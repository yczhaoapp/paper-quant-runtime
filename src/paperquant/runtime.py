from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import NoReturn, Protocol, cast

from paperquant.compiler import compile_run, fingerprint
from paperquant.engine import BacktestEngine, DecisionStrategy
from paperquant.models import (
    Artifact,
    ContractFault,
    DatasetDeclaration,
    EngineCapability,
    ErrorCode,
    Failure,
    MarketEvent,
    RunPolicy,
    RunReport,
    Stage,
    TrainingRequest,
    WorkerReceipt,
)
from paperquant.output import atomic_bytes, prepare_output


class TrainableStrategy(DecisionStrategy, Protocol):
    def train(self, request: TrainingRequest) -> bytes: ...

    def load(self, payload: bytes) -> None: ...


def _fail(run_id: str, stage: Stage, code: ErrorCode, message: str, **details: str) -> NoReturn:
    raise ContractFault(
        Failure(run_id=run_id, stage=stage, code=code, message=message, details=details)
    )


def _persist_model(
    *, run_id: str, strategy_id: str, request: TrainingRequest, payload: bytes, root: Path
) -> Artifact:
    content_sha256 = hashlib.sha256(payload).hexdigest()
    relative = f"models/{content_sha256}.bin"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=target.parent, prefix=".model-", delete=False)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, target)
    finally:
        handle.close()
        if os.path.exists(handle.name):
            os.unlink(handle.name)
    if hashlib.sha256(target.read_bytes()).hexdigest() != content_sha256:
        _fail(run_id, Stage.LOAD, ErrorCode.MODEL_CORRUPT, "Saved model differs from trained bytes")
    return Artifact(
        strategy_id=strategy_id,
        training_sha256=fingerprint(request),
        content_sha256=content_sha256,
        format="strategy-native-v1",
        relative_path=relative,
    )


def _execute(
    *,
    run_id: str,
    strategy: DecisionStrategy,
    dataset: DatasetDeclaration,
    engine_capability: EngineCapability,
    policy: RunPolicy,
    events: tuple[MarketEvent, ...],
    engine: BacktestEngine,
    output: Path,
    training: TrainingRequest | None = None,
    worker_receipt: WorkerReceipt | None = None,
) -> RunReport:
    if (policy.sandbox == "strict") != (worker_receipt is not None):
        _fail(
            run_id,
            Stage.COMPILE,
            ErrorCode.SANDBOX_UNAVAILABLE,
            "Worker evidence is missing or misplaced",
        )
    try:
        actual_capability = engine.capability(engine_capability.data_fields)
    except Exception as exc:
        _fail(run_id, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED,
              "Engine could not declare its capabilities", cause=type(exc).__name__)
    if fingerprint(actual_capability) != fingerprint(engine_capability):
        _fail(
            run_id,
            Stage.COMPILE,
            ErrorCode.ENGINE_UNSUPPORTED,
            "Advertised engine capability differs from the selected engine",
        )
    declaration = strategy.declaration
    plan, effective = compile_run(
        run_id=run_id,
        strategy=declaration,
        dataset=dataset,
        engine=engine_capability,
        policy=policy,
        events=events,
        training=training,
    )
    artifact = None
    if declaration.training_required:
        if training is None or training.dataset_id != dataset.dataset_id or not training.samples:
            _fail(
                run_id,
                Stage.TRAIN,
                ErrorCode.TRAINING_FAILED,
                "Training request is absent or mismatched",
            )
        if not hasattr(strategy, "train") or not hasattr(strategy, "load"):
            _fail(
                run_id, Stage.TRAIN, ErrorCode.TRAINING_FAILED, "Strategy lacks train/load methods"
            )
        trainable = cast(TrainableStrategy, strategy)
        try:
            payload = trainable.train(training)
        except ContractFault:
            raise
        except Exception as exc:
            _fail(
                run_id,
                Stage.TRAIN,
                ErrorCode.TRAINING_FAILED,
                "Strategy training raised an exception",
                cause=type(exc).__name__,
            )
        if type(payload) is not bytes or not payload:
            _fail(
                run_id, Stage.TRAIN, ErrorCode.TRAINING_FAILED, "Training returned no model bytes"
            )
        if fingerprint(training) != plan.training_sha256:
            _fail(
                run_id,
                Stage.TRAIN,
                ErrorCode.TRAINING_FAILED,
                "Training request changed during training",
            )
        artifact = _persist_model(
            run_id=run_id,
            strategy_id=declaration.strategy_id,
            request=training,
            payload=payload,
            root=output,
        )
        stored = output / artifact.relative_path
        if not stored.is_file():
            _fail(run_id, Stage.LOAD, ErrorCode.MODEL_MISSING, "Model file disappeared")
        model_bytes = stored.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != artifact.content_sha256:
            _fail(run_id, Stage.LOAD, ErrorCode.MODEL_CORRUPT, "Model digest differs")
        try:
            trainable.load(model_bytes)
        except Exception as exc:
            _fail(
                run_id,
                Stage.LOAD,
                ErrorCode.MODEL_CORRUPT,
                "Model reloading failed",
                cause=type(exc).__name__,
            )
    elif training is not None:
        _fail(
            run_id, Stage.TRAIN, ErrorCode.TRAINING_FAILED, "Rule strategy received training data"
        )

    if fingerprint(strategy.declaration) != plan.declaration_sha256:
        _fail(
            run_id, Stage.INFER, ErrorCode.INPUT_INVALID, "Strategy declaration changed during run"
        )
    decisions, orders, fills, accounts = engine.run(plan=plan, strategy=strategy, events=effective)
    if len(decisions) != len(effective) or len(accounts) != len(effective):
        _fail(run_id, Stage.BACKTEST, ErrorCode.BACKTEST_FAILED,
              "Engine returned incomplete decision or account traces")
    for event, decision, account in zip(effective, decisions, accounts, strict=True):
        if (decision.event_id != event.event_id or decision.event_time != event.available_time
            or account.timestamp != event.available_time):
            _fail(run_id, Stage.BACKTEST, ErrorCode.BACKTEST_FAILED,
                  "Engine event and account traces are misaligned")
        for action in decision.actions:
            if action.kind not in plan.allowed_actions:
                _fail(run_id, Stage.INFER, ErrorCode.ACTION_INVALID,
                      "Engine reported an undeclared strategy action")
            if hasattr(action, "symbol") and action.symbol not in plan.symbols:
                _fail(run_id, Stage.INFER, ErrorCode.ACTION_INVALID,
                      "Engine reported an unknown action symbol")
    accepted = {(order.order_id, order.symbol) for order in orders
                if order.status == "accepted"}
    completed = {(order.order_id, order.symbol) for order in orders
                 if order.status == "filled"}
    if any((fill.order_id, fill.symbol) not in accepted & completed for fill in fills):
        _fail(run_id, Stage.BACKTEST, ErrorCode.BACKTEST_FAILED,
              "Engine fill lacks accepted and filled order states")
    report = RunReport(
        run_id=run_id,
        plan=plan,
        artifact=artifact,
        worker_receipt=worker_receipt,
        source_events_sha256=fingerprint(events),
        effective_events_sha256=fingerprint(effective),
        decisions=decisions,
        orders=orders,
        fills=fills,
        accounts=accounts,
        final_equity=accounts[-1].equity,
        logs=(
            f"validated {len(events)} source events and {len(effective)} effective events",
            f"executed {len(decisions)} decisions, {len(fills)} fills",
        ),
    )
    atomic_bytes(output / "report.json", (report.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return report


def run(
    *,
    run_id: str,
    strategy: DecisionStrategy,
    dataset: DatasetDeclaration,
    engine_capability: EngineCapability,
    policy: RunPolicy,
    events: tuple[MarketEvent, ...],
    engine: BacktestEngine,
    output: Path,
    training: TrainingRequest | None = None,
) -> RunReport:
    """In-process development API; strict execution is only reached via a package worker."""
    prepare_output(output)
    return _execute(
        run_id=run_id,
        strategy=strategy,
        dataset=dataset,
        engine_capability=engine_capability,
        policy=policy,
        events=events,
        engine=engine,
        output=output,
        training=training,
    )
