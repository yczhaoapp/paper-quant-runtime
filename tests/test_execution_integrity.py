"""Adversarial checks of public execution and independently verified evidence."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest
from test_contract import ScheduledRule, TinyLearner, bars, dataset, declaration

from paperquant.backtrader_engine import BacktraderEngine
from paperquant.cli import main
from paperquant.compiler import compile_run, fingerprint
from paperquant.engine import ReferenceEngine
from paperquant.evidence import verify_bundle, verify_bundle_file
from paperquant.models import (
    ContractFault,
    ErrorCode,
    Granularity,
    NoOp,
    RunPolicy,
    Stage,
    SubmitOrder,
    TargetPosition,
    TrainingRequest,
    TrainingSample,
)
from paperquant.package import replay_bundle, write_manifest
from paperquant.runtime import run

ROOT = Path(__file__).resolve().parents[1]


def execute(strategy, engine, output: Path):
    events = bars()
    source = dataset(events)
    return run(run_id="integrity", strategy=strategy, dataset=source,
               engine_capability=engine.capability(source.fields), policy=RunPolicy(),
               events=events, engine=engine, output=output)


@pytest.mark.parametrize("explicit_first", [False, True])
def test_mixed_actions_cannot_create_the_same_order_twice(
    explicit_first: bool, tmp_path: Path
) -> None:
    class Collision:
        declaration = declaration().model_copy(update={
            "actions": frozenset({"none", "target_position", "submit_order"})})

        def decide(self, event, account):
            target = TargetPosition(symbol=event.symbol, quantity=Decimal(1), reason="target")
            explicit = SubmitOrder(client_order_id="target:0:AAA", symbol=event.symbol,
                                   side="buy", quantity=Decimal(1), reason="explicit")
            return (explicit, target) if explicit_first else (target, explicit)

    with pytest.raises(ContractFault) as caught:
        execute(Collision(), ReferenceEngine(), tmp_path)
    assert caught.value.failure.code == ErrorCode.ORDER_REJECTED
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "bundle.json").exists()


@pytest.mark.parametrize("engine_type", [ReferenceEngine, BacktraderEngine])
@pytest.mark.parametrize("explicit_first", [False, True])
def test_order_namespace_is_shared_across_events(engine_type, explicit_first, tmp_path) -> None:
    class Collision:
        declaration = declaration().model_copy(update={
            "actions": frozenset({"none", "target_position", "submit_order"})})

        def decide(self, event, account):
            if event.event_id == "bar-0":
                if explicit_first:
                    return (SubmitOrder(client_order_id="target:1:AAA", symbol="AAA",
                                        side="buy", quantity=Decimal(1), reason="first"),)
                return (TargetPosition(symbol="AAA", quantity=Decimal(1), reason="first"),)
            if event.event_id == "bar-1":
                if explicit_first:
                    return (TargetPosition(symbol="AAA", quantity=Decimal(2), reason="next"),)
                return (SubmitOrder(client_order_id="target:0:AAA", symbol="AAA",
                                    side="buy", quantity=Decimal(1), reason="next"),)
            return (NoOp(reason="wait"),)

    with pytest.raises(ContractFault) as caught:
        execute(Collision(), engine_type(), tmp_path)
    assert caught.value.failure.code == ErrorCode.ORDER_REJECTED


def test_order_ids_are_unique_across_symbols(tmp_path) -> None:
    class Collision:
        declaration = declaration().model_copy(update={
            "data": declaration().data.model_copy(update={"symbols": frozenset({"AAA", "BBB"})}),
            "actions": frozenset({"target_position", "submit_order"})})

        def decide(self, event, account):
            return (TargetPosition(symbol="AAA", quantity=Decimal(1), reason="target"),
                    SubmitOrder(client_order_id="target:0:AAA", symbol="BBB", side="buy",
                                quantity=Decimal(1), reason="different symbol"))

    events = tuple(event.model_copy(update={"symbol": "BBB"}) if index % 2 else event
                   for index, event in enumerate(bars()))
    source = dataset(events).model_copy(update={
        "symbols": frozenset({"AAA", "BBB"}), "content_sha256": fingerprint(events)})
    engine = ReferenceEngine()
    with pytest.raises(ContractFault) as caught:
        run(run_id="symbol-collision", strategy=Collision(), dataset=source,
            engine_capability=engine.capability(source.fields), policy=RunPolicy(),
            events=events, engine=engine, output=tmp_path)
    assert caught.value.failure.code == ErrorCode.ORDER_REJECTED


@pytest.mark.parametrize("mutation", ["conversion_parameters", "effective_prices"])
def test_offline_recompilation_checks_conversion_content(mutation, tmp_path) -> None:
    source_events = tuple(event.model_copy(update={"symbol": "SOURCE"}) for event in bars())
    source = dataset(source_events).model_copy(update={
        "symbols": frozenset({"SOURCE"}), "content_sha256": fingerprint(source_events)})
    engine = ReferenceEngine()
    run(run_id="mapped", strategy=ScheduledRule(), dataset=source,
        engine_capability=engine.capability(source.fields),
        policy=RunPolicy(symbol_map={"SOURCE": "AAA"}), events=source_events,
        engine=engine, output=tmp_path)
    bundle = verify_bundle_file(tmp_path / "bundle.json")
    plan, report = bundle.report.plan, bundle.report
    conversion = plan.conversions[0]
    if mutation == "conversion_parameters":
        plan = plan.model_copy(update={"conversions": (
            conversion.model_copy(update={"parameters": {"SOURCE": "OTHER"}}),)})
    else:
        events = tuple(event.model_copy(update={"values": {
            key: value + Decimal(10) if key in {"open", "high", "low", "close"} else value
            for key, value in event.values.items()}}) for event in bundle.effective_events)
        digest = fingerprint(events)
        plan = plan.model_copy(update={"effective_events_sha256": digest, "conversions": (
            conversion.model_copy(update={"output_sha256": digest}),)})
        report = report.model_copy(update={"effective_events_sha256": digest})
        bundle = bundle.model_copy(update={"effective_events": events})
    bundle = bundle.model_copy(update={"report": report.model_copy(update={"plan": plan})})
    with pytest.raises(ValueError, match="recompiled"):
        verify_bundle(bundle)


def test_runtime_rejects_an_external_engine_with_duplicate_order_creation(tmp_path) -> None:
    class DuplicateOrders(ReferenceEngine):
        def run(self, **kwargs):
            decisions, orders, fills, accounts = super().run(**kwargs)
            return decisions, (orders[0], *orders), fills, accounts

    with pytest.raises(ContractFault) as caught:
        execute(ScheduledRule(), DuplicateOrders(), tmp_path)
    assert caught.value.failure.code == ErrorCode.BACKTEST_FAILED
    assert "multiple creation" in caught.value.failure.details["reason"]
    assert not (tmp_path / "bundle.json").exists()


@pytest.mark.parametrize("engine_type", [ReferenceEngine, BacktraderEngine])
@pytest.mark.parametrize("field,value", [
    ("max_abs_position", None), ("max_order_quantity", None),
    ("allowed_actions", frozenset({"none", "submit_order", "target_position"})),
    ("symbols", frozenset({"AAA", "UNDECLARED"})),
    ("required_fields", frozenset({"open"})), ("granularity", Granularity.TICK),
    ("sandbox", "strict"), ("financing_mode", "unbounded_margin"),
])
def test_public_engines_reject_every_forged_projection(engine_type, field, value) -> None:
    class Bounded(ScheduledRule):
        declaration = declaration().model_copy(update={"max_order_quantity": Decimal(1)})

    events = bars()
    engine = engine_type()
    plan, effective = compile_run(
        run_id="projection", strategy=Bounded.declaration, dataset=dataset(events),
        engine=engine.capability(dataset(events).fields), engine_profile=engine.profile(),
        policy=RunPolicy(), events=events,
    )
    with pytest.raises(ContractFault) as caught:
        engine.run(plan=plan.model_copy(update={field: value}),
                   strategy=Bounded(), events=effective)
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


@pytest.mark.parametrize("field,value", [
    ("max_abs_position", None), ("max_order_quantity", Decimal(100)),
    ("allowed_actions", frozenset({"none", "target_position", "submit_order"})),
    ("symbols", frozenset({"AAA", "OTHER"})), ("required_fields", frozenset({"open"})),
    ("granularity", Granularity.TICK), ("sandbox", "strict"),
    ("financing_mode", "unbounded_margin"),
])
def test_offline_verification_recompiles_all_plan_fields(field, value, tmp_path) -> None:
    execute(ScheduledRule(), ReferenceEngine(), tmp_path)
    bundle = verify_bundle_file(tmp_path / "bundle.json")
    forged = bundle.model_copy(update={"report": bundle.report.model_copy(update={
        "plan": bundle.report.plan.model_copy(update={field: value})})})
    with pytest.raises(ValueError, match="recompiled"):
        verify_bundle(forged)
    path = tmp_path / "forged.json"
    path.write_text(forged.model_dump_json(), encoding="utf-8")
    assert main(["verify-bundle", "--bundle", str(path)]) == 2
    assert main(["replay-bundle", "--bundle", str(path),
                 "--package", str(ROOT / "examples/basic_rule")]) == 2


@pytest.mark.parametrize("mutation", [
    "duplicate_creation", "duplicate_fill", "missing_fill", "missing_terminal",
    "wrong_terminal_symbol", "repeated_terminal", "unknown_pending", "over_limit_target",
    "over_limit_account", "over_limit_fill", "undeclared_action",
])
def test_offline_verification_rejects_invalid_order_and_risk_traces(mutation, tmp_path) -> None:
    execute(ScheduledRule(), ReferenceEngine(), tmp_path)
    bundle = verify_bundle_file(tmp_path / "bundle.json")
    report = bundle.report
    orders, fills = report.orders, report.fills
    if mutation == "duplicate_creation":
        report = report.model_copy(update={"orders": (orders[0], *orders)})
    elif mutation == "duplicate_fill":
        report = report.model_copy(update={"fills": (fills[0], *fills)})
    elif mutation == "missing_fill":
        report = report.model_copy(update={"fills": fills[1:]})
    elif mutation == "missing_terminal":
        report = report.model_copy(update={"orders": tuple(
            order for order in orders if order.status != "expired")})
    elif mutation == "wrong_terminal_symbol":
        report = report.model_copy(update={"orders": tuple(
            order.model_copy(update={"symbol": "OTHER"}) if order.status == "filled" else order
            for order in orders)})
    elif mutation == "repeated_terminal":
        report = report.model_copy(update={"orders": (*orders, orders[-1])})
    elif mutation == "unknown_pending":
        account = report.accounts[0].model_copy(update={"open_order_ids": ("unknown",)})
        report = report.model_copy(update={"accounts": (account, *report.accounts[1:])})
    elif mutation == "over_limit_target":
        action = TargetPosition(symbol="AAA", quantity=Decimal(3), reason="too large")
        decision = report.decisions[0].model_copy(update={"actions": (action,)})
        report = report.model_copy(update={"decisions": (decision, *report.decisions[1:])})
    elif mutation == "over_limit_account":
        account = report.accounts[1]
        position = account.positions[0].model_copy(update={"quantity": Decimal(3)})
        account = account.model_copy(update={"positions": (position,)})
        report = report.model_copy(update={"accounts": (
            report.accounts[0], account, *report.accounts[2:])})
    elif mutation == "over_limit_fill":
        report = report.model_copy(update={"fills": (
            fills[0].model_copy(update={"quantity": Decimal(3)}), *fills[1:])})
    elif mutation == "undeclared_action":
        action = SubmitOrder(client_order_id="new", symbol="AAA", side="buy",
                             quantity=Decimal(1), reason="undeclared")
        decision = report.decisions[0].model_copy(update={"actions": (action,)})
        report = report.model_copy(update={"decisions": (decision, *report.decisions[1:])})
    with pytest.raises(ValueError):
        verify_bundle(bundle.model_copy(update={"report": report}))


@pytest.mark.parametrize("engine_type", [ReferenceEngine, BacktraderEngine])
@pytest.mark.parametrize("parameter", ["initial_cash", "fee_rate"])
@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_engine_configuration_rejects_nonfinite_numbers(engine_type, parameter, value) -> None:
    with pytest.raises(ValueError, match="finite"):
        engine_type(**{parameter: Decimal(value)})


@pytest.mark.parametrize("engine", ["reference", "backtrader"])
@pytest.mark.parametrize("parameter", ["cash", "fee-rate"])
@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity", "invalid", ""])
def test_cli_nonfinite_configuration_closes_attempt(engine, parameter, value, tmp_path) -> None:
    for filename in ("report.json", "bundle.json", "paper-run.json"):
        (tmp_path / filename).write_text('{"status":"passed"}', encoding="utf-8")
    assert main([
        "run", "--package", str(ROOT / "examples/basic_rule"),
        "--dataset", str(ROOT / "examples/data/dataset.json"),
        "--events", str(ROOT / "examples/data/events.json"),
        "--engine", engine, f"--{parameter}={value}", "--output", str(tmp_path),
    ]) == 2
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["code"] == ErrorCode.INPUT_INVALID
    assert failure["stage"] == Stage.INPUT
    assert json.loads((tmp_path / "attempt.json").read_text())["status"] == "failed"
    for filename in ("report.json", "bundle.json", "paper-run.json"):
        assert not (tmp_path / filename).exists()


def test_post_execution_engine_exception_is_structured(tmp_path) -> None:
    class BrokenProfile(ReferenceEngine):
        ran = False

        def run(self, **kwargs):
            result = super().run(**kwargs)
            self.ran = True
            return result

        def profile(self):
            if self.ran:
                raise RuntimeError("profile lost after matching")
            return super().profile()

    with pytest.raises(ContractFault) as caught:
        execute(ScheduledRule(), BrokenProfile(), tmp_path)
    assert caught.value.failure.stage == Stage.BACKTEST
    assert caught.value.failure.code == ErrorCode.BACKTEST_FAILED
    assert caught.value.failure.details["cause"] == "RuntimeError"
    assert not (tmp_path / "report.json").exists()


def test_model_io_error_keeps_load_stage(tmp_path, monkeypatch) -> None:
    import paperquant.runtime as runtime

    def broken(**kwargs):
        raise OSError("model store failed")

    monkeypatch.setattr(runtime, "_persist_model", broken)
    events = bars()
    source = dataset(events)
    training = TrainingRequest(dataset_id=source.dataset_id, seed=7, samples=(
        TrainingSample(features={"close": Decimal(1)}, target=Decimal(1)),))
    with pytest.raises(ContractFault) as caught:
        run(run_id="model-io", strategy=TinyLearner(), dataset=source,
            engine_capability=ReferenceEngine().capability(source.fields), policy=RunPolicy(),
            events=events, engine=ReferenceEngine(), output=tmp_path, training=training)
    assert caught.value.failure.stage == Stage.LOAD
    assert caught.value.failure.code == ErrorCode.MODEL_CORRUPT


def test_cli_report_publication_error_cleans_partial_success(tmp_path, monkeypatch) -> None:
    import paperquant.runtime as runtime

    original = runtime.atomic_bytes

    def broken(path, payload):
        if path.name == "report.json":
            raise OSError("report publication failed")
        return original(path, payload)

    monkeypatch.setattr(runtime, "atomic_bytes", broken)
    assert main([
        "run", "--package", str(ROOT / "examples/basic_rule"),
        "--dataset", str(ROOT / "examples/data/dataset.json"),
        "--events", str(ROOT / "examples/data/events.json"), "--output", str(tmp_path),
    ]) == 2
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["stage"] == Stage.REPORT
    assert failure["code"] == ErrorCode.BACKTEST_FAILED
    assert json.loads((tmp_path / "attempt.json").read_text())["status"] == "failed"
    assert not (tmp_path / "bundle.json").exists()
    assert not (tmp_path / "report.json").exists()


def test_replay_metadata_error_is_structured(tmp_path) -> None:
    assert main([
        "run", "--package", str(ROOT / "examples/basic_rule"),
        "--dataset", str(ROOT / "examples/data/dataset.json"),
        "--events", str(ROOT / "examples/data/events.json"), "--output", str(tmp_path),
    ]) == 0

    class BrokenProfile(ReferenceEngine):
        def profile(self):
            raise RuntimeError("replay metadata unavailable")

    with pytest.raises(ContractFault) as caught:
        replay_bundle(bundle_path=tmp_path / "bundle.json",
                      package_dir=ROOT / "examples/basic_rule", engine=BrokenProfile())
    assert caught.value.failure.stage == Stage.BACKTEST
    assert caught.value.failure.code == ErrorCode.BACKTEST_FAILED


@pytest.mark.skipif(os.environ.get("PAPERQUANT_TEST_DOCKER") != "1",
                    reason="requires the strict worker release gate")
def test_strict_worker_cannot_make_order_collision_succeed(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    spec = declaration().model_copy(update={
        "actions": frozenset({"none", "target_position", "submit_order"})})
    source = '''from decimal import Decimal
from paperquant.strategy_api import StrategyDeclaration, SubmitOrder, TargetPosition
class Collision:
    declaration = StrategyDeclaration.model_validate(SPEC)
    def decide(self, event, account):
        return (TargetPosition(symbol=event.symbol, quantity=Decimal(1), reason="target"),
                SubmitOrder(client_order_id="target:0:AAA", symbol=event.symbol,
                            side="buy", quantity=Decimal(1), reason="explicit"))
'''.replace("SPEC", repr(spec.model_dump(mode="json")))
    (package / "strategy.py").write_text(source, encoding="utf-8")
    write_manifest(package, spec, "Collision")
    events = bars()
    event_file, dataset_file = tmp_path / "events.json", tmp_path / "dataset.json"
    event_file.write_text(json.dumps([event.model_dump(mode="json") for event in events]))
    dataset_file.write_text(dataset(events).model_dump_json())
    output = tmp_path / "strict"
    assert main([
        "run", "--package", str(package), "--dataset", str(dataset_file),
        "--events", str(event_file), "--policy", str(ROOT / "examples/strict-policy.json"),
        "--output", str(output),
    ]) == 2
    failure = json.loads((output / "failure.json").read_text())
    assert failure["code"] == ErrorCode.ORDER_REJECTED
    assert json.loads((output / "attempt.json").read_text())["status"] == "failed"
    assert not (output / "bundle.json").exists()
    assert not (output / "report.json").exists()
