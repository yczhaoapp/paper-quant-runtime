"""CNN supplement gate: six real-paper paths plus an explicit controlled lifecycle."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import fingerprint
from paperquant.evidence import verify_bundle_file
from paperquant.models import (
    DatasetDeclaration,
    MarketEvent,
    RunPolicy,
    TrainingRequest,
    TrainingSample,
)
from scripts.accept_six_papers import accept, invoke, relative
from scripts.build_six_paper_inputs import save
from scripts.cnn_evidence import audit_execution
from scripts.cnn_release import selection_lock
from scripts.cnn_research import load_strategy
from scripts.gate_support import check_source_identity, new_attempt, source_identity, write_json
from scripts.two_paper_quality import check_contract_lock

ROOT = Path(__file__).resolve().parents[1]


def controlled_inputs(folder, options):
    rng = random.Random(19)
    samples = []
    for index in range(96):
        label = index % 2
        slope = (1 if label else -1) * rng.uniform(0.002, 0.008)
        price = rng.uniform(500, 1000)
        raw = []
        for t in range(24):
            close = price * math.exp(slope * t + rng.uniform(-0.0003, 0.0003))
            raw.append([close * 0.999, close * 1.001, close * 0.998, close, rng.uniform(80, 120)])
        features = {
            f"v_{t}_{j}": Decimal(str(v)) for t, row in enumerate(raw) for j, v in enumerate(row)
        }
        features.update(
            partition=Decimal(index >= 72),
            window_end_ms=Decimal(index + 1),
            **{"cfg_" + key: Decimal(str(value)) for key, value in options.items()},
        )
        samples.append(TrainingSample(features=features, target=Decimal(label)))
    training = TrainingRequest(
        dataset_id="supplement.grouped_cnn_lstm", seed=19, samples=tuple(samples)
    )
    events = []
    price = 500.0
    rng = random.Random(7019)
    origin = datetime(2025, 1, 1, tzinfo=UTC)
    for i in range(256):
        direction = 1 if (i // 64) % 2 == 0 else -1
        opened = price
        price *= math.exp(direction * 0.006 + rng.uniform(-0.0002, 0.0002))
        start = origin + timedelta(hours=i * 4)
        close_time = start + timedelta(hours=4)
        events.append(
            MarketEvent(
                event_id=f"controlled:{i}",
                symbol="BTCUSDT",
                bar_open_time=start,
                event_time=close_time,
                available_time=close_time,
                values={
                    "open": Decimal(str(opened)),
                    "high": Decimal(str(max(opened, price) * 1.001)),
                    "low": Decimal(str(min(opened, price) * 0.998)),
                    "close": Decimal(str(price)),
                    "volume": Decimal(str(rng.uniform(80, 120))),
                },
            )
        )
    dataset = DatasetDeclaration(
        contract_version="1.1",
        dataset_id=training.dataset_id,
        granularity="minute",
        bar_seconds=14400,
        market_kind="bar",
        fields=frozenset(events[0].values),
        symbols=frozenset({"BTCUSDT"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    for name, value in (("dataset", dataset), ("events", events), ("training", training)):
        save(folder / (name + ".json"), value)
    write_json(
        folder / "lineage.json",
        {
            "type": "controlled_learnability_simulation",
            "training_seed": 19,
            "evaluation_seed": 7019,
            "training_windows": 72,
            "validation_windows": 24,
            "evaluation_bars": 256,
            "configuration": options,
            "scope": "trained CNN crosses unchanged thresholds on learnable patterns",
            "exclusions": "controlled simulation; no paper or held-out performance claim",
        },
    )


def controlled_lifecycle(attempt, *, strict):
    lock = selection_lock()
    inputs = attempt / "inputs"
    inputs.mkdir()
    controlled_inputs(inputs, lock["configuration"])
    rows, reports = {}, {}
    for mode in ("host", "strict") if strict else ("host",):
        folder = attempt / mode
        folder.mkdir()
        save(
            folder / "policy.json",
            RunPolicy(
                financing_mode="unbounded_margin",
                sandbox="strict" if mode == "strict" else "development",
            ),
        )
        command = [
            "paper-run",
            "--recipe",
            "research/six-paper/cases/grouped_cnn_lstm/recipe.json",
            "--dataset",
            relative(inputs / "dataset.json"),
            "--events",
            relative(inputs / "events.json"),
            "--training",
            relative(inputs / "training.json"),
            "--policy",
            relative(folder / "policy.json"),
            "--engine",
            "reference",
            "--cash",
            "100",
            "--fee-rate",
            "0.0001",
            "--output",
            relative(folder / "run"),
        ]
        if invoke(command, folder):
            raise ValueError("Controlled trained-CNN lifecycle failed: " + mode)
        bundle = verify_bundle_file(folder / "run/bundle.json")
        report = bundle.report
        if report.artifact is None or not report.fills:
            raise ValueError("Controlled lifecycle lacks a trained artifact or natural fills")
        nonzero = [
            a
            for d in report.decisions
            for a in d.actions
            if a.kind == "target_position" and a.quantity != 0
        ]
        if not nonzero:
            raise ValueError("Controlled lifecycle never crossed the unchanged thresholds")
        # Recompute the saved model, never substitute a stored action sequence.
        model = load_strategy()()
        payload = (folder / "run" / report.artifact.relative_path).read_bytes()
        if json.loads(payload)["options"] != lock["configuration"]:
            raise ValueError("Controlled model did not use the selected CNN configuration")
        model.load(payload)
        raw = [
            [float(event.values[key]) for key in ("open", "high", "low", "close", "volume")]
            for event in bundle.effective_events
        ]
        predictions = []
        for index in range(model.history_bars - 1, len(raw)):
            probability = model.forward(
                model.channels(raw[index - model.history_bars + 1 : index + 1])
            ).value
            lower, upper = model.theta / (1 + model.theta), 1 / (1 + model.theta)
            predictions.append(
                {
                    "event_id": bundle.effective_events[index].event_id,
                    "probability": probability,
                    "signal": 1 if probability >= upper else -1 if probability <= lower else 0,
                }
            )
        diagnostic = audit_execution(
            report,
            bundle.effective_events,
            predictions,
            history=model.history_bars,
            theta=model.theta,
        )
        write_json(folder / "execution-diagnostic.json", diagnostic)
        write_json(folder / "prediction-audit.json", {"predictions": predictions})
        rows[mode] = {
            "runtime": "succeeded",
            "cold_replay": "passed",
            "fills": len(report.fills),
            "nonzero_targets": len(nonzero),
            "orders": len(report.orders),
            "training_sha256": report.plan.training_sha256,
            "model_sha256": report.artifact.content_sha256,
            "package_source_sha256": report.plan.package_source_sha256,
            "configuration": lock["configuration"],
            "execution_diagnostic": diagnostic,
            "worker_receipt": report.worker_receipt.model_dump(mode="json")
            if report.worker_receipt
            else None,
        }
        reports[mode] = report
    if strict:
        left, right = reports["host"], reports["strict"]
        if (left.decisions, left.orders, left.fills, left.accounts) != (
            right.decisions,
            right.orders,
            right.fills,
            right.accounts,
        ):
            raise ValueError("Controlled host/strict traces differ")
    return {
        "scope": "controlled learnability and actual trained-model execution only",
        "real_market_prediction_claim": False,
        "model_relationship": "same implementation and configuration; separately trained model",
        "modes": rows,
        "host_strict_trace_comparison": "passed" if strict else "not_run",
    }


def close_evidence(six, controlled):
    case = next(row for row in six["cases"] if row["strategy_id"] == "supplement.grouped_cnn_lstm")
    if (
        case["source_mapping"] != "passed"
        or case["local_method_oracles"] != "passed"
        or not six.get("method_oracle_junit_sha256")
    ):
        raise ValueError("CNN closure lacks executed source and method validation")
    modes = {}
    for mode, natural in case["modes"].items():
        control = controlled["modes"][mode]
        quality = natural["adaptation_quality"]
        diagnostic = quality["execution_diagnostic"]
        if (
            natural["runtime"] != "succeeded"
            or natural["cold_replay"] != "passed"
            or natural["training"] != "executed_and_reloaded"
            or control["runtime"] != "succeeded"
            or control["cold_replay"] != "passed"
            or not control["fills"]
            or not control["nonzero_targets"]
            or diagnostic["package_source_sha256"] != control["package_source_sha256"]
            or natural["training_sha256"] == control["training_sha256"]
            or diagnostic["model_sha256"] == control["model_sha256"]
            or diagnostic["confidence"] != control["execution_diagnostic"]["confidence"]
        ):
            raise ValueError("Natural and controlled CNN evidence is incomplete or conflated")
        modes[mode] = {
            "status": {
                "source_mapping": case["source_mapping"],
                "declared_local_method_oracles": case["local_method_oracles"],
                "contract_lifecycle": "passed",
                "cold_replay": natural["cold_replay"],
                "training_quality_threshold": quality["evidence_status"][
                    "training_quality_threshold"
                ],
                "generalization_advantage": "not_established",
                "full_paper_experiment": "not_reproduced",
                "controlled_execution_path": "passed",
            },
            "natural_market": {
                "evaluation": quality["cnn_refinement"]["evaluation_window"],
                "execution": diagnostic,
                "training_fit": quality["method_fitting"],
                "prediction_metrics": quality["test_metrics"],
                "report_sha256": natural["report_sha256"],
                "bundle_sha256": natural["bundle_sha256"],
            },
            "controlled_simulation": control,
        }
    return {
        "status": "passed",
        "status_scope": "evidence consistency and declared local method/contract validation",
        "method_oracle_junit_sha256": six["method_oracle_junit_sha256"],
        "model_relationship": controlled["model_relationship"],
        "host_strict_natural_trace_comparison": case.get("host_strict_trace_comparison", "not_run"),
        "host_strict_controlled_trace_comparison": controlled["host_strict_trace_comparison"],
        "threshold_changed": False,
        "modes": modes,
    }


def gate(output, *, strict=False, require_clean=False, controls_only=False):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = new_attempt()
    receipt = output / "cnn-refinement.json"
    write_json(receipt, state)
    attempt = output / "attempts" / state["attempt_id"]
    try:
        attempt.mkdir(parents=True)
        identity = source_identity(ROOT, require_clean=require_clean)
        state.update(identity, strict=strict)
        state["contract_preservation"] = check_contract_lock()
        state["selection"] = selection_lock()
        if not controls_only:
            six = accept(attempt / "six-paper", strict=strict, require_clean=require_clean)
            state["six_paper_attempt_id"] = six["attempt_id"]
            if six["status"] != "passed":
                raise ValueError("Six-paper gate failed: " + six.get("reason", "unknown"))
        state["controlled_lifecycle"] = controlled_lifecycle(attempt, strict=strict)
        if not controls_only:
            closure = close_evidence(six, state["controlled_lifecycle"])
            closure.update(
                identity, attempt_id=state["attempt_id"], six_paper_attempt_id=six["attempt_id"]
            )
            write_json(attempt / "cnn-evidence.json", closure)
            state["cnn_evidence"] = closure
        state["scope"] = (
            "controlled lifecycle only"
            if controls_only
            else ("six source-bound adaptations, CNN holdout and baselines, controlled execution")
        )
        state["market_prediction_or_profit_guarantee"] = False
        check_source_identity(ROOT, identity, require_clean=require_clean)
        state["status"] = "passed"
    except Exception as exc:
        state.update(status="failed", cause=type(exc).__name__, reason=str(exc)[:1000])
    state["completed_at"] = datetime.now(UTC).isoformat()
    write_json(receipt, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    result = gate(
        args.output,
        strict=args.strict,
        require_clean=args.require_clean,
        controls_only=args.controls_only,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "reason": result.get("reason"),
                "receipt": str(args.output / "cnn-refinement.json"),
            }
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 2)
