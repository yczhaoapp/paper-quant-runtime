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
from scripts.cnn_release import selection_lock
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
        rows[mode] = {
            "runtime": "succeeded",
            "cold_replay": "passed",
            "fills": len(report.fills),
            "nonzero_targets": len(nonzero),
            "orders": len(report.orders),
            "training_sha256": report.plan.training_sha256,
            "model_sha256": report.artifact.content_sha256,
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
        "modes": rows,
        "host_strict_trace_comparison": "passed" if strict else "not_run",
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
        state["scope"] = (
            "controlled lifecycle only"
            if controls_only
            else (
                "six source-bound adaptations, CNN holdout and baselines, controlled execution"
            )
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
