"""Separate method fitting, predictions and trades from contract execution status."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from decimal import Decimal
from pathlib import Path

from scripts.gate_support import write_json

ROOT = Path(__file__).resolve().parents[1]


def check_contract_lock():
    lock = json.loads((ROOT / "research/two-paper/contract-lock.json").read_text())
    actual = {
        p.relative_to(ROOT).as_posix()
        for directory in ("src", "schemas")
        for p in (ROOT / directory).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    if actual != set(lock["files"]):
        raise ValueError("Public contract/runtime inventory changed")
    for name, digest in lock["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Public contract/runtime source changed: " + name)
    return {
        "status": "passed",
        "baseline_commit": lock["baseline_commit"],
        "byte_identical_files": len(actual),
    }


def quality_for_run(run_dir, report, bundle):
    name = report.plan.strategy_id.removeprefix("supplement.")
    if name not in {"grouped_cnn_lstm", "gmm_knn"}:
        return None
    source = ROOT / f"strategies/supplement.{name}/strategy.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != report.plan.package_source_sha256:
        raise ValueError("Quality audit source differs from executed package")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "GroupedCNNLSTM" if name == "grouped_cnn_lstm" else "GMMConditionalKNN")
    payload = (run_dir / report.artifact.relative_path).read_bytes()
    model = cls()
    model.load(payload)
    state = json.loads(payload)
    rows = [
        [float(e.values[k]) for k in ("open", "high", "low", "close", "volume")]
        for e in bundle.effective_events
    ]
    training = state["training"]
    traces, predicted, truth = [], [], []
    if name == "grouped_cnn_lstm":
        for i in range(23, len(rows)):
            p = model.forward(model.channels(rows[i - 23 : i + 1])).value
            direction = 1 if p >= 2 / 3 else -1 if p <= 1 / 3 else 0
            action = report.decisions[i].actions[0]
            actual_direction = 1 if action.quantity > 0 else -1 if action.quantity < 0 else 0
            if direction != actual_direction or f"p={p:.8f}" not in action.reason:
                raise ValueError("CNN recomputed signal differs from runtime decision")
            traces.append(
                {
                    "event_id": bundle.effective_events[i].event_id,
                    "probability": p,
                    "signal": direction,
                }
            )
            if i + 1 < len(rows):
                predicted.append(p)
                truth.append(int(rows[i + 1][3] > rows[i][3]))
        test_metrics = cls.metrics(predicted, truth)
        prior = training["selected_train"]["label_counts"]["1"] / training["samples"]
        test_metrics["training_prior_bce"] = -sum(
            y * math.log(prior) + (1 - y) * math.log(1 - prior) for y in truth
        ) / len(truth)
        fitting = {
            "optimization_effective": training["selected_epoch"] > 0
            and training["selected_train"]["bce"] < training["initial_train"]["bce"] - 1e-5,
            "validation_bce_improved": training["selected_validation"]["bce"]
            < training["initial_validation"]["bce"],
            "paper_quality_threshold_met": training["paper_quality_threshold_met"],
            "paper_quality_scope": "printed proxy only; no original convergence claim",
        }
        if not fitting["optimization_effective"]:
            raise ValueError("CNN selected model lacks measured real-data fitting improvement")
    else:
        threshold = state["threshold"]
        for start in range(0, len(rows) - 59, 60):
            closes = [row[3] for row in rows[start : start + 60]]
            pred = model.predict(closes[:59])
            action = report.decisions[start + 58].actions[0]
            if action.quantity != Decimal(pred) * Decimal("0.001"):
                raise ValueError("GMM recomputed prediction differs from runtime decision")
            ret = math.log(closes[59] / closes[58])
            actual = 1 if ret > threshold else -1 if ret < -threshold else 0
            predicted.append(pred)
            truth.append(actual)
            traces.append(
                {
                    "event_id": bundle.effective_events[start + 58].event_id,
                    "predicted": pred,
                    "actual": actual,
                    "minute_return": ret,
                }
            )
        test_metrics = cls.classification_metrics(truth, predicted)
        fitting = {
            "training_samples": training["blocks"],
            "label_counts": training["label_counts"],
            "minimum_class_support": min(training["label_counts"].values()),
            "validation_fitting": False,
        }
        if fitting["minimum_class_support"] < 12:
            raise ValueError("Expanded GMM training still lacks declared per-class support")
    nonzero = [
        a
        for d in report.decisions
        for a in d.actions
        if a.kind == "target_position" and a.quantity != 0
    ]
    result = {
        "runtime": report.status,
        "method_fitting": fitting,
        "test_metrics": test_metrics,
        "trade_coverage": {
            "fills": len(report.fills),
            "nonzero_targets": len(nonzero),
            "long_targets": sum(a.quantity > 0 for a in nonzero),
            "short_targets": sum(a.quantity < 0 for a in nonzero),
        },
        "validation": training.get("selected_validation", training.get("validation")),
        "paper_experiment_reproduced": False,
        "prediction_audit": "recomputed from executed model and matched to actual decisions",
        "test_used_for_selection": False,
    }
    write_json(run_dir / "quality.json", result)
    write_json(run_dir / "prediction-audit.json", {"predictions": traces})
    return result
