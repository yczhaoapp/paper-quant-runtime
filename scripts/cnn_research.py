"""Development-only fitting and independent baselines; no unseen data selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

from paperquant.models import TrainingRequest, TrainingSample
from scripts.build_six_paper_inputs import raw_bars
from scripts.gate_support import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research/cnn-refinement/experiment-plan.json"


def load_strategy(path=None):
    path = path or ROOT / "strategies/supplement.grouped_cnn_lstm/strategy.py"
    spec = importlib.util.spec_from_file_location("cnn_research_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GroupedCNNLSTM


def development_rows(*, history=24):
    plan = json.loads(PLAN.read_text())
    prior = raw_bars("4h", [plan["history_month"]]) if history > 24 else []
    train = raw_bars("4h", plan["training"])
    val = raw_bars("4h", plan["validation"])
    all_rows = prior + train + val
    splits = []
    for start, length in ((len(prior), len(train)), (len(prior) + len(train), len(val))):
        split = []
        for end in range(max(history - 1, start), start + length - 1):
            split.append(
                (
                    all_rows[end][0],
                    [r[1] for r in all_rows[end - history + 1 : end + 1]],
                    int(all_rows[end + 1][1][3] > all_rows[end][1][3]),
                )
            )
        splits.append(split)
    return splits


def request_for(options, *, history=64, seed=23):
    samples = []
    for partition, rows in enumerate(development_rows(history=history)):
        for stamp, window, label in rows:
            features = {
                f"v_{t}_{j}": Decimal(str(value))
                for t, row in enumerate(window)
                for j, value in enumerate(row)
            }
            features.update(partition=Decimal(partition), window_end_ms=Decimal(stamp))
            features.update({"cfg_" + k: Decimal(str(v)) for k, v in options.items()})
            samples.append(TrainingSample(features=features, target=Decimal(label)))
    return TrainingRequest(
        dataset_id="supplement.grouped_cnn_lstm", seed=seed, samples=tuple(samples)
    )


def metrics(probabilities, labels, prior):
    p, y = np.asarray(probabilities), np.asarray(labels)
    losses = -(
        y * np.log(np.clip(p, 1e-12, 1 - 1e-12))
        + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1 - 1e-12))
    )
    positives, negatives = p[y == 1], p[y == 0]
    auc = float(
        np.mean(
            (positives[:, None] > negatives[None, :])
            + 0.5 * (positives[:, None] == negatives[None, :])
        )
    )
    return {
        "samples": len(y),
        "bce": float(np.mean(losses)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "auc": auc,
        "brier": float(np.mean((p - y) ** 2)),
        "training_prior_bce": float(np.mean(-y * math.log(prior) - (1 - y) * math.log(1 - prior))),
        "training_majority_accuracy": float(np.mean(y == (prior >= 0.5))),
        "descriptive_majority_accuracy": float(max(np.mean(y), 1 - np.mean(y))),
        "probability_min": float(p.min()),
        "probability_max": float(p.max()),
        "unchanged_threshold_signals": int(np.sum((p <= 1 / 3) | (p >= 2 / 3))),
    }


def summary_features(cls, rows, *, mode=0):
    channels = np.asarray(
        [cls.features(window, mode=mode) if mode else cls.features(window) for _, window, _ in rows]
    )
    return np.concatenate(
        (channels[:, -1, :], channels.mean(axis=1), channels[:, -1, :] - channels[:, 0, :]), axis=1
    )


def baseline(*, history=24, mode=0, output, package=None):
    cls = load_strategy(package)
    train, validation = development_rows(history=history)
    x, vx = summary_features(cls, train, mode=mode), summary_features(cls, validation, mode=mode)
    y, vy = np.array([r[2] for r in train]), np.array([r[2] for r in validation])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    x, vx = (x - mean) / scale, (vx - mean) / scale
    prior = float(y.mean())
    settings = json.loads(PLAN.read_text())["baselines"]
    candidates = []
    for ridge in settings["ridge"]:
        w, bias = np.zeros(x.shape[1]), math.log(prior / (1 - prior))
        mw, vw = np.zeros_like(w), np.zeros_like(w)
        mb, vb = 0.0, 0.0
        for step in range(1, settings["epochs"] + 1):
            p = 1 / (1 + np.exp(-np.clip(x @ w + bias, -40, 40)))
            grad = x.T @ (p - y) / len(y) + ridge * w
            gb = float(np.mean(p - y))
            mw, vw = 0.9 * mw + 0.1 * grad, 0.999 * vw + 0.001 * grad * grad
            mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb * gb
            rate = settings["learning_rate"]
            w -= rate * (mw / (1 - 0.9**step)) / (np.sqrt(vw / (1 - 0.999**step)) + 1e-8)
            bias -= rate * (mb / (1 - 0.9**step)) / (math.sqrt(vb / (1 - 0.999**step)) + 1e-8)
        prediction = 1 / (1 + np.exp(-np.clip(vx @ w + bias, -40, 40)))
        candidates.append(
            {
                "ridge": ridge,
                "validation": metrics(prediction, vy, prior),
                "weights": w.tolist(),
                "bias": bias,
            }
        )
    selected = min(candidates, key=lambda c: c["validation"]["bce"])
    result = {
        "scope": "independent research baseline, not a catalog strategy or paper reproduction",
        "test_acquired": False,
        "feature_mode": mode,
        "history": history,
        "training_samples": len(y),
        "validation_samples": len(vy),
        "training_prior": prior,
        "means": mean.tolist(),
        "scales": scale.tolist(),
        "candidates": candidates,
        "selected": selected,
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "baseline": str(output),
                "selected": selected["ridge"],
                "validation": selected["validation"],
            }
        ),
        flush=True,
    )


def fit(options, *, output, seed=23):
    model = load_strategy()()
    request = request_for(options, history=int(options.get("history_bars", 64)), seed=seed)
    started = time.perf_counter()
    payload = model.train(request)
    elapsed = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".bin").write_bytes(payload)
    state = json.loads(payload)
    result = {
        "elapsed_host_seconds": elapsed,
        "options": options,
        "seed": seed,
        "package_source_sha256": hashlib.sha256(
            (ROOT / "strategies/supplement.grouped_cnn_lstm/strategy.py").read_bytes()
        ).hexdigest(),
        "training_sha256": hashlib.sha256(request.model_dump_json().encode()).hexdigest(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "training": state["training"],
        "holdout_used": False,
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "fit": str(output),
                "seconds": elapsed,
                "selected_epoch": state["training"]["selected_epoch"],
                "validation": state["training"]["selected_validation"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("baseline", "fit"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--options", default="{}")
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.operation == "baseline":
        baseline(history=args.history, mode=args.mode, output=args.output, package=args.package)
    else:
        fit(json.loads(args.options), output=args.output, seed=args.seed)
