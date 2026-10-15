"""Check the selected upstream configuration and audit held-out predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from paperquant.compiler import fingerprint
from scripts.cnn_evidence import evaluation_window
from scripts.cnn_research import metrics

ROOT = Path(__file__).resolve().parents[1]


def selection_lock():
    path = ROOT / "research/cnn-refinement/selection-lock.json"
    lock = json.loads(path.read_text())
    for name, expected in (
        ("research/cnn-refinement/experiment-plan.json", lock["experiment_plan_sha256"]),
        ("strategies/supplement.grouped_cnn_lstm/strategy.py", lock["package_source_sha256"]),
    ):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("CNN release differs from its development selection lock")
    candidates = lock["candidates"]
    best = min(candidates, key=lambda c: c["validation_bce"])
    if (
        best["trial"] != lock["selected_trial"]
        or best["options"] != lock["configuration"]
        or lock["budget_probe_status"] != "passed"
        or lock["holdout_used_for_selection"] is not False
    ):
        raise ValueError("CNN selection is not bound to validation and resource evidence")
    return lock


def audit_cnn(model, state, bundle, rows, traces, truth, predicted):
    lock = selection_lock()
    plan = json.loads((ROOT / "research/cnn-refinement/experiment-plan.json").read_text())
    window = evaluation_window(bundle.effective_events, model.history_bars, plan)
    if len(traces) != window["prediction_events"] or len(truth) != window["labelled_predictions"]:
        raise ValueError("CNN prediction and scoring counts differ from executed evaluation dates")
    if (
        state["options"] != lock["configuration"]
        or state["seed"] != lock["seed"]
        or fingerprint(bundle.training_request) != lock["training_request_canonical_sha256"]
    ):
        raise ValueError("CNN actually trained inputs differ from the selected configuration")
    baseline = lock["baseline"]
    prior = baseline["training_prior"]
    baseline_probabilities = []
    history = model.history_bars
    features = []
    for i in range(history - 1, len(rows) - 1):
        feature = np.asarray(model.features(rows[i - 23 : i + 1]))
        features.append(
            np.concatenate((feature[-1], feature.mean(axis=0), feature[-1] - feature[0]))
        )
    x = (np.asarray(features) - baseline["means"]) / baseline["scales"]
    selected = baseline["selected"]
    baseline_probabilities = (
        1 / (1 + np.exp(-np.clip(x @ selected["weights"] + selected["bias"], -40, 40)))
    ).tolist()
    cnn_metrics = metrics(predicted, truth, prior)
    logistic_metrics = metrics(baseline_probabilities, truth, prior)
    for record, probability in zip(traces[:-1], baseline_probabilities, strict=True):
        record["logistic_baseline_probability"] = probability
    return {
        "configuration_lock_sha256": hashlib.sha256(
            (ROOT / "research/cnn-refinement/selection-lock.json").read_bytes()
        ).hexdigest(),
        "selection": lock["selection_metric"],
        "cnn": cnn_metrics,
        "logistic_baseline": logistic_metrics,
        "prediction_effectiveness": {
            "beats_training_prior_bce": cnn_metrics["bce"] < cnn_metrics["training_prior_bce"],
            "beats_logistic_bce": cnn_metrics["bce"] < logistic_metrics["bce"],
            "auc_above_half": cnn_metrics["auc"] > 0.5,
            "profitability_established": False,
            "interpretation": "descriptive metric comparisons; no statistical significance "
            "or stable generalization advantage established",
        },
        "prior_negative_result": lock["prior_negative_result"],
        "feature_selection_disclosure": lock["feature_selection_disclosure"],
        "protocol_deviation": lock["protocol_deviation"],
        "evaluation_window": window,
        "scope": "local method adaptation; prediction-bar months "
        + ", ".join(window["actual_months"]),
    }
