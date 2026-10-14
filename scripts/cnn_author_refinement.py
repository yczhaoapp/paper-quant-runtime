"""Predeclared author-inspired comparisons using development data only."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from paperquant.compiler import fingerprint
from paperquant.evidence import verify_bundle_file
from paperquant.models import DatasetDeclaration, RunPolicy
from paperquant.package import write_manifest
from scripts.accept_six_papers import invoke, relative
from scripts.build_six_paper_inputs import bar_events, raw_bars, save
from scripts.cnn_research import fit, load_strategy, request_for
from scripts.gate_support import write_json

ROOT = Path(__file__).resolve().parents[1]


def select(output, *, strict):
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "research/audit-refinement/prior-cnn-selection-lock.json").read_text())
    plan_path = ROOT / "research/cnn-refinement/experiment-plan.json"
    plan = json.loads(plan_path.read_text())
    configurations = [("standard-reference", {**old["configuration"], "scaler": 0})]
    configurations += [
        ("minmax-lr" + str(rate), {**old["configuration"], "scaler": 1, "learning_rate": rate})
        for rate in (0.0001, 0.001)
    ]
    candidates = []
    for name, options in configurations:
        receipt = output / (name + ".json")
        fit(options, output=receipt, seed=23)
        result = json.loads(receipt.read_text())
        candidates.append(
            {
                "trial": name,
                "options": options,
                "validation_bce": result["training"]["selected_validation"]["bce"],
                "train_bce": result["training"]["selected_train"]["bce"],
                "selected_epoch": result["training"]["selected_epoch"],
                "host_seconds": result["elapsed_host_seconds"],
                "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            }
        )
    selected = min(candidates, key=lambda row: row["validation_bce"])
    options = selected["options"]
    request = request_for(options, history=options["history_bars"], seed=23)
    cls = load_strategy()
    write_manifest(
        ROOT / "strategies/supplement.grouped_cnn_lstm", cls.declaration, "GroupedCNNLSTM"
    )
    probe = output / "budget-probe"
    probe.mkdir()
    events = bar_events(raw_bars("4h", plan["validation"])[-48:], 14400)
    dataset = DatasetDeclaration(
        contract_version="1.1",
        dataset_id=request.dataset_id,
        granularity="minute",
        bar_seconds=14400,
        market_kind="bar",
        fields=frozenset(events[0].values),
        symbols=frozenset({"BTCUSDT"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    save(probe / "training.json", request)
    save(probe / "events.json", events)
    save(probe / "dataset.json", dataset)
    models = {}
    for mode in ("host", "strict") if strict else ("host",):
        folder = probe / mode
        folder.mkdir()
        save(
            folder / "policy.json",
            RunPolicy(sandbox="strict" if mode == "strict" else "development"),
        )
        args = [
            "run",
            "--package",
            "strategies/supplement.grouped_cnn_lstm",
            "--training",
            relative(probe / "training.json"),
            "--dataset",
            relative(probe / "dataset.json"),
            "--events",
            relative(probe / "events.json"),
            "--policy",
            relative(folder / "policy.json"),
            "--output",
            relative(folder / "run"),
        ]
        if invoke(args, folder):
            raise ValueError("Selected candidate failed its declared resource probe: " + mode)
        bundle = verify_bundle_file(folder / "run/bundle.json")
        models[mode] = bundle.report.artifact.content_sha256
    if not strict:
        raise ValueError("A release selection lock requires the strict resource probe")
    lock = {
        **old,
        "version": "2.0",
        "locked_at": datetime.now(UTC).isoformat(),
        "experiment_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "package_source_sha256": hashlib.sha256(
            (ROOT / "strategies/supplement.grouped_cnn_lstm/strategy.py").read_bytes()
        ).hexdigest(),
        "candidates": candidates,
        "selected_trial": selected["trial"],
        "configuration": options,
        "training_request_canonical_sha256": fingerprint(request),
        "seed": 23,
        "budget_probe_status": "passed",
        "host_payload_sha256": models["host"],
        "strict_payload_sha256": models["strict"],
        "holdout_acquired_at_lock": False,
        "holdout_used_for_selection": False,
        "protocol_deviation": "none for this new 2025Q1 holdout; prior protocol "
        "retained in archived prior lock",
        "feature_selection_disclosure": "Fixed window-relative channels; predeclared train-only "
        "standard vs author-inspired MinMax and two learning "
        "rates; all development candidates retained",
        "prior_negative_result": {
            "version": "0.2.2",
            "commit": plan["reference_commit"],
            "test_period": ["2024-10", "2024-11", "2024-12"],
            "fills": 0,
            "test_bce": 0.6921765,
            "auc": 0.47811,
            "evidence": "prior independent supplement retained; rounded metrics are descriptive",
        },
    }
    for key in (
        "post_evaluation_source_amendment",
        "pre_format_selection_lock_sha256",
        "model_parameter_max_abs_environment_difference",
    ):
        lock.pop(key, None)
    write_json(ROOT / "research/cnn-refinement/selection-lock.json", lock)
    print(json.dumps({"selected": selected, "lock_persisted_before_acquisition": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    select(args.output.resolve(), strict=args.strict)
