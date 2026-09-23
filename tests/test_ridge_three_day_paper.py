from __future__ import annotations

import importlib.util
import json
import math
import random
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import (
    DatasetDeclaration,
    MarketEvent,
    RunPolicy,
    TrainingRequest,
    TrainingSample,
)
from paperquant.package import inspect_package, run_package
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/supervised.ridge_three_day_price"
FIXTURE = ROOT / "examples/public_ridge"
FIELDS = ("open", "high", "low", "close", "volume")
NAMES = tuple(f"{field}_lag{lag}" for lag in (2, 1, 0) for field in FIELDS)


def _learner():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("ridge_three_day", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RidgeThreeDayPrice()


def _inputs() -> tuple[DatasetDeclaration, tuple[MarketEvent, ...], TrainingRequest]:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def test_public_ridge_lineage_reconstructs_every_window_and_label() -> None:
    claim = json.loads((ROOT / "research/claims/ridge_three_day_price.json").read_text())
    assert claim["source"]["version"] == "arXiv:2310.09903v5"
    inspect_package("ridge-paper", PACKAGE)
    dataset, events, training = _inputs()
    source_dataset, source_events = load_plotly_apple(ROOT / "data/public/finance-charts-apple.csv")
    lineage = json.loads((FIXTURE / "lineage.json").read_text())
    assert lineage["source_file_sha256"] == APPLE_SHA256
    assert lineage["source_event_sha256"] == source_dataset.content_sha256
    assert lineage["training_sha256"] == fingerprint(training)
    assert lineage["evaluation_sha256"] == fingerprint(events) == dataset.content_sha256
    assert lineage["train_feature_end_rows"] == [2, 317]
    assert lineage["train_label_rows"] == [5, 320]
    assert lineage["evaluation_rows"] == [321, 505]
    assert events == source_events[321:]
    assert len(training.samples) == 316
    assert training.dataset_id == dataset.dataset_id
    for index, sample in enumerate(training.samples, start=2):
        assert sample.features == {
            f"{field}_lag{lag}": source_events[index - lag].values[field]
            for lag in (2, 1, 0)
            for field in FIELDS
        }
        assert sample.target == source_events[index + 3].values["close"]
    last_label_time = source_events[320].available_time
    first_open = events[0].bar_open_time
    assert first_open is not None and last_label_time < first_open
    assert datetime.fromisoformat(lineage["last_label_available_time"]) == last_label_time
    assert datetime.fromisoformat(lineage["first_evaluation_open_time"]) == first_open


def test_ridge_train_reload_predict_and_backtest(tmp_path: Path) -> None:
    dataset, events, training = _inputs()
    report = run_package(
        run_id="public-ridge",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
        training=training,
    )
    assert report.status == "succeeded"
    assert report.artifact is not None
    assert report.artifact.training_sha256 == report.plan.training_sha256 == fingerprint(training)
    assert len(report.decisions) == 185
    assert [decision.actions[0].kind for decision in report.decisions[:2]] == ["none", "none"]
    assert all(decision.actions[0].kind == "prediction" for decision in report.decisions[2:])
    assert len(report.fills) > 1
    assert all(fill.timestamp in {event.bar_open_time for event in events} for fill in report.fills)
    model = json.loads((tmp_path / report.artifact.relative_path).read_text())
    assert len(model["weights"]) == 15
    assert model["penalty"] == 1.0


@pytest.mark.parametrize("seed", [3, 17, 71])
def test_random_ridge_fit_satisfies_independent_normal_equations(seed: int) -> None:
    generator = random.Random(seed)
    samples = []
    for _ in range(80):
        features = {}
        for name in NAMES:
            base = 1000 if name.startswith("volume") else 100
            features[name] = Decimal(base + generator.randrange(-40, 41))
        outcome = Decimal(100 + generator.randrange(-25, 26))
        samples.append(TrainingSample(features=features, target=outcome))
    request = TrainingRequest(dataset_id=f"random-ridge-{seed}", seed=seed, samples=tuple(samples))
    model = json.loads(_learner().train(request))
    X = [
        [
            (float(sample.features[name]) - model["means"][column]) / model["scales"][column]
            for column, name in enumerate(NAMES)
        ]
        for sample in samples
    ]
    y = [float(sample.target) for sample in samples]
    weights = model["weights"]
    residuals = [
        model["target_mean"] + math.fsum(a * b for a, b in zip(row, weights, strict=True)) - target
        for row, target in zip(X, y, strict=True)
    ]
    assert abs(math.fsum(residuals)) < 1e-6
    for column, weight in enumerate(weights):
        gradient = math.fsum(error * row[column] for error, row in zip(residuals, X, strict=True))
        assert gradient + model["penalty"] * weight == pytest.approx(0, abs=1e-6)
    changed = request.model_copy(
        update={
            "samples": (
                samples[0].model_copy(update={"target": samples[0].target + Decimal(20)}),
                *samples[1:],
            )
        }
    )
    assert _learner().train(changed) != _learner().train(request)
