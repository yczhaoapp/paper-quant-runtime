from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import DatasetDeclaration, MarketEvent, RunPolicy, TrainingRequest
from paperquant.package import inspect_package, run_package
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/public_gaussian"
PACKAGE = ROOT / "strategies/supervised.gaussian_nb_direction"
OHLCV = ("open", "high", "low", "close", "volume")


def _inputs():  # type: ignore[no-untyped-def]
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def _features(values: dict[str, Decimal]) -> tuple[float, float, float, float]:
    opening, high, low, close, volume = (float(values[key]) for key in OHLCV)
    return (
        (close - opening) / opening,
        (high - low) / opening,
        (close - low) / (high - low) if high > low else 0.5,
        math.log1p(volume),
    )


def _posterior(model: dict[str, object], features: tuple[float, ...]) -> Decimal:
    scores = []
    for label in ("0", "1"):
        group = model[label]
        log_score = math.log(group["prior"])
        for value, mean, variance in zip(
            features, group["means"], group["variances"], strict=True
        ):
            log_score += -0.5 * math.log(2 * math.pi * variance)
            log_score += -0.5 * (value - mean) ** 2 / variance
        scores.append(log_score)
    top = max(scores)
    positive = math.exp(scores[1] - top)
    probability = positive / (math.exp(scores[0] - top) + positive)
    return Decimal(str(round(probability, 8)))


def test_public_labels_are_chronological_and_rebuildable() -> None:
    claim = json.loads((ROOT / "research/claims/gaussian_nb_direction.json").read_text())
    assert claim["source"]["version"] == "arXiv:2107.13148v3"
    inspect_package("gaussian-paper", PACKAGE)
    dataset, events, training = _inputs()
    original, source = load_plotly_apple(ROOT / "data/public/finance-charts-apple.csv")
    lineage = json.loads((FIXTURE / "lineage.json").read_text())
    assert lineage["source_file_sha256"] == APPLE_SHA256
    assert lineage["source_event_sha256"] == original.content_sha256
    assert lineage["training_sha256"] == fingerprint(training)
    assert lineage["evaluation_sha256"] == fingerprint(events) == dataset.content_sha256
    assert events == source[321:]
    assert len(training.samples) == 320
    for index, sample in enumerate(training.samples):
        assert sample.features == {key: source[index].values[key] for key in OHLCV}
        assert sample.target == Decimal(int(source[index + 1].values["close"]
                                            > source[index].values["close"]))
    assert source[320].available_time < events[0].bar_open_time
    assert (
        datetime.fromisoformat(lineage["last_label_available_time"])
        == source[320].available_time
    )
    assert datetime.fromisoformat(lineage["first_evaluation_open_time"]) == events[0].bar_open_time


@pytest.mark.parametrize("sample_count", [100, 200, 320])
def test_gaussian_training_and_posterior_match_independent_oracle(
    sample_count: int, tmp_path: Path
) -> None:
    dataset, events, complete = _inputs()
    training = complete.model_copy(update={"samples": complete.samples[:sample_count]})
    report = run_package(
        run_id=f"gaussian-{sample_count}",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
        training=training,
    )
    assert report.artifact is not None
    assert report.artifact.training_sha256 == fingerprint(training)
    assert len(report.fills) > 0
    model = json.loads((tmp_path / report.artifact.relative_path).read_text())
    assert set(model) == {"0", "1"}
    for label in (0, 1):
        rows = [_features(sample.features) for sample in training.samples
                if sample.target == Decimal(label)]
        assert len(rows) >= 2
        assert model[str(label)]["prior"] == pytest.approx(len(rows) / sample_count, abs=1e-12)
        for column in range(4):
            values = [row[column] for row in rows]
            assert model[str(label)]["means"][column] == pytest.approx(
                statistics.fmean(values), abs=1e-11
            )
            assert model[str(label)]["variances"][column] == pytest.approx(
                max(statistics.pvariance(values), 1e-8), abs=1e-11
            )
    expected = [_posterior(model, _features(event.values)) for event in events]
    actual = [decision.actions[0].value for decision in report.decisions]
    assert actual == expected
    assert min(actual) < Decimal("0.5") < max(actual)


def test_label_inversion_changes_both_class_distributions(tmp_path: Path) -> None:
    dataset, events, training = _inputs()
    inverted = training.model_copy(update={"samples": tuple(
        sample.model_copy(update={"target": Decimal(1) - sample.target})
        for sample in training.samples
    )})
    artifacts = []
    for name, request in (("original", training), ("inverted", inverted)):
        output = tmp_path / name
        report = run_package(
            run_id=name,
            package_dir=PACKAGE,
            dataset=dataset,
            engine_capability=reference_capability(dataset.fields),
            policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
            events=events,
            engine=ReferenceEngine(),
            output=output,
            training=request,
        )
        assert report.artifact is not None
        artifacts.append(json.loads((output / report.artifact.relative_path).read_text()))
    first, second = artifacts
    assert first["0"] == second["1"]
    assert first["1"] == second["0"]
