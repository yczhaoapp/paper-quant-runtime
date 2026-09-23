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
from paperquant.models import DatasetDeclaration, MarketEvent, RunPolicy, RunReport, TrainingRequest
from paperquant.package import run_package
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/public_direction"
PACKAGE = ROOT / "strategies/supervised.logistic_direction"
FEATURES = ("open", "high", "low", "close", "volume")


def _inputs() -> tuple[DatasetDeclaration, tuple[MarketEvent, ...], TrainingRequest]:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def _run(tmp_path: Path, training: TrainingRequest) -> RunReport:
    dataset, events, _ = _inputs()
    return run_package(
        run_id="public-direction",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
        training=training,
    )


def test_training_labels_and_split_reconstruct_from_pinned_raw_market() -> None:
    claim = json.loads((ROOT / "research/claims/logistic_direction.json").read_text())
    assert claim["source"]["version"] == "arXiv:2310.16855v1"
    dataset, events, training = _inputs()
    source_declaration, source_events = load_plotly_apple(
        ROOT / "data/public/finance-charts-apple.csv"
    )
    lineage = json.loads((FIXTURE / "lineage.json").read_text())

    assert lineage["source_file_sha256"] == APPLE_SHA256
    assert lineage["source_event_sha256"] == source_declaration.content_sha256
    assert lineage["training_sha256"] == fingerprint(training)
    assert lineage["evaluation_sha256"] == fingerprint(events) == dataset.content_sha256
    assert lineage["train_feature_rows"] == [0, 319]
    assert lineage["train_label_rows"] == [1, 320]
    assert lineage["evaluation_rows"] == [321, 505]
    assert events == source_events[321:]
    assert len(training.samples) == 320
    assert training.dataset_id == dataset.dataset_id

    for index, sample in enumerate(training.samples):
        assert sample.features == {name: source_events[index].values[name] for name in FEATURES}
        assert sample.target == Decimal(
            int(source_events[index + 1].values["close"] > source_events[index].values["close"])
        )
    last_label = source_events[320].available_time
    first_evaluation_open = events[0].bar_open_time
    assert first_evaluation_open is not None
    assert last_label < first_evaluation_open
    assert datetime.fromisoformat(lineage["last_label_available_time"]) == last_label
    assert datetime.fromisoformat(lineage["first_evaluation_open_time"]) == first_evaluation_open


def test_public_logistic_training_reload_and_trades(tmp_path: Path) -> None:
    _, events, training = _inputs()
    report = _run(tmp_path, training)
    assert report.status == "succeeded"
    assert len(report.decisions) == len(events) == 185
    assert len(report.fills) > 1
    assert report.artifact is not None
    assert report.artifact.training_sha256 == fingerprint(training)
    assert report.plan.training_sha256 == report.artifact.training_sha256
    model = json.loads((tmp_path / report.artifact.relative_path).read_text())
    assert len(model["weights"]) == 6
    probabilities = [decision.actions[0].value for decision in report.decisions]
    assert all(Decimal(0) < value < Decimal(1) for value in probabilities)
    assert max(probabilities) - min(probabilities) > Decimal("0.05")
    assert any(value < Decimal("0.5") for value in probabilities)
    assert any(value > Decimal("0.5") for value in probabilities)
    for fill in report.fills:
        assert fill.timestamp in {event.bar_open_time for event in events}


@pytest.mark.parametrize("portion", [2, 3, 5])
def test_training_is_label_sensitive_and_reduces_log_loss(portion: int, tmp_path: Path) -> None:
    _, _, complete = _inputs()
    subset = complete.model_copy(update={"samples": complete.samples[: 60 * portion]})
    inverted = subset.model_copy(
        update={
            "samples": tuple(
                sample.model_copy(update={"target": Decimal(1) - sample.target})
                for sample in subset.samples
                if sample.target is not None
            )
        }
    )
    original_report = _run(tmp_path / "original", subset)
    inverted_report = _run(tmp_path / "inverted", inverted)
    assert original_report.artifact is not None
    assert inverted_report.artifact is not None
    original = json.loads(
        (tmp_path / "original" / original_report.artifact.relative_path).read_text()
    )
    opposite = json.loads(
        (tmp_path / "inverted" / inverted_report.artifact.relative_path).read_text()
    )
    assert original["means"] == opposite["means"]
    assert original["scales"] == opposite["scales"]
    for weight, inverse_weight in zip(original["weights"], opposite["weights"], strict=True):
        assert weight == pytest.approx(-inverse_weight, abs=1e-10)
    assert original_report.artifact.content_sha256 != inverted_report.artifact.content_sha256

    rows = [[float(sample.features[name]) for name in FEATURES] for sample in subset.samples]
    columns = list(zip(*rows, strict=True))
    independent_means = [statistics.fmean(column) for column in columns]
    independent_scales = [max(statistics.pstdev(column), 1e-12) for column in columns]
    assert original["means"] == pytest.approx(independent_means, abs=1e-10)
    assert original["scales"] == pytest.approx(independent_scales, abs=1e-10)
    design = [
        [1.0] + [
            (row[index] - independent_means[index]) / independent_scales[index]
            for index in range(5)
        ]
        for row in rows
    ]
    labels = [float(sample.target) for sample in subset.samples]
    reference = [0.0] * 6
    for _ in range(1000):
        estimates = [
            1.0 / (1.0 + math.exp(-math.fsum(a * b for a, b in zip(reference, row, strict=True))))
            for row in design
        ]
        residuals = [estimate - label for estimate, label in zip(estimates, labels, strict=True)]
        reference = [
            weight
            - 0.01
            * math.fsum(error * row[column] for error, row in zip(residuals, design, strict=True))
            / len(design)
            for column, weight in enumerate(reference)
        ]
    assert original["weights"] == pytest.approx(reference, abs=1e-10)

    def probability(sample_features: dict[str, Decimal]) -> float:
        standardized = [1.0] + [
            (float(sample_features[name]) - original["means"][index])
            / original["scales"][index]
            for index, name in enumerate(FEATURES)
        ]
        score = sum(a * b for a, b in zip(original["weights"], standardized, strict=True))
        return 1.0 / (1.0 + math.exp(-score))

    actual_loss = -sum(
        float(sample.target) * math.log(probability(sample.features))
        + (1.0 - float(sample.target)) * math.log1p(-probability(sample.features))
        for sample in subset.samples
        if sample.target is not None
    ) / len(subset.samples)
    assert actual_loss < math.log(2)
