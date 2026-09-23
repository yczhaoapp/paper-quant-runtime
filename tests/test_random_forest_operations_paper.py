from __future__ import annotations

import json
import math
import statistics
from collections import Counter
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
FIXTURE = ROOT / "examples/public_forest"
PACKAGE = ROOT / "strategies/supervised.random_forest_operations"
OHLCV = ("open", "high", "low", "close", "volume")


def _inputs() -> tuple[DatasetDeclaration, tuple[MarketEvent, ...], TrainingRequest]:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def _independent_label(closes: tuple[Decimal, ...], index: int) -> int:
    entry = closes[index]
    limits = {1: (Decimal("1.03"), Decimal("0.985")),
              -1: (Decimal("0.97"), Decimal("1.015"))}
    active = {1, -1}
    for future in closes[index + 1 : index + 6]:
        relative = future / entry
        for side in (1, -1):
            if side not in active:
                continue
            gain, loss = limits[side]
            if (side == 1 and relative >= gain) or (side == -1 and relative <= gain):
                return side
            if (side == 1 and relative <= loss) or (side == -1 and relative >= loss):
                active.remove(side)
    return 0


def _features(values: dict[str, Decimal]) -> tuple[float, float, float]:
    opening = float(values["open"])
    return (
        (float(values["close"]) - opening) / opening,
        (float(values["high"]) - float(values["low"])) / opening,
        math.log1p(float(values["volume"])),
    )


def _vote(model: dict[str, object], values: dict[str, Decimal]) -> int:
    features = _features(values)
    votes = Counter()
    for tree in model["trees"]:
        node = tree
        while "leaf" not in node:
            node = node["left" if features[node["feature"]] <= node["threshold"]
                        else "right"]
        votes[node["leaf"]] += 1
    return max((-1, 0, 1), key=lambda label: (votes[label], -abs(label), -label))


def _run(tmp_path: Path, training: TrainingRequest):  # type: ignore[no-untyped-def]
    dataset, events, _ = _inputs()
    return run_package(
        run_id="forest-operations",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
        training=training,
    )


def test_operation_labels_and_chronology_rebuild_from_public_bars() -> None:
    claim = json.loads((ROOT / "research/claims/random_forest_operations.json").read_text())
    assert claim["source"]["version"] == "arXiv:1301.4944v1"
    inspect_package("forest-paper", PACKAGE)
    dataset, events, training = _inputs()
    original, source = load_plotly_apple(ROOT / "data/public/finance-charts-apple.csv")
    closes = tuple(event.values["close"] for event in source)
    lineage = json.loads((FIXTURE / "lineage.json").read_text())
    assert lineage["source_file_sha256"] == APPLE_SHA256
    assert lineage["source_event_sha256"] == original.content_sha256
    assert lineage["training_sha256"] == fingerprint(training)
    assert lineage["evaluation_sha256"] == fingerprint(events) == dataset.content_sha256
    assert events == source[321:]
    assert len(training.samples) == 316
    for index, sample in enumerate(training.samples):
        assert sample.features == {field: source[index].values[field] for field in OHLCV}
        assert sample.target == Decimal(_independent_label(closes, index))
    assert Counter(sample.target for sample in training.samples).keys() == {
        Decimal(-1), Decimal(0), Decimal(1)
    }
    assert source[320].available_time < events[0].bar_open_time
    assert datetime.fromisoformat(lineage["last_label_available_time"]) < events[0].bar_open_time


@pytest.mark.parametrize("seed", [5, 23, 73])
def test_bootstrap_forest_votes_match_independent_tree_traversal(
    seed: int, tmp_path: Path
) -> None:
    _, events, complete = _inputs()
    training = complete.model_copy(update={"seed": seed})
    report = _run(tmp_path, training)
    assert report.artifact is not None
    assert report.artifact.training_sha256 == fingerprint(training)
    assert len(report.fills) > 0
    model = json.loads((tmp_path / report.artifact.relative_path).read_text())
    assert len(model["trees"]) == 15
    for tree in model["trees"]:
        stack = [(tree, 0)]
        while stack:
            node, depth = stack.pop()
            assert depth <= 4
            if "leaf" in node:
                assert node["leaf"] in (-1, 0, 1)
            else:
                assert node["feature"] in (0, 1, 2)
                assert math.isfinite(node["threshold"])
                stack.extend(((node["left"], depth + 1), (node["right"], depth + 1)))
    actual = [int(decision.actions[0].value) for decision in report.decisions]
    expected = [_vote(model, event.values) for event in events]
    assert actual == expected
    assert len(set(actual)) >= 2
    assert all(fill.timestamp in {event.bar_open_time for event in events}
               for fill in report.fills)


def test_training_labels_and_seed_change_forest_artifact(tmp_path: Path) -> None:
    _, _, training = _inputs()
    flipped = training.model_copy(update={"samples": tuple(
        sample.model_copy(update={"target": -sample.target}) for sample in training.samples
    )})
    reports = [_run(tmp_path / name, request) for name, request in (
        ("first", training), ("flipped", flipped),
        ("different-seed", training.model_copy(update={"seed": 71})),
    )]
    assert all(report.artifact is not None for report in reports)
    assert len({report.artifact.content_sha256 for report in reports}) == 3
    forecasts = [[decision.actions[0].value for decision in report.decisions]
                 for report in reports]
    assert forecasts[0] != forecasts[1]
    assert statistics.mean(int(value) for value in forecasts[0]) != statistics.mean(
        int(value) for value in forecasts[1]
    )
