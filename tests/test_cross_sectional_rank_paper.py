from __future__ import annotations

import json
import runpy
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

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/cross_sectional_rank"
PACKAGE = ROOT / "strategies/supervised.cross_sectional_rank"
SYMBOLS = ("ALPHA", "BETA", "GAMMA")
FEATURES = ("signal", "liquidity")


def _inputs() -> tuple[DatasetDeclaration, tuple[MarketEvent, ...], TrainingRequest]:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def _run(tmp_path: Path, training: TrainingRequest):  # type: ignore[no-untyped-def]
    dataset, events, _ = _inputs()
    return run_package(
        run_id="cross-sectional-rank",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
        training=training,
    )


def test_panel_rebuilds_and_training_precedes_all_evaluation_opens() -> None:
    claim = json.loads((ROOT / "research/claims/cross_sectional_rank.json").read_text())
    assert claim["claims"][0]["locator"] == "Section 2.3, equation (8), PDF page 12"
    inspect_package("cross-sectional-paper", PACKAGE)
    dataset, events, training = _inputs()
    panel = runpy.run_path(str(ROOT / "scripts/build_cross_sectional_fixture.py"))["build_panel"]()
    lineage = json.loads((FIXTURE / "lineage.json").read_text())
    assert lineage["source_panel_sha256"] == fingerprint(panel)
    assert lineage["training_sha256"] == fingerprint(training)
    assert lineage["evaluation_sha256"] == fingerprint(events) == dataset.content_sha256
    assert events == panel[121 * 3:]
    assert len(training.samples) == 120 * 3
    for day in range(120):
        for asset in range(3):
            sample = training.samples[day * 3 + asset]
            current = panel[day * 3 + asset]
            following = panel[(day + 1) * 3 + asset]
            assert sample.features == {field: current.values[field] for field in FEATURES}
            assert sample.target == following.values["close"] / current.values["close"] - 1
    assert panel[120 * 3].available_time < events[0].bar_open_time
    assert datetime.fromisoformat(lineage["last_label_available_time"]) < events[0].bar_open_time


@pytest.mark.parametrize("count", [120, 240, 360])
def test_elastic_net_kkt_and_forecast_rank_oracle(count: int, tmp_path: Path) -> None:
    _, events, complete = _inputs()
    training = complete.model_copy(update={"samples": complete.samples[:count]})
    report = _run(tmp_path, training)
    assert report.artifact is not None
    assert report.artifact.training_sha256 == fingerprint(training)
    assert len(report.fills) > 10
    model = json.loads((tmp_path / report.artifact.relative_path).read_text())
    raw = [[float(sample.features[field]) for field in FEATURES]
           for sample in training.samples]
    targets = [float(sample.target) for sample in training.samples]
    means = [statistics.fmean(row[column] for row in raw) for column in range(2)]
    scales = [max(statistics.pstdev(row[column] for row in raw), 1e-12)
              for column in range(2)]
    assert model["means"] == pytest.approx(means, abs=1e-10)
    assert model["scales"] == pytest.approx(scales, abs=1e-10)
    assert model["intercept"] == pytest.approx(statistics.fmean(targets), abs=1e-10)
    standardized = [[(row[column] - means[column]) / scales[column]
                     for column in range(2)] for row in raw]
    residuals = [target - model["intercept"]
                 - sum(a * b for a, b in zip(row, model["weights"], strict=True))
                 for target, row in zip(targets, standardized, strict=True)]
    for column, weight in enumerate(model["weights"]):
        gradient = statistics.fmean(row[column] * residual
                                    for row, residual in zip(standardized, residuals, strict=True))
        signed_l1 = 0.0005 if weight > 0 else -0.0005 if weight < 0 else 0.0
        if weight:
            assert gradient == pytest.approx(signed_l1 + 0.0005 * weight, abs=1e-10)
        else:
            assert abs(gradient) <= 0.0005 + 1e-10

    for day in range(len(events) // 3):
        day_events = events[day * 3 : day * 3 + 3]
        day_decisions = report.decisions[day * 3 : day * 3 + 3]
        predictions = {}
        for event, decision in zip(day_events, day_decisions, strict=True):
            vector = [float(event.values[field]) for field in FEATURES]
            score = model["intercept"] + sum(
                model["weights"][column] * (vector[column] - model["means"][column])
                / model["scales"][column]
                for column in range(2)
            )
            assert decision.actions[0].value == Decimal(str(round(score, 10)))
            predictions[event.symbol] = score
        low = min(SYMBOLS, key=lambda symbol: (predictions[symbol], symbol))
        high = max(SYMBOLS, key=lambda symbol: (predictions[symbol], symbol))
        account = report.accounts[day * 3 + 2]
        positions = {position.symbol: position.quantity for position in account.positions}
        desired = {symbol: Decimal(0) for symbol in SYMBOLS}
        desired[low], desired[high] = Decimal(-1), Decimal(1)
        actual_targets = {action.symbol: action.quantity
                          for action in day_decisions[2].actions[1:]
                          if action.kind == "target_position"}
        assert actual_targets == {symbol: amount for symbol, amount in desired.items()
                                  if positions.get(symbol, Decimal(0)) != amount}
        assert all(decision.actions[1].kind == "none" for decision in day_decisions[:2])


def test_return_sign_inversion_changes_fit_and_rank(tmp_path: Path) -> None:
    _, _, training = _inputs()
    opposite = training.model_copy(update={"samples": tuple(
        sample.model_copy(update={"target": -sample.target}) for sample in training.samples
    )})
    first = _run(tmp_path / "first", training)
    second = _run(tmp_path / "second", opposite)
    assert first.artifact is not None and second.artifact is not None
    original = json.loads((tmp_path / "first" / first.artifact.relative_path).read_text())
    inverted = json.loads((tmp_path / "second" / second.artifact.relative_path).read_text())
    assert inverted["means"] == original["means"]
    assert inverted["scales"] == original["scales"]
    assert inverted["weights"] == pytest.approx([-weight for weight in original["weights"]],
                                                  abs=1e-10)
    assert first.artifact.content_sha256 != second.artifact.content_sha256
    assert first.decisions != second.decisions
