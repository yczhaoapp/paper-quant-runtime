from __future__ import annotations

import importlib.util
import json
import math
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import (
    AccountSnapshot,
    DatasetDeclaration,
    MarketEvent,
    Position,
    RunPolicy,
    TrainingRequest,
)
from paperquant.package import inspect_package, run_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/reinforcement.risk_averse_bandit"
FIXTURE = ROOT / "examples/bandit_l1"


def _policy():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("bandit_policy", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RiskAverseBandit()


def _inputs() -> tuple[DatasetDeclaration, tuple[MarketEvent, ...], TrainingRequest]:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def _reference_update(parameters: dict[str, float], context: float, reward: float) -> None:
    before_a = parameters["a"]
    before_b = parameters["b"]
    parameters["a"] += context**2
    parameters["b"] += context * reward
    parameters["c"] += 0.5
    parameters["d"] += (reward**2 + before_b**2 / before_a
                         - parameters["b"]**2 / parameters["a"]) / 2


@pytest.mark.parametrize("seed", [3, 29, 71])
def test_scalar_normal_gamma_updates_and_thompson_scores_match_reference(seed: int) -> None:
    claim = json.loads((ROOT / "research/claims/risk_averse_bandit.json").read_text())
    assert claim["source"]["version"] == "arXiv:2206.12463v1"
    inspect_package("bandit-paper", PACKAGE)
    _, _, complete = _inputs()
    training = complete.model_copy(update={"seed": seed})
    policy = _policy()
    model = json.loads(policy.train(training))
    expected = {str(arm): {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0}
                for arm in (-1, 0, 1)}
    for sample in training.samples:
        assert sample.reward is not None and sample.action is not None
        _reference_update(expected[str(sample.action)],
                          float(sample.features["context"]), float(sample.reward))
    for arm in (-1, 0, 1):
        for field in ("a", "b", "c", "d"):
            assert model["posterior"][str(arm)][field] == pytest.approx(
                expected[str(arm)][field], abs=1e-12
            )
    policy.load(json.dumps(model).encode())
    independent = random.Random(seed)
    context = 0.37
    scores = policy._sample_scores(context)
    for arm in (-1, 0, 1):
        parameters = expected[str(arm)]
        precision = independent.gammavariate(parameters["c"], 1 / parameters["d"])
        mean = independent.gauss(parameters["b"] / parameters["a"],
                                 math.sqrt(1 / (precision * parameters["a"])))
        assert scores[arm] == pytest.approx(context * mean - 0.1 / precision, abs=1e-12)


def test_online_update_uses_arm_held_during_observed_interval() -> None:
    _, events, training = _inputs()
    policy = _policy()
    policy.load(policy.train(training))
    initial = json.loads(json.dumps(policy.posterior))
    instant = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)

    def account(quantity: int, offset: int) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp=instant + timedelta(seconds=offset),
            cash=Decimal(100000),
            equity=Decimal(100000),
            positions=((Position(symbol="DEMO", quantity=Decimal(quantity),
                                 average_price=Decimal(100), realized_pnl=Decimal(0),
                                 unrealized_pnl=Decimal(0)),) if quantity else ()),
            open_order_ids=(),
        )

    policy.decide(events[0], account(0, 0))
    policy.decide(events[1], account(1, 1))
    after_first = json.loads(json.dumps(policy.posterior))
    assert after_first["1"] == initial["1"]
    assert after_first["0"]["c"] == initial["0"]["c"] + 0.5
    context = (float(events[1].values["bid_size"]) - float(events[1].values["ask_size"])) / 100
    first_mid = float((events[1].values["bid_price"] + events[1].values["ask_price"]) / 2)
    next_mid = float((events[2].values["bid_price"] + events[2].values["ask_price"]) / 2)
    reference = after_first["1"].copy()
    _reference_update(reference, context, (next_mid - first_mid) / first_mid)
    policy.decide(events[2], account(1, 2))
    for field in reference:
        assert policy.posterior["1"][field] == pytest.approx(reference[field], abs=1e-12)
    assert policy.posterior["-1"] == initial["-1"]


def test_risk_penalty_lowers_sampled_scores_with_same_random_draws() -> None:
    _, _, training = _inputs()
    payload = _policy().train(training)
    mild = _policy()
    strong = _policy()
    mild.load(payload)
    strong.load(payload)
    strong.RISK_TOLERANCE = 2.0
    low_risk_scores = mild._sample_scores(0.4)
    high_risk_scores = strong._sample_scores(0.4)
    assert all(high_risk_scores[arm] < low_risk_scores[arm] for arm in (-1, 0, 1))


def test_bandit_lifecycle_and_online_actions(tmp_path: Path) -> None:
    dataset, events, training = _inputs()
    report = run_package(
        run_id="risk-averse-bandit",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
        training=training,
    )
    assert report.artifact is not None
    assert report.artifact.training_sha256 == fingerprint(training)
    assert len(report.decisions) == len(events) == 40
    assert len(report.fills) > 10
    assert {fill.side for fill in report.fills} == {"buy", "sell"}
    selected = {decision.actions[0].reason.rsplit(" ", 1)[-1]
                for decision in report.decisions}
    assert selected == {"-1", "0", "1"}
    assert all(fill.timestamp in {event.available_time for event in events}
               for fill in report.fills)
