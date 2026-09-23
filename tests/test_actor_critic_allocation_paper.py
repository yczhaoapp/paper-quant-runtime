from __future__ import annotations

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
from paperquant.models import DatasetDeclaration, MarketEvent, RunPolicy, TrainingRequest
from paperquant.package import inspect_package, run_package
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/reinforcement.actor_critic_allocation"
FIXTURE = ROOT / "examples/public_actor_critic"
OHLCV = ("open", "high", "low", "close", "volume")


def _inputs() -> tuple[DatasetDeclaration, tuple[MarketEvent, ...], TrainingRequest]:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def _state(values: dict[str, Decimal]) -> tuple[float, float, float]:
    opening = float(values["open"])
    closing = float(values["close"])
    return (1.0, 10 * (closing - opening) / opening,
            10 * (float(values["high"]) - float(values["low"])) / opening)


def _reference_fit(training: TrainingRequest) -> tuple[list[float], list[float]]:
    actor = [0.0] * 3
    critic = [0.0] * 3
    generator = random.Random(training.seed)
    for sample in training.samples:
        assert sample.next_features is not None
        current = _state(sample.features)
        following = _state(sample.next_features)
        location = math.fsum(a * x for a, x in zip(actor, current, strict=True))
        sampled = generator.gauss(location, 0.2)
        risky = 1 / (1 + math.exp(-sampled))
        close = float(sample.features["close"])
        future = float(sample.next_features["close"])
        reward = math.log(1 + risky * (future / close - 1))
        old_value = math.fsum(v * x for v, x in zip(critic, current, strict=True))
        next_value = (0 if sample.terminal else
                      math.fsum(v * x for v, x in zip(critic, following, strict=True)))
        td_error = reward + next_value - old_value
        actor = [weight + td_error * (sampled - location) / 0.04 * component
                 for weight, component in zip(actor, current, strict=True)]
        critic = [weight + 0.1 * td_error * component
                  for weight, component in zip(critic, current, strict=True)]
    return actor, critic


def test_training_transition_lineage_is_disjoint_from_public_evaluation() -> None:
    claim = json.loads((ROOT / "research/claims/actor_critic_allocation.json").read_text())
    assert claim["source"]["version"] == "arXiv:1911.11880v2"
    inspect_package("actor-critic-paper", PACKAGE)
    dataset, events, training = _inputs()
    original, all_events = load_plotly_apple(ROOT / "data/public/finance-charts-apple.csv")
    lineage = json.loads((FIXTURE / "lineage.json").read_text())
    assert lineage["source_file_sha256"] == APPLE_SHA256
    assert lineage["source_event_sha256"] == original.content_sha256
    assert lineage["training_sha256"] == fingerprint(training)
    assert lineage["evaluation_sha256"] == fingerprint(events) == dataset.content_sha256
    assert events == all_events[321:]
    assert len(training.samples) == 320
    for index, sample in enumerate(training.samples):
        assert sample.features == {field: all_events[index].values[field] for field in OHLCV}
        assert sample.next_features == {field: all_events[index + 1].values[field]
                                        for field in OHLCV}
        assert sample.terminal == (index == 319)
        assert sample.action is None and sample.reward is None
    assert all_events[320].available_time < events[0].bar_open_time
    assert (
        datetime.fromisoformat(lineage["last_successor_available_time"])
        < events[0].bar_open_time
    )


@pytest.mark.parametrize("seed", [3, 17, 71])
def test_on_policy_actor_and_td_critic_match_independent_reference(
    seed: int, tmp_path: Path
) -> None:
    dataset, events, complete = _inputs()
    training = complete.model_copy(update={"seed": seed})
    report = run_package(
        run_id=f"actor-critic-{seed}",
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
    model = json.loads((tmp_path / report.artifact.relative_path).read_text())
    actor, critic = _reference_fit(training)
    assert model["actor"] == pytest.approx(actor, abs=1e-10)
    assert model["critic"] == pytest.approx(critic, abs=1e-10)
    assert len(report.fills) > 100
    weights = []
    for decision, account, source_event in zip(
        report.decisions, report.accounts, events, strict=True
    ):
        latent = math.fsum(a * x for a, x in zip(model["actor"],
                                                   _state(source_event.values), strict=True))
        weight = Decimal(str(round(1 / (1 + math.exp(-latent)), 10)))
        weights.append(weight)
        assert decision.actions[0].value == weight
        expected_shares = (weight * account.equity
                           / source_event.values["close"]).quantize(Decimal("0.000001"))
        assert decision.actions[1].quantity == expected_shares
    assert Decimal(0) < min(weights) < max(weights) < Decimal(1)


def test_gaussian_policy_score_gradient_has_expected_finite_difference() -> None:
    features = (1.0, 0.3, -0.2)
    weights = [0.2, -0.1, 0.4]
    sampled_latent = 0.37
    variance = 0.04
    for column in range(3):
        def log_density(theta: list[float]) -> float:
            center = sum(a * b for a, b in zip(theta, features, strict=True))
            return -0.5 * (sampled_latent - center) ** 2 / variance

        upper = weights.copy()
        lower = weights.copy()
        upper[column] += 1e-6
        lower[column] -= 1e-6
        numerical = (log_density(upper) - log_density(lower)) / 2e-6
        center = sum(a * b for a, b in zip(weights, features, strict=True))
        analytical = (sampled_latent - center) * features[column] / variance
        assert numerical == pytest.approx(analytical, abs=1e-9)


def test_successor_price_change_alters_fitted_policy(tmp_path: Path) -> None:
    dataset, events, training = _inputs()
    changed = list(training.samples)
    sample = changed[80]
    assert sample.next_features is not None
    successor = dict(sample.next_features)
    successor["close"] = successor["close"] * Decimal("1.01")
    successor["high"] = max(successor["high"], successor["close"])
    changed[80] = sample.model_copy(update={"next_features": successor})
    modified = training.model_copy(update={"samples": tuple(changed)})
    artifacts = []
    for name, request in (("original", training), ("changed", modified)):
        report = run_package(
            run_id=name,
            package_dir=PACKAGE,
            dataset=dataset,
            engine_capability=reference_capability(dataset.fields),
            policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
            events=events,
            engine=ReferenceEngine(),
            output=tmp_path / name,
            training=request,
        )
        assert report.artifact is not None
        artifacts.append(report.artifact.content_sha256)
    assert artifacts[0] != artifacts[1]
