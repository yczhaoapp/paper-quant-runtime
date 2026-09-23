from __future__ import annotations

import importlib.util
import json
import math
import random
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
PACKAGE = ROOT / "strategies/reinforcement.pairs_actor_critic"
FIXTURE = ROOT / "examples/pairs_actor_critic"


def _strategy_class():
    spec = importlib.util.spec_from_file_location("pair_under_test", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PairsActorCritic


def _inputs():
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes())
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def _reference_fit(request):
    policy = [[0.0 for _ in range(5)] for _ in range(3)]
    values = [0.0] * 5
    generator = random.Random(request.seed)

    def features(row, position):
        left = float(row["left_price"])
        return [1.0, position, 10 * float(row["spread"]) / left,
                min(3, max(-3, float(row["z_score"]))) / 3, float(row["zone"]) / 2]

    for _ in range(5):
        position = 0
        for transition in request.samples:
            assert transition.next_features is not None
            current = features(transition.features, position)
            logits = [math.fsum(w*x for w, x in zip(row, current, strict=True))
                      for row in policy]
            weights = [math.exp(value - max(logits)) for value in logits]
            distribution = [weight / sum(weights) for weight in weights]
            draw = generator.random()
            index = (0 if draw < distribution[0] else
                     1 if draw < distribution[0] + distribution[1] else 2)
            action = (-1, 0, 1)[index]
            following = features(transition.next_features, action)
            a = float(transition.features["left_price"])
            b = float(transition.features["right_price"])
            a_next = float(transition.next_features["left_price"])
            b_next = float(transition.next_features["right_price"])
            pnl = action * ((a_next-a)/a - (b_next-b)/b) / 2
            reward = pnl - 0.0002 * abs(action-position)
            baseline = math.fsum(v*x for v, x in zip(values, current, strict=True))
            next_value = 0 if transition.terminal else 0.95 * math.fsum(
                v*x for v, x in zip(values, following, strict=True))
            advantage = reward + next_value - baseline
            for row_index, row in enumerate(policy):
                for column in range(5):
                    row[column] += 0.5 * advantage * (
                        int(row_index == index) - distribution[row_index]) * current[column]
            for column in range(5):
                values[column] += 0.2 * advantage * current[column]
            position = 0 if transition.terminal else action
    return policy, values


def test_paper_binding_and_disjoint_chronology() -> None:
    claim = json.loads((ROOT / "research/claims/pairs_actor_critic.json").read_text())
    assert claim["source"]["url"] == "https://arxiv.org/pdf/2407.16103"
    inspect_package("pair-paper", PACKAGE)
    dataset, events, request = _inputs()
    lineage = json.loads((FIXTURE / "lineage.json").read_text())
    assert lineage["training_sha256"] == fingerprint(request)
    assert lineage["evaluation_sha256"] == fingerprint(events) == dataset.content_sha256
    assert len(request.samples) == 119 and len(events) == 200
    assert all(sample.action is None and sample.reward is None for sample in request.samples)
    assert datetime.fromisoformat(lineage["last_training_successor_available_time"]) < (
        events[0].bar_open_time)
    assert all(left.symbol == "ALPHA" and right.symbol == "BETA"
               for left, right in zip(events[::2], events[1::2], strict=True))


@pytest.mark.parametrize("seed", [7, 23, 91])
def test_discrete_a2c_updates_match_independent_reference(seed: int, tmp_path: Path) -> None:
    dataset, events, original = _inputs()
    request = original.model_copy(update={"seed": seed})
    report = run_package(
        run_id=f"pair-a2c-{seed}", package_dir=PACKAGE, dataset=dataset,
        engine_capability=reference_capability(dataset.fields), policy=RunPolicy(),
        events=events, engine=ReferenceEngine(), output=tmp_path, training=request,
    )
    assert report.artifact is not None
    assert report.artifact.training_sha256 == fingerprint(request)
    model = json.loads((tmp_path / report.artifact.relative_path).read_text())
    policy, values = _reference_fit(request)
    for actual, expected in zip(model["actor"], policy, strict=True):
        assert actual == pytest.approx(expected, abs=1e-10)
    assert model["critic"] == pytest.approx(values, abs=1e-10)
    assert len(report.fills) > 20
    pair_decisions = [d.actions for d in report.decisions if d.actions[0].kind == "prediction"]
    assert len(pair_decisions) == 80
    assert {-1, 1} <= {int(actions[0].value) for actions in pair_decisions}
    for actions in pair_decisions:
        assert actions[1].symbol == "ALPHA" and actions[2].symbol == "BETA"
        assert actions[1].quantity * actions[2].quantity <= 0


def test_causal_rolling_regression_matches_independent_statistics() -> None:
    strategy = _strategy_class()
    for phase in (0, 2, 5, 11):
        xs = [Decimal(str(100 + 0.7*i + 0.2*math.sin(i+phase))) for i in range(21)]
        ys = [Decimal(str(4 + 1.3*float(x) + 0.3*math.cos(i/2+phase)))
              for i, x in enumerate(xs)]
        fit = statistics.linear_regression([float(x) for x in xs[:-1]],
                                           [float(y) for y in ys[:-1]])
        residuals = [float(y) - fit.intercept - fit.slope*float(x)
                     for x, y in zip(xs[:-1], ys[:-1], strict=True)]
        spread = float(ys[-1]) - fit.intercept - fit.slope*float(xs[-1])
        expected_z = (spread-statistics.mean(residuals))/statistics.pstdev(residuals)
        observed_spread, observed_z, _ = strategy.observation(ys, xs)
        assert observed_spread == pytest.approx(spread, abs=1e-9)
        assert observed_z == pytest.approx(expected_z, abs=1e-8)
        changed_future = ys + [Decimal("999")]
        assert strategy.observation(ys, xs) == strategy.observation(
            changed_future[:-1], xs)


def test_softmax_score_property_and_reward_sensitivity(tmp_path: Path) -> None:
    state = [1.0, -1.0, 0.15, 0.4, 0.5]
    actor = [[0.1, -0.2, 0.3, 0.0, -0.1],
             [-0.1, 0.1, 0.0, 0.2, 0.1],
             [0.2, 0.0, -0.1, -0.3, 0.0]]
    for chosen in range(3):
        logits = [sum(a*b for a, b in zip(row, state, strict=True)) for row in actor]
        weights = [math.exp(v-max(logits)) for v in logits]
        probabilities = [w/sum(weights) for w in weights]
        for row in range(3):
            perturbed = [list(weights_row) for weights_row in actor]
            perturbed[row][3] += 1e-6
            upper = [sum(a*b for a, b in zip(r, state, strict=True)) for r in perturbed]
            log_upper = upper[chosen] - math.log(sum(math.exp(v) for v in upper))
            perturbed[row][3] -= 2e-6
            lower = [sum(a*b for a, b in zip(r, state, strict=True)) for r in perturbed]
            log_lower = lower[chosen] - math.log(sum(math.exp(v) for v in lower))
            analytical = (int(row == chosen)-probabilities[row])*state[3]
            assert (log_upper-log_lower)/2e-6 == pytest.approx(analytical, abs=1e-9)

    dataset, events, request = _inputs()
    changed = list(request.samples)
    sample = changed[31]
    assert sample.next_features is not None
    successor = dict(sample.next_features)
    successor["left_price"] *= Decimal("1.05")
    changed[31] = sample.model_copy(update={"next_features": successor})
    altered = request.model_copy(update={"samples": tuple(changed)})
    digests = []
    for name, training in (("original", request), ("changed", altered)):
        report = run_package(
            run_id=name, package_dir=PACKAGE, dataset=dataset,
            engine_capability=reference_capability(dataset.fields), policy=RunPolicy(),
            events=events, engine=ReferenceEngine(), output=tmp_path/name,
            training=training,
        )
        assert report.artifact is not None
        digests.append(report.artifact.content_sha256)
    assert digests[0] != digests[1]
