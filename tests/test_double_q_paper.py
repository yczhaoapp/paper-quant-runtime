from __future__ import annotations

import importlib.util
import json
import random
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperquant.cli import main
from paperquant.compiler import fingerprint
from paperquant.models import TrainingRequest, TrainingSample
from paperquant.package import inspect_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/reinforcement.double_q_market_making"
FIXTURE = ROOT / "examples/l1_queues"
ACTIONS = (0, 1, -1)


def learner(alpha: Decimal, gamma: Decimal):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("double_q_policy", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DoubleQInventory(alpha=alpha, gamma=gamma)


def features(pressure: int, inventory: int) -> dict[str, Decimal]:
    bid, ask = (75, 25) if pressure > 0 else (25, 75)
    return {
        "bid_size": Decimal(bid),
        "ask_size": Decimal(ask),
        "inventory": Decimal(inventory),
    }


def seed_with_choices(choices: tuple[bool, ...]) -> int:
    for seed in range(10000):
        generator = random.Random(seed)
        if tuple(generator.randrange(2) == 0 for _ in choices) == choices:
            return seed
    raise AssertionError("No seed realizes the requested independent table choices")


@pytest.mark.parametrize(
    ("alpha", "gamma"),
    [
        (Decimal("0.25"), Decimal("0")),
        (Decimal("0.25"), Decimal("0.8")),
        (Decimal("0.5"), Decimal("1")),
    ],
)
def test_double_q_selects_with_one_table_and_evaluates_with_other(
    alpha: Decimal, gamma: Decimal
) -> None:
    successor = features(1, 0)
    current = features(-1, 0)
    transitions = (
        TrainingSample(features=successor, action=1, reward=Decimal(10), terminal=True),
        TrainingSample(features=successor, action=0, reward=Decimal(3), terminal=True),
        TrainingSample(features=successor, action=1, reward=Decimal(-4), terminal=True),
        TrainingSample(features=successor, action=0, reward=Decimal(7), terminal=True),
        TrainingSample(
            features=current,
            action=-1,
            reward=Decimal(2),
            next_features=successor,
        ),
    )
    seeds = (
        seed_with_choices((True, True, False, False, True)),
        seed_with_choices((False, False, True, True, False)),
    )
    models = [
        json.loads(
            learner(alpha, gamma).train(
                TrainingRequest(dataset_id="double-oracle", seed=seed, samples=transitions)
            )
        )
        for seed in seeds
    ]
    expected = alpha * (Decimal(2) - gamma * alpha * Decimal(4))
    assert Decimal(models[0]["qa"]["-1:0:-1"]) == expected
    assert Decimal(models[1]["qb"]["-1:0:-1"]) == expected
    assert models[0]["qa"] == models[1]["qb"]
    assert models[0]["qb"] == models[1]["qa"]
    if alpha > 0 and gamma > 0:
        assert expected != alpha * (Decimal(2) + gamma * alpha * Decimal(7))
        assert expected != alpha * (Decimal(2) + gamma * alpha * Decimal(10))


@pytest.mark.parametrize("seed", [2, 19, 71])
def test_seeded_random_transitions_match_independent_double_bellman_oracle(seed: int) -> None:
    generator = random.Random(seed * 13)
    samples = []
    states = [features(pressure, inventory) for pressure in (-1, 1) for inventory in (-1, 0, 1)]
    for _ in range(45):
        terminal = generator.randrange(4) == 0
        samples.append(
            TrainingSample(
                features=generator.choice(states),
                action=generator.choice(ACTIONS),
                reward=Decimal(generator.randrange(-6, 7)),
                next_features=None if terminal else generator.choice(states),
                terminal=terminal,
            )
        )
    alpha = Decimal("0.3")
    gamma = Decimal("0.7")
    request = TrainingRequest(dataset_id=f"random-{seed}", seed=seed, samples=tuple(samples))
    model = json.loads(learner(alpha, gamma).train(request))

    tables: tuple[dict[str, Decimal], dict[str, Decimal]] = ({}, {})
    branches = random.Random(seed)

    def state(item: dict[str, Decimal]) -> str:
        direction = 1 if item["bid_size"] > item["ask_size"] else -1
        holding = int(item["inventory"])
        return f"{direction}:{holding}"

    for sample in samples:
        selected_index = branches.randrange(2)
        selected = tables[selected_index]
        other = tables[1 - selected_index]
        key = f"{state(sample.features)}:{sample.action}"
        old = selected.get(key, Decimal(0))
        future = Decimal(0)
        if not sample.terminal:
            assert sample.next_features is not None
            successor = state(sample.next_features)
            best = max(
                ACTIONS,
                key=lambda action: selected.get(f"{successor}:{action}", Decimal(0)),
            )
            future = other.get(f"{successor}:{best}", Decimal(0))
        assert sample.reward is not None
        selected[key] = (Decimal(1) - alpha) * old + alpha * (sample.reward + gamma * future)

    assert {key: Decimal(value) for key, value in model["qa"].items()} == tables[0]
    assert {key: Decimal(value) for key, value in model["qb"].items()} == tables[1]
    assert tables[0] != tables[1]


@pytest.mark.parametrize("seed", [3, 17, 71])
def test_behavior_action_uses_combined_tables_and_is_swap_invariant(seed: int) -> None:
    generator = random.Random(seed)
    event = SimpleNamespace(
        symbol="DEMO",
        values={"price": Decimal("100"), "bid_size": Decimal("75"),
                "ask_size": Decimal("25")},
    )
    account = SimpleNamespace(positions=())
    for _ in range(40):
        qa = {f"1:0:{action}": Decimal(generator.randrange(-20, 21))
              for action in ACTIONS}
        qb = {f"1:0:{action}": Decimal(generator.randrange(-20, 21))
              for action in ACTIONS}
        expected = max(ACTIONS, key=lambda action: qa[f"1:0:{action}"] + qb[f"1:0:{action}"])
        for left, right in ((qa, qb), (qb, qa)):
            policy = learner(Decimal("0.2"), Decimal("0.8"))
            policy.qa, policy.qb = left, right
            action = policy.decide(event, account)[0]
            if expected == 0:
                assert action.kind == "none"
            else:
                assert action.kind == "target_position"
                assert action.quantity == Decimal(expected)


def test_double_q_lifecycle_and_artifact_binding(tmp_path: Path) -> None:
    inspect_package("double-q-check", PACKAGE)
    training = TrainingRequest.model_validate_json((FIXTURE / "sarsa_training.json").read_bytes())
    output = tmp_path / "double-q-run"
    assert (
        main(
            [
                "run",
                "--package",
                str(PACKAGE),
                "--dataset",
                str(FIXTURE / "dataset.json"),
                "--events",
                str(FIXTURE / "events.json"),
                "--training",
                str(FIXTURE / "sarsa_training.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["artifact"]["training_sha256"] == fingerprint(training)
    assert report["artifact"]["strategy_id"] == "reinforcement.double_q_market_making"
    assert len(report["decisions"]) == 30
    assert len(report["fills"]) > 3
    assert {fill["side"] for fill in report["fills"]} == {"buy", "sell"}
