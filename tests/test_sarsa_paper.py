from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

from paperquant.cli import main
from paperquant.compiler import fingerprint
from paperquant.models import TrainingRequest, TrainingSample
from paperquant.package import inspect_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/reinforcement.sarsa_inventory"
FIXTURE = ROOT / "examples/l1_queues"


def learner(alpha: Decimal, gamma: Decimal):
    spec = importlib.util.spec_from_file_location("sarsa_policy", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InventorySarsa(alpha=alpha, gamma=gamma)


def features(pressure: int, inventory: int) -> dict[str, Decimal]:
    bid, ask = (75, 25) if pressure > 0 else (25, 75)
    return {
        "bid_size": Decimal(bid),
        "ask_size": Decimal(ask),
        "inventory": Decimal(inventory),
    }


@pytest.mark.parametrize(
    ("alpha", "gamma", "reward"),
    [
        (Decimal("0"), Decimal("0.8"), Decimal("3")),
        (Decimal("0.2"), Decimal("0"), Decimal("-2")),
        (Decimal("0.4"), Decimal("0.8"), Decimal("3")),
        (Decimal("0.8"), Decimal("1"), Decimal("-1")),
    ],
)
def test_sarsa_uses_actual_next_action_not_greedy_maximum(
    alpha: Decimal, gamma: Decimal, reward: Decimal
) -> None:
    next_state = features(1, 1)
    request = TrainingRequest(
        dataset_id="oracle",
        seed=5,
        samples=(
            TrainingSample(features=next_state, action=0, reward=Decimal("10"), terminal=True),
            TrainingSample(features=next_state, action=1, reward=Decimal("2"), terminal=True),
            TrainingSample(
                features=features(-1, 0),
                action=-1,
                reward=reward,
                next_features=next_state,
                next_action=1,
            ),
            TrainingSample(features=next_state, action=1, reward=Decimal("0"), terminal=True),
        ),
    )
    table = json.loads(learner(alpha, gamma).train(request))["q"]
    expected = alpha * (reward + gamma * alpha * Decimal("2"))
    assert Decimal(table["-1:0:-1"]) == expected
    if alpha > 0 and gamma > 0:
        q_learning_value = alpha * (reward + gamma * alpha * Decimal("10"))
        assert Decimal(table["-1:0:-1"]) != q_learning_value


def test_terminal_transition_cannot_bootstrap() -> None:
    request = TrainingRequest(
        dataset_id="terminal-oracle",
        seed=1,
        samples=(
            TrainingSample(
                features=features(1, 0),
                action=1,
                reward=Decimal("-3"),
                next_features=features(-1, 0),
                terminal=True,
            ),
        ),
    )
    table = json.loads(learner(Decimal("0.25"), Decimal("0.9")).train(request))["q"]
    assert Decimal(table["1:0:1"]) == Decimal("-0.75")


def test_wrong_next_behavior_action_is_rejected() -> None:
    request = TrainingRequest(
        dataset_id="invalid-oracle",
        seed=1,
        samples=(
            TrainingSample(
                features=features(1, 0),
                action=1,
                reward=Decimal("1"),
                next_features=features(1, 1),
                next_action=-1,
            ),
            TrainingSample(features=features(1, 1), action=1, reward=Decimal("1"), terminal=True),
        ),
    )
    with pytest.raises(ValueError, match="next action"):
        learner(Decimal("0.4"), Decimal("0.8")).train(request)


def test_reinforcement_package_lifecycle_and_artifact_provenance(tmp_path: Path) -> None:
    inspect_package("sarsa-check", PACKAGE)
    training = TrainingRequest.model_validate_json((FIXTURE / "sarsa_training.json").read_bytes())
    output = tmp_path / "sarsa-run"
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
    assert report["artifact"]["strategy_id"] == "reinforcement.sarsa_inventory"
    assert len(report["decisions"]) == 30
    assert report["fills"]
    assert {fill["side"] for fill in report["fills"]} == {"buy", "sell"}
