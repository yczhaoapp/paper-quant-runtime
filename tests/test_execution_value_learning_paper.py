from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import DatasetDeclaration, MarketEvent, RunPolicy, TrainingRequest
from paperquant.package import inspect_package, run_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/reinforcement.execution_value_learning"
FIXTURE = ROOT / "examples/execution_lattice"


def _inputs() -> tuple[DatasetDeclaration, tuple[MarketEvent, ...], TrainingRequest]:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    return dataset, events, training


def _load_policy():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("execution_policy", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExecutionValueLearning()


def _state(features: dict[str, Decimal]) -> tuple[int, int, int]:
    return (int(features["time_remaining"]), int(features["remaining"]),
            int(features["pressure"]))


def _actions(time: int, inventory: int) -> tuple[int, ...]:
    return (0,) if inventory == 0 else (inventory,) if time == 1 else tuple(
        range(min(2, inventory) + 1)
    )


def _reference_values(training: TrainingRequest) -> dict[str, Decimal]:
    grouped = defaultdict(list)
    for sample in training.samples:
        grouped[(_state(sample.features), sample.action)].append(sample)
    values = {}
    for time in range(1, 5):
        for inventory in range(5):
            for pressure in (-1, 0, 1):
                state = (time, inventory, pressure)
                for action in _actions(time, inventory):
                    observations = grouped[(state, action)]
                    assert observations
                    payoffs = []
                    for sample in observations:
                        assert sample.reward is not None
                        continuation = Decimal(0)
                        if time > 1:
                            assert sample.next_features is not None
                            successor = _state(sample.next_features)
                            continuation = max(
                                values[f"{successor[0]}:{successor[1]}:{successor[2]}:{future}"]
                                for future in _actions(successor[0], successor[1])
                            )
                        else:
                            assert sample.terminal and sample.next_features is None
                        payoffs.append(sample.reward + continuation)
                    values[f"{time}:{inventory}:{pressure}:{action}"] = (
                        sum(payoffs) / Decimal(len(payoffs))
                    )
    return values


def test_complete_training_lattice_and_backward_values_match_reference() -> None:
    claim = json.loads((ROOT / "research/claims/execution_value_learning.json").read_text())
    assert len(claim["claims"]) == 3
    inspect_package("execution-paper", PACKAGE)
    dataset, events, training = _inputs()
    assert len(events) == dataset.event_count == 20
    assert fingerprint(events) == dataset.content_sha256
    assert all(events[index].values["time_remaining"] == Decimal(4 - index % 5)
               for index in range(20))
    model = json.loads(_load_policy().train(training))
    expected = _reference_values(training)
    assert set(model["q"]) == set(expected)
    for key, value in expected.items():
        assert abs(Decimal(model["q"][key]) - value) < Decimal("1e-24")
    assert len({int(key.split(":")[0]) for key in model["q"]}) == 4


@pytest.mark.parametrize("index", [0, 17, 73])
def test_reward_perturbation_changes_only_current_and_earlier_time_values(index: int) -> None:
    _, _, training = _inputs()
    target = training.samples[index]
    assert target.reward is not None
    changed = list(training.samples)
    changed[index] = target.model_copy(update={"reward": target.reward + Decimal(1)})
    modified = training.model_copy(update={"samples": tuple(changed)})
    old_table = json.loads(_load_policy().train(training))["q"]
    new_table = json.loads(_load_policy().train(modified))["q"]
    before = {key: Decimal(value) for key, value in old_table.items()}
    after = {key: Decimal(value) for key, value in new_table.items()}
    state = _state(target.features)
    key = f"{state[0]}:{state[1]}:{state[2]}:{target.action}"
    assert before[key] != after[key]
    for other in before:
        if int(other.split(":")[0]) < state[0]:
            assert before[other] == after[other]


def test_execution_training_save_reload_and_episode_fills(tmp_path: Path) -> None:
    dataset, events, training = _inputs()
    report = run_package(
        run_id="execution-value-paper",
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
    assert len(report.fills) >= 8
    assert {fill.side for fill in report.fills} == {"buy", "sell"}
    model = json.loads((tmp_path / report.artifact.relative_path).read_text())
    values = {key: Decimal(value) for key, value in model["q"].items()}
    for event, account, decision in zip(events, report.accounts, report.decisions, strict=True):
        time = int(event.values["time_remaining"])
        if time == 0:
            continue
        held = next((position.quantity for position in account.positions), Decimal(0))
        inventory = 4 - int(held)
        bid, ask = event.values["bid_size"], event.values["ask_size"]
        pressure = 1 if bid > ask else -1 if bid < ask else 0
        expected_action = max(
            _actions(time, inventory),
            key=lambda action: values[f"{time}:{inventory}:{pressure}:{action}"],
        )
        assert decision.actions[0].value == Decimal(expected_action)
        if time == 1:
            assert expected_action == inventory
    assert any(order.reason == "reset finished execution episode" for order in report.orders)


def test_missing_lattice_action_fails_closed() -> None:
    _, _, training = _inputs()
    reduced = training.model_copy(update={"samples": tuple(
        sample for sample in training.samples
        if not (sample.features["time_remaining"] == 4
                and sample.features["remaining"] == 4
                and sample.features["pressure"] == 1 and sample.action == 2)
    )})
    with pytest.raises(ValueError, match="misses a state-action pair"):
        _load_policy().train(reduced)
