"""Create a complete finite-horizon synthetic execution transition lattice."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import (
    DatasetDeclaration,
    Granularity,
    MarketEvent,
    TrainingRequest,
    TrainingSample,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "fixture.synthetic-execution-lattice"


def state(time: int, remaining: int, pressure: int) -> dict[str, Decimal]:
    return {
        "time_remaining": Decimal(time),
        "remaining": Decimal(remaining),
        "pressure": Decimal(pressure),
    }


def actions(time: int, remaining: int) -> tuple[int, ...]:
    if remaining == 0:
        return (0,)
    if time == 1:
        return (remaining,)
    return tuple(range(min(2, remaining) + 1))


def build_training() -> TrainingRequest:
    transitions = []
    for time in range(1, 5):
        for remaining in range(5):
            for pressure in (-1, 0, 1):
                for action in actions(time, remaining):
                    future_pressures = (0,) if time == 1 else (-1, 0, 1)
                    for future_pressure in future_pressures:
                        reward = (
                            Decimal(action) * Decimal(pressure) / 5
                            - Decimal(action * action) / 20
                        )
                        transitions.append(TrainingSample(
                            features=state(time, remaining, pressure),
                            action=action,
                            reward=reward,
                            terminal=time == 1,
                            next_features=(None if time == 1 else state(
                                time - 1, remaining - action, future_pressure
                            )),
                        ))
    return TrainingRequest(dataset_id=DATASET_ID, seed=0, samples=tuple(transitions))


def build_events() -> tuple[MarketEvent, ...]:
    beginning = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    events = []
    pressures = (1, -1, 0, 1, 0, -1, 1, 1, -1, 0, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0)
    for index, pressure in enumerate(pressures):
        bid, ask = ((75, 25) if pressure > 0 else
                    (25, 75) if pressure < 0 else (50, 50))
        price = Decimal(100) + Decimal(index % 5) / 10
        events.append(MarketEvent(
            event_id=f"execution-{index:02d}",
            symbol="DEMO",
            event_time=beginning + timedelta(seconds=index),
            available_time=beginning + timedelta(seconds=index),
            values={
                "price": price,
                "bid_price": price - Decimal("0.01"),
                "ask_price": price + Decimal("0.01"),
                "bid_size": Decimal(bid),
                "ask_size": Decimal(ask),
                "time_remaining": Decimal(4 - index % 5),
            },
        ))
    return tuple(events)


def main() -> None:
    events = build_events()
    training = build_training()
    dataset = DatasetDeclaration(
        dataset_id=DATASET_ID,
        granularity=Granularity.TICK,
        fields=frozenset(events[0].values),
        symbols=frozenset({"DEMO"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    destination = ROOT / "examples/execution_lattice"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("dataset.json", dataset),
        ("events.json", events),
        ("training.json", training),
    ):
        (destination / filename).write_text(
            json.dumps(canonical_document(value), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
