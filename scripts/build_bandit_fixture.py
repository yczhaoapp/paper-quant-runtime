"""Build separate synthetic reward observations and L1 online evaluation ticks."""

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
DATASET_ID = "fixture.synthetic-bandit-l1"


def build_training() -> TrainingRequest:
    samples = []
    for index in range(18):
        context = Decimal(((index * 7) % 11) - 5) / 10
        for arm in (-1, 0, 1):
            reward = (
                Decimal(arm) * context * Decimal("0.04")
                + (Decimal("0.003") if (index + arm) % 2 else Decimal("-0.003"))
            )
            samples.append(TrainingSample(
                features={"context": context},
                action=arm,
                reward=reward,
            ))
    return TrainingRequest(dataset_id=DATASET_ID, seed=29, samples=tuple(samples))


def build_events() -> tuple[MarketEvent, ...]:
    beginning = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    midpoint = Decimal(100)
    events = []
    for index in range(40):
        direction = Decimal(1) if index % 4 in (0, 1) else Decimal(-1)
        midpoint += direction * (Decimal("0.06") + Decimal(index % 3) / 100)
        bid_size = Decimal(20 + (index * 13) % 70)
        ask_size = Decimal(100) - bid_size
        events.append(MarketEvent(
            event_id=f"bandit-{index:02d}",
            symbol="DEMO",
            event_time=beginning + timedelta(seconds=index),
            available_time=beginning + timedelta(seconds=index),
            values={
                "price": midpoint,
                "bid_price": midpoint - Decimal("0.01"),
                "ask_price": midpoint + Decimal("0.01"),
                "bid_size": bid_size,
                "ask_size": ask_size,
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
    destination = ROOT / "examples/bandit_l1"
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
