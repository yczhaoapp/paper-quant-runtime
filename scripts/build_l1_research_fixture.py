"""Build separate synthetic learning samples and tick events for L1 integration."""

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
DATASET_ID = "fixture.synthetic-l1-queues"


def main() -> None:
    training_pairs = [(20 + index * 3, 80 - index * 2) for index in range(20)]
    training = TrainingRequest(
        dataset_id=DATASET_ID,
        seed=7,
        samples=tuple(
            TrainingSample(
                features={"bid_size": Decimal(bid), "ask_size": Decimal(ask)},
                target=Decimal("1") if bid > ask else Decimal("0"),
            )
            for bid, ask in training_pairs
        ),
    )
    start = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    queue_pairs = [(75, 25), (70, 30), (55, 45), (30, 70), (25, 75), (40, 60)] * 5
    events = []
    for index, (bid_size, ask_size) in enumerate(queue_pairs):
        midpoint = Decimal("100") + Decimal(index % 5) / Decimal("10")
        events.append(
            MarketEvent(
                event_id=f"l1-{index:03d}",
                symbol="DEMO",
                event_time=start + timedelta(seconds=index),
                available_time=start + timedelta(seconds=index),
                values={
                    "price": midpoint,
                    "bid_price": midpoint - Decimal("0.01"),
                    "ask_price": midpoint + Decimal("0.01"),
                    "bid_size": Decimal(bid_size),
                    "ask_size": Decimal(ask_size),
                },
            )
        )
    events = tuple(events)
    destination = ROOT / "examples/l1_queues"
    destination.mkdir(parents=True, exist_ok=True)
    dataset = DatasetDeclaration(
        dataset_id=DATASET_ID,
        granularity=Granularity.TICK,
        fields=frozenset(events[0].values),
        symbols=frozenset({"DEMO"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    for name, value in (
        ("training.json", training),
        ("events.json", events),
        ("dataset.json", dataset),
    ):
        (destination / name).write_text(
            json.dumps(canonical_document(value), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
