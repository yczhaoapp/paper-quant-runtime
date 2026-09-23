"""Generate an explicit synthetic two-level book for quote-contract testing."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, Granularity, MarketEvent

ROOT = Path(__file__).resolve().parents[1]
PATTERN = (Decimal("100"), Decimal("101.4"), Decimal("100"), Decimal("98.6"))


def main() -> None:
    start = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    events = []
    for index in range(32):
        midpoint = PATTERN[index % len(PATTERN)]
        bid = midpoint - Decimal("0.01")
        ask = midpoint + Decimal("0.01")
        event_time = start + timedelta(seconds=index)
        events.append(
            MarketEvent(
                event_id=f"l2-{index:03d}",
                symbol="DEMO",
                event_time=event_time,
                available_time=event_time,
                values={
                    "price": midpoint,
                    "bid_price": bid,
                    "ask_price": ask,
                    "bid_size": Decimal(12 + index % 7),
                    "ask_size": Decimal(11 + index % 5),
                    "bid_price_2": bid - Decimal("0.01"),
                    "ask_price_2": ask + Decimal("0.01"),
                    "bid_size_2": Decimal(21 + index % 3),
                    "ask_size_2": Decimal(20 + index % 4),
                },
            )
        )
    series = tuple(events)
    dataset = DatasetDeclaration(
        dataset_id="fixture.synthetic-l2-book",
        granularity=Granularity.TICK,
        fields=frozenset(series[0].values),
        symbols=frozenset({"DEMO"}),
        event_count=len(series),
        content_sha256=fingerprint(series),
    )
    destination = ROOT / "examples/l2_book"
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (("events.json", series), ("dataset.json", dataset)):
        (destination / name).write_text(
            json.dumps(canonical_document(value), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
