"""Build a deterministic synthetic daily market for formula and lifecycle checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, Granularity, MarketEvent

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    prices = [100] * 20 + list(range(101, 116)) + list(range(114, 84, -1))
    start = datetime(2026, 1, 1, 16, tzinfo=UTC)
    events = tuple(
        MarketEvent(
            event_id=f"synthetic-day-{index:03d}",
            symbol="DEMO",
            event_time=start + timedelta(days=index),
            available_time=start + timedelta(days=index),
            bar_open_time=start + timedelta(days=index, hours=-6),
            values={
                "open": Decimal(price),
                "high": Decimal(price + 1),
                "low": Decimal(price - 1),
                "close": Decimal(price),
                "volume": Decimal("1000"),
            },
        )
        for index, price in enumerate(prices)
    )
    destination = ROOT / "examples/daily_vma"
    destination.mkdir(parents=True, exist_ok=True)
    dataset = DatasetDeclaration(
        dataset_id="fixture.synthetic-daily-vma",
        granularity=Granularity.DAY,
        session_open=time(10),
        session_close=time(16),
        fields=frozenset(events[0].values),
        symbols=frozenset({"DEMO"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    (destination / "events.json").write_text(
        json.dumps(canonical_document(events), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (destination / "dataset.json").write_text(
        json.dumps(canonical_document(dataset), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
