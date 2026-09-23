"""Construct a synchronized three-symbol synthetic formation and trading sample."""

from __future__ import annotations

import json
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, Granularity, MarketEvent

ROOT = Path(__file__).resolve().parents[1]


def build_events() -> tuple[MarketEvent, ...]:
    events = []
    beginning = datetime(2026, 1, 1, 16, tzinfo=UTC)
    for index in range(379):
        alpha = Decimal(100) + Decimal(index) / 100
        if index < 252:
            beta = alpha + (Decimal("0.2") if index % 2 else Decimal("-0.2"))
        else:
            step = index - 252
            offset = (
                Decimal(5) if step < 8 else
                Decimal(-1) if step < 20 else
                Decimal(-5) if step < 28 else
                Decimal(1) if step < 36 else
                Decimal("0.05") if step % 2 else Decimal("-0.05")
            )
            beta = alpha + offset
        gamma = Decimal(100) + Decimal(index) * Decimal("0.08")
        for symbol, price in (("ALPHA", alpha), ("BETA", beta), ("GAMMA", gamma)):
            events.append(
                MarketEvent(
                    event_id=f"pair-{index:03d}-{symbol}",
                    symbol=symbol,
                    bar_open_time=beginning + timedelta(days=index, hours=-7),
                    event_time=beginning + timedelta(days=index),
                    available_time=beginning + timedelta(days=index),
                    values={
                        "open": price,
                        "high": price + Decimal(1),
                        "low": price - Decimal(1),
                        "close": price,
                        "volume": Decimal(1000),
                    },
                )
            )
    return tuple(events)


def main() -> None:
    events = build_events()
    dataset = DatasetDeclaration(
        dataset_id="fixture.synthetic-three-stock-pairs",
        granularity=Granularity.DAY,
        session_open=time(9),
        session_close=time(16),
        fields=frozenset(events[0].values),
        symbols=frozenset({"ALPHA", "BETA", "GAMMA"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    destination = ROOT / "examples/three_stock_pairs"
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (("events.json", events), ("dataset.json", dataset)):
        (destination / name).write_text(
            json.dumps(canonical_document(value), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
