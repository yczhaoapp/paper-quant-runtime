"""Generate the committed sample's data declaration and source digest."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from examples.basic_rule.strategy import MovingAverageRule
from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, Granularity, MarketEvent
from paperquant.package import write_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    package = ROOT / "examples/basic_rule"
    strategy = MovingAverageRule()
    write_manifest(package, strategy.declaration, "MovingAverageRule")

    start = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    prices = (100, 101, 102, 104, 103, 101, 100, 102, 105, 106)
    events = tuple(
        MarketEvent(
            event_id=f"demo-{index:02d}",
            symbol="DEMO",
            event_time=start + timedelta(minutes=index),
            available_time=start + timedelta(minutes=index),
            bar_open_time=start + timedelta(minutes=index, seconds=-30),
            values={
                "open": Decimal(price),
                "high": Decimal(price + 1),
                "low": Decimal(price - 1),
                "close": Decimal(price),
                "volume": Decimal(100 + index),
            },
        )
        for index, price in enumerate(prices)
    )
    data_dir = ROOT / "examples/data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "events.json").write_text(
        json.dumps(canonical_document(events), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    dataset = DatasetDeclaration(
        dataset_id="example.minute-bars",
        granularity=Granularity.MINUTE,
        fields=frozenset(events[0].values),
        symbols=frozenset({"DEMO"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    (data_dir / "dataset.json").write_text(
        json.dumps(canonical_document(dataset), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
