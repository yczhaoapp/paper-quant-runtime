"""Deterministic adapters for pinned, locally bundled public market samples."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from paperquant.compiler import fingerprint
from paperquant.models import DatasetDeclaration, Granularity, MarketEvent

PLOTLY_COMMIT = "0c447c47b757ad74edecab31f0d72f849d2e67c2"
APPLE_SHA256 = "24c7604edfd5afe862ddb9f9535e2fd7351f43711bfc442bca52055e15e37bcd"
APPLE_COLUMNS = (
    "Date",
    "AAPL.Open",
    "AAPL.High",
    "AAPL.Low",
    "AAPL.Close",
    "AAPL.Volume",
    "AAPL.Adjusted",
    "dn",
    "mavg",
    "up",
    "direction",
)


def load_plotly_apple(path: Path) -> tuple[DatasetDeclaration, tuple[MarketEvent, ...]]:
    """Use raw OHLCV only; upstream adjusted/indicator columns are excluded."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != APPLE_SHA256:
        raise ValueError("Pinned public CSV hash differs")
    reader = csv.DictReader(payload.decode("utf-8-sig").splitlines())
    if tuple(reader.fieldnames or ()) != APPLE_COLUMNS:
        raise ValueError("Public CSV header differs")
    exchange_zone = ZoneInfo("America/New_York")
    events: list[MarketEvent] = []
    last_date = None
    for index, row in enumerate(reader):
        try:
            day = datetime.strptime(row["Date"], "%Y-%m-%d").date()
            values = {
                field: Decimal(row[f"AAPL.{field.capitalize()}"])
                for field in ("open", "high", "low", "close", "volume")
            }
        except (ValueError, InvalidOperation, KeyError) as exc:
            raise ValueError(f"Invalid public CSV row {index + 2}") from exc
        if last_date is not None and day <= last_date:
            raise ValueError("Public CSV dates are not strictly increasing")
        last_date = day
        if (
            not all(value.is_finite() for value in values.values())
            or min(values[name] for name in ("open", "high", "low", "close")) <= 0
            or values["volume"] < 0
            or values["high"] < max(values["open"], values["close"], values["low"])
            or values["low"] > min(values["open"], values["close"], values["high"])
        ):
            raise ValueError(f"Invalid OHLCV in public CSV row {index + 2}")
        opening = datetime.combine(day, time(9, 30), exchange_zone)
        closing = datetime.combine(day, time(16), exchange_zone)
        events.append(
            MarketEvent(
                event_id=f"plotly-aapl-{day.isoformat()}",
                symbol="AAPL",
                bar_open_time=opening,
                event_time=closing,
                available_time=closing,
                values=values,
            )
        )
    if len(events) != 506:
        raise ValueError("Pinned public CSV row count differs")
    series = tuple(events)
    declaration = DatasetDeclaration(
        dataset_id=f"public.plotly-aapl.{PLOTLY_COMMIT[:12]}",
        granularity=Granularity.DAY,
        fields=frozenset({"open", "high", "low", "close", "volume"}),
        symbols=frozenset({"AAPL"}),
        event_count=len(series),
        content_sha256=fingerprint(series),
    )
    return declaration, series
