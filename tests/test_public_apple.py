from __future__ import annotations

from datetime import time, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from paperquant.compiler import compile_run, fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import (
    ContractFault,
    DataNeed,
    DatasetDeclaration,
    ErrorCode,
    Granularity,
    MarketEvent,
    RunPolicy,
    StrategyDeclaration,
    StrategyKind,
)
from paperquant.package import run_package
from paperquant.public_data import load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/public/finance-charts-apple.csv"


def test_pinned_public_csv_has_real_bars_and_explicit_open_clock() -> None:
    declaration, events = load_plotly_apple(CSV)
    assert declaration.event_count == len(events) == 506
    assert declaration.content_sha256 == fingerprint(events)
    assert events[0].event_time.date().isoformat() == "2015-02-17"
    assert events[-1].event_time.date().isoformat() == "2017-02-16"
    assert events[0].bar_open_time is not None
    assert events[0].bar_open_time < events[0].available_time
    assert events[50].bar_open_time is not None
    assert events[50].bar_open_time.utcoffset() != events[0].bar_open_time.utcoffset()
    assert all(event.values["high"] >= event.values["low"] for event in events)
    assert all("mavg" not in event.values and "adjusted" not in event.values for event in events)


def test_public_daily_run_matches_independent_formula_and_fill_clock(tmp_path: Path) -> None:
    declaration, events = load_plotly_apple(CSV)
    report = run_package(
        run_id="public-aapl-formula",
        package_dir=ROOT / "strategies/rule.moving_average_crossover",
        dataset=declaration,
        engine_capability=reference_capability(declaration.fields),
        policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )
    assert report.status == "succeeded"
    assert len(report.decisions) == len(events)
    assert len(report.fills) > 10
    assert len(report.plan.conversions) == 1
    assert report.plan.conversions[0].transformation == "symbol_map"

    closes = [event.values["close"] for event in events]
    for index, decision in enumerate(report.decisions):
        if index < 19:
            assert decision.actions[0].kind == "none"
            continue
        fast = sum(closes[index - 1 : index + 1], Decimal(0)) / 2
        slow = sum(closes[index - 19 : index + 1], Decimal(0)) / 20
        wanted = Decimal(1) if fast > slow else Decimal(-1) if fast < slow else Decimal(0)
        if decision.actions[0].kind == "target_position":
            assert decision.actions[0].quantity == wanted

    by_id = {order.order_id: order for order in report.orders if order.status == "accepted"}
    opens = {event.bar_open_time: event.values["open"] for event in events}
    for fill in report.fills:
        assert fill.order_id in by_id
        assert fill.timestamp > by_id[fill.order_id].timestamp
        assert fill.timestamp in opens
        assert fill.price == opens[fill.timestamp]


def test_public_csv_bytes_are_pinned(tmp_path: Path) -> None:
    changed = tmp_path / "changed.csv"
    changed.write_bytes(CSV.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash differs"):
        load_plotly_apple(changed)


@pytest.mark.parametrize("problem", ["missing_open", "overlapping_open"])
def test_bar_clock_cannot_hide_lookahead(problem: str) -> None:
    _, public_events = load_plotly_apple(CSV)
    first, second = public_events[:2]
    if problem == "missing_open":
        second = second.model_copy(update={"bar_open_time": None})
    else:
        second = second.model_copy(
            update={"bar_open_time": first.available_time - timedelta(seconds=1)}
        )
    events: tuple[MarketEvent, ...] = (first, second)
    dataset = DatasetDeclaration(
        dataset_id="clock-defect",
        granularity=Granularity.DAY,
        session_timezone="America/New_York",
        session_open=time(9, 30),
        session_close=time(16),
        fields=frozenset(events[0].values),
        symbols=frozenset({"AAPL"}),
        event_count=2,
        content_sha256=fingerprint(events),
    )
    with pytest.raises(ContractFault) as caught:
        compile_run(
            run_id="clock-defect",
            strategy=_small_daily_declaration(),
            dataset=dataset,
            engine=reference_capability(dataset.fields),
            policy=RunPolicy(),
            events=events,
        )
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


def _small_daily_declaration() -> StrategyDeclaration:
    return StrategyDeclaration(
        strategy_id="test.daily",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"close"}),
            symbols=frozenset({"AAPL"}),
        ),
        actions=frozenset({"none"}),
        training_required=False,
        source="test",
    )
