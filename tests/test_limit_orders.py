from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import (
    ContractFault,
    DataNeed,
    DatasetDeclaration,
    ErrorCode,
    Granularity,
    MarketEvent,
    NoOp,
    RunPolicy,
    StrategyDeclaration,
    StrategyKind,
    SubmitOrder,
)
from paperquant.runtime import run


class OneQuote:
    def __init__(self, *, side: str, limit: Decimal, granularity: Granularity) -> None:
        self.side = side
        self.limit = limit
        self.count = 0
        self.declaration = StrategyDeclaration(
            strategy_id="test.one-quote",
            kind=StrategyKind.RULE,
            data=DataNeed(
                granularity=granularity,
                fields=frozenset({"price"})
                if granularity == Granularity.TICK
                else frozenset({"open", "close"}),
                symbols=frozenset({"AAA"}),
                minimum_events_per_symbol=2,
            ),
            actions=frozenset({"submit_order", "none"}),
            training_required=False,
            source="test",
            max_abs_position=Decimal(2),
        )

    def decide(self, event, account):  # type: ignore[no-untyped-def]
        del account
        self.count += 1
        if self.count > 1:
            return (NoOp(reason="one order only"),)
        return (
            SubmitOrder(
                client_order_id="first",
                symbol=event.symbol,
                side=self.side,
                quantity=Decimal(1),
                limit_price=self.limit,
                reason="test quote",
            ),
        )


def _tick_events(*, quotes: bool = True) -> tuple[MarketEvent, ...]:
    start = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    values = (
        {"price": Decimal(100), "bid_price": Decimal("99.9"), "ask_price": Decimal("100.1")},
        {"price": Decimal(100), "bid_price": Decimal("99.8"), "ask_price": Decimal("100.2")},
    )
    return tuple(
        MarketEvent(
            event_id=f"tick-{index}",
            symbol="AAA",
            event_time=start + timedelta(seconds=index),
            available_time=start + timedelta(seconds=index),
            values=row if quotes else {"price": row["price"]},
        )
        for index, row in enumerate(values)
    )


def _run(strategy: OneQuote, events: tuple[MarketEvent, ...], tmp_path: Path):  # type: ignore[no-untyped-def]
    declaration = DatasetDeclaration(
        dataset_id="test-quote",
        granularity=strategy.declaration.data.granularity,
        fields=frozenset(events[0].values),
        symbols=frozenset({"AAA"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    return run(
        run_id="quote-test",
        strategy=strategy,
        dataset=declaration,
        engine_capability=reference_capability(declaration.fields),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )


@pytest.mark.parametrize(
    ("side", "limit", "filled", "expected_price"),
    [
        ("buy", Decimal("100.2"), True, Decimal("100.2")),
        ("buy", Decimal("100.1"), False, None),
        ("sell", Decimal("99.8"), True, Decimal("99.8")),
        ("sell", Decimal("99.9"), False, None),
    ],
)
def test_limit_matches_next_tick_touch_or_expires(
    side: str, limit: Decimal, filled: bool, expected_price: Decimal | None, tmp_path: Path
) -> None:
    report = _run(
        OneQuote(side=side, limit=limit, granularity=Granularity.TICK), _tick_events(), tmp_path
    )
    assert len(report.fills) == int(filled)
    assert [order.status for order in report.orders] == [
        "accepted",
        "filled" if filled else "expired",
    ]
    if filled:
        assert report.fills[0].price == expected_price
        assert report.orders[1].reason == "next_event_marketable_limit_fill"
    else:
        assert report.orders[1].reason == "next_event_limit_not_marketable"


def test_limit_never_uses_future_bar_low_to_fill_at_its_open(tmp_path: Path) -> None:
    start = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    events = tuple(
        MarketEvent(
            event_id=f"bar-{index}",
            symbol="AAA",
            bar_open_time=start + timedelta(minutes=index),
            event_time=start + timedelta(minutes=index, seconds=30),
            available_time=start + timedelta(minutes=index, seconds=30),
            values={
                "open": Decimal(101),
                "high": Decimal(102),
                "low": Decimal(98),
                "close": Decimal(100),
            },
        )
        for index in range(2)
    )
    report = _run(
        OneQuote(side="buy", limit=Decimal(99), granularity=Granularity.MINUTE), events, tmp_path
    )
    assert not report.fills
    assert report.orders[-1].status == "expired"


def test_missing_next_tick_quotes_fail_structurally(tmp_path: Path) -> None:
    with pytest.raises(ContractFault) as caught:
        _run(
            OneQuote(side="buy", limit=Decimal(101), granularity=Granularity.TICK),
            _tick_events(quotes=False),
            tmp_path,
        )
    assert caught.value.failure.code == ErrorCode.DATA_FIELD_MISSING


def test_last_event_order_is_expired_not_left_accepted(tmp_path: Path) -> None:
    strategy = OneQuote(side="buy", limit=Decimal(101), granularity=Granularity.TICK)
    original_decide = strategy.decide

    def decide_at_end(event, account):  # type: ignore[no-untyped-def]
        if event.event_id == "tick-0":
            return (NoOp(reason="wait"),)
        return original_decide(event, account)

    strategy.decide = decide_at_end  # type: ignore[method-assign]
    report = _run(strategy, _tick_events(), tmp_path)
    assert not report.fills
    assert [order.status for order in report.orders] == ["accepted", "expired"]
    assert report.orders[-1].reason == "no_next_event"
