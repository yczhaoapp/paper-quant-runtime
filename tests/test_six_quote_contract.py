"""Counterexamples and independent ledgers for contract1.1 passive orders."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from paperquant.compiler import compile_run, fingerprint
from paperquant.engine import ReferenceEngine
from paperquant.models import (
    CancelOrder,
    ContractFault,
    DataNeed,
    DatasetDeclaration,
    EngineCapability,
    MarketEvent,
    NoOp,
    RunPolicy,
    StrategyDeclaration,
    SubmitOrder,
)
from paperquant.passive_engine import PassiveQuoteEngine
from paperquant.runtime import run


def events(prices, *, flow=0, draw="0.9"):
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return tuple(
        MarketEvent(
            event_id=f"e{i}",
            symbol="MM",
            event_time=now + timedelta(seconds=i),
            available_time=now + timedelta(seconds=i),
            values={
                "bid_price": Decimal(str(price)) - Decimal("0.005"),
                "ask_price": Decimal(str(price)) + Decimal("0.005"),
                "bid_size": Decimal(10),
                "ask_size": Decimal(10),
                "buy_flow": Decimal(flow),
                "sell_flow": Decimal(flow),
                "uniform_buy": Decimal(draw),
                "uniform_sell": Decimal(draw),
            },
        )
        for i, price in enumerate(prices)
    )


def declaration():
    return StrategyDeclaration(
        contract_version="1.1",
        strategy_id="quote-probe",
        kind="rule",
        data=DataNeed(
            granularity="tick",
            market_kind="quote_l1",
            fields=frozenset({"bid_price", "ask_price", "bid_size", "ask_size"}),
            symbols=frozenset({"MM"}),
        ),
        actions=frozenset({"none", "submit_order", "cancel_order"}),
        training_required=False,
        source="urn:independent-ledger-probe",
        max_abs_position=Decimal(1),
        max_order_quantity=Decimal(1),
    )


def execute(strategy, stream, output, *, fee=Decimal(0), engine=None):
    engine = engine or PassiveQuoteEngine(initial_cash=Decimal(1000), fee_rate=fee)
    dataset = DatasetDeclaration(
        contract_version="1.1",
        dataset_id="quote-probe",
        granularity="tick",
        market_kind="quote_l1",
        fields=frozenset(stream[0].values),
        symbols=frozenset({"MM"}),
        event_count=len(stream),
        content_sha256=fingerprint(stream),
    )
    return run(
        run_id="quote-test",
        strategy=strategy,
        dataset=dataset,
        engine_capability=engine.capability(dataset.fields),
        policy=RunPolicy(),
        events=stream,
        engine=engine,
        output=output,
    )


@pytest.mark.parametrize("base", [20, 50, 100])
@pytest.mark.parametrize("fee", [Decimal(0), Decimal("0.0001")])
def test_adverse_fills_and_complete_account_ledger_match_independent_arithmetic(
    base, fee, tmp_path
):
    class Strategy:
        declaration = declaration()

        def decide(self, event, account):
            if event.event_id == "e0":
                return (
                    SubmitOrder(
                        client_order_id="buy",
                        symbol="MM",
                        side="buy",
                        quantity=Decimal(1),
                        limit_price=event.values["bid_price"],
                        reason="probe",
                    ),
                )
            if event.event_id == "e2":
                return (
                    SubmitOrder(
                        client_order_id="sell",
                        symbol="MM",
                        side="sell",
                        quantity=Decimal(1),
                        limit_price=event.values["ask_price"],
                        reason="probe",
                    ),
                )
            return (NoOp(reason="wait"),)

    report = execute(
        Strategy(), events([base, base - 0.01, base + 0.1, base + 0.12]), tmp_path, fee=fee
    )
    buy = Decimal(base) - Decimal("0.005")
    sell = Decimal(base) + Decimal("0.105")
    assert [fill.price for fill in report.fills] == [buy, sell]
    assert report.accounts[1].positions[0].unrealized_pnl == Decimal("-0.005")
    assert report.accounts[-1].positions[0].realized_pnl == Decimal("0.11")
    assert report.accounts[-1].positions[0].quantity == 0
    assert report.final_equity == Decimal(1000) + sell - buy - fee * (sell + buy)
    assert all(o.reason == "adverse_quote_cross" for o in report.orders if o.status == "filled")


@pytest.mark.parametrize(
    "draw,filled", [("0.01", True), ("0.199", True), ("0.2", False), ("0.9", False)]
)
def test_nonadverse_flow_respects_probability_boundary_and_no_same_event_fill(
    draw, filled, tmp_path
):
    class Strategy:
        declaration = declaration()

        def decide(self, event, account):
            return (
                (
                    SubmitOrder(
                        client_order_id="buy",
                        symbol="MM",
                        side="buy",
                        quantity=Decimal(1),
                        limit_price=event.values["bid_price"],
                        reason="probe",
                    ),
                )
                if event.event_id == "e0"
                else (NoOp(reason="wait"),)
            )

    stream = events([50, 50], flow=1, draw=draw)
    report = execute(Strategy(), stream, tmp_path)
    assert len(report.fills) == int(filled)
    if filled:
        assert report.fills[0].timestamp == stream[1].available_time
        assert report.orders[-1].reason == "nonadverse_order_flow"
    else:
        assert report.orders[-1].status == "expired"


def test_cancelled_quote_cannot_fill_on_later_cross(tmp_path):
    class Strategy:
        declaration = declaration()

        def decide(self, event, account):
            if event.event_id == "e0":
                return (
                    SubmitOrder(
                        client_order_id="buy",
                        symbol="MM",
                        side="buy",
                        quantity=Decimal(1),
                        limit_price=event.values["bid_price"],
                        reason="probe",
                    ),
                )
            if event.event_id == "e1":
                return (
                    CancelOrder(
                        client_order_id="buy", symbol="MM", reason="withdraw before next cross"
                    ),
                )
            return (NoOp(reason="wait"),)

    report = execute(Strategy(), events([50, 50, 49]), tmp_path)
    assert not report.fills
    assert [o.status for o in report.orders] == ["accepted", "cancelled"]


def test_unknown_cancel_and_duplicate_quote_fail_structurally(tmp_path):
    for duplicate in (False, True):

        class Strategy:
            declaration = declaration()

            def decide(self, event, account, duplicate=duplicate):
                if not duplicate:
                    return (CancelOrder(client_order_id="absent", symbol="MM", reason="probe"),)
                return tuple(
                    SubmitOrder(
                        client_order_id=identifier,
                        symbol="MM",
                        side="buy",
                        quantity=Decimal(1),
                        limit_price=event.values["bid_price"],
                        reason="probe",
                    )
                    for identifier in ("first", "second")
                )

        with pytest.raises(ContractFault) as error:
            execute(Strategy(), events([50, 50]), tmp_path / str(duplicate))
        assert error.value.failure.code == "ORDER_REJECTED"
        assert error.value.failure.fallback_used is False
        assert not (tmp_path / str(duplicate) / "report.json").exists()


def test_legacy_engine_refuses_cancel_semantics_before_running(tmp_path):
    class Strategy:
        declaration = declaration()

        def decide(self, event, account):
            raise AssertionError("unsupported engine must refuse before inference")

    with pytest.raises(ContractFault) as error:
        execute(Strategy(), events([50, 50]), tmp_path, engine=ReferenceEngine())
    assert error.value.failure.code == "ENGINE_UNSUPPORTED"
    with pytest.raises(ValidationError):
        StrategyDeclaration.model_validate(
            {**declaration().model_dump(), "contract_version": "1.0"}
        )


def test_direct_passive_engine_is_bound_to_plan_strategy_and_event_bytes():
    class Strategy:
        declaration = declaration()

        def decide(self, event, account):
            return (NoOp(reason="probe"),)

    stream = events([50, 50])
    engine = PassiveQuoteEngine()
    dataset = DatasetDeclaration(
        contract_version="1.1",
        dataset_id="probe",
        granularity="tick",
        market_kind="quote_l1",
        fields=frozenset(stream[0].values),
        symbols=frozenset({"MM"}),
        event_count=2,
        content_sha256=fingerprint(stream),
    )
    plan, effective = compile_run(
        run_id="probe",
        strategy=Strategy.declaration,
        dataset=dataset,
        engine=engine.capability(dataset.fields),
        engine_profile=engine.profile(),
        policy=RunPolicy(),
        events=stream,
    )
    for mutation in ("strategy", "events", "capability"):
        strategy = Strategy()
        current = effective
        bound = plan
        if mutation == "strategy":
            strategy.declaration = strategy.declaration.model_copy(
                update={"strategy_id": "different"}
            )
        if mutation == "events":
            current = (effective[0], effective[1].model_copy(update={"event_id": "different"}))
        if mutation == "capability":
            bound = plan.model_copy(
                update={
                    "engine_capability": EngineCapability(
                        engine_id="forged",
                        granularities=frozenset({"tick"}),
                        data_fields=dataset.fields,
                        actions=frozenset({"none"}),
                    )
                }
            )
        with pytest.raises(ContractFault):
            engine.run(plan=bound, strategy=strategy, events=current)
