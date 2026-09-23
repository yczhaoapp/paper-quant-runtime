from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from paperquant.backtrader_engine import BacktraderEngine
from paperquant.cli import main
from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.evidence import verify_bundle_file
from paperquant.models import (
    AccountSnapshot,
    ContractFault,
    DataNeed,
    DatasetDeclaration,
    ErrorCode,
    MarketEvent,
    NoOp,
    RunPolicy,
    StrategyDeclaration,
    SubmitOrder,
    TargetPosition,
    TrainingRequest,
)
from paperquant.package import run_package
from paperquant.runtime import run

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.filterwarnings(
    "ignore:datetime.datetime.utcnow\\(\\) is deprecated:DeprecationWarning")


class PositionCycle:
    declaration = StrategyDeclaration(
        strategy_id="outside.native.position-cycle", kind="rule",
        data=DataNeed(granularity="minute", fields=frozenset({
            "open", "high", "low", "close", "volume"}), symbols=frozenset({"AAA"})),
        actions=frozenset({"target_position", "none"}), training_required=False,
        source="urn:test:native-position-cycle", max_abs_position=Decimal("1"),
        max_order_quantity=Decimal("2"),
    )

    def decide(self, event: MarketEvent, account: AccountSnapshot):
        del account
        target = {"bar-0": Decimal(1), "bar-2": Decimal(-1),
                  "bar-4": Decimal(0)}.get(event.event_id)
        if target is None:
            return (NoOp(reason="hold"),)
        return (TargetPosition(symbol="AAA", quantity=target, reason="position cycle"),)


def _cycle_events() -> tuple[MarketEvent, ...]:
    start = datetime(2026, 1, 2, 10, tzinfo=UTC)
    closes = (100, 100, 100, 98, 97, 98)
    opens = (100, 101, 100, 99, 97, 98)
    return tuple(MarketEvent(
        event_id=f"bar-{index}", symbol="AAA",
        event_time=start + timedelta(minutes=index),
        available_time=start + timedelta(minutes=index),
        bar_open_time=start + timedelta(minutes=index, seconds=-30),
        values={"open": Decimal(opening), "high": Decimal(max(opening, close)),
                "low": Decimal(min(opening, close)), "close": Decimal(close),
                "volume": Decimal(100)},
    ) for index, (opening, close) in enumerate(zip(opens, closes, strict=True)))


@pytest.mark.parametrize("fee_rate", [Decimal("0"), Decimal("0.01")])
def test_native_broker_fills_and_profit_fields_match_independent_cycle(
    fee_rate: Decimal, tmp_path: Path,
) -> None:
    events = _cycle_events()
    dataset = DatasetDeclaration(dataset_id="native-cycle", granularity="minute",
        fields=frozenset(events[0].values), symbols=frozenset({"AAA"}),
        event_count=len(events), content_sha256=fingerprint(events))
    native = BacktraderEngine(initial_cash=Decimal(1000), fee_rate=fee_rate)
    actual = run(run_id="native-cycle", strategy=PositionCycle(), dataset=dataset,
        engine_capability=native.capability(dataset.fields), policy=RunPolicy(),
        events=events, engine=native, output=tmp_path / "native")
    reference = run(run_id="reference-cycle", strategy=PositionCycle(), dataset=dataset,
        engine_capability=reference_capability(dataset.fields), policy=RunPolicy(),
        events=events, engine=ReferenceEngine(initial_cash=Decimal(1000),
                                              fee_rate=fee_rate),
        output=tmp_path / "reference")
    assert actual.decisions == reference.decisions
    assert [(fill.side, fill.quantity, fill.price, fill.timestamp) for fill in actual.fills] == [
        (fill.side, fill.quantity, fill.price, fill.timestamp) for fill in reference.fills]
    assert actual.accounts[3].positions[0].realized_pnl == Decimal(-2)
    assert actual.accounts[3].positions[0].unrealized_pnl == Decimal(1)
    assert actual.accounts[4].positions[0].unrealized_pnl == Decimal(2)
    assert actual.accounts[-1].positions[0].realized_pnl == Decimal(-1)
    assert actual.accounts[-1].positions[0].quantity == 0
    assert abs(actual.final_equity - reference.final_equity) < Decimal("0.000001")
    assert all(fill.fee == fill.quantity * fill.price * fee_rate for fill in actual.fills)
    verify_bundle_file(tmp_path / "native/bundle.json")


@pytest.mark.parametrize(("strategy_id", "fixture_name", "needs_training"), [
    ("rule.moving_average_crossover", "public_aapl", False),
    ("supervised.logistic_direction", "public_direction", True),
    ("reinforcement.actor_critic_allocation", "public_actor_critic", True),
])
def test_three_strategy_families_run_through_native_broker(
    strategy_id: str, fixture_name: str, needs_training: bool, tmp_path: Path,
) -> None:
    fixture = ROOT / "examples" / fixture_name
    dataset = DatasetDeclaration.model_validate_json((fixture / "dataset.json").read_bytes())
    events = tuple(MarketEvent.model_validate(item) for item in json.loads(
        (fixture / "events.json").read_text()))
    policy = RunPolicy.model_validate_json((fixture / "policy.json").read_bytes())
    training = (TrainingRequest.model_validate_json((fixture / "training.json").read_bytes())
                if needs_training else None)
    engine = BacktraderEngine()
    report = run_package(run_id=f"native-{strategy_id}",
        package_dir=ROOT / "strategies" / strategy_id,
        dataset=dataset, engine_capability=engine.capability(dataset.fields),
        policy=policy, events=events, engine=engine, output=tmp_path, training=training)
    assert report.plan.engine_id == "paperquant.backtrader"
    assert len(report.decisions) == len(events)
    assert report.fills
    assert (report.artifact is not None) == needs_training
    assert all(fill.timestamp >= next(order.timestamp for order in report.orders
           if order.order_id == fill.order_id and order.status == "accepted")
           for fill in report.fills)
    verify_bundle_file(tmp_path / "bundle.json")


def test_native_engine_rejects_multi_symbol_mode(tmp_path: Path) -> None:
    baseline = _cycle_events()
    last = baseline[-1]
    extra = last.model_copy(update={
        "event_id": "another-symbol", "symbol": "BBB",
        "event_time": last.event_time + timedelta(minutes=1),
        "available_time": last.available_time + timedelta(minutes=1),
        "bar_open_time": last.bar_open_time + timedelta(minutes=1),
    })
    events = (*baseline, extra)
    strategy = PositionCycle()
    source = DatasetDeclaration(dataset_id="native-multi", granularity="minute",
        fields=frozenset(events[0].values), symbols=frozenset({"AAA", "BBB"}),
        event_count=len(events), content_sha256=fingerprint(events))
    declaration = strategy.declaration.model_copy(update={
        "data": strategy.declaration.data.model_copy(update={
            "symbols": frozenset({"AAA", "BBB"})})})
    strategy.declaration = declaration
    native = BacktraderEngine()
    with pytest.raises(ContractFault) as caught:
        run(run_id="native-multi", strategy=strategy, dataset=source,
            engine_capability=native.capability(source.fields), policy=RunPolicy(),
            events=events, engine=native, output=tmp_path)
    assert caught.value.failure.code == ErrorCode.ENGINE_UNSUPPORTED
    assert caught.value.failure.stage == "compile"


def test_native_engine_rejects_unsupported_financing_before_inference(
    tmp_path: Path,
) -> None:
    events = _cycle_events()
    source = DatasetDeclaration(dataset_id="native-financing", granularity="minute",
        fields=frozenset(events[0].values), symbols=frozenset({"AAA"}),
        event_count=len(events), content_sha256=fingerprint(events))
    native = BacktraderEngine()
    with pytest.raises(ContractFault) as caught:
        run(run_id="native-financing", strategy=PositionCycle(), dataset=source,
            engine_capability=native.capability(source.fields),
            policy=RunPolicy(financing_mode="unbounded_margin"),
            events=events, engine=native, output=tmp_path)
    assert caught.value.failure.code == ErrorCode.ENGINE_UNSUPPORTED
    assert caught.value.failure.stage == "compile"


def test_native_engine_rejects_limit_order_without_market_fallback(tmp_path: Path) -> None:
    class LimitStrategy(PositionCycle):
        declaration = PositionCycle.declaration.model_copy(update={
            "actions": frozenset({"submit_order"})})

        def decide(self, event: MarketEvent, account: AccountSnapshot):
            del account
            return (SubmitOrder(client_order_id=event.event_id, symbol=event.symbol,
                side="buy", quantity=Decimal(1), limit_price=Decimal(1),
                reason="not marketable"),)

    events = _cycle_events()
    source = DatasetDeclaration(dataset_id="native-limit", granularity="minute",
        fields=frozenset(events[0].values), symbols=frozenset({"AAA"}),
        event_count=len(events), content_sha256=fingerprint(events))
    native = BacktraderEngine()
    with pytest.raises(ContractFault) as caught:
        run(run_id="native-limit", strategy=LimitStrategy(), dataset=source,
            engine_capability=native.capability(source.fields), policy=RunPolicy(),
            events=events, engine=native, output=tmp_path)
    assert caught.value.failure.code == ErrorCode.ENGINE_UNSUPPORTED
    assert not (tmp_path / "report.json").exists()


def test_native_engine_uses_the_same_cli_and_replay_boundary(tmp_path: Path) -> None:
    fixture = ROOT / "examples/public_aapl"
    output = tmp_path / "native-cli"
    assert main([
        "run", "--engine", "backtrader",
        "--package", str(ROOT / "strategies/rule.moving_average_crossover"),
        "--dataset", str(fixture / "dataset.json"),
        "--events", str(fixture / "events.json"),
        "--policy", str(fixture / "policy.json"),
        "--output", str(output),
    ]) == 0
    assert main(["replay-bundle", "--bundle", str(output / "bundle.json"),
                 "--package", str(ROOT / "strategies/rule.moving_average_crossover")]) == 0
