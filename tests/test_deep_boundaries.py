from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from test_contract import TinyLearner, bars, dataset, declaration

from paperquant.compiler import compile_run, fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.evidence import verify_bundle_file
from paperquant.models import (
    AccountSnapshot,
    ContractFault,
    DataNeed,
    DatasetDeclaration,
    Decision,
    EngineCapability,
    EngineProfile,
    ErrorCode,
    Granularity,
    MarketEvent,
    NoOp,
    RunPolicy,
    StrategyDeclaration,
    SubmitOrder,
    TrainingRequest,
    TrainingSample,
)
from paperquant.runtime import run


def test_cross_symbol_order_waits_for_a_causally_available_open(tmp_path: Path) -> None:
    close = datetime(2026, 1, 2, 10, tzinfo=UTC)
    events = (
        MarketEvent(event_id="a", symbol="AAA", event_time=close,
                    available_time=close, bar_open_time=close-timedelta(seconds=30),
                    values={"open": Decimal(100), "close": Decimal(100)}),
        MarketEvent(event_id="b", symbol="BBB", event_time=close,
                    available_time=close, bar_open_time=close-timedelta(seconds=30),
                    values={"open": Decimal(50), "close": Decimal(50)}),
        MarketEvent(event_id="c", symbol="BBB", event_time=close+timedelta(minutes=1),
                    available_time=close+timedelta(minutes=1),
                    bar_open_time=close+timedelta(seconds=30),
                    values={"open": Decimal(51), "close": Decimal(51)}),
    )
    class CrossSymbol:
        declaration = StrategyDeclaration(
            strategy_id="unseen.cross-symbol", kind="rule", training_required=False,
            source="urn:test:causality",
            data=DataNeed(granularity="minute", fields=frozenset({"open", "close"}),
                          symbols=frozenset({"AAA", "BBB"})),
            actions=frozenset({"submit_order", "none"}),
        )
        def decide(self, event: MarketEvent, account: AccountSnapshot):
            del account
            if event.event_id == "a":
                return (SubmitOrder(client_order_id="cross", symbol="BBB", side="buy",
                                    quantity=Decimal(1), reason="cross"),)
            return (NoOp(reason="wait"),)
    source = DatasetDeclaration(dataset_id="cross", granularity="minute",
                                fields=frozenset({"open", "close"}),
                                symbols=frozenset({"AAA", "BBB"}),
                                event_count=len(events), content_sha256=fingerprint(events))
    report = run(run_id="cross", strategy=CrossSymbol(), dataset=source,
                 engine_capability=reference_capability(source.fields), policy=RunPolicy(),
                 events=events, engine=ReferenceEngine(), output=tmp_path)
    assert len(report.fills) == 1
    assert report.fills[0].timestamp == events[2].bar_open_time
    assert report.fills[0].timestamp >= close


def test_short_bars_cannot_claim_daily_granularity() -> None:
    events = bars()
    day_dataset = dataset(events).model_copy(update={
        "granularity": Granularity.DAY,
        "session_open": events[0].bar_open_time.time(),
        "session_close": events[0].event_time.time(),
    })
    day_strategy = declaration().model_copy(update={
        "data": declaration().data.model_copy(update={"granularity": Granularity.DAY})})
    with pytest.raises(ContractFault) as caught:
        compile_run(run_id="false-day", strategy=day_strategy, dataset=day_dataset,
                    engine=reference_capability(day_dataset.fields), policy=RunPolicy(),
                    events=events)
    assert caught.value.failure.code == ErrorCode.TIMEFRAME_MISMATCH


@pytest.mark.parametrize("update", [{"side": "BUY"}, {"quantity": Decimal(-1)}])
def test_unvalidated_host_action_is_rejected(update: dict, tmp_path: Path) -> None:
    class InvalidOrder:
        declaration = declaration().model_copy(update={"actions": frozenset({"submit_order"})})
        def decide(self, event: MarketEvent, account: AccountSnapshot):
            del account
            legal = SubmitOrder(client_order_id=event.event_id, symbol=event.symbol,
                                side="buy", quantity=Decimal(1), reason="test")
            return (legal.model_copy(update=update),)
    events = bars()
    with pytest.raises(ContractFault) as caught:
        run(run_id="invalid-host-action", strategy=InvalidOrder(), dataset=dataset(events),
            engine_capability=reference_capability(dataset(events).fields),
            policy=RunPolicy(), events=events, engine=ReferenceEngine(), output=tmp_path)
    assert caught.value.failure.code == ErrorCode.ACTION_INVALID


def test_strategy_cannot_mutate_bound_market_input(tmp_path: Path) -> None:
    class Mutator:
        declaration = declaration().model_copy(update={"actions": frozenset({"none"})})
        def decide(self, event: MarketEvent, account: AccountSnapshot):
            del account
            event.values["close"] = Decimal(999999)
            return (NoOp(reason="mutated copy"),)
    events = bars()
    original_hash = fingerprint(events)
    report = run(run_id="immutable-input", strategy=Mutator(), dataset=dataset(events),
                 engine_capability=reference_capability(dataset(events).fields),
                 policy=RunPolicy(), events=events, engine=ReferenceEngine(), output=tmp_path)
    assert fingerprint(events) == original_hash == report.source_events_sha256
    verify_bundle_file(tmp_path / "bundle.json")


def test_train_memory_cannot_replace_a_recoverable_artifact(tmp_path: Path) -> None:
    class MemoryOnly(TinyLearner):
        def train(self, request: TrainingRequest) -> bytes:
            del request
            self.learned = Decimal(42)
            return b"not-a-model"
        def load(self, payload: bytes) -> None:
            del payload
        def decide(self, event: MarketEvent, account: AccountSnapshot):
            del event, account
            return (NoOp(reason=str(self.learned)),)
    events = bars()
    training = TrainingRequest(dataset_id="local-bars", seed=5,
        samples=(TrainingSample(features={"close": Decimal(100)}, target=Decimal(1)),))
    with pytest.raises(ContractFault) as caught:
        run(run_id="memory-only", strategy=MemoryOnly(), dataset=dataset(events),
            engine_capability=reference_capability(dataset(events).fields),
            policy=RunPolicy(), events=events, engine=ReferenceEngine(),
            output=tmp_path, training=training)
    assert caught.value.failure.stage == "infer"
    assert not (tmp_path / "report.json").exists()


def test_host_factory_cannot_reuse_a_trainable_instance(tmp_path: Path) -> None:
    class ReusedLearner(TinyLearner):
        def restart_for_inference(self) -> ReusedLearner:
            return self

    learner = ReusedLearner()
    events = bars()
    training = TrainingRequest(dataset_id="local-bars", seed=5,
        samples=(TrainingSample(features={"close": Decimal(100)}, target=Decimal(1)),))
    with pytest.raises(ContractFault) as caught:
        run(run_id="reused-instance", strategy=learner, dataset=dataset(events),
            engine_capability=reference_capability(dataset(events).fields),
            policy=RunPolicy(), events=events, engine=ReferenceEngine(),
            output=tmp_path, training=training,
            strategy_factory=learner.restart_for_inference)
    assert caught.value.failure.stage == "load"
    assert caught.value.failure.code == ErrorCode.MODEL_CORRUPT
    assert not (tmp_path / "report.json").exists()


def test_execution_settings_change_plan_and_offline_bundle(tmp_path: Path) -> None:
    class Idle:
        declaration = declaration().model_copy(update={"actions": frozenset({"none"})})
        def decide(self, event: MarketEvent, account: AccountSnapshot):
            del event, account
            return (NoOp(reason="idle"),)
    events = bars()
    reports = [run(run_id=f"cash-{cash}", strategy=Idle(), dataset=dataset(events),
                   engine_capability=reference_capability(dataset(events).fields),
                   policy=RunPolicy(), events=events,
                   engine=ReferenceEngine(initial_cash=Decimal(cash)),
                   output=tmp_path / str(cash)) for cash in (1000, 2000)]
    assert reports[0].plan.engine_profile_sha256 != reports[1].plan.engine_profile_sha256
    bundle_path = tmp_path / "1000/bundle.json"
    verify_bundle_file(bundle_path)
    changed = json.loads(bundle_path.read_text())
    changed["report"]["plan"]["engine_profile"]["settings"]["initial_cash"] = "2000"
    bundle_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="engine profile"):
        verify_bundle_file(bundle_path)


def test_quote_only_external_engine_needs_no_reference_price(tmp_path: Path) -> None:
    now = datetime(2026, 1, 2, 10, tzinfo=UTC)
    events = tuple(MarketEvent(event_id=f"quote-{index}", symbol="UNSEEN",
        event_time=now+timedelta(seconds=index), available_time=now+timedelta(seconds=index),
        values={"bid_price": Decimal(100+index), "ask_price": Decimal(101+index)})
        for index in range(2))
    fields = frozenset(events[0].values)
    source = DatasetDeclaration(dataset_id="quotes", granularity="tick", fields=fields,
        symbols=frozenset({"UNSEEN"}), event_count=len(events),
        content_sha256=fingerprint(events))
    class Idle:
        declaration = StrategyDeclaration(strategy_id="outside.catalog.quote-only", kind="rule",
            data=DataNeed(granularity="tick", fields=fields, symbols=frozenset({"UNSEEN"})),
            actions=frozenset({"none"}), training_required=False, source="urn:test:quote")
        def decide(self, event: MarketEvent, account: AccountSnapshot):
            del event, account
            return (NoOp(reason="quote-only"),)
    class QuoteEngine:
        def capability(self, fields: frozenset[str]) -> EngineCapability:
            return EngineCapability(engine_id="outside.quote", granularities=frozenset({"tick"}),
                                    data_fields=fields, actions=frozenset({"none"}))
        def profile(self) -> EngineProfile:
            return EngineProfile(engine_id="outside.quote",
                                 settings={"initial_cash": "1000", "matching": "none"})
        def run(self, *, plan, strategy, events):
            accounts = tuple(AccountSnapshot(timestamp=event.available_time,
                cash=Decimal(1000), equity=Decimal(1000), positions=(), open_order_ids=())
                for event in events)
            decisions = tuple(Decision(event_id=event.event_id,
                event_time=event.available_time,
                actions=strategy.decide(event, account))
                for event, account in zip(events, accounts, strict=True))
            return decisions, (), (), accounts
    engine = QuoteEngine()
    report = run(run_id="quote-only", strategy=Idle(), dataset=source,
        engine_capability=engine.capability(fields), policy=RunPolicy(),
        events=events, engine=engine, output=tmp_path)
    assert report.plan.required_fields == fields
    assert report.final_equity == 1000
