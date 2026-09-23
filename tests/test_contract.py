from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from paperquant.compiler import compile_run, fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import (
    AccountSnapshot,
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
    TargetPosition,
    TrainingRequest,
    TrainingSample,
)
from paperquant.runtime import run


def bars() -> tuple[MarketEvent, ...]:
    start = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    prices = (100, 102, 101, 104)
    return tuple(
        MarketEvent(
            event_id=f"bar-{index}",
            symbol="AAA",
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


def dataset(events: tuple[MarketEvent, ...]) -> DatasetDeclaration:
    return DatasetDeclaration(
        dataset_id="local-bars",
        granularity=Granularity.MINUTE,
        fields=frozenset(events[0].values),
        symbols=frozenset({"AAA"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )


def declaration(*, learning: bool = False) -> StrategyDeclaration:
    return StrategyDeclaration(
        strategy_id="sample-learning" if learning else "sample-rule",
        kind=StrategyKind.SUPERVISED if learning else StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.MINUTE,
            fields=frozenset({"close", "open"}),
            symbols=frozenset({"AAA"}),
        ),
        actions=frozenset({"target_position", "none"}),
        training_required=learning,
        source="public-method-description",
        max_abs_position=Decimal("2"),
    )


class ScheduledRule:
    declaration = declaration()

    def __init__(self) -> None:
        self.seen = 0

    def decide(
        self, event: MarketEvent, account: AccountSnapshot
    ) -> tuple[TargetPosition | NoOp, ...]:
        del account
        self.seen += 1
        quantity = Decimal("1") if self.seen < 3 else Decimal("0")
        return (TargetPosition(symbol=event.symbol, quantity=quantity, reason="schedule"),)


class TinyLearner:
    declaration = declaration(learning=True)

    def __init__(self) -> None:
        self.threshold: Decimal | None = None

    def train(self, request: TrainingRequest) -> bytes:
        targets = [sample.target for sample in request.samples]
        assert all(target is not None for target in targets)
        threshold = sum(target for target in targets if target is not None) / len(targets)
        return str(threshold).encode()

    def load(self, payload: bytes) -> None:
        self.threshold = Decimal(payload.decode())

    def decide(self, event: MarketEvent, account: AccountSnapshot) -> tuple[NoOp, ...]:
        del event, account
        assert self.threshold is not None
        return (NoOp(reason="model-loaded"),)


def test_rule_lifecycle_fills_only_on_next_event(tmp_path: Path) -> None:
    events = bars()
    report = run(
        run_id="rule-run",
        strategy=ScheduledRule(),
        dataset=dataset(events),
        engine_capability=reference_capability(frozenset(events[0].values)),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(initial_cash=Decimal("1000")),
        output=tmp_path,
    )
    assert [fill.price for fill in report.fills] == [Decimal("102"), Decimal("104")]
    assert report.fills[0].timestamp == events[1].bar_open_time
    assert report.accounts[-1].positions[0].realized_pnl == Decimal("2")
    assert report.final_equity == Decimal("1002")
    assert (tmp_path / "report.json").is_file()


def test_learning_lifecycle_saves_and_reloads_exact_model(tmp_path: Path) -> None:
    events = bars()
    training = TrainingRequest(
        dataset_id="local-bars",
        seed=11,
        samples=(
            TrainingSample(features={"close": Decimal("100")}, target=Decimal("1")),
            TrainingSample(features={"close": Decimal("101")}, target=Decimal("3")),
        ),
    )
    learner = TinyLearner()
    report = run(
        run_id="learning-run",
        strategy=learner,
        dataset=dataset(events),
        engine_capability=reference_capability(frozenset(events[0].values)),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
        training=training,
    )
    assert learner.threshold == 2
    assert report.artifact is not None
    assert report.artifact.training_sha256 == fingerprint(training)
    assert (tmp_path / report.artifact.relative_path).read_bytes() == b"2"
    assert len(report.decisions) == len(events)


def test_missing_trained_bytes_cannot_be_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import paperquant.runtime as runtime_module

    original = runtime_module._persist_model

    def remove_after_save(**kwargs):
        artifact = original(**kwargs)
        (kwargs["root"] / artifact.relative_path).unlink()
        return artifact

    monkeypatch.setattr(runtime_module, "_persist_model", remove_after_save)
    events = bars()
    training = TrainingRequest(
        dataset_id="local-bars", seed=11,
        samples=(TrainingSample(features={"close": Decimal(100)}, target=Decimal(1)),),
    )
    with pytest.raises(ContractFault) as caught:
        run(
            run_id="missing-model", strategy=TinyLearner(), dataset=dataset(events),
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(), events=events, engine=ReferenceEngine(),
            output=tmp_path, training=training,
        )
    assert caught.value.failure.code == ErrorCode.MODEL_MISSING
    assert not (tmp_path / "report.json").exists()


def test_target_derived_fill_obeys_per_order_quantity_limit(tmp_path: Path) -> None:
    class TooLargeTarget:
        declaration = declaration().model_copy(
            update={"max_order_quantity": Decimal(1)})

        def decide(self, event: MarketEvent, account: AccountSnapshot):
            del account
            return (TargetPosition(symbol=event.symbol, quantity=Decimal(2), reason="probe"),)

    events = bars()
    with pytest.raises(ContractFault) as caught:
        run(
            run_id="large-derived-order", strategy=TooLargeTarget(), dataset=dataset(events),
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(), events=events, engine=ReferenceEngine(), output=tmp_path,
        )
    assert caught.value.failure.code == ErrorCode.ORDER_REJECTED
    assert not (tmp_path / "report.json").exists()


def test_cash_only_rejects_unfunded_fill_and_margin_is_explicit(tmp_path: Path) -> None:
    events = bars()
    args = dict(
        dataset=dataset(events),
        engine_capability=reference_capability(frozenset(events[0].values)),
        events=events,
        engine=ReferenceEngine(initial_cash=Decimal("1")),
    )
    with pytest.raises(ContractFault) as caught:
        run(
            run_id="cash-only", strategy=ScheduledRule(), policy=RunPolicy(),
            output=tmp_path / "cash-only", **args,
        )
    assert caught.value.failure.code == ErrorCode.ORDER_REJECTED
    assert caught.value.failure.details["available_cash"] == "1"
    assert not (tmp_path / "cash-only/report.json").exists()

    report = run(
        run_id="margin", strategy=ScheduledRule(),
        policy=RunPolicy(financing_mode="unbounded_margin"),
        output=tmp_path / "margin", **args,
    )
    assert report.plan.financing_mode == "unbounded_margin"
    assert report.plan.policy.financing_mode == "unbounded_margin"
    assert min(account.cash for account in report.accounts) < 0
    assert report.final_equity == Decimal("3")


def test_engine_exception_is_structured_backtest_failure(tmp_path: Path) -> None:
    class BrokenEngine(ReferenceEngine):
        def run(self, *, plan, strategy, events):
            raise RuntimeError("engine-internal-failure")

    events = bars()
    with pytest.raises(ContractFault) as caught:
        run(
            run_id="broken-engine", strategy=ScheduledRule(), dataset=dataset(events),
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(), events=events, engine=BrokenEngine(), output=tmp_path,
        )
    assert caught.value.failure.stage == "backtest"
    assert caught.value.failure.code == ErrorCode.BACKTEST_FAILED
    assert caught.value.failure.details == {"cause": "RuntimeError"}
    assert not (tmp_path / "report.json").exists()


def test_engine_malformed_trace_is_structured_backtest_failure(tmp_path: Path) -> None:
    class MalformedEngine(ReferenceEngine):
        def run(self, *, plan, strategy, events):
            return (), (), (), ("not-an-account",)

    events = bars()
    with pytest.raises(ContractFault) as caught:
        run(
            run_id="malformed-engine", strategy=ScheduledRule(), dataset=dataset(events),
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(), events=events, engine=MalformedEngine(), output=tmp_path,
        )
    assert caught.value.failure.code == ErrorCode.BACKTEST_FAILED
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("failure", ["field", "timeframe", "symbol", "hash"])
def test_compiler_rejects_unavailable_inputs(failure: str) -> None:
    events = bars()
    source = dataset(events)
    spec = declaration()
    if failure == "field":
        spec = spec.model_copy(
            update={"data": spec.data.model_copy(update={"fields": frozenset({"missing"})})}
        )
    elif failure == "timeframe":
        spec = spec.model_copy(
            update={"data": spec.data.model_copy(update={"granularity": Granularity.DAY})}
        )
    elif failure == "symbol":
        spec = spec.model_copy(
            update={"data": spec.data.model_copy(update={"symbols": frozenset({"OTHER"})})}
        )
    else:
        source = source.model_copy(update={"content_sha256": "0" * 64})
    with pytest.raises(ContractFault) as caught:
        compile_run(
            run_id="bad-run",
            strategy=spec,
            dataset=source,
            engine=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(),
            events=events,
        )
    assert caught.value.failure.code in {
        ErrorCode.DATA_FIELD_MISSING,
        ErrorCode.TIMEFRAME_MISMATCH,
        ErrorCode.SYMBOL_MAPPING_FAILED,
        ErrorCode.INPUT_INVALID,
    }
    assert caught.value.failure.fallback_used is False


def test_strict_mode_fails_without_attested_worker(tmp_path: Path) -> None:
    events = bars()
    with pytest.raises(ContractFault) as caught:
        run(
            run_id="strict-run",
            strategy=ScheduledRule(),
            dataset=dataset(events),
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(sandbox="strict"),
            events=events,
            engine=ReferenceEngine(),
            output=tmp_path,
        )
    assert caught.value.failure.code == ErrorCode.SANDBOX_UNAVAILABLE


def test_training_request_mutation_cannot_reuse_planned_evidence(tmp_path: Path) -> None:
    events = bars()
    training = TrainingRequest(
        dataset_id="local-bars",
        seed=7,
        samples=(TrainingSample(features={"close": Decimal("100")}, target=Decimal("1")),),
    )

    class MutatingLearner(TinyLearner):
        def train(self, request: TrainingRequest) -> bytes:
            request.samples[0].features["close"] = Decimal("999")
            return super().train(request)

    with pytest.raises(ContractFault) as caught:
        run(
            run_id="mutated-training",
            strategy=MutatingLearner(),
            dataset=dataset(events),
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(),
            events=events,
            engine=ReferenceEngine(),
            output=tmp_path,
            training=training,
        )
    assert caught.value.failure.code == ErrorCode.TRAINING_FAILED
    assert not list(tmp_path.glob("models/*"))


def test_explicit_symbol_mapping_records_both_event_hashes() -> None:
    events = bars()
    mapped = tuple(event.model_copy(update={"symbol": "SOURCE"}) for event in events)
    source = DatasetDeclaration(
        dataset_id="mapped-bars",
        granularity=Granularity.MINUTE,
        fields=frozenset(events[0].values),
        symbols=frozenset({"SOURCE"}),
        event_count=len(mapped),
        content_sha256=fingerprint(mapped),
    )
    plan, effective = compile_run(
        run_id="mapped-run",
        strategy=declaration(),
        dataset=source,
        engine=reference_capability(frozenset(events[0].values)),
        policy=RunPolicy(symbol_map={"SOURCE": "AAA"}),
        events=mapped,
    )
    assert effective == events
    assert len(plan.conversions) == 1
    assert plan.conversions[0].input_sha256 == fingerprint(mapped)
    assert plan.conversions[0].output_sha256 == fingerprint(events)
    assert plan.conversions[0].lossy is False
    assert plan.conversions[0].policy_switch == "symbol_map"
    assert plan.conversions[0].can_disable is True
    assert plan.conversions[0].source_event_count == len(mapped)
    assert plan.conversions[0].output_event_count == len(events)
    assert plan.conversions[0].affected_symbols == ("SOURCE",)
    assert plan.conversions[0].reason


def test_minute_to_day_aggregation_requires_explicit_policy() -> None:
    events = bars()
    spec = declaration().model_copy(
        update={
            "data": DataNeed(
                granularity=Granularity.DAY,
                fields=frozenset({"open", "close", "volume"}),
                symbols=frozenset({"AAA"}),
            )
        }
    )
    with pytest.raises(ContractFault) as caught:
        compile_run(
            run_id="aggregation-denied",
            strategy=spec,
            dataset=dataset(events),
            engine=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(),
            events=events,
        )
    assert caught.value.failure.code == ErrorCode.TIMEFRAME_MISMATCH
    plan, daily = compile_run(
        run_id="aggregation-allowed",
        strategy=spec,
        dataset=dataset(events),
        engine=reference_capability(frozenset(events[0].values)),
        policy=RunPolicy(allow_day_aggregation=True),
        events=events,
    )
    assert len(daily) == 1
    assert daily[0].values["open"] == events[0].values["open"]
    assert daily[0].values["close"] == events[-1].values["close"]
    assert daily[0].values["volume"] == sum(event.values["volume"] for event in events)
    assert plan.conversions[0].lossy is True
    assert plan.conversions[0].policy_switch == "allow_day_aggregation"
    assert plan.conversions[0].can_disable is True
    assert plan.conversions[0].source_event_count == len(events)
    assert plan.conversions[0].output_event_count == len(daily)
    assert plan.conversions[0].affected_symbols == ("AAA",)


def test_late_bar_cannot_reorder_a_daily_close_silently() -> None:
    original = bars()[:3]
    late = original[1].model_copy(
        update={"available_time": original[1].available_time + timedelta(minutes=5)})
    arrivals = (original[0], original[2], late)
    source = dataset(arrivals).model_copy(
        update={"event_count": len(arrivals), "content_sha256": fingerprint(arrivals)})
    spec = declaration().model_copy(update={"data": DataNeed(
        granularity=Granularity.DAY,
        fields=frozenset({"open", "close", "volume"}),
        symbols=frozenset({"AAA"}),
    )})
    with pytest.raises(ContractFault) as caught:
        compile_run(
            run_id="late-aggregation", strategy=spec, dataset=source,
            engine=reference_capability(source.fields),
            policy=RunPolicy(allow_day_aggregation=True), events=arrivals,
        )
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


def test_direct_engine_call_rejects_misattributed_strategy() -> None:
    events = bars()
    plan, effective = compile_run(
        run_id="bound-run",
        strategy=declaration(),
        dataset=dataset(events),
        engine=reference_capability(frozenset(events[0].values)),
        policy=RunPolicy(),
        events=events,
    )

    class DifferentRule(ScheduledRule):
        declaration = declaration().model_copy(update={"source": "different-research-claim"})

    with pytest.raises(ContractFault) as caught:
        ReferenceEngine().run(plan=plan, strategy=DifferentRule(), events=effective)
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


def test_direct_engine_call_rejects_forged_capability_snapshot() -> None:
    events = bars()
    plan, effective = compile_run(
        run_id="bound-engine", strategy=declaration(), dataset=dataset(events),
        engine=reference_capability(frozenset(events[0].values)),
        policy=RunPolicy(), events=events,
    )
    forged = plan.model_copy(update={"engine_capability": plan.engine_capability.model_copy(
        update={"actions": frozenset({"none"})})})
    with pytest.raises(ContractFault) as caught:
        ReferenceEngine().run(plan=forged, strategy=ScheduledRule(), events=effective)
    assert caught.value.failure.code == ErrorCode.ENGINE_UNSUPPORTED


def test_reference_engine_rejects_other_engine_plan(tmp_path: Path) -> None:
    events = bars()
    capability = reference_capability(frozenset(events[0].values)).model_copy(
        update={"engine_id": "unrelated.engine"}
    )
    with pytest.raises(ContractFault) as caught:
        run(
            run_id="wrong-engine",
            strategy=ScheduledRule(),
            dataset=dataset(events),
            engine_capability=capability,
            policy=RunPolicy(),
            events=events,
            engine=ReferenceEngine(),
            output=tmp_path,
        )
    assert caught.value.failure.code == ErrorCode.ENGINE_UNSUPPORTED


def test_declaration_fingerprint_is_independent_of_set_insertion_order() -> None:
    first = declaration()
    second = first.model_copy(
        update={
            "actions": frozenset(["none", "target_position"]),
            "data": first.data.model_copy(update={"fields": frozenset(["open", "close"])}),
        }
    )
    assert fingerprint(first) == fingerprint(second)
