"""Market-type and timing checks use independent source events, not catalog fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.compiler import compile_run, fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.evidence import verify_bundle
from paperquant.models import (
    ContractFault,
    DataNeed,
    DatasetDeclaration,
    EngineCapability,
    ErrorCode,
    MarketEvent,
    RunBundle,
    RunPolicy,
    StrategyDeclaration,
    TrainingRequest,
    TrainingSample,
)
from paperquant.package import run_package

NOW = datetime(2026, 1, 2, 10, tzinfo=UTC)


def _compile(
    events: tuple[MarketEvent, ...], *, granularity: str = "tick",
    market_kind: str | None = None, book_depth: int | None = None,
    minimum_book_depth: int | None = None, bar_seconds: int | None = None,
    required_bar_seconds: int | None = None, max_staleness_seconds: int | None = None,
) -> None:
    fields = frozenset(events[0].values)
    dataset = DatasetDeclaration(
        dataset_id="independent-market", granularity=granularity, fields=fields,
        symbols=frozenset({"AAA"}), event_count=len(events),
        content_sha256=fingerprint(events), market_kind=market_kind,
        book_depth=book_depth, bar_seconds=bar_seconds,
    )
    strategy = StrategyDeclaration(
        strategy_id="outside.catalog.market", kind="rule", source="urn:market-test",
        data=DataNeed(
            granularity=granularity, fields=fields, symbols=frozenset({"AAA"}),
            market_kind=market_kind, minimum_book_depth=minimum_book_depth,
            bar_seconds=required_bar_seconds,
            max_staleness_seconds=max_staleness_seconds,
        ),
        actions=frozenset({"none"}), training_required=False,
    )
    capability = EngineCapability(
        engine_id="outside.market", granularities=frozenset({granularity}),
        data_fields=fields, actions=frozenset({"none"}),
    )
    compile_run(run_id="semantic-test", strategy=strategy, dataset=dataset,
                engine=capability, policy=RunPolicy(), events=events)


def _event(values: dict[str, str], *, available_delay: int = 0,
           bar_seconds: int | None = None, trade_id: str | None = None) -> MarketEvent:
    return MarketEvent(
        event_id="outside-1", symbol="AAA", event_time=NOW,
        available_time=NOW + timedelta(seconds=available_delay),
        bar_open_time=NOW - timedelta(seconds=bar_seconds) if bar_seconds else None,
        trade_id=trade_id,
        values={key: Decimal(value) for key, value in values.items()},
    )


@pytest.mark.parametrize("bid,ask", [("100", "100"), ("102", "101"), ("0", "101")])
def test_crossed_locked_or_nonpositive_l1_fails_structurally(bid: str, ask: str) -> None:
    event = _event({"bid_price": bid, "ask_price": ask,
                    "bid_size": "5", "ask_size": "7"})
    with pytest.raises(ContractFault) as caught:
        _compile((event,), market_kind="quote_l1")
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID
    assert caught.value.failure.details["event_id"] == event.event_id


@pytest.mark.parametrize("field,value", [
    ("bid_price_2", "100"), ("ask_price_2", "100.5"),
    ("bid_size_2", "0"),
])
def test_l2_ordering_and_depth_are_checked(field: str, value: str) -> None:
    values = {"bid_price": "100", "ask_price": "101",
              "bid_size": "5", "ask_size": "7",
              "bid_price_2": "99", "ask_price_2": "102",
              "bid_size_2": "4", "ask_size_2": "6"}
    _compile((_event(values),), market_kind="book_l2", book_depth=2,
             minimum_book_depth=2)
    values[field] = value
    with pytest.raises(ContractFault) as caught:
        _compile((_event(values),), market_kind="book_l2", book_depth=2,
                 minimum_book_depth=2)
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


def test_strategy_book_depth_is_a_negotiated_requirement() -> None:
    event = _event({"bid_price": "100", "ask_price": "101",
                    "bid_size": "5", "ask_size": "7",
                    "bid_price_2": "99", "ask_price_2": "102",
                    "bid_size_2": "4", "ask_size_2": "6"})
    with pytest.raises(ContractFault) as caught:
        _compile((event,), market_kind="book_l2", book_depth=2,
                 minimum_book_depth=3)
    assert caught.value.failure.code == ErrorCode.DATA_FIELD_MISSING


def test_l2_third_level_must_be_complete_and_contiguous() -> None:
    values = {"bid_price": "100", "ask_price": "101",
              "bid_size": "5", "ask_size": "7",
              "bid_price_2": "99", "ask_price_2": "102",
              "bid_size_2": "4", "ask_size_2": "6",
              "bid_price_3": "98", "ask_price_3": "103",
              "bid_size_3": "3", "ask_size_3": "5"}
    _compile((_event(values),), market_kind="book_l2", book_depth=3,
             minimum_book_depth=3)
    del values["ask_size_2"]
    with pytest.raises(ContractFault) as caught:
        _compile((_event(values),), market_kind="book_l2", book_depth=3,
                 minimum_book_depth=3)
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


def test_explicit_trade_identity_and_size_are_required() -> None:
    values = {"price": "100.2", "trade_size": "3"}
    _compile((_event(values, trade_id="trade-1"),), market_kind="trade")
    with pytest.raises(ContractFault) as caught:
        _compile((_event(values),), market_kind="trade")
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


def test_five_minute_bars_and_staleness_are_declared_independently() -> None:
    values = {"open": "100", "high": "102", "low": "99",
              "close": "101", "volume": "300"}
    event = _event(values, bar_seconds=300, available_delay=9)
    _compile((event,), granularity="minute", market_kind="bar",
             bar_seconds=300, required_bar_seconds=300, max_staleness_seconds=9)
    for overrides, expected in [
        ({"required_bar_seconds": 60}, ErrorCode.TIMEFRAME_MISMATCH),
        ({"max_staleness_seconds": 8}, ErrorCode.INPUT_INVALID),
    ]:
        options = {"granularity": "minute", "market_kind": "bar",
                   "bar_seconds": 300, "required_bar_seconds": 300,
                   "max_staleness_seconds": 9, **overrides}
        with pytest.raises(ContractFault) as caught:
            _compile((event,), **options)
        assert caught.value.failure.code == expected
    shorter = _event(values, bar_seconds=299)
    with pytest.raises(ContractFault) as caught:
        _compile((shorter,), granularity="minute", market_kind="bar",
                 bar_seconds=300, required_bar_seconds=300)
    assert caught.value.failure.code == ErrorCode.TIMEFRAME_MISMATCH


def test_invalid_ohlc_is_rejected_even_when_all_fields_exist() -> None:
    event = _event({"open": "100", "high": "101", "low": "99",
                    "close": "102", "volume": "5"}, bar_seconds=300)
    with pytest.raises(ContractFault) as caught:
        _compile((event,), granularity="minute", market_kind="bar",
                 bar_seconds=300, required_bar_seconds=300)
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


def _compile_rl(samples: tuple[TrainingSample, ...]) -> None:
    event = _event({"price": "100"})
    dataset = DatasetDeclaration(
        dataset_id="trajectory", granularity="tick", fields=frozenset({"price"}),
        symbols=frozenset({"AAA"}), event_count=1, content_sha256=fingerprint((event,)),
    )
    strategy = StrategyDeclaration(
        strategy_id="outside.catalog.trajectory", kind="reinforcement",
        source="urn:trajectory-test", training_required=True,
        data=DataNeed(granularity="tick", fields=frozenset({"price"}),
                      symbols=frozenset({"AAA"})),
        actions=frozenset({"none"}),
    )
    capability = EngineCapability(
        engine_id="outside.trajectory", granularities=frozenset({"tick"}),
        data_fields=frozenset({"price"}), actions=frozenset({"none"}),
    )
    compile_run(run_id="trajectory-test", strategy=strategy, dataset=dataset,
                engine=capability, policy=RunPolicy(), events=(event,),
                training=TrainingRequest(dataset_id="trajectory", seed=3, samples=samples))


def _transition(episode: str, step: int, *, terminal: bool = False,
                truncated: bool = False) -> TrainingSample:
    return TrainingSample(features={"price": Decimal(100)}, action=0,
                          reward=Decimal(1), episode_id=episode, step=step,
                          terminal=terminal, truncated=truncated)


def test_rl_episode_boundaries_and_truncation_are_explicit() -> None:
    _compile_rl((_transition("a", 0), _transition("a", 1, truncated=True),
                 _transition("b", 0, terminal=True)))
    invalid = (
        (_transition("a", 0), _transition("b", 0, terminal=True)),
        (_transition("a", 0, terminal=True), _transition("a", 1, terminal=True)),
        (_transition("a", 0), _transition("a", 2, terminal=True)),
        (_transition("a", 0, terminal=True), _transition("b", 0)),
    )
    for samples in invalid:
        with pytest.raises(ContractFault) as caught:
            _compile_rl(samples)
        assert caught.value.failure.code == ErrorCode.TRAINING_FAILED


def test_old_rl_samples_remain_compatible_without_invented_episode_ids() -> None:
    _compile_rl((TrainingSample(features={"price": Decimal(100)}, action=0,
                                reward=Decimal(1), terminal=True),))


def test_portable_bundle_rejects_resigned_crossed_market(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples/l1_queues"
    dataset = DatasetDeclaration.model_validate_json((fixture / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (fixture / "events.json").read_bytes())
    output = tmp_path / "market-run"
    run_package(run_id="bundle-market", package_dir=root / "strategies/rule.microprice_toy",
                dataset=dataset, engine_capability=reference_capability(dataset.fields),
                policy=RunPolicy(), events=events, engine=ReferenceEngine(), output=output)
    bundle = RunBundle.model_validate_json((output / "bundle.json").read_bytes())
    first = bundle.source_events[0]
    invalid = first.model_copy(update={
        "values": {**first.values, "ask_price": first.values["bid_price"] - Decimal("1")}
    })
    changed = (invalid, *bundle.source_events[1:])
    new_hash = fingerprint(changed)
    new_dataset = bundle.dataset_declaration.model_copy(update={"content_sha256": new_hash})
    new_plan = bundle.report.plan.model_copy(update={
        "dataset_sha256": fingerprint(new_dataset), "effective_events_sha256": new_hash,
    })
    new_report = bundle.report.model_copy(update={
        "plan": new_plan, "source_events_sha256": new_hash,
        "effective_events_sha256": new_hash,
    })
    resigned = bundle.model_copy(update={
        "report": new_report, "dataset_declaration": new_dataset,
        "source_events": changed, "effective_events": changed,
    })
    with pytest.raises(ValueError, match="source market semantics"):
        verify_bundle(resigned)
