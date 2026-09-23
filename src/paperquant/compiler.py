from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, time
from decimal import Decimal
from typing import Any, NoReturn
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from paperquant.models import (
    ContractFault,
    Conversion,
    DatasetDeclaration,
    EngineCapability,
    EngineProfile,
    ErrorCode,
    ExecutionPlan,
    Failure,
    Granularity,
    MarketEvent,
    RunPolicy,
    Stage,
    StrategyDeclaration,
    TrainingRequest,
)


def canonical_document(value: Any) -> Any:
    if isinstance(value, MarketEvent) and value.bar_open_time is None:
        # Preserve the canonical tick payload when adding a bar-only field.
        return canonical_document(value.model_dump(mode="python", exclude={"bar_open_time"}))
    if isinstance(value, BaseModel):
        return canonical_document(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): canonical_document(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        items = (canonical_document(item) for item in value)
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [canonical_document(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, time)):
        return value.isoformat()
    return value


def fingerprint(value: BaseModel | tuple[MarketEvent, ...] | dict[str, str]) -> str:
    payload = json.dumps(
        canonical_document(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject(run_id: str, stage: Stage, code: ErrorCode, message: str, **details: str) -> NoReturn:
    raise ContractFault(
        Failure(run_id=run_id, stage=stage, code=code, message=message, details=details)
    )


def _check_source(
    run_id: str, dataset: DatasetDeclaration, events: tuple[MarketEvent, ...]
) -> None:
    if len(events) != dataset.event_count or fingerprint(events) != dataset.content_sha256:
        _reject(
            run_id, Stage.INPUT, ErrorCode.INPUT_INVALID, "Declared event hash or count differs"
        )
    ids: set[str] = set()
    previous: tuple[datetime, str] | None = None
    last_available_by_symbol: dict[str, datetime] = {}
    sessions: set[tuple[str, str]] = set()
    try:
        session_zone = ZoneInfo(dataset.session_timezone)
    except ZoneInfoNotFoundError:
        _reject(run_id, Stage.INPUT, ErrorCode.INPUT_INVALID,
                "Dataset session timezone is unknown")
    if dataset.granularity == Granularity.DAY and (
        dataset.session_open is None or dataset.session_close is None
    ):
        _reject(run_id, Stage.INPUT, ErrorCode.TIMEFRAME_MISMATCH,
                "Daily data requires an explicit session open and close")
    for event in events:
        if event.event_id in ids or event.symbol not in dataset.symbols:
            _reject(
                run_id, Stage.INPUT, ErrorCode.INPUT_INVALID, "Duplicate event or unknown symbol"
            )
        ids.add(event.event_id)
        if not dataset.fields <= event.values.keys():
            _reject(run_id, Stage.INPUT, ErrorCode.DATA_FIELD_MISSING, "Actual event lacks fields")
        if dataset.granularity != Granularity.TICK:
            if event.bar_open_time is None:
                _reject(run_id, Stage.INPUT, ErrorCode.INPUT_INVALID, "Bar open time is required")
            prior = last_available_by_symbol.get(event.symbol)
            if prior is not None and event.bar_open_time < prior:
                _reject(
                    run_id,
                    Stage.INPUT,
                    ErrorCode.INPUT_INVALID,
                    "Bar open precedes the prior signal availability",
                )
            duration = (event.event_time - event.bar_open_time).total_seconds()
            if dataset.granularity == Granularity.MINUTE and not 0 <= duration <= 60:
                _reject(run_id, Stage.INPUT, ErrorCode.TIMEFRAME_MISMATCH,
                        "Minute bar duration is outside one minute", event_id=event.event_id)
            if dataset.granularity == Granularity.DAY:
                if not 3600 <= duration <= 86400:
                    _reject(run_id, Stage.INPUT, ErrorCode.TIMEFRAME_MISMATCH,
                            "Daily bar does not span a trading session",
                            event_id=event.event_id)
                opening = event.bar_open_time.astimezone(session_zone).time().replace(tzinfo=None)
                closing = event.event_time.astimezone(session_zone).time().replace(tzinfo=None)
                if opening != dataset.session_open or closing != dataset.session_close:
                    _reject(run_id, Stage.INPUT, ErrorCode.TIMEFRAME_MISMATCH,
                            "Daily bar does not match its declared trading session",
                            event_id=event.event_id)
                session = (event.symbol,
                           event.bar_open_time.astimezone(session_zone).date().isoformat())
                if session in sessions:
                    _reject(run_id, Stage.INPUT, ErrorCode.TIMEFRAME_MISMATCH,
                            "More than one daily bar exists for a symbol and session",
                            event_id=event.event_id)
                sessions.add(session)
        elif event.bar_open_time is not None:
            _reject(run_id, Stage.INPUT, ErrorCode.INPUT_INVALID, "Tick has a bar open time")
        ordering = (event.available_time, event.event_id)
        if previous is not None and ordering <= previous:
            _reject(
                run_id, Stage.INPUT, ErrorCode.INPUT_INVALID, "Events are not in availability order"
            )
        previous = ordering
        last_available_by_symbol[event.symbol] = event.available_time


def _map_symbols(
    run_id: str,
    events: tuple[MarketEvent, ...],
    source: frozenset[str],
    required: frozenset[str],
    policy: RunPolicy,
) -> tuple[MarketEvent, ...]:
    mapping = policy.symbol_map
    if set(mapping) != set(source - required) or set(mapping.values()) != set(required - source):
        _reject(
            run_id,
            Stage.COMPILE,
            ErrorCode.SYMBOL_MAPPING_FAILED,
            "An explicit one-to-one mapping is required for missing symbols",
        )
    if len(set(mapping.values())) != len(mapping):
        _reject(run_id, Stage.COMPILE, ErrorCode.SYMBOL_MAPPING_FAILED, "Mapping is not injective")
    return tuple(
        event.model_copy(update={"symbol": mapping.get(event.symbol, event.symbol)})
        for event in events
    )


def _aggregate_days(
    run_id: str, events: tuple[MarketEvent, ...], session_timezone: str
) -> tuple[MarketEvent, ...]:
    required = {"open", "high", "low", "close", "volume"}
    if any(not required <= event.values.keys() for event in events):
        _reject(
            run_id,
            Stage.COMPILE,
            ErrorCode.DATA_FIELD_MISSING,
            "Daily aggregation requires complete OHLCV bars",
        )
    groups: dict[tuple[str, str], list[MarketEvent]] = defaultdict(list)
    zone = ZoneInfo(session_timezone)
    for event in events:
        assert event.bar_open_time is not None
        key = (event.symbol, event.bar_open_time.astimezone(zone).date().isoformat())
        groups[key].append(event)
    result: list[MarketEvent] = []
    for (symbol, day), bars in groups.items():
        bars.sort(key=lambda item: item.event_time)
        values = {
            "open": bars[0].values["open"],
            "high": max(item.values["high"] for item in bars),
            "low": min(item.values["low"] for item in bars),
            "close": bars[-1].values["close"],
            "volume": sum((item.values["volume"] for item in bars), Decimal("0")),
        }
        result.append(
            MarketEvent(
                event_id=f"day:{day}:{symbol}",
                symbol=symbol,
                event_time=bars[-1].event_time,
                available_time=max(item.available_time for item in bars),
                bar_open_time=bars[0].bar_open_time,
                values=values,
            )
        )
    return tuple(sorted(result, key=lambda item: (item.available_time, item.event_id)))


def _check_effective_timing(run_id: str, events: tuple[MarketEvent, ...]) -> None:
    last_available: dict[str, datetime] = {}
    for event in events:
        if event.bar_open_time is not None:
            previous = last_available.get(event.symbol)
            if previous is not None and event.bar_open_time < previous:
                _reject(run_id, Stage.INPUT, ErrorCode.INPUT_INVALID,
                        "Converted bar opens before previous signal became available")
        last_available[event.symbol] = event.available_time


def compile_run(
    *,
    run_id: str,
    strategy: StrategyDeclaration,
    dataset: DatasetDeclaration,
    engine: EngineCapability,
    engine_profile: EngineProfile | None = None,
    policy: RunPolicy,
    events: tuple[MarketEvent, ...],
    training: TrainingRequest | None = None,
    package_source_sha256: str | None = None,
    package_class_name: str | None = None,
) -> tuple[ExecutionPlan, tuple[MarketEvent, ...]]:
    _check_source(run_id, dataset, events)
    if engine_profile is None:
        engine_profile = EngineProfile(engine_id=engine.engine_id,
                                       settings={"mode": "compile-only"})
    if engine_profile.engine_id != engine.engine_id:
        _reject(run_id, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED,
                "Execution profile belongs to another engine")
    if strategy.training_required:
        if training is None or training.dataset_id != dataset.dataset_id or not training.samples:
            _reject(
                run_id,
                Stage.TRAIN,
                ErrorCode.TRAINING_FAILED,
                "Training request is absent or mismatched",
            )
    elif training is not None:
        _reject(run_id, Stage.TRAIN, ErrorCode.TRAINING_FAILED, "Rule strategy cannot train")
    if not strategy.actions <= engine.actions:
        _reject(
            run_id, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED, "Engine lacks declared actions"
        )
    if (engine.max_symbols is not None and len(strategy.data.symbols) > engine.max_symbols):
        _reject(run_id, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED,
                "Engine cannot support the declared number of symbols")
    if policy.financing_mode not in engine.financing_modes:
        _reject(run_id, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED,
                "Engine cannot support the declared financing mode")
    if not strategy.data.fields <= dataset.fields:
        missing = ",".join(sorted(strategy.data.fields - dataset.fields))
        _reject(
            run_id,
            Stage.COMPILE,
            ErrorCode.DATA_FIELD_MISSING,
            "Dataset lacks fields",
            missing=missing,
        )
    execution_fields = engine.execution_fields.get(strategy.data.granularity,
                                                    frozenset())

    conversions: list[Conversion] = []
    effective = events
    if strategy.data.symbols != dataset.symbols:
        mapped = _map_symbols(run_id, effective, dataset.symbols, strategy.data.symbols, policy)
        conversions.append(
            Conversion(
                transformation="symbol_map",
                reason="Strategy symbols differ from the declared dataset symbols",
                policy_switch="symbol_map",
                source_event_count=len(effective),
                output_event_count=len(mapped),
                affected_symbols=tuple(sorted(policy.symbol_map)),
                input_sha256=fingerprint(effective),
                output_sha256=fingerprint(mapped),
                parameters=policy.symbol_map,
                lossy=False,
            )
        )
        effective = mapped
    elif policy.symbol_map:
        _reject(
            run_id, Stage.COMPILE, ErrorCode.SYMBOL_MAPPING_FAILED, "Unused mapping is forbidden"
        )

    if strategy.data.granularity != dataset.granularity:
        if not (
            dataset.granularity == Granularity.MINUTE
            and strategy.data.granularity == Granularity.DAY
            and policy.allow_day_aggregation
        ):
            _reject(
                run_id,
                Stage.COMPILE,
                ErrorCode.TIMEFRAME_MISMATCH,
                "No authorized conversion can satisfy the required granularity",
            )
        if (policy.aggregation_session_open is None
            or policy.aggregation_session_close is None):
            _reject(run_id, Stage.COMPILE, ErrorCode.TIMEFRAME_MISMATCH,
                    "Day aggregation requires an explicit target session")
        aggregated = _aggregate_days(run_id, effective, dataset.session_timezone)
        conversions.append(
            Conversion(
                transformation="minute_to_day",
                reason="Strategy requires daily bars while the dataset provides minute bars",
                policy_switch="allow_day_aggregation",
                source_event_count=len(effective),
                output_event_count=len(aggregated),
                affected_symbols=tuple(sorted({event.symbol for event in effective})),
                input_sha256=fingerprint(effective),
                output_sha256=fingerprint(aggregated),
                parameters={"calendar": dataset.session_timezone,
                            "session_open": policy.aggregation_session_open.isoformat(),
                            "session_close": policy.aggregation_session_close.isoformat()},
                lossy=True,
            )
        )
        effective = aggregated
    elif policy.allow_day_aggregation:
        _reject(
            run_id, Stage.COMPILE, ErrorCode.TIMEFRAME_MISMATCH, "Unused aggregation is forbidden"
        )

    _check_effective_timing(run_id, effective)
    if strategy.data.granularity != dataset.granularity:
        effective_declaration = DatasetDeclaration(
            dataset_id=dataset.dataset_id,
            granularity=strategy.data.granularity,
            fields=frozenset.intersection(*(frozenset(item.values) for item in effective)),
            symbols=strategy.data.symbols,
            event_count=len(effective),
            content_sha256=fingerprint(effective),
            session_timezone=dataset.session_timezone,
            session_open=policy.aggregation_session_open,
            session_close=policy.aggregation_session_close,
        )
        _check_source(run_id, effective_declaration, effective)

    if strategy.data.granularity not in engine.granularities:
        _reject(run_id, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED, "Engine lacks granularity")
    required_fields = strategy.data.fields | execution_fields
    if not required_fields <= set.intersection(*(set(item.values) for item in effective)):
        _reject(run_id, Stage.COMPILE, ErrorCode.DATA_FIELD_MISSING,
                "Effective dataset lacks engine execution fields")
    if not required_fields <= engine.data_fields:
        _reject(run_id, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED, "Engine lacks market fields")
    counts = {
        symbol: sum(event.symbol == symbol for event in effective)
        for symbol in strategy.data.symbols
    }
    if any(count < strategy.data.minimum_events_per_symbol for count in counts.values()):
        _reject(run_id, Stage.INPUT, ErrorCode.INPUT_INVALID, "Insufficient events for a symbol")
    if any(not required_fields <= event.values.keys() for event in effective):
        _reject(run_id, Stage.INPUT, ErrorCode.DATA_FIELD_MISSING, "Effective event lacks fields")
    return (
        ExecutionPlan(
            run_id=run_id,
            strategy_id=strategy.strategy_id,
            dataset_id=dataset.dataset_id,
            engine_id=engine.engine_id,
            sandbox=policy.sandbox,
            financing_mode=policy.financing_mode,
            granularity=strategy.data.granularity,
            symbols=strategy.data.symbols,
            required_fields=required_fields,
            allowed_actions=strategy.actions,
            max_abs_position=strategy.max_abs_position,
            max_order_quantity=strategy.max_order_quantity,
            declaration_sha256=fingerprint(strategy),
            dataset_sha256=fingerprint(dataset),
            effective_events_sha256=fingerprint(effective),
            training_sha256=fingerprint(training) if training is not None else None,
            engine_capability=engine,
            engine_sha256=fingerprint(engine),
            engine_profile=engine_profile,
            engine_profile_sha256=fingerprint(engine_profile),
            policy=policy,
            policy_sha256=fingerprint(policy),
            package_source_sha256=package_source_sha256,
            package_class_name=package_class_name,
            conversions=tuple(conversions),
        ),
        effective,
    )
