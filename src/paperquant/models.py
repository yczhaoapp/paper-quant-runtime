from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_POSITIVE_DECIMAL_WIRE: dict[str, Any] = {
    "anyOf": [
        {"type": "number", "exclusiveMinimum": 0},
        {"type": "string", "pattern": (
            r"^\+?(?:0*[1-9]\d*(?:\.\d*)?|0*\.\d*[1-9]\d*)(?:[eE][+-]?\d+)?$"
        )},
    ]
}
_NONNEGATIVE_DECIMAL_WIRE: dict[str, Any] = {
    "anyOf": [
        {"type": "number", "minimum": 0},
        {"type": "string", "pattern": (
            r"^(?:\+?(?:(?:0|[1-9]\d*)(?:\.\d*)?|\.\d+)|-0+(?:\.0*)?)(?:[eE][+-]?\d+)?$"
        )},
    ]
}
_OPTIONAL_POSITIVE_DECIMAL_WIRE: dict[str, Any] = {
    "anyOf": [*_POSITIVE_DECIMAL_WIRE["anyOf"], {"type": "null"}]
}


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class Granularity(StrEnum):
    TICK = "tick"
    MINUTE = "minute"
    DAY = "day"


class StrategyKind(StrEnum):
    RULE = "rule"
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"


class Stage(StrEnum):
    COMPILE = "compile"
    INPUT = "input"
    TRAIN = "train"
    LOAD = "load"
    INFER = "infer"
    BACKTEST = "backtest"
    REPORT = "report"


class ErrorCode(StrEnum):
    SCHEMA_VERSION = "SCHEMA_VERSION"
    DATA_FIELD_MISSING = "DATA_FIELD_MISSING"
    TIMEFRAME_MISMATCH = "TIMEFRAME_MISMATCH"
    SYMBOL_MAPPING_FAILED = "SYMBOL_MAPPING_FAILED"
    ENGINE_UNSUPPORTED = "ENGINE_UNSUPPORTED"
    INPUT_INVALID = "INPUT_INVALID"
    ACTION_INVALID = "ACTION_INVALID"
    ORDER_REJECTED = "ORDER_REJECTED"
    MODEL_MISSING = "MODEL_MISSING"
    MODEL_CORRUPT = "MODEL_CORRUPT"
    TRAINING_FAILED = "TRAINING_FAILED"
    BACKTEST_FAILED = "BACKTEST_FAILED"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"


class Failure(Model):
    run_id: str
    stage: Stage
    code: ErrorCode
    message: str
    details: dict[str, str] = Field(default_factory=dict)
    fallback_used: Literal[False] = False


class ContractFault(Exception):
    def __init__(self, failure: Failure) -> None:
        self.failure = failure
        super().__init__(failure.message)


class DataNeed(Model):
    granularity: Granularity
    fields: frozenset[str]
    symbols: frozenset[str]
    minimum_events_per_symbol: int = Field(default=1, ge=1)


class StrategyDeclaration(Model):
    contract_version: Literal["1.0"] = "1.0"
    strategy_id: str
    kind: StrategyKind
    data: DataNeed
    actions: frozenset[Literal["none", "prediction", "target_position", "submit_order"]]
    training_required: bool
    source: str
    max_abs_position: Decimal | None = Field(default=None, gt=0,
                                              json_schema_extra=_OPTIONAL_POSITIVE_DECIMAL_WIRE)
    max_order_quantity: Decimal | None = Field(default=None, gt=0,
                                               json_schema_extra=_OPTIONAL_POSITIVE_DECIMAL_WIRE)

    @model_validator(mode="after")
    def check_training(self) -> StrategyDeclaration:
        if self.kind != StrategyKind.RULE and not self.training_required:
            raise ValueError("learning strategies must declare training")
        return self


class DatasetDeclaration(Model):
    contract_version: Literal["1.0"] = "1.0"
    dataset_id: str
    granularity: Granularity
    fields: frozenset[str]
    symbols: frozenset[str]
    event_count: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_timezone: str = "UTC"
    session_open: time | None = None
    session_close: time | None = None


class EngineCapability(Model):
    engine_id: str
    granularities: frozenset[Granularity]
    data_fields: frozenset[str]
    actions: frozenset[str]
    execution_fields: dict[Granularity, frozenset[str]] = Field(default_factory=dict)
    max_symbols: int | None = Field(default=None, ge=1)
    financing_modes: frozenset[Literal["cash_only", "unbounded_margin"]] = frozenset({
        "cash_only", "unbounded_margin"})


class EngineProfile(Model):
    engine_id: str
    settings: dict[str, str]


class RunPolicy(Model):
    symbol_map: dict[str, str] = Field(default_factory=dict)
    allow_day_aggregation: bool = False
    aggregation_session_open: time | None = None
    aggregation_session_close: time | None = None
    financing_mode: Literal["cash_only", "unbounded_margin"] = "cash_only"
    sandbox: Literal["development", "strict"] = "development"


class Conversion(Model):
    transformation: Literal["symbol_map", "minute_to_day"]
    reason: str
    policy_switch: Literal["symbol_map", "allow_day_aggregation"]
    can_disable: Literal[True] = True
    source_event_count: int = Field(ge=1)
    output_event_count: int = Field(ge=1)
    affected_symbols: tuple[str, ...]
    input_sha256: str
    output_sha256: str
    parameters: dict[str, str]
    lossy: bool


class ExecutionPlan(Model):
    run_id: str
    strategy_id: str
    dataset_id: str
    engine_id: str
    sandbox: Literal["development", "strict"]
    financing_mode: Literal["cash_only", "unbounded_margin"]
    granularity: Granularity
    symbols: frozenset[str]
    required_fields: frozenset[str]
    allowed_actions: frozenset[str]
    max_abs_position: Decimal | None
    max_order_quantity: Decimal | None
    declaration_sha256: str
    dataset_sha256: str
    effective_events_sha256: str
    training_sha256: str | None
    engine_capability: EngineCapability
    engine_sha256: str
    engine_profile: EngineProfile
    engine_profile_sha256: str
    policy: RunPolicy
    policy_sha256: str
    package_source_sha256: str | None = None
    package_class_name: str | None = None
    conversions: tuple[Conversion, ...] = ()


class MarketEvent(Model):
    event_id: str
    symbol: str
    event_time: datetime
    available_time: datetime
    bar_open_time: datetime | None = None
    values: dict[str, Decimal]

    @model_validator(mode="after")
    def check_time(self) -> MarketEvent:
        if self.event_time.tzinfo is None or self.available_time.tzinfo is None:
            raise ValueError("market times must include a timezone")
        if self.available_time < self.event_time:
            raise ValueError("available_time cannot precede event_time")
        if self.bar_open_time is not None:
            if self.bar_open_time.tzinfo is None:
                raise ValueError("bar_open_time must include a timezone")
            if self.bar_open_time > self.event_time:
                raise ValueError("bar_open_time cannot follow event_time")
        return self


class NoOp(Model):
    kind: Literal["none"] = "none"
    reason: str


class Prediction(Model):
    kind: Literal["prediction"] = "prediction"
    symbol: str
    value: Decimal
    reason: str


class TargetPosition(Model):
    kind: Literal["target_position"] = "target_position"
    symbol: str
    quantity: Decimal
    reason: str


class SubmitOrder(Model):
    kind: Literal["submit_order"] = "submit_order"
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: Decimal = Field(gt=0, json_schema_extra=_POSITIVE_DECIMAL_WIRE)
    limit_price: Decimal | None = Field(default=None, gt=0,
                                        json_schema_extra=_OPTIONAL_POSITIVE_DECIMAL_WIRE)
    reason: str


Action = Annotated[NoOp | Prediction | TargetPosition | SubmitOrder, Field(discriminator="kind")]


class Position(Model):
    symbol: str
    quantity: Decimal
    average_price: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal


class AccountSnapshot(Model):
    timestamp: datetime
    cash: Decimal
    equity: Decimal
    positions: tuple[Position, ...]
    open_order_ids: tuple[str, ...]


class Fill(Model):
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: Decimal = Field(gt=0, json_schema_extra=_POSITIVE_DECIMAL_WIRE)
    price: Decimal = Field(gt=0, json_schema_extra=_POSITIVE_DECIMAL_WIRE)
    fee: Decimal = Field(ge=0, json_schema_extra=_NONNEGATIVE_DECIMAL_WIRE)
    timestamp: datetime


class OrderEvent(Model):
    order_id: str
    symbol: str
    timestamp: datetime
    status: Literal["accepted", "filled", "rejected", "expired"]
    reason: str


class Decision(Model):
    event_id: str
    event_time: datetime
    actions: tuple[Action, ...]


class TrainingSample(Model):
    features: dict[str, Decimal]
    target: Decimal | None = None
    reward: Decimal | None = None
    next_features: dict[str, Decimal] | None = None
    action: int | None = None
    next_action: int | None = None
    terminal: bool = False


class TrainingRequest(Model):
    dataset_id: str
    seed: int
    samples: tuple[TrainingSample, ...]


class Artifact(Model):
    strategy_id: str
    training_sha256: str
    content_sha256: str
    format: str
    relative_path: str


class WorkerReceipt(Model):
    protocol_version: Literal["1.0"] = "1.0"
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    container_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    controls: tuple[str, ...]


class RunReport(Model):
    run_id: str
    status: Literal["succeeded"] = "succeeded"
    plan: ExecutionPlan
    artifact: Artifact | None
    worker_receipt: WorkerReceipt | None = None
    training_worker_receipt: WorkerReceipt | None = None
    source_events_sha256: str
    effective_events_sha256: str
    decisions: tuple[Decision, ...]
    orders: tuple[OrderEvent, ...]
    fills: tuple[Fill, ...]
    accounts: tuple[AccountSnapshot, ...]
    final_equity: Decimal
    logs: tuple[str, ...]

    @model_validator(mode="after")
    def check_worker_receipt(self) -> RunReport:
        if (self.plan.sandbox == "strict") != (self.worker_receipt is not None):
            raise ValueError(
                "strict reports require a worker receipt; development reports forbid it"
            )
        if (self.plan.sandbox == "strict" and self.artifact is not None) != (
            self.training_worker_receipt is not None
        ):
            raise ValueError("strict trained reports require both worker receipts")
        return self


class RunBundle(Model):
    report: RunReport
    strategy_declaration: StrategyDeclaration
    dataset_declaration: DatasetDeclaration
    source_events: tuple[MarketEvent, ...]
    effective_events: tuple[MarketEvent, ...]
    training_request: TrainingRequest | None
    model_base64: str | None
    package_source: str | None
