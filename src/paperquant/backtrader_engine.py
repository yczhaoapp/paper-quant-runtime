"""Narrow native Backtrader bridge for causal, single-symbol OHLCV bars."""

from __future__ import annotations

from datetime import UTC
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from typing import Any, NoReturn

from pydantic import TypeAdapter, ValidationError

from paperquant.compiler import fingerprint
from paperquant.engine import DecisionStrategy
from paperquant.models import (
    AccountSnapshot,
    Action,
    ContractFault,
    Decision,
    EngineCapability,
    EngineProfile,
    ErrorCode,
    ExecutionPlan,
    Failure,
    Fill,
    Granularity,
    MarketEvent,
    NoOp,
    OrderEvent,
    Position,
    Prediction,
    Stage,
    SubmitOrder,
    TargetPosition,
)

_ACTIONS = TypeAdapter(tuple[Action, ...])
_BAR_FIELDS = frozenset({"open", "high", "low", "close", "volume"})


def _fail(plan: ExecutionPlan, stage: Stage, code: ErrorCode, message: str,
          **details: str) -> NoReturn:
    raise ContractFault(Failure(run_id=plan.run_id, stage=stage, code=code,
                                message=message, details=details))


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


class BacktraderEngine:
    """Run the native Backtrader broker and translate its completed orders and account.

    This adapter deliberately supports one symbol, cash-only financing, market
    orders, and one next-bar execution opportunity. Unsupported modes fail before
    any strategy inference instead of being translated to another matching rule.
    """

    def __init__(self, *, initial_cash: Decimal = Decimal("100000"),
                 fee_rate: Decimal = Decimal("0")) -> None:
        if initial_cash <= 0 or not Decimal("0") <= fee_rate < Decimal("1"):
            raise ValueError("initial cash must be positive and fee rate between zero and one")
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate

    def capability(self, fields: frozenset[str]) -> EngineCapability:
        return EngineCapability(
            engine_id="paperquant.backtrader",
            granularities=frozenset({Granularity.MINUTE, Granularity.DAY}),
            data_fields=fields,
            actions=frozenset({"none", "prediction", "target_position", "submit_order"}),
            max_symbols=1,
            financing_modes=frozenset({"cash_only"}),
            execution_fields={
                Granularity.MINUTE: _BAR_FIELDS,
                Granularity.DAY: _BAR_FIELDS,
            },
        )

    def profile(self) -> EngineProfile:
        try:
            release = version("backtrader")
        except PackageNotFoundError as exc:
            raise RuntimeError("Backtrader optional dependency is unavailable") from exc
        return EngineProfile(engine_id="paperquant.backtrader", settings={
            "backtrader_version": release,
            "initial_cash": str(self.initial_cash),
            "fee_rate": str(self.fee_rate),
            "matching": "native-broker-next-bar-open-market",
            "accounting": "native-broker-cash-equity-position-and-fill-ledger",
            "limit_orders": "unsupported",
            "partial_fills": "unsupported",
            "financing": "cash_only",
            "slippage": "none",
        })

    def run(self, *, plan: ExecutionPlan, strategy: DecisionStrategy,
            events: tuple[MarketEvent, ...]) -> tuple[
                tuple[Decision, ...], tuple[OrderEvent, ...], tuple[Fill, ...],
                tuple[AccountSnapshot, ...],
            ]:
        if (plan.engine_id != "paperquant.backtrader"
            or fingerprint(plan.engine_capability) != plan.engine_sha256
            or fingerprint(self.capability(plan.engine_capability.data_fields))
               != plan.engine_sha256
            or fingerprint(plan.engine_profile) != plan.engine_profile_sha256
            or fingerprint(self.profile()) != plan.engine_profile_sha256
            or fingerprint(plan.policy) != plan.policy_sha256
            or fingerprint(strategy.declaration) != plan.declaration_sha256
            or fingerprint(events) != plan.effective_events_sha256):
            _fail(plan, Stage.BACKTEST, ErrorCode.INPUT_INVALID,
                  "Native engine context differs from the execution plan")
        if (plan.granularity not in {Granularity.MINUTE, Granularity.DAY}
            or len(plan.symbols) != 1 or plan.financing_mode != "cash_only"):
            _fail(plan, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED,
                  "Native engine supports one bar symbol and cash-only financing")
        if not events or any(event.symbol not in plan.symbols for event in events):
            _fail(plan, Stage.INPUT, ErrorCode.INPUT_INVALID,
                  "Native engine received an unknown or empty event stream")
        for event in events:
            values = event.values
            if event.bar_open_time is None or not _BAR_FIELDS <= values.keys():
                _fail(plan, Stage.INPUT, ErrorCode.DATA_FIELD_MISSING,
                      "Native engine requires complete OHLCV bars")
            if (min(values[key] for key in ("open", "high", "low", "close")) <= 0
                or values["high"] < max(values["open"], values["close"])
                or values["low"] > min(values["open"], values["close"])
                or values["volume"] < 0):
                _fail(plan, Stage.INPUT, ErrorCode.INPUT_INVALID,
                      "Native engine received an inconsistent OHLCV bar")

        try:
            import backtrader as bt  # type: ignore[import-untyped]
        except ImportError:
            _fail(plan, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED,
                  "Backtrader optional dependency is unavailable")

        decisions: list[Decision] = []
        orders: list[OrderEvent] = []
        fills: list[Fill] = []
        accounts: list[AccountSnapshot] = []
        known_ids: set[str] = set()
        pending: dict[int, str] = {}
        realized = Decimal("0")
        prior_quantity = Decimal("0")
        prior_average = Decimal("0")

        class EventFeed(bt.feeds.DataBase):  # type: ignore[misc]
            def start(self) -> None:
                super().start()
                self._source = iter(events)

            def _load(self) -> bool:
                try:
                    event = next(self._source)
                except StopIteration:
                    return False
                moment = event.event_time.astimezone(UTC).replace(tzinfo=None)
                self.lines.datetime[0] = bt.date2num(moment)
                for field in _BAR_FIELDS:
                    getattr(self.lines, field)[0] = float(event.values[field])
                self.lines.openinterest[0] = 0
                return True

        class Bridge(bt.Strategy):  # type: ignore[misc]
            def __init__(self) -> None:
                self.index = 0

            def notify_order(self, native: Any) -> None:
                nonlocal realized, prior_quantity, prior_average
                if native.status in (native.Submitted, native.Accepted):
                    return
                order_id = pending.pop(native.ref, None)
                if order_id is None:
                    _fail(plan, Stage.BACKTEST, ErrorCode.BACKTEST_FAILED,
                          "Native broker reported an untracked order")
                if native.status != native.Completed:
                    if native.status in (native.Margin, native.Rejected):
                        _fail(plan, Stage.BACKTEST, ErrorCode.ORDER_REJECTED,
                              "Native broker rejected an order", order_id=order_id,
                              native_status=native.getstatusname())
                    _fail(plan, Stage.BACKTEST, ErrorCode.ENGINE_UNSUPPORTED,
                          "Native broker returned an unsupported order state",
                          order_id=order_id, native_status=native.getstatusname())
                if self.index >= len(events):
                    _fail(plan, Stage.BACKTEST, ErrorCode.BACKTEST_FAILED,
                          "Native fill has no matching market bar", order_id=order_id)
                event = events[self.index]
                fill_time = event.bar_open_time
                assert fill_time is not None
                signed = _decimal(native.executed.size)
                price = _decimal(native.executed.price)
                fee = _decimal(native.executed.comm)
                if signed == 0 or price <= 0 or fee < 0:
                    _fail(plan, Stage.BACKTEST, ErrorCode.BACKTEST_FAILED,
                          "Native broker returned an invalid fill", order_id=order_id)
                if prior_quantity * signed < 0:
                    closed = min(abs(prior_quantity), abs(signed))
                    realized += closed * (price - prior_average) * (
                        Decimal("1") if prior_quantity > 0 else Decimal("-1"))
                native_position = self.getposition(self.data)
                prior_quantity = _decimal(native_position.size)
                prior_average = _decimal(native_position.price)
                fills.append(Fill(order_id=order_id, symbol=event.symbol,
                                  side="buy" if signed > 0 else "sell",
                                  quantity=abs(signed), price=price, fee=fee,
                                  timestamp=fill_time))
                orders.append(OrderEvent(order_id=order_id, symbol=event.symbol,
                                         timestamp=fill_time, status="filled",
                                         reason="native_backtrader_completed"))

            def next(self) -> None:
                event = events[self.index]
                native_position = self.getposition(self.data)
                quantity = _decimal(native_position.size)
                average = _decimal(native_position.price)
                if quantity != prior_quantity or average != prior_average:
                    _fail(plan, Stage.BACKTEST, ErrorCode.BACKTEST_FAILED,
                          "Native position differs from its executed-fill ledger")
                positions: tuple[Position, ...] = (
                    Position(symbol=event.symbol, quantity=quantity,
                             average_price=average, realized_pnl=realized,
                             unrealized_pnl=quantity * (event.values["close"] - average)),
                )
                if quantity == 0 and realized == 0:
                    positions = ()
                account = AccountSnapshot(
                    timestamp=event.available_time,
                    cash=_decimal(self.broker.getcash()),
                    equity=_decimal(self.broker.getvalue()),
                    positions=positions,
                    open_order_ids=tuple(pending.values()),
                )
                accounts.append(account)
                try:
                    returned = strategy.decide(
                        MarketEvent.model_validate(event.model_dump(mode="python")),
                        AccountSnapshot.model_validate(account.model_dump(mode="python")),
                    )
                except ContractFault:
                    raise
                except Exception as exc:
                    _fail(plan, Stage.INFER, ErrorCode.BACKTEST_FAILED,
                          "Strategy inference failed in native engine",
                          cause=type(exc).__name__, reason=str(exc)[:200],
                          event_id=event.event_id)
                if type(returned) is not tuple:
                    _fail(plan, Stage.INFER, ErrorCode.ACTION_INVALID,
                          "Actions must be a tuple")
                try:
                    actions = _ACTIONS.validate_python(
                        tuple(item.model_dump(mode="python") for item in returned))
                except (AttributeError, TypeError, ValueError, ValidationError):
                    _fail(plan, Stage.INFER, ErrorCode.ACTION_INVALID,
                          "Strategy actions do not satisfy the wire schema")
                traded = False
                for action in actions:
                    if action.kind not in plan.allowed_actions:
                        _fail(plan, Stage.INFER, ErrorCode.ACTION_INVALID,
                              "Action was not declared")
                    if isinstance(action, (Prediction, TargetPosition, SubmitOrder)):
                        if action.symbol != event.symbol:
                            _fail(plan, Stage.INFER, ErrorCode.ACTION_INVALID,
                                  "Native engine cannot execute another symbol")
                    if isinstance(action, (NoOp, Prediction)):
                        continue
                    if traded:
                        _fail(plan, Stage.INFER, ErrorCode.ENGINE_UNSUPPORTED,
                              "Native adapter supports one trade action per bar")
                    traded = True
                    if isinstance(action, TargetPosition):
                        if (plan.max_abs_position is not None
                            and abs(action.quantity) > plan.max_abs_position):
                            _fail(plan, Stage.INFER, ErrorCode.ORDER_REJECTED,
                                  "Target exceeds position limit")
                        signed = action.quantity - quantity
                        order_id = f"target:{len(decisions)}:{event.symbol}"
                    elif isinstance(action, SubmitOrder):
                        if action.limit_price is not None:
                            _fail(plan, Stage.INFER, ErrorCode.ENGINE_UNSUPPORTED,
                                  "Native adapter does not support limit orders")
                        if action.client_order_id in known_ids:
                            _fail(plan, Stage.INFER, ErrorCode.ORDER_REJECTED,
                                  "Duplicate order ID")
                        known_ids.add(action.client_order_id)
                        signed = action.quantity if action.side == "buy" else -action.quantity
                        order_id = action.client_order_id
                    else:
                        _fail(plan, Stage.INFER, ErrorCode.ACTION_INVALID,
                              "Unknown action")
                    if signed == 0:
                        continue
                    if (plan.max_order_quantity is not None
                        and abs(signed) > plan.max_order_quantity):
                        _fail(plan, Stage.INFER, ErrorCode.ORDER_REJECTED,
                              "Order exceeds quantity limit", order_id=order_id)
                    if (plan.max_abs_position is not None
                        and abs(quantity + signed) > plan.max_abs_position):
                        _fail(plan, Stage.INFER, ErrorCode.ORDER_REJECTED,
                              "Order exceeds position limit", order_id=order_id)
                    native = (self.buy if signed > 0 else self.sell)(size=float(abs(signed)))
                    if native is None:
                        _fail(plan, Stage.BACKTEST, ErrorCode.ORDER_REJECTED,
                              "Native broker did not accept the order", order_id=order_id)
                    pending[native.ref] = order_id
                    orders.append(OrderEvent(order_id=order_id, symbol=event.symbol,
                                             timestamp=event.available_time, status="accepted",
                                             reason=action.reason))
                decisions.append(Decision(event_id=event.event_id,
                                          event_time=event.available_time, actions=actions))
                self.index += 1

        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(EventFeed())
        cerebro.addstrategy(Bridge)
        cerebro.broker.setcash(float(self.initial_cash))
        cerebro.broker.setcommission(commission=float(self.fee_rate))
        cerebro.run(runonce=False, preload=False)
        if len(decisions) != len(events):
            _fail(plan, Stage.BACKTEST, ErrorCode.BACKTEST_FAILED,
                  "Native engine did not visit every input event")
        for order_id in pending.values():
            orders.append(OrderEvent(order_id=order_id, symbol=events[-1].symbol,
                                     timestamp=events[-1].available_time,
                                     status="expired", reason="no_next_event"))
        return tuple(decisions), tuple(orders), tuple(fills), tuple(accounts)
