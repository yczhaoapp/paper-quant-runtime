from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, NoReturn, Protocol

from paperquant.compiler import fingerprint
from paperquant.models import (
    AccountSnapshot,
    Action,
    ContractFault,
    Decision,
    EngineCapability,
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
    StrategyDeclaration,
    SubmitOrder,
    TargetPosition,
)


@dataclass(frozen=True)
class _Holding:
    quantity: Decimal = Decimal("0")
    average: Decimal = Decimal("0")
    realized: Decimal = Decimal("0")

    def traded(self, signed_quantity: Decimal, price: Decimal) -> _Holding:
        before = self.quantity
        after = before + signed_quantity
        if before == 0 or before * signed_quantity > 0:
            average = (abs(before) * self.average + abs(signed_quantity) * price) / abs(after)
            return _Holding(after, average, self.realized)
        closed = min(abs(before), abs(signed_quantity))
        direction = Decimal("1") if before > 0 else Decimal("-1")
        gain = closed * (price - self.average) * direction
        average = Decimal("0") if after == 0 else price if before * after < 0 else self.average
        return _Holding(after, average, self.realized + gain)


@dataclass(frozen=True)
class _Intent:
    order_id: str
    symbol: str
    target: Decimal | None
    signed_quantity: Decimal | None
    limit_price: Decimal | None = None


class DecisionStrategy(Protocol):
    declaration: StrategyDeclaration

    def decide(self, event: MarketEvent, account: AccountSnapshot) -> tuple[Action, ...]: ...


class BacktestEngine(Protocol):
    """Trusted engine boundary; strategy training remains with the runtime."""

    def capability(self, fields: frozenset[str]) -> EngineCapability: ...

    def run(
        self, *, plan: ExecutionPlan, strategy: DecisionStrategy,
        events: tuple[MarketEvent, ...],
    ) -> tuple[
        tuple[Decision, ...], tuple[OrderEvent, ...], tuple[Fill, ...], tuple[AccountSnapshot, ...]
    ]: ...


def reference_capability(fields: frozenset[str]) -> EngineCapability:
    return EngineCapability(
        engine_id="paperquant.reference",
        granularities=frozenset(Granularity),
        data_fields=fields,
        actions=frozenset({"none", "prediction", "target_position", "submit_order"}),
    )


class ReferenceEngine:
    """One-event-latency fill engine with one canonical account and no silent orders."""

    def __init__(
        self, *, initial_cash: Decimal = Decimal("100000"), fee_rate: Decimal = Decimal("0")
    ) -> None:
        if initial_cash <= 0 or fee_rate < 0:
            raise ValueError("initial cash must be positive and fees nonnegative")
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate

    def capability(self, fields: frozenset[str]) -> EngineCapability:
        return reference_capability(fields)

    @staticmethod
    def _fail(
        plan: ExecutionPlan, stage: Stage, code: ErrorCode, message: str, **details: str
    ) -> NoReturn:
        raise ContractFault(
            Failure(run_id=plan.run_id, stage=stage, code=code, message=message, details=details)
        )

    @staticmethod
    def _price(event: MarketEvent, *, fill: bool, granularity: Granularity) -> Decimal:
        field = "price" if granularity == Granularity.TICK else "open" if fill else "close"
        price = event.values[field]
        if price <= 0:
            raise ValueError(f"{field} must be positive")
        return price

    def run(
        self,
        *,
        plan: ExecutionPlan,
        strategy: DecisionStrategy,
        events: tuple[MarketEvent, ...],
    ) -> tuple[
        tuple[Decision, ...], tuple[OrderEvent, ...], tuple[Fill, ...], tuple[AccountSnapshot, ...]
    ]:
        if (
            strategy.declaration.strategy_id != plan.strategy_id
            or fingerprint(strategy.declaration) != plan.declaration_sha256
        ):
            self._fail(
                plan,
                Stage.BACKTEST,
                ErrorCode.INPUT_INVALID,
                "Strategy declaration differs from plan",
            )
        if (
            plan.engine_id != "paperquant.reference"
            or plan.engine_capability.engine_id != plan.engine_id
            or fingerprint(plan.engine_capability) != plan.engine_sha256
            or fingerprint(self.capability(plan.engine_capability.data_fields))
            != plan.engine_sha256
        ):
            self._fail(plan, Stage.BACKTEST, ErrorCode.ENGINE_UNSUPPORTED,
                       "Plan capability is not bound to the selected engine")
        if fingerprint(events) != plan.effective_events_sha256:
            self._fail(
                plan, Stage.INPUT, ErrorCode.INPUT_INVALID, "Effective events differ from plan"
            )
        cash = self.initial_cash
        holdings: dict[str, _Holding] = {}
        marks: dict[str, Decimal] = {}
        pending: list[_Intent] = []
        decisions: list[Decision] = []
        orders: list[OrderEvent] = []
        fills: list[Fill] = []
        snapshots: list[AccountSnapshot] = []
        known_order_ids: set[str] = set()

        for event in events:
            if event.symbol not in plan.symbols or not plan.required_fields <= event.values.keys():
                self._fail(
                    plan, Stage.INPUT, ErrorCode.INPUT_INVALID, "Effective event breaks the plan"
                )
            try:
                mark = self._price(event, fill=False, granularity=plan.granularity)
                fill_price = self._price(event, fill=True, granularity=plan.granularity)
            except (KeyError, ValueError) as exc:
                self._fail(plan, Stage.INPUT, ErrorCode.INPUT_INVALID, str(exc))
            if plan.granularity != Granularity.TICK and event.bar_open_time is None:
                self._fail(plan, Stage.INPUT, ErrorCode.INPUT_INVALID, "Bar open time is required")
            fill_time = event.bar_open_time or event.available_time
            marks[event.symbol] = mark

            remaining: list[_Intent] = []
            for intent in pending:
                if intent.symbol != event.symbol:
                    remaining.append(intent)
                    continue
                current = holdings.get(intent.symbol, _Holding())
                signed = intent.signed_quantity
                if intent.target is not None:
                    signed = intent.target - current.quantity
                assert signed is not None
                if signed == 0:
                    orders.append(
                        OrderEvent(
                            order_id=intent.order_id,
                            symbol=intent.symbol,
                            timestamp=fill_time,
                            status="expired",
                            reason="target_already_satisfied",
                        )
                    )
                    continue
                maximum = plan.max_order_quantity
                if maximum is not None and abs(signed) > maximum:
                    self._fail(
                        plan, Stage.BACKTEST, ErrorCode.ORDER_REJECTED,
                        "Fill exceeds per-order quantity limit", order_id=intent.order_id,
                    )
                execution_price = fill_price
                fill_reason = "next_event_market_fill"
                if intent.limit_price is not None:
                    if plan.granularity == Granularity.TICK:
                        try:
                            bid = event.values["bid_price"]
                            ask = event.values["ask_price"]
                        except KeyError:
                            self._fail(
                                plan,
                                Stage.INPUT,
                                ErrorCode.DATA_FIELD_MISSING,
                                "Limit order requires next-event L1 quotes",
                            )
                        if bid <= 0 or ask <= bid:
                            self._fail(
                                plan,
                                Stage.INPUT,
                                ErrorCode.INPUT_INVALID,
                                "Next-event L1 quotes are invalid",
                            )
                        touch = ask if signed > 0 else bid
                    else:
                        touch = fill_price
                    marketable = (
                        intent.limit_price >= touch
                        if signed > 0
                        else intent.limit_price <= touch
                    )
                    if not marketable:
                        orders.append(
                            OrderEvent(
                                order_id=intent.order_id,
                                symbol=intent.symbol,
                                timestamp=fill_time,
                                status="expired",
                                reason="next_event_limit_not_marketable",
                            )
                        )
                        continue
                    execution_price = touch
                    fill_reason = "next_event_marketable_limit_fill"
                limit = plan.max_abs_position
                if limit is not None and abs(current.quantity + signed) > limit:
                    self._fail(
                        plan,
                        Stage.BACKTEST,
                        ErrorCode.ORDER_REJECTED,
                        "Fill exceeds position limit",
                        order_id=intent.order_id,
                    )
                fee = abs(signed) * execution_price * self.fee_rate
                cash -= signed * execution_price + fee
                holdings[intent.symbol] = current.traded(signed, execution_price)
                side: Literal["buy", "sell"] = "buy" if signed > 0 else "sell"
                fills.append(
                    Fill(
                        order_id=intent.order_id,
                        symbol=intent.symbol,
                        side=side,
                        quantity=abs(signed),
                        price=execution_price,
                        fee=fee,
                        timestamp=fill_time,
                    )
                )
                orders.append(
                    OrderEvent(
                        order_id=intent.order_id,
                        symbol=intent.symbol,
                        timestamp=fill_time,
                        status="filled",
                        reason=fill_reason,
                    )
                )
            pending = remaining

            positions = tuple(
                Position(
                    symbol=symbol,
                    quantity=holding.quantity,
                    average_price=holding.average,
                    realized_pnl=holding.realized,
                    unrealized_pnl=holding.quantity * (marks[symbol] - holding.average),
                )
                for symbol, holding in sorted(holdings.items())
            )
            equity = cash + sum(
                (holding.quantity * marks[symbol] for symbol, holding in holdings.items()),
                Decimal("0"),
            )
            snapshot = AccountSnapshot(
                timestamp=event.available_time,
                cash=cash,
                equity=equity,
                positions=positions,
                open_order_ids=tuple(intent.order_id for intent in pending),
            )
            snapshots.append(snapshot)
            try:
                actions = strategy.decide(event, snapshot)
            except ContractFault:
                raise
            except Exception as exc:
                self._fail(
                    plan,
                    Stage.INFER,
                    ErrorCode.BACKTEST_FAILED,
                    "Strategy inference failed",
                    cause=type(exc).__name__,
                )
            if type(actions) is not tuple:
                self._fail(plan, Stage.INFER, ErrorCode.ACTION_INVALID, "Actions must be a tuple")
            target_symbols: set[str] = set()
            for action in actions:
                if not isinstance(action, (NoOp, Prediction, TargetPosition, SubmitOrder)):
                    self._fail(plan, Stage.INFER, ErrorCode.ACTION_INVALID, "Unknown action object")
                if action.kind not in plan.allowed_actions:
                    self._fail(
                        plan, Stage.INFER, ErrorCode.ACTION_INVALID, "Action was not declared"
                    )
                if isinstance(action, (Prediction, TargetPosition, SubmitOrder)):
                    if action.symbol not in plan.symbols:
                        self._fail(
                            plan, Stage.INFER, ErrorCode.ACTION_INVALID, "Unknown action symbol"
                        )
                if isinstance(action, TargetPosition):
                    if action.symbol in target_symbols:
                        self._fail(
                            plan, Stage.INFER, ErrorCode.ACTION_INVALID, "Duplicate target intent"
                        )
                    target_symbols.add(action.symbol)
                    limit = plan.max_abs_position
                    if limit is not None and abs(action.quantity) > limit:
                        self._fail(
                            plan, Stage.INFER, ErrorCode.ORDER_REJECTED, "Target exceeds limit"
                        )
                    order_id = f"target:{len(decisions)}:{action.symbol}"
                    pending.append(_Intent(order_id, action.symbol, action.quantity, None))
                elif isinstance(action, SubmitOrder):
                    if action.client_order_id in known_order_ids:
                        self._fail(
                            plan, Stage.INFER, ErrorCode.ORDER_REJECTED, "Duplicate order ID"
                        )
                    known_order_ids.add(action.client_order_id)
                    maximum = plan.max_order_quantity
                    if maximum is not None and action.quantity > maximum:
                        self._fail(
                            plan, Stage.INFER, ErrorCode.ORDER_REJECTED, "Order exceeds limit"
                        )
                    signed = action.quantity if action.side == "buy" else -action.quantity
                    pending.append(
                        _Intent(
                            action.client_order_id,
                            action.symbol,
                            None,
                            signed,
                            action.limit_price,
                        )
                    )
                else:
                    continue
                orders.append(
                    OrderEvent(
                        order_id=order_id
                        if isinstance(action, TargetPosition)
                        else action.client_order_id,
                        symbol=action.symbol,
                        timestamp=event.available_time,
                        status="accepted",
                        reason=action.reason,
                    )
                )
            decisions.append(
                Decision(event_id=event.event_id, event_time=event.available_time, actions=actions)
            )

        for intent in pending:
            orders.append(
                OrderEvent(
                    order_id=intent.order_id,
                    symbol=intent.symbol,
                    timestamp=events[-1].available_time,
                    status="expired",
                    reason="no_next_event",
                )
            )
        return tuple(decisions), tuple(orders), tuple(fills), tuple(snapshots)
