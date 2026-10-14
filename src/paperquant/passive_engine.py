"""Bounded complete-fill passive quote simulation with explicit cancellation.

Adverse fills cross an existing quote; nonadverse fills require independently
supplied order-flow and uniform draws. This is a declared simulator, not a native
exchange, queue-priority model or endogenous full limit-order book.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, NoReturn

from pydantic import TypeAdapter, ValidationError

from paperquant.compiler import fingerprint, validate_plan_projection
from paperquant.engine import DecisionStrategy, _Holding
from paperquant.models import (
    AccountSnapshot,
    Action,
    CancelOrder,
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
    Stage,
    SubmitOrder,
)

_ACTIONS = TypeAdapter(tuple[Action, ...])
FIELDS = frozenset(
    {"bid_price", "ask_price", "buy_flow", "sell_flow", "uniform_buy", "uniform_sell"}
)


@dataclass(frozen=True)
class RestingQuote:
    identifier: str
    symbol: str
    side: Literal["buy", "sell"]
    price: Decimal


class PassiveQuoteEngine:
    def __init__(
        self,
        *,
        initial_cash: Decimal = Decimal("100000"),
        fee_rate: Decimal = Decimal("0"),
        nonadverse_probability: Decimal = Decimal("0.2"),
    ) -> None:
        if (
            not initial_cash.is_finite()
            or initial_cash <= 0
            or not fee_rate.is_finite()
            or fee_rate < 0
            or not nonadverse_probability.is_finite()
            or not 0 <= nonadverse_probability <= 1
        ):
            raise ValueError("Invalid passive simulator configuration")
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.nonadverse_probability = nonadverse_probability

    def capability(self, fields: frozenset[str]) -> EngineCapability:
        return EngineCapability(
            engine_id="paperquant.passive_quote",
            granularities=frozenset({Granularity.TICK}),
            data_fields=fields,
            actions=frozenset({"none", "submit_order", "cancel_order"}),
            execution_fields={Granularity.TICK: FIELDS},
            max_symbols=1,
        )

    def profile(self) -> EngineProfile:
        return EngineProfile(
            engine_id="paperquant.passive_quote",
            settings={
                "initial_cash": str(self.initial_cash),
                "fee_rate": str(self.fee_rate),
                "nonadverse_probability": str(self.nonadverse_probability),
                "matching": "resting-quote-adverse-cross-or-explicit-order-flow",
                "order_lifetime": "until-fill-cancel-or-end-of-stream",
                "quantity": "unit-complete-fill-only",
                "queue_priority": "not_modelled",
                "mark": "current-bid-ask-midpoint",
                "inference": "after-current-event-fills",
            },
        )

    @staticmethod
    def _fail(plan: ExecutionPlan, stage: Stage, code: ErrorCode, message: str) -> NoReturn:
        raise ContractFault(Failure(run_id=plan.run_id, stage=stage, code=code, message=message))

    def run(
        self, *, plan: ExecutionPlan, strategy: DecisionStrategy, events: tuple[MarketEvent, ...]
    ) -> tuple[
        tuple[Decision, ...], tuple[OrderEvent, ...], tuple[Fill, ...], tuple[AccountSnapshot, ...]
    ]:
        validate_plan_projection(plan, strategy.declaration)
        if (
            fingerprint(strategy.declaration) != plan.declaration_sha256
            or strategy.declaration.strategy_id != plan.strategy_id
            or fingerprint(events) != plan.effective_events_sha256
            or fingerprint(plan.policy) != plan.policy_sha256
            or plan.financing_mode != plan.policy.financing_mode
        ):
            self._fail(
                plan, Stage.INPUT, ErrorCode.INPUT_INVALID, "Passive execution context mismatch"
            )
        if (
            plan.engine_id != "paperquant.passive_quote"
            or fingerprint(self.capability(plan.engine_capability.data_fields))
            != plan.engine_sha256
            or fingerprint(plan.engine_capability) != plan.engine_sha256
            or fingerprint(self.profile()) != plan.engine_profile_sha256
            or fingerprint(plan.engine_profile) != plan.engine_profile_sha256
        ):
            self._fail(
                plan, Stage.COMPILE, ErrorCode.ENGINE_UNSUPPORTED, "Passive engine binding mismatch"
            )
        pending: dict[str, RestingQuote] = {}
        used: set[str] = set()
        cash, holding = self.initial_cash, _Holding()
        decisions: list[Decision] = []
        orders: list[OrderEvent] = []
        fills: list[Fill] = []
        accounts: list[AccountSnapshot] = []
        for event in events:
            values = event.values
            if (
                event.symbol not in plan.symbols
                or not plan.required_fields <= values.keys()
                or not FIELDS <= values.keys()
            ):
                self._fail(
                    plan, Stage.INPUT, ErrorCode.DATA_FIELD_MISSING, "Passive event fields missing"
                )
            bid, ask = values["bid_price"], values["ask_price"]
            if (
                bid <= 0
                or ask <= bid
                or any(not 0 <= values[k] < 1 for k in ("uniform_buy", "uniform_sell"))
                or values["buy_flow"] < 0
                or values["sell_flow"] < 0
            ):
                self._fail(
                    plan, Stage.INPUT, ErrorCode.INPUT_INVALID, "Invalid quote or flow event"
                )
            for identifier, quote in tuple(pending.items()):
                adverse = bid < quote.price if quote.side == "buy" else ask > quote.price
                flow = values["sell_flow"] if quote.side == "buy" else values["buy_flow"]
                draw = values["uniform_sell"] if quote.side == "buy" else values["uniform_buy"]
                nonadverse = flow > 0 and draw < self.nonadverse_probability
                if not (adverse or nonadverse):
                    continue
                signed = Decimal(1) if quote.side == "buy" else Decimal(-1)
                fee = quote.price * self.fee_rate
                next_cash = cash - signed * quote.price - fee
                if (
                    plan.max_abs_position is not None
                    and abs(holding.quantity + signed) > plan.max_abs_position
                ) or (plan.financing_mode == "cash_only" and next_cash < 0):
                    self._fail(
                        plan,
                        Stage.BACKTEST,
                        ErrorCode.ORDER_REJECTED,
                        "Passive fill exceeds risk or cash",
                    )
                cash, holding = next_cash, holding.traded(signed, quote.price)
                fills.append(
                    Fill(
                        order_id=identifier,
                        symbol=quote.symbol,
                        side=quote.side,
                        quantity=Decimal(1),
                        price=quote.price,
                        fee=fee,
                        timestamp=event.available_time,
                    )
                )
                orders.append(
                    OrderEvent(
                        order_id=identifier,
                        symbol=quote.symbol,
                        timestamp=event.available_time,
                        status="filled",
                        reason="adverse_quote_cross" if adverse else "nonadverse_order_flow",
                    )
                )
                del pending[identifier]
            mid = (bid + ask) / 2
            positions = (
                ()
                if holding.quantity == 0 and holding.realized == 0
                else (
                    Position(
                        symbol=event.symbol,
                        quantity=holding.quantity,
                        average_price=holding.average,
                        realized_pnl=holding.realized,
                        unrealized_pnl=holding.quantity * (mid - holding.average),
                    ),
                )
            )
            snapshot = AccountSnapshot(
                timestamp=event.available_time,
                cash=cash,
                equity=cash + holding.quantity * mid,
                positions=positions,
                open_order_ids=tuple(pending),
            )
            accounts.append(snapshot)
            try:
                raw = strategy.decide(
                    MarketEvent.model_validate(event.model_dump(mode="python")),
                    AccountSnapshot.model_validate(snapshot.model_dump(mode="python")),
                )
                if type(raw) is not tuple:
                    raise ValueError("Actions must be a tuple")
                actions = _ACTIONS.validate_python(tuple(a.model_dump(mode="python") for a in raw))
            except (ValueError, AttributeError, TypeError, ValidationError) as exc:
                raise ContractFault(
                    Failure(
                        run_id=plan.run_id,
                        stage=Stage.INFER,
                        code=ErrorCode.ACTION_INVALID,
                        message="Invalid passive strategy action",
                        details={"cause": type(exc).__name__},
                    )
                ) from exc
            decisions.append(
                Decision(event_id=event.event_id, event_time=event.available_time, actions=actions)
            )
            for action in actions:
                if action.kind not in plan.allowed_actions:
                    self._fail(
                        plan, Stage.INFER, ErrorCode.ACTION_INVALID, "Undeclared passive action"
                    )
                if isinstance(action, NoOp):
                    continue
                if action.symbol != event.symbol:
                    self._fail(
                        plan, Stage.INFER, ErrorCode.ORDER_REJECTED, "Passive order symbol mismatch"
                    )
                if isinstance(action, CancelOrder):
                    if action.client_order_id not in pending:
                        self._fail(
                            plan,
                            Stage.INFER,
                            ErrorCode.ORDER_REJECTED,
                            "Cancellation names no live order",
                        )
                    del pending[action.client_order_id]
                    orders.append(
                        OrderEvent(
                            order_id=action.client_order_id,
                            symbol=event.symbol,
                            timestamp=event.available_time,
                            status="cancelled",
                            reason=action.reason,
                        )
                    )
                elif isinstance(action, SubmitOrder):
                    if (
                        action.quantity != 1
                        or action.limit_price is None
                        or action.client_order_id in used
                        or (
                            plan.max_order_quantity is not None
                            and action.quantity > plan.max_order_quantity
                        )
                        or any(q.side == action.side for q in pending.values())
                    ):
                        self._fail(
                            plan,
                            Stage.INFER,
                            ErrorCode.ORDER_REJECTED,
                            "Unsupported or conflicting passive order",
                        )
                    signed = Decimal(1) if action.side == "buy" else Decimal(-1)
                    if (
                        plan.max_abs_position is not None
                        and abs(holding.quantity + signed) > plan.max_abs_position
                    ):
                        self._fail(
                            plan,
                            Stage.INFER,
                            ErrorCode.ORDER_REJECTED,
                            "Posted order exceeds inventory bound",
                        )
                    touch = bid if action.side == "buy" else ask
                    if action.limit_price != touch:
                        self._fail(
                            plan,
                            Stage.INFER,
                            ErrorCode.ENGINE_UNSUPPORTED,
                            "Only current top-of-book posting is supported",
                        )
                    used.add(action.client_order_id)
                    pending[action.client_order_id] = RestingQuote(
                        action.client_order_id, event.symbol, action.side, action.limit_price
                    )
                    orders.append(
                        OrderEvent(
                            order_id=action.client_order_id,
                            symbol=event.symbol,
                            timestamp=event.available_time,
                            status="accepted",
                            reason="resting_top_of_book",
                        )
                    )
                else:
                    self._fail(
                        plan,
                        Stage.INFER,
                        ErrorCode.ACTION_INVALID,
                        "Passive engine cannot execute this action",
                    )
        for quote in pending.values():
            orders.append(
                OrderEvent(
                    order_id=quote.identifier,
                    symbol=quote.symbol,
                    timestamp=events[-1].available_time,
                    status="expired",
                    reason="end_of_stream",
                )
            )
        return tuple(decisions), tuple(orders), tuple(fills), tuple(accounts)
