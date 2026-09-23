from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    SubmitOrder,
)


class TimeSlicedExecution:
    """Submit a fixed parent quantity in equal event-time child orders."""

    declaration = StrategyDeclaration(
        strategy_id="rule.time_sliced_execution",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "close"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=13,
        ),
        actions=frozenset({"submit_order", "none"}),
        training_required=False,
        source="https://arxiv.org/pdf/2208.06244v1",
        max_abs_position=Decimal("12"),
        max_order_quantity=Decimal("12"),
    )

    def __init__(self, *, parent_quantity=Decimal("12"), slices=12, side="buy"):
        parent_quantity = Decimal(parent_quantity)
        if parent_quantity <= 0 or slices < 1 or side not in ("buy", "sell"):
            raise ValueError("invalid execution schedule")
        self.parent_quantity = parent_quantity
        self.slices = slices
        self.side = side
        self.child_quantity = parent_quantity / Decimal(slices)
        self.event_count = {}

    def decide(self, event, account):
        count = self.event_count.get(event.symbol, 0)
        self.event_count[event.symbol] = count + 1
        if count >= self.slices:
            return (NoOp(reason="parent execution schedule completed"),)
        return (
            SubmitOrder(
                client_order_id=f"twap-{event.symbol}-{count:04d}",
                symbol=event.symbol,
                side=self.side,
                quantity=self.child_quantity,
                reason=f"scheduled child {count + 1}/{self.slices}",
            ),
        )
