from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class ChannelBreakout:
    """TRB(50, 0.01) signal with a ten-session fixed holding period."""

    declaration = StrategyDeclaration(
        strategy_id="rule.channel_breakout",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"close"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=51,
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=False,
        source="https://arxiv.org/pdf/1504.04254",
        max_abs_position=Decimal("1"),
    )

    def __init__(self):
        self.history = []
        self.exit_at = None

    def decide(self, event, account):
        del account
        price = event.values["close"]
        previous = self.history[-1] if self.history else None
        prior_range = self.history[-50:]
        self.history.append(price)
        index = len(self.history) - 1

        if self.exit_at is not None:
            if index >= self.exit_at:
                self.exit_at = None
                return (
                    TargetPosition(symbol=event.symbol, quantity=Decimal("0"), reason="TRB exit"),
                )
            return (NoOp(reason="fixed holding period"),)

        if len(prior_range) != 50 or previous is None:
            return (NoOp(reason="fifty-session warmup"),)
        ceiling = max(prior_range) * Decimal("1.01")
        floor = min(prior_range) * Decimal("0.99")
        if previous < ceiling and price > ceiling:
            self.exit_at = index + 10
            return (TargetPosition(symbol=event.symbol, quantity=Decimal("1"), reason="TRB long"),)
        if previous > floor and price < floor:
            self.exit_at = index + 10
            return (
                TargetPosition(symbol=event.symbol, quantity=Decimal("-1"), reason="TRB short"),
            )
        return (NoOp(reason="no range break"),)
