from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class MovingAverageCrossover:
    declaration = StrategyDeclaration(
        strategy_id="rule.moving_average_crossover",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"close"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=20,
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=False,
        source="https://arxiv.org/pdf/1504.04254",
        max_abs_position=Decimal("1"),
    )

    def __init__(self):
        self.closes = []

    def decide(self, event, account):
        self.closes.append(event.values["close"])
        if len(self.closes) < 20:
            return (NoOp(reason="twenty-day warmup"),)
        short_mean = sum(self.closes[-2:], Decimal("0")) / Decimal("2")
        long_mean = sum(self.closes[-20:], Decimal("0")) / Decimal("20")
        desired = (
            Decimal("1")
            if short_mean > long_mean
            else Decimal("-1")
            if short_mean < long_mean
            else Decimal("0")
        )
        current = next(
            (
                position.quantity
                for position in account.positions
                if position.symbol == event.symbol
            ),
            Decimal("0"),
        )
        if current == desired:
            return (NoOp(reason="position already matches signal"),)
        return (TargetPosition(symbol=event.symbol, quantity=desired, reason="VMA(2,20,0)"),)
