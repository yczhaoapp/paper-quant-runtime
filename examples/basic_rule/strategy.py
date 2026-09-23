from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class MovingAverageRule:
    declaration = StrategyDeclaration(
        strategy_id="example.moving_average",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.MINUTE,
            fields=frozenset({"open", "close", "volume"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=5,
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=False,
        source="contract-example: moving-average crossover",
        max_abs_position=Decimal("1"),
    )

    def __init__(self):
        self.closes = []

    def decide(self, event, account):
        self.closes.append(event.values["close"])
        if len(self.closes) < 4:
            return (NoOp(reason="warmup"),)
        fast = sum(self.closes[-2:]) / 2
        slow = sum(self.closes[-4:]) / 4
        target = Decimal("1") if fast > slow else Decimal("0")
        return (TargetPosition(symbol=event.symbol, quantity=target, reason="ma-cross"),)
