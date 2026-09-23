from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    Prediction,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class WeightedMicropriceToy:
    """The paper's Brownian-imbalance toy case, with a declared trading mapping."""

    declaration = StrategyDeclaration(
        strategy_id="rule.microprice_toy",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.TICK,
            fields=frozenset({"price", "bid_price", "ask_price", "bid_size", "ask_size"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=2,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=False,
        source="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694",
        max_abs_position=Decimal("1"),
    )

    @staticmethod
    def fair_price(values):
        bid = values["bid_price"]
        ask = values["ask_price"]
        bid_size = values["bid_size"]
        ask_size = values["ask_size"]
        if bid <= 0 or ask <= bid or bid_size <= 0 or ask_size <= 0:
            raise ValueError("L1 book must have positive ordered quotes and sizes")
        return (ask * bid_size + bid * ask_size) / (bid_size + ask_size)

    def decide(self, event, account):
        fair = self.fair_price(event.values)
        trade_price = event.values["price"]
        if trade_price <= 0:
            raise ValueError("last trade price must be positive")
        forecast = Prediction(
            symbol=event.symbol,
            value=fair,
            reason="Brownian-imbalance toy fair price",
        )
        desired = (
            Decimal(1) if fair > trade_price else Decimal(-1) if fair < trade_price else Decimal(0)
        )
        current = next(
            (
                position.quantity
                for position in account.positions
                if position.symbol == event.symbol
            ),
            Decimal(0),
        )
        if current == desired:
            return (forecast, NoOp(reason="position already matches toy signal"))
        return (
            forecast,
            TargetPosition(symbol=event.symbol, quantity=desired, reason="toy-price-to-position"),
        )
