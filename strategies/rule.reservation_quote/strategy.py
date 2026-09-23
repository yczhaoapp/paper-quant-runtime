from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    Prediction,
    StrategyDeclaration,
    StrategyKind,
    SubmitOrder,
)


class ReservationQuote:
    """Finite-horizon inventory-aware quotes with next-event limit-order expiry."""

    declaration = StrategyDeclaration(
        strategy_id="rule.reservation_quote",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.TICK,
            fields=frozenset(
                {
                    "price",
                    "bid_price",
                    "ask_price",
                    "bid_size",
                    "ask_size",
                    "bid_price_2",
                    "ask_price_2",
                    "bid_size_2",
                    "ask_size_2",
                }
            ),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=32,
        ),
        actions=frozenset({"prediction", "submit_order", "none"}),
        training_required=False,
        source="https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf",
        max_abs_position=Decimal("3"),
        max_order_quantity=Decimal("1"),
    )

    def __init__(
        self,
        *,
        risk_aversion=Decimal("0.1"),
        volatility=Decimal("2"),
        arrival_decay=Decimal("1.5"),
        horizon_events=31,
    ):
        self.risk_aversion = Decimal(risk_aversion)
        self.volatility = Decimal(volatility)
        self.arrival_decay = Decimal(arrival_decay)
        self.horizon_events = horizon_events
        if min(self.risk_aversion, self.volatility, self.arrival_decay) <= 0:
            raise ValueError("quote calibration must be positive")
        if horizon_events < 2:
            raise ValueError("quote horizon must contain at least two events")
        self.seen = {}

    def quotes(self, midpoint, inventory, remaining):
        gamma = self.risk_aversion
        sigma2 = self.volatility * self.volatility
        inventory_term = gamma * sigma2 * remaining
        reservation = midpoint - inventory * inventory_term
        spread = (
            inventory_term + Decimal(2) / gamma * (Decimal(1) + gamma / self.arrival_decay).ln()
        )
        return reservation, reservation - spread / 2, reservation + spread / 2

    @staticmethod
    def _check_book(values):
        bid = values["bid_price"]
        ask = values["ask_price"]
        lower_bid = values["bid_price_2"]
        upper_ask = values["ask_price_2"]
        if not (0 < lower_bid < bid < ask < upper_ask):
            raise ValueError("L2 price levels are crossed or unordered")
        if any(values[name] <= 0 for name in ("bid_size", "ask_size", "bid_size_2", "ask_size_2")):
            raise ValueError("L2 price levels require positive size")
        return (bid + ask) / 2

    def decide(self, event, account):
        midpoint = self._check_book(event.values)
        index = self.seen.get(event.symbol, 0)
        self.seen[event.symbol] = index + 1
        remaining = max(Decimal(0), Decimal(self.horizon_events - index) / self.horizon_events)
        inventory = next(
            (
                position.quantity
                for position in account.positions
                if position.symbol == event.symbol
            ),
            Decimal(0),
        )
        reservation, bid_quote, ask_quote = self.quotes(midpoint, inventory, remaining)
        actions = [
            Prediction(symbol=event.symbol, value=reservation, reason="inventory reservation price")
        ]
        if index >= self.horizon_events:
            actions.append(NoOp(reason="finite quote horizon completed"))
            return tuple(actions)
        if inventory < Decimal(2):
            actions.append(
                SubmitOrder(
                    client_order_id=f"maker-bid-{index:04d}",
                    symbol=event.symbol,
                    side="buy",
                    quantity=Decimal(1),
                    limit_price=bid_quote,
                    reason="inventory-aware bid quote",
                )
            )
        if inventory > Decimal(-2):
            actions.append(
                SubmitOrder(
                    client_order_id=f"maker-ask-{index:04d}",
                    symbol=event.symbol,
                    side="sell",
                    quantity=Decimal(1),
                    limit_price=ask_quote,
                    reason="inventory-aware ask quote",
                )
            )
        return tuple(actions)
