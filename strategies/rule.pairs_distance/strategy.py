from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class DistancePair:
    """One minimum-distance pair, fixed 252-session formation and 126-session trading."""

    SYMBOLS = ("ALPHA", "BETA", "GAMMA")
    FORMATION = 252
    TRADING = 126
    NOTIONAL = Decimal("100")

    declaration = StrategyDeclaration(
        strategy_id="rule.pairs_distance",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"close"}),
            symbols=frozenset(SYMBOLS),
            minimum_events_per_symbol=FORMATION + TRADING,
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=False,
        source="https://repec.som.yale.edu/icfpub/publications/2573.pdf",
        max_abs_position=Decimal("5"),
    )

    def __init__(self):
        self.history = {symbol: [] for symbol in self.SYMBOLS}
        self.session = None
        self.seen = set()
        self.pair = None
        self.sigma = None
        self.direction = 0
        self.quantities = None

    def _select_pair(self):
        normalized = {
            symbol: [price / self.history[symbol][0] for price in prices]
            for symbol, prices in self.history.items()
        }
        distances = {
            pair: sum((left - right) ** 2 for left, right in zip(
                normalized[pair[0]], normalized[pair[1]], strict=True
            ))
            for pair in (("ALPHA", "BETA"), ("ALPHA", "GAMMA"), ("BETA", "GAMMA"))
        }
        self.pair = min(distances, key=distances.__getitem__)
        first, second = self.pair
        differences = [left - right for left, right in zip(
            normalized[first], normalized[second], strict=True
        )]
        mean = sum(differences) / Decimal(len(differences))
        self.sigma = (sum((value - mean) ** 2 for value in differences)
                      / Decimal(len(differences))).sqrt()
        if self.sigma == 0:
            raise ValueError("Formation spread has zero volatility")

    def _targets(self, direction, reason):
        first, second = self.pair
        if direction == 0:
            return tuple(TargetPosition(symbol=symbol, quantity=Decimal(0), reason=reason)
                         for symbol in self.pair)
        if self.quantities is None:
            self.quantities = (
                self.NOTIONAL / self.history[first][-1],
                self.NOTIONAL / self.history[second][-1],
            )
        return (
            TargetPosition(symbol=first, quantity=direction * self.quantities[0], reason=reason),
            TargetPosition(symbol=second, quantity=-direction * self.quantities[1], reason=reason),
        )

    def decide(self, event, account):
        del account
        day = event.event_time.date()
        if self.session is None or day != self.session:
            if self.seen and len(self.seen) != len(self.SYMBOLS):
                raise ValueError("Pair session is missing one or more symbols")
            self.session = day
            self.seen = set()
        if event.symbol in self.seen:
            raise ValueError("Duplicate symbol in pair session")
        self.seen.add(event.symbol)
        self.history[event.symbol].append(event.values["close"])
        if len(self.seen) < len(self.SYMBOLS):
            return (NoOp(reason="await complete daily cross-section"),)

        count = len(self.history[self.SYMBOLS[0]])
        if count == self.FORMATION:
            self._select_pair()
            return (NoOp(reason="formation pair selected"),)
        if count <= self.FORMATION:
            return (NoOp(reason="formation window"),)
        if count >= self.FORMATION + self.TRADING:
            return (NoOp(reason="trading interval ended"),)
        first, second = self.pair
        spread = (self.history[first][-1] / self.history[first][0]
                  - self.history[second][-1] / self.history[second][0])
        if self.direction:
            crossed = spread >= 0 if self.direction > 0 else spread <= 0
            if crossed or count == self.FORMATION + self.TRADING - 1:
                self.direction = 0
                self.quantities = None
                return self._targets(0, "pair convergence or interval exit")
            return (NoOp(reason="open pair has not crossed"),)
        if abs(spread) > Decimal(2) * self.sigma:
            self.direction = -1 if spread > 0 else 1
            self.quantities = None
            return self._targets(self.direction, "two-sigma pair divergence")
        return (NoOp(reason="pair spread below entry threshold"),)
