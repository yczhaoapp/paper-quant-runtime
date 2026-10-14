import math
from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class GraphMatchingPairs:
    declaration = StrategyDeclaration(
        contract_version="1.1",
        strategy_id="supplement.graph_matching",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "close"}),
            symbols=frozenset({"PAIR_A", "PAIR_B", "PAIR_C", "PAIR_D"}),
            minimum_events_per_symbol=505,
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=False,
        source="https://arxiv.org/pdf/2403.07998v1",
        max_abs_position=Decimal("100"),
        max_order_quantity=Decimal("200"),
    )

    def __init__(self):
        self.symbols = tuple(sorted(self.declaration.data.symbols))
        self.history = {symbol: [] for symbol in self.symbols}
        self.frame = {}
        self.frame_time = None
        self.pairs = ()
        self.month = None

    @staticmethod
    def regression(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        denominator = sum((v - mx) ** 2 for v in x)
        if denominator <= 1e-16:
            raise ValueError("Singular log-price regression")
        beta = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=False)) / denominator
        intercept = my - beta * mx
        residual = [b - intercept - beta * a for a, b in zip(x, y, strict=False)]
        return intercept, beta, residual

    @staticmethod
    def adf_statistic(residual):
        # One-lag augmented Dickey-Fuller regression, intercept included.
        rows = [
            [1.0, residual[t - 1], residual[t - 1] - residual[t - 2]]
            for t in range(2, len(residual))
        ]
        y = [residual[t] - residual[t - 1] for t in range(2, len(residual))]
        if len(rows) <= 3:
            raise ValueError("Too few ADF observations")
        gram = [[sum(row[i] * row[j] for row in rows) for j in range(3)] for i in range(3)]
        augmented = [gram[i] + [float(i == j) for j in range(3)] for i in range(3)]
        for i in range(3):
            pivot = max(range(i, 3), key=lambda j: abs(augmented[j][i]))
            augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
            if abs(augmented[i][i]) <= 1e-16:
                raise ValueError("Singular single-lag ADF regression")
            scale = augmented[i][i]
            augmented[i] = [v / scale for v in augmented[i]]
            for j in range(3):
                if j != i:
                    scale = augmented[j][i]
                    augmented[j] = [
                        a - scale * b for a, b in zip(augmented[j], augmented[i], strict=False)
                    ]
        inverse = [row[3:] for row in augmented]
        rhs = [sum(row[i] * target for row, target in zip(rows, y, strict=False)) for i in range(3)]
        coef = [sum(a * b for a, b in zip(row, rhs, strict=False)) for row in inverse]
        sse = sum(
            (target - sum(a * b for a, b in zip(row, coef, strict=False))) ** 2
            for row, target in zip(rows, y, strict=False)
        )
        error = math.sqrt(sse / (len(rows) - 3) * inverse[1][1])
        if error <= 1e-16:
            raise ValueError("Zero ADF standard error")
        return coef[1] / error

    @staticmethod
    def maximum_matching(symbols, weights):
        # Exact subset dynamic programming; bounded universe replaces blossom.
        memo = {}

        def solve(remaining):
            if not remaining:
                return 0.0, ()
            if remaining in memo:
                return memo[remaining]
            first, rest = remaining[0], remaining[1:]
            best = solve(rest)
            for second in rest:
                weight, pairs = solve(tuple(x for x in rest if x != second))
                candidate = weight + weights[(first, second)], ((first, second),) + pairs
                if candidate[0] > best[0]:
                    best = candidate
            memo[remaining] = best
            return best

        return solve(tuple(symbols))[1]

    def decide(self, event, account):
        del account
        if self.frame and self.frame_time != event.available_time:
            raise ValueError("Daily pair universe is not synchronized")
        self.frame_time = event.available_time
        self.frame[event.symbol] = float(event.values["close"])
        if len(self.frame) != len(self.symbols):
            return (NoOp(reason="wait for synchronized daily pair universe"),)
        prices, self.frame = self.frame, {}
        if min(len(v) for v in self.history.values()) < 504:
            for symbol in self.symbols:
                self.history[symbol].append(math.log(prices[symbol]))
            return (NoOp(reason="two-year 504-session calibration warmup"),)
        month = (event.event_time.year, event.event_time.month)
        if month != self.month:
            weights = {}
            for i, left in enumerate(self.symbols):
                for right in self.symbols[i + 1 :]:
                    _, _, residual = self.regression(
                        self.history[left][-504:], self.history[right][-504:]
                    )
                    weights[left, right] = -self.adf_statistic(residual)
            self.pairs = self.maximum_matching(self.symbols, weights)
            self.month = month
        quantities = {symbol: 0.0 for symbol in self.symbols}
        for left, right in self.pairs:
            intercept, beta, residual = self.regression(
                self.history[left][-504:], self.history[right][-504:]
            )
            spread = math.log(prices[right]) - intercept - beta * math.log(prices[left])
            # Paper calls this a z-score while its prose says variance; use SD
            # as required by the z-score definition and disclose that ambiguity.
            sd = math.sqrt(sum(v * v for v in residual) / (len(residual) - 2))
            if sd <= 1e-16:
                raise ValueError("Degenerate spread scale")
            z = max(-3.0, min(3.0, spread / sd))
            signal = 1 if z >= 2 else -1 if z <= -2 else 0
            quantities[left] = signal * beta * 1000.0 / prices[left]
            quantities[right] = -signal * 1000.0 / prices[right]
        for symbol in self.symbols:
            self.history[symbol].append(math.log(prices[symbol]))
            self.history[symbol] = self.history[symbol][-504:]
        return tuple(
            TargetPosition(
                symbol=symbol,
                quantity=Decimal(f"{quantities[symbol]:.8f}"),
                reason="maximum-weight matching plus causal z-score spread",
            )
            for symbol in self.symbols
        )
