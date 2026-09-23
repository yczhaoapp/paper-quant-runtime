import json
import math
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


class CrossSectionalRank:
    """Two-feature elastic net forecast followed by one-long/one-short ranking."""

    SYMBOLS = ("ALPHA", "BETA", "GAMMA")
    FEATURES = ("signal", "liquidity")
    PENALTY = 0.001
    RIDGE_SHARE = 0.5

    declaration = StrategyDeclaration(
        strategy_id="supervised.cross_sectional_rank",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "close", "signal", "liquidity"}),
            symbols=frozenset(SYMBOLS),
            minimum_events_per_symbol=2,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=True,
        source="https://dachxiu.chicagobooth.edu/download/ML.pdf",
        max_abs_position=Decimal(1),
    )

    def __init__(self):
        self.model = None
        self.session = None
        self.scores = {}

    @classmethod
    def _vector(cls, values):
        vector = [float(values[field]) for field in cls.FEATURES]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("nonfinite cross-sectional characteristic")
        return vector

    @staticmethod
    def _soft_threshold(value, threshold):
        if value > threshold:
            return value - threshold
        if value < -threshold:
            return value + threshold
        return 0.0

    def train(self, request):
        if len(request.samples) < 60:
            raise ValueError("at least sixty labeled asset-days are required")
        observations = []
        targets = []
        for sample in request.samples:
            if sample.target is None or not math.isfinite(float(sample.target)):
                raise ValueError("finite next-period return required")
            observations.append(self._vector(sample.features))
            targets.append(float(sample.target))
        count = len(observations)
        means = [sum(row[column] for row in observations) / count for column in range(2)]
        scales = [max((sum((row[column] - means[column]) ** 2 for row in observations)
                       / count) ** 0.5, 1e-12) for column in range(2)]
        rows = [[(row[column] - means[column]) / scales[column] for column in range(2)]
                for row in observations]
        intercept = sum(targets) / count
        centered = [target - intercept for target in targets]
        weights = [0.0, 0.0]
        for _ in range(1000):
            old = weights.copy()
            for column in range(2):
                residual_dot = sum(
                    row[column] * (target - row[1 - column] * weights[1 - column])
                    for row, target in zip(rows, centered, strict=True)
                ) / count
                square = sum(row[column] ** 2 for row in rows) / count
                weights[column] = self._soft_threshold(
                    residual_dot, self.PENALTY * (1 - self.RIDGE_SHARE)
                ) / (square + self.PENALTY * self.RIDGE_SHARE)
            if max(abs(left - right) for left, right in zip(weights, old, strict=True)) < 1e-15:
                break
        return json.dumps(
            {
                "intercept": round(intercept, 12),
                "weights": [round(value, 12) for value in weights],
                "means": [round(value, 12) for value in means],
                "scales": [round(value, 12) for value in scales],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"intercept", "weights", "means", "scales"}:
            raise ValueError("unexpected elastic net model fields")
        if any(len(model[field]) != 2 for field in ("weights", "means", "scales")):
            raise ValueError("wrong elastic net model dimension")
        numbers = [model["intercept"], *model["weights"], *model["means"], *model["scales"]]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers):
            raise ValueError("nonfinite elastic net model")
        if any(value <= 0 for value in model["scales"]):
            raise ValueError("invalid characteristic scale")
        self.model = model

    def decide(self, event, account):
        if self.model is None:
            raise ValueError("model not loaded")
        day = event.event_time.date()
        if self.session is None or day != self.session:
            if self.scores and len(self.scores) != len(self.SYMBOLS):
                raise ValueError("incomplete daily cross-section")
            self.session = day
            self.scores = {}
        if event.symbol in self.scores:
            raise ValueError("duplicate symbol in daily cross-section")
        vector = self._vector(event.values)
        score = self.model["intercept"] + sum(
            weight * (value - mean) / scale
            for weight, value, mean, scale in zip(
                self.model["weights"], vector, self.model["means"],
                self.model["scales"], strict=True
            )
        )
        self.scores[event.symbol] = score
        prediction = Prediction(
            symbol=event.symbol,
            value=Decimal(str(round(score, 10))),
            reason="elastic net next-return forecast",
        )
        if len(self.scores) < len(self.SYMBOLS):
            return (prediction, NoOp(reason="await complete forecast cross-section"))
        ranked = sorted(self.SYMBOLS, key=lambda symbol: (self.scores[symbol], symbol))
        targets = {symbol: Decimal(0) for symbol in self.SYMBOLS}
        targets[ranked[0]] = Decimal(-1)
        targets[ranked[-1]] = Decimal(1)
        current = {position.symbol: position.quantity for position in account.positions}
        actions = [prediction]
        actions.extend(
            TargetPosition(symbol=symbol, quantity=targets[symbol], reason="forecast rank")
            for symbol in self.SYMBOLS if current.get(symbol, Decimal(0)) != targets[symbol]
        )
        if len(actions) == 1:
            actions.append(NoOp(reason="ranked positions unchanged"))
        return tuple(actions)
