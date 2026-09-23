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


class DailyDirectionLogistic:
    """Five-feature next-close direction logistic model with deterministic batch updates."""

    declaration = StrategyDeclaration(
        strategy_id="supervised.logistic_direction",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "high", "low", "close", "volume"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=2,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/2310.16855",
        max_abs_position=Decimal("1"),
    )

    FEATURES = ("open", "high", "low", "close", "volume")
    ALPHA = 0.01
    EPOCHS = 1000

    def __init__(self):
        self.weights = None
        self.means = None
        self.scales = None

    @staticmethod
    def _sigmoid(value):
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    @classmethod
    def _vector(cls, values):
        vector = [float(values[name]) for name in cls.FEATURES]
        if not all(math.isfinite(number) for number in vector):
            raise ValueError("nonfinite OHLCV feature")
        if min(vector[:4]) <= 0 or vector[4] < 0:
            raise ValueError("invalid OHLCV feature")
        return vector

    def train(self, request):
        if len(request.samples) < 50:
            raise ValueError("at least fifty labeled daily bars are required")
        raw = []
        labels = []
        for sample in request.samples:
            if sample.target not in (Decimal("0"), Decimal("1")):
                raise ValueError("direction label must be binary")
            raw.append(self._vector(sample.features))
            labels.append(float(sample.target))
        if len(set(labels)) != 2:
            raise ValueError("both directional classes are required")
        count = len(raw)
        means = [sum(row[column] for row in raw) / count for column in range(5)]
        scales = [
            max(
                math.sqrt(sum((row[column] - means[column]) ** 2 for row in raw) / count),
                1e-12,
            )
            for column in range(5)
        ]
        samples = [
            [1.0] + [(row[column] - means[column]) / scales[column] for column in range(5)]
            for row in raw
        ]
        weights = [0.0] * 6
        for _ in range(self.EPOCHS):
            gradient = [0.0] * 6
            for features, label in zip(samples, labels, strict=True):
                estimate = self._sigmoid(sum(a * b for a, b in zip(weights, features, strict=True)))
                for column in range(6):
                    gradient[column] += (estimate - label) * features[column]
            for column in range(6):
                weights[column] -= self.ALPHA * gradient[column] / count
        # Host math libraries can differ at the final binary digit. Persist a
        # declared decimal precision so the same fit has identical model bytes.
        return json.dumps(
            {
                "weights": [round(value, 12) for value in weights],
                "means": [round(value, 12) for value in means],
                "scales": [round(value, 12) for value in scales],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"weights", "means", "scales"}:
            raise ValueError("unexpected model fields")
        weights = [float(value) for value in model["weights"]]
        means = [float(value) for value in model["means"]]
        scales = [float(value) for value in model["scales"]]
        if (len(weights), len(means), len(scales)) != (6, 5, 5):
            raise ValueError("invalid logistic model dimensions")
        if not all(math.isfinite(value) for value in weights + means + scales):
            raise ValueError("nonfinite logistic model")
        if any(value <= 0 for value in scales):
            raise ValueError("invalid feature scale")
        self.weights = weights
        self.means = means
        self.scales = scales

    def decide(self, event, account):
        if self.weights is None or self.means is None or self.scales is None:
            raise ValueError("model not loaded")
        raw = self._vector(event.values)
        features = [1.0] + [
            (raw[column] - self.means[column]) / self.scales[column]
            for column in range(5)
        ]
        score = sum(a * b for a, b in zip(self.weights, features, strict=True))
        probability = self._sigmoid(score)
        published_probability = Decimal(str(round(probability, 8)))
        forecast = Prediction(
            symbol=event.symbol,
            value=published_probability,
            reason="next-close up probability",
        )
        desired = (
            Decimal("1")
            if published_probability > Decimal("0.5")
            else Decimal("-1")
            if published_probability < Decimal("0.5")
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
            return (forecast, NoOp(reason="position already matches forecast"))
        return (
            forecast,
            TargetPosition(symbol=event.symbol, quantity=desired, reason="forecast-to-position"),
        )
