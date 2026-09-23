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

FIELDS = ("open", "high", "low", "close", "volume")


class RidgeThreeDayPrice:
    """Ridge fit on three completed OHLCV bars for a three-day close forecast."""

    FIELDS = FIELDS
    FEATURES = tuple(f"{field}_lag{lag}" for lag in (2, 1, 0) for field in FIELDS)
    PENALTY = 1.0

    declaration = StrategyDeclaration(
        strategy_id="supervised.ridge_three_day_price",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset(FIELDS),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=4,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/2310.09903v5",
        max_abs_position=Decimal("1"),
    )

    def __init__(self):
        self.model = None
        self.history = {}

    @classmethod
    def _vector(cls, features):
        if set(features) != set(cls.FEATURES):
            raise ValueError("three-day OHLCV window is incomplete")
        values = [float(features[name]) for name in cls.FEATURES]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite OHLCV feature")
        if any(value <= 0 for index, value in enumerate(values) if index % 5 != 4):
            raise ValueError("nonpositive OHLC price")
        if any(value < 0 for index, value in enumerate(values) if index % 5 == 4):
            raise ValueError("negative volume")
        return values

    @staticmethod
    def _solve(matrix, right):
        size = len(right)
        augmented = [row[:] + [value] for row, value in zip(matrix, right, strict=True)]
        for column in range(size):
            pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
            if abs(augmented[pivot][column]) < 1e-10:
                raise ValueError("ridge normal equations are singular")
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            divisor = augmented[column][column]
            for index in range(column, size + 1):
                augmented[column][index] /= divisor
            for row in range(column + 1, size):
                multiplier = augmented[row][column]
                for index in range(column, size + 1):
                    augmented[row][index] -= multiplier * augmented[column][index]
        coefficients = [0.0] * size
        for row in range(size - 1, -1, -1):
            coefficients[row] = augmented[row][size] - sum(
                augmented[row][column] * coefficients[column] for column in range(row + 1, size)
            )
        return coefficients

    def train(self, request):
        if len(request.samples) < 50:
            raise ValueError("at least fifty labeled windows are required")
        raw = []
        targets = []
        for sample in request.samples:
            if sample.target is None or not sample.target.is_finite() or sample.target <= 0:
                raise ValueError("three-day closing price target is invalid")
            raw.append(self._vector(sample.features))
            targets.append(float(sample.target))
        size = len(raw)
        width = len(self.FEATURES)
        means = [sum(row[column] for row in raw) / size for column in range(width)]
        scales = [
            max(
                math.sqrt(sum((row[column] - means[column]) ** 2 for row in raw) / size),
                1.0,
            )
            for column in range(width)
        ]
        centered = [
            [(row[column] - means[column]) / scales[column] for column in range(width)]
            for row in raw
        ]
        target_mean = sum(targets) / size
        gram = [[0.0] * width for _ in range(width)]
        right = [0.0] * width
        for row, target in zip(centered, targets, strict=True):
            for column in range(width):
                right[column] += row[column] * (target - target_mean)
                for other in range(width):
                    gram[column][other] += row[column] * row[other]
        for column in range(width):
            gram[column][column] += self.PENALTY
        weights = self._solve(gram, right)
        return json.dumps(
            {
                "weights": [round(value, 10) for value in weights],
                "means": [round(value, 10) for value in means],
                "scales": [round(value, 10) for value in scales],
                "target_mean": round(target_mean, 10),
                "penalty": self.PENALTY,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"weights", "means", "scales", "target_mean", "penalty"}:
            raise ValueError("unexpected ridge model fields")
        width = len(self.FEATURES)
        if any(len(model[name]) != width for name in ("weights", "means", "scales")):
            raise ValueError("invalid ridge model dimensions")
        values = model["weights"] + model["means"] + model["scales"]
        values += [model["target_mean"], model["penalty"]]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("nonfinite ridge model")
        if any(value <= 0 for value in model["scales"]):
            raise ValueError("invalid ridge feature scale")
        if model["penalty"] != self.PENALTY:
            raise ValueError("ridge penalty differs")
        self.model = model
        self.history = {}

    def decide(self, event, account):
        if self.model is None:
            raise ValueError("model not loaded")
        bars = self.history.setdefault(event.symbol, [])
        bars.append(event.values)
        if len(bars) > 3:
            bars.pop(0)
        if len(bars) < 3:
            return (NoOp(reason="three completed bars required"),)
        features = {
            f"{field}_lag{lag}": bars[index][field]
            for index, lag in enumerate((2, 1, 0))
            for field in self.FIELDS
        }
        values = self._vector(features)
        score = self.model["target_mean"] + sum(
            weight * (value - mean) / scale
            for weight, value, mean, scale in zip(
                self.model["weights"],
                values,
                self.model["means"],
                self.model["scales"],
                strict=True,
            )
        )
        if not math.isfinite(score) or score <= 0:
            raise ValueError("ridge forecast is invalid")
        forecast = Decimal(str(round(score, 6)))
        prediction = Prediction(symbol=event.symbol, value=forecast, reason="close in three days")
        close = event.values["close"]
        desired = (
            Decimal(1) if forecast > close else Decimal(-1) if forecast < close else Decimal(0)
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
            return (prediction, NoOp(reason="position already matches forecast"))
        return (
            prediction,
            TargetPosition(symbol=event.symbol, quantity=desired, reason="forecast-to-position"),
        )
