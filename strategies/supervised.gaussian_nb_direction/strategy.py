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


class GaussianNBDirection:
    """Gaussian Naive Bayes component with four close-time OHLCV features."""

    declaration = StrategyDeclaration(
        strategy_id="supervised.gaussian_nb_direction",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "high", "low", "close", "volume"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=2,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/2107.13148v3",
        max_abs_position=Decimal(1),
    )

    VARIANCE_FLOOR = 1e-8

    def __init__(self):
        self.model = None

    @staticmethod
    def _features(values):
        opening = float(values["open"])
        high = float(values["high"])
        low = float(values["low"])
        close = float(values["close"])
        volume = float(values["volume"])
        if not all(math.isfinite(value) for value in (opening, high, low, close, volume)):
            raise ValueError("nonfinite OHLCV")
        if opening <= 0 or low <= 0 or low > high or not low <= close <= high or volume < 0:
            raise ValueError("invalid OHLCV")
        return (
            (close - opening) / opening,
            (high - low) / opening,
            (close - low) / (high - low) if high > low else 0.5,
            math.log1p(volume),
        )

    def train(self, request):
        if len(request.samples) < 50:
            raise ValueError("fifty labeled bars are required")
        by_class = {0: [], 1: []}
        for sample in request.samples:
            if sample.target not in (Decimal(0), Decimal(1)):
                raise ValueError("direction label must be binary")
            by_class[int(sample.target)].append(self._features(sample.features))
        if min(len(rows) for rows in by_class.values()) < 2:
            raise ValueError("each class needs at least two bars")
        model = {}
        total = len(request.samples)
        for label, rows in by_class.items():
            means = [sum(row[column] for row in rows) / len(rows) for column in range(4)]
            variances = [max(
                sum((row[column] - means[column]) ** 2 for row in rows) / len(rows),
                self.VARIANCE_FLOOR,
            ) for column in range(4)]
            model[str(label)] = {
                "prior": round(len(rows) / total, 12),
                "means": [round(value, 12) for value in means],
                "variances": [round(value, 12) for value in variances],
            }
        return json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"0", "1"}:
            raise ValueError("missing Gaussian class")
        for label in ("0", "1"):
            group = model[label]
            if set(group) != {"prior", "means", "variances"}:
                raise ValueError("unexpected Gaussian model field")
            if len(group["means"]) != 4 or len(group["variances"]) != 4:
                raise ValueError("wrong Gaussian dimension")
            numbers = [group["prior"], *group["means"], *group["variances"]]
            if not all(type(number) in (int, float) and math.isfinite(number)
                       for number in numbers):
                raise ValueError("nonfinite Gaussian model")
            if group["prior"] <= 0 or any(value <= 0 for value in group["variances"]):
                raise ValueError("invalid Gaussian probability or variance")
        if abs(model["0"]["prior"] + model["1"]["prior"] - 1) > 1e-9:
            raise ValueError("class priors do not sum to one")
        self.model = model

    def decide(self, event, account):
        if self.model is None:
            raise ValueError("model not loaded")
        features = self._features(event.values)
        log_scores = []
        for label in ("0", "1"):
            group = self.model[label]
            score = math.log(group["prior"])
            for value, mean, variance in zip(
                features, group["means"], group["variances"], strict=True
            ):
                score -= 0.5 * (math.log(2 * math.pi * variance)
                                + (value - mean) ** 2 / variance)
            log_scores.append(score)
        difference = log_scores[1] - log_scores[0]
        probability = (1 / (1 + math.exp(-difference)) if difference >= 0
                       else math.exp(difference) / (1 + math.exp(difference)))
        published = Decimal(str(round(probability, 8)))
        prediction = Prediction(
            symbol=event.symbol,
            value=published,
            reason="Gaussian Naive Bayes next-close up probability",
        )
        desired = Decimal(1) if published > Decimal("0.5") else Decimal(-1)
        current = next((position.quantity for position in account.positions
                        if position.symbol == event.symbol), Decimal(0))
        if current == desired:
            return (prediction, NoOp(reason="position matches Gaussian forecast"))
        return (prediction, TargetPosition(
            symbol=event.symbol, quantity=desired, reason="Gaussian forecast-to-position"
        ))
