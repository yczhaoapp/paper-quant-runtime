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


class QueueImbalanceLogistic:
    declaration = StrategyDeclaration(
        strategy_id="supervised.queue_imbalance",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.TICK,
            fields=frozenset({"price", "bid_price", "ask_price", "bid_size", "ask_size"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=2,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/1512.03492",
        max_abs_position=Decimal("1"),
    )

    def __init__(self):
        self.intercept = None
        self.slope = None

    @staticmethod
    def _imbalance(bid_size, ask_size):
        if bid_size < 0 or ask_size < 0 or bid_size + ask_size <= 0:
            raise ValueError("best-quote queue sizes must be nonnegative with positive sum")
        return float((bid_size - ask_size) / (bid_size + ask_size))

    @staticmethod
    def _sigmoid(value):
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def train(self, request):
        if len(request.samples) < 4:
            raise ValueError("at least four labeled queue observations are required")
        examples = []
        for sample in request.samples:
            if sample.target not in (Decimal("0"), Decimal("1")):
                raise ValueError("next-midprice label must be binary")
            examples.append(
                (
                    self._imbalance(sample.features["bid_size"], sample.features["ask_size"]),
                    float(sample.target),
                )
            )
        if len({label for _, label in examples}) < 2:
            raise ValueError("both movement directions are needed to fit the classifier")
        intercept = 0.0
        slope = 0.0
        rate = 0.4 / len(examples)
        for _ in range(600):
            gradient_intercept = 0.0
            gradient_slope = 0.0
            for imbalance, label in examples:
                residual = label - self._sigmoid(intercept + slope * imbalance)
                gradient_intercept += residual
                gradient_slope += residual * imbalance
            intercept += rate * gradient_intercept
            slope += rate * gradient_slope
        return json.dumps(
            {"intercept": intercept, "slope": slope}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"intercept", "slope"}:
            raise ValueError("unexpected model fields")
        intercept = float(model["intercept"])
        slope = float(model["slope"])
        if not math.isfinite(intercept) or not math.isfinite(slope):
            raise ValueError("nonfinite model coefficient")
        self.intercept = intercept
        self.slope = slope

    def decide(self, event, account):
        if self.intercept is None or self.slope is None:
            raise ValueError("model not loaded")
        bid = event.values["bid_price"]
        ask = event.values["ask_price"]
        if bid <= 0 or ask <= bid:
            raise ValueError("invalid best quote")
        imbalance = self._imbalance(event.values["bid_size"], event.values["ask_size"])
        probability = self._sigmoid(self.intercept + self.slope * imbalance)
        prediction = Prediction(
            symbol=event.symbol,
            value=Decimal(str(probability)),
            reason="next mid-price up probability",
        )
        desired = (
            Decimal("1")
            if probability > 0.5
            else Decimal("-1")
            if probability < 0.5
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
            return (prediction, NoOp(reason="position already matches prediction"))
        return (
            prediction,
            TargetPosition(
                symbol=event.symbol, quantity=desired, reason="prediction-to-position adapter"
            ),
        )
