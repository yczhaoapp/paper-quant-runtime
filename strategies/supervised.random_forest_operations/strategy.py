import json
import math
import random
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


class RandomForestOperations:
    """Bootstrap trees vote on long, short or no-operation classes."""

    TREE_COUNT = 15
    MAX_DEPTH = 4
    MIN_LEAF = 4
    MAX_HOLD = 5
    GAIN = Decimal("0.03")
    LOSS = Decimal("0.015")

    declaration = StrategyDeclaration(
        strategy_id="supervised.random_forest_operations",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "high", "low", "close", "volume"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=2,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/1301.4944v1",
        max_abs_position=Decimal(1),
    )

    def __init__(self):
        self.trees = None
        self.held_days = 0

    @staticmethod
    def _features(values):
        opening = float(values["open"])
        high = float(values["high"])
        low = float(values["low"])
        close = float(values["close"])
        volume = float(values["volume"])
        if not all(math.isfinite(value) for value in (opening, high, low, close, volume)):
            raise ValueError("nonfinite market feature")
        if opening <= 0 or low <= 0 or low > high or not low <= close <= high or volume < 0:
            raise ValueError("invalid OHLCV bar")
        return ((close - opening) / opening, (high - low) / opening, math.log1p(volume))

    @staticmethod
    def _gini(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return 1.0 - sum((amount / total) ** 2 for amount in counts.values())

    @staticmethod
    def _majority(counts):
        return max((-1, 0, 1), key=lambda label: (counts[label], -abs(label), -label))

    def _grow(self, examples, depth, generator):
        counts = {label: sum(row[1] == label for row in examples) for label in (-1, 0, 1)}
        majority = self._majority(counts)
        if (
            depth >= self.MAX_DEPTH
            or len(examples) < 2 * self.MIN_LEAF
            or max(counts.values()) == len(examples)
        ):
            return {"leaf": majority}
        parent = self._gini(counts)
        best = None
        for feature in sorted(generator.sample(range(3), 2)):
            ordered = sorted(examples, key=lambda row: row[0][feature])
            left = {-1: 0, 0: 0, 1: 0}
            right = counts.copy()
            for split in range(1, len(ordered)):
                label = ordered[split - 1][1]
                left[label] += 1
                right[label] -= 1
                if split < self.MIN_LEAF or len(ordered) - split < self.MIN_LEAF:
                    continue
                low = ordered[split - 1][0][feature]
                high = ordered[split][0][feature]
                if low == high:
                    continue
                gain = parent - (
                    split * self._gini(left) + (len(ordered) - split) * self._gini(right)
                ) / len(ordered)
                threshold = (low + high) / 2
                candidate = (gain, -feature, -threshold)
                if best is None or candidate > best[0]:
                    best = (candidate, feature, threshold)
        if best is None or best[0][0] <= 0:
            return {"leaf": majority}
        _, feature, threshold = best
        left_rows = [row for row in examples if row[0][feature] <= threshold]
        right_rows = [row for row in examples if row[0][feature] > threshold]
        return {
            "feature": feature,
            "threshold": round(threshold, 12),
            "left": self._grow(left_rows, depth + 1, generator),
            "right": self._grow(right_rows, depth + 1, generator),
        }

    def train(self, request):
        if len(request.samples) < 80:
            raise ValueError("eighty labeled bars are required")
        examples = []
        for sample in request.samples:
            if sample.target not in (Decimal(-1), Decimal(0), Decimal(1)):
                raise ValueError("operation class must be -1, 0 or 1")
            examples.append((self._features(sample.features), int(sample.target)))
        if len({label for _, label in examples}) < 2:
            raise ValueError("at least two operation classes are required")
        generator = random.Random(request.seed)
        trees = []
        for _ in range(self.TREE_COUNT):
            sample = [examples[generator.randrange(len(examples))] for _ in examples]
            trees.append(self._grow(sample, 0, generator))
        return json.dumps({"trees": trees}, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def _validate_tree(cls, node, depth):
        if not isinstance(node, dict) or depth > cls.MAX_DEPTH:
            raise ValueError("invalid forest tree")
        if set(node) == {"leaf"}:
            if type(node["leaf"]) is not int or node["leaf"] not in (-1, 0, 1):
                raise ValueError("invalid tree leaf")
            return
        if set(node) != {"feature", "threshold", "left", "right"}:
            raise ValueError("invalid tree split")
        if type(node["feature"]) is not int or node["feature"] not in range(3):
            raise ValueError("invalid tree feature")
        if type(node["threshold"]) not in (int, float) or not math.isfinite(node["threshold"]):
            raise ValueError("invalid tree threshold")
        cls._validate_tree(node["left"], depth + 1)
        cls._validate_tree(node["right"], depth + 1)

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"trees"} or len(model["trees"]) != self.TREE_COUNT:
            raise ValueError("invalid forest model")
        for tree in model["trees"]:
            self._validate_tree(tree, 0)
        self.trees = model["trees"]

    @staticmethod
    def _classify(node, features):
        while "leaf" not in node:
            node = node["left"] if features[node["feature"]] <= node["threshold"] else node["right"]
        return node["leaf"]

    def decide(self, event, account):
        if self.trees is None:
            raise ValueError("forest model not loaded")
        features = self._features(event.values)
        votes = {label: 0 for label in (-1, 0, 1)}
        for tree in self.trees:
            votes[self._classify(tree, features)] += 1
        choice = self._majority(votes)
        prediction = Prediction(
            symbol=event.symbol,
            value=Decimal(choice),
            reason="random forest operation vote",
        )
        position = next((item for item in account.positions if item.symbol == event.symbol), None)
        current = position.quantity if position is not None else Decimal(0)
        if current:
            self.held_days += 1
            entry = position.average_price
            gain = current * (event.values["close"] / entry - 1)
            if gain >= self.GAIN or gain <= -self.LOSS or self.held_days >= self.MAX_HOLD:
                self.held_days = 0
                return (prediction, TargetPosition(
                    symbol=event.symbol, quantity=Decimal(0), reason="forest operation exit"
                ))
            return (prediction, NoOp(reason="forest operation remains open"))
        self.held_days = 0
        if choice == 0:
            return (prediction, NoOp(reason="forest votes no operation"))
        return (prediction, TargetPosition(
            symbol=event.symbol, quantity=Decimal(choice), reason="forest operation entry"
        ))
