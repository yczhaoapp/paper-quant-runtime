import json
import math
import random
from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    MarketKind,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class GMMConditionalKNN:
    declaration = StrategyDeclaration(
        contract_version="1.1",
        strategy_id="supplement.gmm_knn",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.MINUTE,
            bar_seconds=60,
            market_kind=MarketKind.BAR,
            fields=frozenset({"open", "high", "low", "close", "volume"}),
            symbols=frozenset({"BTCUSDT"}),
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/2409.03762v1",
        max_abs_position=Decimal("0.01"),
        max_order_quantity=Decimal("0.02"),
    )

    def __init__(self):
        self.block = []

    @staticmethod
    def features(closes, threshold):
        if len(closes) != 59 or threshold <= 0:
            raise ValueError("Features require 59 past closes and a positive threshold")
        changes = [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False)]
        peaks = [
            (closes[i - 1] - 2 * closes[i] + closes[i + 1], closes[i])
            for i in range(1, 58)
            if closes[i] > closes[i - 1] and closes[i] > closes[i + 1]
        ]
        if len(peaks) >= 2:
            mx, my = sum(p[0] for p in peaks) / len(peaks), sum(p[1] for p in peaks) / len(peaks)
            den = sum((p[0] - mx) ** 2 for p in peaks)
            slope = sum((x - mx) * (y - my) for x, y in peaks) / den if den > 1e-16 else 0.0
            intercept = my - slope * mx
        else:
            mx, my, slope, intercept = (
                (peaks[0][0], peaks[0][1], 0.0, peaks[0][1])
                if peaks
                else (0.0, closes[-1], 0.0, closes[-1])
            )
        return [
            sum(v > threshold for v in changes) / 58,
            sum(v < -threshold for v in changes) / 58,
            closes[-1],
            intercept,
            slope,
            mx,
            my,
            (sum(closes[-6:]) / 6) / (sum(closes) / 59) - 1,
        ]

    @staticmethod
    def responsibilities(row, model):
        logp = []
        for weight, mean, variance in zip(
            model["weights"], model["means"], model["variances"], strict=False
        ):
            logp.append(
                math.log(weight)
                - 0.5
                * sum(
                    math.log(2 * math.pi * v) + (x - m) ** 2 / v
                    for x, m, v in zip(row, mean, variance, strict=False)
                )
            )
        maximum = max(logp)
        denominator = sum(math.exp(v - maximum) for v in logp)
        return [math.exp(v - maximum) / denominator for v in logp], maximum + math.log(denominator)

    @classmethod
    def fit_gmm(cls, rows, k, seed):
        rng = random.Random(seed)
        chosen = rng.sample(rows, k)
        model = {
            "weights": [1 / k] * k,
            "means": [list(r) for r in chosen],
            "variances": [[1.0] * 8 for _ in range(k)],
        }
        for _iteration in range(25):
            responsibilities = [cls.responsibilities(row, model)[0] for row in rows]
            counts = [sum(r[c] for r in responsibilities) for c in range(k)]
            if min(counts) <= 1e-12:
                raise ValueError("GMM component collapsed")
            means = [
                [
                    sum(r[c] * row[j] for r, row in zip(responsibilities, rows, strict=False))
                    / counts[c]
                    for j in range(8)
                ]
                for c in range(k)
            ]
            variances = [
                [
                    sum(
                        r[c] * (row[j] - means[c][j]) ** 2
                        for r, row in zip(responsibilities, rows, strict=False)
                    )
                    / counts[c]
                    + 1e-3
                    for j in range(8)
                ]
                for c in range(k)
            ]
            model = {
                "weights": [n / len(rows) for n in counts],
                "means": means,
                "variances": variances,
            }
        likelihood = sum(cls.responsibilities(row, model)[1] for row in rows)
        # Diagonal Gaussian: means + diagonal variance + k-1 independent weights.
        model["bic"] = -2 * likelihood + (2 * k * 8 + k - 1) * math.log(len(rows))
        return model

    def train(self, request):
        if len(request.samples) < 12:
            raise ValueError("GMM/KNN needs twelve training blocks")
        closes = [[float(s.features[f"close_{i}"]) for i in range(60)] for s in request.samples]
        returns = [math.log(row[-1] / row[-2]) for row in closes]
        continuous_closes = [value for row in closes for value in row]
        minute_returns = [
            math.log(b / a) for a, b in zip(continuous_closes, continuous_closes[1:], strict=False)
        ]
        ordered = sorted(minute_returns)
        index = (len(ordered) - 1) * 0.04
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        threshold = abs(ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo]))
        if threshold <= 0:
            raise ValueError("Training quantile yields a degenerate label threshold")
        rows = [self.features(row[:59], threshold) for row in closes]
        center = [sum(row[j] for row in rows) / len(rows) for j in range(8)]
        scales = [
            math.sqrt(sum((row[j] - center[j]) ** 2 for row in rows) / len(rows)) or 1.0
            for j in range(8)
        ]
        normalized = [
            [(v - m) / s for v, m, s in zip(row, center, scales, strict=False)] for row in rows
        ]
        candidates = [self.fit_gmm(normalized, k, request.seed + k) for k in range(1, 5)]
        model = min(candidates, key=lambda m: m["bic"])
        labels = [1 if r > threshold else -1 if r < -threshold else 0 for r in returns]
        assignments = [
            max(range(len(model["weights"])), key=lambda c: self.responsibilities(row, model)[0][c])
            for row in normalized
        ]
        groups = [
            [
                (row, label)
                for row, label, assignment in zip(normalized, labels, assignments, strict=False)
                if assignment == c
            ]
            for c in range(len(model["weights"]))
        ]
        if any(not group for group in groups):
            raise ValueError("Selected GMM has an empty KNN training cluster")
        return json.dumps(
            {
                "threshold": threshold,
                "center": center,
                "scales": scales,
                "gmm": model,
                "groups": groups,
                "training": {
                    "seed": request.seed,
                    "blocks": len(rows),
                    "bic_by_k": [m["bic"] for m in candidates],
                    "label_counts": {str(label): labels.count(label) for label in (-1, 0, 1)},
                    "cluster_counts": [len(g) for g in groups],
                    "covariance": "diagonal",
                    "neighbors": 1,
                },
            },
            sort_keys=True,
        ).encode()

    def load(self, payload):
        self.model = json.loads(payload)

    def predict(self, closes):
        row = self.features(closes, self.model["threshold"])
        normalized = [
            (v - m) / s
            for v, m, s in zip(row, self.model["center"], self.model["scales"], strict=False)
        ]
        probabilities, _ = self.responsibilities(normalized, self.model["gmm"])
        cluster = max(range(len(probabilities)), key=lambda c: probabilities[c])
        nearest = min(
            self.model["groups"][cluster],
            key=lambda pair: sum((a - b) ** 2 for a, b in zip(normalized, pair[0], strict=False)),
        )
        return nearest[1]

    def decide(self, event, account):
        del account
        self.block.append(float(event.values["close"]))
        if len(self.block) == 59:
            decision = self.predict(self.block)
            return (
                TargetPosition(
                    symbol=event.symbol,
                    quantity=Decimal(decision) * Decimal("0.001"),
                    reason="train-only GMM cluster and nearest-neighbor minute-60 prediction",
                ),
            )
        if len(self.block) == 60:
            self.block = []
            return (
                TargetPosition(
                    symbol=event.symbol,
                    quantity=Decimal(0),
                    reason="one-minute prediction horizon completed",
                ),
            )
        return (NoOp(reason="collecting causal 59-minute feature block"),)
