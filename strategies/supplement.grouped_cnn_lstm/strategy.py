import json
import math
import random
from decimal import Decimal

from paperquant.strategy_api import (
    Conv1D,
    DataNeed,
    DenseMLP,
    Granularity,
    LSTMCell,
    MarketKind,
    NoOp,
    Scalar,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
    gradient_step,
)


class GroupedCNNLSTM:
    declaration = StrategyDeclaration(
        contract_version="1.1",
        strategy_id="supplement.grouped_cnn_lstm",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.MINUTE,
            bar_seconds=14400,
            market_kind=MarketKind.BAR,
            fields=frozenset({"open", "high", "low", "close", "volume"}),
            symbols=frozenset({"BTCUSDT"}),
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/2205.08382v1",
        max_abs_position=Decimal("2"),
        max_order_quantity=Decimal("4"),
    )

    def __init__(self):
        self.history = []
        self.theta = 0.5

    def initialize(self, seed):
        rng = random.Random(seed)
        self.encoders = [Conv1D(2, 1, 3, rng, padding=1), Conv1D(3, 1, 3, rng, padding=1)]
        self.decoders = [Conv1D(1, 2, 3, rng, padding=1), Conv1D(1, 3, 3, rng, padding=1)]
        self.first = [Conv1D(5, 2, 5, rng), Conv1D(1, 1, 5, rng), Conv1D(1, 1, 5, rng)]
        self.second = [Conv1D(2, 2, 3, rng), Conv1D(1, 1, 3, rng), Conv1D(1, 1, 3, rng)]
        self.lstm = LSTMCell(4, 4, rng)
        self.head = DenseMLP((4, 4, 1), rng)

    @staticmethod
    def features(bars):
        # Same causal features for upstream training and runtime inference.
        closes = [float(row[3]) for row in bars]
        result = []
        for i, row in enumerate(bars):
            window = closes[max(0, i - 13) : i + 1]
            changes = [b - a for a, b in zip(window, window[1:], strict=False)]
            gains = sum(max(0.0, v) for v in changes)
            losses = sum(max(0.0, -v) for v in changes)
            rsi = gains / (gains + losses) if gains + losses else 0.5
            mean = sum(window) / len(window)
            sd = math.sqrt(sum((v - mean) ** 2 for v in window) / len(window))
            result.append(
                list(map(float, row))
                + [
                    mean,
                    sum(closes[max(0, i - 4) : i + 1]) / min(i + 1, 5),
                    rsi,
                    closes[i] / window[0] - 1,
                    sd / mean,
                ]
            )
        return result

    def channels(self, rows):
        features = self.features(rows)
        return [[(row[j] - self.means[j]) / self.scales[j] for row in features] for j in range(10)]

    @staticmethod
    def activate(channels):
        return [[v.tanh() for v in channel] for channel in channels]

    def forward(self, channels):
        price = self.activate(self.encoders[0](channels[5:7]))
        nonprice = self.activate(self.encoders[1](channels[7:10]))
        groups = [channels[:5], price, nonprice]
        convolved = []
        for group, first, second in zip(groups, self.first, self.second, strict=False):
            first_output = self.activate(first(group))
            pooled = [
                [
                    max(channel[t : t + 2], key=lambda v: v.value)
                    for t in range(0, len(channel) - 1, 2)
                ]
                for channel in first_output
            ]
            convolved.extend(self.activate(second(pooled)))
        hidden = [Scalar(0.0) for _ in range(4)]
        cell = [Scalar(0.0) for _ in range(4)]
        for t in range(len(convolved[0])):
            hidden, cell = self.lstm([c[t] for c in convolved], hidden, cell)
        return self.head(hidden)[0].sigmoid()

    def parameters(self):
        parts = self.encoders + self.decoders + self.first + self.second + [self.lstm, self.head]
        return [p for layer in parts for p in layer.parameters()]

    def train(self, request):
        if len(request.samples) < 8:
            raise ValueError("Grouped CNN/LSTM needs eight labeled windows")
        self.initialize(request.seed)
        windows = [
            [[float(sample.features[f"v_{t}_{j}"]) for j in range(5)] for t in range(24)]
            for sample in request.samples
        ]
        labels = [float(sample.target) for sample in request.samples]
        if not set(labels) <= {0.0, 1.0}:
            raise ValueError("Binary next-close labels are required")
        features = [row for window in windows for row in self.features(window)]
        self.means = [sum(row[j] for row in features) / len(features) for j in range(10)]
        self.scales = [
            math.sqrt(sum((row[j] - self.means[j]) ** 2 for row in features) / len(features)) or 1.0
            for j in range(10)
        ]
        channels = [self.channels(window) for window in windows]
        before = [p.value for p in self.parameters()]
        ae_losses = []
        for _epoch in range(2):
            for window in channels:
                total = Scalar(0.0)
                for group, encoder, decoder in zip(
                    [window[5:7], window[7:10]], self.encoders, self.decoders, strict=False
                ):
                    reconstructed = decoder(self.activate(encoder(group)))
                    total += sum(
                        (
                            (pred - actual) ** 2
                            for predictions, actuals in zip(reconstructed, group, strict=False)
                            for pred, actual in zip(predictions, actuals, strict=False)
                        ),
                        Scalar(0.0),
                    ) / (len(group) * 24)
                total.backward()
                gradient_step(
                    [p for layer in self.encoders + self.decoders for p in layer.parameters()], 0.01
                )
                ae_losses.append(total.value)
        losses = []
        for _epoch in range(4):
            epoch_losses = []
            for window, label in zip(channels, labels, strict=False):
                probability = self.forward(window)
                loss = -(
                    label * (probability + 1e-9).log()
                    + (1 - label) * (1 - probability + 1e-9).log()
                )
                loss.backward()
                params = [
                    p
                    for layer in self.encoders + self.first + self.second + [self.lstm, self.head]
                    for p in layer.parameters()
                ]
                gradient_step(params, 0.025)
                epoch_losses.append(loss.value)
            losses.append(sum(epoch_losses) / len(epoch_losses))
        after = [p.value for p in self.parameters()]
        return json.dumps(
            {
                "seed": request.seed,
                "parameters": after,
                "means": self.means,
                "scales": self.scales,
                "warmup": windows[-1],
                "training": {
                    "samples": len(windows),
                    "supervised_epochs": 4,
                    "ae_epochs": 2,
                    "bce_by_epoch": losses,
                    "ae_first_loss": ae_losses[0],
                    "ae_last_loss": ae_losses[-1],
                    "parameter_change_l1": sum(
                        abs(a - b) for a, b in zip(after, before, strict=False)
                    ),
                    "paper_quality_proxy": 10 ** -losses[-1],
                    "paper_quality_threshold_met": 10 ** -losses[-1] >= 0.8,
                    "scope": (
                        "bounded engineering training; paper convergence criterion "
                        "is reported separately"
                    ),
                },
            },
            sort_keys=True,
        ).encode()

    def load(self, payload):
        state = json.loads(payload)
        self.initialize(state["seed"])
        parameters = self.parameters()
        if len(parameters) != len(state["parameters"]):
            raise ValueError("CNN/LSTM model architecture differs")
        for parameter, value in zip(parameters, state["parameters"], strict=False):
            parameter.value = float(value)
        self.means, self.scales = state["means"], state["scales"]
        self.history = []

    def decide(self, event, account):
        self.history = (
            self.history
            + [[float(event.values[k]) for k in ("open", "high", "low", "close", "volume")]]
        )[-24:]
        if len(self.history) < 24:
            return (NoOp(reason="CNN/LSTM 24-candle warmup"),)
        probability = self.forward(self.channels(self.history)).value
        upper, lower = 1 / (1 + self.theta), self.theta / (1 + self.theta)
        direction = 1 if probability >= upper else -1 if probability <= lower else 0
        quantity = (
            direction
            * abs(probability - 0.5)
            * float(account.equity)
            / float(event.values["close"])
        )
        return (
            TargetPosition(
                symbol=event.symbol,
                quantity=Decimal(f"{quantity:.8f}"),
                reason=f"CNN/LSTM p={probability:.8f}; Eq15 confidence and Eq bet sizing",
            ),
        )
