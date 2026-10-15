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
        self.history_bars = 24
        self.feature_mode, self.activation = 0, 0

    def initialize(self, seed, hidden=8, *, feature_mode=0, activation=0):
        self.hidden = hidden
        self.feature_mode, self.activation = feature_mode, activation
        rng = random.Random(seed)
        self.encoders = [Conv1D(2, 1, 3, rng, padding=1), Conv1D(3, 1, 3, rng, padding=1)]
        self.decoders = [Conv1D(1, 2, 3, rng, padding=1), Conv1D(1, 3, 3, rng, padding=1)]
        self.first = [Conv1D(5, 2, 5, rng), Conv1D(1, 1, 5, rng), Conv1D(1, 1, 5, rng)]
        self.second = [Conv1D(2, 2, 3, rng), Conv1D(1, 1, 3, rng), Conv1D(1, 1, 3, rng)]
        self.lstm = LSTMCell(4, hidden, rng)
        self.head = DenseMLP((hidden, hidden, 1), rng)

    @staticmethod
    def features(bars, *, mode=0):
        if mode:
            return GroupedCNNLSTM.causal_features(bars, mode=mode)
        bars = bars[-24:]
        # Same causal features for upstream training and runtime inference.
        closes = [float(row[3]) for row in bars]
        base = closes[0]
        volume_scale = sum(float(row[4]) for row in bars) / len(bars)
        if base <= 0 or min(closes) <= 0 or volume_scale <= 0:
            raise ValueError("CNN windows require positive prices and average volume")
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
                [float(v) / base - 1 for v in row[:4]]
                + [float(row[4]) / volume_scale - 1]
                + [
                    mean / base - 1,
                    (sum(closes[max(0, i - 4) : i + 1]) / min(i + 1, 5)) / base - 1,
                    rsi,
                    closes[i] / window[0] - 1,
                    sd / mean,
                ]
            )
        return result

    @staticmethod
    def causal_features(bars, *, mode=1):
        if len(bars) < 64:
            raise ValueError("Causal CNN indicators require 64 completed bars")
        closes = [float(row[3]) for row in bars]
        if min(closes) <= 0:
            raise ValueError("Causal CNN prices must be positive")
        start = len(bars) - 24
        base = closes[start]
        volume_scale = sum(float(row[4]) for row in bars[start:]) / 24
        if volume_scale <= 0:
            raise ValueError("Causal CNN average volume must be positive")
        ema14, ema28 = closes[0], closes[0]
        gain, loss = 0.0, 0.0
        result = []
        for i, row in enumerate(bars):
            ema14 += 2 / 15 * (closes[i] - ema14)
            ema28 += 2 / 29 * (closes[i] - ema28)
            change = closes[i] - closes[i - 1] if i else 0.0
            if 1 <= i <= 14:
                gain += max(change, 0) / 14
                loss += max(-change, 0) / 14
            elif i > 14:
                gain = (13 * gain + max(change, 0)) / 14
                loss = (13 * loss + max(-change, 0)) / 14
            if i < start:
                continue
            high = max(float(r[1]) for r in bars[i - 13 : i + 1])
            low = min(float(r[2]) for r in bars[i - 13 : i + 1])
            rsi = gain / (gain + loss) if gain + loss else 0.5
            williams = (closes[i] - low) / (high - low) if high > low else 0.5
            roc = closes[i] / closes[i - 14] - 1
            if mode == 1:
                indicators = [ema14 / base - 1, ema28 / base - 1, rsi, williams, roc]
            else:
                window = closes[i - 13 : i + 1]
                mean = sum(window) / 14
                sd = math.sqrt(sum((v - mean) ** 2 for v in window) / 14)
                indicators = [
                    mean / base - 1,
                    sum(closes[i - 4 : i + 1]) / 5 / base - 1,
                    rsi,
                    roc,
                    sd / mean,
                ]
            result.append(
                [float(v) / base - 1 for v in row[:4]]
                + [float(row[4]) / volume_scale - 1]
                + indicators
            )
        return result

    def channels(self, rows):
        features = self.features(rows, mode=self.feature_mode)
        return [[(row[j] - self.means[j]) / self.scales[j] for row in features] for j in range(10)]

    @staticmethod
    def activate(channels, *, activation=0):
        return [[v.relu() if activation else v.tanh() for v in channel] for channel in channels]

    def forward(self, channels):
        price = self.activate(self.encoders[0](channels[5:7]))
        nonprice = self.activate(self.encoders[1](channels[7:10]))
        groups = [channels[:5], price, nonprice]
        convolved = []
        for group, first, second in zip(groups, self.first, self.second, strict=False):
            first_output = self.activate(first(group), activation=self.activation)
            pooled = [
                [
                    max(channel[t : t + 2], key=lambda v: v.value)
                    for t in range(0, len(channel) - 1, 2)
                ]
                for channel in first_output
            ]
            convolved.extend(self.activate(second(pooled), activation=self.activation))
        hidden = [Scalar(0.0) for _ in range(self.hidden)]
        cell = [Scalar(0.0) for _ in range(self.hidden)]
        for t in range(len(convolved[0])):
            hidden, cell = self.lstm([c[t] for c in convolved], hidden, cell)
        head_hidden = [
            sum((w * x for w, x in zip(row, hidden, strict=True)), b)
            for row, b in zip(self.head.weights[0], self.head.biases[0], strict=True)
        ]
        head_hidden = [v.relu() if self.activation else v.tanh() for v in head_hidden]
        return sum(
            (w * x for w, x in zip(self.head.weights[1][0], head_hidden, strict=True)),
            self.head.biases[1][0],
        ).sigmoid()

    def parameters(self):
        parts = self.encoders + self.decoders + self.first + self.second + [self.lstm, self.head]
        return [p for layer in parts for p in layer.parameters()]

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))

    @staticmethod
    def conv_fast(layer, inputs, *, activate=True, activation=0, prepared=None):
        length = len(inputs[0]) + 2 * layer.padding - layer.kernel + 1
        padded = [[0.0] * layer.padding + row + [0.0] * layer.padding for row in inputs]
        patches = [[x for row in padded for x in row[t : t + layer.kernel]] for t in range(length)]
        output = []
        weights_flat = (
            prepared
            if prepared is not None
            else [[w.value for row in group for w in row] for group in layer.weights]
        )
        for weights, bias in zip(weights_flat, layer.biases, strict=True):
            channel = []
            for t in range(length):
                value = bias.value + math.sumprod(weights, patches[t])
                channel.append(
                    (max(0.0, value) if activation else math.tanh(value)) if activate else value
                )
            output.append(channel)
        return output

    @staticmethod
    def conv_backward(
        layer, inputs, output, upstream, gradient, *, activate=True, activation=0,
        input_gradient=True,
    ):
        padded = [[0.0] * layer.padding + row + [0.0] * layer.padding for row in inputs]
        result = [[0.0] * len(row) for row in padded] if input_gradient else []
        for c, (weights, bias) in enumerate(zip(layer.weights, layer.biases, strict=True)):
            derivatives = [
                incoming * ((output[c][t] > 0) if activation else (1 - output[c][t] ** 2))
                if activate else incoming
                for t, incoming in enumerate(upstream[c])
            ]
            if not any(derivatives):
                continue
            for derivative in derivatives:
                gradient[bias] += derivative
            # Each weight gradient is a dot product over output positions.
            # Frozen upstream layers need parameter gradients, not input gradients.
            for row, input_row in zip(weights, padded, strict=True):
                for k, weight in enumerate(row):
                    gradient[weight] += math.sumprod(
                        derivatives, input_row[k : k + len(derivatives)]
                    )
            if input_gradient:
                for t, derivative in enumerate(derivatives):
                    if derivative == 0.0:
                        continue
                    for row, input_row in zip(weights, result, strict=True):
                        for k, weight in enumerate(row):
                            input_row[t + k] += derivative * weight.value
        return (
            [row[layer.padding : len(row) - layer.padding] for row in result]
            if layer.padding else result
        )

    def snapshot_weights(self):
        convs = self.encoders + self.decoders + self.first + self.second
        dense = self.lstm.gates + [self.head]
        snapshot = {
            "conv": {
                layer: [[w.value for row in group for w in row] for group in layer.weights]
                for layer in convs
            },
            "dense": {
                layer: [[[w.value for w in row] for row in weights] for weights in layer.weights]
                for layer in dense
            },
        }
        snapshot["dense_columns"] = {
            layer: list(zip(*values[0], strict=True))
            for layer, values in snapshot["dense"].items()
        }
        return snapshot

    def fast_prediction(
        self,
        channels,
        *,
        label=None,
        groups=None,
        prepared=None,
        return_hidden=False,
        sequence=None,
    ):
        """Same convolution/LSTM/head equations with analytic backpropagation.

        The Scalar path remains an independent executable autodiff reference for
        gradient checks. Both paths read exactly the same parameter objects.
        """
        prepared = prepared or self.snapshot_weights()
        frozen = groups is not None
        frozen_conv = sequence is not None
        if not frozen and not frozen_conv:
            encoded = [
                self.conv_fast(net, rows, prepared=prepared["conv"][net])
                for net, rows in zip(self.encoders, [channels[5:7], channels[7:10]], strict=True)
            ]
            groups = [channels[:5], *encoded]
        caches = []
        if sequence is None:
            sequence = []
        for group, first, second in (
            [] if frozen_conv else zip(groups, self.first, self.second, strict=True)
        ):
            first_out = self.conv_fast(
                first, group, activation=self.activation, prepared=prepared["conv"][first]
            )
            indices = [
                [max(range(t, t + 2), key=lambda i: row[i]) for t in range(0, len(row) - 1, 2)]
                for row in first_out
            ]
            pooled = [
                [row[i] for i in selected] for row, selected in zip(first_out, indices, strict=True)
            ]
            second_out = self.conv_fast(
                second, pooled, activation=self.activation, prepared=prepared["conv"][second]
            )
            sequence.extend(second_out)
            caches.append((group, first_out, indices, pooled, second_out))
        hidden, cell = [0.0] * self.hidden, [0.0] * self.hidden
        recurrent = []
        for t in range(len(sequence[0])):
            combined = hidden + [row[t] for row in sequence]
            gates = []
            for index, gate in enumerate(self.lstm.gates):
                values = [
                    math.sumprod(row, combined) + bias.value
                    for row, bias in zip(prepared["dense"][gate][0], gate.biases[0], strict=True)
                ]
                gates.append([math.tanh(v) if index == 0 else self.sigmoid(v) for v in values])
            candidate, update, forget, output = gates
            previous_cell = cell
            cell = [
                u * c + f * old
                for u, c, f, old in zip(update, candidate, forget, previous_cell, strict=True)
            ]
            hidden = [o * math.tanh(c) for o, c in zip(output, cell, strict=True)]
            if label is not None:
                recurrent.append((combined, previous_cell, cell, gates))
        if return_hidden:
            return hidden
        head_hidden = [
            math.sumprod(row, hidden) + b.value
            for row, b in zip(prepared["dense"][self.head][0], self.head.biases[0], strict=True)
        ]
        head_hidden = [max(0.0, v) if self.activation else math.tanh(v) for v in head_hidden]
        logit = math.sumprod(prepared["dense"][self.head][1][0], head_hidden)
        logit += self.head.biases[1][0].value
        probability = self.sigmoid(logit)
        if label is None:
            return probability
        loss = max(logit, 0) - label * logit + math.log1p(math.exp(-abs(logit)))
        gradient = {p: 0.0 for p in self.parameters()}
        derivative = probability - label
        gradient[self.head.biases[1][0]] += derivative
        for weight, value in zip(self.head.weights[1][0], head_hidden, strict=True):
            gradient[weight] += derivative * value
        dh = [0.0] * self.hidden
        for i, (weights, bias) in enumerate(
            zip(self.head.weights[0], self.head.biases[0], strict=True)
        ):
            dz = (
                derivative
                * self.head.weights[1][0][i].value
                * ((head_hidden[i] > 0) if self.activation else (1 - head_hidden[i] ** 2))
            )
            gradient[bias] += dz
            for j, weight in enumerate(weights):
                gradient[weight] += dz * hidden[j]
                dh[j] += dz * weight.value
        if not any(dh):
            # A zero upstream derivative leaves every trunk gradient at zero.
            return probability, loss, gradient
        dc = [0.0] * self.hidden
        dsequence = [] if frozen_conv else [[0.0] * len(row) for row in sequence]
        # Parameter gradients can be reduced over time after the recurrence
        # gradients are known; parameter values stay fixed throughout this pass.
        reverse_columns = list(zip(*(row[0] for row in reversed(recurrent)), strict=True))
        gate_signals = [[] for _ in self.lstm.gates]
        for t in reversed(range(len(recurrent))):
            combined, previous_cell, current_cell, gates = recurrent[t]
            candidate, update, forget, output = gates
            tanh_cell = [math.tanh(c) for c in current_cell]
            cell_gradient = [
                old + h * o * (1 - c * c)
                for old, h, o, c in zip(dc, dh, output, tanh_cell, strict=True)
            ]
            gate_gradients = [
                [
                    g * u * (1 - c * c)
                    for g, u, c in zip(cell_gradient, update, candidate, strict=True)
                ],
                [
                    g * c * u * (1 - u)
                    for g, c, u in zip(cell_gradient, candidate, update, strict=True)
                ],
                [
                    g * old * f * (1 - f)
                    for g, old, f in zip(cell_gradient, previous_cell, forget, strict=True)
                ],
                [h * c * o * (1 - o) for h, c, o in zip(dh, tanh_cell, output, strict=True)],
            ]
            dcombined = [0.0] * (self.hidden if frozen_conv else len(combined))
            for gate, dg, signals in zip(
                self.lstm.gates, gate_gradients, gate_signals, strict=True
            ):
                signals.append(dg)
                for j in range(len(dcombined)):
                    dcombined[j] += math.sumprod(dg, prepared["dense_columns"][gate][j])
            dh = dcombined[: self.hidden]
            dc = [g * f for g, f in zip(cell_gradient, forget, strict=True)]
            for row, value in zip(dsequence, dcombined[self.hidden :], strict=True):
                row[t] = value
        for gate, signals in zip(self.lstm.gates, gate_signals, strict=True):
            derivatives = list(zip(*signals, strict=True))
            for weights, bias, values in zip(
                gate.weights[0], gate.biases[0], derivatives, strict=True
            ):
                for value in values:
                    gradient[bias] += value
                for weight, column in zip(weights, reverse_columns, strict=True):
                    gradient[weight] += math.sumprod(values, column)
        offset = 0
        for index, (group, first_out, indices, pooled, second_out) in enumerate(caches):
            width = len(second_out)
            dpool = self.conv_backward(
                self.second[index],
                pooled,
                second_out,
                dsequence[offset : offset + width],
                gradient,
                activation=self.activation,
            )
            offset += width
            dfirst = [[0.0] * len(row) for row in first_out]
            for row, selected, upstream in zip(dfirst, indices, dpool, strict=True):
                for i, value in zip(selected, upstream, strict=True):
                    row[i] += value
            dgroup = self.conv_backward(
                self.first[index], group, first_out, dfirst, gradient, activation=self.activation,
                input_gradient=bool(index and not frozen),
            )
            if index and not frozen:
                self.conv_backward(
                    self.encoders[index - 1],
                    channels[5:7] if index == 1 else channels[7:10],
                    encoded[index - 1],
                    dgroup,
                    gradient,
                    input_gradient=False,
                )
        return probability, loss, gradient

    def convolution_sequence(self, channels):
        encoded = [
            self.conv_fast(net, group)
            for net, group in zip(self.encoders, [channels[5:7], channels[7:10]], strict=True)
        ]
        sequence = []
        for group, first, second in zip(
            [channels[:5], *encoded], self.first, self.second, strict=True
        ):
            output = self.conv_fast(first, group, activation=self.activation)
            pooled = [[max(row[t : t + 2]) for t in range(0, len(row) - 1, 2)] for row in output]
            sequence.extend(self.conv_fast(second, pooled, activation=self.activation))
        return sequence

    @staticmethod
    def adam_step(parameters, gradient, optimizer, rate):
        optimizer["step"] += 1
        t = optimizer["step"]
        norm = math.sqrt(sum(gradient[p] ** 2 for p in parameters))
        clip = min(1.0, 5.0 / max(norm, 1e-12))
        for p in parameters:
            g = gradient[p] * clip
            m, v = optimizer["moments"].get(p, (0.0, 0.0))
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            optimizer["moments"][p] = (m, v)
            p.value -= rate * (m / (1 - 0.9**t)) / (math.sqrt(v / (1 - 0.999**t)) + 1e-8)
            if not math.isfinite(p.value):
                raise ValueError("CNN optimization produced a nonfinite parameter")

    def head_prediction(self, hidden, *, label=None):
        values = [
            sum(w.value * x for w, x in zip(row, hidden, strict=True)) + b.value
            for row, b in zip(self.head.weights[0], self.head.biases[0], strict=True)
        ]
        values = [max(0.0, v) if self.activation else math.tanh(v) for v in values]
        logit = sum(w.value * x for w, x in zip(self.head.weights[1][0], values, strict=True))
        logit += self.head.biases[1][0].value
        probability = self.sigmoid(logit)
        if label is None:
            return probability
        gradient = {p: 0.0 for p in self.head.parameters()}
        derivative = probability - label
        gradient[self.head.biases[1][0]] = derivative
        for weight, value in zip(self.head.weights[1][0], values, strict=True):
            gradient[weight] = derivative * value
        for i, (weights, bias) in enumerate(
            zip(self.head.weights[0], self.head.biases[0], strict=True)
        ):
            dz = (
                derivative
                * self.head.weights[1][0][i].value
                * ((values[i] > 0) if self.activation else (1 - values[i] ** 2))
            )
            gradient[bias] = dz
            for weight, value in zip(weights, hidden, strict=True):
                gradient[weight] = dz * value
        return probability, gradient

    @staticmethod
    def metrics(probabilities, labels):
        clipped = [min(1 - 1e-12, max(1e-12, p)) for p in probabilities]
        return {
            "samples": len(labels),
            "bce": -sum(
                y * math.log(p) + (1 - y) * math.log(1 - p)
                for p, y in zip(clipped, labels, strict=True)
            )
            / len(labels),
            "accuracy": sum((p >= 0.5) == y for p, y in zip(probabilities, labels, strict=True))
            / len(labels),
            "majority_accuracy": max(labels.count(0), labels.count(1)) / len(labels),
            "probability_min": min(probabilities),
            "probability_max": max(probabilities),
            "label_counts": {str(i): labels.count(i) for i in (0, 1)},
        }

    def train(self, request):
        if not request.samples:
            raise ValueError("CNN training request is empty")
        defaults = {
            "max_epochs": 8,
            "ae_epochs": 2,
            "batch_size": 32,
            "patience": 4,
            "hidden": 8,
            "learning_rate": 0.003,
            "history_bars": 24,
            "feature_mode": 0,
            "activation": 0,
            "freeze_encoders": 1,
            "head_epochs": 0,
            "recurrent_epochs": 0,
            "scaler": 0,
        }
        options = {
            key: float(request.samples[0].features.get("cfg_" + key, value))
            for key, value in defaults.items()
        }
        for key in defaults.keys() - {"learning_rate"}:
            minimum = (
                0
                if key
                in {
                    "feature_mode",
                    "activation",
                    "freeze_encoders",
                    "head_epochs",
                    "recurrent_epochs",
                    "scaler",
                }
                else 1
            )
            if options[key] != int(options[key]) or not minimum <= options[key] <= 128:
                raise ValueError("Invalid integer training setting: " + key)
            options[key] = int(options[key])
        if not 0.00001 <= options["learning_rate"] <= 0.1 or options["hidden"] not in (4, 8, 12):
            raise ValueError("Invalid CNN optimizer or width setting")
        if (
            options["feature_mode"] not in (0, 1, 2)
            or options["activation"] not in (0, 1)
            or options["freeze_encoders"] not in (0, 1)
            or options["scaler"] not in (0, 1)
            or options["history_bars"] not in (24, 64)
            or options["feature_mode"]
            and options["history_bars"] != 64
        ):
            raise ValueError("Invalid CNN feature/staging setting")
        if any(
            float(s.features.get("cfg_" + key, default)) != options[key]
            for s in request.samples
            for key, default in defaults.items()
        ):
            raise ValueError("CNN settings must be identical across the training request")
        self.initialize(
            request.seed,
            options["hidden"],
            feature_mode=options["feature_mode"],
            activation=options["activation"],
        )
        self.history_bars = options["history_bars"]
        windows = [
            [
                [float(sample.features[f"v_{t}_{j}"]) for j in range(5)]
                for t in range(self.history_bars)
            ]
            for sample in request.samples
        ]
        labels = [float(sample.target) for sample in request.samples]
        if not set(labels) <= {0.0, 1.0}:
            raise ValueError("Binary next-close labels are required")
        partitions = [float(s.features.get("partition", 0)) for s in request.samples]
        if not set(partitions) <= {0, 1} or partitions != sorted(partitions):
            raise ValueError("Training then chronological validation partitions are required")
        train_indices = [i for i, split in enumerate(partitions) if split == 0]
        val_indices = [i for i, split in enumerate(partitions) if split == 1]
        if len(train_indices) < 8 or len(val_indices) < 4:
            raise ValueError("CNN requires eight training and four validation windows")
        stamps = [float(s.features["window_end_ms"]) for s in request.samples]
        if any(b <= a for a, b in zip(stamps, stamps[1:], strict=False)):
            raise ValueError("CNN window end times must strictly increase")
        features = [
            row for i in train_indices for row in self.features(windows[i], mode=self.feature_mode)
        ]
        if options["scaler"]:
            # Author-inspired train-only MinMax normalization. Validation/test
            # values may lie outside [0,1]; never refit or clip those partitions.
            self.means = [min(row[j] for row in features) for j in range(10)]
            self.scales = [
                max(row[j] for row in features) - self.means[j] or 1.0 for j in range(10)
            ]
        else:
            self.means = [sum(row[j] for row in features) / len(features) for j in range(10)]
            self.scales = [
                math.sqrt(sum((row[j] - self.means[j]) ** 2 for row in features) / len(features))
                or 1.0
                for j in range(10)
            ]
        channels = [self.channels(window) for window in windows]
        before = [p.value for p in self.parameters()]
        rng = random.Random(request.seed)
        ae_optimizer = {"step": 0, "moments": {}}
        ae_parameters = [p for layer in self.encoders + self.decoders for p in layer.parameters()]
        ae_losses = []
        for _epoch in range(options["ae_epochs"]):
            order = list(train_indices)
            rng.shuffle(order)
            total_loss = 0.0
            for start in range(0, len(order), options["batch_size"]):
                batch = order[start : start + options["batch_size"]]
                gradient = {p: 0.0 for p in ae_parameters}
                prepared = self.snapshot_weights()
                for i in batch:
                    for group, encoder, decoder in zip(
                        [channels[i][5:7], channels[i][7:10]],
                        self.encoders,
                        self.decoders,
                        strict=True,
                    ):
                        encoded = self.conv_fast(encoder, group, prepared=prepared["conv"][encoder])
                        decoded = self.conv_fast(
                            decoder, encoded, activate=False, prepared=prepared["conv"][decoder]
                        )
                        scale = len(group) * 24
                        errors = [
                            [pred - actual for pred, actual in zip(a, b, strict=True)]
                            for a, b in zip(decoded, group, strict=True)
                        ]
                        total_loss += sum(v * v for row in errors for v in row) / scale
                        upstream = [[2 * v / scale / len(batch) for v in row] for row in errors]
                        dencoded = self.conv_backward(
                            decoder, encoded, decoded, upstream, gradient, activate=False
                        )
                        self.conv_backward(
                            encoder, group, encoded, dencoded, gradient, input_gradient=False
                        )
                self.adam_step(ae_parameters, gradient, ae_optimizer, options["learning_rate"])
            ae_losses.append(total_loss / len(order))
        train_labels = [labels[i] for i in train_indices]
        val_labels = [labels[i] for i in val_indices]
        encoder_after_pretraining = [p.value for net in self.encoders for p in net.parameters()]
        frozen_groups = None
        if options["freeze_encoders"]:
            frozen_groups = [
                [
                    row[:5],
                    self.conv_fast(self.encoders[0], row[5:7]),
                    self.conv_fast(self.encoders[1], row[7:10]),
                ]
                for row in channels
            ]

        def score(indices, targets):
            prepared = self.snapshot_weights()
            return self.metrics(
                [
                    self.fast_prediction(
                        channels[i],
                        groups=frozen_groups[i] if frozen_groups else None,
                        prepared=prepared,
                    )
                    for i in indices
                ],
                targets,
            )

        initial_train, initial_validation = (
            score(train_indices, train_labels),
            score(val_indices, val_labels),
        )
        best_loss, best_epoch = initial_validation["bce"], 0
        best = [p.value for p in self.parameters()]
        patience = 0
        optimizer = {"step": 0, "moments": {}}
        parameters = [
            p
            for layer in (
                ([] if frozen_groups else self.encoders)
                + self.first
                + self.second
                + [self.lstm, self.head]
            )
            for p in layer.parameters()
        ]
        trace = []
        for epoch in range(1, options["max_epochs"] + 1):
            order = list(train_indices)
            rng.shuffle(order)
            epoch_probabilities, epoch_labels = [], []
            for start in range(0, len(order), options["batch_size"]):
                batch = order[start : start + options["batch_size"]]
                accumulated = {p: 0.0 for p in parameters}
                prepared = self.snapshot_weights()
                for i in batch:
                    probability, _, gradient = self.fast_prediction(
                        channels[i],
                        label=labels[i],
                        prepared=prepared,
                        groups=frozen_groups[i] if frozen_groups else None,
                    )
                    epoch_probabilities.append(probability)
                    epoch_labels.append(labels[i])
                    for p in parameters:
                        accumulated[p] += gradient[p] / len(batch)
                self.adam_step(parameters, accumulated, optimizer, options["learning_rate"])
            train_score = self.metrics(epoch_probabilities, epoch_labels)
            val_score = score(val_indices, val_labels)
            trace.append(
                {
                    "epoch": epoch,
                    "stage": "cnn_lstm_head",
                    "train": train_score,
                    "validation": val_score,
                    "train_scope": "online pre-update minibatch probabilities",
                }
            )
            if val_score["bce"] < best_loss - 1e-6:
                best_loss, best_epoch, patience = val_score["bce"], epoch, 0
                best = [p.value for p in self.parameters()]
            else:
                patience += 1
            if patience >= options["patience"]:
                break
        joint_epochs = len(trace)
        # Start the optional final stage from the best validation checkpoint.
        for p, value in zip(self.parameters(), best, strict=True):
            p.value = value
        recurrent_epochs = 0
        if options["recurrent_epochs"]:
            sequences = [self.convolution_sequence(row) for row in channels]
            recurrent_parameters = [
                p for layer in [self.lstm, self.head] for p in layer.parameters()
            ]
            optimizer = {"step": 0, "moments": {}}
            patience = 0
            for recurrent_epoch in range(1, options["recurrent_epochs"] + 1):
                order = list(train_indices)
                rng.shuffle(order)
                epoch_probabilities, epoch_labels = [], []
                for start in range(0, len(order), options["batch_size"]):
                    batch = order[start : start + options["batch_size"]]
                    gradient = {p: 0.0 for p in recurrent_parameters}
                    prepared = self.snapshot_weights()
                    for i in batch:
                        probability, _, sample_gradient = self.fast_prediction(
                            channels[i], label=labels[i], sequence=sequences[i], prepared=prepared
                        )
                        epoch_probabilities.append(probability)
                        epoch_labels.append(labels[i])
                        for p in recurrent_parameters:
                            gradient[p] += sample_gradient[p] / len(batch)
                    self.adam_step(
                        recurrent_parameters, gradient, optimizer, options["learning_rate"]
                    )
                prepared = self.snapshot_weights()
                val_score = self.metrics(
                    [
                        self.fast_prediction(channels[i], sequence=sequences[i], prepared=prepared)
                        for i in val_indices
                    ],
                    val_labels,
                )
                recurrent_epochs = recurrent_epoch
                epoch = joint_epochs + recurrent_epoch
                trace.append(
                    {
                        "epoch": epoch,
                        "stage": "frozen_conv_lstm_head",
                        "train": self.metrics(epoch_probabilities, epoch_labels),
                        "validation": val_score,
                        "train_scope": "online pre-update minibatch probabilities",
                    }
                )
                if val_score["bce"] < best_loss - 1e-6:
                    best_loss, best_epoch, patience = val_score["bce"], epoch, 0
                    best = [p.value for p in self.parameters()]
                else:
                    patience += 1
                if patience >= options["patience"]:
                    break
            for p, value in zip(self.parameters(), best, strict=True):
                p.value = value
        head_epochs = 0
        if options["head_epochs"]:
            hidden_vectors = [self.fast_prediction(row, return_hidden=True) for row in channels]
            head_parameters = self.head.parameters()
            optimizer = {"step": 0, "moments": {}}
            patience = 0
            for head_epoch in range(1, options["head_epochs"] + 1):
                order = list(train_indices)
                rng.shuffle(order)
                epoch_probabilities, epoch_labels = [], []
                for start in range(0, len(order), options["batch_size"]):
                    batch = order[start : start + options["batch_size"]]
                    gradient = {p: 0.0 for p in head_parameters}
                    for i in batch:
                        probability, sample_gradient = self.head_prediction(
                            hidden_vectors[i], label=labels[i]
                        )
                        epoch_probabilities.append(probability)
                        epoch_labels.append(labels[i])
                        for p in head_parameters:
                            gradient[p] += sample_gradient[p] / len(batch)
                    self.adam_step(head_parameters, gradient, optimizer, options["learning_rate"])
                val_score = self.metrics(
                    [self.head_prediction(hidden_vectors[i]) for i in val_indices], val_labels
                )
                head_epochs = head_epoch
                epoch = joint_epochs + recurrent_epochs + head_epoch
                trace.append(
                    {
                        "epoch": epoch,
                        "stage": "frozen_trunk_head",
                        "train": self.metrics(epoch_probabilities, epoch_labels),
                        "validation": val_score,
                        "train_scope": "online pre-update minibatch probabilities",
                    }
                )
                if val_score["bce"] < best_loss - 1e-6:
                    best_loss, best_epoch, patience = val_score["bce"], epoch, 0
                    best = [p.value for p in self.parameters()]
                else:
                    patience += 1
                if patience >= options["patience"]:
                    break
            for p, value in zip(self.parameters(), best, strict=True):
                p.value = value
        selected_train, selected_validation = (
            score(train_indices, train_labels),
            score(val_indices, val_labels),
        )
        after = [p.value for p in self.parameters()]
        return json.dumps(
            {
                "seed": request.seed,
                "model_format": 3,
                "options": options,
                "parameters": after,
                "means": self.means,
                "scales": self.scales,
                "training": {
                    "samples": len(train_indices),
                    "validation_samples": len(val_indices),
                    "supervised_epochs": len(trace),
                    "joint_epochs": joint_epochs,
                    "head_epochs": head_epochs,
                    "recurrent_epochs": recurrent_epochs,
                    "encoder_frozen": bool(options["freeze_encoders"]),
                    "encoder_parameters_after_pretraining": encoder_after_pretraining,
                    "encoder_parameters_after_supervised": [
                        p.value for net in self.encoders for p in net.parameters()
                    ],
                    "selected_epoch": best_epoch,
                    "options": options,
                    "optimizer": "minibatch Adam with global gradient norm clip5",
                    "scaler": "training-only MinMax"
                    if options["scaler"]
                    else "training-only standard",
                    "checkpoint_selection": "minimum chronological validation BCE",
                    "initial_train": initial_train,
                    "initial_validation": initial_validation,
                    "selected_train": selected_train,
                    "selected_validation": selected_validation,
                    "epoch_trace": trace,
                    "ae_epochs": options["ae_epochs"],
                    "bce_by_epoch": [epoch["train"]["bce"] for epoch in trace],
                    "ae_first_loss": ae_losses[0],
                    "ae_last_loss": ae_losses[-1],
                    "ae_loss_by_epoch": ae_losses,
                    "parameter_change_l1": sum(
                        abs(a - b) for a, b in zip(after, before, strict=False)
                    ),
                    "paper_quality_proxy": 10 ** -selected_train["bce"],
                    "paper_quality_threshold_met": 10 ** -selected_train["bce"] >= 0.8,
                    "scope": (
                        "chronological validation-selected local method adaptation; "
                        "printed paper quality proxy reported separately"
                    ),
                },
            },
            sort_keys=True,
        ).encode()

    def load(self, payload):
        state = json.loads(payload)
        if state.get("model_format") != 3:
            raise ValueError("CNN model format differs from this strategy implementation")
        self.initialize(
            state["seed"],
            state["options"]["hidden"],
            feature_mode=state["options"]["feature_mode"],
            activation=state["options"]["activation"],
        )
        self.history_bars = state["options"]["history_bars"]
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
        )[-self.history_bars :]
        if len(self.history) < self.history_bars:
            return (NoOp(reason="CNN/LSTM causal indicator warmup"),)
        probability = self.fast_prediction(self.channels(self.history))
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
