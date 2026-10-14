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

    def initialize(self, seed, hidden=8):
        self.hidden = hidden
        rng = random.Random(seed)
        self.encoders = [Conv1D(2, 1, 3, rng, padding=1), Conv1D(3, 1, 3, rng, padding=1)]
        self.decoders = [Conv1D(1, 2, 3, rng, padding=1), Conv1D(1, 3, 3, rng, padding=1)]
        self.first = [Conv1D(5, 2, 5, rng), Conv1D(1, 1, 5, rng), Conv1D(1, 1, 5, rng)]
        self.second = [Conv1D(2, 2, 3, rng), Conv1D(1, 1, 3, rng), Conv1D(1, 1, 3, rng)]
        self.lstm = LSTMCell(4, hidden, rng)
        self.head = DenseMLP((hidden, hidden, 1), rng)

    @staticmethod
    def features(bars):
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
        hidden = [Scalar(0.0) for _ in range(self.hidden)]
        cell = [Scalar(0.0) for _ in range(self.hidden)]
        for t in range(len(convolved[0])):
            hidden, cell = self.lstm([c[t] for c in convolved], hidden, cell)
        return self.head(hidden)[0].sigmoid()

    def parameters(self):
        parts = self.encoders + self.decoders + self.first + self.second + [self.lstm, self.head]
        return [p for layer in parts for p in layer.parameters()]

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))

    @staticmethod
    def conv_fast(layer, inputs, *, activate=True):
        length = len(inputs[0]) + 2 * layer.padding - layer.kernel + 1
        padded = [[0.0] * layer.padding + row + [0.0] * layer.padding for row in inputs]
        output = []
        for weights, bias in zip(layer.weights, layer.biases, strict=True):
            channel = []
            for t in range(length):
                value = bias.value
                for row, input_row in zip(weights, padded, strict=True):
                    for k, weight in enumerate(row):
                        value += weight.value * input_row[t + k]
                channel.append(math.tanh(value) if activate else value)
            output.append(channel)
        return output

    @staticmethod
    def conv_backward(layer, inputs, output, upstream, gradient, *, activate=True):
        padded = [[0.0] * layer.padding + row + [0.0] * layer.padding for row in inputs]
        result = [[0.0] * len(row) for row in padded]
        for c, (weights, bias) in enumerate(zip(layer.weights, layer.biases, strict=True)):
            for t, incoming in enumerate(upstream[c]):
                derivative = incoming * (1 - output[c][t] ** 2) if activate else incoming
                gradient[bias] += derivative
                for row, input_row, input_gradient in zip(weights, padded, result, strict=True):
                    for k, weight in enumerate(row):
                        gradient[weight] += derivative * input_row[t + k]
                        input_gradient[t + k] += derivative * weight.value
        return (
            [row[layer.padding : len(row) - layer.padding] for row in result]
            if layer.padding
            else result
        )

    def fast_prediction(self, channels, *, label=None):
        """Same convolution/LSTM/head equations with analytic backpropagation.

        The Scalar path remains an independent executable autodiff reference for
        gradient checks. Both paths read exactly the same parameter objects.
        """
        encoded = [
            self.conv_fast(net, rows)
            for net, rows in zip(self.encoders, [channels[5:7], channels[7:10]], strict=True)
        ]
        groups = [channels[:5], *encoded]
        caches, sequence = [], []
        for group, first, second in zip(groups, self.first, self.second, strict=True):
            first_out = self.conv_fast(first, group)
            indices = [
                [max(range(t, t + 2), key=lambda i: row[i]) for t in range(0, len(row) - 1, 2)]
                for row in first_out
            ]
            pooled = [
                [row[i] for i in selected] for row, selected in zip(first_out, indices, strict=True)
            ]
            second_out = self.conv_fast(second, pooled)
            sequence.extend(second_out)
            caches.append((group, first_out, indices, pooled, second_out))
        hidden, cell = [0.0] * self.hidden, [0.0] * self.hidden
        recurrent = []
        for t in range(len(sequence[0])):
            combined = hidden + [row[t] for row in sequence]
            gates = []
            for index, gate in enumerate(self.lstm.gates):
                values = [
                    sum(w.value * x for w, x in zip(row, combined, strict=True)) + bias.value
                    for row, bias in zip(gate.weights[0], gate.biases[0], strict=True)
                ]
                gates.append([math.tanh(v) if index == 0 else self.sigmoid(v) for v in values])
            candidate, update, forget, output = gates
            previous_cell = cell
            cell = [
                u * c + f * old
                for u, c, f, old in zip(update, candidate, forget, previous_cell, strict=True)
            ]
            hidden = [o * math.tanh(c) for o, c in zip(output, cell, strict=True)]
            recurrent.append((combined, previous_cell, cell, gates))
        head_hidden = [
            math.tanh(sum(w.value * x for w, x in zip(row, hidden, strict=True)) + b.value)
            for row, b in zip(self.head.weights[0], self.head.biases[0], strict=True)
        ]
        logit = sum(w.value * x for w, x in zip(self.head.weights[1][0], head_hidden, strict=True))
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
            dz = derivative * self.head.weights[1][0][i].value * (1 - head_hidden[i] ** 2)
            gradient[bias] += dz
            for j, weight in enumerate(weights):
                gradient[weight] += dz * hidden[j]
                dh[j] += dz * weight.value
        dc = [0.0] * self.hidden
        dsequence = [[0.0] * len(row) for row in sequence]
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
            dcombined = [0.0] * len(combined)
            for gate, dg in zip(self.lstm.gates, gate_gradients, strict=True):
                for i, (weights, bias) in enumerate(
                    zip(gate.weights[0], gate.biases[0], strict=True)
                ):
                    gradient[bias] += dg[i]
                    for j, weight in enumerate(weights):
                        gradient[weight] += dg[i] * combined[j]
                        dcombined[j] += dg[i] * weight.value
            dh = dcombined[: self.hidden]
            dc = [g * f for g, f in zip(cell_gradient, forget, strict=True)]
            for row, value in zip(dsequence, dcombined[self.hidden :], strict=True):
                row[t] = value
        offset = 0
        for index, (group, first_out, indices, pooled, second_out) in enumerate(caches):
            width = len(second_out)
            dpool = self.conv_backward(
                self.second[index], pooled, second_out, dsequence[offset : offset + width], gradient
            )
            offset += width
            dfirst = [[0.0] * len(row) for row in first_out]
            for row, selected, upstream in zip(dfirst, indices, dpool, strict=True):
                for i, value in zip(selected, upstream, strict=True):
                    row[i] += value
            dgroup = self.conv_backward(self.first[index], group, first_out, dfirst, gradient)
            if index:
                self.conv_backward(
                    self.encoders[index - 1],
                    channels[5:7] if index == 1 else channels[7:10],
                    encoded[index - 1],
                    dgroup,
                    gradient,
                )
        return probability, loss, gradient

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
        }
        options = {
            key: float(request.samples[0].features.get("cfg_" + key, value))
            for key, value in defaults.items()
        }
        for key in defaults.keys() - {"learning_rate"}:
            if options[key] != int(options[key]) or not 1 <= options[key] <= 128:
                raise ValueError("Invalid integer training setting: " + key)
            options[key] = int(options[key])
        if not 0.00001 <= options["learning_rate"] <= 0.1 or options["hidden"] not in (4, 8, 12):
            raise ValueError("Invalid CNN optimizer or width setting")
        if any(
            float(s.features.get("cfg_" + key, default)) != options[key]
            for s in request.samples
            for key, default in defaults.items()
        ):
            raise ValueError("CNN settings must be identical across the training request")
        self.initialize(request.seed, options["hidden"])
        windows = [
            [[float(sample.features[f"v_{t}_{j}"]) for j in range(5)] for t in range(24)]
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
        features = [row for i in train_indices for row in self.features(windows[i])]
        self.means = [sum(row[j] for row in features) / len(features) for j in range(10)]
        self.scales = [
            math.sqrt(sum((row[j] - self.means[j]) ** 2 for row in features) / len(features)) or 1.0
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
                for i in batch:
                    for group, encoder, decoder in zip(
                        [channels[i][5:7], channels[i][7:10]],
                        self.encoders,
                        self.decoders,
                        strict=True,
                    ):
                        encoded = self.conv_fast(encoder, group)
                        decoded = self.conv_fast(decoder, encoded, activate=False)
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
                        self.conv_backward(encoder, group, encoded, dencoded, gradient)
                self.adam_step(ae_parameters, gradient, ae_optimizer, options["learning_rate"])
            ae_losses.append(total_loss / len(order))
        train_labels = [labels[i] for i in train_indices]
        val_labels = [labels[i] for i in val_indices]

        def score(indices, targets):
            return self.metrics([self.fast_prediction(channels[i]) for i in indices], targets)

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
            for layer in self.encoders + self.first + self.second + [self.lstm, self.head]
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
                for i in batch:
                    probability, _, gradient = self.fast_prediction(channels[i], label=labels[i])
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
                "model_format": 2,
                "options": options,
                "parameters": after,
                "means": self.means,
                "scales": self.scales,
                "training": {
                    "samples": len(train_indices),
                    "validation_samples": len(val_indices),
                    "supervised_epochs": len(trace),
                    "selected_epoch": best_epoch,
                    "options": options,
                    "optimizer": "minibatch Adam with global gradient norm clip5",
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
        if state.get("model_format") != 2:
            raise ValueError("CNN model format differs from this strategy implementation")
        self.initialize(state["seed"], state["options"]["hidden"])
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
