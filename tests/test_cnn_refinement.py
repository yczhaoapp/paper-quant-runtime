"""Independent causal indicators and deployed-width staged training checks."""

from __future__ import annotations

import json
import math
import random
from decimal import Decimal

import numpy as np
import pytest

from paperquant.models import TrainingRequest, TrainingSample
from tests.test_two_paper_adaptation import controlled_request, strategy


def cls():
    return strategy("grouped_cnn_lstm", "GroupedCNNLSTM")


def bars(seed, count=64):
    rng = random.Random(seed)
    price, result = 200.0, []
    for _ in range(count):
        opened = price
        price *= math.exp(rng.uniform(-0.02, 0.02))
        result.append(
            [
                opened,
                max(opened, price) * 1.01,
                min(opened, price) * 0.99,
                price,
                rng.uniform(100, 300),
            ]
        )
    return result


@pytest.mark.parametrize("seed", [7, 31, 79])
@pytest.mark.parametrize("mode", [1, 2])
def test_causal_indicator_features_match_independent_prefix_equations(seed, mode):
    raw = np.asarray(bars(seed))
    actual = cls().features(raw.tolist(), mode=mode)
    price, gain, loss = raw[:, 3], 0.0, 0.0
    expected = []
    for i in range(1, len(raw)):
        change = price[i] - price[i - 1]
        if i <= 14:
            gain += max(change, 0) / 14
            loss += max(-change, 0) / 14
        else:
            gain, loss = gain * 13 / 14 + max(change, 0) / 14, loss * 13 / 14 + max(-change, 0) / 14
        if i < 40:
            continue
        prefix = price[: i + 1]
        ema = []
        for period in (14, 28):
            alpha = 2 / (period + 1)
            ema.append(
                price[0] * (1 - alpha) ** i
                + sum(alpha * (1 - alpha) ** (i - j) * price[j] for j in range(1, i + 1))
            )
        if mode == 1:
            highest, lowest = raw[i - 13 : i + 1, 1].max(), raw[i - 13 : i + 1, 2].min()
            indicators = [
                ema[0] / price[40] - 1,
                ema[1] / price[40] - 1,
                gain / (gain + loss),
                (price[i] - lowest) / (highest - lowest),
                price[i] / price[i - 14] - 1,
            ]
        else:
            recent = prefix[-14:]
            indicators = [
                recent.mean() / price[40] - 1,
                prefix[-5:].mean() / price[40] - 1,
                gain / (gain + loss),
                price[i] / price[i - 14] - 1,
                recent.std() / recent.mean(),
            ]
        expected.append(
            list(raw[i, :4] / price[40] - 1) + [raw[i, 4] / raw[40:, 4].mean() - 1] + indicators
        )
    assert np.asarray(actual) == pytest.approx(np.asarray(expected), abs=1e-12)


@pytest.mark.parametrize("mode", [1, 2])
def test_causal_features_use_completed_prefix_and_reject_insufficient_history(mode):
    raw = bars(71, 96)
    prefix = cls().features(raw[:64], mode=mode)
    changed = raw[:64] + [[v * 1000 for v in row] for row in raw[64:]]
    assert cls().features(changed[:64], mode=mode) == prefix
    with pytest.raises(ValueError, match="64 completed"):
        cls().features(raw[:63], mode=mode)


@pytest.mark.parametrize("seed", [5, 19, 61])
@pytest.mark.parametrize("activation", [0, 1])
def test_frozen_encoder_cache_matches_uncached_prediction_and_training_gradients(seed, activation):
    model = cls()()
    model.initialize(seed, 4, activation=activation)
    rng = random.Random(seed + 100)
    channels = [[rng.uniform(-0.8, 0.8) for _ in range(24)] for _ in range(10)]
    groups = [
        channels[:5],
        model.conv_fast(model.encoders[0], channels[5:7]),
        model.conv_fast(model.encoders[1], channels[7:10]),
    ]
    left, loss, gradient = model.fast_prediction(channels, label=1)
    right, cached_loss, cached_gradient = model.fast_prediction(channels, label=1, groups=groups)
    assert right == pytest.approx(left, abs=1e-12)
    assert cached_loss == pytest.approx(loss, abs=1e-12)
    for layer in model.first + model.second + [model.lstm, model.head]:
        for p in layer.parameters():
            assert cached_gradient[p] == pytest.approx(gradient[p], abs=1e-12)
    for layer in model.encoders:
        assert all(cached_gradient[p] == 0 for p in layer.parameters())


def training_request(*, seed=13, count=32, epochs=2, mode=2):
    samples = []
    for index in range(count):
        raw = bars(seed + index)
        features = {
            f"v_{t}_{j}": Decimal(str(v)) for t, row in enumerate(raw) for j, v in enumerate(row)
        }
        features.update(
            partition=Decimal(index >= count * 0.75),
            window_end_ms=Decimal(index + 1),
            cfg_history_bars=Decimal(64),
            cfg_feature_mode=Decimal(mode),
            cfg_hidden=Decimal(4),
            cfg_max_epochs=Decimal(epochs),
            cfg_patience=Decimal(epochs),
            cfg_ae_epochs=Decimal(3),
            cfg_head_epochs=Decimal(2),
        )
        samples.append(TrainingSample(features=features, target=Decimal(index % 2)))
    return TrainingRequest(dataset_id="causal-staging", seed=seed, samples=tuple(samples))


def test_actual_hidden4_training_freezes_encoder_and_excludes_validation_from_fitting():
    request = training_request()
    modified = request.model_copy(
        update={
            "samples": tuple(
                sample.model_copy(
                    update={
                        "target": 1 - sample.target,
                        "features": {
                            k: v * 5 if k.startswith("v_") else v
                            for k, v in sample.features.items()
                        },
                    }
                )
                if sample.features["partition"]
                else sample
                for sample in request.samples
            )
        }
    )
    left, right = json.loads(cls()().train(request)), json.loads(cls()().train(modified))
    assert left["options"]["hidden"] == 4
    for name in ("means", "scales"):
        assert left[name] == right[name]
    for state in (left, right):
        trace = state["training"]
        assert (
            trace["encoder_parameters_after_pretraining"]
            == trace["encoder_parameters_after_supervised"]
        )
    assert (
        left["training"]["encoder_parameters_after_pretraining"]
        == right["training"]["encoder_parameters_after_pretraining"]
    )

    def joint(state):
        return [
            e["train"] for e in state["training"]["epoch_trace"] if e["stage"] == "cnn_lstm_head"
        ]

    assert joint(left) == joint(right)


@pytest.mark.parametrize("activation", [0, 1])
def test_head_refinement_gradients_match_independent_scalar_autodiff(activation):
    from paperquant.strategy_api import Scalar

    model = cls()()
    model.initialize(43, 4, activation=activation)
    hidden = [0.2, -0.4, 0.7, -0.1]
    p, gradient = model.head_prediction(hidden, label=0)
    values = [
        sum((w * x for w, x in zip(row, hidden, strict=True)), b)
        for row, b in zip(model.head.weights[0], model.head.biases[0], strict=True)
    ]
    values = [v.relu() if activation else v.tanh() for v in values]
    logit = sum(
        (w * x for w, x in zip(model.head.weights[1][0], values, strict=True)),
        model.head.biases[1][0],
    )
    probability = logit.sigmoid()
    loss = -(Scalar(1) - probability).log()
    loss.backward()
    assert p == pytest.approx(probability.value)
    for parameter in model.head.parameters():
        assert gradient[parameter] == pytest.approx(parameter.grad, abs=1e-12)


@pytest.mark.parametrize("seed", [5, 19, 61])
def test_cached_convolution_recurrent_stage_preserves_forward_and_recurrent_gradients(seed):
    model = cls()()
    model.initialize(seed, 4, activation=1)
    rng = random.Random(seed)
    channels = [[rng.uniform(-1, 1) for _ in range(24)] for _ in range(10)]
    sequence = model.convolution_sequence(channels)
    p, loss, grad = model.fast_prediction(channels, label=0)
    cp, closs, cgrad = model.fast_prediction(channels, sequence=sequence, label=0)
    assert cp == pytest.approx(p, abs=1e-12)
    assert closs == pytest.approx(loss, abs=1e-12)
    for part in (model.lstm, model.head):
        for param in part.parameters():
            assert cgrad[param] == pytest.approx(grad[param], abs=1e-12)
    for part in model.encoders + model.first + model.second:
        assert all(cgrad[p] == 0 for p in part.parameters())


@pytest.mark.parametrize("seed", [5, 19, 61])
def test_deployed_hidden4_relu_stages_learn_unseen_patterns_without_threshold_changes(seed):
    request = controlled_request(seed, hidden=4)
    options = {
        "activation": 1,
        "ae_epochs": 12,
        "max_epochs": 12,
        "patience": 6,
        "recurrent_epochs": 12,
        "head_epochs": 32,
        "learning_rate": 0.003,
        "batch_size": 32,
    }
    request = request.model_copy(
        update={
            "samples": tuple(
                sample.model_copy(
                    update={
                        "features": {
                            **sample.features,
                            **{"cfg_" + k: Decimal(str(v)) for k, v in options.items()},
                        }
                    }
                )
                for sample in request.samples
            )
        }
    )
    model = cls()()
    payload = model.train(request)
    training = json.loads(payload)["training"]
    val = training["selected_validation"]
    assert val["accuracy"] >= 0.9
    assert training["selected_train"]["bce"] < training["initial_train"]["bce"] * 0.65
    assert val["probability_min"] <= 1 / 3 or val["probability_max"] >= 2 / 3
    restored = cls()()
    restored.load(payload)
    for sample in request.samples[-12:]:
        raw = [[float(sample.features[f"v_{t}_{j}"]) for j in range(5)] for t in range(24)]
        assert restored.fast_prediction(restored.channels(raw)) == pytest.approx(
            model.forward(model.channels(raw)).value, abs=1e-12
        )


def test_holdout_downloader_requires_persisted_selection_before_network_access(
    tmp_path, monkeypatch
):
    from scripts import fetch_two_paper_data

    monkeypatch.setattr(fetch_two_paper_data, "ROOT", tmp_path)
    called = []
    monkeypatch.setattr(
        fetch_two_paper_data.urllib.request, "urlopen", lambda *args, **kwargs: called.append(args)
    )
    with pytest.raises(ValueError, match="persisted selection lock"):
        fetch_two_paper_data.prepare(cnn_holdout=True)
    assert not called
