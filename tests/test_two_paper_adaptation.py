"""Independent gradients, learnability, and chronological partition safeguards."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
from decimal import Decimal
from pathlib import Path

import pytest

from paperquant.models import TrainingRequest, TrainingSample

ROOT = Path(__file__).resolve().parents[1]


def test_contract_lock_distinguishes_unchanged_interfaces_from_execution_policy():
    lock = json.loads((ROOT / "research/two-paper/contract-lock.json").read_text())
    actual = {
        p.relative_to(ROOT).as_posix()
        for directory in ("src", "schemas")
        for p in (ROOT / directory).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    assert actual == set(lock["files"])
    revisions = lock.get("execution_policy_revisions", {})
    assert set(revisions) == {"src/paperquant/sandbox.py"}
    for name, sha in lock["files"].items():
        current = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if name in revisions:
            assert current == revisions[name]["current_sha256"]
            assert current != sha
        else:
            assert current == sha
    from scripts.two_paper_quality import check_contract_lock

    result = check_contract_lock()
    assert result["byte_identical_files"] == len(actual) - 1
    assert result["schema_and_lifecycle"] == "unchanged"


def strategy(name, cls):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"strategies/supplement.{name}/strategy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, cls)


@pytest.mark.parametrize("seed", [3, 17, 43])
@pytest.mark.parametrize("label", [0, 1])
@pytest.mark.parametrize("hidden", [4, 8])
@pytest.mark.parametrize("activation", [0, 1])
def test_analytic_cnn_all_gradients_match_autodiff_and_sampled_finite_differences(
    seed, label, hidden, activation
):
    cls = strategy("grouped_cnn_lstm", "GroupedCNNLSTM")
    model = cls()
    model.initialize(seed, hidden, activation=activation)
    rng = random.Random(seed + 90)
    channels = [[rng.uniform(-1, 1) for _ in range(24)] for _ in range(10)]
    probability, loss, gradient = model.fast_prediction(channels, label=label)
    reference = model.forward(channels)
    expected_loss = -reference.log() if label else -(1 - reference).log()
    expected_loss.backward()
    assert probability == pytest.approx(reference.value, abs=1e-12)
    assert loss == pytest.approx(expected_loss.value, abs=1e-12)
    for parameter in model.parameters():
        assert gradient[parameter] == pytest.approx(parameter.grad, abs=2e-10)
    # Finite differences do not use either backward implementation.
    for part in (model.encoders, model.first, model.second, [model.lstm], [model.head]):
        values = [p for layer in part for p in layer.parameters()]
        for p in rng.sample(values, min(6, len(values))):
            old, eps = p.value, 1e-5
            losses = []
            for delta in (-eps, eps):
                p.value = old + delta
                pred = model.forward(channels).value
                losses.append(-math.log(pred) if label else -math.log(1 - pred))
            p.value = old
            assert gradient[p] == pytest.approx((losses[1] - losses[0]) / (2 * eps), abs=2e-8)


def controlled_request(seed, *, count=96, epochs=16, hidden=4):
    rng = random.Random(seed)
    samples = []
    for index in range(count):
        label = index % 2
        slope = (1 if label else -1) * rng.uniform(0.002, 0.008)
        price = rng.uniform(100, 500)
        rows = []
        for t in range(24):
            close = price * math.exp(slope * t + rng.uniform(-0.0003, 0.0003))
            rows.append([close * 0.999, close * 1.001, close * 0.998, close, rng.uniform(80, 120)])
        features = {
            f"v_{t}_{j}": Decimal(str(v)) for t, row in enumerate(rows) for j, v in enumerate(row)
        }
        features.update(
            partition=Decimal(index >= count * 0.75),
            window_end_ms=Decimal(index + 1),
            cfg_max_epochs=Decimal(epochs),
            cfg_patience=Decimal(epochs),
            cfg_batch_size=Decimal(8),
            cfg_learning_rate=Decimal("0.01"),
            cfg_hidden=Decimal(hidden),
        )
        samples.append(TrainingSample(features=features, target=Decimal(label)))
    return TrainingRequest(dataset_id="controlled-learnability", seed=seed, samples=tuple(samples))


@pytest.mark.parametrize("seed", [5, 19, 61])
@pytest.mark.parametrize("hidden", [4, 8])
def test_cnn_learns_unseen_controlled_patterns_and_reloads_without_test_branches(seed, hidden):
    cls = strategy("grouped_cnn_lstm", "GroupedCNNLSTM")
    request = controlled_request(seed, hidden=hidden)
    model = cls()
    payload = model.train(request)
    metrics = json.loads(payload)["training"]
    assert metrics["selected_train"]["bce"] < metrics["initial_train"]["bce"] * 0.65
    assert metrics["selected_validation"]["accuracy"] >= 0.9
    assert (
        metrics["selected_validation"]["probability_max"]
        - metrics["selected_validation"]["probability_min"]
        > 0.3
    )
    restored = cls()
    restored.load(payload)
    for sample in request.samples[-12:]:
        rows = [[float(sample.features[f"v_{t}_{j}"]) for j in range(5)] for t in range(24)]
        assert (
            restored.forward(restored.channels(rows)).value
            == model.forward(model.channels(rows)).value
        )


def test_cnn_validation_does_not_fit_scaler_or_change_training_updates():
    cls = strategy("grouped_cnn_lstm", "GroupedCNNLSTM")
    request = controlled_request(13, count=32, epochs=2)
    changed = request.model_copy(
        update={
            "samples": tuple(
                s.model_copy(update={"target": 1 - s.target}) if s.features["partition"] else s
                for s in request.samples
            )
        }
    )
    left, right = json.loads(cls().train(request)), json.loads(cls().train(changed))
    assert left["means"] == right["means"]
    assert left["scales"] == right["scales"]
    assert [e["train"] for e in left["training"]["epoch_trace"]] == [
        e["train"] for e in right["training"]["epoch_trace"]
    ]


@pytest.mark.parametrize("scale", [0.01, 0.5, 10, 10000])
def test_cnn_window_normalization_is_price_and_volume_scale_invariant(scale):
    cls = strategy("grouped_cnn_lstm", "GroupedCNNLSTM")
    rng = random.Random(7)
    bars = [
        [p * 0.99, p * 1.01, p * 0.98, p, rng.uniform(10, 20)] for p in [100 + t for t in range(24)]
    ]
    transformed = [[v * scale for v in row] for row in bars]
    for actual, expected in zip(cls.features(transformed), cls.features(bars), strict=True):
        assert actual == pytest.approx(expected, abs=1e-12)


def test_gmm_validation_cannot_change_quantile_scaler_mixture_or_neighbor_bank():
    cls = strategy("gmm_knn", "GMMConditionalKNN")
    rng = random.Random(73)
    samples = []
    for i in range(48):
        price = rng.uniform(100, 150)
        closes = [price]
        for _ in range(59):
            closes.append(closes[-1] * math.exp(rng.gauss(0, 0.002)))
        features = {f"close_{j}": Decimal(str(v)) for j, v in enumerate(closes)}
        features["partition"] = Decimal(i >= 36)
        samples.append(TrainingSample(features=features))
    request = TrainingRequest(dataset_id="gmm-partitions", seed=31, samples=tuple(samples))
    changed = request.model_copy(
        update={
            "samples": tuple(
                s.model_copy(
                    update={
                        "features": {
                            k: v * 10 if k.startswith("close_") else v
                            for k, v in s.features.items()
                        }
                    }
                )
                if s.features["partition"]
                else s
                for s in request.samples
            )
        }
    )
    left, right = json.loads(cls().train(request)), json.loads(cls().train(changed))
    for key in ("threshold", "center", "scales", "gmm", "groups"):
        assert left[key] == right[key]
    assert left["training"]["blocks"] == 36
    assert left["training"]["validation_blocks"] == 12
