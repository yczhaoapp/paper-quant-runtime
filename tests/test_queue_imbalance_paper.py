from __future__ import annotations

import importlib.util
import json
import math
import random
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.cli import main
from paperquant.compiler import fingerprint
from paperquant.models import DatasetDeclaration, MarketEvent, TrainingRequest, TrainingSample
from paperquant.package import inspect_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/supervised.queue_imbalance"
FIXTURE = ROOT / "examples/l1_queues"


def test_tick_learning_package_trains_reload_predicts_and_trades(tmp_path: Path) -> None:
    inspect_package("l1-check", PACKAGE)
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    training = TrainingRequest.model_validate_json((FIXTURE / "training.json").read_bytes())
    assert dataset.content_sha256 == fingerprint(events)
    assert training.dataset_id == dataset.dataset_id
    output = tmp_path / "tick-run"
    assert (
        main(
            [
                "run",
                "--package",
                str(PACKAGE),
                "--dataset",
                str(FIXTURE / "dataset.json"),
                "--events",
                str(FIXTURE / "events.json"),
                "--training",
                str(FIXTURE / "training.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["artifact"]["training_sha256"] == fingerprint(training)
    model = json.loads((output / report["artifact"]["relative_path"]).read_bytes())
    assert model["slope"] > 0
    assert report["fills"]
    assert {fill["side"] for fill in report["fills"]} == {"buy", "sell"}
    for event, decision in zip(events, report["decisions"], strict=True):
        bid_size = event.values["bid_size"]
        ask_size = event.values["ask_size"]
        imbalance = float((bid_size - ask_size) / (bid_size + ask_size))
        expected = 1 / (1 + math.exp(-(model["intercept"] + model["slope"] * imbalance)))
        prediction = next(
            action for action in decision["actions"] if action["kind"] == "prediction"
        )
        assert float(prediction["value"]) == pytest.approx(expected, rel=1e-12)


def test_training_labels_change_model_and_predictions(tmp_path: Path) -> None:
    original = json.loads((FIXTURE / "training.json").read_text())
    inverted = json.loads((FIXTURE / "training.json").read_text())
    for sample in inverted["samples"]:
        sample["target"] = "0" if Decimal(sample["target"]) == 1 else "1"
    flipped_file = tmp_path / "flipped-training.json"
    flipped_file.write_text(json.dumps(inverted), encoding="utf-8")
    assert original != inverted
    output = tmp_path / "flipped-run"
    assert (
        main(
            [
                "run",
                "--package",
                str(PACKAGE),
                "--dataset",
                str(FIXTURE / "dataset.json"),
                "--events",
                str(FIXTURE / "events.json"),
                "--training",
                str(flipped_file),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    model = json.loads((output / report["artifact"]["relative_path"]).read_bytes())
    assert model["slope"] < 0
    assert report["artifact"]["training_sha256"] == fingerprint(
        TrainingRequest.model_validate(inverted)
    )


@pytest.mark.parametrize("seed", [4, 19, 71])
def test_randomized_training_improves_independent_log_likelihood(seed: int) -> None:
    spec = importlib.util.spec_from_file_location("queue_learner", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    learner = module.QueueImbalanceLogistic()
    generator = random.Random(seed)
    samples = []
    for _ in range(60):
        bid = generator.randint(10, 100)
        ask = generator.randint(10, 100)
        threshold_noise = generator.uniform(-0.2, 0.2)
        imbalance = (bid - ask) / (bid + ask)
        samples.append(
            TrainingSample(
                features={"bid_size": Decimal(bid), "ask_size": Decimal(ask)},
                target=Decimal("1") if imbalance + threshold_noise > 0 else Decimal("0"),
            )
        )
    request = TrainingRequest(dataset_id="random-oracle", seed=seed, samples=tuple(samples))
    fitted = json.loads(learner.train(request))

    def independent_loss(intercept: float, slope: float) -> float:
        total = 0.0
        for sample in samples:
            bid = float(sample.features["bid_size"])
            ask = float(sample.features["ask_size"])
            x = (bid - ask) / (bid + ask)
            z = intercept + slope * x
            total += math.log1p(math.exp(z)) - float(sample.target) * z
        return total

    assert independent_loss(fitted["intercept"], fitted["slope"]) < independent_loss(0, 0)
    assert fitted["slope"] > 0
