"""Build chronological, synthetic two-asset pair transitions and held-out bars."""

from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import (
    DatasetDeclaration,
    Granularity,
    MarketEvent,
    TrainingRequest,
    TrainingSample,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "fixture.synthetic-pair-a2c-holdout"
WINDOW = 20
TRAIN_LAST = 139
EVALUATION_START = 140


def prices() -> tuple[list[Decimal], list[Decimal]]:
    left, right = [], []
    for index in range(240):
        right_value = 100 + 0.035 * index + 0.8 * math.sin(index / 9)
        residual = 1.7 * math.sin(index / 3.5) + 0.3 * math.sin(index / 1.7)
        right.append(Decimal(str(round(right_value, 8))))
        left.append(Decimal(str(round(4 + 0.97 * right_value + residual, 8))))
    return left, right


def observation(left: list[Decimal], right: list[Decimal], index: int) -> dict[str, Decimal]:
    # Standard-library OLS is deliberately independent of the strategy's OLS loop.
    xs = [float(value) for value in right[index-WINDOW:index]]
    ys = [float(value) for value in left[index-WINDOW:index]]
    fit = statistics.linear_regression(xs, ys)
    residuals = [y - fit.intercept - fit.slope * x for x, y in zip(xs, ys, strict=True)]
    current = float(left[index]) - fit.intercept - fit.slope * float(right[index])
    z_score = (current - statistics.mean(residuals)) / statistics.pstdev(residuals)
    zone = (-2 if z_score <= -1.8 else -1 if z_score < -0.4 else
            2 if z_score >= 1.8 else 1 if z_score > 0.4 else 0)
    return {"left_price": left[index], "right_price": right[index],
            "spread": Decimal(str(current)), "z_score": Decimal(str(z_score)),
            "zone": Decimal(zone)}


def build() -> tuple[DatasetDeclaration, tuple[MarketEvent, ...], TrainingRequest, dict]:
    left, right = prices()
    transitions = tuple(TrainingSample(
        features=observation(left, right, index),
        next_features=observation(left, right, index + 1),
        terminal=index == TRAIN_LAST - 1,
    ) for index in range(WINDOW, TRAIN_LAST))
    training = TrainingRequest(dataset_id=DATASET_ID, seed=23, samples=transitions)
    beginning = datetime(2025, 1, 1, 16, tzinfo=UTC)
    events = []
    for index in range(EVALUATION_START, len(left)):
        for symbol, series in (("ALPHA", left), ("BETA", right)):
            close = series[index]
            opening = series[index - 1]
            events.append(MarketEvent(
                event_id=f"pair-a2c-{index:03d}-{symbol}",
                symbol=symbol,
                bar_open_time=beginning + timedelta(days=index, hours=-7),
                event_time=beginning + timedelta(days=index),
                available_time=beginning + timedelta(days=index),
                values={"open": opening, "high": max(opening, close) + Decimal("0.5"),
                        "low": min(opening, close) - Decimal("0.5"), "close": close,
                        "volume": Decimal(1000)},
            ))
    evaluation = tuple(events)
    dataset = DatasetDeclaration(
        dataset_id=DATASET_ID, granularity=Granularity.DAY,
        fields=frozenset(evaluation[0].values), symbols=frozenset({"ALPHA", "BETA"}),
        event_count=len(evaluation), content_sha256=fingerprint(evaluation),
    )
    lineage = {
        "source": "deterministic synthetic, script-generated pair; no live market claim",
        "train_state_days": [WINDOW, TRAIN_LAST-1],
        "train_successor_days": [WINDOW+1, TRAIN_LAST],
        "evaluation_days": [EVALUATION_START, len(left)-1],
        "last_training_successor_available_time": (
            beginning + timedelta(days=TRAIN_LAST)).isoformat(),
        "first_evaluation_open_time": (
            beginning + timedelta(days=EVALUATION_START, hours=-7)).isoformat(),
        "training_sha256": fingerprint(training),
        "evaluation_sha256": fingerprint(evaluation),
    }
    return dataset, evaluation, training, lineage


def main() -> None:
    destination = ROOT / "examples/pairs_actor_critic"
    destination.mkdir(parents=True, exist_ok=True)
    dataset, events, training, lineage = build()
    for filename, value in (("dataset.json", dataset), ("events.json", events),
                            ("training.json", training), ("lineage.json", lineage)):
        (destination / filename).write_text(
            json.dumps(canonical_document(value), indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )


if __name__ == "__main__":
    main()
