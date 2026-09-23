"""Build a synthetic on-policy transition trace for the SARSA contract example."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import canonical_document
from paperquant.models import TrainingRequest, TrainingSample

ROOT = Path(__file__).resolve().parents[1]


def features(pressure: int, inventory: int) -> dict[str, Decimal]:
    bid, ask = (75, 25) if pressure > 0 else (25, 75)
    return {
        "bid_size": Decimal(bid),
        "ask_size": Decimal(ask),
        "inventory": Decimal(inventory),
    }


def main() -> None:
    samples = [
        TrainingSample(
            features=features(pressure, inventory),
            action=pressure,
            reward=Decimal("2"),
            terminal=True,
        )
        for pressure in (1, -1)
        for inventory in (-1, 0, 1)
    ]
    samples.extend(
        (
            TrainingSample(
                features=features(1, 0),
                action=1,
                reward=Decimal("1"),
                next_features=features(1, 1),
                next_action=1,
            ),
            TrainingSample(
                features=features(1, 1),
                action=1,
                reward=Decimal("2"),
                terminal=True,
            ),
        )
    )
    request = TrainingRequest(
        dataset_id="fixture.synthetic-l1-queues", seed=13, samples=tuple(samples)
    )
    path = ROOT / "examples/l1_queues/sarsa_training.json"
    path.write_text(
        json.dumps(canonical_document(request), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
