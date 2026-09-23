"""Build on-policy training transitions and disjoint evaluation from pinned AAPL bars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, TrainingRequest, TrainingSample
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
OHLCV = ("open", "high", "low", "close", "volume")


def main() -> None:
    source = ROOT / "data/public/finance-charts-apple.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != APPLE_SHA256:
        raise ValueError("pinned public CSV changed")
    original, all_events = load_plotly_apple(source)
    evaluation = all_events[321:]
    dataset_id = original.dataset_id + ".actor-critic-holdout"
    training = TrainingRequest(
        dataset_id=dataset_id,
        seed=17,
        samples=tuple(
            TrainingSample(
                features={key: all_events[index].values[key] for key in OHLCV},
                next_features={key: all_events[index + 1].values[key] for key in OHLCV},
                terminal=index == 319,
            )
            for index in range(320)
        ),
    )
    if all_events[320].available_time >= evaluation[0].bar_open_time:
        raise ValueError("last training transition overlaps evaluation open")
    dataset = DatasetDeclaration(
        dataset_id=dataset_id,
        granularity=original.granularity,
        fields=original.fields,
        symbols=original.symbols,
        event_count=len(evaluation),
        content_sha256=fingerprint(evaluation),
    )
    lineage = {
        "source_file_sha256": APPLE_SHA256,
        "source_event_sha256": original.content_sha256,
        "train_state_rows": [0, 319],
        "train_successor_rows": [1, 320],
        "evaluation_rows": [321, len(all_events) - 1],
        "last_successor_available_time": all_events[320].available_time.isoformat(),
        "first_evaluation_open_time": evaluation[0].bar_open_time.isoformat(),
        "reward_rule": "log(1 + sampled risky weight * next-close simple return)",
        "training_sha256": fingerprint(training),
        "evaluation_sha256": fingerprint(evaluation),
    }
    destination = ROOT / "examples/public_actor_critic"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("dataset.json", dataset),
        ("events.json", evaluation),
        ("training.json", training),
        ("lineage.json", lineage),
        ("policy.json", {"symbol_map": {"AAPL": "DEMO"}}),
    ):
        (destination / filename).write_text(
            json.dumps(canonical_document(value), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
