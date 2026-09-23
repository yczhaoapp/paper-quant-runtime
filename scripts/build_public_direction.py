"""Build a chronological training/evaluation split from pinned AAPL daily bars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, TrainingRequest, TrainingSample
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FEATURE_ROWS = 320
EVALUATION_START_ROW = 321
FEATURES = ("open", "high", "low", "close", "volume")


def main() -> None:
    source = ROOT / "data/public/finance-charts-apple.csv"
    declaration, all_events = load_plotly_apple(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == APPLE_SHA256
    evaluation = all_events[EVALUATION_START_ROW:]
    evaluation_id = declaration.dataset_id + ".direction-holdout"
    training = TrainingRequest(
        dataset_id=evaluation_id,
        seed=0,
        samples=tuple(
            TrainingSample(
                features={name: all_events[index].values[name] for name in FEATURES},
                target=int(
                    all_events[index + 1].values["close"]
                    > all_events[index].values["close"]
                ),
            )
            for index in range(TRAIN_FEATURE_ROWS)
        ),
    )
    if not all_events[TRAIN_FEATURE_ROWS].available_time < evaluation[0].bar_open_time:
        raise ValueError("Training labels would overlap the evaluation opening")
    holdout = DatasetDeclaration(
        dataset_id=evaluation_id,
        granularity=declaration.granularity,
        fields=declaration.fields,
        symbols=declaration.symbols,
        event_count=len(evaluation),
        content_sha256=fingerprint(evaluation),
    )
    lineage = {
        "source_file_sha256": APPLE_SHA256,
        "source_event_sha256": declaration.content_sha256,
        "train_feature_rows": [0, TRAIN_FEATURE_ROWS - 1],
        "train_label_rows": [1, TRAIN_FEATURE_ROWS],
        "evaluation_rows": [EVALUATION_START_ROW, len(all_events) - 1],
        "last_label_available_time": all_events[TRAIN_FEATURE_ROWS].available_time.isoformat(),
        "first_evaluation_open_time": evaluation[0].bar_open_time.isoformat(),
        "label_rule": "next close > current close",
        "training_sha256": fingerprint(training),
        "evaluation_sha256": fingerprint(evaluation),
    }
    destination = ROOT / "examples/public_direction"
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("dataset.json", holdout),
        ("events.json", evaluation),
        ("training.json", training),
        ("lineage.json", lineage),
    ):
        (destination / name).write_text(
            json.dumps(canonical_document(value), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
