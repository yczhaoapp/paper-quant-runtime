"""Build a three-bar, three-day-ahead Ridge example from pinned AAPL daily data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, TrainingRequest, TrainingSample
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("open", "high", "low", "close", "volume")
WINDOW = 3
HORIZON = 3
LAST_LABEL_ROW = 320
EVALUATION_START_ROW = 321


def window_features(events, index):
    return {
        f"{field}_lag{lag}": events[index - lag].values[field]
        for lag in (2, 1, 0)
        for field in FIELDS
    }


def main() -> None:
    source = ROOT / "data/public/finance-charts-apple.csv"
    declaration, all_events = load_plotly_apple(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != APPLE_SHA256:
        raise ValueError("Pinned source digest differs")
    evaluation = all_events[EVALUATION_START_ROW:]
    dataset_id = declaration.dataset_id + ".ridge-holdout"
    training = TrainingRequest(
        dataset_id=dataset_id,
        seed=0,
        samples=tuple(
            TrainingSample(
                features=window_features(all_events, index),
                target=all_events[index + HORIZON].values["close"],
            )
            for index in range(WINDOW - 1, LAST_LABEL_ROW - HORIZON + 1)
        ),
    )
    first_open = evaluation[0].bar_open_time
    last_label_time = all_events[LAST_LABEL_ROW].available_time
    if first_open is None or not last_label_time < first_open:
        raise ValueError("Ridge training labels overlap evaluation")
    holdout = DatasetDeclaration(
        dataset_id=dataset_id,
        granularity=declaration.granularity,
        session_timezone=declaration.session_timezone,
        session_open=declaration.session_open,
        session_close=declaration.session_close,
        fields=declaration.fields,
        symbols=declaration.symbols,
        event_count=len(evaluation),
        content_sha256=fingerprint(evaluation),
    )
    lineage = {
        "source_file_sha256": APPLE_SHA256,
        "source_event_sha256": declaration.content_sha256,
        "window_bars": WINDOW,
        "forecast_horizon_bars": HORIZON,
        "train_feature_end_rows": [WINDOW - 1, LAST_LABEL_ROW - HORIZON],
        "train_label_rows": [WINDOW - 1 + HORIZON, LAST_LABEL_ROW],
        "evaluation_rows": [EVALUATION_START_ROW, len(all_events) - 1],
        "last_label_available_time": last_label_time.isoformat(),
        "first_evaluation_open_time": first_open.isoformat(),
        "training_sha256": fingerprint(training),
        "evaluation_sha256": fingerprint(evaluation),
    }
    destination = ROOT / "examples/public_ridge"
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("dataset.json", holdout),
        ("events.json", evaluation),
        ("training.json", training),
        ("lineage.json", lineage),
        ("policy.json", {"symbol_map": {"AAPL": "DEMO"}}),
    ):
        (destination / name).write_text(
            json.dumps(canonical_document(value), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
