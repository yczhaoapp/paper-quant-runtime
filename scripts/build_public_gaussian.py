"""Pin a chronology-safe public AAPL sample for Gaussian NB component validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, TrainingRequest, TrainingSample
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("open", "high", "low", "close", "volume")


def main() -> None:
    source = ROOT / "data/public/finance-charts-apple.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != APPLE_SHA256:
        raise ValueError("pinned public CSV changed")
    original, all_events = load_plotly_apple(source)
    evaluation = all_events[321:]
    dataset_id = original.dataset_id + ".gaussian-holdout"
    training = TrainingRequest(
        dataset_id=dataset_id,
        seed=0,
        samples=tuple(
            TrainingSample(
                features={field: all_events[index].values[field] for field in FEATURES},
                target=int(all_events[index + 1].values["close"]
                           > all_events[index].values["close"]),
            )
            for index in range(320)
        ),
    )
    if all_events[320].available_time >= evaluation[0].bar_open_time:
        raise ValueError("last training label overlaps first evaluation open")
    dataset = DatasetDeclaration(
        dataset_id=dataset_id,
        granularity=original.granularity,
        session_timezone=original.session_timezone,
        session_open=original.session_open,
        session_close=original.session_close,
        fields=original.fields,
        symbols=original.symbols,
        event_count=len(evaluation),
        content_sha256=fingerprint(evaluation),
    )
    lineage = {
        "source_file_sha256": APPLE_SHA256,
        "source_event_sha256": original.content_sha256,
        "train_feature_rows": [0, 319],
        "train_label_rows": [1, 320],
        "evaluation_rows": [321, len(all_events) - 1],
        "last_label_available_time": all_events[320].available_time.isoformat(),
        "first_evaluation_open_time": evaluation[0].bar_open_time.isoformat(),
        "label_rule": "next close > current close",
        "training_sha256": fingerprint(training),
        "evaluation_sha256": fingerprint(evaluation),
    }
    destination = ROOT / "examples/public_gaussian"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, document in (
        ("dataset.json", dataset),
        ("events.json", evaluation),
        ("training.json", training),
        ("lineage.json", lineage),
        ("policy.json", {"symbol_map": {"AAPL": "DEMO"}}),
    ):
        (destination / filename).write_text(
            json.dumps(canonical_document(document), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
