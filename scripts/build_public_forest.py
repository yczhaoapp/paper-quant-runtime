"""Build chronological operation labels from a pinned public OHLCV history."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import canonical_document, fingerprint
from paperquant.models import DatasetDeclaration, TrainingRequest, TrainingSample
from paperquant.public_data import APPLE_SHA256, load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("open", "high", "low", "close", "volume")
GAIN = Decimal("0.03")
LOSS = Decimal("0.015")
MAX_HOLD = 5


def operation_label(closes: tuple[Decimal, ...], index: int) -> int:
    entry = closes[index]
    long_open = True
    short_open = True
    for future in closes[index + 1 : index + MAX_HOLD + 1]:
        change = future / entry - 1
        if long_open and change >= GAIN:
            return 1
        if short_open and change <= -GAIN:
            return -1
        if change <= -LOSS:
            long_open = False
        if change >= LOSS:
            short_open = False
        if not long_open and not short_open:
            break
    return 0


def main() -> None:
    source = ROOT / "data/public/finance-charts-apple.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != APPLE_SHA256:
        raise ValueError("pinned public CSV changed")
    original, all_events = load_plotly_apple(source)
    closes = tuple(event.values["close"] for event in all_events)
    evaluation = all_events[321:]
    dataset_id = original.dataset_id + ".forest-operations-holdout"
    training = TrainingRequest(
        dataset_id=dataset_id,
        seed=23,
        samples=tuple(
            TrainingSample(
                features={field: all_events[index].values[field] for field in FEATURES},
                target=operation_label(closes, index),
            )
            for index in range(316)
        ),
    )
    if all_events[320].available_time >= evaluation[0].bar_open_time:
        raise ValueError("latest operation label overlaps first evaluation opening")
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
        "train_feature_rows": [0, 315],
        "train_label_latest_rows": [5, 320],
        "evaluation_rows": [321, len(all_events) - 1],
        "last_label_available_time": all_events[320].available_time.isoformat(),
        "first_evaluation_open_time": evaluation[0].bar_open_time.isoformat(),
        "label_rule": (
            "first successful 3% stop-gain before 1.5% stop-loss "
            "within 5 closes, else no action"
        ),
        "training_sha256": fingerprint(training),
        "evaluation_sha256": fingerprint(evaluation),
    }
    destination = ROOT / "examples/public_forest"
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
