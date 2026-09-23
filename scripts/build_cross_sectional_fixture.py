"""Generate a chronological synthetic panel with known conditional returns."""

from __future__ import annotations

import json
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
SYMBOLS = ("ALPHA", "BETA", "GAMMA")
DATASET_ID = "fixture.synthetic-cross-sectional-holdout"


def build_panel() -> tuple[MarketEvent, ...]:
    start = datetime(2026, 1, 1, 16, tzinfo=UTC)
    previous = {symbol: Decimal(100 + 5 * index) for index, symbol in enumerate(SYMBOLS)}
    events = []
    for day in range(161):
        for asset, symbol in enumerate(SYMBOLS):
            signal = Decimal(((day * 7 + asset * 11) % 17) - 8) / 8
            liquidity = Decimal(((day * 3 + asset * 5) % 11) - 5) / 5
            opening = previous[symbol]
            if day == 0:
                closing = opening
            else:
                earlier_signal = Decimal((((day - 1) * 7 + asset * 11) % 17) - 8) / 8
                earlier_liquidity = Decimal((((day - 1) * 3 + asset * 5) % 11) - 5) / 5
                noise = Decimal("0.001") if (day + asset) % 2 else Decimal("-0.001")
                closing = opening * (1 + Decimal("0.006") * earlier_signal
                                     + Decimal("0.002") * earlier_liquidity + noise)
            previous[symbol] = closing
            events.append(MarketEvent(
                event_id=f"rank-{day:03d}-{symbol}",
                symbol=symbol,
                bar_open_time=start + timedelta(days=day, hours=-7),
                event_time=start + timedelta(days=day),
                available_time=start + timedelta(days=day),
                values={
                    "open": opening,
                    "close": closing,
                    "signal": signal,
                    "liquidity": liquidity,
                },
            ))
    return tuple(events)


def main() -> None:
    panel = build_panel()
    evaluation = panel[121 * 3:]
    training = TrainingRequest(
        dataset_id=DATASET_ID,
        seed=0,
        samples=tuple(
            TrainingSample(
                features={key: panel[day * 3 + asset].values[key]
                          for key in ("signal", "liquidity")},
                target=(panel[(day + 1) * 3 + asset].values["close"]
                        / panel[day * 3 + asset].values["close"] - 1),
            )
            for day in range(120)
            for asset in range(3)
        ),
    )
    if panel[120 * 3].available_time >= evaluation[0].bar_open_time:
        raise ValueError("training label overlaps evaluation")
    dataset = DatasetDeclaration(
        dataset_id=DATASET_ID,
        granularity=Granularity.DAY,
        fields=frozenset(panel[0].values),
        symbols=frozenset(SYMBOLS),
        event_count=len(evaluation),
        content_sha256=fingerprint(evaluation),
    )
    lineage = {
        "generator": "scripts/build_cross_sectional_fixture.py",
        "source_panel_sha256": fingerprint(panel),
        "training_feature_days": [0, 119],
        "training_label_days": [1, 120],
        "evaluation_days": [121, 160],
        "last_label_available_time": panel[120 * 3].available_time.isoformat(),
        "first_evaluation_open_time": evaluation[0].bar_open_time.isoformat(),
        "training_sha256": fingerprint(training),
        "evaluation_sha256": fingerprint(evaluation),
    }
    destination = ROOT / "examples/cross_sectional_rank"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, document in (
        ("dataset.json", dataset),
        ("events.json", evaluation),
        ("training.json", training),
        ("lineage.json", lineage),
        ("policy.json", {}),
    ):
        (destination / filename).write_text(
            json.dumps(canonical_document(document), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
