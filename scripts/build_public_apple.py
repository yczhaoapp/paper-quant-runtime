"""Create canonical daily events from the locally bundled, pinned public CSV."""

from __future__ import annotations

import json
from pathlib import Path

from paperquant.compiler import canonical_document
from paperquant.public_data import load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    declaration, events = load_plotly_apple(ROOT / "data/public/finance-charts-apple.csv")
    destination = ROOT / "examples/public_aapl"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "dataset.json").write_text(
        json.dumps(canonical_document(declaration), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (destination / "events.json").write_text(
        json.dumps(canonical_document(events), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
