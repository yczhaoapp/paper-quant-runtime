from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def test_research_backlog_has_distinct_methods_and_balanced_targets() -> None:
    catalog = json.loads(Path("research/candidates.json").read_text(encoding="utf-8"))
    entries = catalog["entries"]
    assert Counter(entry["kind"] for entry in entries) == {
        "rule": 6,
        "supervised": 6,
        "reinforcement": 6,
    }
    assert len({entry["id"] for entry in entries}) == len(entries)
    assert len({entry["method"] for entry in entries}) == len(entries)
    assert all(entry["source"].startswith("https://") for entry in entries)
    assert all(entry["key_risk"] for entry in entries)
    assert all(
        entry["source_check"]
        in {"pending", "primary_metadata_confirmed", "primary_full_text_confirmed"}
        for entry in entries
    )
