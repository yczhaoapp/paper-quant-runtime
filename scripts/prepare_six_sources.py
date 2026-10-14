"""Explicit host preparation of six evaluator-pinned source PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from paperquant.papers import extract_text, fetch_paper_source
from scripts.gate_support import new_attempt, write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(*, fetch=False):
    lock = json.loads((ROOT / "research/six-paper/source-lock.json").read_text())
    sources = lock["sources"]
    if len(sources) != 6 or len({s["id"] for s in sources}) != 6:
        raise ValueError("Six distinct evaluator-selected sources are required")
    all_sources = [*sources, *lock.get("supporting_sources", [])]
    if len({source["id"] for source in all_sources}) != len(all_sources):
        raise ValueError("Primary and supporting source identities must be distinct")
    verified = []
    for source in all_sources:
        path = ROOT / source["file"]
        if path.is_symlink() or not path.resolve().is_relative_to(
            (ROOT / "research/source-cache/six").resolve()
        ):
            raise ValueError("Paper cache path escapes its declared directory")
        if not path.exists():
            if not fetch:
                raise FileNotFoundError("Explicit source preparation required: " + source["id"])
            fetch_paper_source(source["url"], source["sha256"], path)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError("Source bytes differ from evaluator pin: " + source["id"])
        pages = extract_text(data, "pdf")

        def normalize(text):
            return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))

        if normalize(source["title"]) not in normalize(" ".join(pages[:2])):
            raise ValueError("Paper title differs from pinned source: " + source["id"])
        verified.append(
            {
                "id": source["id"],
                "sha256": source["sha256"],
                "pages": len(pages),
                "url": source["url"],
                "distribution": source["distribution"],
                "role": source.get("role", "evaluator_selected_paper"),
            }
        )
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / "sources.json"
    state = new_attempt()
    write_json(receipt, state)
    try:
        state.update(sources=prepare(fetch=args.fetch), status="passed")
    except Exception as exc:
        state.update(status="failed", cause=type(exc).__name__, reason=str(exc))
    state["completed_at"] = datetime.now(UTC).isoformat()
    write_json(receipt, state)
    print(json.dumps({"status": state["status"], "receipt": str(receipt)}))
    return 0 if state["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
