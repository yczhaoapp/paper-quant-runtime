"""Verify the exact public source bytes behind every catalog research claim.

Fetching is an explicit host-side preparation step. Strategy workers never fetch papers.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from paperquant.papers import fetch_paper_source

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research/source-lock.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _words(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", folded))


def _resolve_file(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Paper source path must stay within the repository")
    path = ROOT / relative
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Paper source path escapes the repository or is a symlink")
    return path


def _expected_claim_urls() -> set[str]:
    catalog = json.loads((ROOT / "examples/catalog.json").read_text())
    entries = catalog["cases"]
    if len(entries) != 18 or len({item["strategy"] for item in entries}) != 18:
        raise ValueError("Paper source gate requires the eighteen-case catalog")
    urls = set()
    for item in entries:
        strategy_id = item["strategy"]
        claim_path = ROOT / "research/claims" / (strategy_id.split(".", 1)[1] + ".json")
        claim = json.loads(claim_path.read_text())
        if claim["strategy_id"] != strategy_id or not claim["claims"]:
            raise ValueError(f"Research claim is absent or mismatched: {strategy_id}")
        source = claim.get("source", claim.get("trading_source"))
        if not isinstance(source, dict) or not source.get("url"):
            raise ValueError(f"Primary paper source is absent: {strategy_id}")
        urls.add(source["url"])
    return urls


def verify_sources(*, fetch: bool = False) -> list[dict[str, object]]:
    lock_bytes = LOCK.read_bytes()
    lock = json.loads(lock_bytes)
    if lock.get("schema_version") != "1.0" or not isinstance(lock.get("sources"), list):
        raise ValueError("Paper source lock has an unsupported schema")
    sources = lock["sources"]
    ids = [source["source_id"] for source in sources]
    urls = [source["canonical_url"] for source in sources]
    if (
        len(ids) != len(set(ids))
        or len(urls) != len(set(urls))
        or set(urls) != _expected_claim_urls()
    ):
        raise ValueError("Paper source lock does not cover each catalog primary source")
    records: list[dict[str, object]] = []
    for source in sources:
        source_id = source["source_id"]
        expected = source["sha256"]
        retrieval_url = source["retrieval_url"]
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", source_id)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not retrieval_url.startswith("https://")
            or source["format"] != "pdf"
        ):
            raise ValueError(f"Paper source lock entry is invalid: {source_id}")
        path = _resolve_file(source["file"])
        if not path.exists():
            if not fetch:
                raise ValueError(f"Paper source bytes are absent: {source_id}")
            fetch_paper_source(retrieval_url, expected, path)
        data = path.read_bytes()
        if _sha256(data) != expected:
            raise ValueError(f"Paper source bytes differ from pinned digest: {source_id}")
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted or not reader.pages:
            raise ValueError(f"Paper source is not an extractable PDF: {source_id}")
        opening_text = " ".join((page.extract_text() or "") for page in reader.pages[:2])
        if not opening_text.strip() or _words(source["title"]) not in _words(opening_text):
            raise ValueError(f"Paper title is absent from pinned PDF: {source_id}")
        records.append(
            {
                "source_id": source_id,
                "canonical_url": source["canonical_url"],
                "retrieval_url": retrieval_url,
                "sha256": expected,
                "pages": len(reader.pages),
                "file": source["file"],
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="explicitly fetch missing public PDFs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = verify_sources(fetch=args.fetch)
        result: dict[str, object] = {
            "status": "passed",
            "source_lock_sha256": _sha256(LOCK.read_bytes()),
            "catalog_cases": 18,
            "unique_primary_sources": len(records),
            "sources": records,
        }
    except (KeyError, OSError, ValueError, TypeError) as exc:
        result = {"status": "failed", "cause": type(exc).__name__, "reason": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(args.output)}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
