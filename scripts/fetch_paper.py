"""Explicitly retrieve one hash-pinned public paper before offline verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paperquant.papers import fetch_paper_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = fetch_paper_source(args.url, args.sha256, args.output)
    print(json.dumps({"path": str(args.output), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
