"""Build a reviewed, unlisted strategy package from a pinned new-paper method spec.

The paper-to-method interpretation is human reviewed. This builder checks the
actual source pages, then translates the declared equation (16) inputs into a
fresh package; it does not claim to infer an algorithm from arbitrary prose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from paperquant.models import StrategyDeclaration
from paperquant.package import write_manifest
from paperquant.papers import extract_text, fetch_paper_source

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "research/independent/busseti-static-vwap"
CACHE = ROOT / "research/source-cache/busseti2015.pdf"


def _read(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((REVIEW / name).read_text(encoding="utf-8")))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_code(method: dict[str, Any], source_url: str) -> str:
    profile = method["expected_volume_profile"]
    weights = tuple(Decimal(item) for item in profile)
    parent = Decimal(method["parent_quantity"])
    if len(weights) < 2 or any(weight <= 0 for weight in weights) or parent <= 0:
        raise ValueError("VWAP volume profile and parent quantity must be positive")
    if method["granularity"] != "minute" or method["bar_seconds"] != 60:
        raise ValueError("This reviewed paper case requires one-minute bars")
    source = '''from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed, Granularity, NoOp, StrategyDeclaration, StrategyKind, SubmitOrder,
)


class StaticVolumeShare:
    """Equation (16) static expected-volume-share schedule."""

    declaration = StrategyDeclaration(
        strategy_id=STRATEGY_ID,
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.MINUTE,
            bar_seconds=60,
            fields=frozenset({"open", "close", "volume"}),
            symbols=frozenset({SYMBOL}),
            minimum_events_per_symbol=MIN_EVENTS,
        ),
        actions=frozenset({"none", "submit_order"}),
        training_required=False,
        source=SOURCE_URL,
        max_abs_position=Decimal(PARENT),
        max_order_quantity=Decimal(PARENT),
    )

    def __init__(self):
        self.parent = Decimal(PARENT)
        self.profile = tuple(Decimal(x) for x in PROFILE)
        self.total = sum(self.profile)
        self.index = 0
        self.emitted = Decimal("0")

    def decide(self, event, account):
        del account
        index = self.index
        self.index += 1
        if index >= len(self.profile):
            return (NoOp(reason="volume-share parent completed"),)
        # Last-child residual avoids Decimal division rounding leaving inventory.
        quantity = (self.parent - self.emitted if index == len(self.profile) - 1
                    else self.parent * self.profile[index] / self.total)
        self.emitted += quantity
        return (SubmitOrder(
            client_order_id=f"vwap-{event.symbol}-{index:04d}",
            symbol=event.symbol, side="buy", quantity=quantity,
            reason="equation (16) expected-volume share",
        ),)
'''
    replacements = {
        "STRATEGY_ID": repr(method["strategy_id"]),
        "SYMBOL": repr(method["symbol"]),
        "MIN_EVENTS": str(len(weights) + 1),
        "SOURCE_URL": repr(source_url),
        "PARENT": repr(str(parent)),
        "PROFILE": repr(tuple(str(weight) for weight in weights)),
    }
    for name, value in replacements.items():
        source = source.replace(name, value)
    return source


def build_case(source_file: Path, output: Path) -> dict[str, str]:
    source = _read("source.json")
    method = _read("method.json")
    claim = _read("claim.json")
    if output.exists():
        raise ValueError("Output already exists; use a fresh directory for each build")
    raw = source_file.read_bytes()
    if _digest(raw) != source["sha256"]:
        raise ValueError("New-paper PDF differs from the pinned original bytes")
    catalog_urls = {
        document.get("source", document.get("trading_source"))["url"]
        for path in (ROOT / "research/claims").glob("*.json")
        for document in [json.loads(path.read_text())]
    }
    if source["url"] in catalog_urls:
        raise ValueError("Independent paper source is already in the eighteen-case catalog")
    pages = extract_text(raw, "pdf")
    if method["strategy_id"] != claim["strategy_id"] or claim["source"]["url"] != source["url"]:
        raise ValueError("Reviewed method, claim, and original paper disagree")
    for anchor in source["anchors"]:
        page = anchor["page"]
        if (
            page < 1
            or page > len(pages)
            or " ".join(anchor["text"].split()) not in " ".join(pages[page - 1].split())
        ):
            raise ValueError(f"Reviewed source anchor missing on page {page}")
    code = _source_code(method, source["url"])
    declaration = StrategyDeclaration.model_validate(
        {
            "strategy_id": method["strategy_id"],
            "kind": "rule",
            "data": {
                "granularity": "minute",
                "bar_seconds": 60,
                "fields": ["open", "close", "volume"],
                "symbols": [method["symbol"]],
                "minimum_events_per_symbol": len(method["expected_volume_profile"]) + 1,
            },
            "actions": ["none", "submit_order"],
            "training_required": False,
            "source": source["url"],
            "max_abs_position": method["parent_quantity"],
            "max_order_quantity": method["parent_quantity"],
        }
    )
    output.mkdir(parents=True)
    package = output / "package"
    package.mkdir()
    (package / "strategy.py").write_bytes(code.encode("utf-8"))
    write_manifest(package, declaration, "StaticVolumeShare")
    shutil.copyfile(source_file, output / "paper.pdf")
    shutil.copyfile(REVIEW / "claim.json", output / "claim.json")
    spec = {
        "schema_version": "1.0",
        "strategy_id": method["strategy_id"],
        "source_sha256": source["sha256"],
        "fields": ["open", "close", "volume"],
        "steps": [
            {
                "role": "execution",
                "claim_index": 0,
                "paper_rule": (
                    "Total child volume equals the parent order, with all child "
                    "volumes nonnegative."
                ),
                "runtime_rule": (
                    "Emit the residual as the final child and then cease submitting orders."
                ),
                "implementation": "StaticVolumeShare.decide",
                "oracle_node": (
                    "tests/test_independent_busseti.py::"
                    "test_generated_schedule_matches_independent_formula"
                ),
                "adaptation": (
                    "Synthetic minute bars and next-open market fills replace NYSE execution."
                ),
            },
            {
                "role": "execution",
                "claim_index": 1,
                "paper_rule": (
                    "Under constant spread, child volume is the parent size times "
                    "expected market-volume share."
                ),
                "runtime_rule": (
                    "Normalize the reviewed expected-volume profile and submit one "
                    "child per interval."
                ),
                "implementation": "StaticVolumeShare.decide",
                "oracle_node": (
                    "tests/test_independent_busseti.py::"
                    "test_generated_schedule_matches_independent_formula"
                ),
                "adaptation": (
                    "An illustrative profile replaces the paper's estimated "
                    "volume distribution."
                ),
            },
        ],
        "excluded_paper_components": method["excluded_paper_components"],
    }
    spec_bytes = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
    (output / "spec.json").write_bytes(spec_bytes)
    recipe = {
        "schema_version": "1.0",
        "source": {
            "title": source["title"],
            "url": source["url"],
            "file": "paper.pdf",
            "format": "pdf",
            "sha256": source["sha256"],
            "license": source["license"],
        },
        "package": "package",
        "package_source_sha256": _digest(code.encode()),
        "claim_file": "claim.json",
        "spec_file": "spec.json",
        "spec_sha256": _digest(spec_bytes),
        "expected": {
            "strategy_id": method["strategy_id"],
            "kind": "rule",
            "granularity": "minute",
            "bar_seconds": 60,
            "fields": ["open", "close", "volume"],
            "training_required": False,
            "symbols": [method["symbol"]],
            "actions": ["none", "submit_order"],
        },
        "anchors": [
            {"page": a["page"], "text": a["text"], "claim_index": index}
            for index, a in enumerate(source["anchors"])
        ],
        "review_note": (
            "Human-reviewed equation (16) static volume-share case. Synthetic minute data and "
            "illustrative expected volumes do not reproduce the dynamic policy or NYSE experiments."
        ),
    }
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    receipt = {
        "status": "built",
        "paper_sha256": source["sha256"],
        "reviewed_method_sha256": _digest((REVIEW / "method.json").read_bytes()),
        "strategy_source_sha256": _digest(code.encode()),
        "recipe_sha256": _digest((output / "recipe.json").read_bytes()),
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=CACHE)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source.exists() and args.fetch:
        source = _read("source.json")
        fetch_paper_source(source["url"], source["sha256"], args.source)
    print(json.dumps(build_case(args.source, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
