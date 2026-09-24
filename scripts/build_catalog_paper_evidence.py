"""Compile reviewed claim-by-claim paper mappings for catalog strategies.

The reviewed anchor and method maps are inputs, not inferred from arbitrary PDFs.
The output binds each claim to a source page, implementation symbol and oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from paperquant.package import inspect_package
from paperquant.papers import verify_recipe

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "research/deep-evidence"


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: dict[str, Any]) -> bytes:
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def build_catalog_evidence() -> list[str]:
    anchors = _read(REVIEW / "anchors.json")
    methods = _read(REVIEW / "methods.json")
    catalog = _read(ROOT / "examples/catalog.json")
    bindings = _read(ROOT / "research/oracle_bindings.json")["bindings"]
    lock = _read(ROOT / "research/source-lock.json")
    by_url = {entry["canonical_url"]: entry for entry in lock["sources"]}
    supporting_by_url = {
        entry["canonical_url"]: entry for entry in lock.get("supporting_sources", [])
    }
    ids = {entry["strategy"] for entry in catalog["cases"]}
    if set(anchors) != set(methods) or not set(anchors) <= ids:
        raise ValueError("Reviewed anchor and method maps differ from the catalog")
    produced: list[str] = []
    for strategy_id in sorted(anchors):
        suffix = strategy_id.split(".", 1)[1]
        claim_path = ROOT / "research/claims" / f"{suffix}.json"
        claim = _read(claim_path)
        source_claim = claim.get("source", claim.get("trading_source"))
        if not isinstance(source_claim, dict) or not source_claim.get("url"):
            raise ValueError(f"Catalog strategy has no primary source: {strategy_id}")
        source = by_url[source_claim["url"]]
        additional = []
        if algorithm_source := claim.get("algorithm_source"):
            additional.append(supporting_by_url[algorithm_source["url"]])
        manifest, _ = inspect_package(strategy_id, ROOT / "strategies" / strategy_id)
        declaration = manifest.declaration
        claim_count = len(claim["claims"])
        reviewed_anchors = [
            dict(anchor, claim_index=anchor.get("claim_index", index))
            for index, anchor in enumerate(anchors[strategy_id])
        ]
        if (
            claim["strategy_id"] != strategy_id
            or claim_count != len(methods[strategy_id])
            or claim_count != len(bindings[strategy_id]["claim_nodes"])
            or {anchor["claim_index"] for anchor in reviewed_anchors} != set(range(claim_count))
        ):
            raise ValueError(f"Claim, anchor, method and oracle counts differ: {strategy_id}")
        spec_steps: list[dict[str, Any]] = []
        for index, method in enumerate(methods[strategy_id]):
            adaptation_index = method["adaptation_index"]
            oracle_node = bindings[strategy_id]["claim_nodes"][index][0]
            if oracle_node not in bindings[strategy_id]["node_ids"]:
                raise ValueError(f"Unbound claim oracle: {strategy_id} index {index}")
            spec_steps.append(
                {
                    "role": method["role"],
                    "claim_index": index,
                    "paper_rule": claim["claims"][index]["statement"],
                    "runtime_rule": method["runtime_rule"],
                    "implementation": method["implementation"],
                    "oracle_node": oracle_node,
                    "adaptation": claim["adaptations"][adaptation_index],
                }
            )
        spec = {
            "schema_version": "1.0",
            "strategy_id": strategy_id,
            "source_sha256": source["sha256"],
            "additional_source_sha256": [entry["sha256"] for entry in additional],
            "fields": sorted(declaration.data.fields),
            "steps": spec_steps,
            "excluded_paper_components": claim["adaptations"],
        }
        spec_path = ROOT / "research/deep-specs" / f"{suffix}.json"
        spec_data = _write(spec_path, spec)
        expected: dict[str, Any] = {
            "strategy_id": strategy_id,
            "kind": declaration.kind,
            "granularity": declaration.data.granularity,
            "fields": sorted(declaration.data.fields),
            "training_required": declaration.training_required,
            "symbols": sorted(declaration.data.symbols),
            "actions": sorted(declaration.actions),
        }
        for name in ("market_kind", "bar_seconds", "max_staleness_seconds", "minimum_book_depth"):
            value = getattr(declaration.data, name)
            if value is not None:
                expected[name] = value
        recipe = {
            "schema_version": "1.0",
            "source": {
                "title": source["title"],
                "url": source["canonical_url"],
                "file": f"../source-cache/{source['source_id']}.pdf",
                "format": "pdf",
                "sha256": source["sha256"],
                "license": "Redistribution not asserted; prepare hash-pinned local copy",
            },
            "additional_sources": [
                {
                    "title": entry["title"],
                    "url": entry["canonical_url"],
                    "file": f"../source-cache/{entry['source_id']}.pdf",
                    "format": "pdf",
                    "sha256": entry["sha256"],
                    "license": "Redistribution not asserted; prepare hash-pinned local copy",
                }
                for entry in additional
            ],
            "package": f"../../strategies/{strategy_id}",
            "package_source_sha256": manifest.source_sha256,
            "claim_file": f"../claims/{suffix}.json",
            "spec_file": f"../deep-specs/{suffix}.json",
            "spec_sha256": hashlib.sha256(spec_data).hexdigest(),
            "expected": expected,
            "anchors": reviewed_anchors,
            "review_note": (
                "Human-reviewed page anchors and declared method adaptations. "
                "Source text does not by itself prove algorithm or original-experiment fidelity; "
                "see the bound independent oracle and individual claim record."
            ),
        }
        recipe_path = ROOT / "research/deep-recipes" / f"{suffix}.json"
        _write(recipe_path, recipe)
        checked = verify_recipe(recipe_path)
        if checked["strategy_id"] != strategy_id or len(cast(list[Any], checked["anchors"])) != len(
            reviewed_anchors
        ):
            raise ValueError(f"Generated recipe failed complete claim verification: {strategy_id}")
        produced.append(strategy_id)
    return produced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    produced = build_catalog_evidence()
    print(
        json.dumps(
            {"status": "passed", "deep_recipes": len(produced), "strategies": produced},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
