"""Verify all eighteen source-bound strategies and one independent new paper.

Public papers with unasserted redistribution rights are explicitly prepared on
the host. This acceptance run itself is offline unless --fetch is requested.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from paperquant.cli import main as cli
from paperquant.evidence import verify_bundle_file
from paperquant.models import RunReport
from paperquant.output import atomic_bytes
from paperquant.papers import MethodSpec, Recipe, fetch_paper_source, verify_recipe
from scripts.acceptance import _run_oracles
from scripts.build_busseti_case import CACHE, REVIEW, build_case
from scripts.verify_paper_sources import LOCK, verify_sources

ROOT = Path(__file__).resolve().parents[1]
INPUT_LOCK = ROOT / "research/deep-evidence/inputs.json"
INDEPENDENT_NODES = (
    "tests/test_independent_busseti.py::test_generated_schedule_matches_independent_formula[3]",
    "tests/test_independent_busseti.py::test_generated_schedule_matches_independent_formula[17]",
    "tests/test_independent_busseti.py::test_generated_schedule_matches_independent_formula[41]",
    "tests/test_independent_busseti.py::test_generated_schedule_matches_independent_formula[73]",
    "tests/test_independent_busseti.py::test_independent_paper_source_build_and_runtime",
    "tests/test_independent_busseti.py::test_changed_independent_source_fails_before_package_build",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(path: Path, state: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(state, indent=2, sort_keys=True) + "\n").encode())


def _check_inputs(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lock = json.loads(INPUT_LOCK.read_text())
    cases = lock["cases"]
    if lock["schema_version"] != "1.0" or set(cases) != {
        entry["strategy"] for entry in entries
    }:
        raise ValueError("Pinned input evidence does not cover the eighteen-case catalog")
    for entry in entries:
        strategy_id = entry["strategy"]
        pinned = cases[strategy_id]
        expected_files = {"dataset": "dataset.json", "events": "events.json"}
        expected_files.update({key: entry[key] for key in ("policy", "training") if key in entry})
        if pinned["fixture"] != entry["fixture"] or set(pinned["files"]) != set(
            expected_files
        ):
            raise ValueError(f"Pinned fixture mapping differs: {strategy_id}")
        for kind, filename in expected_files.items():
            path = ROOT / "examples" / entry["fixture"] / filename
            record = pinned["files"][kind]
            if record["path"] != path.relative_to(ROOT).as_posix() or record["sha256"] != _digest(
                path
            ):
                raise ValueError(f"Pinned {kind} bytes differ: {strategy_id}")
    return cast(dict[str, dict[str, Any]], cases)


def _check_independent_inputs() -> dict[str, Any]:
    locked = json.loads(INPUT_LOCK.read_text())["independent"]
    if locked["fixture"] != "independent_busseti" or set(locked["files"]) != {
        "dataset", "events"
    }:
        raise ValueError("Independent paper fixture mapping differs")
    for kind in ("dataset", "events"):
        path = ROOT / "examples/independent_busseti" / f"{kind}.json"
        record = locked["files"][kind]
        if record["path"] != path.relative_to(ROOT).as_posix() or record["sha256"] != _digest(
            path
        ):
            raise ValueError(f"Independent paper {kind} bytes differ")
    return cast(dict[str, Any], locked["files"])


def _run_paper(
    recipe_path: Path, fixture: Path, entry: dict[str, Any], output: Path
) -> dict[str, Any]:
    arguments = [
        "paper-run",
        "--recipe",
        str(recipe_path),
        "--dataset",
        str(fixture / "dataset.json"),
        "--events",
        str(fixture / "events.json"),
        "--output",
        str(output),
    ]
    if "policy" in entry:
        arguments.extend(["--policy", str(fixture / entry["policy"])])
    if "training" in entry:
        arguments.extend(["--training", str(fixture / entry["training"])])
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        exit_code = cli(arguments)
    if exit_code != 0:
        raise ValueError(f"paper-run failed: {recipe_path.name}: {stream.getvalue()[:300]}")
    report = RunReport.model_validate_json((output / "report.json").read_bytes())
    verify_bundle_file(output / "bundle.json")
    receipt = json.loads((output / "paper-run.json").read_text())
    checked = verify_recipe(recipe_path)
    if (
        report.status != "succeeded"
        or not report.decisions
        or not report.fills
        or receipt["status"] != "passed"
        or receipt["strategy_id"] != checked["strategy_id"]
        or receipt["source_sha256"] != checked["source_sha256"]
        or tuple(receipt["additional_source_sha256"]) != checked["additional_source_sha256"]
        or receipt["bundle_sha256"] != _digest(output / "bundle.json")
    ):
        raise ValueError(f"Paper execution is incomplete: {checked['strategy_id']}")
    return {
        "strategy_id": checked["strategy_id"],
        "source_sha256": checked["source_sha256"],
        "additional_source_sha256": checked["additional_source_sha256"],
        "method_spec_sha256": checked["method_spec_sha256"],
        "recipe_sha256": _digest(recipe_path),
        "paper_run_sha256": _digest(output / "paper-run.json"),
        "bundle_sha256": _digest(output / "bundle.json"),
        "fills": len(report.fills),
    }


def run_depth(output: Path, *, fetch: bool = False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    attempt = output / "attempts" / attempt_id
    attempt.mkdir(parents=True)
    receipt_path = output / "paper-depth.json"
    state: dict[str, Any] = {
        "status": "running",
        "attempt_id": attempt_id,
        "started_at": datetime.now(UTC).isoformat(),
    }
    _publish(receipt_path, state)
    try:
        source_records = verify_sources(fetch=fetch)
        primary = [record for record in source_records if record["role"] == "primary"]
        supporting = [record for record in source_records if record["role"] == "supporting"]
        if len(primary) != 16 or len(supporting) != 1:
            raise ValueError("All sixteen primary and one algorithm source must verify")
        catalog = json.loads((ROOT / "examples/catalog.json").read_text())
        entries = catalog["cases"]
        pinned_inputs = _check_inputs(entries)
        bindings = json.loads((ROOT / "research/oracle_bindings.json").read_text())["bindings"]
        recipes = sorted(
            (
                *((ROOT / "research/recipes").glob("*.json")),
                *((ROOT / "research/deep-recipes").glob("*.json")),
            )
        )
        if len(entries) != 18 or len(recipes) != 18 or len(bindings) != 18:
            raise ValueError("Eighteen distinct catalog recipes and oracle bindings are required")
        by_strategy: dict[str, Path] = {}
        for path in recipes:
            checked = verify_recipe(path)
            strategy_id = str(checked["strategy_id"])
            if strategy_id in by_strategy:
                raise ValueError(f"Duplicate deep paper recipe: {strategy_id}")
            by_strategy[strategy_id] = path
            recipe = Recipe.model_validate_json(path.read_bytes())
            spec = MethodSpec.model_validate_json((path.parent / recipe.spec_file).read_bytes())
            claim = json.loads((path.parent / recipe.claim_file).read_text())
            if {anchor.claim_index for anchor in recipe.anchors} != set(
                range(len(claim["claims"]))
            ) or {step.claim_index for step in spec.steps} != set(range(len(claim["claims"]))):
                raise ValueError(f"Paper claim lacks a page anchor or method step: {strategy_id}")
            groups = bindings[strategy_id]["claim_nodes"]
            if any(step.oracle_node not in groups[step.claim_index] for step in spec.steps):
                raise ValueError(f"Method oracle does not bind its claim: {strategy_id}")
        if set(by_strategy) != {entry["strategy"] for entry in entries}:
            raise ValueError("Deep recipes differ from the eighteen runnable catalog cases")
        catalog_junit_sha = _run_oracles(attempt / "catalog_oracles", bindings)
        paper_runs = []
        for entry in entries:
            strategy_id = entry["strategy"]
            state["current_case"] = strategy_id
            _publish(receipt_path, state)
            record = _run_paper(
                    by_strategy[strategy_id],
                    ROOT / "examples" / entry["fixture"],
                    entry,
                    attempt / "paper_runs" / strategy_id,
                )
            record["pinned_inputs"] = pinned_inputs[strategy_id]["files"]
            paper_runs.append(record)
        del state["current_case"]
        independent_source = json.loads((REVIEW / "source.json").read_text())
        independent_inputs = _check_independent_inputs()
        if independent_source["url"] in {record["canonical_url"] for record in source_records}:
            raise ValueError("Independent paper duplicates a catalog or supporting source")
        if not CACHE.is_file() and fetch:
            fetch_paper_source(independent_source["url"], independent_source["sha256"], CACHE)
        built = attempt / "independent" / "build"
        build_receipt = build_case(CACHE, built)
        independent_junit_sha = _run_oracles(
            attempt / "independent_oracles",
            {"independent": {"node_ids": INDEPENDENT_NODES}},
        )
        independent = _run_paper(
            built / "recipe.json",
            ROOT / "examples/independent_busseti",
            {},
            attempt / "independent" / "run",
        )
        independent["pinned_inputs"] = independent_inputs
        state.update(
            {
                "status": "passed",
                "finished_at": datetime.now(UTC).isoformat(),
                "catalog_cases": 18,
                "unique_primary_sources": len(primary),
                "supporting_sources": len(supporting),
                "independent_new_sources": 1,
                "source_lock_sha256": _digest(LOCK),
                "input_lock_sha256": _digest(INPUT_LOCK),
                "catalog_oracle_junit_sha256": catalog_junit_sha,
                "independent_oracle_junit_sha256": independent_junit_sha,
                "paper_runs": paper_runs,
                "independent_build": build_receipt,
                "independent_paper_run": independent,
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "cause": type(exc).__name__,
                "reason": str(exc)[:500],
            }
        )
    _publish(receipt_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_depth(args.output, fetch=args.fetch)
    print(
        json.dumps({"status": result["status"], "receipt": str(args.output / "paper-depth.json")})
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
