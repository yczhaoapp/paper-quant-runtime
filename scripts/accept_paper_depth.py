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
from scripts.acceptance import PUBLIC_CASES, _check_paper_run_scope, _run_oracles
from scripts.assurance_axes import validate_assurance
from scripts.build_busseti_case import CACHE, REVIEW, build_case
from scripts.verify_paper_sources import LOCK, verify_sources
from scripts.verify_strict import _git_state, _source_sha256

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


def _check_method_oracle_nodes(declared: tuple[str, ...], passed: tuple[str, ...]) -> None:
    # A pytest parent node selects its parametrized cases; a specific case must match exactly.
    covered = set(passed) | {node.split("[", 1)[0] for node in passed}
    if not declared or not set(declared) <= covered:
        raise ValueError("Method spec refers to an oracle outside this passed attempt")


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


def _check_public_supplement(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_strategy = {entry["strategy"]: entry for entry in entries}
    expected = {
        strategy_id: fixture_name
        for strategy_id, fixture_name in PUBLIC_CASES
        if by_strategy[strategy_id]["fixture"] != fixture_name
    }
    locked = json.loads(INPUT_LOCK.read_text())["public_supplement"]
    if set(locked) != set(expected):
        raise ValueError("Public-data supplement must cover exactly the missing catalog runs")
    for strategy_id, fixture_name in expected.items():
        record = locked[strategy_id]
        fixture = ROOT / "examples" / fixture_name
        expected_files = {"dataset": "dataset.json", "events": "events.json",
                          "policy": "policy.json"}
        if (fixture / "training.json").is_file():
            expected_files["training"] = "training.json"
        if record["fixture"] != fixture_name or set(record["files"]) != set(expected_files):
            raise ValueError(f"Public-data supplement files differ: {strategy_id}")
        for kind, filename in expected_files.items():
            path = fixture / filename
            pinned = record["files"][kind]
            if pinned["path"] != path.relative_to(ROOT).as_posix() or (
                pinned["sha256"] != _digest(path)
            ):
                raise ValueError(f"Pinned public {kind} bytes differ: {strategy_id}")
    return cast(dict[str, dict[str, Any]], locked)


def _run_paper(
    recipe_path: Path, fixture: Path, entry: dict[str, Any], output: Path,
    *, oracle_junit_sha256: str,
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
    _check_paper_run_scope(receipt)
    checked = verify_recipe(recipe_path)
    if (
        report.status != "succeeded"
        or report.plan.sandbox != "development"
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
        "source_mapping": "passed",
        "runtime_validation": "passed",
        "replay_validation": "passed",
        "method_validation": "passed",
        "method_oracle_nodes": checked["method_oracle_nodes"],
        "oracle_junit_sha256": oracle_junit_sha256,
        "source_sha256": checked["source_sha256"],
        "additional_source_sha256": checked["additional_source_sha256"],
        "method_spec_sha256": checked["method_spec_sha256"],
        "recipe_sha256": _digest(recipe_path),
        "paper_run_sha256": _digest(output / "paper-run.json"),
        "bundle_sha256": _digest(output / "bundle.json"),
        "source_events_sha256": report.source_events_sha256,
        "training_sha256": report.plan.training_sha256,
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
        source_tree_sha256 = _source_sha256()
        git_commit, git_clean = _git_state()
        state.update({
            "source_tree_sha256": source_tree_sha256,
            "git_commit": git_commit,
            "git_clean": git_clean,
        })
        _publish(receipt_path, state)
        source_records = verify_sources(fetch=fetch)
        primary = [record for record in source_records if record["role"] == "primary"]
        supporting = [record for record in source_records if record["role"] == "supporting"]
        if len(primary) != 16 or len(supporting) != 1:
            raise ValueError("All sixteen primary and one algorithm source must verify")
        catalog = json.loads((ROOT / "examples/catalog.json").read_text())
        entries = catalog["cases"]
        pinned_inputs = _check_inputs(entries)
        public_supplement = _check_public_supplement(entries)
        assurance = validate_assurance(
            {entry["strategy"] for entry in entries},
            {strategy_id for strategy_id, _ in PUBLIC_CASES},
        )
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
                    oracle_junit_sha256=catalog_junit_sha,
                )
            record["pinned_inputs"] = pinned_inputs[strategy_id]["files"]
            paper_runs.append(record)
        del state["current_case"]
        run_by_strategy = {record["strategy_id"]: record for record in paper_runs}
        source_record = json.loads((ROOT / "data/public/SOURCE.json").read_text())
        source_csv_sha256 = _digest(ROOT / "data/public/finance-charts-apple.csv")
        if source_record["sha256"] != source_csv_sha256:
            raise ValueError("Public market source differs from its pinned raw CSV")
        public_paper_runs = []
        catalog_by_strategy = {entry["strategy"]: entry for entry in entries}
        for strategy_id, fixture_name in PUBLIC_CASES:
            fixture = ROOT / "examples" / fixture_name
            if catalog_by_strategy[strategy_id]["fixture"] == fixture_name:
                record = run_by_strategy[strategy_id]
            else:
                entry = {"policy": "policy.json"}
                if (fixture / "training.json").is_file():
                    entry["training"] = "training.json"
                record = _run_paper(
                    by_strategy[strategy_id],
                    fixture,
                    entry,
                    attempt / "public_supplement" / strategy_id,
                    oracle_junit_sha256=catalog_junit_sha,
                )
                record["pinned_inputs"] = public_supplement[strategy_id]["files"]
            lineage_path = fixture / "lineage.json"
            lineage_sha256 = None
            if lineage_path.is_file():
                lineage = json.loads(lineage_path.read_text())
                if (
                    lineage["source_file_sha256"] != source_csv_sha256
                    or lineage["evaluation_sha256"] != record["source_events_sha256"]
                    or lineage["training_sha256"] != record["training_sha256"]
                ):
                    raise ValueError(f"Public-data lineage differs from paper run: {strategy_id}")
                lineage_sha256 = _digest(lineage_path)
            elif (fixture / "training.json").is_file():
                raise ValueError(f"Public-data training lineage is absent: {strategy_id}")
            public_paper_runs.append({
                "strategy_id": strategy_id,
                "source_mapping": record["source_mapping"],
                "runtime_validation": record["runtime_validation"],
                "method_validation": record["method_validation"],
                "oracle_junit_sha256": record["oracle_junit_sha256"],
                "fixture": fixture_name,
                "raw_source_sha256": source_csv_sha256,
                "lineage_sha256": lineage_sha256,
                "paper_run_sha256": record["paper_run_sha256"],
                "bundle_sha256": record["bundle_sha256"],
                "source_events_sha256": record["source_events_sha256"],
                "training_sha256": record["training_sha256"],
                "pinned_inputs": record["pinned_inputs"],
            })
        if len(public_paper_runs) != assurance["data_counts"]["D1"]:
            raise ValueError("D1 grades differ from actual public-paper runs")
        independent_source = json.loads((REVIEW / "source.json").read_text())
        independent_inputs = _check_independent_inputs()
        if independent_source["url"] in {record["canonical_url"] for record in source_records}:
            raise ValueError("Independent paper duplicates a catalog or supporting source")
        if not CACHE.is_file() and fetch:
            fetch_paper_source(independent_source["url"], independent_source["sha256"], CACHE)
        built = attempt / "independent" / "build"
        build_receipt = build_case(CACHE, built)
        independent_recipe = Recipe.model_validate_json((built / "recipe.json").read_bytes())
        independent_spec = MethodSpec.model_validate_json(
            (built / independent_recipe.spec_file).read_bytes())
        independent_junit_sha = _run_oracles(
            attempt / "independent_oracles",
            {"independent": {"node_ids": INDEPENDENT_NODES}},
        )
        _check_method_oracle_nodes(
            tuple(step.oracle_node for step in independent_spec.steps), INDEPENDENT_NODES)
        independent = _run_paper(
            built / "recipe.json",
            ROOT / "examples/independent_busseti",
            {},
            attempt / "independent" / "run",
            oracle_junit_sha256=independent_junit_sha,
        )
        independent["pinned_inputs"] = independent_inputs
        if source_tree_sha256 != _source_sha256() or (git_commit, git_clean) != _git_state():
            raise ValueError("Paper-depth source tree changed during this attempt")
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
                "assurance": {
                    "reviewed": assurance,
                    "runtime": {
                        "level": "R1",
                        "scope": "host paper intake for eighteen catalog cases and one new case",
                        "attempt_id": attempt_id,
                    },
                },
                "catalog_oracle_junit_sha256": catalog_junit_sha,
                "independent_oracle_junit_sha256": independent_junit_sha,
                "paper_runs": paper_runs,
                "public_paper_runs": public_paper_runs,
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
