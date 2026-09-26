"""Validate reviewed evidence scopes without treating a grade as semantic proof.

Method grades are human judgments bound to exact claims and runnable oracle
nodes. Data and result grades must agree with independently checked fixtures
and claim records. Runtime grades are deliberately absent here: only an
attempt-scoped execution receipt can award R1 or R2.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "research/assurance-axes.json"
LEVELS = {"A0": 0, "A1": 1, "A2": 2}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_assurance(
    catalog_ids: set[str],
    public_ids: set[str],
    *,
    matrix_path: Path = MATRIX,
) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    cases = matrix["cases"]
    if (
        matrix["schema_version"] != "1.0"
        or matrix["assessment_kind"] != "human_reviewed_scope_not_semantic_proof"
        or set(cases) != catalog_ids
        or len(cases) != 18
        or len(public_ids) != 8
        or not public_ids <= catalog_ids
    ):
        raise ValueError("Assurance matrix does not cover the exact catalog and public runs")
    bindings = json.loads((ROOT / "research/oracle_bindings.json").read_text())["bindings"]
    if set(bindings) != catalog_ids:
        raise ValueError("Assurance matrix differs from current oracle bindings")
    for strategy_id, case in cases.items():
        if set(case) != {"core", "strategy", "data", "experiment", "claim_sha256"}:
            raise ValueError(f"Assurance fields differ: {strategy_id}")
        claim_path = ROOT / "research/claims" / f"{strategy_id.split('.', 1)[1]}.json"
        claim = json.loads(claim_path.read_text())
        if (
            claim["strategy_id"] != strategy_id
            or _digest(claim_path) != case["claim_sha256"]
            or case["claim_sha256"] != bindings[strategy_id]["claim_sha256"]
        ):
            raise ValueError(f"Assurance grade is detached from reviewed claims: {strategy_id}")
        for scope in ("core", "strategy"):
            assessment = case[scope]
            if (
                assessment.get("level") not in LEVELS
                or not isinstance(assessment.get("scope"), str)
                or len(assessment["scope"]) < 40
            ):
                raise ValueError(f"Unreviewed assurance scope: {strategy_id} {scope}")
        indices = case["core"].get("claim_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
            or len(indices) != len(set(indices))
            or not set(indices) <= set(range(len(claim["claims"])))
            or any(not bindings[strategy_id]["claim_nodes"][index] for index in indices)
        ):
            raise ValueError(f"Core grade has no exact claim and oracle binding: {strategy_id}")
        if LEVELS[case["strategy"]["level"]] > LEVELS[case["core"]["level"]]:
            raise ValueError(f"Strategy grade exceeds its core evidence: {strategy_id}")
        expected_data = "D1" if strategy_id in public_ids else "D0"
        has_public_claim = "public_market_proxy" in claim["data_fidelity"]
        if case["data"] != expected_data or has_public_claim != (expected_data == "D1"):
            raise ValueError(f"Data grade differs from actual public-data track: {strategy_id}")
        if case["experiment"] != "E0" or claim["experimental_fidelity"] != "not_reproduced":
            raise ValueError(f"Unverified paper-result grade: {strategy_id}")
    independent = matrix["independent"]
    method_path = ROOT / "research/independent/busseti-static-vwap/method.json"
    source_path = ROOT / "research/independent/busseti-static-vwap/source.json"
    method = json.loads(method_path.read_text())
    source = json.loads(source_path.read_text())
    if (
        set(independent) != {
            "strategy_id", "core", "strategy", "data", "experiment",
            "reviewed_method_sha256", "source_sha256",
        }
        or independent["strategy_id"] != method["strategy_id"]
        or independent["reviewed_method_sha256"] != _digest(method_path)
        or independent["source_sha256"] != source["sha256"]
        or independent["core"].get("level") != "A2"
        or independent["core"].get("claim_indices") != [0]
        or independent["strategy"].get("level") != "A0"
        or any(len(independent[scope].get("scope", "")) < 40 for scope in ("core", "strategy"))
        or independent["data"] != "D0"
        or independent["experiment"] != "E0"
        or method["experimental_fidelity"] != "not_reproduced"
    ):
        raise ValueError("Independent paper grade differs from its reviewed build inputs")
    return {
        "matrix_sha256": _digest(matrix_path),
        "assessment_kind": matrix["assessment_kind"],
        "catalog_cases": len(cases),
        "core_counts": dict(
            sorted(Counter(case["core"]["level"] for case in cases.values()).items())
        ),
        "strategy_counts": dict(
            sorted(Counter(case["strategy"]["level"] for case in cases.values()).items())
        ),
        "data_counts": dict(sorted(Counter(case["data"] for case in cases.values()).items())),
        "experiment_counts": dict(
            sorted(Counter(case["experiment"] for case in cases.values()).items())
        ),
        "public_case_ids": sorted(public_ids),
        "independent": {
            "strategy_id": independent["strategy_id"],
            "core_level": independent["core"]["level"],
            "strategy_level": independent["strategy"]["level"],
            "data_level": independent["data"],
            "experiment_level": independent["experiment"],
        },
    }
