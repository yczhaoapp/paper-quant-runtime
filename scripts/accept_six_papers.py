"""Fail-closed acceptance for the six reviewed, bounded paper adaptations."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from paperquant.cli import main as cli
from paperquant.evidence import verify_bundle_file
from paperquant.models import RunReport
from paperquant.output import atomic_bytes
from paperquant.papers import verify_recipe
from scripts.acceptance import _check_paper_run_scope, _run_oracles
from scripts.gate_support import check_source_identity, new_attempt, source_identity, write_json
from scripts.prepare_six_sources import prepare
from scripts.two_paper_quality import check_contract_lock, quality_for_run

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "research/six-paper/catalog.json"
INPUT_LOCK = ROOT / "research/six-paper/input-lock.json"
QUOTE_NODES = [
    "tests/test_six_quote_contract.py::" + name
    for name in (
        "test_adverse_fills_and_complete_account_ledger_match_independent_arithmetic",
        "test_nonadverse_flow_respects_probability_boundary_and_no_same_event_fill",
        "test_cancelled_quote_cannot_fill_on_later_cross",
        "test_unknown_cancel_and_duplicate_quote_fail_structurally",
        "test_legacy_engine_refuses_cancel_semantics_before_running",
        "test_direct_passive_engine_is_bound_to_plan_strategy_and_event_bytes",
    )
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def check_inputs():
    pinned = json.loads(INPUT_LOCK.read_text())["files"]
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "examples/six-paper").rglob("*.json")}
    if actual != set(pinned):
        raise ValueError("Supplemental input set differs from its lock")
    for name, digest in pinned.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Supplemental input changed: " + name)
    market = json.loads((ROOT / "data/six-paper/public-btc/SOURCE.json").read_text())
    for record in market["sources"]:
        if sha(ROOT / "data/six-paper/public-btc" / record["csv"]) != record["csv_sha256"]:
            raise ValueError("Public market CSV differs from official archive preparation")


def expanded_oracles(nodes):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *nodes],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise ValueError("Declared method oracle could not be collected")
    actual = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    if (
        not actual
        or len(actual) != len(set(actual))
        or any(
            not any(node == parent or node.startswith(parent + "[") for parent in nodes)
            for node in actual
        )
        or any(
            not any(node == parent or node.startswith(parent + "[") for node in actual)
            for parent in nodes
        )
    ):
        raise ValueError("Expanded oracle set differs from its declared parent nodes")
    return actual


def invoke(arguments, folder):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli(arguments)
    atomic_bytes(folder / "stdout.txt", stdout.getvalue().encode())
    atomic_bytes(folder / "stderr.txt", stderr.getvalue().encode())
    write_json(
        folder / "invocation.json",
        {"entrypoint": "python -m paperquant.cli", "argv": arguments, "exit_code": code},
    )
    return code


def accept(output, *, strict=False, require_clean=False):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = new_attempt()
    state.update(
        status_scope="source_mapping_contract_runtime_local_oracles_and_cold_replay",
        provenance=("new supplemental execution; original six-PDF extraction records retained"),
        original_sealed_commit="747460544d5dabcafa9421031bdbc97a1bf28282",
        strict=strict,
    )
    receipt = output / "six-paper.json"
    write_json(receipt, state)
    attempt = output / "attempts" / state["attempt_id"]
    try:
        attempt.mkdir(parents=True)
        identity = source_identity(ROOT, require_clean=require_clean)
        state.update(identity)
        state["contract_preservation"] = check_contract_lock()
        semantics = ROOT / "research/six-paper/quote-semantics.json"
        interpretation = json.loads(semantics.read_text())
        if sha(ROOT / "src/paperquant/passive_engine.py") != interpretation["engine_source_sha256"]:
            raise ValueError("Quote simulator differs from its pinned semantic declaration")
        state["passive_quote_semantics"] = interpretation
        state["passive_quote_semantics_sha256"] = sha(semantics)
        state["sources"] = prepare(fetch=False)
        check_inputs()
        catalog = json.loads(CATALOG.read_text())["cases"]
        if (
            len(catalog) != 6
            or len({c["paper_id"] for c in catalog}) != 6
            or len({c["strategy_id"] for c in catalog}) != 6
        ):
            raise ValueError("Six distinct paper-specific adaptations are required")
        base_ids = {
            case["strategy"]
            for case in json.loads((ROOT / "examples/catalog.json").read_text())["cases"]
        }
        actual_packages = {
            directory.name for directory in (ROOT / "strategies").iterdir() if directory.is_dir()
        }
        if actual_packages != base_ids | {case["strategy_id"] for case in catalog}:
            raise ValueError("An unlisted package cannot be omitted from supplemental evidence")
        sources = {source["id"]: source for source in state["sources"]}
        for case in catalog:
            checked = verify_recipe(ROOT / case["recipe"])
            if (
                checked["strategy_id"] != case["strategy_id"]
                or checked["source_sha256"] != sources[case["paper_id"]]["sha256"]
                or checked["source_url"] != sources[case["paper_id"]]["url"]
                or set(checked["method_oracle_nodes"]) != set(case["oracle_nodes"])
            ):
                raise ValueError("Catalog, evaluator source or method oracle binding mismatch")
        bindings = {
            c["strategy_id"]: {"node_ids": expanded_oracles(c["oracle_nodes"])} for c in catalog
        }
        bindings["quote-contract"] = {"node_ids": expanded_oracles(QUOTE_NODES + [
            "tests/test_six_audit_refinement.py::test_interval_probability_simulator_explicitly_allows_off_touch_resting_fill"
        ])}
        oracle_sha = _run_oracles(attempt / "oracles", bindings)
        state["method_oracle_junit_sha256"] = oracle_sha
        image_id = None
        if strict:
            image_id = subprocess.check_output(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    "paperquant-worker:six-paper-0.2",
                ],
                text=True,
                timeout=30,
            ).strip()
        rows = []
        for case in catalog:
            state["current_case"] = case["strategy_id"]
            write_json(receipt, state)
            checked = verify_recipe(ROOT / case["recipe"])
            fixture = ROOT / case["fixture"]
            row = {
                "paper_id": case["paper_id"],
                "strategy_id": case["strategy_id"],
                "source_mapping": "passed",
                "local_method_oracles": "passed",
                "paper_experiment_reproduced": False,
                "modes": {},
            }
            reports = {}
            for mode in ("host", "strict") if strict else ("host",):
                run_dir = attempt / case["strategy_id"] / mode
                run_dir.mkdir(parents=True)
                policy = json.loads((fixture / "policy.json").read_text())
                policy["sandbox"] = "strict" if mode == "strict" else "development"
                write_json(run_dir / "policy.json", policy)
                command = [
                    "paper-run",
                    "--recipe",
                    case["recipe"],
                    "--dataset",
                    case["fixture"] + "/dataset.json",
                    "--events",
                    case["fixture"] + "/events.json",
                    "--policy",
                    relative(run_dir / "policy.json"),
                    "--engine",
                    case["engine"],
                    "--fee-rate",
                    case["fee_rate"],
                    "--output",
                    relative(run_dir / "run"),
                ]
                if case["training_required"]:
                    command.extend(["--training", case["fixture"] + "/training.json"])
                if invoke(command, run_dir) != 0:
                    raise ValueError("Paper runtime failed: " + case["strategy_id"] + "/" + mode)
                bundle = verify_bundle_file(run_dir / "run/bundle.json")
                report = RunReport.model_validate_json((run_dir / "run/report.json").read_bytes())
                paper_receipt = json.loads((run_dir / "run/paper-run.json").read_text())
                _check_paper_run_scope(paper_receipt)
                if (
                    report != bundle.report
                    or paper_receipt["source_sha256"] != checked["source_sha256"]
                    or not report.decisions
                    or report.status != "succeeded"
                ):
                    raise ValueError("Supplemental execution evidence is incomplete")
                if strict and mode == "strict":
                    if report.worker_receipt is None or report.worker_receipt.image_id != image_id:
                        raise ValueError("Strict run is not bound to this worker image")
                training_metrics = None
                if report.artifact:
                    model_path = run_dir / "run" / report.artifact.relative_path
                    if sha(model_path) != report.artifact.content_sha256:
                        raise ValueError("Model differs from report")
                    training_metrics = json.loads(model_path.read_bytes()).get("training")
                row["modes"][mode] = {
                    "runtime": "succeeded",
                    "cold_replay": "passed",
                    "fills": len(report.fills),
                    "decisions": len(report.decisions),
                    "orders": len(report.orders),
                    "final_equity": str(report.final_equity),
                    "training": "executed_and_reloaded"
                    if report.artifact
                    else "not_applicable_rule",
                    "training_sha256": report.plan.training_sha256,
                    "training_metrics": training_metrics,
                    "paper_run_sha256": sha(run_dir / "run/paper-run.json"),
                    "bundle_sha256": sha(run_dir / "run/bundle.json"),
                    "report_sha256": sha(run_dir / "run/report.json"),
                    "zero_trades": not bool(report.fills),
                    "adaptation_quality": quality_for_run(run_dir / "run", report, bundle),
                }
                reports[mode] = report
            if strict:
                left, right = reports["host"], reports["strict"]
                if (left.decisions, left.orders, left.fills, left.accounts) != (
                    right.decisions,
                    right.orders,
                    right.fills,
                    right.accounts,
                ):
                    raise ValueError("Host and strict traces differ: " + case["strategy_id"])
                row["host_strict_trace_comparison"] = "passed"
            rows.append(row)
        del state["current_case"]
        state["cases"] = rows
        state["counts"] = {
            "source_papers": 6,
            "strategy_adaptations": 6,
            "trained_cases": 5,
            "rule_cases": 1,
            "host_backtests": 6,
            "strict_backtests": 6 if strict else 0,
            "zero_trade_host_cases": sum(row["modes"]["host"]["zero_trades"] for row in rows),
        }
        state["research_limits"] = {
            "original_experiments_reproduced": False,
            "arbitrary_pdf_autogeneration": False,
            "pressure_testing_scope": "not part of this supplemental gate",
            "quality_and_trade_counts": "actual outcomes recorded separately from runtime status",
        }
        check_source_identity(ROOT, identity, require_clean=require_clean)
        state["status"] = "passed"
    except Exception as exc:
        state.update(status="failed", cause=type(exc).__name__, reason=str(exc)[:600])
    state["completed_at"] = datetime.now(UTC).isoformat()
    write_json(receipt, state)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    result = accept(args.output, strict=args.strict, require_clean=args.require_clean)
    print(json.dumps({"status": result["status"], "receipt": str(args.output / "six-paper.json")}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
