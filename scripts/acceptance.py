"""Run the declared 6×3 strategy catalog and publish a fail-closed evidence index."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from paperquant.backtrader_engine import BacktraderEngine
from paperquant.cli import main as cli
from paperquant.engine import ReferenceEngine
from paperquant.evidence import verify_bundle_file
from paperquant.models import (
    DatasetDeclaration,
    MarketEvent,
    RunPolicy,
    RunReport,
    TrainingRequest,
)
from paperquant.output import atomic_bytes
from paperquant.package import inspect_package, replay_bundle, run_package
from paperquant.papers import MethodSpec, Recipe, verify_recipe

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CASES = (
    ("rule.moving_average_crossover", "public_aapl"),
    ("rule.channel_breakout", "public_aapl"),
    ("rule.time_sliced_execution", "public_aapl"),
    ("supervised.logistic_direction", "public_direction"),
    ("supervised.ridge_three_day_price", "public_ridge"),
    ("supervised.gaussian_nb_direction", "public_gaussian"),
    ("supervised.random_forest_operations", "public_forest"),
    ("reinforcement.actor_critic_allocation", "public_actor_critic"),
)
NATIVE_CASES = (
    ("rule.moving_average_crossover", "public_aapl"),
    ("supervised.logistic_direction", "public_direction"),
    ("reinforcement.actor_critic_allocation", "public_actor_critic"),
)
EXTERNAL_PAPER_NODES = (
    "tests/test_paper_intake.py::"
    "test_unlisted_paper_package_enters_runtime_without_catalog_changes[12-3]",
    "tests/test_paper_intake.py::"
    "test_unlisted_paper_package_enters_runtime_without_catalog_changes[10-5]",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _check_implementation(claim: dict, strategy_id: str) -> None:
    source = ast.parse((ROOT / "strategies" / strategy_id / "strategy.py").read_text())
    classes = {node.name: node for node in source.body if isinstance(node, ast.ClassDef)}
    for statement in claim["claims"]:
        text = statement["implementation"]
        matches = re.findall(r"\b([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", text)
        if not matches and "scripts/" not in text:
            raise ValueError(f"{strategy_id} claim has no implementation symbol")
        for class_name, member in matches:
            node = classes.get(class_name)
            if node is None:
                raise ValueError(f"{strategy_id} implementation class is absent: {class_name}")
            named = {part.name for part in node.body
                     if isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef))}
            assigned = {
                target.id for part in ast.walk(node)
                if isinstance(part, (ast.Assign, ast.AnnAssign))
                for target in (part.targets if isinstance(part, ast.Assign) else [part.target])
                if isinstance(target, ast.Name)
            }
            attributes = {
                target.attr for part in ast.walk(node)
                if isinstance(part, (ast.Assign, ast.AnnAssign))
                for target in (part.targets if isinstance(part, ast.Assign) else [part.target])
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            }
            if member not in named | assigned | attributes:
                raise ValueError(f"{strategy_id} implementation member is absent: {member}")
        for script in re.findall(r"scripts/[A-Za-z_0-9/]+\.py", text):
            if not (ROOT / script).is_file():
                raise ValueError(f"{strategy_id} implementation script is absent: {script}")


def _run_oracles(output: Path, bindings: dict[str, dict],
                 *, retain_artifacts: bool = False) -> str:
    output.mkdir(parents=True, exist_ok=True)
    nodes = [node for binding in bindings.values() for node in binding["node_ids"]]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("oracle bindings contain no nodes or duplicate nodes")
    junit = output / "oracle-junit.xml"
    command = [sys.executable, "-m", "pytest", "-q", *nodes, "--junitxml", str(junit)]
    if retain_artifacts:
        command.extend(["--basetemp", str(output / "case-artifacts")])
    completed = subprocess.run(
        command,
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    (output / "oracle-run.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8",
    )
    if completed.returncode != 0 or not junit.is_file():
        raise ValueError("bound oracle nodes did not pass this acceptance attempt")
    cases = ET.parse(junit).findall(".//testcase")
    actual = {
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        for case in cases
    }
    if (actual != set(nodes) or len(cases) != len(nodes)
        or any(case.find(tag) is not None for case in cases
               for tag in ("failure", "error", "skipped"))):
        raise ValueError("JUnit nodes differ from passed oracle bindings")
    return digest(junit)


def _run_paper_case(recipe: Path, entry: dict, output: Path, checked: dict) -> str:
    fixture = ROOT / "examples" / entry["fixture"]
    destination = output / "paper_cases" / entry["strategy"]
    policy = json.loads((fixture / entry["policy"]).read_text()) if "policy" in entry else {}
    policy["sandbox"] = "development"
    destination.mkdir(parents=True, exist_ok=True)
    policy_file = destination / "policy.json"
    policy_file.write_text(json.dumps(policy, sort_keys=True) + "\n", encoding="utf-8")
    arguments = [
        "paper-run", "--recipe", str(recipe),
        "--dataset", str(fixture / "dataset.json"),
        "--events", str(fixture / "events.json"),
        "--policy", str(policy_file), "--output", str(destination / "run"),
    ]
    if "training" in entry:
        arguments.extend(["--training", str(fixture / entry["training"])])
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        exit_code = cli(arguments)
    if exit_code != 0:
        raise RuntimeError(f"{entry['strategy']} paper-run failed: {stream.getvalue()}")
    receipt_path = destination / "run/paper-run.json"
    receipt = json.loads(receipt_path.read_text())
    linked = RunReport.model_validate_json((destination / "run/report.json").read_bytes())
    standard = RunReport.model_validate_json(
        (output / entry["strategy"] / "host/run/report.json").read_bytes())
    if (receipt["status"] != "passed" or receipt["strategy_id"] != entry["strategy"]
        or receipt["method_spec_sha256"] != checked["method_spec_sha256"]
        or receipt["bundle_sha256"] != digest(destination / "run/bundle.json")
        or (linked.decisions, linked.orders, linked.fills, linked.accounts)
           != (standard.decisions, standard.orders, standard.fills, standard.accounts)):
        raise ValueError("paper-run receipt differs from the catalog execution")
    return digest(receipt_path)


def _run_case(
    *, entry: dict, output: Path, mode: str, expected_image_id: str | None
) -> tuple[RunReport, str]:
    strategy_id = entry["strategy"]
    fixture = ROOT / "examples" / entry["fixture"]
    package = ROOT / "strategies" / strategy_id
    destination = output / strategy_id / mode
    destination.mkdir(parents=True, exist_ok=True)
    policy = json.loads((fixture / entry["policy"]).read_text()) if "policy" in entry else {}
    policy["sandbox"] = "strict" if mode == "strict" else "development"
    policy_file = destination / "policy.json"
    policy_file.write_text(json.dumps(policy, sort_keys=True) + "\n", encoding="utf-8")
    args = [
        "run", "--package", str(package), "--dataset", str(fixture / "dataset.json"),
        "--events", str(fixture / "events.json"), "--policy", str(policy_file),
        "--output", str(destination / "run"),
    ]
    if "training" in entry:
        args.extend(["--training", str(fixture / entry["training"])])
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        exit_code = cli(args)
    if exit_code != 0:
        raise RuntimeError(f"{strategy_id} {mode} failed: {stream.getvalue().strip()}")
    report_path = destination / "run/report.json"
    report = RunReport.model_validate_json(report_path.read_bytes())
    verify_bundle_file(destination / "run/bundle.json")
    if mode == "host":
        settings = report.plan.engine_profile.settings
        replay_bundle(
            bundle_path=destination / "run/bundle.json", package_dir=package,
            engine=ReferenceEngine(initial_cash=Decimal(settings["initial_cash"]),
                                   fee_rate=Decimal(settings["fee_rate"])),
        )
    if report.plan.strategy_id != strategy_id or not report.decisions or not report.fills:
        raise RuntimeError(f"{strategy_id} {mode} has no complete execution evidence")
    if mode == "strict":
        if report.worker_receipt is None or report.worker_receipt.image_id != expected_image_id:
            raise RuntimeError(f"{strategy_id} strict worker image differs from build")
    elif report.worker_receipt is not None:
        raise RuntimeError(f"{strategy_id} host run contains worker evidence")
    return report, digest(report_path)


def _run_native_case(strategy_id: str, fixture_name: str, output: Path,
                     mode: str) -> RunReport:
    fixture = ROOT / "examples" / fixture_name
    dataset = DatasetDeclaration.model_validate_json((fixture / "dataset.json").read_bytes())
    events = tuple(MarketEvent.model_validate(item) for item in json.loads(
        (fixture / "events.json").read_text()))
    policy = RunPolicy.model_validate_json((fixture / "policy.json").read_bytes())
    policy = policy.model_copy(update={"sandbox": "strict" if mode == "strict"
                                else "development"})
    training_path = fixture / "training.json"
    training = (TrainingRequest.model_validate_json(training_path.read_bytes())
                if training_path.is_file() else None)
    engine = BacktraderEngine()
    destination = output / strategy_id / mode
    report = run_package(
        run_id=f"native-{strategy_id}", package_dir=ROOT / "strategies" / strategy_id,
        dataset=dataset, engine_capability=engine.capability(dataset.fields),
        policy=policy, events=events, engine=engine, output=destination,
        training=training,
    )
    verify_bundle_file(destination / "bundle.json")
    replay_bundle(bundle_path=destination / "bundle.json",
                  package_dir=ROOT / "strategies" / strategy_id,
                  engine=BacktraderEngine())
    if report.plan.engine_id != "paperquant.backtrader" or not report.fills:
        raise ValueError(f"{strategy_id} native execution did not produce real fills")
    return report


def _check_native_differential(native: RunReport, reference: RunReport) -> None:
    tolerance = Decimal("0.000001")
    if native.decisions != reference.decisions or len(native.fills) != len(reference.fills):
        raise ValueError("Native and reference decisions or fill counts differ")
    for actual, expected in zip(native.fills, reference.fills, strict=True):
        if ((actual.order_id, actual.symbol, actual.side, actual.timestamp)
            != (expected.order_id, expected.symbol, expected.side, expected.timestamp)
            or abs(actual.quantity - expected.quantity) > tolerance
            or abs(actual.price - expected.price) > tolerance):
            raise ValueError("Native and reference order execution differs")
    for actual, expected in zip(native.accounts, reference.accounts, strict=True):
        if (abs(actual.cash - expected.cash) > tolerance
            or abs(actual.equity - expected.equity) > tolerance
            or len(actual.positions) != len(expected.positions)):
            raise ValueError("Native and reference account ledgers differ")
        for native_position, reference_position in zip(
            actual.positions, expected.positions, strict=True,
        ):
            if (native_position.symbol != reference_position.symbol
                or abs(native_position.quantity - reference_position.quantity) > tolerance
                or abs(native_position.average_price - reference_position.average_price)
                   > tolerance
                or abs(native_position.realized_pnl - reference_position.realized_pnl)
                   > tolerance
                or abs(native_position.unrealized_pnl - reference_position.unrealized_pnl)
                   > tolerance):
                raise ValueError("Native and reference position profits differ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-image-id", default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "acceptance.json"
    state: dict = {"status": "running", "started_at": datetime.now(UTC).isoformat()}
    publish(receipt, state)
    try:
        catalog = json.loads((ROOT / "examples/catalog.json").read_text())
        entries = catalog["cases"]
        if catalog["schema_version"] != "1.0" or len(entries) != 18:
            raise ValueError("catalog must contain exactly eighteen cases")
        ids = [entry["strategy"] for entry in entries]
        if len(set(ids)) != len(ids) or Counter(value.split(".")[0] for value in ids) != {
            "rule": 6, "supervised": 6, "reinforcement": 6,
        }:
            raise ValueError("catalog must contain six distinct strategies per family")
        candidate_ids = {entry["id"] for entry in json.loads(
            (ROOT / "research/candidates.json").read_text())["entries"]}
        package_ids = {directory.name for directory in (ROOT / "strategies").iterdir()
                       if directory.is_dir()}
        if set(ids) != candidate_ids or set(ids) != package_ids:
            raise ValueError("catalog, candidates and runnable packages differ")
        paper_checks = []
        recipe_paths = sorted((ROOT / "research/recipes").glob("*.json"))
        if len(recipe_paths) < 3:
            raise ValueError("three real-paper intake recipes are required")
        for recipe_path in recipe_paths:
            checked = verify_recipe(recipe_path)
            if checked["strategy_id"] not in ids:
                raise ValueError("paper recipe has no catalog strategy")
            paper_checks.append(checked)
        if (len({check["source_sha256"] for check in paper_checks}) < 3
            or len({check["strategy_id"] for check in paper_checks}) < 3
            or {check["strategy_id"].split(".", 1)[0] for check in paper_checks}
               != {"rule", "supervised", "reinforcement"}):
            raise ValueError("real-paper cases must cover three distinct sources and families")
        records = []
        bindings = json.loads((ROOT / "research/oracle_bindings.json").read_text())["bindings"]
        if set(bindings) != set(ids):
            raise ValueError("oracle bindings do not cover the declared catalog")
        for recipe_path, checked in zip(recipe_paths, paper_checks, strict=True):
            recipe = Recipe.model_validate_json(recipe_path.read_bytes())
            spec = MethodSpec.model_validate_json(
                (recipe_path.parent / recipe.spec_file).read_bytes())
            bound_claim_nodes = bindings[checked["strategy_id"]]["claim_nodes"]
            if any(step.oracle_node not in bound_claim_nodes[step.claim_index]
                   for step in spec.steps):
                raise ValueError("Reviewed method step is not bound to its claim oracle")
        for entry in entries:
            strategy_id = entry["strategy"]
            state["current_case"] = strategy_id
            publish(receipt, state)
            manifest, _ = inspect_package(strategy_id, ROOT / "strategies" / strategy_id)
            claim_path = ROOT / "research/claims" / (strategy_id.split(".", 1)[1] + ".json")
            claim = json.loads(claim_path.read_text())
            if claim["strategy_id"] != strategy_id or not claim["claims"]:
                raise ValueError(f"{strategy_id} has no bound research claim")
            if not (ROOT / claim["oracle"]).is_file():
                raise ValueError(f"{strategy_id} has no independent oracle test")
            _check_implementation(claim, strategy_id)
            binding = bindings[strategy_id]
            if (binding["claim_sha256"] != digest(claim_path)
                or binding["source_sha256"] != manifest.source_sha256
                or binding["oracle_sha256"] != digest(ROOT / claim["oracle"])
                or not binding["node_ids"]
                or len(binding.get("claim_nodes", ())) != len(claim["claims"])
                or any(not group or not set(group) <= set(binding["node_ids"])
                       for group in binding["claim_nodes"])
                or any(not node.startswith(claim["oracle"] + "::")
                       for node in binding["node_ids"])):
                raise ValueError(f"{strategy_id} oracle binding differs from source or claim")
            source = claim.get("source", claim.get("trading_source"))
            if source is None or source["url"] != manifest.declaration.source:
                raise ValueError(f"{strategy_id} research source differs from manifest")
            host, host_digest = _run_case(entry=entry, output=output, mode="host",
                                          expected_image_id=None)
            if (host.artifact is None) != (not manifest.declaration.training_required):
                raise ValueError(f"{strategy_id} has inconsistent training artifact")
            record = {
                "strategy_id": strategy_id,
                "kind": manifest.declaration.kind,
                "source_sha256": manifest.source_sha256,
                "claim_sha256": digest(claim_path),
                "oracle": claim["oracle"],
                "oracle_nodes": binding["node_ids"],
                "claim_oracle_nodes": binding["claim_nodes"],
                "dataset_sha256": host.plan.dataset_sha256,
                "training_sha256": host.plan.training_sha256,
                "host_report_sha256": host_digest,
                "host_bundle_sha256": digest(output / strategy_id / "host/run/bundle.json"),
                "decision_count": len(host.decisions),
                "fill_count": len(host.fills),
                "final_equity": str(host.final_equity),
            }
            if args.strict_image_id is not None:
                strict, strict_digest = _run_case(
                    entry=entry, output=output, mode="strict",
                    expected_image_id=args.strict_image_id,
                )
                if (host.decisions, host.orders, host.fills, host.accounts,
                    host.final_equity) != (strict.decisions, strict.orders, strict.fills,
                                           strict.accounts, strict.final_equity):
                    raise RuntimeError(f"{strategy_id} host/strict execution paths differ")
                if host.artifact is not None and strict.artifact is not None and (
                    host.artifact.content_sha256 != strict.artifact.content_sha256
                ):
                    raise RuntimeError(f"{strategy_id} host/strict model bytes differ")
                record["strict_report_sha256"] = strict_digest
                record["strict_bundle_sha256"] = digest(
                    output / strategy_id / "strict/run/bundle.json")
            records.append(record)
        oracle_junit_sha256 = _run_oracles(output, bindings)
        external_dir = output / "external_paper_oracles"
        external_junit_sha256 = _run_oracles(
            external_dir, {"external": {"node_ids": EXTERNAL_PAPER_NODES}},
            retain_artifacts=True)
        external_paper_cases = []
        external_receipts = sorted(
            (external_dir / "case-artifacts").rglob("paper-run.json"))
        if len(external_receipts) != len(EXTERNAL_PAPER_NODES):
            raise ValueError("Unlisted real-paper executions are missing")
        for paper_receipt_path in external_receipts:
            paper_receipt = json.loads(paper_receipt_path.read_text())
            bundle_path = paper_receipt_path.parent / "bundle.json"
            bundle = verify_bundle_file(bundle_path)
            external_recipe_path = paper_receipt_path.parent.parent / "recipe.json"
            external_recipe = Recipe.model_validate_json(external_recipe_path.read_bytes())
            external_spec = MethodSpec.model_validate_json(
                (external_recipe_path.parent / external_recipe.spec_file).read_bytes())
            if (paper_receipt["status"] != "passed"
                or not paper_receipt["strategy_id"].startswith("outside.catalog.")
                or paper_receipt["source_sha256"] not in {
                    check["source_sha256"] for check in paper_checks}
                or paper_receipt["method_spec_sha256"] == ""
                or any(step.oracle_node not in EXTERNAL_PAPER_NODES
                       for step in external_spec.steps)
                or paper_receipt["bundle_sha256"] != digest(bundle_path)
                or not bundle.report.fills):
                raise ValueError("Unlisted paper-run bundle or source evidence differs")
            external_paper_cases.append({
                "strategy_id": paper_receipt["strategy_id"],
                "source_sha256": paper_receipt["source_sha256"],
                "method_spec_sha256": paper_receipt["method_spec_sha256"],
                "paper_run_sha256": digest(paper_receipt_path),
                "bundle_sha256": digest(bundle_path),
                "fill_count": len(bundle.report.fills),
                "oracle_junit_sha256": external_junit_sha256,
            })
        for checked, recipe_path in zip(paper_checks, recipe_paths, strict=True):
            entry = next(item for item in entries if item["strategy"] == checked["strategy_id"])
            checked["paper_run_sha256"] = _run_paper_case(recipe_path, entry, output, checked)
            checked["oracle_junit_sha256"] = oracle_junit_sha256
        source_record = json.loads((ROOT / "data/public/SOURCE.json").read_text())
        raw_csv_sha256 = digest(ROOT / "data/public/finance-charts-apple.csv")
        if source_record["sha256"] != raw_csv_sha256:
            raise ValueError("public AAPL source bytes differ from their declared provenance")
        public_cases = []
        public_reports: dict[str, RunReport] = {}
        for strategy_id, fixture_name in PUBLIC_CASES:
            fixture = ROOT / "examples" / fixture_name
            entry = {"strategy": strategy_id, "fixture": fixture_name,
                     "policy": "policy.json"}
            if (fixture / "training.json").is_file():
                entry["training"] = "training.json"
            report, report_sha256 = _run_case(
                entry=entry, output=output / "public_cases", mode="host",
                expected_image_id=None,
            )
            public_reports[strategy_id] = report
            lineage_path = fixture / "lineage.json"
            lineage_sha256 = None
            if lineage_path.is_file():
                lineage = json.loads(lineage_path.read_text())
                if (lineage["source_file_sha256"] != raw_csv_sha256
                    or lineage["evaluation_sha256"] != report.source_events_sha256
                    or lineage["training_sha256"] != report.plan.training_sha256):
                    raise ValueError(f"{strategy_id} public-data lineage differs from its run")
                lineage_sha256 = digest(lineage_path)
            public_cases.append({
                "strategy_id": strategy_id,
                "source_csv_sha256": raw_csv_sha256,
                "lineage_sha256": lineage_sha256,
                "source_events_sha256": report.source_events_sha256,
                "training_sha256": report.plan.training_sha256,
                "report_sha256": report_sha256,
                "bundle_sha256": digest(output / "public_cases" / strategy_id
                                        / "host/run/bundle.json"),
                "fill_count": len(report.fills),
            })
        native_cases = []
        for strategy_id, fixture_name in NATIVE_CASES:
            native_root = output / "native_engine_cases"
            native = _run_native_case(strategy_id, fixture_name, native_root, "host")
            _check_native_differential(native, public_reports[strategy_id])
            native_record = {
                "strategy_id": strategy_id,
                "engine_profile_sha256": native.plan.engine_profile_sha256,
                "host_report_sha256": digest(native_root / strategy_id / "host/report.json"),
                "host_bundle_sha256": digest(native_root / strategy_id / "host/bundle.json"),
                "fill_count": len(native.fills),
                "differential_tolerance": "0.000001",
            }
            if args.strict_image_id is not None:
                strict_native = _run_native_case(
                    strategy_id, fixture_name, native_root, "strict")
                _check_native_differential(strict_native, public_reports[strategy_id])
                if (strict_native.decisions != native.decisions
                    or len(strict_native.fills) != len(native.fills)
                    or strict_native.worker_receipt is None
                    or strict_native.worker_receipt.image_id != args.strict_image_id):
                    raise ValueError("Native-engine host and strict worker paths differ")
                native_record["strict_report_sha256"] = digest(
                    native_root / strategy_id / "strict/report.json")
                native_record["strict_bundle_sha256"] = digest(
                    native_root / strategy_id / "strict/bundle.json")
            native_cases.append(native_record)
        schedule_recipe = next(path for path in recipe_paths
                               if Recipe.model_validate_json(path.read_bytes())
                               .expected.strategy_id == "rule.time_sliced_execution")
        schedule_fixture = ROOT / "examples/public_aapl"
        native_paper_output = output / "paper_cases/native_time_sliced"
        with contextlib.redirect_stdout(io.StringIO()) as paper_stdout:
            exit_code = cli([
                "paper-run", "--engine", "backtrader",
                "--recipe", str(schedule_recipe),
                "--dataset", str(schedule_fixture / "dataset.json"),
                "--events", str(schedule_fixture / "events.json"),
                "--policy", str(schedule_fixture / "policy.json"),
                "--output", str(native_paper_output),
            ])
        if exit_code != 0:
            raise ValueError("Native paper-to-strategy run failed: " + paper_stdout.getvalue())
        native_paper = RunReport.model_validate_json(
            (native_paper_output / "report.json").read_bytes())
        _check_native_differential(
            native_paper, public_reports["rule.time_sliced_execution"])
        paper_receipt = json.loads((native_paper_output / "paper-run.json").read_text())
        schedule_check = next(item for item in paper_checks
                              if item["strategy_id"] == "rule.time_sliced_execution")
        if (paper_receipt["method_spec_sha256"] != schedule_check["method_spec_sha256"]
            or paper_receipt["bundle_sha256"]
               != digest(native_paper_output / "bundle.json")):
            raise ValueError("Native paper-run evidence is not bound to its reviewed spec")
        schedule_check["native_paper_run_sha256"] = digest(
            native_paper_output / "paper-run.json")
        state.pop("current_case", None)
        state.update({"status": "passed", "completed_at": datetime.now(UTC).isoformat(),
                      "count": len(records), "strict": args.strict_image_id is not None,
                      "cases": records, "paper_checks": paper_checks,
                      "public_data_cases": public_cases,
                      "native_engine_cases": native_cases,
                      "external_paper_cases": external_paper_cases,
                      "external_paper_oracle_junit_sha256": external_junit_sha256,
                      "oracle_junit_sha256": oracle_junit_sha256})
        publish(receipt, state)
        print(json.dumps({"status": "passed", "count": len(records), "receipt": str(receipt)}))
        return 0
    except Exception as exc:
        state.update({"status": "failed", "completed_at": datetime.now(UTC).isoformat(),
                      "cause": type(exc).__name__, "message": str(exc)})
        publish(receipt, state)
        print(json.dumps({"status": "failed", "receipt": str(receipt),
                          "message": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
