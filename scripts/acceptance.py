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

from paperquant.cli import main as cli
from paperquant.engine import ReferenceEngine
from paperquant.evidence import verify_bundle_file
from paperquant.models import RunReport
from paperquant.output import atomic_bytes
from paperquant.package import inspect_package, replay_bundle
from paperquant.papers import verify_recipe

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


def _run_oracles(output: Path, bindings: dict[str, dict]) -> str:
    nodes = [node for binding in bindings.values() for node in binding["node_ids"]]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("oracle bindings contain no nodes or duplicate nodes")
    junit = output / "oracle-junit.xml"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *nodes, "--junitxml", str(junit)],
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


def _run_paper_case(recipe: Path, entry: dict, output: Path) -> str:
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
        for checked, recipe_path in zip(paper_checks, recipe_paths, strict=True):
            entry = next(item for item in entries if item["strategy"] == checked["strategy_id"])
            checked["paper_run_sha256"] = _run_paper_case(recipe_path, entry, output)
            checked["oracle_junit_sha256"] = oracle_junit_sha256
        source_record = json.loads((ROOT / "data/public/SOURCE.json").read_text())
        raw_csv_sha256 = digest(ROOT / "data/public/finance-charts-apple.csv")
        if source_record["sha256"] != raw_csv_sha256:
            raise ValueError("public AAPL source bytes differ from their declared provenance")
        public_cases = []
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
        state.pop("current_case", None)
        state.update({"status": "passed", "completed_at": datetime.now(UTC).isoformat(),
                      "count": len(records), "strict": args.strict_image_id is not None,
                      "cases": records, "paper_checks": paper_checks,
                      "public_data_cases": public_cases,
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
