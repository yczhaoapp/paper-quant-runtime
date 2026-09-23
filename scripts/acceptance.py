"""Run the declared 6×3 strategy catalog and publish a fail-closed evidence index."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from paperquant.cli import main as cli
from paperquant.models import RunReport
from paperquant.output import atomic_bytes
from paperquant.package import inspect_package
from paperquant.papers import verify_recipe

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


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
        if not recipe_paths:
            raise ValueError("no real-paper intake recipe is available")
        for recipe_path in recipe_paths:
            checked = verify_recipe(recipe_path)
            if checked["strategy_id"] not in ids:
                raise ValueError("paper recipe has no catalog strategy")
            paper_checks.append(checked)
        records = []
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
                "dataset_sha256": host.plan.dataset_sha256,
                "training_sha256": host.plan.training_sha256,
                "host_report_sha256": host_digest,
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
            records.append(record)
        state.pop("current_case", None)
        state.update({"status": "passed", "completed_at": datetime.now(UTC).isoformat(),
                      "count": len(records), "strict": args.strict_image_id is not None,
                      "cases": records, "paper_checks": paper_checks})
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
