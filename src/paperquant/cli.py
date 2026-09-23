from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from paperquant.comparison import Comparison, Observation, compare, from_report
from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.evidence import verify_bundle_file
from paperquant.models import (
    AccountSnapshot,
    Artifact,
    ContractFault,
    Conversion,
    DatasetDeclaration,
    Decision,
    EngineCapability,
    EngineProfile,
    ErrorCode,
    ExecutionPlan,
    Failure,
    Fill,
    MarketEvent,
    OrderEvent,
    Position,
    RunBundle,
    RunPolicy,
    RunReport,
    Stage,
    StrategyDeclaration,
    TrainingRequest,
    WorkerReceipt,
)
from paperquant.output import Attempt, atomic_bytes
from paperquant.package import replay_bundle, run_package
from paperquant.papers import Anchor, ExpectedStrategy, Recipe, Source, verify_recipe

SCHEMAS = {
    model.__name__: model
    for model in (
        StrategyDeclaration,
        DatasetDeclaration,
        EngineCapability,
        EngineProfile,
        RunPolicy,
        ExecutionPlan,
        Conversion,
        MarketEvent,
        AccountSnapshot,
        Position,
        Decision,
        OrderEvent,
        Fill,
        TrainingRequest,
        Artifact,
        RunReport,
        RunBundle,
        Failure,
        Observation,
        Comparison,
        WorkerReceipt,
        Source,
        Anchor,
        ExpectedStrategy,
        Recipe,
    )
}


def export_schemas(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, model in sorted(SCHEMAS.items()):
        (output / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paperquant")
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema")
    schema.add_argument("--output", type=Path, required=True)
    paper_check = commands.add_parser("paper-check")
    paper_check.add_argument("--recipe", type=Path, required=True)
    paper_check.add_argument("--output", type=Path, required=True)
    paper_run = commands.add_parser("paper-run")
    paper_run.add_argument("--recipe", type=Path, required=True)
    paper_run.add_argument("--dataset", type=Path, required=True)
    paper_run.add_argument("--events", type=Path, required=True)
    paper_run.add_argument("--training", type=Path)
    paper_run.add_argument("--policy", type=Path)
    paper_run.add_argument("--output", type=Path, required=True)
    paper_run.add_argument("--cash", type=Decimal, default=Decimal("100000"))
    bundle_check = commands.add_parser("verify-bundle")
    bundle_check.add_argument("--bundle", type=Path, required=True)
    replay = commands.add_parser("replay-bundle")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--package", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--package", type=Path, required=True)
    execute.add_argument("--dataset", type=Path, required=True)
    execute.add_argument("--events", type=Path, required=True)
    execute.add_argument("--training", type=Path)
    execute.add_argument("--policy", type=Path)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--cash", type=Decimal, default=Decimal("100000"))
    observe = commands.add_parser("observe")
    observe.add_argument("--report", type=Path, required=True)
    observe.add_argument("--scenario", required=True)
    observe.add_argument("--method", required=True)
    observe.add_argument("--profile-sha256")
    observe.add_argument("--initial-cash", type=Decimal)
    observe.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--left", type=Path, required=True)
    comparison.add_argument("--right", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "schema":
        export_schemas(args.output)
        return 0
    if args.command == "verify-bundle":
        try:
            bundle = verify_bundle_file(args.bundle)
        except (OSError, ValueError, ValidationError) as exc:
            print(json.dumps({"status": "failed", "reason": str(exc)[:300]}), file=sys.stderr)
            return 2
        print(json.dumps({"status": "passed", "run_id": bundle.report.run_id}))
        return 0
    if args.command == "replay-bundle":
        try:
            bundle = verify_bundle_file(args.bundle)
            settings = bundle.report.plan.engine_profile.settings
            replay_bundle(
                bundle_path=args.bundle, package_dir=args.package,
                engine=ReferenceEngine(initial_cash=Decimal(settings["initial_cash"]),
                                       fee_rate=Decimal(settings["fee_rate"])),
            )
        except (ContractFault, OSError, ValueError, KeyError, ValidationError) as exc:
            print(json.dumps({"status": "failed", "reason": str(exc)[:300]}), file=sys.stderr)
            return 2
        print(json.dumps({"status": "passed", "run_id": bundle.report.run_id}))
        return 0
    if args.command == "paper-check":
        attempt = Attempt(args.output)
        try:
            checked = verify_recipe(args.recipe)
            atomic_bytes(args.output / "report.json",
                         (json.dumps(checked, indent=2, sort_keys=True) + "\n").encode())
        except Exception as exc:
            failure = Failure(
                run_id="paper-check", stage=Stage.INPUT, code=ErrorCode.INPUT_INVALID,
                message="Paper source or reviewed recipe could not be verified",
                details={"cause": type(exc).__name__, "reason": str(exc)},
            )
            attempt.failed(run_id=failure.run_id, code=failure.code,
                           failure=(failure.model_dump_json(indent=2) + "\n").encode())
            print(failure.model_dump_json(), file=sys.stderr)
            return 2
        attempt.succeeded(run_id="paper-check")
        print(json.dumps({"status": "succeeded", "report": str(args.output / "report.json")}))
        return 0
    if args.command == "paper-run":
        attempt = Attempt(args.output)
        try:
            checked = verify_recipe(args.recipe)
            recipe = Recipe.model_validate_json(args.recipe.read_bytes())
            package_dir = (args.recipe.resolve().parent / recipe.package).resolve(strict=True)
            events = TypeAdapter(tuple[MarketEvent, ...]).validate_python(_read_json(args.events))
            dataset = DatasetDeclaration.model_validate(_read_json(args.dataset))
            training = (TrainingRequest.model_validate(_read_json(args.training))
                        if args.training else None)
            policy = (
                RunPolicy.model_validate(_read_json(args.policy)) if args.policy else RunPolicy()
            )
            engine = ReferenceEngine(initial_cash=args.cash)
            report = run_package(
                run_id=f"paper-{fingerprint(events)[:12]}", package_dir=package_dir,
                dataset=dataset, engine_capability=reference_capability(dataset.fields),
                policy=policy, events=events, engine=engine, output=args.output,
                training=training,
            )
            replay_bundle(bundle_path=args.output / "bundle.json",
                          package_dir=package_dir, engine=engine)
            evidence = {
                "status": "passed", "run_id": report.run_id,
                "strategy_id": report.plan.strategy_id,
                "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
                "source_sha256": checked["source_sha256"],
                "claim_sha256": checked["claim_sha256"],
                "package_source_sha256": checked["package_source_sha256"],
                "bundle_sha256": hashlib.sha256(
                    (args.output / "bundle.json").read_bytes()).hexdigest(),
                "report_sha256": hashlib.sha256(
                    (args.output / "report.json").read_bytes()).hexdigest(),
            }
            atomic_bytes(args.output / "paper-run.json",
                         (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
            attempt.succeeded(run_id=report.run_id)
        except (ContractFault, ValidationError, ValueError, OSError, KeyError) as exc:
            failure = exc.failure if isinstance(exc, ContractFault) else Failure(
                run_id="paper-run", stage=Stage.INPUT, code=ErrorCode.INPUT_INVALID,
                message="Paper source or linked runtime could not be verified",
                details={"cause": type(exc).__name__, "reason": str(exc)[:200]},
            )
            (args.output / "paper-run.json").unlink(missing_ok=True)
            attempt.failed(run_id=failure.run_id, code=failure.code,
                           failure=(failure.model_dump_json(indent=2) + "\n").encode())
            print(failure.model_dump_json(), file=sys.stderr)
            return 2
        print(json.dumps({"status": "passed", "receipt": str(args.output / "paper-run.json")}))
        return 0
    if args.command == "observe":
        observation = from_report(
            RunReport.model_validate(_read_json(args.report)),
            scenario_id=args.scenario,
            method_id=args.method,
            execution_profile_sha256=args.profile_sha256,
            initial_cash=args.initial_cash,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            observation.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return 0
    if args.command == "compare":
        result = compare(
            Observation.model_validate(_read_json(args.left)),
            Observation.model_validate(_read_json(args.right)),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return 0 if result.status == "comparable" else 2

    output: Path = args.output
    attempt = Attempt(output)
    try:
        events = TypeAdapter(tuple[MarketEvent, ...]).validate_python(_read_json(args.events))
        dataset = DatasetDeclaration.model_validate(_read_json(args.dataset))
        training = (
            TrainingRequest.model_validate(_read_json(args.training)) if args.training else None
        )
        policy = RunPolicy.model_validate(_read_json(args.policy)) if args.policy else RunPolicy()
        report = run_package(
            run_id=f"run-{fingerprint(events)[:12]}",
            package_dir=args.package,
            dataset=dataset,
            engine_capability=reference_capability(dataset.fields),
            policy=policy,
            events=events,
            engine=ReferenceEngine(initial_cash=args.cash),
            output=output,
            training=training,
        )
    except ContractFault as exc:
        attempt.failed(
            run_id=exc.failure.run_id,
            code=exc.failure.code,
            failure=(exc.failure.model_dump_json(indent=2) + "\n").encode("utf-8"),
        )
        print(exc.failure.model_dump_json(), file=sys.stderr)
        return 2
    except (ValidationError, ValueError, OSError, json.JSONDecodeError) as exc:
        failure = Failure(
            run_id="invalid-input",
            stage=Stage.INPUT,
            code=ErrorCode.INPUT_INVALID,
            message="Input could not be parsed",
            details={"cause": type(exc).__name__},
        )
        attempt.failed(
            run_id=failure.run_id,
            code=failure.code,
            failure=(failure.model_dump_json(indent=2) + "\n").encode("utf-8"),
        )
        print(failure.model_dump_json(), file=sys.stderr)
        return 2
    attempt.succeeded(run_id=report.run_id)
    print(json.dumps({"status": report.status, "report": str(output / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
