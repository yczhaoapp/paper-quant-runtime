from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from paperquant.comparison import Comparison, Observation, compare, from_report
from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import (
    AccountSnapshot,
    Artifact,
    ContractFault,
    Conversion,
    DatasetDeclaration,
    Decision,
    EngineCapability,
    ErrorCode,
    ExecutionPlan,
    Failure,
    Fill,
    MarketEvent,
    OrderEvent,
    Position,
    RunPolicy,
    RunReport,
    Stage,
    StrategyDeclaration,
    TrainingRequest,
    WorkerReceipt,
)
from paperquant.output import Attempt, atomic_bytes
from paperquant.package import run_package
from paperquant.papers import Anchor, ExpectedStrategy, Recipe, Source, verify_recipe

SCHEMAS = {
    model.__name__: model
    for model in (
        StrategyDeclaration,
        DatasetDeclaration,
        EngineCapability,
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
    observe.add_argument("--profile-sha256", required=True)
    observe.add_argument("--initial-cash", type=Decimal, required=True)
    observe.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--left", type=Path, required=True)
    comparison.add_argument("--right", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "schema":
        export_schemas(args.output)
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
