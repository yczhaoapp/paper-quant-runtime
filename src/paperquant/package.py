from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperquant.compiler import canonical_document, compile_run, fingerprint
from paperquant.engine import BacktestEngine
from paperquant.models import (
    ContractFault,
    DatasetDeclaration,
    EngineCapability,
    ErrorCode,
    Failure,
    MarketEvent,
    RunPolicy,
    RunReport,
    Stage,
    StrategyDeclaration,
    TrainingRequest,
)
from paperquant.output import prepare_output
from paperquant.runtime import _execute, run
from paperquant.sandbox import DockerWorkerStrategy


class PackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    declaration: StrategyDeclaration
    class_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z_0-9]*$")
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _fail(run_id: str, code: ErrorCode, message: str) -> NoReturn:
    raise ContractFault(Failure(run_id=run_id, stage=Stage.COMPILE, code=code, message=message))


def _check_source(run_id: str, source: bytes) -> None:
    try:
        parsed = ast.parse(source.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError):
        _fail(run_id, ErrorCode.INPUT_INVALID, "Strategy source is not valid UTF-8 Python")
    allowed = {"paperquant.strategy_api", "math", "statistics", "decimal", "json", "random"}
    forbidden_calls = {"open", "exec", "eval", "compile", "__import__"}
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            if any(alias.name not in allowed for alias in node.names):
                _fail(run_id, ErrorCode.SANDBOX_UNAVAILABLE, "Strategy imports outside the API")
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module not in allowed:
                _fail(run_id, ErrorCode.SANDBOX_UNAVAILABLE, "Strategy imports outside the API")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                _fail(run_id, ErrorCode.SANDBOX_UNAVAILABLE, "Strategy contains a forbidden call")


def inspect_package(run_id: str, root: Path) -> tuple[PackageManifest, bytes]:
    root = root.resolve()
    manifest_path = root / "strategy.json"
    source_path = root / "strategy.py"
    if not manifest_path.is_file() or not source_path.is_file():
        _fail(run_id, ErrorCode.INPUT_INVALID, "Strategy directory lacks required files")
    if manifest_path.is_symlink() or source_path.is_symlink():
        _fail(run_id, ErrorCode.INPUT_INVALID, "Strategy package cannot contain symlink entries")
    try:
        manifest = PackageManifest.model_validate_json(manifest_path.read_bytes())
    except (ValidationError, ValueError):
        _fail(run_id, ErrorCode.INPUT_INVALID, "Strategy manifest does not satisfy the schema")
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != manifest.source_sha256:
        _fail(run_id, ErrorCode.INPUT_INVALID, "Strategy source digest differs from manifest")
    _check_source(run_id, source)
    return manifest, source


def run_package(
    *,
    run_id: str,
    package_dir: Path,
    dataset: DatasetDeclaration,
    engine_capability: EngineCapability,
    policy: RunPolicy,
    events: tuple[MarketEvent, ...],
    engine: BacktestEngine,
    output: Path,
    training: TrainingRequest | None = None,
) -> RunReport:
    prepare_output(output)
    manifest, _ = inspect_package(run_id, package_dir)
    try:
        actual_capability = engine.capability(engine_capability.data_fields)
    except Exception:
        _fail(run_id, ErrorCode.ENGINE_UNSUPPORTED, "Engine cannot declare capabilities")
    if fingerprint(actual_capability) != fingerprint(engine_capability):
        _fail(run_id, ErrorCode.ENGINE_UNSUPPORTED,
              "Advertised capability differs from the selected engine")
    compile_run(
        run_id=run_id,
        strategy=manifest.declaration,
        dataset=dataset,
        engine=engine_capability,
        policy=policy,
        events=events,
        training=training,
    )
    if policy.sandbox == "strict":
        with DockerWorkerStrategy(
            run_id=run_id,
            package_dir=package_dir,
            declaration=manifest.declaration,
            source_sha256=manifest.source_sha256,
        ) as strategy:
            return _execute(
                run_id=run_id,
                strategy=strategy,
                dataset=dataset,
                engine_capability=engine_capability,
                policy=policy,
                events=events,
                engine=engine,
                output=output,
                training=training,
                worker_receipt=strategy.receipt,
            )
    source_path = package_dir.resolve() / "strategy.py"
    module_spec = importlib.util.spec_from_file_location(
        f"paperquant_package_{manifest.source_sha256}", source_path
    )
    if module_spec is None or module_spec.loader is None:
        _fail(run_id, ErrorCode.INPUT_INVALID, "Strategy module cannot be loaded")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
        factory: Any = getattr(module, manifest.class_name)
        strategy = factory()
    except Exception as exc:
        raise ContractFault(
            Failure(
                run_id=run_id,
                stage=Stage.LOAD,
                code=ErrorCode.MODEL_CORRUPT,
                message="Strategy construction failed",
                details={"cause": type(exc).__name__},
            )
        ) from exc
    if not hasattr(strategy, "declaration") or fingerprint(strategy.declaration) != fingerprint(
        manifest.declaration
    ):
        _fail(run_id, ErrorCode.INPUT_INVALID, "Loaded strategy declaration differs from manifest")
    return run(
        run_id=run_id,
        strategy=strategy,
        dataset=dataset,
        engine_capability=engine_capability,
        policy=policy,
        events=events,
        engine=engine,
        output=output,
        training=training,
    )


def write_manifest(root: Path, declaration: StrategyDeclaration, class_name: str) -> None:
    source = (root / "strategy.py").read_bytes()
    manifest = PackageManifest(
        declaration=declaration,
        class_name=class_name,
        source_sha256=hashlib.sha256(source).hexdigest(),
    )
    (root / "strategy.json").write_text(
        json.dumps(canonical_document(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
