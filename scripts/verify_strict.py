"""Build and verify the isolated worker with attempt-scoped, fail-closed receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Direct script execution and python -m scripts.verify_strict share the same helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts.gate_support import (  # noqa: E402
    check_source_identity,
    git_state,
    new_attempt,
    source_sha256,
    write_json,
)
from scripts.strict_test_gate import test_environment, verify_safety_tests  # noqa: E402

ASSURANCE_MATRIX = ROOT / "research/assurance-axes.json"


def _write_json(path: Path, value: dict[str, object]) -> None:
    write_json(path, value)


def _source_sha256() -> str:
    return source_sha256(ROOT)


def _git_state() -> tuple[str | None, bool]:
    return git_state(ROOT)


def _validate_acceptance(value: object, identity: dict[str, object], image_id: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Child acceptance receipt must be a JSON object")
    if (value.get("status") != "passed" or type(value.get("count")) is not int
        or value["count"] != 18 or value.get("strict") is not True
        or value.get("strict_image_id") != image_id
        or not isinstance(value.get("attempt_id"), str) or not value["attempt_id"]
        or not isinstance(value.get("completed_at"), str) or not value["completed_at"]
        or any(value.get(key) != expected for key, expected in identity.items())
        or type(value.get("git_clean")) is not bool):
        raise ValueError("Child acceptance is not bound to this source and strict image")
    cases = value.get("cases")
    catalog_ids = {item["strategy"] for item in json.loads(
        (ROOT / "examples/catalog.json").read_text())["cases"]}
    if (not isinstance(cases, list) or len(cases) != 18
        or any(not isinstance(case, dict) for case in cases)
        or {case.get("strategy_id") for case in cases} != catalog_ids):
        raise ValueError("Child acceptance does not cover the eighteen-case catalog")
    return value


def _run(
    command: list[str], log: Path, *, timeout: int, environment: dict[str, str] | None = None
) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    log.write_text(result.stdout + result.stderr, encoding="utf-8", newline="\n")
    if result.returncode != 0:
        raise RuntimeError(f"command exited {result.returncode}: {command[0]} {command[1]}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    output: Path = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = new_attempt()
    attempt_id = str(state["attempt_id"])
    attempt_dir = output / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    receipt = output / "verification.json"
    image_receipt = output / "image.json"
    image_receipt.unlink(missing_ok=True)
    _write_json(receipt, state)
    phase = "source"
    try:
        before = _source_sha256()
        commit, clean = _git_state()
        identity = {"source_tree_sha256": before, "git_commit": commit, "git_clean": clean}
        state.update(identity)
        state["source_sha256"] = before
        state["git_commit"] = commit
        state["git_clean"] = clean
        _write_json(receipt, state)
        if args.require_clean and (commit is None or not clean):
            raise RuntimeError("release verification requires a clean committed source tree")
        phase = "build"
        _run(
            [
                args.docker,
                "build",
                "--no-cache",
                "-f",
                "Dockerfile.worker",
                "-t",
                "paperquant-worker:local",
                ".",
            ],
            attempt_dir / "build.log",
            timeout=900,
        )
        phase = "inspect"
        image_id = _run(
            [args.docker, "image", "inspect", "--format", "{{.Id}}", "paperquant-worker:local"],
            attempt_dir / "inspect.log",
            timeout=30,
        )
        if not image_id.startswith("sha256:") or len(image_id) != 71:
            raise RuntimeError("image inspection returned no immutable image ID")
        _write_json(image_receipt, {"attempt_id": attempt_id, "image_id": image_id})
        phase = "tests"
        environment = test_environment()
        environment["PAPERQUANT_TEST_DOCKER"] = "1"
        environment["PYTHONPATH"] = "src"
        _run(
            [sys.executable, "-m", "pytest", "-c", "pyproject.toml", "-o", "addopts=",
             "-q", "tests", "--junitxml", str(attempt_dir / "tests.xml")],
            attempt_dir / "tests.log",
            timeout=900,
            environment=environment,
        )
        phase = "safety-tests"
        state["safety_test_gate"] = verify_safety_tests(attempt_dir / "tests.xml")
        _write_json(receipt, state)
        phase = "acceptance"
        _run(
            [sys.executable, "scripts/acceptance.py", "--output", str(attempt_dir / "acceptance"),
             "--strict-image-id", image_id] + (["--require-clean"] if args.require_clean else []),
            attempt_dir / "acceptance.log",
            timeout=900,
            environment=environment,
        )
        acceptance_file = attempt_dir / "acceptance/acceptance.json"
        acceptance = _validate_acceptance(
            json.loads(acceptance_file.read_text()), identity, image_id,
        )
        phase = "source-recheck"
        check_source_identity(ROOT, identity, require_clean=args.require_clean)
        state.update(
            {
                "status": "passed",
                "image_id": image_id,
                "completed_at": datetime.now(UTC).isoformat(),
                "build_log_sha256": hashlib.sha256(
                    (attempt_dir / "build.log").read_bytes()
                ).hexdigest(),
                "test_log_sha256": hashlib.sha256(
                    (attempt_dir / "tests.log").read_bytes()
                ).hexdigest(),
                "acceptance_sha256": hashlib.sha256(acceptance_file.read_bytes()).hexdigest(),
                "acceptance_log_sha256": hashlib.sha256(
                    (attempt_dir / "acceptance.log").read_bytes()
                ).hexdigest(),
            }
        )
        if commit is not None and clean:
            state["runtime_assurance"] = {
                "level": "R2",
                "scope": "eighteen catalog strategies through strict isolated worker",
                "catalog_cases": acceptance["count"],
                "attempt_id": attempt_id,
                "assurance_matrix_sha256": hashlib.sha256(
                    ASSURANCE_MATRIX.read_bytes()
                ).hexdigest(),
            }
        _write_json(receipt, state)
        print(json.dumps({"status": "passed", "receipt": str(receipt), "attempt_id": attempt_id}))
        return 0
    except Exception as exc:
        image_receipt.unlink(missing_ok=True)
        state.pop("runtime_assurance", None)
        state.pop("image_id", None)
        state.update(
            {
                "status": "failed",
                "phase": phase,
                "cause": type(exc).__name__,
                "message": str(exc),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(receipt, state)
        print(json.dumps({"status": "failed", "phase": phase, "receipt": str(receipt)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
