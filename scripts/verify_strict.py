"""Build and verify the isolated worker with attempt-scoped, fail-closed receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "src",
    "tests",
    "strategies",
    "research",
    "examples",
    "scripts",
    "docs",
    "schemas",
    "data",
    ".github",
)
SOURCE_FILES = (
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    "Dockerfile.worker",
    "requirements-worker.txt",
    "pyproject.toml",
    "uv.lock",
    "README.md",
)
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".receipt-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_sha256() -> str:
    paths = [ROOT / filename for filename in SOURCE_FILES]
    for directory in SOURCE_DIRS:
        paths.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and not any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts)
            and path.suffix != ".pyc"
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = head.stdout.strip() if head.returncode == 0 else None
    clean = status.returncode == 0 and not status.stdout.strip()
    return commit, clean


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
    attempt_id = uuid.uuid4().hex
    attempt_dir = output / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    receipt = output / "verification.json"
    image_receipt = output / "image.json"
    image_receipt.unlink(missing_ok=True)
    state: dict[str, object] = {
        "attempt_id": attempt_id,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(receipt, state)
    phase = "source"
    try:
        before = _source_sha256()
        commit, clean = _git_state()
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
        environment = dict(os.environ)
        environment["PAPERQUANT_TEST_DOCKER"] = "1"
        environment["PYTHONPATH"] = "src"
        _run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            attempt_dir / "tests.log",
            timeout=900,
            environment=environment,
        )
        phase = "acceptance"
        _run(
            [sys.executable, "scripts/acceptance.py", "--output", str(attempt_dir / "acceptance"),
             "--strict-image-id", image_id],
            attempt_dir / "acceptance.log",
            timeout=900,
            environment=environment,
        )
        acceptance_file = attempt_dir / "acceptance/acceptance.json"
        acceptance = json.loads(acceptance_file.read_text())
        if (
            acceptance.get("status") != "passed"
            or acceptance.get("count") != 18
            or not acceptance.get("strict")
        ):
            raise RuntimeError("eighteen-case strict acceptance did not pass")
        phase = "source-recheck"
        after = _source_sha256()
        after_commit, after_clean = _git_state()
        if before != after or commit != after_commit or (args.require_clean and not after_clean):
            raise RuntimeError("source tree changed during verification")
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
        _write_json(receipt, state)
        print(json.dumps({"status": "passed", "receipt": str(receipt), "attempt_id": attempt_id}))
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        image_receipt.unlink(missing_ok=True)
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
