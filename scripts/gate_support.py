"""Shared atomic receipts and source identity for independent release gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paperquant.output import atomic_bytes

SOURCE_DIRS = (
    "src", "tests", "strategies", "research", "examples", "scripts", "docs",
    "schemas", "data", ".github",
)
SOURCE_FILES = (
    ".dockerignore", ".gitattributes", ".gitignore", "Dockerfile.worker",
    "requirements-worker.txt", "pyproject.toml", "uv.lock", "README.md",
)
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "source-cache"}


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def new_attempt() -> dict[str, Any]:
    return {"attempt_id": uuid.uuid4().hex, "status": "running",
            "started_at": datetime.now(UTC).isoformat()}


def source_sha256(root: Path) -> str:
    paths = [root / filename for filename in SOURCE_FILES]
    for directory in SOURCE_DIRS:
        paths.extend(path for path in (root / directory).rglob("*") if path.is_file()
                     and not any(part in EXCLUDE_DIRS for part in path.relative_to(root).parts)
                     and path.suffix != ".pyc")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_state(root: Path) -> tuple[str | None, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=False, timeout=30)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                            capture_output=True, text=True, check=False, timeout=30)
    return (head.stdout.strip() if head.returncode == 0 else None,
            status.returncode == 0 and not status.stdout.strip())


def source_identity(root: Path, *, require_clean: bool = False) -> dict[str, Any]:
    digest = source_sha256(root)
    commit, clean = git_state(root)
    if require_clean and (commit is None or not clean):
        raise ValueError("Release verification requires a clean committed source tree")
    return {"source_tree_sha256": digest, "git_commit": commit, "git_clean": clean}


def check_source_identity(root: Path, before: dict[str, Any], *, require_clean: bool) -> None:
    if source_identity(root, require_clean=require_clean) != before:
        raise ValueError("Source tree or Git identity changed during verification")
