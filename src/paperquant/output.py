"""Atomic result files and a fail-closed run attempt marker."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".paperquant-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").unlink(missing_ok=True)
    (root / "bundle.json").unlink(missing_ok=True)
    (root / "paper-run.json").unlink(missing_ok=True)
    (root / "failure.json").unlink(missing_ok=True)


class Attempt:
    def __init__(self, root: Path) -> None:
        prepare_output(root)
        self.root = root
        self.attempt_id = uuid.uuid4().hex
        self.started_at = datetime.now(UTC).isoformat()
        self._publish("running")

    def _publish(self, status: str, **fields: Any) -> None:
        value = {
            "attempt_id": self.attempt_id,
            "status": status,
            "started_at": self.started_at,
            **fields,
        }
        atomic_bytes(
            self.root / "attempt.json",
            (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )

    def succeeded(self, *, run_id: str) -> None:
        report = (self.root / "report.json").read_bytes()
        self._publish(
            "succeeded",
            run_id=run_id,
            report_sha256=hashlib.sha256(report).hexdigest(),
            completed_at=datetime.now(UTC).isoformat(),
        )

    def failed(self, *, run_id: str, code: str, failure: bytes) -> None:
        (self.root / "report.json").unlink(missing_ok=True)
        (self.root / "bundle.json").unlink(missing_ok=True)
        (self.root / "paper-run.json").unlink(missing_ok=True)
        atomic_bytes(self.root / "failure.json", failure)
        self._publish(
            "failed",
            run_id=run_id,
            code=code,
            failure_sha256=hashlib.sha256(failure).hexdigest(),
            completed_at=datetime.now(UTC).isoformat(),
        )
