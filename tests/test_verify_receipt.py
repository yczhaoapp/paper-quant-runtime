from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_failed_new_attempt_invalidates_earlier_success_receipts(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "verification.json").write_text(
        json.dumps({"attempt_id": "old", "status": "passed"}), encoding="utf-8"
    )
    (output / "image.json").write_text(
        json.dumps({"attempt_id": "old", "image_id": "sha256:" + "a" * 64}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_strict.py",
            "--output",
            str(output),
            "--docker",
            sys.executable,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    receipt = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["phase"] == "build"
    assert receipt["attempt_id"] != "old"
    assert "runtime_assurance" not in receipt
    assert not (output / "image.json").exists()
