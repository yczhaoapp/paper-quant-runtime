"""Container-backed strategy protocol client; market and report stay on the host."""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, NoReturn, Self

from pydantic import TypeAdapter

from paperquant.compiler import fingerprint
from paperquant.models import (
    AccountSnapshot,
    Action,
    ContractFault,
    ErrorCode,
    Failure,
    MarketEvent,
    Stage,
    StrategyDeclaration,
    TrainingRequest,
    WorkerReceipt,
)

_ACTIONS = TypeAdapter(tuple[Action, ...])
_MAX_MESSAGE = 8 * 1024 * 1024
_CONTROLS = (
    "network:none",
    "root:read-only",
    "capabilities:none",
    "privilege-escalation:none",
    "user:65532:65532",
    "package:read-only",
    "memory:256m",
    "pids:64",
    "cpus:0.5",
    "tmpfs:/tmp:16m",
)


def _unavailable(run_id: str, message: str, **details: str) -> NoReturn:
    raise ContractFault(
        Failure(
            run_id=run_id,
            stage=Stage.LOAD,
            code=ErrorCode.SANDBOX_UNAVAILABLE,
            message=message,
            details=details,
        )
    )


class DockerWorkerStrategy:
    def __init__(
        self,
        *,
        run_id: str,
        package_dir: Path,
        declaration: StrategyDeclaration,
        source_sha256: str,
        image: str = "paperquant-worker:local",
    ) -> None:
        self.run_id = run_id
        self.declaration = declaration
        self._process: subprocess.Popen[bytes] | None = None
        self._replies: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
        try:
            inspected = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            _unavailable(run_id, "Strict worker image is not available")
        image_id = inspected.stdout.decode("ascii", errors="replace").strip()
        if not image_id.startswith("sha256:") or len(image_id) != 71:
            _unavailable(run_id, "Worker image inspection returned no immutable ID")
        package_path = package_dir.resolve()
        if "," in str(package_path) or "\n" in str(package_path):
            _unavailable(run_id, "Package path cannot be encoded as a Docker bind mount")
        self._cid_directory = tempfile.TemporaryDirectory(prefix="paperquant-container-")
        cidfile = Path(self._cid_directory.name) / "cid"
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--cidfile",
            str(cidfile),
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "0.5",
            "--user",
            "65532:65532",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--mount",
            f"type=bind,source={package_path},target=/package,readonly",
            image_id,
            "/package",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            assert self._process.stdout is not None
            self._reader_thread = threading.Thread(
                target=self._read_worker_lines,
                args=(self._process.stdout,),
                daemon=True,
            )
            self._reader_thread.start()
            ready = self._receive(timeout=30)
            if (
                ready.get("ok") is not True
                or ready.get("protocol_version") != "1.0"
                or ready.get("source_sha256") != source_sha256
                or fingerprint(StrategyDeclaration.model_validate(ready.get("declaration")))
                != fingerprint(declaration)
            ):
                _unavailable(run_id, "Worker did not attest the expected strategy package")
            container_id = cidfile.read_text(encoding="ascii").strip()
            self._verify_container(container_id, package_path)
        except ContractFault:
            self.close()
            raise
        except Exception as exc:
            self.close()
            _unavailable(
                run_id, "Strict worker could not start or attest", cause=type(exc).__name__
            )
        self.receipt = WorkerReceipt(
            image_id=image_id,
            container_id=container_id,
            source_sha256=source_sha256,
            controls=_CONTROLS,
        )

    def _verify_container(self, container_id: str, package_path: Path) -> None:
        if len(container_id) != 64 or any(char not in "0123456789abcdef" for char in container_id):
            _unavailable(self.run_id, "Worker container ID was not recorded")
        try:
            inspected = subprocess.run(
                ["docker", "inspect", "--format", "{{json .}}", container_id],
                capture_output=True,
                check=True,
                timeout=10,
            )
            container = json.loads(inspected.stdout)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            _unavailable(self.run_id, "Worker container could not be inspected")
        try:
            host = container["HostConfig"]
            mounts = container["Mounts"]
            package_mounts = [mount for mount in mounts if mount["Destination"] == "/package"]
            controls_applied = (
                host["NetworkMode"] == "none"
                and host["ReadonlyRootfs"] is True
                and "ALL" in (host["CapDrop"] or [])
                and "no-new-privileges" in (host["SecurityOpt"] or [])
                and host["PidsLimit"] == 64
                and host["Memory"] == 256 * 1024 * 1024
                and host["NanoCpus"] == 500_000_000
                and container["Config"]["User"] == "65532:65532"
                and len(package_mounts) == 1
                and package_mounts[0]["Type"] == "bind"
                and package_mounts[0]["RW"] is False
                and Path(package_mounts[0]["Source"]).resolve() == package_path
                and all(mount["RW"] is False for mount in mounts if mount["Type"] == "bind")
                and "/tmp" in host["Tmpfs"]
                and "noexec" in host["Tmpfs"]["/tmp"]
                and "nosuid" in host["Tmpfs"]["/tmp"]
                and "size=16m" in host["Tmpfs"]["/tmp"]
            )
        except (KeyError, TypeError, AttributeError):
            controls_applied = False
        if not controls_applied:
            _unavailable(self.run_id, "Docker did not apply the required worker restrictions")

    def _read_worker_lines(self, stream: Any) -> None:
        while True:
            line = stream.readline(_MAX_MESSAGE + 1)
            self._replies.put(line if line else None)
            if not line:
                return

    def _receive(self, *, timeout: float = 20) -> dict[str, Any]:
        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            _unavailable(self.run_id, "Strict worker response timed out")
        if line is None:
            _unavailable(self.run_id, "Strict worker closed the protocol stream")
        if len(line) > _MAX_MESSAGE or not line.endswith(b"\n"):
            _unavailable(self.run_id, "Strict worker exceeded message size limit")
        try:
            result = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            _unavailable(self.run_id, "Strict worker returned invalid protocol JSON")
        if not isinstance(result, dict):
            _unavailable(self.run_id, "Strict worker returned a non-object reply")
        return result

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self._process is not None and self._process.stdin is not None
        payload = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > _MAX_MESSAGE:
            raise ValueError("worker request exceeds size limit")
        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            _unavailable(self.run_id, "Strict worker protocol input closed")
        result = self._receive()
        if result.get("ok") is not True:
            raise RuntimeError(f"worker rejected operation: {result.get('cause', 'unknown')}")
        return result

    def train(self, request: TrainingRequest) -> bytes:
        encoded = self._exchange({"op": "train", "request": request.model_dump(mode="json")})
        return base64.b64decode(encoded["payload"], validate=True)

    def load(self, payload: bytes) -> None:
        self._exchange({"op": "load", "payload": base64.b64encode(payload).decode("ascii")})

    def decide(self, event: MarketEvent, account: AccountSnapshot) -> tuple[Action, ...]:
        response = self._exchange(
            {
                "op": "decide",
                "event": event.model_dump(mode="json"),
                "account": account.model_dump(mode="json"),
            }
        )
        return _ACTIONS.validate_python(response["actions"])

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            self._cid_directory.cleanup()
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(b'{"op":"stop"}\n')
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        self._cid_directory.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
