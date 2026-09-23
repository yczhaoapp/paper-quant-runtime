"""Narrow JSON-line protocol for an isolated strategy process."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from paperquant.compiler import fingerprint
from paperquant.models import AccountSnapshot, Action, MarketEvent, TrainingRequest
from paperquant.package import inspect_package

_ACTIONS = TypeAdapter(tuple[Action, ...])


def _reply(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_strategy(package: Path) -> tuple[Any, str]:
    manifest, _ = inspect_package("worker-init", package)
    spec = importlib.util.spec_from_file_location("isolated_strategy", package / "strategy.py")
    if spec is None or spec.loader is None:
        raise ValueError("strategy module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        strategy = getattr(module, manifest.class_name)()
    if fingerprint(strategy.declaration) != fingerprint(manifest.declaration):
        raise ValueError("worker declaration differs from manifest")
    return strategy, manifest.source_sha256


def _handle(strategy: Any, request: dict[str, Any]) -> dict[str, Any]:
    operation = request.get("op")
    if operation == "train":
        training = TrainingRequest.model_validate(request["request"])
        with redirect_stdout(sys.stderr):
            payload = strategy.train(training)
        if type(payload) is not bytes or not payload:
            raise ValueError("training returned no model bytes")
        return {"ok": True, "payload": base64.b64encode(payload).decode("ascii")}
    if operation == "load":
        payload = base64.b64decode(request["payload"], validate=True)
        with redirect_stdout(sys.stderr):
            strategy.load(payload)
        return {"ok": True}
    if operation == "decide":
        event = MarketEvent.model_validate(request["event"])
        account = AccountSnapshot.model_validate(request["account"])
        with redirect_stdout(sys.stderr):
            actions = strategy.decide(event, account)
        if type(actions) is not tuple:
            return {"ok": False, "code": "ACTION_INVALID"}
        try:
            actions = _ACTIONS.validate_python(
                tuple(action.model_dump(mode="python") for action in actions)
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            return {"ok": False, "code": "ACTION_INVALID"}
        return {"ok": True, "actions": _ACTIONS.dump_python(actions, mode="json")}
    raise ValueError("unknown worker operation")


def main() -> int:
    package = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("/package")
    try:
        strategy, source_sha256 = _load_strategy(package)
    except Exception as exc:
        _reply({"ok": False, "stage": "load", "cause": type(exc).__name__})
        return 2
    _reply(
        {
            "ok": True,
            "protocol_version": "1.0",
            "source_sha256": source_sha256,
            "declaration": strategy.declaration.model_dump(mode="json"),
        }
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "stop":
                _reply({"ok": True})
                return 0
            _reply(_handle(strategy, request))
        except Exception as exc:
            _reply({"ok": False, "cause": type(exc).__name__})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
