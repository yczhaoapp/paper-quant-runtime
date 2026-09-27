"""Require the explicitly reviewed safety nodes to pass in the full strict test run."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research/strict-test-nodes.json"


def test_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # Selection/configuration through PYTEST_ADDOPTS or plugin overrides is not
    # accepted in a release gate. Explicit command arguments remain recorded.
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        environment.pop(key, None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def verify_safety_tests(junit: Path, *, lock: Path = LOCK) -> dict[str, Any]:
    raw = lock.read_bytes()
    declaration = json.loads(raw)
    if (not isinstance(declaration, dict) or declaration.get("schema_version") != "1.0"
        or not isinstance(declaration.get("required_nodes"), list)):
        raise ValueError("Safety test lock has an unsupported shape")
    required = declaration["required_nodes"]
    if (not required or any(not isinstance(node, str) or "::" not in node for node in required)
        or len(required) != len(set(required))):
        raise ValueError("Safety test lock has empty, duplicate, or invalid nodes")
    tree = ET.parse(junit)
    results: dict[str, list[ET.Element]] = {}
    cases = tree.findall(".//testcase")
    for case in cases:
        node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        results.setdefault(node, []).append(case)
    missing = [node for node in required if node not in results]
    bad = [node for node in required if node in results and (
        len(results[node]) != 1 or any(results[node][0].find(tag) is not None
                                      for tag in ("failure", "error", "skipped")))]
    if missing or bad:
        raise ValueError("Required safety tests did not pass: " + ", ".join(missing + bad))
    if any(case.find(tag) is not None for case in cases for tag in ("failure", "error")):
        raise ValueError("The full strict test run contains failures or errors")
    return {
        "status": "passed", "required_nodes": required,
        "required_passed": len(required), "reported_testcases": len(cases),
        "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
        "test_lock_sha256": hashlib.sha256(raw).hexdigest(),
    }
