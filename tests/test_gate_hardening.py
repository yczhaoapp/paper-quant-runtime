"""Fault injection proves release receipts fail closed independently of runtime tests."""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts import acceptance, ci_gate, gate_support, verify_paper_sources, verify_strict
from scripts.strict_test_gate import LOCK, verify_safety_tests
from scripts.strict_test_gate import test_environment as release_environment


def _junit(path: Path, nodes: list[str], *, outcome: str | None = None) -> None:
    suite = ET.Element("testsuite")
    for index, node in enumerate(nodes):
        file, name = node.split("::", 1)
        case = ET.SubElement(suite, "testcase", classname=file[:-3].replace("/", "."),
                             name=name)
        if index == 0 and outcome:
            ET.SubElement(case, outcome)
    ET.ElementTree(suite).write(path, encoding="utf-8")


def _required() -> list[str]:
    return json.loads(LOCK.read_text())["required_nodes"]


def test_safety_node_lock_matches_collected_strict_boundaries() -> None:
    import subprocess

    environment = release_environment()
    environment["PAPERQUANT_TEST_DOCKER"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "tests/test_strict_worker.py", "tests/test_unseen_packages.py",
         "tests/test_execution_integrity.py"],
        cwd=verify_strict.ROOT, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {line for line in result.stdout.splitlines()
                if line.startswith("tests/test_strict_worker.py::")
                or "::test_unregistered_strategy_matches_strict_worker[" in line
                or line.endswith("::test_strict_worker_cannot_make_order_collision_succeed")}
    assert len(expected) == 27
    assert set(_required()) == expected


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_required_safety_nodes_cannot_be_skipped_or_failed(tmp_path: Path, outcome: str) -> None:
    junit = tmp_path / "tests.xml"
    _junit(junit, _required(), outcome=outcome)
    with pytest.raises(ValueError, match="Required safety tests"):
        verify_safety_tests(junit)


@pytest.mark.parametrize("change", ["missing", "duplicate", "oracle_only", "absent"])
def test_zero_exit_is_insufficient_without_safety_junit(tmp_path: Path, change: str) -> None:
    nodes = _required()
    if change == "missing":
        nodes = nodes[1:]
    elif change == "duplicate":
        nodes = nodes + [nodes[0]]
    elif change == "oracle_only":
        nodes = ["tests/test_q_oracles.py::test_formula"]
    junit = tmp_path / "tests.xml"
    if change != "absent":
        _junit(junit, nodes)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_safety_tests(junit)


def test_optional_paper_cache_skip_does_not_weaken_required_safety_nodes(tmp_path: Path) -> None:
    junit = tmp_path / "tests.xml"
    _junit(junit, _required())
    tree = ET.parse(junit)
    extra = ET.SubElement(tree.getroot(), "testcase", classname="tests.test_paper_sources",
                          name="test_all_primary_pdfs_match_pins_when_cache_is_prepared")
    ET.SubElement(extra, "skipped")
    tree.write(junit)
    assert verify_safety_tests(junit)["required_passed"] == 27


@pytest.mark.parametrize("declaration", [[], {}, {"schema_version": "1.0", "required_nodes": []},
    {"schema_version": "1.0", "required_nodes": ["a::b", "a::b"]},
    {"schema_version": "1.0", "required_nodes": [4]}])
def test_invalid_safety_lock_is_rejected(tmp_path: Path, declaration: object) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(declaration))
    with pytest.raises(ValueError):
        verify_safety_tests(tmp_path / "absent.xml", lock=lock)


def test_release_pytest_environment_removes_external_selection(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore=tests/test_strict_worker.py -k nonexistent")
    monkeypatch.setenv("PYTEST_PLUGINS", "arbitrary_external_plugin")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")
    env = release_environment()
    assert "PYTEST_ADDOPTS" not in env
    assert "PYTEST_PLUGINS" not in env
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


def test_hash_correct_damaged_pdf_closes_attempt_and_replaces_old_pass(tmp_path: Path,
                                                                    monkeypatch) -> None:
    raw = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n"
    (tmp_path / "paper.pdf").write_bytes(raw)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema_version": "1.0", "sources": [{
        "source_id": "damaged", "canonical_url": "https://example.org/paper",
        "retrieval_url": "https://example.org/paper.pdf", "format": "pdf",
        "file": "paper.pdf", "sha256": hashlib.sha256(raw).hexdigest(),
        "title": "A Paper", "authors": ["A Author"],
    }]}))
    monkeypatch.setattr(verify_paper_sources, "ROOT", tmp_path)
    monkeypatch.setattr(verify_paper_sources, "LOCK", lock)
    monkeypatch.setattr(verify_paper_sources, "_expected_claim_urls",
                        lambda: {"https://example.org/paper"})
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "passed", "attempt_id": "old", "sources": [1]}))
    writes = []
    original = verify_paper_sources.write_json

    def observe(path, value):
        writes.append(dict(value))
        original(path, value)

    monkeypatch.setattr(verify_paper_sources, "write_json", observe)
    monkeypatch.setattr(sys, "argv", ["verify_paper_sources", "--output", str(receipt)])
    assert verify_paper_sources.main() == 2
    result = json.loads(receipt.read_text())
    assert writes[0]["status"] == "running"
    assert result["attempt_id"] == writes[0]["attempt_id"] != "old"
    assert result["status"] == "failed" and result["completed_at"]
    assert result["cause"] == "PaperParseError"
    assert "sources" not in result


@pytest.mark.parametrize("exception", [RuntimeError("parser problem"), KeyError("source"),
                                       TypeError("shape")])
def test_source_gate_closes_other_ordinary_exceptions(tmp_path: Path, monkeypatch,
                                                     exception: Exception) -> None:
    def fail(**kwargs):
        raise exception

    monkeypatch.setattr(verify_paper_sources, "verify_sources", fail)
    path = tmp_path / "sources.json"
    monkeypatch.setattr(sys, "argv", ["verify_paper_sources", "--output", str(path)])
    assert verify_paper_sources.main() == 2
    result = json.loads(path.read_text())
    assert result["status"] == "failed" and result["completed_at"] and result["attempt_id"]


@pytest.mark.parametrize("problem", ["truncated", "array", "null", "bool", "empty",
    "wrong_source", "wrong_commit", "wrong_image", "wrong_count", "wrong_strict", "no_attempt",
    "missing_case", "duplicate_case", "tests_missing", "tests_skipped", "inspect_exception"])
def test_strict_gate_closes_bad_child_results(tmp_path: Path, monkeypatch, problem: str) -> None:
    image = "sha256:" + "a" * 64
    identity = {"source_tree_sha256": "b" * 64, "git_commit": "c" * 40, "git_clean": True}
    catalog = json.loads((verify_strict.ROOT / "examples/catalog.json").read_text())["cases"]
    child = {**identity, "status": "passed", "count": 18, "strict": True,
             "strict_image_id": image, "attempt_id": "child-new", "completed_at": "now",
             "cases": [{"strategy_id": case["strategy"]} for case in catalog]}
    if problem.startswith("wrong_"):
        key, value = {
            "wrong_source": ("source_tree_sha256", "d" * 64),
            "wrong_commit": ("git_commit", "d" * 40), "wrong_image": ("strict_image_id", "bad"),
            "wrong_count": ("count", True), "wrong_strict": ("strict", 1),
        }[problem]
        child[key] = value
    elif problem == "no_attempt":
        child.pop("attempt_id")
    elif problem == "missing_case":
        child["cases"].pop()
    elif problem == "duplicate_case":
        child["cases"][-1] = child["cases"][0]
    encoded = {"truncated": '{"status":', "array": "[]", "null": "null", "bool": "true",
               "empty": "{}"}.get(problem, json.dumps(child))
    monkeypatch.setattr(verify_strict, "_source_sha256", lambda: identity["source_tree_sha256"])
    monkeypatch.setattr(verify_strict, "_git_state", lambda: (identity["git_commit"], True))
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore=tests/test_strict_worker.py")

    def run(command, log, *, timeout, environment=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("fault-injection-only\n")
        if "inspect" in command:
            if problem == "inspect_exception":
                raise ValueError("unexpected inspection data")
            return image
        if "pytest" in command:
            assert "PYTEST_ADDOPTS" not in environment
            assert "--junitxml" in command and "addopts=" in command
            junit = Path(command[command.index("--junitxml") + 1])
            if problem != "tests_missing":
                _junit(junit, _required(), outcome="skipped" if problem == "tests_skipped"
                       else None)
        if "scripts/acceptance.py" in command:
            path = Path(command[command.index("--output") + 1])
            path.mkdir(parents=True)
            (path / "acceptance.json").write_text(encoded)
        return ""

    monkeypatch.setattr(verify_strict, "_run", run)
    (tmp_path / "verification.json").write_text('{"status":"passed","attempt_id":"old"}')
    (tmp_path / "image.json").write_text('{"attempt_id":"old"}')
    monkeypatch.setattr(sys, "argv", ["verify_strict", "--output", str(tmp_path)])
    assert verify_strict.main() == 1
    result = json.loads((tmp_path / "verification.json").read_text())
    assert result["status"] == "failed" and result["completed_at"]
    assert result["attempt_id"] != "old"
    assert "runtime_assurance" not in result and "image_id" not in result
    assert not (tmp_path / "image.json").exists()
    expected_phase = ("safety-tests" if problem.startswith("tests_") else
                      "inspect" if problem == "inspect_exception" else "acceptance")
    assert result["phase"] == expected_phase


@pytest.mark.parametrize("clean", [False, True])
def test_host_receipt_has_attempt_and_source_identity_on_failure(tmp_path: Path, monkeypatch,
                                                               clean: bool) -> None:
    identity = {"source_tree_sha256": "a" * 64, "git_commit": "b" * 40, "git_clean": clean}
    monkeypatch.setattr(acceptance, "source_identity", lambda *a, **kw: identity)

    def fail(*args, **kwargs):
        raise ValueError("runtime failure")

    monkeypatch.setattr(acceptance, "_run_case", fail)
    output = tmp_path / "host"
    monkeypatch.setattr(sys, "argv", ["acceptance", "--output", str(output)])
    assert acceptance.main() == 1
    receipt = json.loads((output / "acceptance.json").read_text())
    assert receipt.items() >= identity.items()
    assert receipt["status"] == "failed" and receipt["attempt_id"] and receipt["completed_at"]


@pytest.mark.parametrize("change", ["source_tree_sha256", "git_commit", "git_clean"])
def test_identity_changes_prevent_final_success(monkeypatch, change: str) -> None:
    before = {"source_tree_sha256": "a" * 64, "git_commit": "b" * 40, "git_clean": True}
    after = {**before, change: False if change == "git_clean" else "changed"}
    monkeypatch.setattr(gate_support, "source_identity", lambda *args, **kwargs: after)
    with pytest.raises(ValueError, match="changed during"):
        gate_support.check_source_identity(Path("."), before, require_clean=False)


def test_release_mode_requires_clean_commit(monkeypatch) -> None:
    monkeypatch.setattr(gate_support, "source_sha256", lambda root: "a" * 64)
    monkeypatch.setattr(gate_support, "git_state", lambda root: ("b" * 40, False))
    with pytest.raises(ValueError, match="clean committed"):
        gate_support.source_identity(Path("."), require_clean=True)


@pytest.mark.parametrize("job", ["host", "strict", "paper-depth"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", None])
def test_aggregate_ci_cannot_pass_when_any_release_job_did_not_succeed(job: str,
                                                                    result: str) -> None:
    needs = {name: {"result": "success"} for name in ("host", "strict", "paper-depth")}
    needs[job]["result"] = result
    with pytest.raises(ValueError, match="must succeed"):
        ci_gate.verify_results(needs)


def test_aggregate_ci_requires_all_three_dependencies() -> None:
    ci_gate.verify_results({name: {"result": "success"}
                            for name in ("host", "strict", "paper-depth")})
    for value in ([], {}, {"host": {"result": "success"}}):
        with pytest.raises(ValueError, match="dependencies"):
            ci_gate.verify_results(value)
