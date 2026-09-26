"""Paper intake failures and the scope of standalone versus method evidence."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from paperquant.cli import main
from paperquant.models import ErrorCode, Failure, Stage
from paperquant.papers import PaperParseError, extract_text, verify_recipe
from scripts.accept_paper_depth import _check_method_oracle_nodes
from scripts.acceptance import _check_paper_run_scope, _run_oracles

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "research/recipes/pardo-time-sliced.json"


def _copy_recipe(root: Path) -> tuple[Path, dict]:
    root.mkdir()
    recipe = json.loads(RECIPE.read_text())
    for key in ("claim_file", "spec_file"):
        original = (RECIPE.parent / recipe[key]).resolve()
        recipe[key] = original.name
        (root / original.name).write_bytes(original.read_bytes())
    (root / "paper.pdf").write_bytes((RECIPE.parent / recipe["source"]["file"]).read_bytes())
    recipe["source"]["file"] = "paper.pdf"
    shutil.copytree((RECIPE.parent / recipe["package"]).resolve(), root / "package",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    recipe["package"] = "package"
    path = root / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    return path, recipe


def _arguments(command: str, recipe: Path, output: Path) -> list[str]:
    arguments = [command, "--recipe", str(recipe), "--output", str(output)]
    if command == "paper-run":
        fixture = ROOT / "examples/public_aapl"
        for flag, filename in (("--dataset", "dataset.json"), ("--events", "events.json"),
                               ("--policy", "policy.json")):
            arguments.extend([flag, str(fixture / filename)])
    return arguments


@pytest.mark.parametrize("command", ["paper-check", "paper-run"])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("truncation", ["header", "half"])
def test_truncated_pdf_finishes_attempt_and_invalidates_success(
    command: str, reuse: bool, truncation: str, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recipe_path, recipe = _copy_recipe(tmp_path / "input")
    output = tmp_path / "run"
    arguments = _arguments(command, recipe_path, output)
    prior_id = None
    if reuse:
        assert main(arguments) == 0
        prior_id = json.loads((output / "attempt.json").read_text())["attempt_id"]
        assert (output / "report.json").is_file()
    capsys.readouterr()
    source = recipe_path.parent / "paper.pdf"
    raw = source.read_bytes()
    damaged = raw[:16 if truncation == "header" else len(raw) // 2]
    source.write_bytes(damaged)
    # Reach the parser, rather than merely testing the source-digest guard.
    recipe["source"]["sha256"] = hashlib.sha256(damaged).hexdigest()
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    assert main(arguments) == 2
    failure_bytes = (output / "failure.json").read_bytes()
    failure = Failure.model_validate_json(failure_bytes)
    assert failure.stage == Stage.INPUT and failure.code == ErrorCode.INPUT_INVALID
    assert failure.details["cause"] == "PaperParseError"
    assert "PDF parsing failed" in str(failure.details["reason"])
    attempt = json.loads((output / "attempt.json").read_text())
    assert attempt["status"] == "failed" and attempt["completed_at"]
    assert attempt["attempt_id"] != prior_id
    assert attempt["failure_sha256"] == hashlib.sha256(failure_bytes).hexdigest()
    assert all(not (output / name).exists()
               for name in ("report.json", "bundle.json", "paper-run.json"))
    assert "Traceback" not in capsys.readouterr().err
    with pytest.raises(PaperParseError, match="PDF parsing failed"):
        extract_text(damaged, "pdf")


def test_page_extraction_errors_use_the_same_failure_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        def extract_text(self) -> str:
            raise RuntimeError("damaged content stream")

    class Reader:
        is_encrypted = False
        pages = [Page()] * 11

    monkeypatch.setattr("paperquant.papers.PdfReader", lambda stream: Reader())
    with pytest.raises(PaperParseError, match="RuntimeError.*damaged content stream"):
        extract_text(b"PDF bytes", "pdf")
    with pytest.raises(PaperParseError, match="RuntimeError.*damaged content stream"):
        verify_recipe(RECIPE)
    output = tmp_path / "run"
    assert main(_arguments("paper-run", RECIPE, output)) == 2
    assert json.loads((output / "attempt.json").read_text())["status"] == "failed"
    assert Failure.model_validate_json((output / "failure.json").read_bytes()).code == (
        ErrorCode.INPUT_INVALID)


@pytest.mark.parametrize("phase", ["replay", "publication"])
def test_failure_after_runtime_removes_partial_success_outputs(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"

    def broken_replay(**kwargs: object) -> None:
        assert (output / "report.json").is_file() and (output / "bundle.json").is_file()
        raise RuntimeError("replay failed unexpectedly")

    def broken_completion(*args: object, **kwargs: object) -> None:
        assert (output / "paper-run.json").is_file()
        raise RuntimeError("attempt finalization failed unexpectedly")

    if phase == "replay":
        monkeypatch.setattr("paperquant.cli.replay_bundle", broken_replay)
    else:
        monkeypatch.setattr("paperquant.output.Attempt.succeeded", broken_completion)
    assert main(_arguments("paper-run", RECIPE, output)) == 2
    assert json.loads((output / "attempt.json").read_text())["status"] == "failed"
    assert (output / "failure.json").is_file()
    assert all(not (output / name).exists()
               for name in ("report.json", "bundle.json", "paper-run.json"))


def test_nonexistent_oracle_cannot_be_mistaken_for_a_method_pass(tmp_path: Path) -> None:
    recipe_path, recipe = _copy_recipe(tmp_path / "input")
    spec_path = recipe_path.parent / recipe["spec_file"]
    spec = json.loads(spec_path.read_text())
    absent = "tests/test_nonexistent_method.py::test_no_such_oracle"
    for step in spec["steps"]:
        step["oracle_node"] = absent
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    recipe["spec_sha256"] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    output = tmp_path / "run"
    assert main(_arguments("paper-run", recipe_path, output)) == 0
    receipt = json.loads((output / "paper-run.json").read_text())
    assert receipt["status_scope"] == "source_mapping_runtime_and_replay"
    assert receipt["source_mapping"] == receipt["runtime_validation"] == (
        receipt["replay_validation"]) == "passed"
    assert receipt["method_validation"] == "not_run"
    assert set(receipt["declared_method_oracle_nodes"]) == {absent}
    _check_paper_run_scope(receipt)
    with pytest.raises(ValueError, match="bound oracle nodes did not pass"):
        _run_oracles(tmp_path / "acceptance", {"method": {"node_ids": [absent]}})
    receipt["method_validation"] = "passed"
    with pytest.raises(ValueError, match="misstates its validation scope"):
        _check_paper_run_scope(receipt)


def test_twap_schedule_locator_matches_the_original_section_heading() -> None:
    recipe = json.loads(RECIPE.read_text())
    claim = json.loads((RECIPE.parent / recipe["claim_file"]).read_text())
    locator = claim["claims"][1]["locator"]
    match = re.fullmatch(r"Section (\d+\.\d+), PDF page (\d+)", locator)
    assert match is not None
    pages = extract_text((RECIPE.parent / recipe["source"]["file"]).read_bytes(), "pdf")
    page = " ".join(pages[int(match[2]) - 1].split())
    assert f"{match[1]} The Execution Algo Class" in page
    assert "volume may be decided a priori" in page


@pytest.mark.parametrize(("declared", "valid"), [
    ("tests/test_oracle.py::test_formula", True),
    ("tests/test_oracle.py::test_formula[3]", True),
    ("tests/test_oracle.py::test_formula[99]", False),
    ("tests/test_oracle.py::test_form", False),
    ("tests/test_oracle.py::test_absent", False),
])
def test_method_validation_resolves_only_passed_pytest_nodes(declared: str, valid: bool) -> None:
    passed = ("tests/test_oracle.py::test_formula[3]", "tests/test_oracle.py::test_formula[17]")
    if valid:
        _check_method_oracle_nodes((declared,), passed)
    else:
        with pytest.raises(ValueError, match="outside this passed attempt"):
            _check_method_oracle_nodes((declared,), passed)
