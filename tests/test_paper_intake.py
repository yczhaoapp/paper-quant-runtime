from __future__ import annotations

import hashlib
import io
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from pypdf import PdfWriter

from paperquant.cli import main
from paperquant.models import StrategyDeclaration
from paperquant.package import write_manifest
from paperquant.papers import extract_text, fetch_paper_source, verify_recipe

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "research/recipes/yang-malik-rl1.json"


def test_paper_run_links_fixed_source_to_cold_replayed_bundle(tmp_path: Path) -> None:
    destination = tmp_path / "paper-run"
    assert main([
        "paper-run", "--recipe", str(ROOT / "research/recipes/pardo-time-sliced.json"),
        "--dataset", str(ROOT / "examples/public_aapl/dataset.json"),
        "--events", str(ROOT / "examples/public_aapl/events.json"),
        "--policy", str(ROOT / "examples/public_aapl/policy.json"),
        "--output", str(destination),
    ]) == 0
    receipt = json.loads((destination / "paper-run.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["strategy_id"] == "rule.time_sliced_execution"
    assert receipt["source_sha256"]
    assert receipt["bundle_sha256"]


@pytest.mark.parametrize(
    ("recipe_file", "fixture_name", "package_name", "training_file", "pages"),
    [
        ("chipwanya-logistic.json", "public_direction", "supervised.logistic_direction",
         "training.json", (2, 3)),
        ("pardo-time-sliced.json", "public_aapl", "rule.time_sliced_execution",
         None, (11, 7)),
    ],
)
def test_additional_real_papers_bind_to_runnable_strategies(
    recipe_file: str, fixture_name: str, package_name: str,
    training_file: str | None, pages: tuple[int, ...], tmp_path: Path,
) -> None:
    recipe = ROOT / "research/recipes" / recipe_file
    checked = verify_recipe(recipe)
    assert checked["strategy_id"] == package_name
    assert [anchor["page"] for anchor in checked["anchors"]] == list(pages)
    source_file = json.loads(recipe.read_text())["source"]["file"]
    source = (recipe.parent / source_file).resolve()
    assert checked["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert main(["paper-check", "--recipe", str(recipe),
                 "--output", str(tmp_path / "paper-check")]) == 0
    fixture = ROOT / "examples" / fixture_name
    arguments = [
        "run", "--package", str(ROOT / "strategies" / package_name),
        "--dataset", str(fixture / "dataset.json"),
        "--events", str(fixture / "events.json"),
        "--policy", str(fixture / "policy.json"),
        "--output", str(tmp_path / "run"),
    ]
    if training_file is not None:
        arguments.extend(["--training", str(fixture / training_file)])
    assert main(arguments) == 0
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["status"] == "succeeded" and report["fills"]


def _copy_recipe(tmp_path: Path) -> tuple[Path, dict]:
    original = json.loads(RECIPE.read_text())
    paper_copy = tmp_path / "paper.pdf"
    paper_copy.write_bytes((ROOT / "research/sources/yang-malik-2024.pdf").read_bytes())
    original["source"]["file"] = "paper.pdf"
    shutil.copytree(ROOT / "strategies/reinforcement.pairs_actor_critic",
                    tmp_path / "strategy", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (tmp_path / "claim.json").write_bytes(
        (ROOT / "research/claims/pairs_actor_critic.json").read_bytes())
    (tmp_path / "spec.json").write_bytes(
        (ROOT / "research/specs/pairs-actor-critic.json").read_bytes())
    original["package"] = "strategy"
    original["claim_file"] = "claim.json"
    original["spec_file"] = "spec.json"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    return path, original


def test_real_cc_by_pdf_is_bound_to_claims_and_executable_package(tmp_path: Path) -> None:
    checked = verify_recipe(RECIPE)
    assert checked["page_count"] == 19
    assert checked["format"] == "pdf"
    assert [anchor["page"] for anchor in checked["anchors"]] == [7, 8, 9, 12]
    assert checked["source_sha256"] == (
        "9b30affe7f856c1add02ff4fb8bb7f3eac670e877a19bf3c91a18bbf4fc98eb3")
    assert main(["paper-check", "--recipe", str(RECIPE),
                 "--output", str(tmp_path / "paper-check")]) == 0
    assert json.loads((tmp_path / "paper-check/attempt.json").read_text())["status"] == (
        "succeeded")
    fixture = ROOT / "examples/pairs_actor_critic"
    assert main([
        "run", "--package", str(ROOT / "strategies/reinforcement.pairs_actor_critic"),
        "--dataset", str(fixture / "dataset.json"),
        "--events", str(fixture / "events.json"),
        "--training", str(fixture / "training.json"),
        "--policy", str(fixture / "policy.json"),
        "--output", str(tmp_path / "pair-run"),
    ]) == 0
    run = json.loads((tmp_path / "pair-run/report.json").read_text())
    assert run["status"] == "succeeded" and len(run["fills"]) > 20


def test_changed_pdf_and_missing_anchor_fail_closed(tmp_path: Path) -> None:
    recipe_path, recipe = _copy_recipe(tmp_path)
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(paper.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest"):
        verify_recipe(recipe_path)
    assert main(["paper-check", "--recipe", str(recipe_path),
                 "--output", str(tmp_path / "check")]) == 2
    assert json.loads((tmp_path / "check/attempt.json").read_text())["status"] == "failed"
    assert not (tmp_path / "check/report.json").exists()

    paper.write_bytes((ROOT / "research/sources/yang-malik-2024.pdf").read_bytes())
    recipe["anchors"][0]["text"] = "an absent private trading formula"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor missing"):
        verify_recipe(recipe_path)


def test_changed_reviewed_method_spec_fails_closed(tmp_path: Path) -> None:
    recipe_path, recipe = _copy_recipe(tmp_path)
    spec_path = tmp_path / "spec.json"
    changed = json.loads(spec_path.read_text())
    changed["steps"][0]["implementation"] = "ImaginaryPaperStrategy.decide"
    spec_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned digest"):
        verify_recipe(recipe_path)
    recipe["spec_sha256"] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation symbol"):
        verify_recipe(recipe_path)


@pytest.mark.parametrize(("parent_quantity", "slices"), [(12, 3), (10, 5)])
def test_unlisted_paper_package_enters_runtime_without_catalog_changes(
    parent_quantity: int, slices: int, tmp_path: Path,
) -> None:
    original = json.loads((ROOT / "research/recipes/pardo-time-sliced.json").read_text())
    strategy_id = f"outside.catalog.paper_schedule_{parent_quantity}_{slices}"
    package = tmp_path / "package"
    package.mkdir()
    source = '''\
from decimal import Decimal
from paperquant.strategy_api import (
    DataNeed, Granularity, NoOp, StrategyDeclaration, StrategyKind, SubmitOrder,
)
class ExternalSchedule:
    declaration = StrategyDeclaration(
        strategy_id="STRATEGY_ID", kind=StrategyKind.RULE,
        data=DataNeed(granularity=Granularity.DAY,
                      fields=frozenset({"open", "close"}),
                      symbols=frozenset({"DEMO"})),
        actions=frozenset({"none", "submit_order"}), training_required=False,
        source="SOURCE_URL",
    )
    def __init__(self):
        self.index = 0
    def decide(self, event, account):
        del account
        self.index += 1
        if self.index > SLICES:
            return (NoOp(reason="schedule complete"),)
        return (SubmitOrder(client_order_id=f"child-{self.index}",
                            symbol=event.symbol, side="buy",
                            quantity=Decimal("CHILD_QUANTITY"),
                            reason="reviewed advance schedule"),)
'''.replace("STRATEGY_ID", strategy_id).replace("SOURCE_URL", original["source"]["url"])
    source = source.replace("SLICES", str(slices)).replace(
        "CHILD_QUANTITY", str(parent_quantity // slices))
    # Preserve the reviewed source bytes on every platform, including Windows.
    (package / "strategy.py").write_bytes(source.encode("utf-8"))
    declaration = StrategyDeclaration.model_validate({
        "strategy_id": strategy_id, "kind": "rule",
        "data": {"granularity": "day", "fields": ["open", "close"],
                 "symbols": ["DEMO"]},
        "actions": ["none", "submit_order"], "training_required": False,
        "source": original["source"]["url"],
    })
    write_manifest(package, declaration, "ExternalSchedule")
    claim = {
        "strategy_id": strategy_id,
        "source": {"url": original["source"]["url"]},
        "claims": [{"locator": "Section 3.3"}, {"locator": "Section 2.4, PDF page 7"}],
    }
    (tmp_path / "claim.json").write_text(json.dumps(claim), encoding="utf-8")
    spec = {
        "schema_version": "1.0", "strategy_id": strategy_id,
        "source_sha256": original["source"]["sha256"],
        "fields": ["open", "close"],
        "steps": [{
            "role": "execution", "claim_index": index,
            "paper_rule": "Choose an advance child-order schedule from the paper.",
            "runtime_rule": f"Emit {slices} equal children totaling {parent_quantity} shares.",
            "implementation": "ExternalSchedule.decide",
            "oracle_node": ("tests/test_paper_intake.py::"
                            "test_unlisted_paper_package_enters_runtime_without_catalog_changes"
                            f"[{parent_quantity}-{slices}]"),
            "adaptation": "Daily market orders replace intraday order-book child orders.",
        } for index in (0, 1)],
        "excluded_paper_components": ["The source's RL agent is outside this schedule case."],
    }
    spec_bytes = (json.dumps(spec, sort_keys=True) + "\n").encode()
    (tmp_path / "spec.json").write_bytes(spec_bytes)
    (tmp_path / "paper.pdf").write_bytes(
        (ROOT / "research/sources/pardo-2022.pdf").read_bytes())
    original["source"]["file"] = "paper.pdf"
    original["package"] = "package"
    original["package_source_sha256"] = hashlib.sha256(
        (package / "strategy.py").read_bytes()).hexdigest()
    original["claim_file"] = "claim.json"
    original["spec_file"] = "spec.json"
    original["spec_sha256"] = hashlib.sha256(spec_bytes).hexdigest()
    original["expected"]["strategy_id"] = strategy_id
    original["review_note"] = (
        "Unlisted reviewed schedule uses a different parent quantity and number of children. "
        "The real paper source stays fixed while the strategy package is generated externally.")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(original), encoding="utf-8")
    assert strategy_id not in (ROOT / "examples/catalog.json").read_text()
    assert verify_recipe(recipe_path)["strategy_id"] == strategy_id
    output = tmp_path / "run"
    assert main([
        "paper-run", "--recipe", str(recipe_path),
        "--dataset", str(ROOT / "examples/public_aapl/dataset.json"),
        "--events", str(ROOT / "examples/public_aapl/events.json"),
        "--policy", str(ROOT / "examples/public_aapl/policy.json"),
        "--output", str(output),
    ]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["plan"]["strategy_id"] == strategy_id
    assert len(report["fills"]) == slices
    assert sum(Decimal(fill["quantity"]) for fill in report["fills"]) == parent_quantity
    receipt = json.loads((output / "paper-run.json").read_text())
    assert receipt["method_spec_sha256"] == original["spec_sha256"]


def test_html_and_tex_text_adapters_are_explicitly_limited() -> None:
    html = (
        b"<html><head><style>hidden</style></head><body><h1>Pair strategy</h1>"
        b"<p>A &amp; B</p><script>secret()</script></body></html>"
    )
    extracted_html = extract_text(html, "html")
    assert len(extracted_html) == 1
    assert "Pair strategy" in extracted_html[0] and "A & B" in extracted_html[0]
    assert "hidden" not in extracted_html[0] and "secret" not in extracted_html[0]
    tex = b"\\section{Pair strategy}\nA % note\n\\textbf{spread}\\n"
    extracted_tex = extract_text(tex, "tex")
    assert "Pair strategy" in extracted_tex[0] and "spread" in extracted_tex[0]
    assert "note" not in extracted_tex[0]


def test_scanned_pdf_is_not_falsely_treated_as_text() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    from io import BytesIO

    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="no extractable text"):
        extract_text(output.getvalue(), "pdf")


def test_explicit_fetch_publishes_only_digest_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        def geturl(self) -> str:
            return "https://papers.example/article.pdf"

    content = b"%PDF-1.7 research paper bytes"
    monkeypatch.setattr("paperquant.papers.urllib.request.urlopen",
                        lambda request, timeout: Response(content))
    destination = tmp_path / "paper.pdf"
    destination.write_bytes(b"earlier verified paper")
    with pytest.raises(ValueError, match="differs"):
        fetch_paper_source("https://papers.example/article.pdf", "0" * 64, destination)
    assert destination.read_bytes() == b"earlier verified paper"
    assert not list(tmp_path.glob(".paper-*"))
    digest = hashlib.sha256(content).hexdigest()
    assert fetch_paper_source("https://papers.example/article.pdf", digest, destination) == digest
    assert destination.read_bytes() == content
    assert not list(tmp_path.glob(".paper-*"))
    with pytest.raises(ValueError, match="HTTPS"):
        fetch_paper_source("http://papers.example/article.pdf", digest, destination)
