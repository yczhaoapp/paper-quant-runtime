"""Fail-closed controls for complete original-paper acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.accept_paper_depth as depth
from paperquant.papers import verify_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_all_eighteen_recipes_cover_every_reviewed_claim_when_sources_ready() -> None:
    recipes = [*(ROOT / "research/recipes").glob("*.json"),
               *(ROOT / "research/deep-recipes").glob("*.json")]
    assert len(recipes) == 18
    if any(not (path.parent / json.loads(path.read_text())["source"]["file"]).is_file()
           for path in recipes):
        pytest.skip("Prepare the hash-pinned paper cache for the full offline check")
    checked = [verify_recipe(path) for path in recipes]
    assert len({item["strategy_id"] for item in checked}) == 18
    double_q = next(item for item in checked if item["strategy_id"] == (
        "reinforcement.double_q_market_making"))
    assert len(double_q["additional_source_sha256"]) == 1
    assert {anchor["source_index"] for anchor in double_q["anchors"]} == {0, 1}


def test_second_algorithm_pdf_cannot_be_replaced_without_detection(tmp_path: Path) -> None:
    recipe_path = ROOT / "research/deep-recipes/double_q_market_making.json"
    recipe = json.loads(recipe_path.read_text())
    original = (recipe_path.parent / recipe["source"]["file"]).resolve()
    if not original.is_file():
        pytest.skip("Prepare the hash-pinned paper cache for the second-source test")
    (tmp_path / "primary.pdf").write_bytes(original.read_bytes())
    (tmp_path / "secondary.pdf").write_bytes(b"wrong algorithm paper")
    recipe["source"]["file"] = "primary.pdf"
    recipe["additional_sources"][0]["file"] = "secondary.pdf"
    changed = tmp_path / "recipe.json"
    changed.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="pinned digest"):
        verify_recipe(changed)


def test_changed_training_payload_fails_pinned_depth_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = json.loads(depth.INPUT_LOCK.read_text())
    training = lock["cases"]["supervised.logistic_direction"]["files"]["training"]
    training["sha256"] = "0" * 64
    changed = tmp_path / "inputs.json"
    changed.write_text(json.dumps(lock))
    monkeypatch.setattr(depth, "INPUT_LOCK", changed)
    catalog = json.loads((ROOT / "examples/catalog.json").read_text())
    with pytest.raises(ValueError, match="Pinned training bytes differ"):
        depth._check_inputs(catalog["cases"])


def test_failed_new_attempt_invalidates_previous_success_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "paper-depth.json"
    receipt.write_text('{"status":"passed","attempt_id":"prior"}')

    def unavailable(*, fetch: bool = False) -> list[dict[str, object]]:
        del fetch
        raise ValueError("source unavailable this attempt")

    monkeypatch.setattr(depth, "verify_sources", unavailable)
    result = depth.run_depth(tmp_path)
    published = json.loads(receipt.read_text())
    assert result["status"] == published["status"] == "failed"
    assert published["attempt_id"] != "prior"
    assert "source unavailable this attempt" in published["reason"]
