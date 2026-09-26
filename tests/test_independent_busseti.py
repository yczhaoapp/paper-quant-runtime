"""Independent new-paper case: source, reviewed build, formula, and runtime."""

from __future__ import annotations

import json
import random
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperquant.cli import main
from paperquant.papers import verify_recipe
from scripts.build_busseti_case import CACHE, REVIEW, _source_code, build_case

ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((REVIEW / "source.json").read_text())
METHOD = json.loads((REVIEW / "method.json").read_text())


@pytest.mark.parametrize("seed", [3, 17, 41, 73])
def test_generated_schedule_matches_independent_formula(seed: int) -> None:
    randomizer = random.Random(seed)
    weights = [Decimal(randomizer.randint(1, 20)) for _ in range(randomizer.randint(2, 8))]
    parent = Decimal(randomizer.randint(5, 50))
    method = {
        **METHOD,
        "parent_quantity": str(parent),
        "expected_volume_profile": [str(value) for value in weights],
    }
    namespace: dict = {}
    exec(_source_code(method, SOURCE["url"]), namespace)
    strategy = namespace["StaticVolumeShare"]()
    event = SimpleNamespace(symbol="DEMO")
    actions = [strategy.decide(event, None)[0] for _ in range(len(weights) + 2)]
    actual = [action.quantity for action in actions[: len(weights)]]
    expected = [parent * weight / sum(weights) for weight in weights[:-1]]
    expected.append(parent - sum(expected))
    assert actual == expected
    assert sum(actual) == parent
    assert all(action.kind == "none" for action in actions[len(weights) :])


def test_independent_paper_source_build_and_runtime(tmp_path: Path) -> None:
    if not CACHE.is_file():
        pytest.skip("Fetch pinned Busseti PDF before this offline source test")
    build = tmp_path / "built"
    receipt = build_case(CACHE, build)
    checked = verify_recipe(build / "recipe.json")
    assert checked["source_sha256"] == SOURCE["sha256"]
    assert checked["strategy_id"] == METHOD["strategy_id"]
    assert receipt["strategy_source_sha256"] == checked["package_source_sha256"]
    fixture = ROOT / "examples/independent_busseti"
    output = tmp_path / "run"
    assert (
        main(
            [
                "paper-run",
                "--recipe",
                str(build / "recipe.json"),
                "--dataset",
                str(fixture / "dataset.json"),
                "--events",
                str(fixture / "events.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    run = json.loads((output / "report.json").read_text())
    quantities = [Decimal(fill["quantity"]) for fill in run["fills"]]
    assert quantities == [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")]
    assert sum(quantities) == Decimal("10")
    assert run["status"] == "succeeded"
    assert json.loads((output / "paper-run.json").read_text())["status"] == "passed"


def test_changed_independent_source_fails_before_package_build(tmp_path: Path) -> None:
    if not CACHE.is_file():
        pytest.skip("Fetch pinned Busseti PDF before this offline source test")
    changed = tmp_path / "changed.pdf"
    changed.write_bytes(CACHE.read_bytes() + b"different bytes")
    with pytest.raises(ValueError, match="pinned original bytes"):
        build_case(changed, tmp_path / "built")
    assert not (tmp_path / "built").exists()
