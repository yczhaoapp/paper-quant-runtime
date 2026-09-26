"""Assurance levels are scoped reviewed claims, not transferable badges."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.accept_paper_depth import _check_public_supplement
from scripts.acceptance import PUBLIC_CASES
from scripts.assurance_axes import MATRIX, validate_assurance

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "examples/catalog.json").read_text())["cases"]
CATALOG_IDS = {entry["strategy"] for entry in CATALOG}
PUBLIC_IDS = {strategy_id for strategy_id, _ in PUBLIC_CASES}


def test_four_axes_do_not_transfer_core_or_host_evidence_to_full_method_or_strict() -> None:
    result = validate_assurance(CATALOG_IDS, PUBLIC_IDS)
    assert result["catalog_cases"] == 18
    assert result["core_counts"] == {"A1": 5, "A2": 13}
    assert result["strategy_counts"] == {"A0": 10, "A1": 8}
    assert result["data_counts"] == {"D0": 10, "D1": 8}
    assert result["experiment_counts"] == {"E0": 18}
    assert "runtime" not in json.loads(MATRIX.read_text())
    assert result["independent"]["strategy_level"] == "A0"


@pytest.mark.parametrize(
    ("strategy_id", "mutation", "message"),
    [
        ("reinforcement.actor_critic_allocation",
         lambda item: item["strategy"].update({"level": "A2"}),
         "Strategy grade exceeds"),
        ("supervised.queue_imbalance",
         lambda item: item.update({"data": "D1"}),
         "Data grade differs"),
        ("rule.channel_breakout",
         lambda item: item.update({"claim_sha256": "0" * 64}),
         "detached from reviewed claims"),
        ("rule.moving_average_crossover",
         lambda item: item.update({"runtime": "R2"}),
         "Assurance fields differ"),
    ],
)
def test_assurance_grade_cannot_outlive_its_scope_or_evidence(
    tmp_path: Path, strategy_id: str, mutation: Callable[[dict[str, Any]], Any], message: str,
) -> None:
    matrix = json.loads(MATRIX.read_text())
    mutation(matrix["cases"][strategy_id])
    changed = tmp_path / "matrix.json"
    changed.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match=message):
        validate_assurance(CATALOG_IDS, PUBLIC_IDS, matrix_path=changed)


def test_public_supplement_pins_both_additional_d1_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import accept_paper_depth as depth

    locked = json.loads(depth.INPUT_LOCK.read_text())
    assert set(_check_public_supplement(CATALOG)) == {
        "rule.moving_average_crossover", "rule.channel_breakout"
    }
    locked["public_supplement"]["rule.channel_breakout"]["files"]["events"]["sha256"] = (
        "0" * 64
    )
    changed = tmp_path / "inputs.json"
    changed.write_text(json.dumps(locked))
    monkeypatch.setattr(depth, "INPUT_LOCK", changed)
    with pytest.raises(ValueError, match="Pinned public events bytes differ"):
        _check_public_supplement(CATALOG)
