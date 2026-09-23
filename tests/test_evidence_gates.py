from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_ACCEPTANCE = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                 / "scripts/acceptance.py"))
_check_implementation = _ACCEPTANCE["_check_implementation"]
_run_oracles = _ACCEPTANCE["_run_oracles"]


def test_nonexistent_claim_symbol_fails_before_acceptance() -> None:
    root = Path(__file__).resolve().parents[1]
    claim = json.loads((root / "research/claims/sarsa_inventory.json").read_text())
    claim["claims"][0]["implementation"] = "ClassThatDoesNotExist.fit"
    with pytest.raises(ValueError, match="implementation class is absent"):
        _check_implementation(claim, "reinforcement.sarsa_inventory")


def test_empty_or_missing_oracle_nodes_cannot_pass(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no nodes"):
        _run_oracles(tmp_path, {"strategy": {"node_ids": []}})
    with pytest.raises(ValueError, match="did not pass"):
        _run_oracles(tmp_path, {"strategy": {
            "node_ids": ["tests/test_evidence_gates.py::test_deleted_oracle_node"]
        }})
