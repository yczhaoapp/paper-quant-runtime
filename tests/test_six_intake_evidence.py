"""Source binding, trajectory causality and fail-closed supplement controls."""

from __future__ import annotations

import importlib.util
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

import scripts.accept_six_papers as gate
from paperquant.models import TrainingRequest
from paperquant.papers import verify_recipe
from scripts.acceptance import _check_package_inventory
from scripts.build_six_paper_inputs import CLASSES, quote_events

ROOT = Path(__file__).resolve().parents[1]


def test_twap_children_follow_actual_warmup_and_declared_interval():
    for side in (-1, 1):
        stream = quote_events(81, hawkes=True, meta_side=side, steps=1972)
        assert stream[0].event_id == "quote:00470"
        start = stream[0].event_time
        assert all((e.event_time - start).total_seconds() >= 0 for e in stream)
        children = [e for e in stream if e.values["meta_child"] == 1]
        assert len(children) == 8
        for i, child in enumerate(children):
            elapsed = int(child.event_id.split(":")[1]) * 0.213
            assert 100 + 40 * i <= elapsed < 100 + 40 * i + 0.213
            assert child.values["twap_present"] == side
            assert child.values["buy_flow" if side > 0 else "sell_flow"] >= 1
    assert not any(e.values["meta_child"] for e in quote_events(81, hawkes=True, steps=1972))


@pytest.mark.parametrize("name,dt", [("sac_passive_quotes", 0.001), ("ppo_twap_aware", 0.213)])
def test_training_rewards_and_next_observations_are_bound_to_actual_quote_ledger(name, dt):
    folder = ROOT / "examples/six-paper" / name
    request = TrainingRequest.model_validate_json((folder / "training.json").read_bytes())
    for path in sorted((folder / "collection").glob("*.json")):
        trace = json.loads(path.read_text())
        samples = [s for s in request.samples if s.episode_id == f"episode-{trace['episode']}"]
        accounts, events = trace["accounts"], trace["source_events"]
        assert len(samples) == len(accounts) - 1 == len(events) - 1
        assert sum(s.terminal for s in samples) == 1 and samples[-1].terminal
        for i, sample in enumerate(samples):
            quantity = sum(float(p["quantity"]) for p in accounts[i + 1]["positions"])
            expected = float(accounts[i + 1]["equity"]) - float(accounts[i]["equity"])
            expected -= 0.001 * abs(quantity) * dt
            assert float(sample.reward) == pytest.approx(expected, abs=2e-10)
            assert float(sample.next_features["inventory"]) == quantity
            assert float(sample.next_features["price"]) == pytest.approx(
                (
                    float(events[i + 1]["values"]["bid_price"])
                    + float(events[i + 1]["values"]["ask_price"])
                )
                / 2
            )
            if name == "sac_passive_quotes":
                latent = sample.features["behavior_latent"]
                assert sample.action == (1 if latent > 1 / 3 else -1 if latent < -1 / 3 else 0)


@pytest.mark.parametrize("name", ["sac_passive_quotes", "ppo_twap_aware"])
def test_rl_training_responds_to_changed_reward_without_fixture_specific_branches(name):
    request = TrainingRequest.model_validate_json(
        (ROOT / "examples/six-paper" / name / "training.json").read_bytes()
    )
    samples = request.samples[:32]
    samples = (*samples[:-1], samples[-1].model_copy(update={"terminal": True}))
    original = request.model_copy(update={"samples": samples})
    changed = original.model_copy(
        update={
            "samples": tuple(
                s.model_copy(update={"reward": s.reward + Decimal("0.31")}) for s in samples
            )
        }
    )
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "strategies" / ("supplement." + name) / "strategy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, CLASSES[name])
    left = json.loads(cls().train(original))
    right = json.loads(cls().train(changed))
    assert left["training"]["parameter_change_l1"] > 0
    assert right["training"]["parameter_change_l1"] > 0
    assert left["training"]["reward_sum"] != right["training"]["reward_sum"]
    keys = ("actor", "critics") if name == "sac_passive_quotes" else ("decision", "action", "value")
    assert any(left[key] != right[key] for key in keys)


def test_multi_algorithm_mapping_checks_all_sources_and_rejects_claim_substitution(tmp_path):
    original = ROOT / "research/six-paper/cases/ppo_twap_aware/recipe.json"
    recipe = json.loads(original.read_text())
    if any(
        not (original.parent / s["file"]).is_file()
        for s in [recipe["source"], *recipe["additional_sources"]]
    ):
        pytest.skip("Prepare the explicit supporting paper cache")
    checked = verify_recipe(original)
    assert len(checked["additional_source_sha256"]) == 2
    assert {a["source_index"] for a in checked["anchors"]} == {0, 1, 2}
    claims = json.loads((original.parent / recipe["claim_file"]).read_text())
    # A complete relative-path recipe fixture also works when the repository
    # and the system temporary directory are on different Windows volumes.
    for key in ("package", "claim_file", "spec_file"):
        source = (original.parent / recipe[key]).resolve()
        destination = tmp_path / key
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copyfile(source, destination)
        recipe[key] = destination.name
    for index, source in enumerate([recipe["source"], *recipe["additional_sources"]]):
        destination = tmp_path / f"paper-{index}.pdf"
        shutil.copyfile((original.parent / source["file"]).resolve(), destination)
        source["file"] = destination.name
    claims["algorithm_sources"][1]["url"] = "https://example.org/unrelated-algorithm"
    (tmp_path / "claims.json").write_text(json.dumps(claims))
    recipe["claim_file"] = "claims.json"
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="Supporting algorithm sources"):
        verify_recipe(tmp_path / "recipe.json")


def test_supplement_gate_invalidates_old_success_when_new_source_preparation_fails(
    tmp_path, monkeypatch
):
    receipt = tmp_path / "six-paper.json"
    receipt.write_text('{"status":"passed","attempt_id":"old"}')

    def unavailable(**kwargs):
        raise ValueError("missing new source")

    monkeypatch.setattr(gate, "prepare", unavailable)
    result = gate.accept(tmp_path)
    actual = json.loads(receipt.read_text())
    assert actual == result and actual["status"] == "failed"
    assert actual["attempt_id"] != "old" and actual["completed_at"]
    assert "missing new source" in actual["reason"]


def test_nonexistent_oracle_parent_is_rejected_before_runtime():
    with pytest.raises(ValueError, match="could not be collected"):
        gate.expanded_oracles(["tests/test_six_paper_methods.py::nonexistent_method"])


def test_separate_catalog_cannot_hide_an_unlisted_package_or_duplicate_case():
    base = {"rule.example"}
    supplement = [{"strategy_id": f"supplement.example{i}"} for i in range(6)]
    identifiers = {case["strategy_id"] for case in supplement}
    assert _check_package_inventory(base, base, base | identifiers, supplement) == identifiers
    with pytest.raises(ValueError, match="runnable packages differ"):
        _check_package_inventory(
            base, base, base | identifiers | {"supplement.unlisted"}, supplement
        )
    with pytest.raises(ValueError, match="runnable packages differ"):
        _check_package_inventory(base, base, base | identifiers, supplement[:-1] + [supplement[0]])
