from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_contract import bars, dataset, declaration

from paperquant.cli import main
from paperquant.models import ErrorCode, Failure, RunReport
from paperquant.package import write_manifest

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "examples/strict-policy.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("PAPERQUANT_TEST_DOCKER") != "1",
    reason="strict container verification is a separate required release gate",
)


@pytest.mark.parametrize(
    ("package", "fixture", "training"),
    [
        ("rule.moving_average_crossover", "daily_vma", None),
        ("rule.channel_breakout", "daily_vma", None),
        ("rule.microprice_toy", "l1_queues", None),
        ("rule.reservation_quote", "l2_book", None),
        ("rule.pairs_distance", "three_stock_pairs", None),
        ("supervised.queue_imbalance", "l1_queues", "training.json"),
        ("reinforcement.sarsa_inventory", "l1_queues", "sarsa_training.json"),
        ("reinforcement.double_q_market_making", "l1_queues", "sarsa_training.json"),
        ("reinforcement.execution_value_learning", "execution_lattice", "training.json"),
        ("reinforcement.risk_averse_bandit", "bandit_l1", "training.json"),
        ("reinforcement.pairs_actor_critic", "pairs_actor_critic", "training.json"),
    ],
)
def test_strict_worker_matches_host_behavior(
    package: str, fixture: str, training: str | None, tmp_path: Path
) -> None:
    base = [
        "--package",
        str(ROOT / "strategies" / package),
        "--dataset",
        str(ROOT / "examples" / fixture / "dataset.json"),
        "--events",
        str(ROOT / "examples" / fixture / "events.json"),
    ]
    if training is not None:
        base.extend(["--training", str(ROOT / "examples" / fixture / training)])
    host_output = tmp_path / "host"
    strict_output = tmp_path / "strict"
    assert main(["run", *base, "--output", str(host_output)]) == 0
    assert main(["run", *base, "--policy", str(STRICT), "--output", str(strict_output)]) == 0
    host = RunReport.model_validate_json((host_output / "report.json").read_bytes())
    strict = RunReport.model_validate_json((strict_output / "report.json").read_bytes())
    assert host.plan.sandbox == "development"
    assert strict.plan.sandbox == "strict"
    assert strict.worker_receipt is not None
    assert strict.worker_receipt.source_sha256
    assert len(strict.worker_receipt.container_id) == 64
    assert strict.decisions == host.decisions
    assert strict.orders == host.orders
    assert strict.fills == host.fills
    assert strict.accounts == host.accounts
    assert strict.final_equity == host.final_equity


@pytest.mark.parametrize(
    ("package", "fixture_name", "training_file", "symbol_map"),
    [
        ("supervised.logistic_direction", "public_direction", "training.json", {"AAPL": "DEMO"}),
        ("supervised.ridge_three_day_price", "public_ridge", "training.json", {"AAPL": "DEMO"}),
        ("supervised.gaussian_nb_direction", "public_gaussian", "training.json", {"AAPL": "DEMO"}),
        ("supervised.random_forest_operations", "public_forest", "training.json", {"AAPL": "DEMO"}),
        (
            "reinforcement.actor_critic_allocation",
            "public_actor_critic",
            "training.json",
            {"AAPL": "DEMO"},
        ),
        ("supervised.cross_sectional_rank", "cross_sectional_rank", "training.json", {}),
        ("rule.time_sliced_execution", "public_aapl", None, {"AAPL": "DEMO"}),
    ],
)
def test_extended_strategy_matches_strict_worker(
    package: str,
    fixture_name: str,
    training_file: str | None,
    symbol_map: dict[str, str],
    tmp_path: Path,
) -> None:
    fixture = ROOT / "examples" / fixture_name
    base = [
        "run",
        "--package",
        str(ROOT / "strategies" / package),
        "--dataset",
        str(fixture / "dataset.json"),
        "--events",
        str(fixture / "events.json"),
    ]
    if training_file is not None:
        base.extend(["--training", str(fixture / training_file)])
    strict_policy = tmp_path / "strict-policy.json"
    strict_policy.write_text(
        json.dumps({"sandbox": "strict", "symbol_map": symbol_map}),
        encoding="utf-8",
    )
    host_output = tmp_path / "host"
    strict_output = tmp_path / "strict"
    assert (
        main([*base, "--policy", str(fixture / "policy.json"), "--output", str(host_output)])
        == 0
    )
    assert main([*base, "--policy", str(strict_policy), "--output", str(strict_output)]) == 0
    host = RunReport.model_validate_json((host_output / "report.json").read_bytes())
    strict = RunReport.model_validate_json((strict_output / "report.json").read_bytes())
    assert strict.worker_receipt is not None
    assert (strict.artifact is None) == (host.artifact is None)
    if strict.artifact is not None and host.artifact is not None:
        assert strict.artifact.content_sha256 == host.artifact.content_sha256
    assert strict.decisions == host.decisions
    assert strict.orders == host.orders
    assert strict.fills == host.fills
    assert strict.accounts == host.accounts
    assert strict.final_equity == host.final_equity


def test_dynamic_import_cannot_write_to_package_mount(tmp_path: Path) -> None:
    package = tmp_path / "probe"
    package.mkdir()
    source = """\
import json
from paperquant.strategy_api import NoOp, StrategyDeclaration, StrategyKind, DataNeed, Granularity

class Probe:
    declaration = StrategyDeclaration(
        strategy_id="sample-rule",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.MINUTE,
            fields=frozenset({"close", "open"}),
            symbols=frozenset({"AAA"}),
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=False,
        source="public-method-description",
        max_abs_position=2,
    )

    def decide(self, event, account):
        builtins = json.__dict__["__builtins__"]
        importer = (
            builtins["__import__"]
            if isinstance(builtins, dict)
            else getattr(builtins, "__import__")
        )
        path = importer("pathlib").Path("/package/injected.txt")
        path.write_text("unexpected write")
        return (NoOp(reason="probe"),)
"""
    (package / "strategy.py").write_text(source, encoding="utf-8")
    write_manifest(package, declaration(), "Probe")
    market = bars()
    dataset_file = tmp_path / "dataset.json"
    events_file = tmp_path / "events.json"
    dataset_file.write_text(dataset(market).model_dump_json(), encoding="utf-8")
    events_file.write_text(
        json.dumps([event.model_dump(mode="json") for event in market]), encoding="utf-8"
    )
    output = tmp_path / "strict-probe"
    assert (
        main(
            [
                "run",
                "--package",
                str(package),
                "--dataset",
                str(dataset_file),
                "--events",
                str(events_file),
                "--policy",
                str(STRICT),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    failure = Failure.model_validate_json((output / "failure.json").read_bytes())
    assert failure.code == ErrorCode.BACKTEST_FAILED
    assert not (package / "injected.txt").exists()
    assert not (output / "report.json").exists()
