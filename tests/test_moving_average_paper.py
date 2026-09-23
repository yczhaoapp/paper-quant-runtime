from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.cli import main
from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import DatasetDeclaration, Granularity, MarketEvent, RunPolicy
from paperquant.package import inspect_package, run_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/rule.moving_average_crossover"


def _market(seed: int) -> tuple[MarketEvent, ...]:
    generator = random.Random(seed)
    close = Decimal("100")
    events = []
    start = datetime(2026, 1, 1, 16, tzinfo=UTC)
    for index in range(80):
        close += Decimal(generator.choice((-5, -3, -1, 1, 3, 5)))
        events.append(
            MarketEvent(
                event_id=f"day-{index:03d}",
                symbol="DEMO",
                event_time=start + timedelta(days=index),
                available_time=start + timedelta(days=index),
                bar_open_time=start + timedelta(days=index, hours=-6),
                values={
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": Decimal("1000"),
                },
            )
        )
    return tuple(events)


@pytest.mark.parametrize("seed", [2, 17, 29])
def test_paper_vma_signal_matches_independent_rolling_formula(seed: int, tmp_path: Path) -> None:
    inspect_package("paper-package", PACKAGE)
    claim = json.loads((ROOT / "research/claims/moving_average_crossover.json").read_text())
    assert claim["source"]["version"] == "arXiv:1504.04254v1"
    events = _market(seed)
    fields = frozenset(events[0].values)
    dataset = DatasetDeclaration(
        dataset_id=f"oracle-{seed}",
        granularity=Granularity.DAY,
        fields=fields,
        symbols=frozenset({"DEMO"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    report = run_package(
        run_id=f"paper-oracle-{seed}",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(fields),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )
    assert len(report.decisions) == 80
    for index, decision in enumerate(report.decisions):
        action = decision.actions[0]
        if index < 19:
            assert action.kind == "none"
            continue
        closes = [event.values["close"] for event in events[: index + 1]]
        short = sum(closes[-2:]) / 2
        long = sum(closes[-20:]) / 20
        expected = Decimal("1") if short > long else Decimal("-1") if short < long else Decimal("0")
        actual_position = report.accounts[index].positions
        current = actual_position[0].quantity if actual_position else Decimal("0")
        if current == expected:
            assert action.kind == "none"
        else:
            assert action.kind == "target_position"
            assert action.quantity == expected

    assert {
        action.quantity
        for decision in report.decisions
        for action in decision.actions
        if action.kind == "target_position"
    } >= {Decimal("-1"), Decimal("1")}


def test_committed_daily_fixture_runs_with_two_opposing_trades(tmp_path: Path) -> None:
    fixture = ROOT / "examples/daily_vma"
    dataset = DatasetDeclaration.model_validate_json((fixture / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (fixture / "events.json").read_bytes()
    )
    assert dataset.content_sha256 == fingerprint(events)
    output = tmp_path / "daily-run"
    assert (
        main(
            [
                "run",
                "--package",
                str(PACKAGE),
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
    report = json.loads((output / "report.json").read_text())
    assert len(report["fills"]) == 2
    assert [fill["side"] for fill in report["fills"]] == ["buy", "sell"]
