from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import DatasetDeclaration, Granularity, MarketEvent, RunPolicy
from paperquant.package import run_package
from paperquant.public_data import load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/rule.channel_breakout"


def _series(seed: int) -> tuple[MarketEvent, ...]:
    generator = random.Random(seed)
    level = Decimal("100")
    start = datetime(2026, 1, 1, 16, tzinfo=UTC)
    result = []
    for index in range(230):
        level += Decimal(generator.choice((-3, -2, -1, 0, 1, 2, 3)))
        if index in (52, 87, 122, 157, 192):
            level += Decimal(generator.choice((-14, 14)))
        result.append(
            MarketEvent(
                event_id=f"oracle-day-{index}",
                symbol="DEMO",
                bar_open_time=start + timedelta(days=index, hours=-6),
                event_time=start + timedelta(days=index),
                available_time=start + timedelta(days=index),
                values={
                    "open": level,
                    "high": level + 1,
                    "low": level - 1,
                    "close": level,
                    "volume": Decimal("1000"),
                },
            )
        )
    return tuple(result)


def _independent_signal(prices: list[Decimal]) -> list[Decimal | None]:
    expected: list[Decimal | None] = []
    next_exit: int | None = None
    for index, price in enumerate(prices):
        if next_exit is not None:
            if index == next_exit:
                expected.append(Decimal(0))
                next_exit = None
            else:
                expected.append(None)
            continue
        if index < 50:
            expected.append(None)
            continue
        earlier = prices[index - 50 : index]
        resistance = Decimal("1.01") * max(earlier)
        support = Decimal("0.99") * min(earlier)
        if prices[index - 1] < resistance < price:
            expected.append(Decimal(1))
            next_exit = index + 10
        elif prices[index - 1] > support > price:
            expected.append(Decimal(-1))
            next_exit = index + 10
        else:
            expected.append(None)
    return expected


@pytest.mark.parametrize("seed", [3, 17, 41, 73])
def test_paper_trb_matches_independent_crossing_and_holding_oracle(
    seed: int, tmp_path: Path
) -> None:
    claim = json.loads((ROOT / "research/claims/channel_breakout.json").read_text())
    assert claim["source"]["version"] == "arXiv:1504.04254v1"
    events = _series(seed)
    fields = frozenset(events[0].values)
    dataset = DatasetDeclaration(
        dataset_id=f"trb-oracle-{seed}",
        granularity=Granularity.DAY,
        fields=fields,
        symbols=frozenset({"DEMO"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    report = run_package(
        run_id=f"trb-oracle-{seed}",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(fields),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )
    expected = _independent_signal([event.values["close"] for event in events])
    actual = [
        decision.actions[0].quantity
        if decision.actions[0].kind == "target_position"
        else None
        for decision in report.decisions
    ]
    assert actual == expected
    assert len(report.fills) > 0
    assert set(actual) >= {None, Decimal(0)}


def test_channel_breakout_runs_on_pinned_public_bars(tmp_path: Path) -> None:
    dataset, events = load_plotly_apple(ROOT / "data/public/finance-charts-apple.csv")
    report = run_package(
        run_id="public-trb",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )
    expected = _independent_signal([event.values["close"] for event in events])
    actual = [
        action.quantity if action.kind == "target_position" else None
        for decision in report.decisions
        for action in decision.actions
    ]
    assert actual == expected
    assert len(report.fills) > 10
    assert report.plan.conversions[0].parameters == {"AAPL": "DEMO"}
