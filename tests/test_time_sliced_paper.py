from __future__ import annotations

import importlib.util
import json
import random
from decimal import Decimal
from pathlib import Path

import pytest

from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import RunPolicy
from paperquant.package import inspect_package, run_package
from paperquant.public_data import load_plotly_apple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/rule.time_sliced_execution"
CSV = ROOT / "data/public/finance-charts-apple.csv"


def _strategy(*, parent: Decimal, slices: int, side: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("time_sliced", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TimeSlicedExecution(parent_quantity=parent, slices=slices, side=side)


@pytest.mark.parametrize("seed", [3, 17, 71])
def test_random_parent_schedules_preserve_volume_and_terminal_silence(seed: int) -> None:
    claim = json.loads((ROOT / "research/claims/time_sliced_execution.json").read_text())
    assert claim["source"]["version"] == "arXiv:2208.06244v1"
    inspect_package("twap-paper", PACKAGE)
    generator = random.Random(seed)
    _, events = load_plotly_apple(CSV)
    for _ in range(12):
        slices = generator.randrange(2, 11)
        quantity = Decimal(slices * generator.randrange(1, 12 // slices + 1))
        side = generator.choice(("buy", "sell"))
        strategy = _strategy(parent=quantity, slices=slices, side=side)
        output = [strategy.decide(event, None)[0] for event in events[: slices + 3]]
        children = output[:slices]
        assert all(action.kind == "submit_order" for action in children)
        assert all(action.side == side for action in children)
        assert all(action.quantity == quantity / Decimal(slices) for action in children)
        assert sum((action.quantity for action in children), Decimal(0)) == quantity
        assert len({action.client_order_id for action in children}) == slices
        assert all(action.kind == "none" for action in output[slices:])


def test_public_twap_has_twelve_child_orders_and_next_open_fills(tmp_path: Path) -> None:
    dataset, events = load_plotly_apple(CSV)
    report = run_package(
        run_id="public-time-sliced",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(symbol_map={"AAPL": "DEMO"}),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )
    assert report.status == "succeeded"
    assert len(report.decisions) == len(events) == 506
    assert len(report.fills) == 12
    assert [decision.actions[0].kind for decision in report.decisions[:12]] == ["submit_order"] * 12
    assert all(decision.actions[0].kind == "none" for decision in report.decisions[12:])
    assert [fill.quantity for fill in report.fills] == [Decimal(1)] * 12
    assert [fill.timestamp for fill in report.fills] == [
        event.bar_open_time for event in events[1:13]
    ]
    assert report.accounts[-1].positions[0].quantity == Decimal(12)
