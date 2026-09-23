from __future__ import annotations

import importlib.util
import json
import random
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import DatasetDeclaration, MarketEvent, RunPolicy
from paperquant.package import inspect_package, run_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/rule.reservation_quote"
FIXTURE = ROOT / "examples/l2_book"


def _maker(**parameters):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("reservation_quote", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReservationQuote(**parameters)


@pytest.mark.parametrize("seed", [3, 17, 71])
def test_random_equations_match_independent_reservation_and_spread(seed: int) -> None:
    claim = json.loads((ROOT / "research/claims/reservation_quote.json").read_text())
    assert [item["locator"] for item in claim["claims"]] == [
        "Equation (29), PDF page 5",
        "Equation (30), PDF page 5",
    ]
    inspect_package("reservation-paper", PACKAGE)
    generator = random.Random(seed)
    for _ in range(40):
        midpoint = Decimal(generator.randrange(9000, 11001)) / 100
        inventory = Decimal(generator.randrange(-2, 3))
        remaining = Decimal(generator.randrange(0, 101)) / 100
        gamma = Decimal(generator.randrange(1, 11)) / 100
        volatility = Decimal(generator.randrange(1, 6))
        decay = Decimal(generator.randrange(5, 26)) / 10
        reservation, bid, ask = _maker(
            risk_aversion=gamma, volatility=volatility, arrival_decay=decay
        ).quotes(midpoint, inventory, remaining)
        risk_term = gamma * volatility**2 * remaining
        expected_reservation = midpoint - inventory * risk_term
        expected_width = risk_term + (Decimal(2) / gamma) * (Decimal(1) + gamma / decay).ln()
        assert abs(reservation - expected_reservation) < Decimal("1e-24")
        assert abs((ask - bid) - expected_width) < Decimal("1e-24")
        assert abs((bid + ask) / 2 - reservation) < Decimal("1e-24")
        higher_inventory = _maker(
            risk_aversion=gamma, volatility=volatility, arrival_decay=decay
        ).quotes(midpoint, inventory + 1, remaining)
        assert higher_inventory[1] <= bid
        assert higher_inventory[2] <= ask


def test_l2_quote_run_has_price_bound_limits_and_real_fills(tmp_path: Path) -> None:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    assert len(dataset.fields) == 9
    report = run_package(
        run_id="reservation-quote",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )
    assert report.status == "succeeded"
    assert len(report.decisions) == len(events) == 32
    assert len(report.fills) > 10
    assert {order.status for order in report.orders} == {"accepted", "filled", "expired"}
    by_time = {event.available_time: event for event in events}
    for fill in report.fills:
        event = by_time[fill.timestamp]
        expected = event.values["ask_price" if fill.side == "buy" else "bid_price"]
        assert fill.price == expected
        assert fill.timestamp > events[0].available_time
    for index, (decision, account, event) in enumerate(
        zip(report.decisions, report.accounts, events, strict=True)
    ):
        inventory = next((position.quantity for position in account.positions), Decimal(0))
        midpoint = (event.values["bid_price"] + event.values["ask_price"]) / 2
        remaining = max(Decimal(0), Decimal(31 - index) / 31)
        independently_expected = midpoint - inventory * Decimal("0.1") * Decimal(4) * remaining
        assert decision.actions[0].kind == "prediction"
        assert decision.actions[0].value == independently_expected
    assert report.decisions[-1].actions[-1].kind == "none"


def test_invalid_l2_depth_is_rejected_before_quoting() -> None:
    with pytest.raises(ValueError, match="crossed or unordered"):
        _maker()._check_book(
            {
                "bid_price": Decimal(100),
                "ask_price": Decimal(101),
                "bid_price_2": Decimal(102),
                "ask_price_2": Decimal(103),
                "bid_size": Decimal(1),
                "ask_size": Decimal(1),
                "bid_size_2": Decimal(1),
                "ask_size_2": Decimal(1),
            }
        )
