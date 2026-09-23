from __future__ import annotations

import importlib.util
import json
import random
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import AccountSnapshot, DatasetDeclaration, MarketEvent, RunPolicy
from paperquant.package import inspect_package, run_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/rule.microprice_toy"
FIXTURE = ROOT / "examples/l1_queues"


def _strategy():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("microprice_toy", PACKAGE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WeightedMicropriceToy()


@pytest.mark.parametrize("seed", [3, 17, 71])
def test_toy_price_matches_independent_rational_formula_and_trade_mapping(seed: int) -> None:
    claim = json.loads((ROOT / "research/claims/microprice_toy.json").read_text())
    assert claim["source"]["companion_slides"].endswith("Stoikov.pdf")
    inspect_package("microprice-paper", PACKAGE)
    generator = random.Random(seed)
    account = AccountSnapshot(
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        cash=Decimal(100000),
        equity=Decimal(100000),
        positions=(),
        open_order_ids=(),
    )
    policy = _strategy()
    for index in range(50):
        bid_cents = generator.randrange(5000, 20000)
        spread_cents = generator.randrange(1, 6)
        bid_size = generator.randrange(1, 101)
        ask_size = generator.randrange(1, 101)
        bid = Decimal(bid_cents) / 100
        ask = Decimal(bid_cents + spread_cents) / 100
        rational = Fraction(
            (bid_cents + spread_cents) * bid_size + bid_cents * ask_size,
            100 * (bid_size + ask_size),
        )
        expected = Decimal(rational.numerator) / Decimal(rational.denominator)
        for trade_price, expected_target in (
            (bid, Decimal(1)),
            (ask, Decimal(-1)),
        ):
            event = MarketEvent(
                event_id=f"book-{index}-{expected_target}",
                symbol="DEMO",
                event_time=account.timestamp,
                available_time=account.timestamp,
                values={
                    "price": trade_price,
                    "bid_price": bid,
                    "ask_price": ask,
                    "bid_size": Decimal(bid_size),
                    "ask_size": Decimal(ask_size),
                },
            )
            prediction, action = policy.decide(event, account)
            assert abs(prediction.value - expected) < Decimal("1e-24")
            assert bid < prediction.value < ask
            assert action.kind == "target_position"
            assert action.quantity == expected_target


def test_synthetic_l1_toy_price_completes_runtime_and_trades(tmp_path: Path) -> None:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    report = run_package(
        run_id="microprice-toy",
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )
    assert report.status == "succeeded"
    assert len(report.decisions) == dataset.event_count
    assert len(report.fills) > 2
    for event, decision in zip(events, report.decisions, strict=True):
        bid = event.values["bid_price"]
        ask = event.values["ask_price"]
        imbalance = event.values["bid_size"] / (event.values["bid_size"] + event.values["ask_size"])
        expected = bid + imbalance * (ask - bid)
        assert abs(decision.actions[0].value - expected) < Decimal("1e-24")


def test_invalid_book_fails_instead_of_producing_a_price() -> None:
    policy = _strategy()
    with pytest.raises(ValueError, match="positive ordered quotes"):
        policy.fair_price(
            {
                "bid_price": Decimal(101),
                "ask_price": Decimal(100),
                "bid_size": Decimal(1),
                "ask_size": Decimal(1),
            }
        )
