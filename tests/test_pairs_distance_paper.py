from __future__ import annotations

import json
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import DatasetDeclaration, MarketEvent, RunPolicy
from paperquant.package import inspect_package, run_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "strategies/rule.pairs_distance"
FIXTURE = ROOT / "examples/three_stock_pairs"
SYMBOLS = ("ALPHA", "BETA", "GAMMA")


def _events(swap: bool) -> tuple[MarketEvent, ...]:
    original = TypeAdapter(tuple[MarketEvent, ...]).validate_json(
        (FIXTURE / "events.json").read_bytes()
    )
    if not swap:
        return original
    days = [original[offset : offset + 3] for offset in range(0, len(original), 3)]
    result = []
    for day in days:
        by_symbol = {event.symbol: event for event in day}
        for event in day:
            replacement = (
                "GAMMA" if event.symbol == "BETA" else
                "BETA" if event.symbol == "GAMMA" else event.symbol
            )
            result.append(event.model_copy(update={"values": by_symbol[replacement].values}))
    return tuple(result)


def _oracle(
    events: tuple[MarketEvent, ...],
) -> tuple[tuple[str, str], list[tuple[Decimal, ...] | None]]:
    days = [events[offset : offset + 3] for offset in range(0, len(events), 3)]
    prices = {symbol: [next(event.values["close"] for event in day if event.symbol == symbol)
                       for day in days] for symbol in SYMBOLS}
    normalized = {symbol: [price / prices[symbol][0] for price in series]
                  for symbol, series in prices.items()}
    pairs = list(combinations(SYMBOLS, 2))
    chosen = min(pairs, key=lambda pair: sum(
        (a - b) ** 2 for a, b in zip(normalized[pair[0]][:252],
                                      normalized[pair[1]][:252], strict=True)
    ))
    spread_series = [a - b for a, b in zip(normalized[chosen[0]][:252],
                                            normalized[chosen[1]][:252], strict=True)]
    mean = sum(spread_series) / Decimal(252)
    sigma = (sum((spread - mean) ** 2 for spread in spread_series) / Decimal(252)).sqrt()
    assert sigma > 0
    direction = 0
    expected = []
    for index in range(len(days)):
        if index < 252 or index >= 377:
            expected.append(None)
            continue
        spread = normalized[chosen[0]][index] - normalized[chosen[1]][index]
        if direction:
            if (direction > 0 and spread >= 0) or (direction < 0 and spread <= 0) or index == 376:
                direction = 0
                expected.append((Decimal(0), Decimal(0)))
            else:
                expected.append(None)
        elif abs(spread) > 2 * sigma:
            direction = -1 if spread > 0 else 1
            amounts = (Decimal(100) / prices[chosen[0]][index],
                       Decimal(100) / prices[chosen[1]][index])
            expected.append((direction * amounts[0], -direction * amounts[1]))
        else:
            expected.append(None)
    return chosen, expected


@pytest.mark.parametrize("swap", [False, True])
def test_minimum_distance_pair_and_trades_match_independent_oracle(
    swap: bool, tmp_path: Path
) -> None:
    claim = json.loads((ROOT / "research/claims/pairs_distance.json").read_text())
    assert len(claim["claims"]) == 3
    inspect_package("pairs-inspection", PACKAGE)
    events = _events(swap)
    dataset = DatasetDeclaration(
        dataset_id=f"synthetic-pairs-{'swapped' if swap else 'base'}",
        granularity="day",
        fields=frozenset(events[0].values),
        symbols=frozenset(SYMBOLS),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    report = run_package(
        run_id=dataset.dataset_id,
        package_dir=PACKAGE,
        dataset=dataset,
        engine_capability=reference_capability(dataset.fields),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(),
        output=tmp_path,
    )
    pair, expected = _oracle(events)
    actual = []
    for day in range(len(events) // 3):
        actions = report.decisions[3 * day + 2].actions
        if actions[0].kind == "target_position":
            assert tuple(action.symbol for action in actions) == pair
            actual.append(tuple(action.quantity for action in actions))
        else:
            actual.append(None)
    assert actual == expected
    assert pair == (("ALPHA", "GAMMA") if swap else ("ALPHA", "BETA"))
    assert len(report.fills) >= 8
    assert {fill.symbol for fill in report.fills} == set(pair)
    assert all(report.decisions[3 * day].actions[0].kind == "none"
               for day in range(len(events) // 3))


def test_fixture_is_pinned_and_synchronized() -> None:
    dataset = DatasetDeclaration.model_validate_json((FIXTURE / "dataset.json").read_bytes())
    events = _events(False)
    assert dataset.content_sha256 == fingerprint(events)
    assert dataset.event_count == 3 * 379
    for offset in range(0, len(events), 3):
        day = events[offset : offset + 3]
        assert tuple(event.symbol for event in day) == SYMBOLS
        assert len({event.available_time for event in day}) == 1
