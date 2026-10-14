"""Independent multi-step checks for audited method and simulator boundaries."""

import json
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from paperquant.models import AccountSnapshot, MarketEvent, NoOp, SubmitOrder, TrainingRequest
from paperquant.numerics import Scalar
from tests.test_six_paper_methods import case
from tests.test_six_quote_contract import declaration, events, execute


@pytest.mark.parametrize("seed", [11, 29, 83])
def test_ou_complete_rolling_entry_hold_and_median_exit(seed):
    agent = case("ou_percentile", "OUPercentilePairs")()
    rng = np.random.default_rng(seed)
    history = rng.normal(size=90).tolist()
    agent.parameters = {"mu": 0, "sigma": 1}
    agent.z_history = history.copy()
    position = 0
    seen = set()
    for day in range(200):
        lower, median, upper = np.quantile(history[-90:], [0.25, 0.5, 0.75])
        spread = float(rng.normal() * 3)
        if position == 0:
            position = -1 if spread > upper else 1 if spread < lower else 0
        elif (position > 0 and spread >= median) or (position < 0 and spread <= median):
            position = 0
        seen.add(position)
        stamp = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day)
        account = AccountSnapshot(
            timestamp=stamp,
            cash=Decimal(100000),
            equity=Decimal(100000),
            positions=(),
            open_order_ids=(),
        )
        for symbol, price in (("OU_A", 200 + spread), ("OU_B", 200)):
            actions = agent.decide(
                MarketEvent(
                    event_id=f"{day}:{symbol}",
                    symbol=symbol,
                    event_time=stamp,
                    available_time=stamp,
                    values={"close": Decimal(str(price))},
                ),
                account,
            )
        assert [int(a.quantity) for a in actions] == [position, -position]
        history.append(spread)
        assert agent.z_history == pytest.approx(history[-90:])
    assert seen == {-1, 0, 1}


@pytest.mark.parametrize("seed", [7, 31, 79])
def test_ppo_multistep_gradient_reset_matches_independent_finite_difference(seed):
    cls = case("ppo_twap_aware", "PPOTWAPAware")
    rng = random.Random(seed)
    parameters = [Scalar(rng.uniform(-0.2, 0.2)) for _ in range(3)]
    rate = 0.001
    for step in range(48):
        mode, reward = step % 4, rng.uniform(-0.3, 0.3)

        def objective(values, mode=mode, reward=reward):
            x, y, value = values
            # Skip/mask exclude action parameters; a clipped constant excludes both actors.
            actor = x * y if mode == 0 else x * 0.25 if mode == 1 else 0.0
            return actor + 0.5 * (value - reward) ** 2

        before = [p.value for p in parameters]
        eps = 1e-6
        expected = []
        for i, old in enumerate(before):
            plus, minus = before.copy(), before.copy()
            plus[i] += eps
            minus[i] -= eps
            expected.append(old - rate * (objective(plus) - objective(minus)) / (2 * eps))
        loss = 0.5 * (parameters[2] - reward) ** 2
        if mode == 0:
            loss += parameters[0] * parameters[1]
        elif mode == 1:
            loss += parameters[0] * 0.25
        cls.update_parameters(loss, parameters, rate)
        assert [p.value for p in parameters] == pytest.approx(expected, abs=1e-11)
        if mode in (2, 3):
            assert [p.value for p in parameters[:2]] == before[:2]


def test_ppo_real_training_never_updates_parameters_outside_current_loss(monkeypatch):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ppo_gradient_probe", root / "strategies/supplement.ppo_twap_aware/strategy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backward, optimizer = Scalar.backward, module.gradient_step
    visited, observations = set(), []

    def observe_backward(loss):
        visited.clear()
        pending = [loss]
        while pending:
            node = pending.pop()
            if node not in visited:
                visited.add(node)
                pending.extend(node.children)
        backward(loss)

    def observe_step(parameters, *args, **kwargs):
        absent = [p for p in parameters if p not in visited]
        before = [p.value for p in absent]
        assert all(p.grad == 0 for p in absent)
        optimizer(parameters, *args, **kwargs)
        assert [p.value for p in absent] == before
        observations.append(len(absent))

    monkeypatch.setattr(Scalar, "backward", observe_backward)
    monkeypatch.setattr(module, "gradient_step", observe_step)
    request = TrainingRequest.model_validate_json(
        (root / "examples/six-paper/ppo_twap_aware/training.json").read_bytes()
    )
    model = json.loads(module.PPOTWAPAware().train(request))
    assert len(observations) == model["training"]["optimizer_steps"] == 128
    assert any(observations)
    assert model["training"]["parameter_change_l1"] > 0


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_interval_probability_simulator_explicitly_allows_off_touch_resting_fill(side, tmp_path):
    class Probe:
        declaration = declaration()

        def decide(self, event, account):
            if event.event_id == "e0":
                return (
                    SubmitOrder(
                        client_order_id="quote",
                        symbol="MM",
                        side=side,
                        quantity=Decimal(1),
                        limit_price=event.values["bid_price" if side == "buy" else "ask_price"],
                        reason="interval probability boundary",
                    ),
                )
            return (NoOp(reason="wait"),)

    stream = events([50, 51 if side == "buy" else 49], flow=1, draw="0.1")
    report = execute(Probe(), stream, tmp_path)
    assert len(report.fills) == 1
    assert report.fills[0].price == stream[0].values["bid_price" if side == "buy" else "ask_price"]
    assert report.orders[-1].reason == "nonadverse_order_flow"
    # This tests a declared interval-probability approximation, not exchange price priority.


def test_sac_collected_rewards_match_separate_marked_wealth_and_penalty():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    samples = json.loads(
        (root / "examples/six-paper/sac_passive_quotes/training.json").read_text()
    )["samples"]
    traces = sorted((root / "examples/six-paper/sac_passive_quotes/collection").glob("*.json"))
    assert traces and samples
    for trace_path in traces:
        trace = json.loads(trace_path.read_text())
        assert trace["fills"]
    # Exact per-transition reward is independently checked below through the collector ledger.
    from scripts.build_six_paper_inputs import collect_episode, quote_events

    cls = case("sac_passive_quotes", "SACPassiveQuotes")
    transitions, trace = collect_episode(cls, quote_events(137), seed=37)
    accounts = trace["accounts"]
    for sample, previous, current in zip(transitions, accounts[:-1], accounts[1:], strict=True):
        wealth_change = Decimal(current["equity"]) - Decimal(previous["equity"])
        inventory = sum(abs(Decimal(p["quantity"])) for p in current["positions"])
        expected = wealth_change - Decimal(".001") * inventory * Decimal(".001")
        assert float(sample.reward) == pytest.approx(float(expected), abs=1e-10)


def test_sac_execution_uses_trained_policy_and_inventory_bounded_quote_ledger():
    from pathlib import Path

    from paperquant.compiler import compile_run, fingerprint
    from paperquant.models import DatasetDeclaration, RunPolicy
    from paperquant.passive_engine import PassiveQuoteEngine
    from scripts.build_six_paper_inputs import quote_events

    root = Path(__file__).resolve().parents[1]
    request = TrainingRequest.model_validate_json(
        (root / "examples/six-paper/sac_passive_quotes/training.json").read_bytes()
    )
    cls = case("sac_passive_quotes", "SACPassiveQuotes")
    payload = cls().train(request)
    agent = cls()
    agent.load(payload)
    stream = quote_events(81037, steps=200)
    stream = tuple(
        e.model_copy(
            update={
                "values": {
                    **e.values,
                    "buy_flow": Decimal(1),
                    "sell_flow": Decimal(1),
                    "uniform_buy": Decimal(0),
                    "uniform_sell": Decimal(0),
                }
            }
        )
        for e in stream
    )
    engine = PassiveQuoteEngine()
    dataset = DatasetDeclaration(
        contract_version="1.1",
        dataset_id=request.dataset_id,
        granularity="tick",
        market_kind="quote_l1",
        fields=frozenset(stream[0].values),
        symbols=frozenset({"MM"}),
        event_count=len(stream),
        content_sha256=fingerprint(stream),
    )
    plan, effective = compile_run(
        run_id="sac-execution-oracle",
        strategy=agent.declaration,
        dataset=dataset,
        engine=engine.capability(dataset.fields),
        engine_profile=engine.profile(),
        policy=RunPolicy(),
        events=stream,
        training=request,
    )
    _, orders, fills, accounts = engine.run(plan=plan, strategy=agent, events=effective)
    assert fills and any(o.reason == "nonadverse_order_flow" for o in orders)
    assert all(abs(p.quantity) <= 5 for a in accounts for p in a.positions)
    cash = Decimal(100000)
    by_time = {}
    for fill in fills:
        by_time.setdefault(fill.timestamp, []).append(fill)
    for event, account in zip(stream, accounts, strict=True):
        for fill in by_time.pop(account.timestamp, []):
            cash -= fill.price * (1 if fill.side == "buy" else -1)
        assert account.cash == cash
        mark = (event.values["bid_price"] + event.values["ask_price"]) / 2
        assert account.equity == cash + sum(p.quantity * mark for p in account.positions)
    assert not by_time
