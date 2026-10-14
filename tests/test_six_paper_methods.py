"""Independent formula and algorithm oracles for the six bounded adaptations."""

from __future__ import annotations

import importlib.util
import itertools
import math
import random
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from paperquant.models import AccountSnapshot, MarketEvent
from paperquant.numerics import Conv1D, LSTMCell, Scalar

ROOT = Path(__file__).resolve().parents[1]


def case(name, cls):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "strategies" / ("supplement." + name) / "strategy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, cls)


@pytest.mark.parametrize("seed", [3, 17, 41, 73])
def test_matching_is_global_optimum_and_has_no_shared_assets(seed):
    cls = case("graph_matching", "GraphMatchingPairs")
    rng = random.Random(seed)
    for count in (4, 5, 6):
        symbols = tuple(str(i) for i in range(count))
        for _ in range(12):
            edges = list(itertools.combinations(symbols, 2))
            weights = {edge: rng.uniform(-3, 5) for edge in edges}
            actual = cls.maximum_matching(symbols, weights)
            used = [vertex for edge in actual for vertex in edge]
            assert len(used) == len(set(used))
            # Exhaustive edge-subset oracle differs from subset-DP implementation.
            candidates = []
            for size in range(count // 2 + 1):
                for chosen in itertools.combinations(edges, size):
                    vertices = [v for edge in chosen for v in edge]
                    if len(vertices) == len(set(vertices)):
                        candidates.append(sum(weights[edge] for edge in chosen))
            assert sum(weights[edge] for edge in actual) == pytest.approx(max(candidates))


def test_matching_avoids_the_greedy_counterexample():
    cls = case("graph_matching", "GraphMatchingPairs")
    weights = {
        ("A", "B"): 10,
        ("A", "C"): 9,
        ("A", "D"): 0,
        ("B", "C"): 0,
        ("B", "D"): 9,
        ("C", "D"): 1,
    }
    actual = cls.maximum_matching(tuple("ABCD"), weights)
    assert sum(weights[edge] for edge in actual) == 18


@pytest.mark.parametrize("seed", [7, 19, 43])
def test_single_lag_adf_matches_numpy_ols_and_standard_error(seed):
    cls = case("graph_matching", "GraphMatchingPairs")
    rng = np.random.default_rng(seed)
    for rho in (0.2, 0.6, 0.95):
        residual = [0.0]
        for noise in rng.normal(0, 0.4, 200):
            residual.append(rho * residual[-1] + float(noise))
        values = np.array(residual)
        x = np.column_stack((np.ones(len(values) - 2), values[1:-1], np.diff(values[:-1])))
        y = np.diff(values)[1:]
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        variance = ((y - x @ coef) ** 2).sum() / (len(y) - 3)
        expected = coef[1] / np.sqrt(variance * np.linalg.inv(x.T @ x)[1, 1])
        assert cls.adf_statistic(residual) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("seed", [5, 23, 59])
def test_ou_calibration_matches_independent_ar1_equations(seed):
    cls = case("ou_percentile", "OUPercentilePairs")
    rng = np.random.default_rng(seed)
    for rho in (0.4, 0.8, 0.96):
        values = [2.0]
        for noise in rng.normal(0, 0.3, 400):
            values.append(2 + rho * (values[-1] - 2) + float(noise))
        x = np.column_stack((values[:-1], np.ones(len(values) - 1)))
        y = np.array(values[1:])
        a, b = np.linalg.lstsq(x, y, rcond=None)[0]
        errors = y - x @ np.array([a, b])
        sigma_e = math.sqrt(float(errors @ errors) / (len(errors) - 2))
        for dt in (0.5, 1, 2):
            model = cls.calibrate(values, dt)
            assert model["lambda"] == pytest.approx(-math.log(a) / dt)
            assert model["mu"] == pytest.approx(b / (1 - a))
            assert model["sigma"] == pytest.approx(
                sigma_e * math.sqrt(-2 * math.log(a) / (dt * (1 - a * a)))
            )


def test_ou_refuses_nonstationary_calibration_and_quantiles_match_numpy():
    cls = case("ou_percentile", "OUPercentilePairs")
    with pytest.raises(ValueError):
        cls.calibrate([float(i) for i in range(120)])
    rng = random.Random(13)
    for _ in range(32):
        values = [rng.gauss(0, 1) for _ in range(90)]
        for probability in (0.25, 0.5, 0.75):
            assert cls.quantile(values, probability) == pytest.approx(
                np.quantile(values, probability)
            )


@pytest.mark.parametrize("seed", [2, 11, 37])
def test_autodiff_gradients_match_independent_math_and_finite_differences(seed):
    rng = random.Random(seed)
    for _ in range(30):
        xv, yv = rng.uniform(-2, 2), rng.uniform(-2, 2)
        x, y = Scalar(xv), Scalar(yv)
        output = (x * y + x * x).tanh() + x.sigmoid() + (y * y + 1).log()
        output.backward()
        t = math.tanh(xv * yv + xv * xv)
        sigmoid = 1 / (1 + math.exp(-xv))
        expected_x = (1 - t * t) * (yv + 2 * xv) + sigmoid * (1 - sigmoid)
        expected_y = (1 - t * t) * xv + 2 * yv / (yv * yv + 1)
        assert x.grad == pytest.approx(expected_x)
        assert y.grad == pytest.approx(expected_y)


def test_convolution_and_lstm_match_separate_numpy_equations():
    rng = random.Random(17)
    conv = Conv1D(2, 3, 3, rng)
    channels = [[rng.gauss(0, 1) for _ in range(9)] for _ in range(2)]
    actual = conv(channels)
    for c in range(3):
        weights = np.array([[v.value for v in row] for row in conv.weights[c]])
        for t in range(7):
            expected = (
                float((weights * np.array(channels)[:, t : t + 3]).sum()) + conv.biases[c].value
            )
            assert actual[c][t].value == pytest.approx(expected)
    cell = LSTMCell(3, 2, rng)
    hidden, memory = [Scalar(0.2), Scalar(-0.3)], [Scalar(-0.1), Scalar(0.4)]
    inputs = [0.7, -0.2, 0.5]
    h, c = cell(inputs, hidden, memory)
    combined = np.array([0.2, -0.3, *inputs])
    gates = []
    for gate in cell.gates:
        weights = np.array([[v.value for v in row] for row in gate.weights[0]])
        bias = np.array([v.value for v in gate.biases[0]])
        gates.append(weights @ combined + bias)
    expected_cell = 1 / (1 + np.exp(-gates[1])) * np.tanh(gates[0]) + 1 / (
        1 + np.exp(-gates[2])
    ) * np.array([-0.1, 0.4])
    expected_hidden = 1 / (1 + np.exp(-gates[3])) * np.tanh(expected_cell)
    assert [v.value for v in c] == pytest.approx(expected_cell)
    assert [v.value for v in h] == pytest.approx(expected_hidden)


def test_cnn_training_updates_all_model_stages_and_confidence_equations():
    cls = case("grouped_cnn_lstm", "GroupedCNNLSTM")
    model = cls()
    model.initialize(29)
    model.means, model.scales = [0] * 10, [1] * 10
    channels = [[0.1 * math.sin(i + j) + 0.2 for i in range(24)] for j in range(10)]
    loss = -model.forward(channels).log()
    loss.backward()
    for stage in (model.encoders, model.first, model.second, [model.lstm], [model.head]):
        assert any(abs(p.grad) > 1e-14 for network in stage for p in network.parameters())
    for theta in (0.25, 0.5, 1.0):
        assert 1 / (1 + theta) + theta / (1 + theta) == pytest.approx(1)


def test_gmm_density_bic_and_training_cluster_predictions_are_real():
    cls = case("gmm_knn", "GMMConditionalKNN")
    row = [0.3] * 8
    model = {
        "weights": [0.4, 0.6],
        "means": [[0.0] * 8, [1.0] * 8],
        "variances": [[0.5] * 8, [2.0] * 8],
    }
    probabilities, log_likelihood = cls.responsibilities(row, model)
    density = [
        w
        * math.prod(
            math.exp(-((x - m) ** 2) / (2 * v)) / math.sqrt(2 * math.pi * v)
            for x, m, v in zip(row, means, variance, strict=False)
        )
        for w, means, variance in zip(
            model["weights"], model["means"], model["variances"], strict=False
        )
    ]
    assert probabilities == pytest.approx([d / sum(density) for d in density])
    assert log_likelihood == pytest.approx(math.log(sum(density)))
    rng = random.Random(41)
    rows = [[rng.gauss(0, 0.2) + offset for _ in range(8)] for offset in (0, 3) for _ in range(20)]
    fitted = cls.fit_gmm(rows, 2, 11)
    ll = sum(cls.responsibilities(row, fitted)[1] for row in rows)
    assert fitted["bic"] == pytest.approx(-2 * ll + (2 * 2 * 8 + 2 - 1) * math.log(len(rows)))


def test_59_minute_features_match_independent_peak_regression_and_have_no_future_slot():
    cls = case("gmm_knn", "GMMConditionalKNN")
    rng = np.random.default_rng(67)
    for _ in range(20):
        closes = 100 + np.sin(np.arange(59) * 0.6) + rng.normal(0, 0.1, 59)
        changes = np.diff(np.log(closes))
        indices = [
            i for i in range(1, 58) if closes[i] > closes[i - 1] and closes[i] > closes[i + 1]
        ]
        curvatures = np.array([closes[i - 1] - 2 * closes[i] + closes[i + 1] for i in indices])
        heights = closes[indices]
        slope, intercept = np.linalg.lstsq(
            np.column_stack((curvatures, np.ones(len(indices)))), heights, rcond=None
        )[0]
        threshold = 0.002
        expected = [
            np.mean(changes > threshold),
            np.mean(changes < -threshold),
            closes[-1],
            intercept,
            slope,
            curvatures.mean(),
            heights.mean(),
            closes[-6:].mean() / closes.mean() - 1,
        ]
        assert cls.features(list(closes), threshold) == pytest.approx(expected)
        with pytest.raises(ValueError):
            cls.features([*closes, 999999], threshold)


def test_knn_is_not_a_constant_hold_classifier():
    cls = case("gmm_knn", "GMMConditionalKNN")
    rng = random.Random(89)
    windows = [[100 + rng.gauss(0, 1) for _ in range(59)] for _ in range(9)]
    features = [cls.features(w, 0.01) for w in windows]
    labels = [-1, 0, 1] * 3
    agent = cls()
    agent.model = {
        "threshold": 0.01,
        "center": [0] * 8,
        "scales": [1] * 8,
        "gmm": {"weights": [1], "means": [[0] * 8], "variances": [[10000] * 8]},
        "groups": [list(zip(features, labels, strict=False))],
    }
    for window, label in zip(windows, labels, strict=False):
        assert agent.predict(window) == label


def test_sac_soft_bellman_terminal_and_entropy_properties():
    cls = case("sac_passive_quotes", "SACPassiveQuotes")
    rng = random.Random(43)
    for _ in range(100):
        reward, q1, q2, logp = [rng.uniform(-4, 4) for _ in range(4)]
        gamma, alpha = rng.random(), rng.random()
        assert cls.soft_target(reward, False, q1, q2, logp, gamma, alpha) == pytest.approx(
            reward + gamma * (min(q1, q2) - alpha * logp)
        )
        assert cls.soft_target(reward, True, q1, q2, logp, gamma, alpha) == reward
        assert cls.soft_target(reward, False, q1, q2, logp, 0, alpha) == reward
    agent = cls()
    agent.initialize(7)
    for state in ([0, 0], [2, 1], [-3, 0.2]):
        for noise in (-2, 0, 2):
            action, logp = agent.policy(state, noise)
            assert -1 <= action.value <= 1
            assert math.isfinite(logp.value)


def test_ppo_clipping_changes_gradient_by_advantage_sign_and_masks_infeasible_actions():
    cls = case("ppo_twap_aware", "PPOTWAPAware")
    for advantage in (-3, -0.2, 0, 0.5, 4):
        for ratio in (0.1, 0.79, 0.9, 1, 1.1, 1.21, 3):
            expected = min(ratio * advantage, min(1.2, max(0.8, ratio)) * advantage)
            assert cls.clipped_surrogate(ratio, advantage) == pytest.approx(expected)
            scalar = Scalar(ratio)
            objective = cls.clipped_surrogate(scalar, advantage)
            objective.backward()
            unclipped = ratio * advantage
            clipped = min(1.2, max(0.8, ratio)) * advantage
            assert scalar.grad == pytest.approx(advantage if unclipped <= clipped else 0)
    agent = cls()
    agent.initialize(13)
    for inventory in (-10, 0, 10):
        for bid_live, ask_live in itertools.product((0, 1), repeat=2):
            features = {
                "price": 50,
                "inventory": inventory,
                "lambda_buy": 0.3,
                "lambda_sell": 0.4,
                "twap_present": 1,
                "bid_live": bid_live,
                "ask_live": ask_live,
            }
            probabilities = [p.value for p in agent.probabilities(features)]
            assert sum(probabilities) == pytest.approx(1)
            if inventory == 10 or bid_live:
                assert probabilities[1] == 0
            if inventory == -10 or ask_live:
                assert probabilities[2] == 0
            if not bid_live:
                assert probabilities[3] == 0
            if not ask_live:
                assert probabilities[4] == 0


def test_self_imitation_uses_only_positive_return_advantage():
    cls = case("ppo_twap_aware", "PPOTWAPAware")
    rng = random.Random(83)
    for _ in range(50):
        old_value, target = rng.uniform(-2, 2), rng.uniform(-2, 2)
        logp, value = Scalar(-0.7), Scalar(old_value)
        loss = cls.self_imitation_loss(logp, value, target)
        loss.backward()
        positive = max(target - old_value, 0)
        assert loss.value == pytest.approx(0.7 * positive + 0.5 * positive**2)
        assert logp.grad == pytest.approx(-positive)
        assert value.grad == pytest.approx(-positive)


def test_graph_trading_signal_matches_independent_log_regression():
    cls = case("graph_matching", "GraphMatchingPairs")
    rng = np.random.default_rng(53)
    model = cls()
    xs = np.cumsum(rng.normal(0, 0.01, 504)) + 4.5
    ys = 0.1 + 0.8 * xs + rng.normal(0, 0.01, 504)
    model.history = {
        "PAIR_A": list(xs),
        "PAIR_B": list(ys),
        "PAIR_C": list(xs + 0.2),
        "PAIR_D": list(ys - 0.1),
    }
    model.pairs = (("PAIR_A", "PAIR_B"), ("PAIR_C", "PAIR_D"))
    model.month = (2025, 1)
    beta0, beta1 = np.linalg.lstsq(np.column_stack((np.ones(504), xs)), ys, rcond=None)[0]
    residual = ys - beta0 - beta1 * xs
    scale = float(np.sqrt(residual @ residual / 502))
    account = AccountSnapshot(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        cash=Decimal(100000),
        equity=Decimal(100000),
        positions=(),
        open_order_ids=(),
    )
    for shift in (-4, 0, 4):
        local = cls()
        local.history = {k: list(v) for k, v in model.history.items()}
        local.pairs = model.pairs
        local.month = model.month
        logs = {
            "PAIR_A": float(xs[-1]),
            "PAIR_B": float(beta0 + beta1 * xs[-1] + shift * scale),
            "PAIR_C": float(xs[-1] + 0.2),
            "PAIR_D": float(beta0 + beta1 * xs[-1] + shift * scale - 0.1),
        }
        actual = ()
        for symbol in sorted(logs):
            event = MarketEvent(
                event_id=symbol,
                symbol=symbol,
                event_time=account.timestamp,
                available_time=account.timestamp,
                values={"close": Decimal(str(math.exp(logs[symbol])))},
            )
            actual = local.decide(event, account)
        quantities = {action.symbol: float(action.quantity) for action in actual}
        direction = 1 if shift >= 2 else -1 if shift <= -2 else 0
        assert quantities["PAIR_A"] == pytest.approx(
            direction * beta1 * 1000 / math.exp(logs["PAIR_A"])
        )
        assert quantities["PAIR_B"] == pytest.approx(-direction * 1000 / math.exp(logs["PAIR_B"]))


def test_confidence_and_bet_size_mapping_uses_real_probability_boundaries(monkeypatch):
    cls = case("grouped_cnn_lstm", "GroupedCNNLSTM")
    account = AccountSnapshot(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        cash=Decimal(1000),
        equity=Decimal(1000),
        positions=(),
        open_order_ids=(),
    )
    event = MarketEvent(
        event_id="probe",
        symbol="BTCUSDT",
        event_time=account.timestamp,
        available_time=account.timestamp,
        values={
            "open": Decimal(100),
            "high": Decimal(101),
            "low": Decimal(99),
            "close": Decimal(100),
            "volume": Decimal(10),
        },
    )
    for theta in (0.25, 0.5, 1):
        for probability in (0.05, 0.25, 0.3333, 0.5, 0.6667, 0.8, 0.99):
            agent = cls()
            agent.history = [[100, 101, 99, 100, 10]] * 24
            agent.theta = theta
            monkeypatch.setattr(agent, "channels", lambda rows: None)
            monkeypatch.setattr(
                agent, "forward", lambda inputs, probability=probability: Scalar(probability)
            )
            action = agent.decide(event, account)[0]
            direction = (
                1
                if probability >= 1 / (1 + theta)
                else -1
                if probability <= theta / (1 + theta)
                else 0
            )
            assert float(action.quantity) == pytest.approx(direction * abs(probability - 0.5) * 10)
