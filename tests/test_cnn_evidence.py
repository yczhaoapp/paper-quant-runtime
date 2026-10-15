"""Reject mismatched dates, targets and explanations in supplemental evidence."""

from __future__ import annotations

import copy
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from paperquant.models import AccountSnapshot, MarketEvent, NoOp, OrderEvent, TargetPosition
from scripts.cnn_evidence import audit_execution, evaluation_window


def quarter_events():
    start = datetime(2025, 1, 1, tzinfo=UTC) - timedelta(hours=23 * 4)
    return tuple(
        MarketEvent(
            event_id=str(index),
            symbol="BTCUSDT",
            bar_open_time=start + timedelta(hours=4 * index),
            event_time=start + timedelta(hours=4 * (index + 1)),
            available_time=start + timedelta(hours=4 * (index + 1)),
            values={"close": Decimal(100)},
        )
        for index in range(563)
    )


def test_evaluation_scope_uses_actual_bar_dates_and_retains_final_unlabelled_prediction():
    result = evaluation_window(
        quarter_events(), 24, {"evaluation": ["2025-01", "2025-02", "2025-03"]}
    )
    assert result["prediction_events"] == 540
    assert result["labelled_predictions"] == 539
    assert result["warmup_events"] == 23
    assert result["last_bar_open"] == "2025-03-31T20:00:00+00:00"
    assert result["last_prediction_available"] == "2025-04-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "mutation", ["old_quarter", "shift", "gap", "no_bar_time", "shortened", "stale_warmup"]
)
def test_evaluation_scope_rejects_report_plan_and_actual_event_mismatch(mutation):
    events = list(quarter_events())
    plan = {"evaluation": ["2025-01", "2025-02", "2025-03"]}
    if mutation == "old_quarter":
        plan["evaluation"] = ["2024-10", "2024-11", "2024-12"]
    elif mutation == "shift":
        events = [
            event.model_copy(update={"bar_open_time": event.bar_open_time + timedelta(hours=4)})
            for event in events
        ]
    elif mutation == "gap":
        del events[200]
    elif mutation == "no_bar_time":
        events[100] = events[100].model_copy(update={"bar_open_time": None})
    elif mutation == "stale_warmup":
        events[0] = events[0].model_copy(
            update={"bar_open_time": datetime(2024, 9, 27, tzinfo=UTC)}
        )
    else:
        del events[-1]
    with pytest.raises(ValueError, match="dates|timestamps"):
        evaluation_window(events, 24, plan)


def execution(probabilities, theta=0.5, seed=7):
    rng = random.Random(seed)
    history = 3
    events = quarter_events()[: len(probabilities) + history - 1]
    decisions, accounts, predictions, orders = [], [], [], []
    lower, upper = theta / (1 + theta), 1 / (1 + theta)
    neutral = all(lower < probability < upper for probability in probabilities)
    for index, event in enumerate(events):
        equity = Decimal(10000) if neutral else Decimal(str(rng.uniform(100, 10000)))
        account = AccountSnapshot(
            timestamp=event.event_time,
            cash=Decimal(10000),
            equity=equity,
            positions=(),
            open_order_ids=(),
        )
        accounts.append(account)
        if index < history - 1:
            action = NoOp(reason="warmup")
        else:
            probability = probabilities[index - history + 1]
            direction = (probability >= upper) - (probability <= lower)
            target = direction * abs(probability - 0.5) * float(equity) / 100
            action = TargetPosition(
                symbol=event.symbol,
                quantity=Decimal(f"{target:.8f}"),
                reason=f"p={probability:.8f}",
            )
            predictions.append(
                {"event_id": event.event_id, "probability": probability, "signal": direction}
            )
            for status, reason in (
                ("accepted", action.reason),
                ("expired", "target_already_satisfied"),
            ):
                orders.append(
                    OrderEvent(
                        order_id=str(index),
                        symbol=event.symbol,
                        timestamp=event.event_time,
                        status=status,
                        reason=reason,
                    )
                )
        decisions.append(SimpleNamespace(event_id=event.event_id, actions=(action,)))
    report = SimpleNamespace(
        status="succeeded",
        artifact=SimpleNamespace(content_sha256="a" * 64),
        plan=SimpleNamespace(training_sha256="b" * 64, package_source_sha256="c" * 64),
        decisions=decisions,
        accounts=accounts,
        orders=orders,
        fills=(),
        final_equity=accounts[0].equity,
    )
    return report, events, predictions


@pytest.mark.parametrize("theta", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", [7, 31, 79])
def test_zero_fill_explanation_checks_many_probabilities_and_unique_order_lifecycles(theta, seed):
    rng = random.Random(seed)
    lower, upper = theta / (1 + theta), 1 / (1 + theta)
    values = [rng.uniform(lower + 1e-6, upper - 1e-6) for _ in range(64)]
    report, events, predictions = execution(values, theta, seed)
    result = audit_execution(report, events, predictions, history=3, theta=theta)
    assert result["targets"] == {"long": 0, "short": 0, "zero": 64}
    assert result["distinct_order_ids"] == 64
    assert result["order_state_records"] == 128
    assert result["zero_trade_reason"].startswith("confidence_threshold_not_crossed")


@pytest.mark.parametrize(
    "mutation",
    [
        "quantity",
        "probability",
        "signal",
        "rejected",
        "expiry_reason",
        "missing_order",
        "duplicate_order",
        "decision_id",
        "warmup",
    ],
)
def test_zero_trade_explanation_rejects_inconsistent_actual_evidence(mutation):
    report, events, predictions = execution([0.51, 0.52, 0.55])
    if mutation == "quantity":
        action = report.decisions[2].actions[0]
        report.decisions[2].actions = (action.model_copy(update={"quantity": Decimal(1)}),)
    elif mutation == "probability":
        predictions[0]["probability"] = 0.9
    elif mutation == "signal":
        predictions[0]["signal"] = 1
    elif mutation == "rejected":
        report.orders[-1] = report.orders[-1].model_copy(update={"status": "rejected"})
    elif mutation == "expiry_reason":
        report.orders[-1] = report.orders[-1].model_copy(update={"reason": "end_of_data"})
    elif mutation == "missing_order":
        del report.orders[-1]
    elif mutation == "duplicate_order":
        report.orders[-1] = report.orders[-1].model_copy(
            update={"order_id": report.orders[0].order_id}
        )
    elif mutation == "decision_id":
        report.decisions[2].event_id = "different-event"
    else:
        report.decisions[0].actions = (report.decisions[2].actions[0],)
    with pytest.raises(ValueError):
        audit_execution(report, events, predictions, history=3, theta=0.5)


def test_threshold_boundaries_are_checked_and_nonzero_targets_do_not_get_zero_signal_explanation():
    report, events, predictions = execution([1 / 3, 2 / 3, 0.5])
    result = audit_execution(report, events, predictions, history=3, theta=0.5)
    assert result["targets"] == {"long": 1, "short": 1, "zero": 1}
    assert result["zero_trade_reason"] == "requires_separate_execution_explanation"


def test_final_no_next_event_is_disclosed_and_rejected_at_an_earlier_event():
    report, events, predictions = execution([0.51, 0.52, 0.55])
    report.orders[-1] = report.orders[-1].model_copy(update={"reason": "no_next_event"})
    result = audit_execution(report, events, predictions, history=3, theta=0.5)
    assert result["order_terminal_reason_counts"] == {
        "target_already_satisfied": 2,
        "no_next_event": 1,
    }
    report.orders[1] = report.orders[1].model_copy(update={"reason": "no_next_event"})
    with pytest.raises(ValueError):
        audit_execution(report, events, predictions, history=3, theta=0.5)


@pytest.mark.parametrize(
    "mutation", ["training", "model", "source", "oracle", "control_fills", "threshold"]
)
def test_closure_rejects_conflated_or_missing_method_and_control_evidence(mutation):
    from scripts.accept_cnn_refinement import close_evidence

    natural = {
        "runtime": "succeeded",
        "cold_replay": "passed",
        "training": "executed_and_reloaded",
        "training_sha256": "natural-training",
        "report_sha256": "report",
        "bundle_sha256": "bundle",
        "adaptation_quality": {
            "execution_diagnostic": {
                "package_source_sha256": "source",
                "model_sha256": "natural-model",
                "confidence": {"theta": 0.5},
            },
            "evidence_status": {"training_quality_threshold": "not_met"},
            "cnn_refinement": {"evaluation_window": {}},
            "method_fitting": {},
            "test_metrics": {},
        },
    }
    control = {
        "runtime": "succeeded",
        "cold_replay": "passed",
        "fills": 1,
        "nonzero_targets": 1,
        "package_source_sha256": "source",
        "training_sha256": "control-training",
        "model_sha256": "control-model",
        "execution_diagnostic": {"confidence": {"theta": 0.5}},
    }
    case = {
        "strategy_id": "supplement.grouped_cnn_lstm",
        "source_mapping": "passed",
        "local_method_oracles": "passed",
        "modes": {"host": natural},
    }
    six = {"cases": [case], "method_oracle_junit_sha256": "junit"}
    controlled = {
        "modes": {"host": control},
        "model_relationship": "separately trained",
        "host_strict_trace_comparison": "not_run",
    }
    assert (
        close_evidence(six, controlled)["modes"]["host"]["status"]["training_quality_threshold"]
        == "not_met"
    )
    controlled = copy.deepcopy(controlled)
    if mutation == "training":
        controlled["modes"]["host"]["training_sha256"] = natural["training_sha256"]
    elif mutation == "model":
        controlled["modes"]["host"]["model_sha256"] = "natural-model"
    elif mutation == "source":
        controlled["modes"]["host"]["package_source_sha256"] = "other-source"
    elif mutation == "oracle":
        six["method_oracle_junit_sha256"] = None
    elif mutation == "threshold":
        controlled["modes"]["host"]["execution_diagnostic"]["confidence"]["theta"] = 0.25
    else:
        controlled["modes"]["host"]["fills"] = 0
    with pytest.raises(ValueError):
        close_evidence(six, controlled)
