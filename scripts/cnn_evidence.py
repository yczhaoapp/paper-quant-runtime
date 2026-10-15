"""Audit executed CNN decisions without changing the strategy or runtime contract."""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal


def evaluation_window(events, history, plan):
    months = plan["evaluation"]
    starts = [datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC) for month in months]
    if not starts or months != sorted(set(months)):
        raise ValueError("Evaluation months must be ordered and distinct")
    last = starts[-1]
    end = datetime(last.year + (last.month == 12), last.month % 12 + 1, 1, tzinfo=UTC)
    evaluated = events[history - 1 :]
    if history < 1 or not evaluated:
        raise ValueError("Evaluation has no post-warmup predictions")
    if any(event.bar_open_time is None for event in evaluated):
        raise ValueError("CNN evaluation requires explicit bar opening timestamps")
    actual = [event.bar_open_time.astimezone(UTC) for event in evaluated]
    all_starts = [event.bar_open_time for event in events]
    if any(stamp is None for stamp in all_starts) or any(
        (right - left).total_seconds() != 14400
        for left, right in zip(all_starts, all_starts[1:], strict=False)
    ):
        raise ValueError("CNN warmup and evaluation dates are not contiguous")
    if (
        sorted({stamp.strftime("%Y-%m") for stamp in actual}) != months
        or actual[0] != starts[0]
        or any(not starts[0] <= stamp < end for stamp in actual)
        or any(
            (right - left).total_seconds() != 14400
            for left, right in zip(actual, actual[1:], strict=False)
        )
        or (end - actual[-1]).total_seconds() != 14400
    ):
        raise ValueError("Executed prediction dates differ from the complete evaluation plan")
    return {
        "planned_months": months,
        "actual_months": sorted({stamp.strftime("%Y-%m") for stamp in actual}),
        "timestamp_basis": "UTC bar_open_time; predictions available at completed bar close",
        "first_bar_open": actual[0].isoformat(),
        "last_bar_open": actual[-1].isoformat(),
        "first_prediction_available": evaluated[0].available_time.isoformat(),
        "last_prediction_available": evaluated[-1].available_time.isoformat(),
        "source_events": len(events),
        "warmup_events": history - 1,
        "first_warmup_bar_open": events[0].bar_open_time.astimezone(UTC).isoformat(),
        "warmup_contiguous": True,
        "prediction_events": len(evaluated),
        "labelled_predictions": len(evaluated) - 1,
        "last_prediction_label": "unavailable; final prediction retained, excluded from scores",
    }


def audit_execution(report, events, predictions, *, history, theta):
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("CNN confidence parameter must be finite and positive")
    if (
        report.status != "succeeded"
        or report.artifact is None
        or len(report.decisions) != len(events)
        or len(report.accounts) != len(events)
        or len(predictions) != len(events) - history + 1
        or not predictions
    ):
        raise ValueError("CNN execution audit lacks the complete trained-model lifecycle")
    lower, upper = theta / (1 + theta), 1 / (1 + theta)
    directions = Counter()
    for index, (event, decision, account) in enumerate(
        zip(events, report.decisions, report.accounts, strict=True)
    ):
        if event.event_id != decision.event_id or len(decision.actions) != 1:
            raise ValueError("CNN decision sequence differs from executed events")
        action = decision.actions[0]
        if index < history - 1:
            if action.kind != "none":
                raise ValueError("CNN warmup produced an unexpected action")
            continue
        record = predictions[index - history + 1]
        probability = record["probability"]
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("CNN probability is invalid")
        direction = 1 if probability >= upper else -1 if probability <= lower else 0
        quantity = (
            direction
            * abs(probability - 0.5)
            * float(account.equity)
            / float(event.values["close"])
        )
        expected = Decimal(f"{quantity:.8f}")
        if (
            record["event_id"] != event.event_id
            or record["signal"] != direction
            or action.kind != "target_position"
            or action.symbol != event.symbol
            or action.quantity != expected
            or f"p={probability:.8f}" not in action.reason
        ):
            raise ValueError("CNN probability, confidence rule and actual target differ")
        directions[direction] += 1
    statuses = Counter(order.status for order in report.orders)
    reasons = Counter(order.reason for order in report.orders if order.status != "accepted")
    zero = not report.fills
    neutral = directions[0] == len(predictions)
    flat = all(
        position.quantity == 0 for account in report.accounts for position in account.positions
    )
    zero_reason = "not_zero_trade" if not zero else "requires_separate_execution_explanation"
    if zero and neutral:
        lifecycles = {}
        for order in report.orders:
            lifecycles.setdefault(order.order_id, []).append(order)

        def valid_zero_order(orders):
            if [order.status for order in orders] != ["accepted", "expired"]:
                return False
            return orders[-1].reason == "target_already_satisfied" or (
                orders[-1].reason == "no_next_event"
                and orders[0].timestamp == events[-1].available_time
                and orders[-1].timestamp == events[-1].available_time
            )

        if (
            not flat
            or len(lifecycles) != len(predictions)
            or any(not valid_zero_order(orders) for orders in lifecycles.values())
            or report.final_equity != report.accounts[0].equity
            or any(account.cash != report.accounts[0].cash for account in report.accounts)
            or any(
                account.equity != report.accounts[0].equity or account.cash != account.equity
                for account in report.accounts
            )
        ):
            raise ValueError("Zero targets are not supported by flat accounts and order lifecycles")
        zero_reason = "confidence_threshold_not_crossed; zero targets already satisfied"
    return {
        "status": "passed",
        "training_and_reload": "executed; saved artifact and bundle checked by enclosing gate",
        "training_sha256": report.plan.training_sha256,
        "model_sha256": report.artifact.content_sha256,
        "package_source_sha256": report.plan.package_source_sha256,
        "events": len(events),
        "warmup_events": history - 1,
        "predictions": len(predictions),
        "confidence": {"theta": theta, "lower": lower, "upper": upper},
        "probability_min": min(record["probability"] for record in predictions),
        "probability_max": max(record["probability"] for record in predictions),
        "targets": {"long": directions[1], "short": directions[-1], "zero": directions[0]},
        "order_state_records": len(report.orders),
        "distinct_order_ids": len({order.order_id for order in report.orders}),
        "order_status_counts": dict(statuses),
        "order_terminal_reason_counts": dict(reasons),
        "fills": len(report.fills),
        "zero_trade_reason": zero_reason,
        "accounts_remained_flat": flat,
        "prediction_audit": "independently recomputed saved-model probabilities "
        "match actual targets",
        "financial_performance_claim": False,
    }
