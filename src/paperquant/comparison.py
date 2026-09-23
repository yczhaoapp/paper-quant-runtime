from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field

from paperquant.compiler import fingerprint
from paperquant.models import (
    Model,
    NoOp,
    Prediction,
    RunReport,
    SubmitOrder,
    TargetPosition,
)


class ObservedAction(Model):
    event_id: str
    kind: Literal["none", "prediction", "target_position", "submit_order"]
    symbol: str | None = None
    quantity: Decimal | None = None
    side: Literal["buy", "sell"] | None = None
    value: Decimal | None = None
    client_order_id: str | None = None
    limit_price: Decimal | None = None


class ObservedFill(Model):
    event_time: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: Decimal
    price: Decimal
    fee: Decimal


class Observation(Model):
    scenario_id: str
    method_id: str
    source_events_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    effective_events_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_cash: Decimal
    final_equity: Decimal
    actions: tuple[ObservedAction, ...]
    fills: tuple[ObservedFill, ...]


class Comparison(Model):
    status: Literal["comparable", "incomparable"]
    reasons: tuple[str, ...]
    equity_delta: Decimal | None = None
    same_actions: bool | None = None
    same_fills: bool | None = None
    action_count_delta: int | None = None
    fill_count_delta: int | None = None


def from_report(
    report: RunReport,
    *,
    scenario_id: str,
    method_id: str,
    execution_profile_sha256: str | None = None,
    initial_cash: Decimal | None = None,
) -> Observation:
    profile = report.plan.engine_profile
    bound_profile_sha256 = fingerprint(profile.settings)
    if execution_profile_sha256 is not None and execution_profile_sha256 != bound_profile_sha256:
        raise ValueError("comparison profile differs from bound engine settings")
    bound_cash = profile.settings.get("initial_cash")
    if bound_cash is not None:
        if initial_cash is not None and initial_cash != Decimal(bound_cash):
            raise ValueError("comparison cash differs from bound engine settings")
        initial_cash = Decimal(bound_cash)
    if initial_cash is None:
        raise ValueError("external engine must declare or supply initial cash")
    actions = []
    for decision in report.decisions:
        for action in decision.actions:
            if isinstance(action, NoOp):
                actions.append(ObservedAction(event_id=decision.event_id, kind="none"))
            elif isinstance(action, Prediction):
                actions.append(
                    ObservedAction(
                        event_id=decision.event_id,
                        kind="prediction",
                        symbol=action.symbol,
                        value=action.value,
                    )
                )
            elif isinstance(action, TargetPosition):
                actions.append(
                    ObservedAction(
                        event_id=decision.event_id,
                        kind="target_position",
                        symbol=action.symbol,
                        quantity=action.quantity,
                    )
                )
            elif isinstance(action, SubmitOrder):
                actions.append(
                    ObservedAction(
                        event_id=decision.event_id,
                        kind="submit_order",
                        symbol=action.symbol,
                        quantity=action.quantity,
                        side=action.side,
                        client_order_id=action.client_order_id,
                        limit_price=action.limit_price,
                    )
                )
    return Observation(
        scenario_id=scenario_id,
        method_id=method_id,
        source_events_sha256=report.source_events_sha256,
        effective_events_sha256=report.effective_events_sha256,
        execution_profile_sha256=bound_profile_sha256,
        initial_cash=initial_cash,
        final_equity=report.final_equity,
        actions=tuple(actions),
        fills=tuple(
            ObservedFill(
                event_time=fill.timestamp.isoformat(),
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                price=fill.price,
                fee=fill.fee,
            )
            for fill in report.fills
        ),
    )


def compare(left: Observation, right: Observation) -> Comparison:
    identity_fields = (
        "scenario_id",
        "method_id",
        "source_events_sha256",
        "effective_events_sha256",
        "execution_profile_sha256",
        "initial_cash",
    )
    reasons = tuple(
        field for field in identity_fields if getattr(left, field) != getattr(right, field)
    )
    if reasons:
        return Comparison(status="incomparable", reasons=reasons)
    return Comparison(
        status="comparable",
        reasons=(),
        equity_delta=right.final_equity - left.final_equity,
        same_actions=left.actions == right.actions,
        same_fills=left.fills == right.fills,
        action_count_delta=len(right.actions) - len(left.actions),
        fill_count_delta=len(right.fills) - len(left.fills),
    )
