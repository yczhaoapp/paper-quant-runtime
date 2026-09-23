"""Offline checks for the portable inputs, code identity, and run trace."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from paperquant.compiler import fingerprint
from paperquant.market_semantics import validate_market_event
from paperquant.models import RunBundle


def verify_bundle(bundle: RunBundle) -> None:
    report = bundle.report
    plan = report.plan
    checks = {
        "run identity": report.run_id == plan.run_id,
        "strategy declaration": fingerprint(bundle.strategy_declaration) == plan.declaration_sha256,
        "strategy ID": bundle.strategy_declaration.strategy_id == plan.strategy_id,
        "dataset declaration": fingerprint(bundle.dataset_declaration) == plan.dataset_sha256,
        "dataset ID": bundle.dataset_declaration.dataset_id == plan.dataset_id,
        "source event count": len(bundle.source_events) == bundle.dataset_declaration.event_count,
        "source dataset hash": (
            fingerprint(bundle.source_events) == bundle.dataset_declaration.content_sha256
        ),
        "source report hash": fingerprint(bundle.source_events) == report.source_events_sha256,
        "effective plan hash": fingerprint(bundle.effective_events) == plan.effective_events_sha256,
        "effective report hash": (
            fingerprint(bundle.effective_events) == report.effective_events_sha256
        ),
        "training request": (
            (fingerprint(bundle.training_request) if bundle.training_request is not None else None)
            == plan.training_sha256
        ),
        "engine capability": fingerprint(plan.engine_capability) == plan.engine_sha256,
        "engine profile": fingerprint(plan.engine_profile) == plan.engine_profile_sha256,
        "engine identity": plan.engine_profile.engine_id == plan.engine_id,
        "policy": fingerprint(plan.policy) == plan.policy_sha256,
        "final equity": bool(report.accounts) and report.final_equity == report.accounts[-1].equity,
    }
    try:
        for event in bundle.source_events:
            validate_market_event(bundle.dataset_declaration, event)
        checks["source market semantics"] = True
    except ValueError:
        checks["source market semantics"] = False
    maximum_age = bundle.strategy_declaration.data.max_staleness_seconds
    if maximum_age is not None:
        checks["effective market staleness"] = all(
            (event.available_time - event.event_time).total_seconds() <= maximum_age
            for event in bundle.effective_events
        )
    if (bundle.package_source is None) != (plan.package_source_sha256 is None):
        checks["package source presence"] = False
    elif bundle.package_source is not None:
        source_hash = hashlib.sha256(bundle.package_source.encode()).hexdigest()
        checks["package source hash"] = source_hash == plan.package_source_sha256
        checks["package class"] = bool(plan.package_class_name)
        if report.worker_receipt is not None:
            checks["worker source hash"] = report.worker_receipt.source_sha256 == source_hash
        if report.training_worker_receipt is not None:
            checks["training worker source hash"] = (
                report.training_worker_receipt.source_sha256 == source_hash
            )
    if report.artifact is None:
        checks["model absent"] = bundle.model_base64 is None and bundle.training_request is None
    else:
        try:
            payload = base64.b64decode(bundle.model_base64 or "", validate=True)
        except ValueError:
            payload = b""
        checks["model payload"] = (
            bool(payload)
            and hashlib.sha256(payload).hexdigest() == report.artifact.content_sha256
            and report.artifact.training_sha256 == plan.training_sha256
            and report.artifact.strategy_id == plan.strategy_id
        )
    for previous, current in zip(plan.conversions, plan.conversions[1:], strict=False):
        checks[f"conversion chain {current.transformation}"] = (
            previous.output_sha256 == current.input_sha256
        )
    if plan.conversions:
        checks["conversion source"] = (
            plan.conversions[0].input_sha256 == report.source_events_sha256
        )
        checks["conversion result"] = (
            plan.conversions[-1].output_sha256 == report.effective_events_sha256
        )
    else:
        checks["no hidden conversion"] = (
            report.source_events_sha256 == report.effective_events_sha256
        )
    accepted = {(order.order_id, order.symbol): order.timestamp for order in report.orders
                if order.status == "accepted"}
    filled = {(order.order_id, order.symbol) for order in report.orders
              if order.status == "filled"}
    checks["order chronology"] = all(
        (fill.order_id, fill.symbol) in accepted
        and (fill.order_id, fill.symbol) in filled
        and accepted[(fill.order_id, fill.symbol)] <= fill.timestamp
        for fill in report.fills
    )
    checks["trace length"] = (
        len(report.decisions) == len(bundle.effective_events)
        and len(report.accounts) == len(bundle.effective_events)
    )
    for event, decision, account in zip(
        bundle.effective_events, report.decisions, report.accounts, strict=False
    ):
        if (decision.event_id != event.event_id or decision.event_time != event.available_time
            or account.timestamp != event.available_time):
            checks["trace alignment"] = False
            break
    if failed := [name for name, okay in checks.items() if not okay]:
        raise ValueError("Run bundle verification failed: " + ", ".join(failed))


def verify_bundle_file(path: Path) -> RunBundle:
    bundle = RunBundle.model_validate_json(path.read_bytes())
    verify_bundle(bundle)
    return bundle
