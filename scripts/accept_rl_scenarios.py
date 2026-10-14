"""Retain all predeclared RL paths and separately labelled controlled fill probes."""

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import fingerprint
from paperquant.evidence import verify_bundle_file
from paperquant.models import DatasetDeclaration, RunPolicy, TrainingRequest
from scripts.accept_six_papers import invoke, relative
from scripts.build_six_paper_inputs import collect_episode, package, quote_events, save
from scripts.gate_support import check_source_identity, new_attempt, source_identity, write_json

ROOT = Path(__file__).resolve().parents[1]


def checked_ledger(report, events):
    cash = Decimal(100000)
    quantity = average = realized = Decimal(0)
    fees = Decimal(0)
    fills = {}
    for fill in report.fills:
        fills.setdefault(fill.timestamp, []).append(fill)
    for event, account in zip(events, report.accounts, strict=True):
        for fill in fills.pop(account.timestamp, []):
            signed = fill.quantity if fill.side == "buy" else -fill.quantity
            new = quantity + signed
            if quantity == 0 or quantity * signed > 0:
                average = (abs(quantity) * average + fill.quantity * fill.price) / abs(new)
            else:
                realized += (
                    min(abs(quantity), fill.quantity)
                    * (fill.price - average)
                    * (1 if quantity > 0 else -1)
                )
                average = Decimal(0) if new == 0 else fill.price if quantity * new < 0 else average
            quantity = new
            cash -= signed * fill.price + fill.fee
            fees += fill.fee
        assert account.cash == cash
        if account.positions:
            position = account.positions[0]
            assert (position.quantity, position.average_price, position.realized_pnl) == (
                quantity,
                average,
                realized,
            )
            mark = (event.values["bid_price"] + event.values["ask_price"]) / 2
            assert account.equity == cash + quantity * mark
            assert position.unrealized_pnl == quantity * (mark - average)
    assert not fills
    return {
        "status": "passed",
        "snapshots": len(report.accounts),
        "fills": len(report.fills),
        "fees": str(fees),
    }


def gate(output, *, strict, require_clean):
    output.mkdir(parents=True, exist_ok=True)
    state = new_attempt()
    receipt = output / "rl-scenarios.json"
    write_json(receipt, state)
    attempt = output / "attempts" / state["attempt_id"]
    try:
        identity = source_identity(ROOT, require_clean=require_clean)
        state.update(identity)
        plan = json.loads((ROOT / "research/audit-refinement/rl-scenario-plan.json").read_text())
        scenarios = []
        for seed in plan["evaluation_seeds"]:
            scenarios.append(("sac_passive_quotes", f"natural-{seed}", quote_events(seed), None))
            for side in plan["ppo_meta_sides"]:
                scenarios.append(
                    (
                        "ppo_twap_aware",
                        f"rho{side}-seed{seed}",
                        quote_events(seed, hawkes=True, meta_side=side, steps=1972),
                        None,
                    )
                )
        controlled = tuple(
            event.model_copy(
                update={
                    "values": {
                        **event.values,
                        "buy_flow": Decimal(1),
                        "sell_flow": Decimal(1),
                        "uniform_buy": Decimal(0),
                        "uniform_sell": Decimal(0),
                    }
                }
            )
            for event in quote_events(81037, steps=200)
        )
        # Build a second trained model specifically for the labelled branch probe.
        cls = package("sac_passive_quotes")
        samples, collection = collect_episode(cls, controlled, seed=37)
        controlled_training = TrainingRequest(
            dataset_id="supplement.sac_passive_quotes", seed=37, samples=tuple(samples)
        )
        scenarios.append(
            (
                "sac_passive_quotes",
                "controlled-nonadverse-train-and-evaluate",
                controlled,
                controlled_training,
            )
        )
        results = []
        attempt.mkdir(parents=True)
        for name, scenario, events, training in scenarios:
            folder = attempt / name / scenario
            folder.mkdir(parents=True)
            dataset = DatasetDeclaration(
                contract_version="1.1",
                dataset_id="supplement." + name,
                granularity="tick",
                market_kind="quote_l1",
                fields=frozenset(events[0].values),
                symbols=frozenset({"MM"}),
                event_count=len(events),
                content_sha256=fingerprint(events),
            )
            save(folder / "events.json", events)
            save(folder / "dataset.json", dataset)
            if training is not None:
                save(folder / "training.json", training)
                save(folder / "collection.json", collection)
                if not any(
                    o["reason"] == "nonadverse_order_flow" and o["status"] == "filled"
                    for o in collection["orders"]
                ):
                    raise ValueError(
                        "Controlled training did not exercise its declared fill branch"
                    )
            training_file = (
                folder / "training.json"
                if training
                else ROOT / f"examples/six-paper/{name}/training.json"
            )
            row = {
                "strategy": "supplement." + name,
                "scenario": scenario,
                "scope": "controlled branch validation"
                if training
                else "predeclared natural simulation path",
                "source_events_sha256": fingerprint(events),
                "modes": {},
            }
            reports = {}
            for mode in ("host", "strict") if strict else ("host",):
                target = folder / mode
                target.mkdir()
                save(
                    target / "policy.json",
                    RunPolicy(sandbox="strict" if mode == "strict" else "development"),
                )
                args = [
                    "paper-run",
                    "--recipe",
                    f"research/six-paper/cases/{name}/recipe.json",
                    "--dataset",
                    relative(folder / "dataset.json"),
                    "--events",
                    relative(folder / "events.json"),
                    "--training",
                    relative(training_file),
                    "--policy",
                    relative(target / "policy.json"),
                    "--engine",
                    "passive",
                    "--fee-rate",
                    "0.0001" if name == "ppo_twap_aware" else "0",
                    "--output",
                    relative(target / "run"),
                ]
                if invoke(args, target):
                    raise ValueError("RL scenario failed: " + scenario + "/" + mode)
                report = verify_bundle_file(target / "run/bundle.json").report
                reports[mode] = report
                nonadverse = sum(
                    o.status == "filled" and o.reason == "nonadverse_order_flow"
                    for o in report.orders
                )
                if training is not None and nonadverse == 0:
                    raise ValueError("Controlled trained-model execution missed its fill branch")
                row["modes"][mode] = {
                    "status": report.status,
                    "fills": len(report.fills),
                    "nonadverse_fills": nonadverse,
                    "final_equity": str(report.final_equity),
                    "training_sha256": report.plan.training_sha256,
                    "model_sha256": report.artifact.content_sha256,
                    "ledger": checked_ledger(report, events),
                    "cold_replay": "passed",
                }
            if strict:
                left, right = reports["host"], reports["strict"]
                if (left.decisions, left.orders, left.fills, left.accounts) != (
                    right.decisions,
                    right.orders,
                    right.fills,
                    right.accounts,
                ):
                    raise ValueError("RL scenario host/strict trace mismatch")
            row["host_strict_traces"] = "passed" if strict else "not_run"
            results.append(row)
            state.update(results=results, current_scenario=scenario)
            write_json(receipt, state)
            print(scenario, row["modes"]["host"]["fills"], flush=True)
        check_source_identity(ROOT, identity, require_clean=require_clean)
        state.update(
            status="passed",
            scenario_count=len(results),
            strict=strict,
            method_validation="see six-paper gate on the same source identity",
            original_experiments_reproduced=False,
            profitability_claim=False,
        )
    except Exception as exc:
        state.update(status="failed", cause=type(exc).__name__, reason=str(exc))
    state["completed_at"] = datetime.now(UTC).isoformat()
    write_json(receipt, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    result = gate(args.output.resolve(), strict=args.strict, require_clean=args.require_clean)
    print(json.dumps({"status": result["status"], "reason": result.get("reason")}))
    raise SystemExit(0 if result["status"] == "passed" else 2)
