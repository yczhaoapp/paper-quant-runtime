"""Build declared public and controlled fixtures without inspecting expected results."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from paperquant.compiler import canonical_document, compile_run, fingerprint
from paperquant.models import (
    DatasetDeclaration,
    MarketEvent,
    RunPolicy,
    TrainingRequest,
    TrainingSample,
)
from paperquant.package import write_manifest
from paperquant.passive_engine import PassiveQuoteEngine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/six-paper"
CLASSES = {
    "graph_matching": "GraphMatchingPairs",
    "ou_percentile": "OUPercentilePairs",
    "grouped_cnn_lstm": "GroupedCNNLSTM",
    "gmm_knn": "GMMConditionalKNN",
    "sac_passive_quotes": "SACPassiveQuotes",
    "ppo_twap_aware": "PPOTWAPAware",
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical_document(value), indent=2, sort_keys=True) + "\n")


def package(name):
    root = ROOT / "strategies" / ("supplement." + name)
    spec = importlib.util.spec_from_file_location(name, root / "strategy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, CLASSES[name])
    write_manifest(root, cls.declaration, CLASSES[name])
    return cls


def publish(name, events, *, dataset=None, training=None, provenance, policy=None):
    path = OUTPUT / name
    events = tuple(events)
    settings = {
        "contract_version": "1.1",
        "granularity": "tick" if events[0].bar_open_time is None else "day",
        **(dataset or {}),
    }
    dataset = DatasetDeclaration(
        dataset_id="supplement." + name,
        fields=frozenset(events[0].values),
        symbols=frozenset(e.symbol for e in events),
        event_count=len(events),
        content_sha256=fingerprint(events),
        **settings,
    )
    save(path / "events.json", [e.model_dump(mode="json") for e in events])
    save(path / "dataset.json", dataset)
    if training is not None:
        save(path / "training.json", training.model_dump(mode="json"))
    save(
        path / "policy.json",
        (policy or RunPolicy(financing_mode="unbounded_margin")).model_dump(mode="json"),
    )
    save(
        path / "lineage.json",
        {
            **provenance,
            "events_sha256": fingerprint(events),
            "training_sha256": fingerprint(training) if training is not None else None,
            "event_count": len(events),
            "first_event": events[0].event_time.isoformat(),
            "last_event": events[-1].event_time.isoformat(),
        },
    )
    return dataset


def daily_events(prices, symbols, *, start=datetime(2020, 1, 1, tzinfo=UTC)):
    result = []
    day = start
    previous = prices[0]
    for i, row in enumerate(prices):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        for symbol, opening, closing in zip(symbols, previous, row, strict=False):
            result.append(
                MarketEvent(
                    event_id=f"day:{i:04}:{symbol}",
                    symbol=symbol,
                    bar_open_time=day.replace(hour=9),
                    event_time=day.replace(hour=16),
                    available_time=day.replace(hour=16),
                    values={"open": Decimal(str(opening)), "close": Decimal(str(closing))},
                )
            )
        previous, day = row, day + timedelta(days=1)
    return result


def graph_inputs():
    rng = random.Random(240307998)
    rows = []
    common, residuals = math.log(100), [0.0] * 4
    for _ in range(650):
        common += rng.gauss(0.0001, 0.005)
        residuals = [0.65 * r + rng.gauss(0, 0.012) for r in residuals]
        rows.append(
            [
                math.exp(common + offset + noise)
                for offset, noise in zip((0.0, 0.1, -0.1, 0.2), residuals, strict=False)
            ]
        )
    publish(
        "graph_matching",
        daily_events(rows, ("PAIR_A", "PAIR_B", "PAIR_C", "PAIR_D")),
        dataset={"session_open": "09:00:00", "session_close": "16:00:00"},
        provenance={
            "type": "controlled_simulation",
            "seed": 240307998,
            "model": "common log random walk plus AR1 idiosyncratic spreads",
            "calibration_sessions": 504,
            "exclusions": "not S&P500 universe or empirical 2017-2023 experiment",
        },
    )


def ou_inputs():
    rng = random.Random(241212458)
    rows, spreads = [], []
    base, spread = 100.0, 0.0
    for _ in range(800):
        base += rng.gauss(0, 0.15)
        spread = 0.90 * spread + rng.gauss(0, 0.35)
        spreads.append(spread)
        rows.append([base + spread, base])
    training = TrainingRequest(
        dataset_id="supplement.ou_percentile",
        seed=17,
        samples=tuple(TrainingSample(features={"spread": Decimal(str(v))}) for v in spreads[:504]),
    )
    publish(
        "ou_percentile",
        daily_events(rows[504:], ("OU_A", "OU_B"), start=datetime(2022, 1, 3, tzinfo=UTC)),
        dataset={"session_open": "09:00:00", "session_close": "16:00:00"},
        training=training,
        provenance={
            "type": "controlled_simulation",
            "seed": 241212458,
            "model": "Gaussian OU-compatible AR1 price spread",
            "training_sessions": 504,
            "evaluation_sessions": 296,
            "split": "first 504 only; no evaluation data in OU fit",
            "exclusions": "preselected pair instead of private WRDS universe and manual selection",
        },
    )


def raw_bars(interval, months):
    result = []
    for month in months:
        path = ROOT / "data/six-paper/public-btc" / f"BTCUSDT-{interval}-{month}.csv"
        with path.open() as handle:
            for row in csv.reader(handle):
                result.append((int(row[0]), [float(v) for v in row[1:6]]))
    return result


def bar_events(rows, seconds):
    result = []
    for stamp, values in rows:
        opened = datetime.fromtimestamp(stamp / 1000, UTC)
        closed = opened + timedelta(seconds=seconds)
        result.append(
            MarketEvent(
                event_id=f"btc:{stamp}",
                symbol="BTCUSDT",
                bar_open_time=opened,
                event_time=closed,
                available_time=closed,
                values={
                    key: Decimal(str(v))
                    for key, v in zip(
                        ("open", "high", "low", "close", "volume"), values, strict=False
                    )
                },
            )
        )
    return result


def cnn_inputs():
    train = raw_bars("4h", ["2024-01", "2024-02"])
    evaluation = raw_bars("4h", ["2024-03"])
    # Data-independent stride bounds the miniature neural training workload.
    indices = list(range(23, len(train) - 1, 14))
    samples = []
    for end in indices:
        window = train[end - 23 : end + 1]
        features = {
            f"v_{t}_{j}": Decimal(str(value))
            for t, (_, row) in enumerate(window)
            for j, value in enumerate(row)
        }
        label = Decimal(int(train[end + 1][1][3] > train[end][1][3]))
        samples.append(TrainingSample(features=features, target=label))
    request = TrainingRequest(
        dataset_id="supplement.grouped_cnn_lstm", seed=23, samples=tuple(samples)
    )
    publish(
        "grouped_cnn_lstm",
        bar_events(evaluation, 14400),
        dataset={"granularity": "minute", "bar_seconds": 14400, "market_kind": "bar"},
        training=request,
        provenance={
            "type": "public_market_data",
            "source": "data/six-paper/public-btc/SOURCE.json",
            "training": "2024-01 through 2024-02 UTC, 24-candle windows at stride 14",
            "evaluation": "2024-03 UTC",
            "availability": "completed bar exclusive end, archive close time plus 1ms",
            "exclusions": (
                "Binance sample replaces original 2017-2021 crypto exchange"
                " data; no original profitability claim"
            ),
        },
    )


def gmm_inputs():
    rows = raw_bars("1m", ["2024-01"])[: 7 * 24 * 60]
    split = 4 * 24 * 60
    train, evaluation = rows[:split], rows[split:]
    samples = []
    for start in range(0, len(train) - 59, 60):
        samples.append(
            TrainingSample(
                features={
                    f"close_{i}": Decimal(str(row[1][3]))
                    for i, row in enumerate(train[start : start + 60])
                }
            )
        )
    request = TrainingRequest(dataset_id="supplement.gmm_knn", seed=31, samples=tuple(samples))
    publish(
        "gmm_knn",
        bar_events(evaluation, 60),
        dataset={"granularity": "minute", "bar_seconds": 60, "market_kind": "bar"},
        training=request,
        provenance={
            "type": "public_market_data",
            "source": "data/six-paper/public-btc/SOURCE.json",
            "training": "2024-01-01 through 2024-01-04 UTC",
            "evaluation": "2024-01-05 through 2024-01-07 UTC",
            "split": "training-only labels, scaler, GMM, BIC and neighbor bank",
            "exclusions": (
                "single BTC market and seven-day window replace original "
                "three markets and six months"
            ),
        },
    )


def quote_events(seed, *, hawkes=False, meta_side=0, steps=1000, meta_start=100):
    rng = random.Random(seed)
    dt = 0.213 if hawkes else 0.001
    bid_intensity = ask_intensity = 0.275
    mid = 50.0
    events = []
    for step in range(steps + 1):
        elapsed = step * dt
        present = meta_side if hawkes and meta_start <= elapsed < 420 else 0
        child = int(
            present != 0
            and (
                elapsed - dt < meta_start
                or int((elapsed - meta_start) / 40) != int((elapsed - dt - meta_start) / 40)
            )
        )
        if hawkes:
            bid_intensity = 0.275 + (bid_intensity - 0.275) * math.exp(-dt)
            ask_intensity = 0.275 + (ask_intensity - 0.275) * math.exp(-dt)
        buy = int(rng.random() < 1 - math.exp(-bid_intensity * dt)) + int(child and present > 0)
        sell = int(rng.random() < 1 - math.exp(-ask_intensity * dt)) + int(child and present < 0)
        if hawkes:
            bid_intensity += 0.3 * buy
            ask_intensity += 0.3 * sell
            mid += 0.01 * (buy - sell) + rng.gauss(0, 0.01 * math.sqrt(dt))
        else:
            # Eq15 and Table1: eta=0; total variance = .1^2+.1^2+.1^2.
            mid += rng.gauss(0, math.sqrt(0.03 * dt))
        timestamp = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=elapsed)
        values = {
            "bid_price": mid - 0.005,
            "ask_price": mid + 0.005,
            "bid_size": 10,
            "ask_size": 10,
            "buy_flow": buy,
            "sell_flow": sell,
            "uniform_buy": rng.random(),
            "uniform_sell": rng.random(),
            "lambda_buy": bid_intensity,
            "lambda_sell": ask_intensity,
            "twap_present": present,
            "meta_child": child,
        }
        events.append(
            MarketEvent(
                event_id=f"quote:{step:05}",
                symbol="MM",
                event_time=timestamp,
                available_time=timestamp,
                values={k: Decimal(str(v)) for k, v in values.items()},
            )
        )
    return tuple(
        event
        for event in events
        if not hawkes
        or (event.event_time - datetime(2025, 1, 1, tzinfo=UTC)).total_seconds() >= 100
    )


def collect_episode(cls, events, *, seed, ppo=False, episode=0):
    agent = cls()
    if ppo:
        agent.initialize(seed)
    rng = random.Random(seed + episode + 100)
    records = []

    class Recorder:
        declaration = cls.declaration

        def decide(self, event, account):
            from paperquant.models import CancelOrder, NoOp, SubmitOrder

            inventory = float(account.positions[0].quantity) if account.positions else 0.0
            price = float((event.values["bid_price"] + event.values["ask_price"]) / 2)
            live = {order.split(":")[0]: order for order in account.open_order_ids}
            features = {"price": price, "inventory": inventory}
            if ppo:
                features.update(
                    lambda_buy=float(event.values["lambda_buy"]),
                    lambda_sell=float(event.values["lambda_sell"]),
                    twap_present=float(event.values["twap_present"]),
                    bid_live=int("buy" in live),
                    ask_live=int("sell" in live),
                )
                probabilities = agent.probabilities(features)
                draw, cumulative = rng.random(), 0.0
                action = 4
                for i, probability in enumerate(probabilities):
                    cumulative += probability.value
                    if draw < cumulative:
                        action = i
                        break
                features["old_log_probability"] = math.log(probabilities[action].value)
                if action in (3, 4):
                    side = "buy" if action == 3 else "sell"
                    actions = (
                        CancelOrder(
                            client_order_id=live[side],
                            symbol="MM",
                            reason="on-policy rollout cancel",
                        ),
                    )
                elif action in (1, 2):
                    side = "buy" if action == 1 else "sell"
                    actions = (
                        SubmitOrder(
                            client_order_id=f"{side}:{event.event_id}",
                            symbol="MM",
                            side=side,
                            quantity=Decimal(1),
                            limit_price=event.values["bid_price" if side == "buy" else "ask_price"],
                            reason="on-policy rollout quote",
                        ),
                    )
                else:
                    actions = (NoOp(reason="on-policy decision skips"),)
            else:
                latent = rng.uniform(-1, 1)
                action = 1 if latent > 1 / 3 else -1 if latent < -1 / 3 else 0
                if not -5 <= inventory + action <= 5:
                    latent, action = 0.0, 0
                features["behavior_latent"] = latent
                actions = tuple(
                    CancelOrder(client_order_id=order, symbol="MM", reason="renew interval control")
                    for order in account.open_order_ids
                )
                if action:
                    side = "buy" if action == 1 else "sell"
                    actions += (
                        SubmitOrder(
                            client_order_id=f"{side}:{event.event_id}",
                            symbol="MM",
                            side=side,
                            quantity=Decimal(1),
                            limit_price=event.values["bid_price" if action == 1 else "ask_price"],
                            reason="exploratory quote replay",
                        ),
                    )
                if not actions:
                    actions = (NoOp(reason="exploratory hold"),)
            records.append((features, action, account.equity))
            return actions

    engine = PassiveQuoteEngine(fee_rate=Decimal("0.0001") if ppo else Decimal(0))
    dataset = DatasetDeclaration(
        contract_version="1.1",
        dataset_id="collection",
        granularity="tick",
        market_kind="quote_l1",
        fields=frozenset(events[0].values),
        symbols=frozenset({"MM"}),
        event_count=len(events),
        content_sha256=fingerprint(events),
    )
    dummy = TrainingRequest(
        dataset_id="collection",
        seed=seed,
        samples=(TrainingSample(features={"collection": Decimal(1)}),),
    )
    plan, effective = compile_run(
        run_id="collect",
        strategy=cls.declaration,
        dataset=dataset,
        engine=engine.capability(dataset.fields),
        engine_profile=engine.profile(),
        policy=RunPolicy(),
        events=events,
        training=dummy,
    )
    decisions, orders, fills, accounts = engine.run(
        plan=plan, strategy=Recorder(), events=effective
    )
    samples = []
    dt = 0.213 if ppo else 0.001
    for i in range(len(records) - 1):
        features, action, equity = records[i]
        next_features = records[i + 1][0]
        reward = (
            float(records[i + 1][2] - equity) - 0.001 * abs(float(next_features["inventory"])) * dt
        )
        samples.append(
            TrainingSample(
                features={k: Decimal(str(v)) for k, v in features.items()},
                action=action,
                reward=Decimal(str(reward)),
                next_features={k: Decimal(str(v)) for k, v in next_features.items()},
                terminal=i == len(records) - 2,
                episode_id=f"episode-{episode}",
                step=i,
            )
        )
    trace = {
        "role": "behavior trajectory collection, not trained evaluation",
        "collector": "scripts/build_six_paper_inputs.py::collect_episode",
        "engine_profile": engine.profile().model_dump(mode="json"),
        "seed": seed,
        "episode": episode,
        "source_events_sha256": fingerprint(events),
        "actions": [d.model_dump(mode="json") for d in decisions],
        "orders": [o.model_dump(mode="json") for o in orders],
        "fills": [f.model_dump(mode="json") for f in fills],
        "accounts": [a.model_dump(mode="json") for a in accounts],
        "source_events": [e.model_dump(mode="json") for e in events],
    }
    return samples, trace


def rl_inputs(name, *, ppo=False):
    cls = package(name)
    seed = 43 if ppo else 37
    samples = []
    for episode in range(3 if ppo else 4):
        events = quote_events(
            seed + episode + 100,
            hawkes=ppo,
            meta_side=episode - 1 if ppo else 0,
            steps=1972 if ppo else 1000,
            meta_start=250 if ppo else 100,
        )
        transitions, trace = collect_episode(cls, events, seed=seed, ppo=ppo, episode=episode)
        samples.extend(transitions)
        save(OUTPUT / name / "collection" / f"episode-{episode}.json", trace)
    request = TrainingRequest(dataset_id="supplement." + name, seed=seed, samples=tuple(samples))
    evaluation = quote_events(
        seed + 9999, hawkes=ppo, meta_side=1 if ppo else 0, steps=1972 if ppo else 1000
    )
    publish(
        name,
        evaluation,
        dataset={"contract_version": "1.1", "market_kind": "quote_l1"},
        training=request,
        policy=RunPolicy(),
        provenance={
            "type": "controlled_simulation",
            "training_seed": seed,
            "evaluation_seed": seed + 9999,
            "training_episodes": 3 if ppo else 4,
            "behavior": "initial on-policy decision/action rollout"
            if ppo
            else "uniform continuous latent, quantized and inventory-bounded quote control replay",
            "dt": 0.213 if ppo else 0.001,
            "episode_seconds": 1972 * 0.213 if ppo else 1,
            "fee_rate": "0.0001 per fill" if ppo else "0",
            "nonadverse_probability": 0.2,
            "exclusions": (
                "reduced quote simulator; queue priority, partial fills and"
                " endogenous maker impact absent"
            ),
            "time_scope": "100-second warmup and 320-second trading window"
            if ppo
            else "Table1 dt=.001 and T=1 retained",
            "metaorder_scope": (
                "rPOV2 interval40s in training and evaluation; training observes a late-start "
                "prefix from t250, evaluation observes eight children from t100"
            )
            if ppo
            else "not_applicable",
            "paper_training_budget": "3 versus original 140+ PPO episodes"
            if ppo
            else "4 versus original 1000 SAC episodes",
        },
    )


def main():
    for name in CLASSES:
        package(name)
    graph_inputs()
    ou_inputs()
    cnn_inputs()
    gmm_inputs()
    rl_inputs("sac_passive_quotes")
    rl_inputs("ppo_twap_aware", ppo=True)
    files = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in OUTPUT.rglob("*.json")
    }
    save(ROOT / "research/six-paper/input-lock.json", {"files": files})
    print(json.dumps({"status": "built", "cases": 6, "files": len(files)}))


if __name__ == "__main__":
    main()
