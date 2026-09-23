import json
import math
import random
from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    Prediction,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class RiskAverseBandit:
    """Scalar disjoint normal-gamma Thompson sampling with online updates."""

    ARMS = (-1, 0, 1)
    RISK_TOLERANCE = 0.1

    declaration = StrategyDeclaration(
        strategy_id="reinforcement.risk_averse_bandit",
        kind=StrategyKind.REINFORCEMENT,
        data=DataNeed(
            granularity=Granularity.TICK,
            fields=frozenset({"price", "bid_price", "ask_price", "bid_size", "ask_size"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=3,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/2206.12463v1",
        max_abs_position=Decimal(1),
    )

    def __init__(self):
        self.posterior = None
        self.generator = None
        self.previous_mid = None
        self.previous_context = None
        self.previous_held_arm = None

    @staticmethod
    def _context(values):
        bid = float(values["bid_size"])
        ask = float(values["ask_size"])
        bid_price = float(values["bid_price"])
        ask_price = float(values["ask_price"])
        if not all(math.isfinite(value) for value in (bid, ask, bid_price, ask_price)):
            raise ValueError("nonfinite L1 book")
        if bid < 0 or ask < 0 or bid + ask <= 0 or bid_price <= 0 or ask_price <= bid_price:
            raise ValueError("invalid L1 book")
        return (bid - ask) / (bid + ask), (bid_price + ask_price) / 2

    @staticmethod
    def _prior():
        return {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0}

    @staticmethod
    def _update(parameters, context, reward):
        if not math.isfinite(context) or not math.isfinite(reward):
            raise ValueError("nonfinite context or reward")
        old_a, old_b = parameters["a"], parameters["b"]
        new_a = old_a + context * context
        new_b = old_b + context * reward
        parameters["a"] = new_a
        parameters["b"] = new_b
        parameters["c"] += 0.5
        parameters["d"] += 0.5 * (
            reward * reward + old_b * old_b / old_a - new_b * new_b / new_a
        )
        if parameters["d"] <= 0:
            raise ValueError("nonpositive posterior rate")

    def train(self, request):
        if len(request.samples) < 12:
            raise ValueError("bandit needs historical observed rewards")
        posterior = {str(arm): self._prior() for arm in self.ARMS}
        counts = {arm: 0 for arm in self.ARMS}
        for sample in request.samples:
            if sample.action not in self.ARMS or sample.reward is None:
                raise ValueError("bandit observation lacks arm or reward")
            context = float(sample.features["context"])
            self._update(posterior[str(sample.action)], context, float(sample.reward))
            counts[sample.action] += 1
        if min(counts.values()) < 2:
            raise ValueError("each arm requires two observed rewards")
        return json.dumps(
            {"posterior": posterior, "seed": request.seed},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"posterior", "seed"} or type(model["seed"]) is not int:
            raise ValueError("invalid bandit model")
        if set(model["posterior"]) != {str(arm) for arm in self.ARMS}:
            raise ValueError("missing bandit arm")
        for parameters in model["posterior"].values():
            if set(parameters) != {"a", "b", "c", "d"}:
                raise ValueError("invalid posterior fields")
            if not all(type(value) in (int, float) and math.isfinite(value)
                       for value in parameters.values()):
                raise ValueError("nonfinite posterior")
            if parameters["a"] <= 0 or parameters["c"] <= 0 or parameters["d"] <= 0:
                raise ValueError("invalid posterior scale")
        self.posterior = model["posterior"]
        self.generator = random.Random(model["seed"])
        self.previous_mid = None
        self.previous_context = None
        self.previous_held_arm = None

    def _sample_scores(self, context):
        scores = {}
        for arm in self.ARMS:
            parameters = self.posterior[str(arm)]
            precision = self.generator.gammavariate(parameters["c"], 1 / parameters["d"])
            mean = self.generator.gauss(
                parameters["b"] / parameters["a"],
                math.sqrt(1 / (precision * parameters["a"])),
            )
            scores[arm] = context * mean - self.RISK_TOLERANCE / precision
        return scores

    def decide(self, event, account):
        if self.posterior is None or self.generator is None:
            raise ValueError("bandit model not loaded")
        context, midpoint = self._context(event.values)
        if self.previous_mid is not None:
            realized = self.previous_held_arm * (midpoint - self.previous_mid) / self.previous_mid
            self._update(
                self.posterior[str(self.previous_held_arm)], self.previous_context, realized
            )
        position = next(
            (item.quantity for item in account.positions if item.symbol == event.symbol),
            Decimal(0),
        )
        if position not in (Decimal(-1), Decimal(0), Decimal(1)):
            raise ValueError("bandit requires unit positions")
        scores = self._sample_scores(context)
        chosen = max(self.ARMS, key=lambda arm: scores[arm])
        self.previous_mid = midpoint
        self.previous_context = context
        self.previous_held_arm = int(position)
        prediction = Prediction(
            symbol=event.symbol,
            value=Decimal(str(round(scores[chosen], 10))),
            reason=f"mean-variance posterior sample arm {chosen}",
        )
        if Decimal(chosen) == position:
            return (prediction, NoOp(reason="sampled arm already held"))
        return (prediction, TargetPosition(
            symbol=event.symbol, quantity=Decimal(chosen), reason="risk-averse sampled arm"
        ))
