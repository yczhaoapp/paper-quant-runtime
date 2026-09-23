"""Reduced discrete A2C pair trader with causal rolling-regression observations."""

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


class PairsActorCritic:
    SYMBOLS = ("ALPHA", "BETA")
    WINDOW = 20
    ACTIONS = (-1, 0, 1)
    ACTOR_RATE = 0.5
    CRITIC_RATE = 0.2
    DISCOUNT = 0.95
    COST = 0.0002
    EPOCHS = 5

    declaration = StrategyDeclaration(
        strategy_id="reinforcement.pairs_actor_critic",
        kind=StrategyKind.REINFORCEMENT,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "high", "low", "close", "volume"}),
            symbols=frozenset(SYMBOLS),
            minimum_events_per_symbol=WINDOW + 2,
        ),
        actions=frozenset({"none", "prediction", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/2407.16103",
        max_abs_position=Decimal("10000"),
    )

    def __init__(self):
        self.actor = None
        self.critic = None
        self.prices = {symbol: [] for symbol in self.SYMBOLS}
        self.day = None
        self.seen = set()
        self.position = 0

    @staticmethod
    def _dot(weights, state):
        return sum(weight * component for weight, component in zip(weights, state, strict=True))

    @classmethod
    def observation(cls, left_history, right_history):
        """Fit on preceding observations only, then score the current completed close."""
        if len(left_history) != len(right_history) or len(left_history) <= cls.WINDOW:
            raise ValueError("incomplete synchronized pair history")
        left = [float(value) for value in left_history[-cls.WINDOW - 1:-1]]
        right = [float(value) for value in right_history[-cls.WINDOW - 1:-1]]
        if min(left + right) <= 0 or not all(math.isfinite(x) for x in left + right):
            raise ValueError("invalid pair prices")
        mean_left = sum(left) / cls.WINDOW
        mean_right = sum(right) / cls.WINDOW
        denominator = sum((value - mean_right) ** 2 for value in right)
        if denominator <= 0:
            raise ValueError("constant hedge regressor")
        slope = sum((a - mean_left) * (b - mean_right)
                    for a, b in zip(left, right, strict=True)) / denominator
        intercept = mean_left - slope * mean_right
        residuals = [a - intercept - slope * b
                     for a, b in zip(left, right, strict=True)]
        mean_residual = sum(residuals) / cls.WINDOW
        sigma = math.sqrt(sum((x - mean_residual) ** 2 for x in residuals) / cls.WINDOW)
        if sigma <= 1e-12:
            raise ValueError("constant pair spread")
        current_left = float(left_history[-1])
        current_right = float(right_history[-1])
        if current_left <= 0 or current_right <= 0:
            raise ValueError("invalid current pair prices")
        spread = current_left - intercept - slope * current_right
        z_score = (spread - mean_residual) / sigma
        zone = (-2 if z_score <= -1.8 else -1 if z_score < -0.4 else
                2 if z_score >= 1.8 else 1 if z_score > 0.4 else 0)
        return spread, z_score, zone

    @staticmethod
    def _state(features, position):
        left = float(features["left_price"])
        right = float(features["right_price"])
        spread = float(features["spread"])
        z_score = float(features["z_score"])
        zone = float(features["zone"])
        if (left <= 0 or right <= 0 or not all(math.isfinite(x) for x in
            (left, right, spread, z_score, zone)) or zone not in (-2, -1, 0, 1, 2)):
            raise ValueError("invalid pair training observation")
        return (1.0, float(position), spread / left * 10,
                max(-3.0, min(3.0, z_score)) / 3, zone / 2)

    @classmethod
    def _probabilities(cls, actor, state):
        logits = [cls._dot(row, state) for row in actor]
        peak = max(logits)
        weights = [math.exp(value - peak) for value in logits]
        total = sum(weights)
        return [value / total for value in weights]

    def train(self, request):
        if len(request.samples) < 60:
            raise ValueError("at least sixty pair transitions are required")
        actor = [[0.0] * 5 for _ in self.ACTIONS]
        critic = [0.0] * 5
        rng = random.Random(request.seed)
        for _ in range(self.EPOCHS):
            previous = 0
            for sample in request.samples:
                if sample.next_features is None:
                    raise ValueError("pair transition has no successor")
                state = self._state(sample.features, previous)
                probabilities = self._probabilities(actor, state)
                draw = rng.random()
                chosen = next(index for index, probability in enumerate(
                    (probabilities[0], probabilities[0] + probabilities[1], 1.0)
                ) if draw < probability)
                action = self.ACTIONS[chosen]
                successor = self._state(sample.next_features, action)
                left = float(sample.features["left_price"])
                right = float(sample.features["right_price"])
                left_next = float(sample.next_features["left_price"])
                right_next = float(sample.next_features["right_price"])
                pair_return = (left_next / left - right_next / right) / 2
                reward = action * pair_return - self.COST * abs(action - previous)
                value = self._dot(critic, state)
                continuation = (0 if sample.terminal else
                                self.DISCOUNT * self._dot(critic, successor))
                advantage = reward + continuation - value
                for row in range(3):
                    score = (1.0 if row == chosen else 0.0) - probabilities[row]
                    for column in range(5):
                        actor[row][column] += self.ACTOR_RATE * advantage * score * state[column]
                for column in range(5):
                    critic[column] += self.CRITIC_RATE * advantage * state[column]
                if not all(math.isfinite(value) for row in actor for value in row):
                    raise ValueError("pair actor diverged")
                if not all(math.isfinite(value) for value in critic):
                    raise ValueError("pair critic diverged")
                previous = 0 if sample.terminal else action
        return json.dumps({"actor": [[round(v, 12) for v in row] for row in actor],
                           "critic": [round(v, 12) for v in critic]},
                          sort_keys=True, separators=(",", ":")).encode()

    def load(self, payload):
        model = json.loads(payload.decode())
        if set(model) != {"actor", "critic"} or len(model["actor"]) != 3:
            raise ValueError("invalid pair actor-critic model")
        if any(len(row) != 5 for row in model["actor"]) or len(model["critic"]) != 5:
            raise ValueError("invalid pair actor-critic dimensions")
        numbers = [*model["critic"], *(v for row in model["actor"] for v in row)]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers):
            raise ValueError("nonfinite pair model")
        self.actor = model["actor"]
        self.critic = model["critic"]

    def decide(self, event, account):
        if self.actor is None or self.critic is None:
            raise ValueError("pair actor-critic model not loaded")
        day = event.event_time.date()
        if day != self.day:
            if self.day is not None and len(self.seen) != 2:
                raise ValueError("incomplete prior pair session")
            self.day = day
            self.seen = set()
        if event.symbol in self.seen:
            raise ValueError("duplicate pair symbol")
        self.seen.add(event.symbol)
        self.prices[event.symbol].append(event.values["close"])
        if len(self.seen) < 2:
            return (NoOp(reason="await synchronized pair close"),)
        if len(self.prices[self.SYMBOLS[0]]) <= self.WINDOW:
            return (NoOp(reason="causal regression formation window"),)
        left, right = self.SYMBOLS
        spread, z_score, zone = self.observation(self.prices[left], self.prices[right])
        features = {"left_price": self.prices[left][-1],
                    "right_price": self.prices[right][-1],
                    "spread": Decimal(str(spread)),
                    "z_score": Decimal(str(z_score)), "zone": Decimal(zone)}
        state = self._state(features, self.position)
        probabilities = self._probabilities(self.actor, state)
        chosen = max(range(3), key=lambda index: probabilities[index])
        self.position = self.ACTIONS[chosen]
        notional = account.equity / 2
        if notional <= 0:
            raise ValueError("pair account equity must be positive")
        left_shares = (Decimal(self.position) * notional / features["left_price"]
                       ).quantize(Decimal("0.000001"))
        right_shares = (-Decimal(self.position) * notional / features["right_price"]
                        ).quantize(Decimal("0.000001"))
        return (Prediction(symbol=left, value=Decimal(self.position),
                           reason="discrete A2C pair direction; spread and zone observed"),
                TargetPosition(symbol=left, quantity=left_shares,
                               reason="pair first leg target after synchronized close"),
                TargetPosition(symbol=right, quantity=right_shares,
                               reason="pair opposite leg target after synchronized close"))
