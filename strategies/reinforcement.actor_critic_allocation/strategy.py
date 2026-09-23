import json
import math
import random
from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    Prediction,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class ActorCriticAllocation:
    """On-policy logistic-normal actor, linear TD critic, one risky asset plus cash."""

    ACTOR_RATE = 1.0
    CRITIC_RATE = 0.1
    NOISE = 0.2

    declaration = StrategyDeclaration(
        strategy_id="reinforcement.actor_critic_allocation",
        kind=StrategyKind.REINFORCEMENT,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "high", "low", "close", "volume"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=2,
        ),
        actions=frozenset({"prediction", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/1911.11880v2",
        max_abs_position=Decimal(10000),
    )

    def __init__(self):
        self.actor = None
        self.critic = None

    @staticmethod
    def _sigmoid(value):
        if value >= 0:
            return 1 / (1 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1 + exponential)

    @staticmethod
    def _state(values):
        opening = float(values["open"])
        high = float(values["high"])
        low = float(values["low"])
        close = float(values["close"])
        volume = float(values["volume"])
        if not all(math.isfinite(value) for value in (opening, high, low, close, volume)):
            raise ValueError("nonfinite OHLCV")
        if opening <= 0 or low <= 0 or high < low or not low <= close <= high or volume < 0:
            raise ValueError("invalid OHLCV")
        return (1.0, 10 * (close / opening - 1), 10 * (high - low) / opening)

    @staticmethod
    def _dot(weights, features):
        return sum(weight * feature for weight, feature in zip(weights, features, strict=True))

    def train(self, request):
        if len(request.samples) < 80:
            raise ValueError("at least eighty chronological transitions are required")
        generator = random.Random(request.seed)
        actor = [0.0, 0.0, 0.0]
        critic = [0.0, 0.0, 0.0]
        for sample in request.samples:
            if sample.next_features is None:
                raise ValueError("allocation transition lacks next completed bar")
            state = self._state(sample.features)
            successor = self._state(sample.next_features)
            current_close = float(sample.features["close"])
            next_close = float(sample.next_features["close"])
            latent_mean = self._dot(actor, state)
            latent_action = generator.gauss(latent_mean, self.NOISE)
            risky_weight = self._sigmoid(latent_action)
            gross = 1 + risky_weight * (next_close / current_close - 1)
            if gross <= 0:
                raise ValueError("nonpositive portfolio gross return")
            reward = math.log(gross)
            value = self._dot(critic, state)
            continuation = 0.0 if sample.terminal else self._dot(critic, successor)
            advantage = reward + continuation - value
            score_factor = (latent_action - latent_mean) / (self.NOISE**2)
            for column in range(3):
                critic[column] += self.CRITIC_RATE * advantage * state[column]
                actor[column] += self.ACTOR_RATE * advantage * score_factor * state[column]
            if not all(math.isfinite(value) for value in actor + critic):
                raise ValueError("actor-critic diverged")
        return json.dumps(
            {
                "actor": [round(value, 12) for value in actor],
                "critic": [round(value, 12) for value in critic],
                "noise": self.NOISE,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"actor", "critic", "noise"}:
            raise ValueError("invalid actor-critic model fields")
        if len(model["actor"]) != 3 or len(model["critic"]) != 3:
            raise ValueError("invalid actor-critic dimension")
        numbers = [*model["actor"], *model["critic"], model["noise"]]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers):
            raise ValueError("nonfinite actor-critic model")
        if model["noise"] <= 0:
            raise ValueError("invalid policy noise")
        self.actor = model["actor"]
        self.critic = model["critic"]

    def decide(self, event, account):
        if self.actor is None or self.critic is None:
            raise ValueError("actor-critic model not loaded")
        state = self._state(event.values)
        risky_weight = self._sigmoid(self._dot(self.actor, state))
        shares = (
            Decimal(str(round(risky_weight, 10))) * account.equity / event.values["close"]
        ).quantize(Decimal("0.000001"))
        if shares < 0:
            raise ValueError("negative risky allocation")
        return (
            Prediction(
                symbol=event.symbol,
                value=Decimal(str(round(risky_weight, 10))),
                reason="actor risky-asset weight; remaining weight is cash",
            ),
            TargetPosition(
                symbol=event.symbol,
                quantity=shares,
                reason="actor weight adapted to next-open target shares",
            ),
        )
