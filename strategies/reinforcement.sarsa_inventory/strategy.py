import json
from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class InventorySarsa:
    declaration = StrategyDeclaration(
        strategy_id="reinforcement.sarsa_inventory",
        kind=StrategyKind.REINFORCEMENT,
        data=DataNeed(
            granularity=Granularity.TICK,
            fields=frozenset({"price", "bid_size", "ask_size"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=2,
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/1804.04216",
        max_abs_position=Decimal("1"),
    )
    ACTIONS = (0, 1, -1)

    def __init__(self, alpha=Decimal("0.4"), gamma=Decimal("0.8")):
        if not Decimal("0") <= alpha <= Decimal("1") or not Decimal("0") <= gamma <= Decimal("1"):
            raise ValueError("learning rate and discount must be in [0,1]")
        self.alpha = alpha
        self.gamma = gamma
        self.values = None

    @staticmethod
    def _state(features):
        bid = features["bid_size"]
        ask = features["ask_size"]
        inventory = features["inventory"]
        if bid < 0 or ask < 0 or bid + ask <= 0:
            raise ValueError("invalid best-quote sizes")
        imbalance = (bid - ask) / (bid + ask)
        pressure = 1 if imbalance > Decimal("0.1") else -1 if imbalance < Decimal("-0.1") else 0
        holding = 1 if inventory > 0 else -1 if inventory < 0 else 0
        return f"{pressure}:{holding}"

    @staticmethod
    def _key(state, action):
        return f"{state}:{action}"

    def train(self, request):
        if not request.samples or not request.samples[-1].terminal:
            raise ValueError("training episodes must finish with a terminal transition")
        table = {}
        for index, sample in enumerate(request.samples):
            if sample.reward is None or sample.action not in self.ACTIONS:
                raise ValueError("transition lacks a supported action or reward")
            state = self._state(sample.features)
            key = self._key(state, sample.action)
            old = table.get(key, Decimal("0"))
            if sample.terminal:
                if sample.next_action is not None:
                    raise ValueError("terminal transition cannot bootstrap")
                continuation = Decimal("0")
            else:
                if sample.next_features is None or sample.next_action not in self.ACTIONS:
                    raise ValueError("SARSA requires the behavior policy's next action")
                if index + 1 >= len(request.samples):
                    raise ValueError("nonterminal transition cannot end an episode")
                successor = request.samples[index + 1]
                if (
                    sample.next_features != successor.features
                    or sample.next_action != successor.action
                ):
                    raise ValueError("next action must be the next observed behavior action")
                next_state = self._state(sample.next_features)
                continuation = table.get(self._key(next_state, sample.next_action), Decimal("0"))
            table[key] = old + self.alpha * (sample.reward + self.gamma * continuation - old)
        return json.dumps(
            {
                "alpha": str(self.alpha),
                "gamma": str(self.gamma),
                "q": {key: str(value) for key, value in sorted(table.items())},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def load(self, payload):
        encoded = json.loads(payload.decode("utf-8"))
        if not isinstance(encoded, dict) or set(encoded) != {"alpha", "gamma", "q"}:
            raise ValueError("invalid value table")
        self.alpha = Decimal(encoded["alpha"])
        self.gamma = Decimal(encoded["gamma"])
        if not Decimal("0") <= self.alpha <= Decimal("1") or not Decimal(
            "0"
        ) <= self.gamma <= Decimal("1"):
            raise ValueError("invalid model parameters")
        if not isinstance(encoded["q"], dict):
            raise ValueError("invalid value table")
        self.values = {str(key): Decimal(value) for key, value in encoded["q"].items()}

    def decide(self, event, account):
        if self.values is None:
            raise ValueError("model not loaded")
        current = next(
            (
                position.quantity
                for position in account.positions
                if position.symbol == event.symbol
            ),
            Decimal("0"),
        )
        state = self._state(
            {
                "bid_size": event.values["bid_size"],
                "ask_size": event.values["ask_size"],
                "inventory": current,
            }
        )
        action = max(
            self.ACTIONS,
            key=lambda candidate: self.values.get(self._key(state, candidate), Decimal("0")),
        )
        desired = Decimal(action)
        if desired == current:
            return (NoOp(reason="inventory already matches on-policy decision"),)
        return (
            TargetPosition(
                symbol=event.symbol,
                quantity=desired,
                reason="SARSA table action adapted to target inventory",
            ),
        )
