import json
import random
from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class DoubleQInventory:
    declaration = StrategyDeclaration(
        strategy_id="reinforcement.double_q_market_making",
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

    def __init__(self, alpha=Decimal("0.001"), gamma=Decimal("0.97")):
        if not Decimal(0) <= alpha <= Decimal(1) or not Decimal(0) <= gamma <= Decimal(1):
            raise ValueError("alpha and gamma must lie in [0,1]")
        self.alpha = alpha
        self.gamma = gamma
        self.qa = None
        self.qb = None

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
        if not request.samples:
            raise ValueError("Double Q requires transitions")
        generator = random.Random(request.seed)
        qa = {}
        qb = {}
        for sample in request.samples:
            if sample.reward is None or sample.action not in self.ACTIONS:
                raise ValueError("transition lacks a supported action or reward")
            current_state = self._state(sample.features)
            key = self._key(current_state, sample.action)
            update_a = generator.randrange(2) == 0
            selected = qa if update_a else qb
            other = qb if update_a else qa
            old = selected.get(key, Decimal(0))
            continuation = Decimal(0)
            if not sample.terminal:
                if sample.next_features is None:
                    raise ValueError("nonterminal transition lacks next state")
                successor = self._state(sample.next_features)
                selected_action = max(
                    self.ACTIONS,
                    key=lambda action: selected.get(self._key(successor, action), Decimal(0)),
                )
                continuation = other.get(self._key(successor, selected_action), Decimal(0))
            selected[key] = old + self.alpha * (
                sample.reward + self.gamma * continuation - old
            )
        return json.dumps(
            {
                "alpha": str(self.alpha),
                "gamma": str(self.gamma),
                "qa": {key: str(value) for key, value in sorted(qa.items())},
                "qb": {key: str(value) for key, value in sorted(qb.items())},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if not isinstance(model, dict) or set(model) != {"alpha", "gamma", "qa", "qb"}:
            raise ValueError("invalid Double Q model")
        alpha = Decimal(model["alpha"])
        gamma = Decimal(model["gamma"])
        if not Decimal(0) <= alpha <= Decimal(1) or not Decimal(0) <= gamma <= Decimal(1):
            raise ValueError("invalid Double Q hyperparameters")
        if not isinstance(model["qa"], dict) or not isinstance(model["qb"], dict):
            raise ValueError("invalid Double Q tables")
        self.alpha = alpha
        self.gamma = gamma
        self.qa = {str(key): Decimal(value) for key, value in model["qa"].items()}
        self.qb = {str(key): Decimal(value) for key, value in model["qb"].items()}

    def decide(self, event, account):
        if self.qa is None or self.qb is None:
            raise ValueError("Double Q model not loaded")
        current = next(
            (
                position.quantity
                for position in account.positions
                if position.symbol == event.symbol
            ),
            Decimal(0),
        )
        state = self._state(
            {
                "bid_size": event.values["bid_size"],
                "ask_size": event.values["ask_size"],
                "inventory": current,
            }
        )
        chosen = max(
            self.ACTIONS,
            key=lambda action: self.qa.get(self._key(state, action), Decimal(0))
            + self.qb.get(self._key(state, action), Decimal(0)),
        )
        target = Decimal(chosen)
        if target == current:
            return (NoOp(reason="inventory already matches Double Q policy"),)
        return (
            TargetPosition(
                symbol=event.symbol,
                quantity=target,
                reason="Double Q action adapted to target inventory",
            ),
        )
