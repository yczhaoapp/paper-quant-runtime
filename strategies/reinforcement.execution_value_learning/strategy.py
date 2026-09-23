import json
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


class ExecutionValueLearning:
    """Backward empirical action values on time, inventory and market pressure."""

    TOTAL = 4
    HORIZON = 4

    declaration = StrategyDeclaration(
        strategy_id="reinforcement.execution_value_learning",
        kind=StrategyKind.REINFORCEMENT,
        data=DataNeed(
            granularity=Granularity.TICK,
            fields=frozenset({"price", "bid_size", "ask_size", "time_remaining"}),
            symbols=frozenset({"DEMO"}),
            minimum_events_per_symbol=5,
        ),
        actions=frozenset({"prediction", "none", "target_position"}),
        training_required=True,
        source="https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf",
        max_abs_position=Decimal(TOTAL),
    )

    def __init__(self):
        self.values = None

    @staticmethod
    def _pressure(bid, ask):
        if bid < 0 or ask < 0 or bid + ask <= 0:
            raise ValueError("invalid best-quote sizes")
        imbalance = (bid - ask) / (bid + ask)
        return 1 if imbalance > Decimal("0.1") else -1 if imbalance < Decimal("-0.1") else 0

    @classmethod
    def _state(cls, features):
        time = features["time_remaining"]
        remaining = features["remaining"]
        pressure = features["pressure"]
        if time != int(time) or not 0 <= time <= cls.HORIZON:
            raise ValueError("invalid time state")
        if remaining != int(remaining) or not 0 <= remaining <= cls.TOTAL:
            raise ValueError("invalid remaining inventory state")
        if pressure not in (-1, 0, 1):
            raise ValueError("invalid market pressure state")
        return int(time), int(remaining), int(pressure)

    @staticmethod
    def _key(state, action):
        return f"{state[0]}:{state[1]}:{state[2]}:{action}"

    @staticmethod
    def _actions(time, remaining):
        if remaining == 0:
            return (0,)
        if time == 1:
            return (remaining,)
        return tuple(range(min(2, remaining) + 1))

    def train(self, request):
        if not request.samples:
            raise ValueError("execution transitions are required")
        grouped = {}
        for sample in request.samples:
            if sample.reward is None or sample.action is None:
                raise ValueError("execution transition lacks reward or action")
            state = self._state(sample.features)
            time, remaining, _ = state
            if time < 1 or sample.action not in self._actions(time, remaining):
                raise ValueError("unsupported execution action")
            if time == 1:
                if not sample.terminal or sample.next_features is not None:
                    raise ValueError("horizon terminal action cannot bootstrap")
                successor = None
            else:
                if sample.terminal or sample.next_features is None:
                    raise ValueError("nonterminal action requires next private state")
                successor = self._state(sample.next_features)
                if successor[:2] != (time - 1, remaining - sample.action):
                    raise ValueError("execution transition changes private state incorrectly")
            grouped.setdefault((state, sample.action), []).append((sample.reward, successor))

        table = {}
        for time in range(1, self.HORIZON + 1):
            for remaining in range(self.TOTAL + 1):
                for pressure in (-1, 0, 1):
                    state = (time, remaining, pressure)
                    for action in self._actions(time, remaining):
                        transitions = grouped.get((state, action))
                        if not transitions:
                            raise ValueError("execution training misses a state-action pair")
                        observed = Decimal(0)
                        count = 0
                        for reward, successor in transitions:
                            continuation = Decimal(0)
                            if successor is not None:
                                candidates = (
                                    table[self._key(successor, next_action)]
                                    for next_action in self._actions(successor[0], successor[1])
                                )
                                continuation = max(candidates)
                            count += 1
                            observed += (reward + continuation - observed) / count
                        table[self._key(state, action)] = observed
        return json.dumps(
            {"q": {key: str(value) for key, value in sorted(table.items())}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def load(self, payload):
        model = json.loads(payload.decode("utf-8"))
        if set(model) != {"q"} or not isinstance(model["q"], dict):
            raise ValueError("invalid execution model")
        table = {str(key): Decimal(value) for key, value in model["q"].items()}
        expected = {
            self._key((time, remaining, pressure), action)
            for time in range(1, self.HORIZON + 1)
            for remaining in range(self.TOTAL + 1)
            for pressure in (-1, 0, 1)
            for action in self._actions(time, remaining)
        }
        if set(table) != expected or not all(value.is_finite() for value in table.values()):
            raise ValueError("incomplete or nonfinite execution table")
        self.values = table

    def decide(self, event, account):
        if self.values is None:
            raise ValueError("execution model not loaded")
        time = event.values["time_remaining"]
        held = next((item.quantity for item in account.positions if item.symbol == event.symbol),
                    Decimal(0))
        if time == 0:
            if held == 0:
                return (NoOp(reason="execution episode already reset"),)
            return (TargetPosition(symbol=event.symbol, quantity=Decimal(0),
                                   reason="reset finished execution episode"),)
        state = self._state({
            "time_remaining": time,
            "remaining": Decimal(self.TOTAL) - held,
            "pressure": self._pressure(event.values["bid_size"], event.values["ask_size"]),
        })
        actions = self._actions(state[0], state[1])
        chosen = max(actions, key=lambda action: self.values[self._key(state, action)])
        prediction = Prediction(
            symbol=event.symbol,
            value=Decimal(chosen),
            reason="backward-learned execution child size",
        )
        if chosen == 0:
            return (prediction, NoOp(reason="wait for later execution interval"))
        return (prediction, TargetPosition(
            symbol=event.symbol,
            quantity=held + chosen,
            reason="execute learned child quantity on next tick",
        ))
