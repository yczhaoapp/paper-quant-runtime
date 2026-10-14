import json
import math
import random
from decimal import Decimal

from paperquant.strategy_api import (
    CancelOrder,
    DataNeed,
    DenseMLP,
    Granularity,
    MarketKind,
    NoOp,
    Scalar,
    StrategyDeclaration,
    StrategyKind,
    SubmitOrder,
    gradient_step,
)


class PPOTWAPAware:
    declaration = StrategyDeclaration(
        contract_version="1.1",
        strategy_id="supplement.ppo_twap_aware",
        kind=StrategyKind.REINFORCEMENT,
        data=DataNeed(
            granularity=Granularity.TICK,
            market_kind=MarketKind.QUOTE_L1,
            fields=frozenset(
                {
                    "bid_price",
                    "ask_price",
                    "bid_size",
                    "ask_size",
                    "lambda_buy",
                    "lambda_sell",
                    "twap_present",
                }
            ),
            symbols=frozenset({"MM"}),
        ),
        actions=frozenset({"none", "submit_order", "cancel_order"}),
        training_required=True,
        source="https://arxiv.org/pdf/2510.27334v1",
        max_abs_position=Decimal(10),
        max_order_quantity=Decimal(1),
    )
    clip = 0.2

    def initialize(self, seed):
        rng = random.Random(seed)
        self.decision = DenseMLP((5, 8, 2), rng)
        self.action = DenseMLP((5, 8, 4), rng)
        self.value = DenseMLP((5, 8, 1), rng)

    @staticmethod
    def state(features):
        return [
            (float(features["price"]) - 50) / 0.1,
            float(features["inventory"]) / 10,
            math.log1p(float(features["lambda_buy"])),
            math.log1p(float(features["lambda_sell"])),
            float(features["twap_present"]),
        ]

    @staticmethod
    def softmax(logits, mask=None):
        mask = [True] * len(logits) if mask is None else mask
        if not any(mask):
            raise ValueError("Empty feasible policy action set")
        maximum = max(v.value for v, valid in zip(logits, mask, strict=False) if valid)
        exponent = [
            (v - maximum).exp() if valid else Scalar(0)
            for v, valid in zip(logits, mask, strict=False)
        ]
        denominator = sum(exponent, Scalar(0))
        return [v / denominator for v in exponent]

    def probabilities(self, features):
        state = self.state(features)
        bid_live = bool(features["bid_live"])
        ask_live = bool(features["ask_live"])
        inventory = float(features["inventory"])
        mask = [
            not bid_live and inventory < 10,
            not ask_live and inventory > -10,
            bid_live,
            ask_live,
        ]
        decision = self.softmax(self.decision(state), [True, any(mask)])
        action = self.softmax(self.action(state), mask) if any(mask) else [Scalar(0)] * 4
        return [decision[0]] + [decision[1] * p for p in action]

    @staticmethod
    def clipped_surrogate(ratio, advantage, clip=0.2):
        value = ratio.value if isinstance(ratio, Scalar) else ratio
        clipped = max(1 - clip, min(1 + clip, value)) * advantage
        unclipped = ratio * advantage
        if isinstance(ratio, Scalar):
            return unclipped if unclipped.value <= clipped else Scalar(clipped)
        return min(unclipped, clipped)

    @staticmethod
    def self_imitation_loss(logp, value, target):
        positive = max(0.0, target - value.value)
        return -logp * positive + 0.5 * (value - target) ** 2 if positive > 0 else Scalar(0)

    def train(self, request):
        if len(request.samples) < 32 or any(
            s.reward is None or s.action is None for s in request.samples
        ):
            raise ValueError("PPO requires a completed action/reward rollout")
        self.initialize(request.seed)
        actor_params = self.decision.parameters() + self.action.parameters()
        params = actor_params + self.value.parameters()
        before = [p.value for p in params]
        returns = [0.0] * len(request.samples)
        accumulator = 0.0
        for i in range(len(request.samples) - 1, -1, -1):
            sample = request.samples[i]
            if sample.terminal:
                accumulator = 0.0
            accumulator = float(sample.reward) + 0.99 * accumulator
            returns[i] = accumulator
        old_log = []
        advantages = []
        for sample, ret in zip(request.samples, returns, strict=False):
            probabilities = self.probabilities(sample.features)
            p = probabilities[sample.action].value
            declared = float(sample.features["old_log_probability"])
            if p <= 0 or abs(math.log(p) - declared) > 1e-8:
                raise ValueError("PPO rollout is not bound to its initial on-policy behavior")
            old_log.append(declared)
            advantages.append(ret - self.value(self.state(sample.features))[0].value)
        mean = sum(advantages) / len(advantages)
        sd = math.sqrt(sum((a - mean) ** 2 for a in advantages) / len(advantages)) or 1
        advantages = [(a - mean) / sd for a in advantages]
        rng = random.Random(request.seed + 2000)
        losses = []
        imitation_updates = 0
        for _epoch in range(4):
            order = rng.sample(range(len(request.samples)), min(32, len(request.samples)))
            for i in order:
                sample = request.samples[i]
                p = self.probabilities(sample.features)[sample.action]
                logp = (p + 1e-12).log()
                ratio = (logp - old_log[i]).exp()
                policy_loss = -self.clipped_surrogate(ratio, advantages[i], self.clip)
                value = self.value(self.state(sample.features))[0]
                positive = max(0.0, returns[i] - value.value)
                # Positive-return advantage only: actor and value SIL losses.
                loss = policy_loss + 0.5 * (value - returns[i]) ** 2
                if positive > 0:
                    loss += 0.05 * self.self_imitation_loss(logp, value, returns[i])
                    imitation_updates += 1
                loss.backward()
                gradient_step(params, 0.002, clip=1)
                losses.append(loss.value)
        after = [p.value for p in params]
        return json.dumps(
            {
                "seed": request.seed,
                "decision": [p.value for p in self.decision.parameters()],
                "action": [p.value for p in self.action.parameters()],
                "value": [p.value for p in self.value.parameters()],
                "training": {
                    "transitions": len(request.samples),
                    "ppo_epochs": 4,
                    "minibatch_samples_per_epoch": 32,
                    "self_imitation_updates": imitation_updates,
                    "parameter_change_l1": sum(
                        abs(a - b) for a, b in zip(before, after, strict=False)
                    ),
                    "reward_sum": sum(float(s.reward) for s in request.samples),
                    "loss_first": losses[0],
                    "loss_last": losses[-1],
                    "behavior_binding": "initial decision/action policy log probabilities verified",
                },
            },
            sort_keys=True,
        ).encode()

    def load(self, payload):
        model = json.loads(payload)
        self.initialize(model["seed"])
        self.inference_rng = random.Random(model["seed"] + 4000)
        for network, key in [
            (self.decision, "decision"),
            (self.action, "action"),
            (self.value, "value"),
        ]:
            if len(network.parameters()) != len(model[key]):
                raise ValueError("PPO architecture mismatch")
            for p, v in zip(network.parameters(), model[key], strict=False):
                p.value = v

    def decide(self, event, account):
        orders = {order.split(":")[0]: order for order in account.open_order_ids}
        inventory = float(account.positions[0].quantity) if account.positions else 0
        features = {
            "price": (float(event.values["bid_price"]) + float(event.values["ask_price"])) / 2,
            "inventory": inventory,
            "lambda_buy": event.values["lambda_buy"],
            "lambda_sell": event.values["lambda_sell"],
            "twap_present": event.values["twap_present"],
            "bid_live": "buy" in orders,
            "ask_live": "sell" in orders,
        }
        probabilities = self.probabilities(features)
        draw = self.inference_rng.random()
        cumulative = 0.0
        action = 4
        for index, probability in enumerate(probabilities):
            cumulative += probability.value
            if draw < cumulative:
                action = index
                break
        if action == 0:
            return (NoOp(reason="PPO decision network skips intervention"),)
        if action in (3, 4):
            side = "buy" if action == 3 else "sell"
            return (
                CancelOrder(
                    client_order_id=orders[side],
                    symbol=event.symbol,
                    reason="PPO cancels live top-of-book quote",
                ),
            )
        side = "buy" if action == 1 else "sell"
        return (
            SubmitOrder(
                client_order_id=f"{side}:{event.event_id}",
                symbol=event.symbol,
                side=side,
                quantity=Decimal(1),
                limit_price=event.values["bid_price" if action == 1 else "ask_price"],
                reason=f"PPO intervention with explicit TWAP awareness {features['twap_present']}",
            ),
        )
