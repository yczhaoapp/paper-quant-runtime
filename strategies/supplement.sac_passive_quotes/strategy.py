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


class SACPassiveQuotes:
    declaration = StrategyDeclaration(
        contract_version="1.1",
        strategy_id="supplement.sac_passive_quotes",
        kind=StrategyKind.REINFORCEMENT,
        data=DataNeed(
            granularity=Granularity.TICK,
            market_kind=MarketKind.QUOTE_L1,
            fields=frozenset({"bid_price", "ask_price", "bid_size", "ask_size"}),
            symbols=frozenset({"MM"}),
        ),
        actions=frozenset({"none", "submit_order", "cancel_order"}),
        training_required=True,
        source="https://arxiv.org/pdf/2410.14504v2",
        max_abs_position=Decimal(5),
        max_order_quantity=Decimal(1),
    )
    alpha, gamma, tau = 0.2, 0.99, 0.01

    def initialize(self, seed):
        rng = random.Random(seed)
        self.actor = DenseMLP((2, 8, 8, 2), rng)
        self.critics = [DenseMLP((3, 8, 8, 1), rng) for _ in range(2)]
        self.targets = [DenseMLP((3, 8, 8, 1), rng) for _ in range(2)]
        for target, critic in zip(self.targets, self.critics, strict=False):
            for a, b in zip(target.parameters(), critic.parameters(), strict=False):
                a.value = b.value

    def state(self, features):
        return [
            (float(features["price"]) - self.center) / self.scale,
            (float(features["inventory"]) + 5) / 10,
        ]

    def policy(self, state, noise):
        mean, raw_scale = self.actor(state)
        log_std = -1 + raw_scale.tanh()
        latent = mean + log_std.exp() * noise
        action = latent.tanh()
        logp = (
            -0.5 * noise**2 - log_std - 0.5 * math.log(2 * math.pi) - (1 - action**2 + 1e-6).log()
        )
        return action, logp

    @staticmethod
    def soft_target(reward, terminal, next_q1, next_q2, log_probability, gamma=0.99, alpha=0.2):
        return (
            reward
            if terminal
            else reward + gamma * (min(next_q1, next_q2) - alpha * log_probability)
        )

    def train(self, request):
        if len(request.samples) < 32 or any(
            s.reward is None or s.action is None or s.next_features is None for s in request.samples
        ):
            raise ValueError("SAC requires 32 action/reward/next-state transitions")
        for sample in request.samples:
            latent = float(sample.features["behavior_latent"])
            control = 1 if latent > 1 / 3 else -1 if latent < -1 / 3 else 0
            if not -1 <= latent <= 1 or sample.action != control:
                raise ValueError("SAC reward action does not match its collected latent")
        self.initialize(request.seed)
        prices = [float(s.features["price"]) for s in request.samples]
        self.center = sum(prices) / len(prices)
        self.scale = math.sqrt(sum((p - self.center) ** 2 for p in prices) / len(prices))
        if self.scale <= 1e-12:
            raise ValueError("SAC training prices have no variation")
        rng = random.Random(request.seed + 1000)
        params = self.actor.parameters() + [
            p for network in self.critics for p in network.parameters()
        ]
        before = [p.value for p in params]
        losses = []
        for _iteration in range(128):
            sample = request.samples[rng.randrange(len(request.samples))]
            state = self.state(sample.features)
            next_state = self.state(sample.next_features)
            action, logp = self.policy(next_state, rng.gauss(0, 1))
            future = [network(next_state + [action.value])[0].value for network in self.targets]
            target = self.soft_target(
                float(sample.reward), sample.terminal, *future, logp.value, self.gamma, self.alpha
            )
            estimates = [
                network(state + [float(sample.features["behavior_latent"])])[0]
                for network in self.critics
            ]
            critic_loss = sum(((q - target) ** 2 / 2 for q in estimates), Scalar(0))
            critic_loss.backward()
            gradient_step([p for network in self.critics for p in network.parameters()], 0.01)
            action, logp = self.policy(state, rng.gauss(0, 1))
            estimates = [network(state + [action])[0] for network in self.critics]
            minimum = min(estimates, key=lambda q: q.value)
            actor_loss = self.alpha * logp - minimum
            actor_loss.backward()
            gradient_step(self.actor.parameters(), 0.005)
            for target_net, critic in zip(self.targets, self.critics, strict=False):
                for p, q in zip(target_net.parameters(), critic.parameters(), strict=False):
                    p.value = (1 - self.tau) * p.value + self.tau * q.value
            losses.append([critic_loss.value, actor_loss.value])
        after = [p.value for p in params]
        return json.dumps(
            {
                "seed": request.seed,
                "actor": [p.value for p in self.actor.parameters()],
                "critics": [[p.value for p in net.parameters()] for net in self.critics],
                "targets": [[p.value for p in net.parameters()] for net in self.targets],
                "center": self.center,
                "scale": self.scale,
                "training": {
                    "transitions": len(request.samples),
                    "updates": 128,
                    "losses": losses,
                    "parameter_change_l1": sum(
                        abs(a - b) for a, b in zip(before, after, strict=False)
                    ),
                    "reward_sum": sum(float(s.reward) for s in request.samples),
                    "behavior": "collected quote-action replay",
                    "actor": "Gaussian tanh latent mapped to three posting controls",
                },
            },
            sort_keys=True,
        ).encode()

    def load(self, payload):
        model = json.loads(payload)
        self.initialize(model["seed"])
        for network, values in [
            (self.actor, model["actor"]),
            *zip(self.critics, model["critics"], strict=False),
            *zip(self.targets, model["targets"], strict=False),
        ]:
            if len(network.parameters()) != len(values):
                raise ValueError("SAC architecture mismatch")
            for p, v in zip(network.parameters(), values, strict=False):
                p.value = v
        self.center, self.scale = model["center"], model["scale"]
        self.inference_rng = random.Random(model["seed"] + 3000)

    def decide(self, event, account):
        inventory = float(account.positions[0].quantity) if account.positions else 0
        price = (float(event.values["bid_price"]) + float(event.values["ask_price"])) / 2
        state = self.state({"price": price, "inventory": inventory})
        score = self.policy(state, self.inference_rng.gauss(0, 1))[0].value
        # Quantization is explicit: the source gives discrete posting controls
        # while the SAC actor emits a continuous Gaussian action.
        control = (
            1 if score > 1 / 3 and inventory < 5 else -1 if score < -1 / 3 and inventory > -5 else 0
        )
        actions = [
            CancelOrder(
                client_order_id=order, symbol=event.symbol, reason="next-interval posting control"
            )
            for order in account.open_order_ids
        ]
        if control:
            side = "buy" if control == 1 else "sell"
            actions.append(
                SubmitOrder(
                    client_order_id=f"{side}:{event.event_id}",
                    symbol=event.symbol,
                    side=side,
                    quantity=Decimal(1),
                    limit_price=event.values["bid_price" if control == 1 else "ask_price"],
                    reason=f"inventory-bounded SAC posting score {score:.8f}",
                )
            )
        return tuple(actions) if actions else (NoOp(reason="SAC selects no quote"),)
