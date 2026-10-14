import json
import math
from decimal import Decimal

from paperquant.strategy_api import (
    DataNeed,
    Granularity,
    NoOp,
    StrategyDeclaration,
    StrategyKind,
    TargetPosition,
)


class OUPercentilePairs:
    declaration = StrategyDeclaration(
        contract_version="1.1",
        strategy_id="supplement.ou_percentile",
        kind=StrategyKind.SUPERVISED,
        data=DataNeed(
            granularity=Granularity.DAY,
            fields=frozenset({"open", "close"}),
            symbols=frozenset({"OU_A", "OU_B"}),
        ),
        actions=frozenset({"none", "target_position"}),
        training_required=True,
        source="https://arxiv.org/pdf/2412.12458v1",
        max_abs_position=Decimal("1"),
        max_order_quantity=Decimal("2"),
    )

    def __init__(self):
        self.frame = {}
        self.frame_time = None
        self.position = 0

    @staticmethod
    def calibrate(spreads, delta=1.0):
        if len(spreads) < 92 or delta <= 0:
            raise ValueError("OU calibration requires 92 observations and positive dt")
        x, y = spreads[:-1], spreads[1:]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        variance = sum((v - mx) ** 2 for v in x)
        if variance <= 1e-16:
            raise ValueError("OU calibration is singular")
        a = sum((i - mx) * (j - my) for i, j in zip(x, y, strict=False)) / variance
        b = my - a * mx
        if not 0 < a < 1:
            raise ValueError("Fitted AR(1) coefficient is not a stationary OU process")
        residual = [j - a * i - b for i, j in zip(x, y, strict=False)]
        sd = math.sqrt(sum(r * r for r in residual) / (len(residual) - 2))
        if sd <= 0:
            raise ValueError("OU diffusion is degenerate")
        return {
            "a": a,
            "b": b,
            "lambda": -math.log(a) / delta,
            "mu": b / (1 - a),
            "sigma": sd * math.sqrt(-2 * math.log(a) / (delta * (1 - a * a))),
        }

    @staticmethod
    def quantile(values, level):
        ordered = sorted(values)
        location = (len(ordered) - 1) * level
        lo, hi = int(location), min(len(ordered) - 1, int(location) + 1)
        return ordered[lo] + (location - lo) * (ordered[hi] - ordered[lo])

    def train(self, request):
        spreads = [float(s.features["spread"]) for s in request.samples]
        parameters = self.calibrate(spreads)
        z = [(v - parameters["mu"]) / parameters["sigma"] for v in spreads[-90:]]
        return json.dumps(
            {
                "parameters": parameters,
                "z_history": z,
                "training": {
                    "seed": request.seed,
                    "observations": len(spreads),
                    "method": "OLS AR1 and Eq10 OU calibration",
                },
            },
            sort_keys=True,
        ).encode()

    def load(self, payload):
        model = json.loads(payload)
        self.parameters, self.z_history = model["parameters"], model["z_history"]

    def decide(self, event, account):
        del account
        if self.frame and self.frame_time != event.available_time:
            raise ValueError("Daily OU pair is not synchronized")
        self.frame_time = event.available_time
        self.frame[event.symbol] = float(event.values["close"])
        if len(self.frame) < 2:
            return (NoOp(reason="wait for both daily closes"),)
        spread = self.frame["OU_A"] - self.frame["OU_B"]
        self.frame = {}
        z = (spread - self.parameters["mu"]) / self.parameters["sigma"]
        q25, q50, q75 = [self.quantile(self.z_history[-90:], p) for p in (0.25, 0.5, 0.75)]
        if self.position == 0:
            self.position = -1 if z > q75 else 1 if z < q25 else 0
        elif self.position == 1 and z >= q50 or self.position == -1 and z <= q50:
            self.position = 0
        self.z_history = (self.z_history + [z])[-90:]
        return (
            TargetPosition(
                symbol="OU_A", quantity=Decimal(self.position), reason="PIT OU percentile"
            ),
            TargetPosition(symbol="OU_B", quantity=Decimal(-self.position), reason="paired hedge"),
        )
