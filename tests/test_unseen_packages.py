"""Exercise packages created after the fixed eighteen-case release catalog."""

from __future__ import annotations

import os
import uuid
from decimal import Decimal
from pathlib import Path

import pytest
from test_contract import bars, dataset

from paperquant.compiler import fingerprint
from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import (
    DataNeed,
    Granularity,
    RunPolicy,
    StrategyDeclaration,
    StrategyKind,
    TrainingRequest,
    TrainingSample,
)
from paperquant.package import run_package, write_manifest


def _package(root: Path, kind: StrategyKind) -> tuple[Path, TrainingRequest | None]:
    strategy_id = f"outside.catalog.{kind.value}.{uuid.uuid4().hex}"
    declaration = StrategyDeclaration(
        strategy_id=strategy_id,
        kind=kind,
        data=DataNeed(
            granularity=Granularity.MINUTE,
            fields=frozenset({"open", "close"}),
            symbols=frozenset({"AAA"}),
        ),
        actions=frozenset({"target_position"}),
        training_required=kind != StrategyKind.RULE,
        source="urn:paperquant:unseen-package-contract-probe",
        max_abs_position=Decimal("1"),
    )
    root.mkdir()
    source = f'''from decimal import Decimal
from paperquant.strategy_api import (
    DataNeed, Granularity, StrategyDeclaration, StrategyKind, TargetPosition,
)

class UnseenStrategy:
    declaration = StrategyDeclaration(
        strategy_id="{strategy_id}",
        kind=StrategyKind.{kind.name},
        data=DataNeed(granularity=Granularity.MINUTE,
                      fields=frozenset({{"open", "close"}}),
                      symbols=frozenset({{"AAA"}})),
        actions=frozenset({{"target_position"}}),
        training_required={kind != StrategyKind.RULE},
        source="urn:paperquant:unseen-package-contract-probe",
        max_abs_position=Decimal("1"),
    )

    def train(self, request):
        score = sum((sample.reward if self.declaration.kind == StrategyKind.REINFORCEMENT
                     else sample.target) for sample in request.samples)
        return str(score).encode("ascii")

    def load(self, payload):
        self.score = Decimal(payload.decode("ascii"))

    def decide(self, event, account):
        score = Decimal("1") if self.declaration.kind == StrategyKind.RULE else self.score
        return (TargetPosition(symbol=event.symbol,
                               quantity=Decimal("1") if score > 0 else Decimal("0"),
                               reason="unseen package decision"),)
'''
    (root / "strategy.py").write_text(source, encoding="utf-8")
    write_manifest(root, declaration, "UnseenStrategy")
    training = None
    if kind != StrategyKind.RULE:
        training = TrainingRequest(
            dataset_id="local-bars",
            seed=17,
            samples=(TrainingSample(
                features={"close": Decimal("100")},
                target=Decimal("1") if kind == StrategyKind.SUPERVISED else None,
                reward=Decimal("2") if kind == StrategyKind.REINFORCEMENT else None,
            ),),
        )
    return root, training


@pytest.mark.parametrize("kind", tuple(StrategyKind))
def test_unregistered_strategy_uses_the_same_host_contract(
    kind: StrategyKind, tmp_path: Path
) -> None:
    package, training = _package(tmp_path / "package", kind)
    events = bars()
    report = run_package(
        run_id="unseen-host", package_dir=package, dataset=dataset(events),
        engine_capability=reference_capability(frozenset(events[0].values)),
        policy=RunPolicy(), events=events, engine=ReferenceEngine(),
        output=tmp_path / "run", training=training,
    )
    assert report.plan.strategy_id.startswith("outside.catalog.")
    assert len(report.decisions) == len(events)
    assert len(report.fills) == 1
    assert (report.artifact is None) == (training is None)
    if training is not None:
        assert report.artifact is not None
        assert report.artifact.training_sha256 == fingerprint(training)


@pytest.mark.skipif(
    os.environ.get("PAPERQUANT_TEST_DOCKER") != "1",
    reason="strict worker runs in the release verification gate",
)
@pytest.mark.parametrize("kind", tuple(StrategyKind))
def test_unregistered_strategy_matches_strict_worker(kind: StrategyKind, tmp_path: Path) -> None:
    package, training = _package(tmp_path / "package", kind)
    events = bars()
    common = dict(
        package_dir=package, dataset=dataset(events),
        engine_capability=reference_capability(frozenset(events[0].values)),
        events=events, engine=ReferenceEngine(), training=training,
    )
    host = run_package(
        run_id="unseen-host", policy=RunPolicy(), output=tmp_path / "host", **common
    )
    strict = run_package(
        run_id="unseen-strict", policy=RunPolicy(sandbox="strict"),
        output=tmp_path / "strict", **common,
    )
    assert strict.worker_receipt is not None
    assert host.decisions == strict.decisions
    assert host.orders == strict.orders
    assert host.fills == strict.fills
    assert host.accounts == strict.accounts
    assert host.final_equity == strict.final_equity
    assert (host.artifact is None) == (strict.artifact is None)
    if host.artifact is not None and strict.artifact is not None:
        assert host.artifact.content_sha256 == strict.artifact.content_sha256
