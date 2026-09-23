from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from test_contract import bars, dataset, declaration

from paperquant.engine import ReferenceEngine, reference_capability
from paperquant.models import ContractFault, ErrorCode, RunPolicy
from paperquant.package import run_package, write_manifest

SOURCE = """\
from decimal import Decimal
from paperquant.strategy_api import (
    DataNeed, Granularity, StrategyDeclaration, StrategyKind, TargetPosition,
)

class ExternalRule:
    declaration = StrategyDeclaration(
        strategy_id="sample-rule",
        kind=StrategyKind.RULE,
        data=DataNeed(
            granularity=Granularity.MINUTE,
            fields=frozenset({"close", "open"}),
            symbols=frozenset({"AAA"}),
        ),
        actions=frozenset({"target_position", "none"}),
        training_required=False,
        source="public-method-description",
        max_abs_position=Decimal("2"),
    )

    def decide(self, event, account):
        return (TargetPosition(symbol=event.symbol, quantity=Decimal("1"), reason="signal"),)
"""


def make_package(root: Path, source: str = SOURCE) -> None:
    root.mkdir()
    (root / "strategy.py").write_text(source, encoding="utf-8")
    write_manifest(root, declaration(), "ExternalRule")


def test_external_directory_runs_after_manifest_and_source_validation(tmp_path: Path) -> None:
    package = tmp_path / "package"
    make_package(package)
    events = bars()
    report = run_package(
        run_id="external-run",
        package_dir=package,
        dataset=dataset(events),
        engine_capability=reference_capability(frozenset(events[0].values)),
        policy=RunPolicy(),
        events=events,
        engine=ReferenceEngine(initial_cash=Decimal("1000")),
        output=tmp_path / "out",
    )
    assert len(report.decisions) == len(events)
    assert len(report.fills) == 1


def test_package_tampering_is_rejected_before_import(tmp_path: Path) -> None:
    package = tmp_path / "package"
    make_package(package)
    (package / "strategy.py").write_text(SOURCE + "\n# mutation\n", encoding="utf-8")
    events = bars()
    with pytest.raises(ContractFault) as caught:
        run_package(
            run_id="tampered-run",
            package_dir=package,
            dataset=dataset(events),
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(),
            events=events,
            engine=ReferenceEngine(),
            output=tmp_path / "out",
        )
    assert caught.value.failure.code == ErrorCode.INPUT_INVALID


def test_data_failure_precedes_strategy_import(tmp_path: Path) -> None:
    package = tmp_path / "package"
    make_package(package, SOURCE + "\nraise RuntimeError('module was imported')\n")
    events = bars()
    spec = dataset(events).model_copy(update={"fields": frozenset({"open"})})
    with pytest.raises(ContractFault) as caught:
        run_package(
            run_id="preload-failure",
            package_dir=package,
            dataset=spec,
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(),
            events=events,
            engine=ReferenceEngine(),
            output=tmp_path / "out",
        )
    assert caught.value.failure.code == ErrorCode.DATA_FIELD_MISSING
