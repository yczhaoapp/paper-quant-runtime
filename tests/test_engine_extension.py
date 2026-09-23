from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from test_contract import TinyLearner, bars, dataset

from paperquant.engine import reference_capability
from paperquant.models import (
    AccountSnapshot,
    ContractFault,
    Decision,
    EngineCapability,
    EngineProfile,
    ErrorCode,
    Granularity,
    RunPolicy,
    TrainingRequest,
    TrainingSample,
)
from paperquant.runtime import run


class SignalOnlyLearner(TinyLearner):
    declaration = TinyLearner.declaration.model_copy(
        update={"actions": frozenset({"none"})})


class IndependentSignalEngine:
    """A second engine driver used to exercise injection and capability binding."""

    def capability(self, fields: frozenset[str]) -> EngineCapability:
        return EngineCapability(
            engine_id="test.independent-signal-engine",
            granularities=frozenset({Granularity.MINUTE}),
            data_fields=fields,
            actions=frozenset({"none"}),
        )

    def profile(self) -> EngineProfile:
        return EngineProfile(engine_id="test.independent-signal-engine",
                             settings={"initial_cash": "1000", "matching": "none",
                                       "fee_rate": "0"})

    def run(self, *, plan, strategy, events):
        assert plan.engine_id == "test.independent-signal-engine"
        decisions = []
        accounts = []
        for event in events:
            snapshot = AccountSnapshot(
                timestamp=event.available_time, cash=Decimal(1000),
                equity=Decimal(1000), positions=(), open_order_ids=(),
            )
            actions = strategy.decide(event, snapshot)
            assert len(actions) == 1 and actions[0].kind == "none"
            decisions.append(Decision(
                event_id=event.event_id, event_time=event.event_time, actions=actions))
            accounts.append(snapshot)
        return tuple(decisions), (), (), tuple(accounts)


def test_separate_engine_executes_train_reload_infer_and_report(tmp_path: Path) -> None:
    events = bars()
    engine = IndependentSignalEngine()
    request = TrainingRequest(
        dataset_id="local-bars", seed=2,
        samples=(TrainingSample(features={"close": Decimal(100)}, target=Decimal(1)),
                 TrainingSample(features={"close": Decimal(101)}, target=Decimal(3))),
    )
    learner = SignalOnlyLearner()
    report = run(
        run_id="signal-engine", strategy=learner, dataset=dataset(events),
        engine_capability=engine.capability(frozenset(events[0].values)),
        policy=RunPolicy(),
        events=events, engine=engine, output=tmp_path, training=request,
    )
    assert learner.threshold is None
    assert report.plan.engine_id == "test.independent-signal-engine"
    assert report.artifact is not None
    assert len(report.decisions) == len(events)
    assert report.final_equity == 1000
    assert (tmp_path / "report.json").is_file()


def test_engine_cannot_execute_another_drivers_capabilities(tmp_path: Path) -> None:
    events = bars()
    with pytest.raises(ContractFault) as caught:
        run(
            run_id="false-capability", strategy=SignalOnlyLearner(), dataset=dataset(events),
            engine_capability=reference_capability(frozenset(events[0].values)),
            policy=RunPolicy(),
            events=events, engine=IndependentSignalEngine(), output=tmp_path,
            training=TrainingRequest(
                dataset_id="local-bars", seed=2,
                samples=(TrainingSample(features={"close": Decimal(100)}, target=Decimal(1)),),
            ),
        )
    assert caught.value.failure.code == ErrorCode.ENGINE_UNSUPPORTED
    assert not (tmp_path / "report.json").exists()
