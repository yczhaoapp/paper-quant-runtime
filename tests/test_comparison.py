from __future__ import annotations

from decimal import Decimal

from paperquant.comparison import Observation, ObservedAction, compare


def observation() -> Observation:
    return Observation(
        scenario_id="shared-market-window",
        method_id="moving-average-crossover",
        source_events_sha256="a" * 64,
        effective_events_sha256="b" * 64,
        execution_profile_sha256="c" * 64,
        initial_cash=Decimal("1000"),
        final_equity=Decimal("1002"),
        actions=(ObservedAction(event_id="one", kind="none"),),
        fills=(),
    )


def test_same_conditions_allow_effect_and_path_comparison() -> None:
    baseline = observation()
    candidate = baseline.model_copy(
        update={
            "final_equity": Decimal("1005"),
            "actions": (
                ObservedAction(
                    event_id="one", kind="target_position", symbol="AAA", quantity=Decimal("1")
                ),
            ),
        }
    )
    result = compare(baseline, candidate)
    assert result.status == "comparable"
    assert result.equity_delta == Decimal("3")
    assert result.same_actions is False
    assert result.action_count_delta == 0


def test_different_execution_profile_blocks_numerical_claim() -> None:
    baseline = observation()
    candidate = baseline.model_copy(
        update={"execution_profile_sha256": "d" * 64, "final_equity": Decimal("1005")}
    )
    result = compare(baseline, candidate)
    assert result.status == "incomparable"
    assert result.reasons == ("execution_profile_sha256",)
    assert result.equity_delta is None
