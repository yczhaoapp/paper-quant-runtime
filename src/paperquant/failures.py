"""Normalize ordinary failures without hiding an existing contract error."""

from collections.abc import Iterator
from contextlib import contextmanager

from paperquant.models import ContractFault, ErrorCode, Failure, Stage


@contextmanager
def failure_boundary(
    run_id: str, stage: Stage, code: ErrorCode, message: str
) -> Iterator[None]:
    try:
        yield
    except ContractFault:
        raise
    except Exception as exc:
        raise ContractFault(Failure(
            run_id=run_id, stage=stage, code=code, message=message,
            details={"cause": type(exc).__name__, "reason": str(exc)[:200]},
        )) from exc
