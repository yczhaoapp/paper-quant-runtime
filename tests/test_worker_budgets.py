"""Training budgets are caller policy, independent of the paper's algorithm."""

from __future__ import annotations

import io
import queue
from types import SimpleNamespace

import pytest

from paperquant.models import ContractFault
from paperquant.sandbox import DockerWorkerStrategy


@pytest.mark.parametrize("budget", [None, 0.25, 60.0])
def test_operation_budgets_apply_to_training_only(budget):
    worker = DockerWorkerStrategy.__new__(DockerWorkerStrategy)
    worker._training_timeout = budget
    worker._process = SimpleNamespace(stdin=io.BytesIO())
    observed = []

    def receive(*, timeout):
        observed.append(timeout)
        return {"ok": True}

    worker._receive = receive
    for operation in ("train", "load", "decide"):
        assert worker._exchange({"op": operation}) == {"ok": True}
    assert observed == [budget, 20, 20]


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_invalid_budget_is_rejected_before_starting_docker(budget):
    with pytest.raises(ValueError, match="training_timeout"):
        DockerWorkerStrategy(
            run_id="invalid-budget", package_dir=None, declaration=None,
            source_sha256="unused", training_timeout=budget,
        )


def test_uncapped_training_still_rejects_a_closed_worker_stream():
    worker = DockerWorkerStrategy.__new__(DockerWorkerStrategy)
    worker.run_id = "closed-stream"
    worker._replies = queue.Queue()
    worker._replies.put(None)
    with pytest.raises(ContractFault, match="closed the protocol stream"):
        worker._receive(timeout=None)
