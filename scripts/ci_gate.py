"""Stable required CI check: every dependent job must finish successfully."""

import json
import os


def verify_results(needs: object) -> None:
    required = {"host", "strict", "paper-depth"}
    if not isinstance(needs, dict) or set(needs) != required:
        raise ValueError("CI gate dependencies are absent or unexpected")
    if any(not isinstance(job, dict) or job.get("result") != "success"
           for job in needs.values()):
        raise ValueError("Every required CI job must succeed; skipped/cancelled jobs cannot pass")


if __name__ == "__main__":
    verify_results(json.loads(os.environ["CI_JOB_RESULTS"]))
    print("All required release jobs succeeded")
