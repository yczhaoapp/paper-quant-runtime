from __future__ import annotations

import json
from pathlib import Path

from paperquant.cli import SCHEMAS, main
from paperquant.compiler import fingerprint
from paperquant.models import DatasetDeclaration, MarketEvent
from paperquant.package import inspect_package

ROOT = Path(__file__).resolve().parents[1]


def test_committed_example_runs_through_directory_cli(tmp_path: Path) -> None:
    output = tmp_path / "run"
    status = main(
        [
            "run",
            "--package",
            str(ROOT / "examples/basic_rule"),
            "--dataset",
            str(ROOT / "examples/data/dataset.json"),
            "--events",
            str(ROOT / "examples/data/events.json"),
            "--output",
            str(output),
        ]
    )
    assert status == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "succeeded"
    assert report["plan"]["strategy_id"] == "example.moving_average"
    assert report["fills"]


def test_example_manifest_and_event_hashes_match_committed_bytes() -> None:
    inspect_package("example-integrity", ROOT / "examples/basic_rule")
    dataset = DatasetDeclaration.model_validate_json(
        (ROOT / "examples/data/dataset.json").read_bytes()
    )
    events = tuple(
        MarketEvent.model_validate(value)
        for value in json.loads((ROOT / "examples/data/events.json").read_text())
    )
    assert dataset.content_sha256 == fingerprint(events)
    assert dataset.event_count == len(events)


def test_schema_export_is_deterministic(tmp_path: Path) -> None:
    assert main(["schema", "--output", str(tmp_path / "schemas")]) == 0
    assert len(list((tmp_path / "schemas").glob("*.schema.json"))) == len(SCHEMAS)
    for name in SCHEMAS:
        assert (tmp_path / "schemas" / f"{name}.schema.json").read_bytes() == (
            ROOT / "schemas" / f"{name}.schema.json"
        ).read_bytes()


def test_sample_report_exports_comparable_observation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "--package",
                str(ROOT / "examples/basic_rule"),
                "--dataset",
                str(ROOT / "examples/data/dataset.json"),
                "--events",
                str(ROOT / "examples/data/events.json"),
                "--output",
                str(run_dir),
            ]
        )
        == 0
    )
    observation = tmp_path / "observation.json"
    assert (
        main(
            [
                "observe",
                "--report",
                str(run_dir / "report.json"),
                "--scenario",
                "sample-minute-bars",
                "--method",
                "moving-average-crossover",
                "--profile-sha256",
                "c" * 64,
                "--initial-cash",
                "100000",
                "--output",
                str(observation),
            ]
        )
        == 0
    )
    compared = tmp_path / "comparison.json"
    assert (
        main(
            [
                "compare",
                "--left",
                str(observation),
                "--right",
                str(observation),
                "--output",
                str(compared),
            ]
        )
        == 0
    )
    result = json.loads(compared.read_text(encoding="utf-8"))
    assert result["status"] == "comparable"
    assert result["equity_delta"] == "0"
    assert result["same_actions"] is True


def test_reused_output_directory_cannot_keep_an_old_success(tmp_path: Path) -> None:
    output = tmp_path / "reused-run"
    common = [
        "run",
        "--package",
        str(ROOT / "examples/basic_rule"),
        "--dataset",
        str(ROOT / "examples/data/dataset.json"),
        "--events",
        str(ROOT / "examples/data/events.json"),
        "--output",
        str(output),
    ]
    assert main(common) == 0
    first_attempt = json.loads((output / "attempt.json").read_text())
    assert first_attempt["status"] == "succeeded"
    assert (output / "report.json").is_file()

    broken = json.loads((ROOT / "examples/data/dataset.json").read_text())
    broken["content_sha256"] = "0" * 64
    broken_dataset = tmp_path / "broken-dataset.json"
    broken_dataset.write_text(json.dumps(broken), encoding="utf-8")
    second = common.copy()
    second[second.index("--dataset") + 1] = str(broken_dataset)
    assert main(second) == 2
    second_attempt = json.loads((output / "attempt.json").read_text())
    assert second_attempt["status"] == "failed"
    assert second_attempt["attempt_id"] != first_attempt["attempt_id"]
    assert not (output / "report.json").exists()
    assert (output / "failure.json").is_file()
