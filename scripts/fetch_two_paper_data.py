"""Fetch fixed public intervals, verifying exchange-published archive checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/six-paper/public-btc"


def prepare(*, cnn_history=False, cnn_holdout=False):
    if cnn_holdout:
        lock_path = ROOT / "research/cnn-refinement/selection-lock.json"
        if not lock_path.is_file():
            raise ValueError("CNN holdout acquisition requires a persisted selection lock")
        lock = json.loads(lock_path.read_text())
        plan = ROOT / "research/cnn-refinement/experiment-plan.json"
        source = ROOT / "strategies/supplement.grouped_cnn_lstm/strategy.py"
        if (
            hashlib.sha256(plan.read_bytes()).hexdigest() != lock["experiment_plan_sha256"]
            or hashlib.sha256(source.read_bytes()).hexdigest() != lock["package_source_sha256"]
        ):
            raise ValueError("CNN selection lock differs from the registered plan or strategy")
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = DATA / "SOURCE.json"
    records = json.loads(manifest.read_text())["sources"]
    by_file = {record["csv"]: record for record in records}
    intervals = [("4h", f"2024-{month:02d}") for month in range(1, 10)]
    intervals += [("1m", f"2024-{month:02d}") for month in range(1, 6)]
    if cnn_history:
        intervals.append(("4h", "2023-12"))
    if cnn_holdout:
        intervals += [("4h", month) for month in json.loads(plan.read_text())["evaluation"]]
    for interval, month in intervals:
        name = f"BTCUSDT-{interval}-{month}"
        path = DATA / (name + ".csv")
        if path.name in by_file:
            if hashlib.sha256(path.read_bytes()).hexdigest() != by_file[path.name]["csv_sha256"]:
                raise ValueError("Existing public CSV differs from its checksum")
            continue
        url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/{interval}/{name}.zip"
        with urllib.request.urlopen(url + ".CHECKSUM", timeout=30) as response:
            checksum_bytes = response.read(4096)
        checksum = checksum_bytes.decode().split()[0]
        with urllib.request.urlopen(url, timeout=30) as response:
            compressed = response.read(32 * 1024 * 1024 + 1)
        if len(compressed) > 32 * 1024 * 1024:
            raise ValueError("Public archive exceeds declared size")
        if hashlib.sha256(compressed).hexdigest() != checksum:
            raise ValueError("Public archive differs from its published checksum")
        with zipfile.ZipFile(io.BytesIO(compressed)) as archive:
            if archive.namelist() != [path.name]:
                raise ValueError("Unexpected archive member")
            content = archive.read(path.name)
        rows = list(csv.reader(io.StringIO(content.decode())))
        stamps = [int(row[0]) for row in rows]
        seconds = 14400 if interval == "4h" else 60
        unit = "microseconds" if month >= "2025-01" else "milliseconds"
        units_per_second = 1000000 if unit == "microseconds" else 1000
        if not stamps or any(
            b - a != seconds * units_per_second for a, b in zip(stamps, stamps[1:], strict=False)
        ):
            raise ValueError("Public data contains a gap, duplicate or invalid order")
        path.write_bytes(content)
        record = {
            "url": url,
            "published_checksum_url": url + ".CHECKSUM",
            "archive_sha256": checksum,
            "csv": path.name,
            "csv_sha256": hashlib.sha256(content).hexdigest(),
            "rows": len(rows),
            "timestamp_unit": unit + " since Unix epoch",
            "timestamp_conversion": "explicit source-metadata conversion to milliseconds in "
            "upstream builder",
            "scope": "BTC/USDT spot OHLCV, UTC; explicit paper exchange/window adaptation",
        }
        records.append(record)
        manifest.write_text(json.dumps({"sources": records}, indent=2) + "\n")
        print(path.name, "checksum verified", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-history", action="store_true")
    parser.add_argument("--cnn-holdout", action="store_true")
    args = parser.parse_args()
    prepare(cnn_history=args.cnn_history, cnn_holdout=args.cnn_holdout)
