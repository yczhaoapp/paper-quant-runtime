"""Regenerate a local research strategy's deterministic manifest and source hash."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from paperquant.package import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("class_name")
    args = parser.parse_args()
    source = args.package / "strategy.py"
    spec = importlib.util.spec_from_file_location("manifest_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy_class = getattr(module, args.class_name)
    write_manifest(args.package, strategy_class.declaration, args.class_name)


if __name__ == "__main__":
    main()
