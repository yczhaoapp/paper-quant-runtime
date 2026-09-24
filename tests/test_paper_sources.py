"""Check that source locking covers the complete executable catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_paper_sources import LOCK, _expected_claim_urls, _resolve_file, verify_sources


def test_catalog_sources_are_pinned_individually() -> None:
    lock = json.loads(LOCK.read_text())
    sources = lock["sources"]
    assert len(sources) == 16
    assert {source["canonical_url"] for source in sources} == _expected_claim_urls()
    assert all(len(source["sha256"]) == 64 for source in sources)


def test_source_paths_reject_escape_and_symlinks(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="within the repository"):
        _resolve_file("../outside.pdf")
    import scripts.verify_paper_sources as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "link.pdf").symlink_to(LOCK)
    with pytest.raises(ValueError, match="symlink"):
        module._resolve_file("link.pdf")


def test_all_primary_pdfs_match_pins_when_cache_is_prepared() -> None:
    lock = json.loads(LOCK.read_text())
    if any(not _resolve_file(source["file"]).is_file() for source in lock["sources"]):
        pytest.skip("Run explicit source preparation before the full offline byte check")
    checked = verify_sources()
    assert len(checked) == 16
    assert all(item["pages"] > 0 for item in checked)
