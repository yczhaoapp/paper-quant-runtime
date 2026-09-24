"""Auditable source-text intake; research interpretation remains a reviewed recipe."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from paperquant.package import inspect_package


class PaperModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(PaperModel):
    title: str
    url: str = Field(pattern=r"^https://")
    file: str
    format: Literal["pdf", "html", "tex"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str


class Anchor(PaperModel):
    page: int = Field(ge=1)
    text: str = Field(min_length=4)
    claim_index: int = Field(ge=0)
    source_index: int = Field(default=0, ge=0)


class ExpectedStrategy(PaperModel):
    strategy_id: str
    kind: Literal["rule", "supervised", "reinforcement"]
    granularity: Literal["tick", "minute", "day"]
    fields: frozenset[str]
    training_required: bool
    symbols: frozenset[str]
    actions: frozenset[str]
    market_kind: Literal["bar", "trade", "quote_l1", "book_l2"] | None = None
    bar_seconds: int | None = Field(default=None, ge=1)
    max_staleness_seconds: int | None = Field(default=None, ge=0)
    minimum_book_depth: int | None = Field(default=None, ge=2)


class Recipe(PaperModel):
    schema_version: Literal["1.0"] = "1.0"
    source: Source
    additional_sources: tuple[Source, ...] = ()
    package: str
    package_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_file: str
    spec_file: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected: ExpectedStrategy
    anchors: tuple[Anchor, ...] = Field(min_length=1)
    review_note: str = Field(min_length=16)


class MethodStep(PaperModel):
    role: Literal["data", "signal", "training", "inference", "execution", "reward"]
    claim_index: int = Field(ge=0)
    paper_rule: str = Field(min_length=12)
    runtime_rule: str = Field(min_length=12)
    implementation: str = Field(min_length=5)
    oracle_node: str = Field(pattern=r"^tests/[^\s]+\.py::[^\s]+$")
    adaptation: str = Field(min_length=12)


class MethodSpec(PaperModel):
    schema_version: Literal["1.0"] = "1.0"
    strategy_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    additional_source_sha256: tuple[str, ...] = ()
    fields: frozenset[str]
    steps: tuple[MethodStep, ...] = Field(min_length=1)
    excluded_paper_components: tuple[str, ...] = Field(min_length=1)


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self.skip += 1
        elif tag in {"p", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def fetch_paper_source(url: str, expected_sha256: str, destination: Path) -> str:
    """Explicit opt-in HTTPS fetch; publish only after byte digest verification."""
    if not url.startswith("https://") or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Paper fetch requires HTTPS and a pinned SHA-256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "paperquant-paper-intake/1.0"})
    descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=".paper-")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as out:
            with urllib.request.urlopen(request, timeout=30) as source:
                if not source.geturl().startswith("https://"):
                    raise ValueError("Paper fetch redirected outside HTTPS")
                digest = hashlib.sha256()
                size = 0
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        raise ValueError("Paper source exceeds 20 MiB")
                    out.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("Fetched paper differs from pinned SHA-256")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return expected_sha256


def extract_text(raw: bytes, format: Literal["pdf", "html", "tex"]) -> tuple[str, ...]:
    """Extract visible text, never infer equations or fill in missing content."""
    if format == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted or not reader.pages:
            raise ValueError("PDF is encrypted or empty")
        pages = tuple(page.extract_text() or "" for page in reader.pages)
        if not any(page.strip() for page in pages):
            raise ValueError("PDF has no extractable text; OCR is not automatic")
        return pages
    decoded = raw.decode("utf-8")
    if format == "html":
        parser = _HTMLText()
        parser.feed(decoded)
        return (" ".join(parser.parts),)
    if format == "tex":
        lines = [re.sub(r"(?<!\\)%.*$", "", line) for line in decoded.splitlines()]
        content = "\n".join(lines)
        content = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", content)
        content = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", content)
        return (content.replace("{", " ").replace("}", " "),)
    raise ValueError("Unsupported paper format")


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _resolve(recipe_file: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("Recipe paths must be relative to the recipe")
    return (recipe_file.parent / relative).resolve(strict=True)


def verify_recipe(recipe_file: Path) -> dict[str, object]:
    recipe_file = recipe_file.resolve(strict=True)
    recipe = Recipe.model_validate_json(recipe_file.read_bytes())
    sources = (recipe.source, *recipe.additional_sources)
    source_digests: list[str] = []
    source_pages: list[list[str]] = []
    for source_index, source in enumerate(sources):
        raw = _resolve(recipe_file, source.file).read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()
        if source_sha != source.sha256:
            raise ValueError("Paper source bytes differ from the recipe's pinned digest")
        source_digests.append(source_sha)
        if source.format == "pdf":
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted or not reader.pages:
                raise ValueError("PDF is encrypted or empty")
            pages = [""] * len(reader.pages)
            for anchor in recipe.anchors:
                if anchor.source_index != source_index:
                    continue
                if anchor.page > len(pages):
                    raise ValueError("Paper anchor page is out of range")
                pages[anchor.page - 1] = reader.pages[anchor.page - 1].extract_text() or ""
        else:
            pages = list(extract_text(raw, source.format))
        source_pages.append(pages)
    package_path = _resolve(recipe_file, recipe.package)
    manifest, strategy_source = inspect_package(recipe.expected.strategy_id, package_path)
    declaration = manifest.declaration
    if manifest.source_sha256 != recipe.package_source_sha256:
        raise ValueError("Strategy package source differs from reviewed recipe")
    expected = recipe.expected
    if (
        declaration.strategy_id != expected.strategy_id
        or declaration.kind != expected.kind
        or declaration.data.granularity != expected.granularity
        or declaration.data.fields != expected.fields
        or declaration.training_required != expected.training_required
        or declaration.data.symbols != expected.symbols
        or declaration.actions != expected.actions
        or declaration.data.market_kind != expected.market_kind
        or declaration.data.bar_seconds != expected.bar_seconds
        or declaration.data.max_staleness_seconds != expected.max_staleness_seconds
        or declaration.data.minimum_book_depth != expected.minimum_book_depth
        or declaration.source != recipe.source.url
    ):
        raise ValueError("Strategy declaration differs from the reviewed paper mapping")
    claim_path = _resolve(recipe_file, recipe.claim_file)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim_source = claim.get("source", claim.get("trading_source"))
    if claim["strategy_id"] != expected.strategy_id or claim_source["url"] != recipe.source.url:
        raise ValueError("Paper claim does not identify the mapped strategy and source")
    declared_algorithm = claim.get("algorithm_source")
    if (recipe.additional_sources or declared_algorithm is not None) and (
        not isinstance(declared_algorithm, dict)
        or tuple(source.url for source in recipe.additional_sources)
        != (declared_algorithm.get("url"),)
    ):
        raise ValueError("Supporting algorithm source differs from the research claim")
    spec_path = _resolve(recipe_file, recipe.spec_file)
    spec_bytes = spec_path.read_bytes()
    spec_sha = hashlib.sha256(spec_bytes).hexdigest()
    if spec_sha != recipe.spec_sha256:
        raise ValueError("Reviewed method spec differs from its pinned digest")
    spec = MethodSpec.model_validate_json(spec_bytes)
    if (
        spec.strategy_id != expected.strategy_id
        or spec.source_sha256 != source_digests[0]
        or spec.additional_source_sha256 != tuple(source_digests[1:])
        or spec.fields != expected.fields
        or {step.claim_index for step in spec.steps} != set(range(len(claim["claims"])))
    ):
        raise ValueError("Reviewed method spec does not cover the source and claims")
    strategy_tree = ast.parse(strategy_source)
    classes = {node.name: node for node in strategy_tree.body if isinstance(node, ast.ClassDef)}
    for step in spec.steps:
        if step.implementation.startswith("scripts/"):
            if not (Path(__file__).resolve().parents[2] / step.implementation).is_file():
                raise ValueError("Reviewed implementation script is absent")
        else:
            parts = step.implementation.split(".")
            if (
                len(parts) != 2
                or parts[0] not in classes
                or not any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == parts[1]
                    for node in classes[parts[0]].body
                )
            ):
                raise ValueError("Reviewed implementation symbol is absent")
    evidence = []
    for anchor in recipe.anchors:
        if (
            anchor.source_index >= len(sources)
            or anchor.page > len(source_pages[anchor.source_index])
            or anchor.claim_index >= len(claim["claims"])
        ):
            raise ValueError("Paper anchor page or claim index is out of range")
        if _normalized(anchor.text) not in _normalized(
            source_pages[anchor.source_index][anchor.page - 1]
        ):
            raise ValueError(
                f"Paper anchor missing on page {anchor.page}: "
                f"{recipe_file.name}, source {anchor.source_index}, text {anchor.text!r}"
            )
        evidence.append(
            {
                "page": anchor.page,
                "text": anchor.text,
                "source_index": anchor.source_index,
                "source_sha256": source_digests[anchor.source_index],
                "claim_index": anchor.claim_index,
                "locator": claim["claims"][anchor.claim_index]["locator"],
            }
        )
    return {
        "strategy_id": expected.strategy_id,
        "paper_title": recipe.source.title,
        "source_url": recipe.source.url,
        "source_sha256": source_digests[0],
        "additional_source_sha256": tuple(source_digests[1:]),
        "format": recipe.source.format,
        "page_count": len(source_pages[0]),
        "additional_page_counts": tuple(len(pages) for pages in source_pages[1:]),
        "package_source_sha256": manifest.source_sha256,
        "claim_sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest(),
        "method_spec_sha256": spec_sha,
        "method_steps": len(spec.steps),
        "method_oracle_nodes": tuple(step.oracle_node for step in spec.steps),
        "anchors": evidence,
        "review_note": recipe.review_note,
        "interpretation": "human-reviewed mapping; anchor checks do not prove semantic fidelity",
    }
