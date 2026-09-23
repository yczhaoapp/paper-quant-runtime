"""Validate the meaning of market fields before an engine sees an event.

The wire format remains a flat decimal map for existing strategy packages. A
dataset declares one stream type, and these checks give that map typed market
semantics instead of treating field presence as proof of a valid quote or bar.
"""

from __future__ import annotations

import re
from decimal import Decimal

from paperquant.models import DatasetDeclaration, Granularity, MarketEvent, MarketKind

_BOOK_FIELD = re.compile(r"^(bid|ask)_(price|size)(?:_([1-9]\d*))?$")
_TOP_PRICES = frozenset({"bid_price", "ask_price"})
_TOP_SIZES = frozenset({"bid_size", "ask_size"})


def _book_levels(fields: frozenset[str] | set[str]) -> set[int]:
    return {int(match.group(3) or 1) for field in fields
            if (match := _BOOK_FIELD.fullmatch(field))}


def market_kind(dataset: DatasetDeclaration) -> MarketKind:
    if dataset.market_kind is not None:
        return dataset.market_kind
    if dataset.granularity != Granularity.TICK:
        return MarketKind.BAR
    levels = _book_levels(dataset.fields)
    if any(level > 1 for level in levels):
        return MarketKind.BOOK_L2
    if levels:
        return MarketKind.QUOTE_L1
    return MarketKind.TRADE


def market_depth(dataset: DatasetDeclaration) -> int:
    return max(_book_levels(dataset.fields), default=0)


def validate_market_event(dataset: DatasetDeclaration, event: MarketEvent) -> None:
    values = event.values
    fields = set(values)
    declared = set(dataset.fields)
    kind = market_kind(dataset)
    if any(not value.is_finite() for value in values.values()):
        raise ValueError("market fields must be finite")
    actual_levels = _book_levels(fields)
    declared_levels = _book_levels(declared)
    if actual_levels != declared_levels:
        raise ValueError("market event and dataset declare different order-book levels")
    if kind == MarketKind.BAR:
        if dataset.granularity == Granularity.TICK or actual_levels:
            raise ValueError("bar stream has tick or order-book fields")
        for field in ("open", "high", "low", "close"):
            if field in values and values[field] <= 0:
                raise ValueError(f"{field} must be positive")
        if "volume" in values and values["volume"] < 0:
            raise ValueError("bar volume cannot be negative")
        if {"high", "low"} <= fields and values["high"] < values["low"]:
            raise ValueError("bar high is below low")
        for field in ("open", "close"):
            if field in values and "high" in values and values[field] > values["high"]:
                raise ValueError(f"bar {field} exceeds high")
            if field in values and "low" in values and values[field] < values["low"]:
                raise ValueError(f"bar {field} falls below low")
        return
    if dataset.granularity != Granularity.TICK or event.bar_open_time is not None:
        raise ValueError("trade and book events must be tick events")
    if kind == MarketKind.TRADE:
        if actual_levels or "price" not in values or values["price"] <= 0:
            raise ValueError("trade stream requires a positive price and no order book")
        if dataset.market_kind == MarketKind.TRADE:
            if not event.trade_id or "trade_size" not in values or values["trade_size"] <= 0:
                raise ValueError("explicit trade stream requires trade_id and trade_size")
        elif "trade_size" in values and values["trade_size"] <= 0:
            raise ValueError("trade_size must be positive")
        return
    if event.trade_id is not None or event.aggressor_side is not None:
        raise ValueError("trade identity cannot appear on a quote or book event")
    if not _TOP_PRICES <= declared or not _TOP_PRICES <= fields:
        raise ValueError("quote stream requires both level-one prices")
    if kind == MarketKind.QUOTE_L1:
        size_fields = fields & _TOP_SIZES
        if size_fields and size_fields != _TOP_SIZES:
            raise ValueError("level-one quote sizes must be a complete pair")
        if dataset.market_kind == MarketKind.QUOTE_L1 and size_fields != _TOP_SIZES:
            raise ValueError("explicit level-one book requires both sizes")
    depth = max(declared_levels, default=0)
    if kind == MarketKind.QUOTE_L1 and depth != 1:
        raise ValueError("level-one quote cannot declare deeper book levels")
    if kind == MarketKind.BOOK_L2 and depth < 2:
        raise ValueError("level-two book requires at least two levels")
    if dataset.book_depth is not None and dataset.book_depth != depth:
        raise ValueError("declared book depth differs from the actual fields")
    if declared_levels != set(range(1, depth + 1)):
        raise ValueError("order-book levels must be contiguous")
    bid_prices: list[Decimal] = []
    ask_prices: list[Decimal] = []
    for level in range(1, depth + 1):
        suffix = "" if level == 1 else f"_{level}"
        names = {f"{side}_{field}{suffix}" for side in ("bid", "ask")
                 for field in ("price", "size")}
        if kind == MarketKind.QUOTE_L1 and not (fields & _TOP_SIZES):
            names = set(_TOP_PRICES)
        if not names <= declared or not names <= fields:
            raise ValueError(f"order-book level {level} is incomplete")
        for name in names:
            if values[name] <= 0:
                raise ValueError(f"order-book {name} must be positive")
        bid_prices.append(values[f"bid_price{suffix}"])
        ask_prices.append(values[f"ask_price{suffix}"])
    if bid_prices[0] >= ask_prices[0]:
        raise ValueError("level-one bid must be below ask")
    if any(left <= right for left, right in zip(bid_prices[:-1], bid_prices[1:], strict=True)):
        raise ValueError("bid prices must decrease through the book")
    if any(left >= right for left, right in zip(ask_prices[:-1], ask_prices[1:], strict=True)):
        raise ValueError("ask prices must increase through the book")
    if "price" in values and values["price"] <= 0:
        raise ValueError("last trade price must be positive")
