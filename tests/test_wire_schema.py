from __future__ import annotations

from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from paperquant.models import Fill, SubmitOrder


@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    [
        ("quantity", "0.25", True),
        ("quantity", "1E+2", True),
        ("quantity", "0", False),
        ("quantity", "-1", False),
        ("quantity", "NaN", False),
        ("limit_price", None, True),
        ("limit_price", "99.5", True),
        ("limit_price", "0", False),
        ("limit_price", "-5", False),
    ],
)
def test_order_wire_constraints_match_model(field: str, value: str | None, accepted: bool) -> None:
    payload = {"client_order_id": "id", "symbol": "AAA", "side": "buy",
               "quantity": "1", "limit_price": "100", "reason": "test"}
    payload[field] = value
    schema_valid = Draft202012Validator(SubmitOrder.model_json_schema()).is_valid(payload)
    try:
        SubmitOrder.model_validate(payload)
        model_valid = True
    except ValidationError:
        model_valid = False
    assert schema_valid is model_valid is accepted


@pytest.mark.parametrize(
    ("fee", "accepted"),
    [("0", True), ("0.01", True), ("-0", True), ("-0.01", False), ("NaN", False)],
)
def test_fill_fee_schema_and_model_agree(fee: str, accepted: bool) -> None:
    from datetime import UTC, datetime
    payload = {"order_id": "one", "symbol": "AAA", "side": "buy",
               "quantity": "1", "price": "100", "fee": fee,
               "timestamp": datetime(2026, 1, 2, tzinfo=UTC).isoformat()}
    schema_valid = Draft202012Validator(Fill.model_json_schema()).is_valid(payload)
    try:
        fill = Fill.model_validate(payload)
        model_valid = True
        assert Draft202012Validator(Fill.model_json_schema()).is_valid(
            fill.model_dump(mode="json"))
    except ValidationError:
        model_valid = False
    assert schema_valid is model_valid is accepted


def test_serialized_positive_decimal_is_valid_wire_data() -> None:
    order = SubmitOrder(client_order_id="id", symbol="AAA", side="sell",
                        quantity=Decimal("1E+2"), limit_price=Decimal("0.25"), reason="test")
    assert Draft202012Validator(SubmitOrder.model_json_schema()).is_valid(
        order.model_dump(mode="json"))
