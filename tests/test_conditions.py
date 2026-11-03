import pytest
from pydantic import ValidationError

from dsl.condition import NumericCondition


def test_valid_numeric_condition():
    condition = NumericCondition(
        field="dividend_change_pct",
        op=">",
        value=10,
    )

    assert condition.field == "dividend_change_pct"
    assert condition.op == ">"
    assert condition.value == 10.0


@pytest.mark.parametrize(
    "changes",
    [
        {"field": "future_return"},
        {"op": "execute"},
        {"value": "10"},
        {"value": True},
        {"value": float("nan")},
        {"value": float("inf")},
        {"unexpected": "argument"},
    ],
)
def test_invalid_numeric_condition_is_rejected(changes):
    inputs = {
        "field": "dividend_change_pct",
        "op": ">",
        "value": 10,
    }
    inputs.update(changes)

    with pytest.raises(ValidationError):
        NumericCondition.model_validate(inputs)