"""Validated numeric conditions for event selection."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    model_validator,
)

NumericField = Literal[
    "dividend_change_pct",
    "dividend_per_share",
    "buyback_amount_usd",
    "buyback_pct_mcap",
    "mcap_usd",
]

ComparisonOperator = Literal[
    ">",
    ">=",
    "<",
    "<=",
    "==",
    "!=",
]


class NumericCondition(BaseModel):
    """One comparison against an allowed numeric event field."""

    model_config = ConfigDict(extra="forbid")

    field: NumericField
    op: ComparisonOperator
    value: float = Field(strict=True, allow_inf_nan=False)
    
class CategoryCondition(BaseModel):
    """A comparison against an allowed category."""

    model_config = ConfigDict(extra="forbid")

    field: Literal["mcap_bucket", "dividend_kind"]
    op: Literal["==", "!="]
    value: StrictStr

    @model_validator(mode="after")
    def validate_category(self) -> "CategoryCondition":
        allowed = {
            "mcap_bucket": {"small", "mid", "large"},
            "dividend_kind": {"regular", "special", "initial"},
        }
        if self.value not in allowed[self.field]:
            raise ValueError(
                f"{self.field} must use one of "
                f"{sorted(allowed[self.field])}"
            )
        return self


class BooleanCondition(BaseModel):
    """A comparison against a true/false field."""

    model_config = ConfigDict(extra="forbid")

    field: Literal["is_new_program"]
    op: Literal["==", "!="]
    value: StrictBool


Condition = NumericCondition | CategoryCondition | BooleanCondition