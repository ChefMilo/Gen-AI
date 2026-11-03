"""Validated event-study signal definitions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsl.condition import Condition
from store.schema import DEFAULT_BENCHMARK, Benchmark, EventType


class TextFilter(BaseModel):
    """Select a text-feature label supplied by the retrieval layer."""

    model_config = ConfigDict(extra="forbid")

    feature: Literal["dividend_framing", "buyback_rationale"]
    equals: str = Field(min_length=1, max_length=100)


class SignalSpec(BaseModel):
    """Instructions for selecting events and measuring their returns."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    benchmark: Benchmark = DEFAULT_BENCHMARK
    conditions: list[Condition] = Field(
        default_factory=list,
        max_length=5,
    )
    text_filter: TextFilter | None = None
    holding_days: int = Field(
        default=5,
        strict=True,
        ge=1,
        le=60,
    )
    direction: Literal["long", "short"] = "long"
    entry: Literal["first_tradeable_date"] = "first_tradeable_date"

    @model_validator(mode="after")
    def validate_event_fields(self) -> "SignalSpec":
        dividend_fields = {
            "dividend_change_pct",
            "dividend_per_share",
            "dividend_kind",
        }
        buyback_fields = {
            "buyback_amount_usd",
            "buyback_pct_mcap",
            "is_new_program",
        }

        forbidden = (
            buyback_fields
            if self.event_type == "dividend"
            else dividend_fields
        )

        for condition in self.conditions:
            if condition.field in forbidden:
                raise ValueError(
                    f"{condition.field} does not apply to "
                    f"{self.event_type} events"
                )

        if self.text_filter is not None:
            expected = {
                "dividend": "dividend_framing",
                "buyback": "buyback_rationale",
            }[self.event_type]

            if self.text_filter.feature != expected:
                raise ValueError(
                    f"{self.event_type} events require "
                    f"text feature {expected}"
                )

        return self