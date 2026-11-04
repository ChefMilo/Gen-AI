"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

Signal spec models built from a field table, until Member 4's SignalSpec can hold
conditions. The fakes (agent.fakes) and the interim ports both build theirs here, so a
condition is validated the same way everywhere.

A field table maps field name -> (event type it applies to, or None for both; kind;
allowed values or None). Kinds: number, category, bool. A text-feature table maps
feature name -> (event type, allowed labels).
"""

from typing import Any, Literal, NamedTuple

import pandas as pd
from pydantic import Field, model_validator

from agent.tools import ToolArgs

FieldTable = dict[str, tuple[str | None, str, tuple | None]]
TextFeatureTable = dict[str, tuple[str, tuple[str, ...]]]

OPS_BY_KIND = {
    "number": ("==", "!=", ">", ">=", "<", "<=", "in"),
    "category": ("==", "!=", "in"),
    "bool": ("==", "!="),
}
DIRECTIONS = ("long", "short")
MAX_CONDITIONS = 5


class SpecModels(NamedTuple):
    condition: type[ToolArgs]
    text_filter: type[ToolArgs] | None
    signal_spec: type[ToolArgs]


def make_spec_models(
    fields: FieldTable,
    text_features: TextFeatureTable,
    event_types: tuple[str, ...],
    holding_days: tuple[int, ...],
    prefix: str = "",
) -> SpecModels:
    """Condition, text filter (None without text features) and signal spec models.

    prefix names the classes (e.g. "Fake"), which only shows in JSON schema titles.
    """
    type_options, day_options = tuple(event_types), tuple(holding_days)

    class Condition(ToolArgs):
        field: Literal[tuple(fields)]  # type: ignore[valid-type]
        op: Literal["==", "!=", ">", ">=", "<", "<=", "in"]
        value: bool | float | str | list[bool | float | str]

        @model_validator(mode="after")
        def _check_value(self) -> "Condition":
            _, kind, allowed = fields[self.field]
            values = self.value if isinstance(self.value, list) else [self.value]
            if (self.op == "in") != isinstance(self.value, list):
                raise ValueError("use a list value with 'in', and a single value otherwise")
            if self.op not in OPS_BY_KIND[kind]:
                ops = " ".join(OPS_BY_KIND[kind])
                raise ValueError(f"{self.field} is a {kind} field; use one of: {ops}")
            if kind == "number" and any(isinstance(v, (bool, str)) for v in values):
                raise ValueError(f"{self.field} is a number field; value must be a number")
            if kind == "bool" and not all(isinstance(v, bool) for v in values):
                raise ValueError(f"{self.field} is a bool field; value must be true or false")
            if allowed and any(v not in allowed for v in values):
                raise ValueError(f"{self.field} must be one of {list(allowed)}")
            return self

    filter_model = None
    if text_features:

        class TextFilter(ToolArgs):
            feature: Literal[tuple(text_features)]  # type: ignore[valid-type]
            equals: str

            @model_validator(mode="after")
            def _check_label(self) -> "TextFilter":
                labels = text_features[self.feature][1]
                if self.equals not in labels:
                    raise ValueError(f"{self.feature} labels are {list(labels)}")
                return self

        filter_model = TextFilter

    class SignalSpec(ToolArgs):
        event_type: Literal[type_options]  # type: ignore[valid-type]
        conditions: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
        if filter_model is not None:
            text_filter: filter_model | None = None  # type: ignore[valid-type]
        holding_days: Literal[day_options] = day_options[0]  # type: ignore[valid-type]
        direction: Literal[DIRECTIONS] = "long"  # type: ignore[valid-type]

        @model_validator(mode="after")
        def _fields_match_event_type(self) -> "SignalSpec":
            for c in self.conditions:
                applies_to = fields[c.field][0]
                if applies_to not in (None, self.event_type):
                    raise ValueError(f"{c.field} only applies to {applies_to} events")
            tf = getattr(self, "text_filter", None)
            if tf and text_features[tf.feature][0] != self.event_type:
                raise ValueError(f"{tf.feature} only applies to "
                                 f"{text_features[tf.feature][0]} events")
            return self

    for cls, name in ((Condition, "Condition"), (filter_model, "TextFilter"),
                      (SignalSpec, "SignalSpec")):
        if cls is not None:
            cls.__name__ = cls.__qualname__ = f"{prefix}{name}"
    return SpecModels(Condition, filter_model, SignalSpec)


def select(events: pd.DataFrame, spec: Any) -> pd.DataFrame:
    """Events of spec.event_type matching every condition and the text filter, if any.

    Text filters read the flattened <feature> label column (agent.interim.data).
    A missing value (NaN) never matches, as in R0 (NaN > 10 is False).
    """
    mask = events["event_type"] == spec.event_type
    for c in spec.conditions:
        col = events[c.field]
        if c.op == "in":
            mask &= col.isin(c.value)
        else:
            mask &= {
                "==": col == c.value, "!=": col != c.value, ">": col > c.value,
                ">=": col >= c.value, "<": col < c.value, "<=": col <= c.value,
            }[c.op]
    text_filter = getattr(spec, "text_filter", None)
    if text_filter:
        mask &= events[text_filter.feature] == text_filter.equals
    return events[mask]
