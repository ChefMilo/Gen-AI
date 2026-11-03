"""Select eligible events using validated signal conditions."""

import operator

import numpy as np
import pandas as pd

from dsl.condition import NumericCondition
from dsl.signal import SignalSpec

COMPARISONS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def select_events(
    events: pd.DataFrame,
    spec: SignalSpec,
) -> pd.DataFrame:
    required = {
        "event_id",
        "event_type",
        "is_new_announcement",
        "extraction_status",
    }
    required.update(c.field for c in spec.conditions)

    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    if events["event_id"].isna().any():
        raise ValueError("Event IDs must not be missing")

    if events["event_id"].duplicated().any():
        raise ValueError("Event IDs must be unique")

    mask = (
        events["event_type"].eq(spec.event_type)
        & events["is_new_announcement"].eq(True)
        & events["extraction_status"].eq("passed")
    ).fillna(False)

    for condition in spec.conditions:
        values = events[condition.field]
        valid = values.notna()

        if isinstance(condition, NumericCondition):
            values = pd.to_numeric(values, errors="coerce")
            valid = values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))

        comparison = COMPARISONS[condition.op](
            values,
            condition.value,
        )

        # Missing values never satisfy a condition, including "!=".
        mask &= valid & comparison.fillna(False)

    if spec.text_filter is not None:
        if "text_features" not in events.columns:
            raise ValueError("Missing event column: text_features")

        feature = spec.text_filter.feature

        def get_label(features):
            if not isinstance(features, dict):
                return None

            value = features.get(feature)
            if isinstance(value, dict):
                return value.get("label")

            # Also accepts store.schema.TextFeature objects.
            return getattr(value, "label", None)

        labels = events["text_features"].map(get_label)
        mask &= labels.eq(spec.text_filter.equals).fillna(False)

    return events.loc[mask].copy()
