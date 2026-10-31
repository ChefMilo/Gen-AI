"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

InterimEventQuery: the query_events port over a universe view's events, behaving like
agent.fakes.FakeEventQuery (counts, counts by year, a compact sample). It stands in for
Member 1's event store. Samples show only identifiers, the date and the condition
fields asked about, so no outcome or bookkeeping column ever reaches the model.
"""

from typing import Any

import pandas as pd

from agent.fakes import _compact
from agent.interim.data import ID_COLUMNS, field_descriptions
from agent.interim.spec import FieldTable, select


class InterimEventQuery:
    def __init__(self, events: pd.DataFrame, fields: FieldTable, signal_spec: type):
        """events: already cut to usable rows in the research period (research_events)."""
        self.events = events
        self.fields = fields
        self.signal_spec = signal_spec

    def query(self, event_type: str, conditions: list[dict], limit: int) -> dict[str, Any]:
        spec = self.signal_spec(event_type=event_type, conditions=conditions)
        rows = select(self.events, spec)
        cols = list(dict.fromkeys([*ID_COLUMNS, *[c.field for c in spec.conditions]]))
        years = pd.to_datetime(rows["first_tradeable_date"]).dt.year.value_counts().sort_index()
        sample = rows[cols].head(limit)
        return {
            "n_events": len(rows),
            "by_year": {str(k): int(v) for k, v in years.items()},
            "sample_columns": cols,
            "sample_rows": [[_compact(v) for v in row] for row in sample.itertuples(index=False)],
        }

    def existing_event_ids(self, event_ids: list[str]) -> set[str]:
        return set(event_ids) & set(self.events["event_id"])

    def available_fields(self) -> dict[str, str]:
        return field_descriptions(self.fields)
