"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

What the agent may see of a universe: usable events in the research period, prices and
benchmarks up to research_end, and the field and text-feature tables built from them.

Time lockout: events need research_start <= first_tradeable_date <= research_end, and
price and benchmark rows after research_end are dropped. A None date leaves that side
open (the runner's restrict_to_period does the same), so with research_end unset the
interim lockout cuts nothing. Prices before research_start stay, as look-back history.
"""

from datetime import date

import pandas as pd

from agent.fakes import FIELD_NOTES, FIELDS
from agent.interim.spec import FieldTable, TextFeatureTable
from universes.fake_events import FAKE_FILER, FAKE_QUOTE_PREFIX
from universes.planted import eligible

BREAKDOWN_COLUMNS = ("sector", "mcap_bucket")  # each offered when the events have it
# Columns a model-facing result may show: identifiers, the date and condition fields.
# Anything else (prices, CARs, placebo bookkeeping, raw quotes) never leaves the port.
ID_COLUMNS = ("event_id", "ticker", "first_tradeable_date")


# Fake events mark their text: quotes start with "[FAKE]" and labels with "fake_". The
# marks are the same on every universe type, but the agent must never see words like
# "fake", so they are stripped from everything model-facing (labels, quotes, passages).
FAKE_LABEL_PREFIX = "fake_"


def display_label(label: str | None) -> str | None:
    return label.removeprefix(FAKE_LABEL_PREFIX) if isinstance(label, str) else label


def display_text(text: str | None) -> str | None:
    if not isinstance(text, str):
        return text
    return text.removeprefix(FAKE_QUOTE_PREFIX).removeprefix(" ")


def as_date(value) -> date | None:
    if value is None or (not isinstance(value, date) and pd.isna(value)):
        return None
    return pd.Timestamp(value).date()


def in_period(days: pd.Series, start: date | None, end: date | None) -> pd.Series:
    days = pd.to_datetime(days)
    keep = pd.Series(True, index=days.index)
    if start is not None:
        keep &= days >= pd.Timestamp(start)
    if end is not None:
        keep &= days <= pd.Timestamp(end)
    return keep


def research_events(events: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    """Usable events (docs/data_schema.md: new announcements that passed extraction) in
    the research period, with each text feature flattened into <feature> (label) and
    <feature>_quote columns (labels and quotes as display_label / display_text show them;
    the nested text_features column keeps the originals)."""
    rows = eligible(events)
    rows = rows[in_period(rows["first_tradeable_date"], start, end)].reset_index(drop=True)
    features = sorted({name for tf in rows["text_features"] for name in (tf or {})})
    for feature in features:
        values = [(tf or {}).get(feature) or {} for tf in rows["text_features"]]
        # object dtype keeps None for missing values (pandas 3 would store NaN in a str column)
        rows[feature] = pd.Series([display_label(v.get("label")) for v in values],
                                  index=rows.index, dtype=object)
        rows[f"{feature}_quote"] = pd.Series([display_text(v.get("quote")) for v in values],
                                             index=rows.index, dtype=object)
    return rows


def prices_until(prices: pd.DataFrame, end: date | None) -> pd.DataFrame:
    if end is None:
        return prices
    return prices[pd.to_datetime(prices["date"]) <= pd.Timestamp(end)].reset_index(drop=True)


def last_event_date(events: pd.DataFrame) -> date | None:
    """The latest first_tradeable_date, the retriever cutoff when research_end is None."""
    if events.empty:
        return None
    return as_date(pd.to_datetime(events["first_tradeable_date"]).max())


def is_fake(events: pd.DataFrame) -> bool:
    """Events made by universes.fake_events: every accession_no has the FAKE_FILER prefix."""
    accessions = events["accession_no"].astype(str)
    return len(accessions) > 0 and bool(accessions.str.startswith(FAKE_FILER).all())


def field_table(events: pd.DataFrame) -> FieldTable:
    """The agent's condition fields that these events actually have (no sector column on
    real Event rows, so sector only appears when a column exists)."""
    return {name: spec for name, spec in FIELDS.items() if name in events.columns}


def field_descriptions(fields: FieldTable) -> dict[str, str]:
    out = {}
    for name, (applies_to, kind, allowed) in fields.items():
        parts = [kind, f"{applies_to} events" if applies_to else "all events"]
        if allowed:
            parts.append("one of " + "/".join(allowed))
        if name in FIELD_NOTES:
            parts.append(FIELD_NOTES[name])
        out[name] = "; ".join(parts)
    return out


def text_feature_table(events: pd.DataFrame) -> TextFeatureTable:
    """feature -> (event type, labels seen on these events), sorted. Labels come from the
    data, as rag.Retriever.available_text_features does."""
    out: dict[str, tuple[str, set]] = {}
    for event_type, features in zip(events["event_type"], events["text_features"],
                                    strict=True):
        for name, value in (features or {}).items():
            if value and value.get("label") is not None:
                out.setdefault(name, (event_type, set()))[1].add(display_label(value["label"]))
    return {name: (t, tuple(sorted(labels))) for name, (t, labels) in sorted(out.items())}


def breakdowns(events: pd.DataFrame) -> tuple[str, ...]:
    return ("year", *[c for c in BREAKDOWN_COLUMNS if c in events.columns])
