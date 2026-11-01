"""Research-period event queries, text features and document evidence."""

import copy
import json

import pandas as pd

from agent.finding import EvidenceQuote
from backtest.selection import select_events
from dsl.signal import SignalSpec
from harness.backtester import GuardrailError, HarnessConfig, eastern_day
from rag.index import SearchFilters

FIELD_NOTES = {
    "dividend_change_pct": "number: percentage change versus the previous dividend",
    "dividend_per_share": "number: dividend in USD per share",
    "buyback_amount_usd": "number: authorised buyback amount in USD",
    "buyback_pct_mcap": "number: buyback as a percentage of market capitalisation",
    "mcap_usd": "number: market capitalisation in USD before the event",
    "mcap_bucket": "category: small, mid, large",
    "dividend_kind": "category: regular, special, initial",
    "is_new_program": "bool: whether the buyback is a new program",
}
FEATURES = {"dividend_framing", "buyback_rationale"}


def snapshot_events(events: pd.DataFrame) -> pd.DataFrame:
    """Pandas deep copies alone do not isolate nested evidence dictionaries."""
    result = events.copy(deep=True)
    for name in result.columns:
        if result[name].dtype == object:
            result[name] = result[name].map(copy.deepcopy)
    return result


def _feature(value):
    if isinstance(value, dict):
        return value
    return value.model_dump() if hasattr(value, "model_dump") else {}


def _contains(quote: str, sources: list[str]) -> bool:
    quote = " ".join(quote.split())
    return bool(quote) and any(quote in " ".join(source.split()) for source in sources)


class ResearchEventQuery:
    """A read-only, structured view; no returns or arbitrary raw columns."""

    def __init__(self, events: pd.DataFrame, config: HarnessConfig):
        required = {
            "event_id",
            "ticker",
            "event_type",
            "first_tradeable_date",
            "acceptance_datetime",
            "is_new_announcement",
            "extraction_status",
        }
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"Missing event-query columns: {sorted(missing)}")
        day = pd.to_datetime(events["first_tradeable_date"]).dt.date
        accepted = events["acceptance_datetime"].map(eastern_day)
        visible = (
            day.ge(config.research_start)
            & day.le(config.research_end)
            & accepted.le(config.research_end)
            & events["is_new_announcement"].eq(True)
            & events["extraction_status"].eq("passed")
        ).fillna(False)
        self._events = (
            snapshot_events(events.loc[visible])
            .sort_values(["first_tradeable_date", "ticker", "event_id"], kind="stable")
            .reset_index(drop=True)
        )
        if self._events["event_id"].isna().any() or self._events["event_id"].duplicated().any():
            raise ValueError("Visible event IDs must be present and unique")
        self._config = config

    @property
    def configuration(self) -> HarnessConfig:
        return self._config

    def query(self, event_type: str, conditions: list[dict], limit: int) -> dict:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 20:
            raise GuardrailError("Query limit must be an integer from 0 to 20")
        spec = SignalSpec(event_type=event_type, conditions=conditions)
        missing = {c.field for c in spec.conditions} - set(self._events.columns)
        if missing:
            raise GuardrailError(f"Unavailable condition fields: {sorted(missing)}")
        rows = select_events(self._events, spec)
        columns = ["event_id", "ticker", "first_tradeable_date", *self.available_fields()]
        sample = rows.loc[:, columns].head(limit)
        # Pandas JSON handles missing values without emitting NaN.
        sample_rows = json.loads(sample.to_json(orient="values", date_format="iso"))
        years = pd.to_datetime(rows["first_tradeable_date"]).dt.year.value_counts().sort_index()
        return {
            "n_events": len(rows),
            "by_year": {str(k): int(v) for k, v in years.items()},
            "sample_columns": columns,
            "sample_rows": sample_rows,
        }

    def existing_event_ids(self, event_ids: list[str]) -> set[str]:
        return set(event_ids) & set(self._events["event_id"])

    def available_fields(self) -> dict[str, str]:
        return {name: note for name, note in FIELD_NOTES.items() if name in self._events.columns}

    def text_feature(self, event_ids: list[str], feature: str) -> list[dict]:
        if feature not in FEATURES:
            raise GuardrailError(f"Unknown text feature: {feature}")
        by_id = self._events.set_index("event_id")
        result = []
        for event_id in event_ids:
            features = (
                by_id.at[event_id, "text_features"]
                if event_id in by_id.index and "text_features" in by_id.columns
                else {}
            )
            value = _feature(features.get(feature)) if isinstance(features, dict) else {}
            result.append(
                {"event_id": event_id, "label": value.get("label"), "quote": value.get("quote")}
            )
        return copy.deepcopy(result)

    def available_text_features(self) -> dict[str, list[str]]:
        labels = {}
        for feature in sorted(FEATURES):
            values = self.text_feature(self._events["event_id"].tolist(), feature)
            found = sorted({v["label"] for v in values if v["label"] is not None})
            if found:
                labels[feature] = found
        return labels

    def validate_event_evidence(self, evidence: EvidenceQuote, measured_ids: list[str]) -> None:
        if evidence.event_id is None:
            return
        if evidence.event_id not in measured_ids:
            raise GuardrailError("Evidence event is not in the signal's measured events")
        rows = self._events.loc[self._events["event_id"].eq(evidence.event_id)]
        if rows.empty:
            raise GuardrailError("Evidence event is outside the research view")
        row = rows.iloc[0]
        values = row.get("evidence", {})
        sources = (
            [v for v in values.values() if isinstance(v, str)] if isinstance(values, dict) else []
        )
        features = row.get("text_features", {})
        if isinstance(features, dict):
            sources.extend(
                v["quote"]
                for value in features.values()
                if isinstance((v := _feature(value)).get("quote"), str)
            )
        if not _contains(evidence.quote, sources):
            raise GuardrailError("Evidence quote is not present in the event's stored sources")


class ResearchRetriever:
    """Wrap a retrieval backend with a controller-owned cutoff and evidence ledger.

    Text features come from the same event snapshot as structured queries, not
    from an independently populated backend feature cache.
    """

    def __init__(self, backend, events: ResearchEventQuery):
        self._backend = backend
        self._events = events
        self._documents: dict[str, list[str]] = {}
        self._check_cutoff()

    @property
    def cutoff(self):
        return self._events.configuration.research_end

    def _check_cutoff(self):
        if getattr(self._backend, "cutoff", None) != self.cutoff:
            raise GuardrailError("Retrieval backend cutoff must match the harness cutoff")

    def search(self, query: str, filters: dict, k: int) -> list[dict]:
        self._check_cutoff()
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
            raise GuardrailError("Search k must be an integer from 1 to 10")
        requested = SearchFilters.model_validate(filters or {})
        end = min(requested.end or self.cutoff, self.cutoff)
        if requested.start is not None and requested.start > end:
            return []
        bounded = requested.model_copy(update={"end": end}).model_dump(
            mode="json", exclude_none=True
        )
        hits = self._backend.search(query, bounded, k)
        safe = []
        allowed = {
            "doc_ref",
            "chunk_id",
            "ticker",
            "company_name",
            "accession_no",
            "items",
            "text",
            "filing_date",
            "score",
        }
        for hit in hits:
            if not isinstance(hit, dict) or not isinstance(hit.get("text"), str):
                raise GuardrailError("Malformed retrieval response")
            try:
                day = eastern_day(hit["filing_date"])
            except (KeyError, TypeError, ValueError) as error:
                raise GuardrailError("Retrieval hit has no valid filing date") from error
            if day > end:
                raise GuardrailError("Retrieval backend returned a post-cutoff document")
            if requested.start is not None and day < requested.start:
                raise GuardrailError("Retrieval backend ignored the requested start date")
            if not isinstance(hit.get("doc_ref"), str) or not hit["doc_ref"]:
                raise GuardrailError("Retrieval hit has no document reference")
            safe.append({key: copy.deepcopy(value) for key, value in hit.items() if key in allowed})
        # Commit source passages only after every returned hit passes validation.
        for hit in safe[:k]:
            self._documents.setdefault(hit["doc_ref"], []).append(hit["text"])
        return safe[:k]

    def text_feature(self, event_ids: list[str], feature: str) -> list[dict]:
        return self._events.text_feature(event_ids, feature)

    def available_text_features(self) -> dict[str, list[str]]:
        return self._events.available_text_features()

    def validate_document_evidence(self, evidence: EvidenceQuote) -> None:
        if evidence.doc_ref is not None and not _contains(
            evidence.quote, self._documents.get(evidence.doc_ref, [])
        ):
            raise GuardrailError("Document evidence must quote a passage retrieved in this session")
