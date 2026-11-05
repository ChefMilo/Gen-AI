"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

Retriever selection and wrappers for R3 / R3-U on a universe view.

- Fake events (universes.fake_events, accession numbers with the FAKE_FILER prefix):
  FakeTextRetriever, over the view's own events (their fake evidence and text-feature
  quotes). The real 8-K index shares no filings with fake events.
- Real events: rag.Retriever (Member 2) with cutoff = research_end, or the last
  first_tradeable_date in the view when research_end is None, wrapped by
  RagRetrieverAdapter.

Both wrappers return the same compact, model-facing shapes:
- search hits {doc_ref, ticker, filing_date, event_ids, text}, event_ids being the
  view's usable events of that filing (by accession_no), text clipped to TEXT_CHARS;
- text features: one {event_id, label, quote} per requested event, label None where the
  event has no such feature.
With mask_dates (agent.mask_dates_in_text) every passage and quote has its dates
replaced by [DATE] (agent.textmask; PR #6, placebo date leak) before clipping.
"""

from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd
from pydantic import ValidationError

from agent.fakes import FakeRetriever
from agent.interim.data import (
    display_text,
    in_period,
    is_fake,
    last_event_date,
    text_feature_table,
)
from agent.textmask import maybe_mask
from agent.tools import ToolError
from store.schema import Event

TEXT_CHARS = 400
FILTER_KEYS = ("ticker", "event_type", "start", "end")


def clip(text: str, limit: int = TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def _event_ids_by_accession(events: pd.DataFrame) -> dict[str, list[str]]:
    return {acc: sorted(g["event_id"]) for acc, g in events.groupby("accession_no")}


def _check_filters(filters: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(filters) - set(FILTER_KEYS))
    if unknown:
        raise ToolError(f"unknown search filters {unknown}; allowed: {list(FILTER_KEYS)}")
    return {k: v for k, v in filters.items() if v is not None}


class _CompactRetriever:
    """Shared output shaping and masking."""

    def __init__(self, events: pd.DataFrame, mask_dates: bool):
        self.events = events
        self.mask_dates = mask_dates
        self._by_accession = _event_ids_by_accession(events)

    def _hit(self, doc_ref: str, ticker: str, accession_no: str, filing_date, text: str) -> dict:
        return {
            "doc_ref": doc_ref,
            "ticker": ticker,
            "filing_date": pd.Timestamp(filing_date).date().isoformat(),
            "event_ids": self._by_accession.get(accession_no, []),
            "text": clip(maybe_mask(text, self.mask_dates)),
        }

    def _feature_rows(self, event_ids: list[str], found: list[dict]) -> list[dict]:
        by_id = {v["event_id"]: v for v in found}
        out = []
        for event_id in event_ids:
            value = by_id.get(event_id) or {}
            out.append({"event_id": event_id, "label": value.get("label"),
                        "quote": maybe_mask(value.get("quote"), self.mask_dates)})
        return out


class FakeTextRetriever(_CompactRetriever):
    """agent.fakes.FakeRetriever over a view's fake events (already research-period only)."""

    def __init__(self, events: pd.DataFrame, mask_dates: bool):
        super().__init__(events, mask_dates)
        self.inner = FakeRetriever(events, research_end=None)
        self._rows = events.set_index("event_id")
        self._features = text_feature_table(events)

    def search(self, query: str, filters: dict[str, Any], k: int) -> list[dict[str, Any]]:
        f = _check_filters(filters)
        hits = self.inner.search(query, {key: f.get(key) for key in ("ticker", "event_type")},
                                 k=len(self.events) * 10)
        if "start" in f or "end" in f:
            days = pd.Series([h["date"] for h in hits], dtype=object)
            keep = in_period(days, f.get("start"), f.get("end")).tolist() if hits else []
            hits = [h for h, ok in zip(hits, keep, strict=True) if ok]
        out = []
        for h in hits[:k]:
            row = self._rows.loc[h["event_id"]]
            out.append(self._hit(h["doc_ref"], row["ticker"], row["accession_no"], h["date"],
                                 display_text(h["text"])))
        return out

    def text_feature(self, event_ids: list[str], feature: str) -> list[dict[str, Any]]:
        if feature not in self._features:
            raise ToolError(f"unknown feature {feature!r}. Available: {list(self._features)}")
        return self._feature_rows(event_ids, self.inner.text_feature(event_ids, feature))

    def available_text_features(self) -> dict[str, list[str]]:
        return {name: list(labels) for name, (_, labels) in self._features.items()}


class RagRetrieverAdapter(_CompactRetriever):
    """Member 2's rag.retriever.Retriever behind the agent's Retriever port."""

    def __init__(self, retriever: Any, events: pd.DataFrame, mask_dates: bool):
        super().__init__(events, mask_dates)
        self.inner = retriever

    def search(self, query: str, filters: dict[str, Any], k: int) -> list[dict[str, Any]]:
        from rag.index import SearchFilters

        f = _check_filters(filters)
        mapped = {"tickers": [f["ticker"]] if "ticker" in f else None,
                  "event_type": f.get("event_type"), "start": f.get("start"),
                  "end": f.get("end")}
        try:
            search_filters = SearchFilters.model_validate(
                {key: v for key, v in mapped.items() if v is not None})
        except ValidationError as e:
            raise ToolError(f"bad search filters ({e.error_count()} errors); allowed: "
                            f"{list(FILTER_KEYS)}, event_type one of dividend/buyback, "
                            "dates as YYYY-MM-DD") from e
        hits = self.inner.search(query, search_filters.model_dump(exclude_none=True), k)
        return [self._hit(h["doc_ref"], h["ticker"], h["accession_no"], h["filing_date"],
                          h["text"]) for h in hits]

    def text_feature(self, event_ids: list[str], feature: str) -> list[dict[str, Any]]:
        available = self.inner.available_text_features()
        if feature not in available:
            raise ToolError(f"unknown feature {feature!r}. Available: {list(available)}")
        return self._feature_rows(event_ids, self.inner.text_feature(event_ids, feature))

    def available_text_features(self) -> dict[str, list[str]]:
        return self.inner.available_text_features()


RagFactory = Callable[[pd.DataFrame, date], Any]  # (events, cutoff) -> rag.Retriever


def as_events(events: pd.DataFrame) -> list[Event]:
    """Rows as store.schema.Event (only the Event columns; NaN becomes None)."""
    from universes.fake_events import rows_as_events

    return rows_as_events(events[[c for c in Event.model_fields if c in events.columns]])


def shared_index_retriever(events: pd.DataFrame, cutoff: date) -> Any:
    """rag.Retriever on the shared index in results/rag_index/ (README, "Retrieval index").
    Needs the [rag] extra and the downloaded index."""
    from rag.build import load_index
    from rag.config import load_rag_config
    from rag.retriever import Retriever

    return Retriever(load_index(load_rag_config()), as_events(events), cutoff)


def retriever_cutoff(events: pd.DataFrame, research_end: date | None) -> date | None:
    return research_end if research_end is not None else last_event_date(events)


def make_retriever(
    events: pd.DataFrame,
    research_end: date | None,
    mask_dates: bool,
    rag_factory: RagFactory = shared_index_retriever,
) -> tuple[Any, str]:
    """(retriever, kind) for a view's research-period events; kind is "fake" or "rag"."""
    if is_fake(events):
        return FakeTextRetriever(events, mask_dates), "fake"
    cutoff = retriever_cutoff(events, research_end)
    return RagRetrieverAdapter(rag_factory(events, cutoff), events, mask_dates), "rag"
