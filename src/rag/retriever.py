"""Retrieval tools for the agent, with the time lockout (agent.ports.Retriever).

The harness sets the cutoff, not the agent: no document filed after it, and no text
feature of an event accepted after it, is ever returned. Only events the backtests use
(is_new_announcement and extraction_status "passed") expose text features.
"""

from datetime import date
from typing import Any

from rag.index import HybridIndex, SearchFilters
from store.schema import Event, TextFeature

HIT_COLUMNS = ["doc_ref", "chunk_id", "ticker", "company_name", "accession_no", "items", "text"]


class Retriever:
    def __init__(self, index: HybridIndex, events: list[Event], cutoff: date):
        self.index = index
        self.cutoff = cutoff
        self._features: dict[str, dict[str, TextFeature]] = {
            e.event_id: e.text_features
            for e in events
            if e.is_new_announcement
            and e.extraction_status == "passed"
            and e.acceptance_datetime.date() <= cutoff
        }

    def search(self, query: str, filters: dict[str, Any], k: int) -> list[dict[str, Any]]:
        """The best passage of each of the top k documents, filed on or before the cutoff."""
        f = SearchFilters.model_validate(filters or {})
        f = f.model_copy(update={"end": min(f.end or self.cutoff, self.cutoff)})
        # A document has several chunks: rank all of them, then keep one per document.
        chunks = self.index.search(query, f, k=len(self.index.chunks))
        # Companies refile the same press release or deck: copies of a passage share the
        # best copy's score, and only the earliest filing is kept.
        chunks["passage"] = chunks["text"].str.split().str.join(" ")
        chunks["score"] = chunks.groupby("passage")["score"].transform("max")
        chunks = chunks.sort_values(
            ["score", "filing_date"], ascending=[False, True], kind="stable"
        )
        hits = chunks.drop_duplicates("passage").drop_duplicates("doc_ref").head(k)
        return [
            {
                **{c: row[c] for c in HIT_COLUMNS},
                "filing_date": row["filing_date"].date().isoformat(),
                "score": round(float(row["score"]), 6),
            }
            for row in hits.to_dict(orient="records")
        ]

    def text_feature(self, event_ids: list[str], feature: str) -> list[dict[str, Any]]:
        """One {event_id, label, quote} per event; None where the feature is unavailable."""
        result = []
        for event_id in event_ids:
            value = self._features.get(event_id, {}).get(feature)
            result.append(
                {
                    "event_id": event_id,
                    "label": value.label if value else None,
                    "quote": value.quote if value else None,
                }
            )
        return result

    def available_text_features(self) -> dict[str, list[str]]:
        """Feature name -> labels seen on events visible before the cutoff."""
        labels: dict[str, set[str]] = {}
        for features in self._features.values():
            for name, value in features.items():
                labels.setdefault(name, set()).add(value.label)
        return {name: sorted(values) for name, values in sorted(labels.items())}
