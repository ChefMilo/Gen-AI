from datetime import date, datetime

import pandas as pd
import pytest
from rag_helpers import make_event

from rag.index import HybridIndex
from rag.retriever import Retriever
from store.schema import TextFeature

CUTOFF = date(2019, 12, 31)


GROWTH = TextFeature(label="increase_confidence", quote="reflecting confidence")
ONE_OFF = TextFeature(label="one_off", quote="special one-time dividend")
EVENTS = [
    make_event("0001-16-000001-dividend-1", datetime(2016, 3, 1, 9), {"dividend_framing": GROWTH}),
    make_event("0002-21-000004-dividend-1", datetime(2021, 2, 4, 9), {"dividend_framing": ONE_OFF}),
    make_event(
        "0009-17-000009-dividend-1",
        datetime(2017, 1, 1, 9),
        {"dividend_framing": ONE_OFF},
        extraction_status="flagged",
    ),
]


@pytest.fixture
def retriever(docs, embedder):
    index = HybridIndex.build(docs, embedder, max_chars=500, candidates=50, rrf_k=60)
    return Retriever(index, EVENTS, cutoff=CUTOFF)


def test_search_never_returns_documents_after_the_cutoff(retriever):
    for filters in ({}, {"end": "2025-01-01"}, {"start": "2020-01-01"}):
        for hit in retriever.search("special one-time dividend", filters, k=10):
            assert date.fromisoformat(hit["filing_date"]) <= CUTOFF


def test_search_returns_one_passage_per_document_with_a_doc_ref(retriever):
    hits = retriever.search("dividend", {}, k=10)
    refs = [h["doc_ref"] for h in hits]
    assert len(refs) == len(set(refs)) > 0
    assert set(hits[0]) == {
        "doc_ref",
        "chunk_id",
        "ticker",
        "company_name",
        "accession_no",
        "filing_date",
        "items",
        "text",
        "score",
    }


def test_search_drops_refiled_copies_and_keeps_the_earliest(docs, embedder):
    original = docs[docs["accession_no"] == "0001-18-000002"].iloc[0]
    refiled = original.copy()
    refiled["accession_no"], refiled["doc_ref"] = "0001-18-000099", "AAA/0001-18-000099/ex99"
    refiled["filing_date"] = original["filing_date"] + pd.Timedelta(days=30)
    refiled["text"] = original["text"].replace(" ", "  ")  # whitespace differences only
    docs = pd.concat([pd.DataFrame([refiled]), docs], ignore_index=True)
    index = HybridIndex.build(docs, embedder, max_chars=500, candidates=50, rrf_k=60)

    hits = Retriever(index, [], cutoff=CUTOFF).search("share repurchase undervalued", {}, k=5)

    refs = [h["doc_ref"] for h in hits]
    assert original["doc_ref"] in refs and refiled["doc_ref"] not in refs


def test_search_respects_k_and_rejects_unknown_filters(retriever):
    assert len(retriever.search("dividend", {}, k=1)) == 1
    with pytest.raises(ValueError):
        retriever.search("dividend", {"sector": "Industrials"}, k=5)


def test_text_feature_hides_events_after_the_cutoff_and_unusable_events(retriever):
    ids = [e.event_id for e in EVENTS] + ["unknown"]
    result = retriever.text_feature(ids, "dividend_framing")
    assert [r["event_id"] for r in result] == ids
    assert result[0] == {
        "event_id": ids[0],
        "label": "increase_confidence",
        "quote": "reflecting confidence",
    }
    assert all(r["label"] is None and r["quote"] is None for r in result[1:])


def test_available_text_features_come_from_visible_events(retriever):
    assert retriever.available_text_features() == {"dividend_framing": ["increase_confidence"]}
