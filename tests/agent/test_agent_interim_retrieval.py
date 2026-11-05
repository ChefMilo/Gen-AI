"""Retriever selection and wrappers, date masking, and grounding of masked quotes."""

import hashlib
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import agent.session as session_module
from agent.interim.data import research_events
from agent.interim.retrieval import (
    TEXT_CHARS,
    FakeTextRetriever,
    RagRetrieverAdapter,
    make_retriever,
)
from agent.session import new_session
from agent.textmask import MASK, mask_dates
from agent.tools import ToolError, dispatch
from rag.index import HybridIndex
from rag.retriever import Retriever
from store.schema import Event, TextFeature

# --- Date masking ---------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "January 22, 2015", "Jan. 22, 2015", "Jan 22, 2015", "22 January 2015", "2015-01-22",
    "1/22/2015", "January 2015", "Sept. 3rd, 2019", "February 05",
])
def test_each_date_format_is_masked_whole(text):
    assert mask_dates(f"declared on {text}.") == f"declared on {MASK}."


def test_text_around_dates_is_kept():
    text = "Record date January 22, 2015; payable on February 05, 2015 to holders."
    assert mask_dates(text) == f"Record date {MASK}; payable on {MASK} to holders."


@pytest.mark.parametrize("text", [
    "we may 2015", "Item 8.01 and 9.01", "a 10% increase", "May 5% growth", "$0.25 per share",
])
def test_non_dates_are_left_alone(text):
    assert mask_dates(text) == text


# --- A tiny real index ----------------------------------------------------------------

DIM = 32


class HashEmbedder:
    name = "hash-agent-test"

    def _vector(self, text):
        v = np.zeros(DIM, dtype=np.float32)
        for token in text.lower().split():
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts):
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, DIM))

    def encode_query(self, text):
        return self._vector(text)


LONG = "The board cited strong cash flow. " * 20
DOCS = [  # ticker, accession, filed, dividend?, buyback?, text
    ("AAA", "0001-16-000001", date(2016, 3, 1), True, False,
     "AAA declared a quarterly cash dividend of $0.25 per share, an increase of 10%, "
     "payable on April 15, 2016 to holders of record on March 31, 2016. " + LONG),
    ("AAA", "0001-18-000002", date(2018, 5, 2), False, True,
     "AAA announced a $50 million share repurchase program; the stock is undervalued."),
    ("BBB", "0002-17-000003", date(2017, 7, 3), True, False,
     "BBB declared a special one-time dividend of $1.00 per share on 7/3/2017."),
    ("BBB", "0002-21-000004", date(2021, 2, 4), True, False,
     "BBB raised its dividend by 20% on February 4, 2021."),
]


def make_index():
    docs = pd.DataFrame([{
        "doc_ref": f"{t}/{acc}/ex99", "ticker": t, "company_name": f"{t} Corp",
        "accession_no": acc, "document": "ex99", "form_type": "8-K", "items": "8.01",
        "filing_date": pd.Timestamp(filed), "is_div_candidate": div,
        "is_buyback_candidate": buy, "text": text,
    } for t, acc, filed, div, buy, text in DOCS])
    return HybridIndex.build(docs, HashEmbedder(), max_chars=1500, candidates=20, rrf_k=60)


def make_events():
    rows = []
    for t, acc, filed, div, _, _ in DOCS:
        event_type = "dividend" if div else "buyback"
        feature = ("dividend_framing", "confidence") if div else ("buyback_rationale",
                                                                    "undervalued")
        accepted = datetime.combine(filed, datetime.min.time()).replace(hour=9)
        rows.append(Event(
            event_id=f"{acc}-{event_type}-1", event_type=event_type, ticker=t,
            cik="0000000001", company_name=f"{t} Corp", accession_no=acc, form_type="8-K",
            items="8.01", source_url="https://www.sec.gov/", acceptance_datetime=accepted,
            first_tradeable_date=filed, is_new_announcement=True,
            text_features={feature[0]: TextFeature(
                label=feature[1], quote=f"as of {filed:%B} {filed.day}, {filed.year}")},
            extraction_status="passed", extracted_at=datetime(2026, 1, 1),
            extractor_version="test",
        ).model_dump())
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def index():
    return make_index()


def rag_for(index, end=None, mask=True):
    events = research_events(make_events(), None, end)
    built = {}

    def factory(rows, cutoff):
        built["cutoff"] = cutoff
        from agent.interim.retrieval import as_events
        return Retriever(index, as_events(rows), cutoff)

    retriever, kind = make_retriever(events, end, mask, rag_factory=factory)
    return retriever, kind, built


def test_real_events_use_the_rag_wrapper_with_cutoff_research_end(index):
    retriever, kind, built = rag_for(index, end=date(2019, 12, 31))
    assert kind == "rag" and isinstance(retriever, RagRetrieverAdapter)
    assert built["cutoff"] == date(2019, 12, 31)
    hits = retriever.search("dividend", {}, 5)
    assert hits and all(h["filing_date"] <= "2019-12-31" for h in hits)


def test_open_end_cutoff_is_the_last_event_date(index):
    _, _, built = rag_for(index, end=None)
    assert built["cutoff"] == date(2021, 2, 4)


def test_rag_hits_are_compact_masked_clipped_and_carry_event_ids(index):
    retriever, _, _ = rag_for(index)
    hits = retriever.search("quarterly cash dividend increase", {"ticker": "AAA"}, 3)
    assert set(hits[0]) == {"doc_ref", "ticker", "filing_date", "event_ids", "text"}
    top = hits[0]
    assert top["doc_ref"] == "AAA/0001-16-000001/ex99"
    assert top["event_ids"] == ["0001-16-000001-dividend-1"]
    assert "April 15, 2016" not in top["text"] and MASK in top["text"]
    assert len(top["text"]) <= TEXT_CHARS + 4 and top["text"].endswith(" ...")


def test_filters_map_to_search_filters(index):
    retriever, _, _ = rag_for(index)
    hits = retriever.search("dividend", {"ticker": "BBB", "event_type": "dividend",
                                         "start": date(2018, 1, 1), "end": None}, 5)
    assert [h["doc_ref"] for h in hits] == ["BBB/0002-21-000004/ex99"]
    with pytest.raises(ToolError, match="allowed"):
        retriever.search("dividend", {"tickers": ["BBB"]}, 3)
    with pytest.raises(ToolError, match="allowed"):
        retriever.search("dividend", {"event_type": "rights_issue"}, 3)


def test_rag_text_features_mask_quotes_and_keep_missing_as_none(index):
    retriever, _, _ = rag_for(index)
    rows = retriever.text_feature(["0001-16-000001-dividend-1", "nope"], "dividend_framing")
    assert rows[0] == {"event_id": "0001-16-000001-dividend-1", "label": "confidence",
                       "quote": f"as of {MASK}"}
    assert rows[1] == {"event_id": "nope", "label": None, "quote": None}
    with pytest.raises(ToolError, match="Available"):
        retriever.text_feature(["x"], "made_up")


def test_masking_can_be_switched_off(index):
    retriever, _, _ = rag_for(index, mask=False)
    hits = retriever.search("special one-time dividend", {"ticker": "BBB"}, 1)
    assert "7/3/2017" in hits[0]["text"]


# --- Fake events: FakeTextRetriever over the view ------------------------------------


@pytest.fixture(scope="module")
def fake_events():
    from universes.fake_events import make_fake_events
    from universes.synthetic import make_synthetic_prices

    prices = make_synthetic_prices([f"T{i:03d}" for i in range(15)], "2016-01-01",
                                   "2018-12-31", seed=1)
    return research_events(make_fake_events(prices, seed=1), None, None)


def test_fake_events_use_the_fake_retriever(fake_events):
    retriever, kind = make_retriever(fake_events, None, True)
    assert kind == "fake" and isinstance(retriever, FakeTextRetriever)
    hits = retriever.search("dividend record payable", {"event_type": "dividend"}, 5)
    assert len(hits) == 5
    assert all(set(h) == {"doc_ref", "ticker", "filing_date", "event_ids", "text"}
               for h in hits)
    assert all("2016" not in h["text"] and "2017" not in h["text"] for h in hits)
    late = retriever.search("dividend", {"start": date(2018, 1, 1)}, 5)
    assert late and all(h["filing_date"] >= "2018-01-01" for h in late)


def test_fake_text_features_use_the_labels_in_the_data(fake_events):
    retriever, _ = make_retriever(fake_events, None, True)
    features = retriever.available_text_features()
    assert set(features["dividend_framing"]) == {"routine", "growth", "cautious"}  # no "fake_"
    buyback = fake_events.loc[fake_events["event_type"] == "buyback", "event_id"].iloc[0]
    assert retriever.text_feature([buyback], "dividend_framing")[0]["label"] is None


# --- Grounding of masked quotes -------------------------------------------------------


def test_a_quote_of_masked_text_grounds_and_unmasked_dates_are_masked_first(
        fake_events, registry, monkeypatch):
    from agent.interim.ports import build_ports

    monkeypatch.setattr(session_module, "code_version", lambda: None)
    view = type("V", (), {})()
    view.events = fake_events
    view.prices = pd.DataFrame(columns=["date", "ticker", "ret", "is_bad_price"])
    view.benchmarks = pd.DataFrame(columns=["date", "ticker", "ret"])
    view.research_start = view.research_end = None
    ports = build_ports(view, budget=3, holding_days=5, min_events=20, with_retriever=True,
                        mask_dates=True)
    config = {"research_start": date(2016, 1, 1),
              "agent": {"model": "m", "mask_dates_in_text": True}}
    s = new_session("R3", config, "u", ports.backtester, ports.events, retriever=ports.retriever,
                    research_end=date(2018, 12, 31), in_memory=True)
    content, _ = dispatch(s, "search_announcements",
                          {"query": "shareholders of record payable", "k": 1})
    hit = json.loads(content)["results"][0]
    masked = hit["text"]
    assert MASK in masked
    original = masked.replace(MASK, "January 22, 2016", 1)
    s.signals["sig-1"], s.tested_signals = {}, {"sig-1"}
    content, is_error = dispatch(s, "report_finding", {
        "signal_id": "sig-1", "rationale": "r",
        "evidence": [{"doc_ref": hit["doc_ref"], "quote": masked},
                     {"doc_ref": hit["doc_ref"], "quote": original},
                     {"doc_ref": hit["doc_ref"], "quote": "never shown to the model"}]})
    assert not is_error, content
    claim = next(e for e in s.trace.events if e.get("entry_kind") == "claim")
    assert [q["grounded"] for q in claim["data"]["evidence"]] == [True, True, False]
