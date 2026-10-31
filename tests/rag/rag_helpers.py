"""Shared test data for tests/rag: a tiny corpus, a hashing embedder and an event factory."""

import hashlib
from datetime import date, datetime

import numpy as np
import pandas as pd

from rag.bm25 import tokenize
from store.schema import Event

DIM = 64


class HashEmbedder:
    """Deterministic stand-in for a sentence-embedding model: hashed bag of words."""

    name = "hash-test"

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for token in tokenize(text):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, DIM))

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)


DOCS = [
    # ticker, accession_no, document, filing_date, items, div, buyback, text
    (
        "AAA",
        "0001-16-000001",
        "ex99",
        date(2016, 3, 1),
        "8.01,9.01",
        True,
        False,
        "Item 8.01 Other Events.\n\nAAA Corp declared a quarterly cash dividend of $0.25 per "
        "share, an increase of 10%, reflecting confidence in future cash flow.",
    ),
    (
        "AAA",
        "0001-18-000002",
        "ex99",
        date(2018, 5, 2),
        "8.01",
        False,
        True,
        "AAA Corp announced a $50 million share repurchase program because the board believes "
        "the stock is undervalued.",
    ),
    (
        "BBB",
        "0002-17-000003",
        "ex99",
        date(2017, 7, 3),
        "2.02,9.01",
        False,
        False,
        "BBB Inc reported second quarter revenue of $120 million and raised full-year guidance.",
    ),
    (
        "BBB",
        "0002-21-000004",
        "ex99",
        date(2021, 2, 4),
        "8.01",
        True,
        False,
        "BBB Inc declared a special one-time dividend of $1.00 per share from the asset sale.",
    ),
    (
        "CCC",
        "0003-19-000005",
        "main",
        date(2019, 9, 5),
        "5.02",
        False,
        False,
        "CCC Co announced the appointment of a new chief financial officer.",
    ),
]


def make_docs() -> pd.DataFrame:
    rows = [
        {
            "doc_ref": f"{t}/{acc}/{doc}",
            "ticker": t,
            "company_name": f"{t} Corp",
            "accession_no": acc,
            "document": doc,
            "form_type": "8-K",
            "items": items,
            "filing_date": pd.Timestamp(filed),
            "is_div_candidate": div,
            "is_buyback_candidate": buy,
            "text": text,
        }
        for t, acc, doc, filed, items, div, buy, text in DOCS
    ]
    return pd.DataFrame(rows)


def make_event(event_id, accepted, features, event_type="dividend", **overrides) -> Event:
    fields = {
        "event_id": event_id,
        "event_type": event_type,
        "ticker": "AAA",
        "cik": "0000000001",
        "company_name": "AAA Corp",
        "accession_no": event_id.rsplit("-", 2)[0],
        "form_type": "8-K",
        "items": "8.01",
        "source_url": "https://www.sec.gov/",
        "acceptance_datetime": accepted,
        "first_tradeable_date": accepted.date(),
        "is_new_announcement": True,
        "text_features": features,
        "extraction_status": "passed",
        "extracted_at": datetime(2026, 1, 1),
        "extractor_version": "test",
    }
    return Event(**(fields | overrides))
