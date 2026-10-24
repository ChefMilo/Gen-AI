"""Hybrid chunk index: BM25 and dense vectors, fused by reciprocal rank (RRF).

search() returns ranked chunks; the agent-facing Retriever (rag.retriever) adds the time
lockout and keeps the best chunk per document.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from rag.bm25 import BM25
from rag.chunking import chunk_documents
from rag.embed import Embedder
from store.schema import EventType

CHUNKS_FILE = "chunks.parquet"
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"

# Until events.parquet lands, event_type filters use the filing index's keyword flags,
# and keep only the passages of those filings that mention the event type.
EVENT_TYPE_FLAG: dict[str, str] = {
    "dividend": "is_div_candidate",
    "buyback": "is_buyback_candidate",
}
EVENT_TYPE_WORDS: dict[str, str] = {
    "dividend": r"\bdividend",
    "buyback": r"\brepurchas|\bbuy[- ]?backs?\b",
}


class SearchFilters(BaseModel):
    """Metadata filters; every one given must match. Dates are filing dates, inclusive."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] | None = None
    start: date | None = None
    end: date | None = None
    event_type: EventType | None = None
    items: list[str] | None = None  # any of these 8-K items, e.g. ["8.01"]
    accession_nos: list[str] | None = None


class HybridIndex:
    def __init__(
        self,
        chunks: pd.DataFrame,
        embeddings: np.ndarray,
        embedder: Embedder,
        candidates: int,
        rrf_k: int,
    ):
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        self.chunks = chunks.reset_index(drop=True).astype({"filing_date": "datetime64[ns]"})
        self.embeddings = embeddings
        self.embedder = embedder
        self.candidates = candidates  # hits taken from each ranking before fusion
        self.rrf_k = rrf_k
        self.bm25 = BM25(self.chunks["text"].tolist())
        self._items = self.chunks["items"].str.split(",").map(set)
        self._event_type_mask = {
            event_type: self.chunks[flag].to_numpy(dtype=bool)
            & self.chunks["text"].str.contains(EVENT_TYPE_WORDS[event_type], case=False).to_numpy()
            for event_type, flag in EVENT_TYPE_FLAG.items()
        }

    @classmethod
    def build(
        cls, docs: pd.DataFrame, embedder: Embedder, max_chars: int, candidates: int, rrf_k: int
    ) -> "HybridIndex":
        chunks = chunk_documents(docs, max_chars)
        embeddings = embedder.encode_documents(chunks["text"].tolist())
        return cls(chunks, embeddings, embedder, candidates, rrf_k)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.chunks.to_parquet(path / CHUNKS_FILE)
        np.save(path / EMBEDDINGS_FILE, self.embeddings)
        (path / META_FILE).write_text(json.dumps({"embedder": self.embedder.name}))

    @classmethod
    def load(cls, path: Path, embedder: Embedder, candidates: int, rrf_k: int) -> "HybridIndex":
        built_with = json.loads((path / META_FILE).read_text())["embedder"]
        if built_with != embedder.name:
            raise ValueError(f"index at {path} was built with {built_with}, not {embedder.name}")
        chunks = pd.read_parquet(path / CHUNKS_FILE)
        return cls(chunks, np.load(path / EMBEDDINGS_FILE), embedder, candidates, rrf_k)

    def _mask(self, f: SearchFilters) -> np.ndarray:
        c = self.chunks
        mask = np.ones(len(c), dtype=bool)
        if f.tickers is not None:
            mask &= c["ticker"].isin(f.tickers).to_numpy()
        if f.start is not None:
            mask &= (c["filing_date"] >= pd.Timestamp(f.start)).to_numpy()
        if f.end is not None:
            mask &= (c["filing_date"] <= pd.Timestamp(f.end)).to_numpy()
        if f.event_type is not None:
            mask &= self._event_type_mask[f.event_type]
        if f.items is not None:
            wanted = set(f.items)
            mask &= self._items.map(lambda items: not wanted.isdisjoint(items)).to_numpy(bool)
        if f.accession_nos is not None:
            mask &= c["accession_no"].isin(f.accession_nos).to_numpy()
        return mask

    def search(self, query: str, filters: SearchFilters, k: int) -> pd.DataFrame:
        """Top k chunks matching the filters, best first, with an RRF "score" column."""
        allowed = np.flatnonzero(self._mask(filters))
        lexical = self.bm25.scores(query)[allowed]
        dense = self.embeddings[allowed] @ self.embedder.encode_query(query)

        fused = np.zeros(len(allowed))
        for scores in (np.where(lexical > 0, lexical, np.nan), dense):  # no keyword hit, no rank
            order = np.argsort(-scores, kind="stable")
            order = order[~np.isnan(scores[order])][: self.candidates]
            fused[order] += 1.0 / (self.rrf_k + np.arange(1, len(order) + 1))

        hits = self.chunks.iloc[allowed].assign(score=fused)
        hits = hits[hits["score"] > 0].sort_values("score", ascending=False, kind="stable")
        return hits.head(k).reset_index(drop=True)
