"""Okapi BM25 keyword scoring, precomputed as a sparse chunk x term weight matrix."""

import re
from collections import Counter

import numpy as np
from scipy import sparse

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")  # keeps "8.01" and "0.25" whole


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.vocab: dict[str, int] = {}
        rows, cols, freqs = [], [], []
        lengths = np.zeros(len(texts))
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                rows.append(i)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                freqs.append(tf)
        rows, cols, tf = np.array(rows, int), np.array(cols, int), np.array(freqs, float)

        n_docs = len(texts)
        df = np.bincount(cols, minlength=len(self.vocab))
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        avg_length = lengths.mean() if n_docs and lengths.mean() > 0 else 1.0
        norm = k1 * (1 - b + b * lengths / avg_length)
        weights = idf[cols] * tf * (k1 + 1) / (tf + norm[rows])
        self.matrix = sparse.csc_array((weights, (rows, cols)), shape=(n_docs, len(self.vocab)))

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every text for the query; 0 where no query term occurs."""
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        return np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()
