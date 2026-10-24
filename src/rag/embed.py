"""Sentence-embedding encoders. Vectors are L2-normalised, so a dot product is a cosine."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str  # stored with a saved index; loading with another model is refused

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """A sentence-transformers model (pip install -e .[rag])."""

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str], progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return vectors.astype(np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, progress=True)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text], progress=False)[0]
