"""Build the hybrid index from $GENAI_DATA_DIR into rag.index_dir.

python -m rag.build
"""

from rag.config import RagConfig, load_rag_config
from rag.corpus import load_corpus
from rag.embed import SentenceTransformerEmbedder
from rag.index import HybridIndex
from store.prices import data_dir


def embedder_for(config: RagConfig) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(config.embedding_model, config.query_prefix)


def load_index(config: RagConfig) -> HybridIndex:
    """The saved index, with the embedder it was built with."""
    return HybridIndex.load(config.index_dir, embedder_for(config), config.candidates, config.rrf_k)


def main() -> None:
    config = load_rag_config()
    docs = load_corpus(data_dir())
    print(f"{len(docs)} documents from {docs['accession_no'].nunique()} filings")
    index = HybridIndex.build(
        docs, embedder_for(config), config.chunk_max_chars, config.candidates, config.rrf_k
    )
    index.save(config.index_dir)
    print(f"{len(index.chunks)} chunks saved to {config.index_dir}")


if __name__ == "__main__":
    main()
