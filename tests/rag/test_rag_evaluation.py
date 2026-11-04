from rag_helpers import HashEmbedder, make_docs

from rag.evaluation import (
    compute_category_summary,
    compute_summary,
    df_to_markdown,
    evaluate_query,
)
from rag.index import HybridIndex


def test_evaluate_query_and_summaries():
    docs = make_docs()
    embedder = HashEmbedder()
    index = HybridIndex.build(docs, embedder, max_chars=500, candidates=10, rrf_k=60)

    query_item = {
        "id": "T01",
        "query": "special one-time dividend asset sale",
        "filters": {"event_type": "dividend"},
        "category": "exact_keyword",
        "relevant_accession_nos": ["0002-21-000004"],
        "description": "BBB Inc special dividend",
    }

    res = evaluate_query(index, query_item, candidates=10, rrf_k=60)
    assert res["id"] == "T01"
    assert res["ranks"]["bm25"] == 1
    assert res["metrics"]["bm25"]["recall@3"] == 1.0
    assert res["metrics"]["bm25"]["mrr"] == 1.0

    summary = compute_summary([res])
    assert len(summary) == 3
    assert "Recall@3" in summary.columns

    cat_summary = compute_category_summary([res])
    assert len(cat_summary) == 3
    assert cat_summary["Category"].iloc[0] == "exact_keyword"

    md = df_to_markdown(summary)
    assert "| Method | Recall@3 |" in md
