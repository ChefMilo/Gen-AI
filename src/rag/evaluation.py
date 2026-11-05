"""Evaluation of RAG retrieval methods: BM25 vs Dense vs Hybrid.

Computes Recall@3, Recall@5, Recall@10, and Mean Reciprocal Rank (MRR)
on labelled queries.
"""

from typing import Any

import numpy as np
import pandas as pd

from rag.index import HybridIndex, SearchFilters

K_VALUES = (3, 5, 10)
METHODS = ("bm25", "dense", "hybrid")


def evaluate_query(
    index: HybridIndex,
    query_item: dict[str, Any],
    candidates: int,
    rrf_k: int,
) -> dict[str, Any]:
    """Evaluate one query across BM25, Dense, and Hybrid."""
    q_text = query_item["query"]
    filters = SearchFilters.model_validate(query_item.get("filters") or {})
    targets = set(query_item["relevant_accession_nos"])

    allowed = np.flatnonzero(index._mask(filters))
    if len(allowed) == 0:
        return {
            "id": query_item["id"],
            "query": q_text,
            "category": query_item.get("category", "general"),
            "targets": list(targets),
            "ranks": {m: None for m in METHODS},
            "metrics": {m: {f"recall@{k}": 0.0 for k in K_VALUES} | {"mrr": 0.0} for m in METHODS},
        }

    chunks_sub = index.chunks.iloc[allowed]
    accs = chunks_sub["accession_no"].to_numpy()

    # 1. Lexical (BM25)
    lexical = index.bm25.scores(q_text)[allowed]

    # 2. Dense (Embedding dot product)
    q_vec = index.embedder.encode_query(q_text)
    dense = index.embeddings[allowed] @ q_vec

    # 3. Hybrid (RRF)
    fused = np.zeros(len(allowed), dtype=np.float64)
    for scores in (np.where(lexical > 0, lexical, np.nan), dense):
        order = np.argsort(-scores, kind="stable")
        valid_order = order[~np.isnan(scores[order])][:candidates]
        fused[valid_order] += 1.0 / (rrf_k + np.arange(1, len(valid_order) + 1))

    scores_by_method = {
        "bm25": lexical,
        "dense": dense,
        "hybrid": fused,
    }

    ranks: dict[str, int | None] = {}
    metrics: dict[str, dict[str, float]] = {}

    for method, scores in scores_by_method.items():
        # Document-level aggregation: max score per accession_no
        df = pd.DataFrame({"acc": accs, "score": scores})
        ranked_docs = df.groupby("acc")["score"].max().sort_values(ascending=False).index.tolist()

        first_rank = None
        for rank, doc_id in enumerate(ranked_docs, 1):
            if doc_id in targets:
                first_rank = rank
                break

        ranks[method] = first_rank
        mrr = (1.0 / first_rank) if first_rank is not None else 0.0
        recalls = {
            f"recall@{k}": 1.0 if (first_rank is not None and first_rank <= k) else 0.0
            for k in K_VALUES
        }
        metrics[method] = {**recalls, "mrr": mrr}

    return {
        "id": query_item["id"],
        "query": q_text,
        "category": query_item.get("category", "general"),
        "targets": list(targets),
        "ranks": ranks,
        "metrics": metrics,
    }


def compute_summary(eval_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate Recall@k and MRR across queries for each method."""
    rows = []
    for m in METHODS:
        r3 = float(np.mean([r["metrics"][m]["recall@3"] for r in eval_results]))
        r5 = float(np.mean([r["metrics"][m]["recall@5"] for r in eval_results]))
        r10 = float(np.mean([r["metrics"][m]["recall@10"] for r in eval_results]))
        mrr = float(np.mean([r["metrics"][m]["mrr"] for r in eval_results]))
        found_ranks = [r["ranks"][m] for r in eval_results if r["ranks"][m] is not None]
        mean_rank = float(np.mean(found_ranks)) if found_ranks else float("nan")

        rows.append({
            "Method": m.upper(),
            "Recall@3": r3,
            "Recall@5": r5,
            "Recall@10": r10,
            "MRR": mrr,
            "Mean Rank": mean_rank,
        })

    return pd.DataFrame(rows)


def compute_category_summary(eval_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate metrics grouped by query category and method."""
    categories = sorted({r["category"] for r in eval_results})
    rows = []
    for cat in categories:
        sub = [r for r in eval_results if r["category"] == cat]
        for m in METHODS:
            r5 = float(np.mean([r["metrics"][m]["recall@5"] for r in sub]))
            mrr = float(np.mean([r["metrics"][m]["mrr"] for r in sub]))
            rows.append({
                "Category": cat,
                "Method": m.upper(),
                "Queries": len(sub),
                "Recall@5": r5,
                "MRR": mrr,
            })
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table without external dependencies like tabulate."""
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        formatted_vals = []
        for val in row:
            if isinstance(val, float):
                formatted_vals.append(f"{val:.3f}")
            else:
                formatted_vals.append(str(val))
        rows.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join([header_line, separator_line] + rows)


def build_markdown_report(
    summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
    eval_results: list[dict[str, Any]],
) -> str:
    lines = [
        "# RAG Retrieval Evaluation Report",
        "",
        "Evaluation of **BM25**, **Dense Embeddings** (`BAAI/bge-small-en-v1.5`), and "
        "**Hybrid RRF** across "
        f"{len(eval_results)} hand-labelled queries against SEC EDGAR 8-K filings.",
        "",
        "## Summary Results",
        "",
        df_to_markdown(summary),
        "",
        "## Breakdown by Query Category",
        "",
        df_to_markdown(cat_summary),
        "",
        "## Per-Query Ranks",
        "",
        "| ID | Category | Query | BM25 Rank | Dense Rank | Hybrid Rank |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in eval_results:
        bm = str(r["ranks"]["bm25"]) if r["ranks"]["bm25"] is not None else ">100"
        de = str(r["ranks"]["dense"]) if r["ranks"]["dense"] is not None else ">100"
        hy = str(r["ranks"]["hybrid"]) if r["ranks"]["hybrid"] is not None else ">100"
        lines.append(f"| {r['id']} | {r['category']} | {r['query']} | {bm} | {de} | {hy} |")

    lines.append("")
    return "\n".join(lines)

