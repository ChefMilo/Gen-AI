"""Offline evaluation of RAG retrieval methods: BM25 vs Dense vs Hybrid.

Evaluates Recall@3, Recall@5, Recall@10 and Mean Reciprocal Rank (MRR) on hand-labelled
queries with ground-truth SEC 8-K filings:

    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --queries data/rag_queries/queries.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.build import load_index  # noqa: E402
from rag.config import load_rag_config  # noqa: E402
from rag.evaluation import (  # noqa: E402
    build_markdown_report,
    compute_category_summary,
    compute_summary,
    evaluate_query,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_queries = REPO_ROOT / "data" / "rag_queries" / "queries.json"
    parser.add_argument("--queries", type=Path, default=default_queries)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "rag_eval")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if not args.queries.is_file():
        sys.stderr.write(f"Error: Queries file not found: {args.queries}\n")
        return 1

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} labelled queries from {args.queries}")

    print("Loading hybrid index (this may take ~20-30s)...")
    cfg = load_rag_config(args.config)
    index = load_index(cfg)
    n_tickers = index.chunks["ticker"].nunique()
    print(f"Loaded index with {len(index.chunks):,} chunks across {n_tickers} tickers.")

    eval_results = []
    for q in queries:
        res = evaluate_query(index, q, candidates=cfg.candidates, rrf_k=cfg.rrf_k)
        eval_results.append(res)

    summary = compute_summary(eval_results)
    cat_summary = compute_category_summary(eval_results)

    print("\n=== SUMMARY RESULTS ===")
    print(summary.to_string(index=False, float_format="{:.3f}".format))

    print("\n=== BY CATEGORY ===")
    print(cat_summary.to_string(index=False, float_format="{:.3f}".format))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "rag_evaluation.json"
    json_path.write_text(json.dumps({
        "summary": summary.to_dict(orient="records"),
        "category_summary": cat_summary.to_dict(orient="records"),
        "results": eval_results,
    }, indent=2), encoding="utf-8")

    md_report = build_markdown_report(summary, cat_summary, eval_results)
    md_path = args.out_dir / "rag_evaluation.md"
    md_path.write_text(md_report, encoding="utf-8")

    print(f"\nArtifacts written to:\n  - {json_path}\n  - {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
