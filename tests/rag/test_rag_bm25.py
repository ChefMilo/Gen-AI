import numpy as np

from rag.bm25 import BM25, tokenize


def test_tokenize_keeps_item_numbers_and_amounts():
    assert tokenize("Item 8.01: a $0.25 Dividend") == ["item", "8.01", "a", "0.25", "dividend"]


def test_scores_rank_matching_documents_and_rare_terms_higher():
    bm25 = BM25(["cash dividend declared", "share repurchase program", "dividend and repurchase"])
    scores = bm25.scores("repurchase")
    assert scores[0] == 0
    assert scores[1] > 0 and scores[2] > 0
    assert np.argmax(bm25.scores("dividend declared")) == 0


def test_unknown_query_terms_score_zero():
    bm25 = BM25(["cash dividend"])
    assert bm25.scores("zebra").tolist() == [0.0]
