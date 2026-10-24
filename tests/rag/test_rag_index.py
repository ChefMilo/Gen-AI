from datetime import date

import pytest

from rag.index import HybridIndex, SearchFilters


@pytest.fixture
def index(docs, embedder):
    return HybridIndex.build(docs, embedder, max_chars=80, candidates=50, rrf_k=60)


def test_search_finds_the_relevant_document(index):
    hits = index.search("share repurchase undervalued", SearchFilters(), k=3)
    assert hits.iloc[0]["doc_ref"] == "AAA/0001-18-000002/ex99"
    assert hits["score"].is_monotonic_decreasing


def test_hits_are_chunks_with_exact_offsets(index, docs):
    hits = index.search("dividend", SearchFilters(), k=50)
    text_by_doc = docs.set_index("doc_ref")["text"]
    for hit in hits.itertuples():
        assert text_by_doc[hit.doc_ref][hit.start : hit.end] == hit.text
        assert hit.chunk_id.startswith(f"{hit.doc_ref}#")


@pytest.mark.parametrize(
    "filters, expected",
    [
        (SearchFilters(tickers=["BBB"]), {"BBB"}),
        (SearchFilters(end=date(2017, 12, 31)), {"AAA", "BBB"}),
        (SearchFilters(start=date(2019, 1, 1)), {"BBB", "CCC"}),
        (SearchFilters(event_type="buyback"), {"AAA"}),
        (SearchFilters(items=["5.02"]), {"CCC"}),
        (SearchFilters(accession_nos=["0002-21-000004"]), {"BBB"}),
    ],
)
def test_filters_restrict_hits(index, filters, expected):
    hits = index.search("dividend share company announced", filters, k=50)
    assert len(hits) > 0
    assert set(hits["ticker"]) <= expected


@pytest.mark.parametrize("event_type, word", [("dividend", "dividend"), ("buyback", "repurchase")])
def test_event_type_keeps_only_passages_that_mention_it(index, event_type, word):
    hits = index.search("company announced", SearchFilters(event_type=event_type), k=50)
    assert len(hits) > 0
    assert hits["text"].str.lower().str.contains(word).all()
    # e.g. the "Item 8.01 Other Events." heading of a dividend filing is not returned
    assert not (hits["text"] == "Item 8.01 Other Events.").any()


def test_filters_that_match_nothing_return_no_hits(index):
    assert index.search("dividend", SearchFilters(tickers=["ZZZ"]), k=5).empty


def test_filters_reject_unknown_keys():
    with pytest.raises(ValueError):
        SearchFilters.model_validate({"company": "AAA"})


def test_save_and_load_round_trip(index, embedder, tmp_path):
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, embedder, candidates=50, rrf_k=60)
    query, filters = "special dividend asset sale", SearchFilters()
    assert loaded.search(query, filters, k=5).equals(index.search(query, filters, k=5))


def test_load_rejects_a_different_embedder(index, embedder, tmp_path):
    index.save(tmp_path)
    embedder.name = "other-model"
    with pytest.raises(ValueError, match="other-model"):
        HybridIndex.load(tmp_path, embedder, candidates=50, rrf_k=60)
