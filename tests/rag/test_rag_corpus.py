from datetime import date

import pandas as pd

from rag.corpus import FILING_INDEX, TEXT_DIR, UNIVERSE_QUALITY, load_corpus


def _write(data_dir, filings, quality, texts):
    (data_dir / FILING_INDEX).parent.mkdir(parents=True)
    pd.DataFrame(filings).to_parquet(data_dir / FILING_INDEX)
    pd.DataFrame(quality).to_csv(data_dir / UNIVERSE_QUALITY, index=False)
    for (ticker, accession, document), text in texts.items():
        folder = data_dir / TEXT_DIR / ticker / accession
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{document}.txt").write_text(text, encoding="utf-8")


def _filing(ticker, accession, filed, div=None):
    return {
        "ticker": ticker,
        "company_name": f"{ticker} Corp",
        "accession_no": accession,
        "form_type": "8-K",
        "filing_date": filed,
        "items": "8.01,9.01",
        "is_div_candidate": div,
        "is_buyback_candidate": False,
    }


def test_load_corpus_follows_universe_quality(tmp_path):
    filings = [
        _filing("KEEP", "a-1", date(2016, 5, 1), div=True),
        _filing("LATE", "b-1", date(2016, 5, 1)),  # before usable_from
        _filing("LATE", "b-2", date(2019, 5, 1)),
        _filing("DROP", "c-1", date(2017, 5, 1)),
        _filing("NEW", "d-1", date(2017, 5, 1)),  # not in universe_quality.csv
    ]
    quality = {
        "ticker": ["KEEP", "LATE", "DROP"],
        "recommendation": ["keep", "keep_from", "drop"],
        "usable_from": [None, "2018-01-01", None],
    }
    texts = {
        ("KEEP", "a-1", "main.htm"): "8-K cover",
        ("KEEP", "a-1", "ex99-1.htm"): "Dividend press release",
        ("LATE", "b-1", "ex99.htm"): "too early",
        ("LATE", "b-2", "ex99.htm"): "valid",
        ("DROP", "c-1", "ex99.htm"): "dropped",
        ("NEW", "d-1", "ex99.htm"): "unknown company",
    }
    _write(tmp_path, filings, quality, texts)

    docs = load_corpus(tmp_path)

    assert sorted(docs["doc_ref"]) == [
        "KEEP/a-1/ex99-1.htm",
        "KEEP/a-1/main.htm",
        "LATE/b-2/ex99.htm",
    ]
    keep = docs[docs["doc_ref"] == "KEEP/a-1/ex99-1.htm"].iloc[0]
    assert keep["text"] == "Dividend press release"
    assert keep["filing_date"] == pd.Timestamp(2016, 5, 1)
    assert docs["is_div_candidate"].dtype == bool
    assert not docs.loc[docs["ticker"] == "LATE", "is_div_candidate"].any()  # None -> False
