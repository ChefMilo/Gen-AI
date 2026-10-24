"""Announcement corpus: one row per text/edgar document, with its filing metadata.

Layout (docs/data_schema.md, "Filings"): text/edgar/<TICKER>/<accession_no>/<document>.txt,
joined to index/filing_index.parquet on ticker + accession_no. Companies and start dates
follow universe_quality.csv.
"""

from pathlib import Path

import pandas as pd

FILING_INDEX = Path("index/filing_index.parquet")
TEXT_DIR = Path("text/edgar")
UNIVERSE_QUALITY = Path("universe_quality.csv")
USABLE = ("keep", "keep_from")

FILING_COLUMNS = [
    "ticker",
    "company_name",
    "accession_no",
    "form_type",
    "items",
    "filing_date",
    "is_div_candidate",
    "is_buyback_candidate",
]
DOC_COLUMNS = ["doc_ref", "document", *FILING_COLUMNS, "text"]


def usable_filings(filings: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Filings of keep / keep_from companies, filed on or after usable_from."""
    quality = quality.set_index("ticker")
    recommendation = filings["ticker"].map(quality["recommendation"])
    usable_from = filings["ticker"].map(pd.to_datetime(quality["usable_from"]))
    keep = recommendation.isin(USABLE) & (
        usable_from.isna() | (filings["filing_date"] >= usable_from)
    )
    return filings[keep]


def load_corpus(data_dir: Path) -> pd.DataFrame:
    """Every text document of every usable filing, one row each (DOC_COLUMNS)."""
    filings = pd.read_parquet(data_dir / FILING_INDEX, columns=FILING_COLUMNS)
    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    for flag in ("is_div_candidate", "is_buyback_candidate"):
        filings[flag] = filings[flag].fillna(False).astype(bool)
    filings["items"] = filings["items"].fillna("")
    filings = usable_filings(filings, pd.read_csv(data_dir / UNIVERSE_QUALITY))

    rows = []
    for filing in filings.to_dict(orient="records"):
        folder = data_dir / TEXT_DIR / filing["ticker"] / filing["accession_no"]
        for path in sorted(folder.glob("*.txt")):
            rows.append(
                {
                    **filing,
                    "doc_ref": f"{filing['ticker']}/{filing['accession_no']}/{path.stem}",
                    "document": path.stem,
                    "text": path.read_text(encoding="utf-8", errors="replace"),
                }
            )
    return pd.DataFrame(rows, columns=DOC_COLUMNS)
