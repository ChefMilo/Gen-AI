"""Gold set: 30 hand-labelled filings for measuring event extraction (data/gold/README.md).

data/gold/filings.csv lists the filings; data/gold/events.csv holds one row per event a
person found in them. score() compares extracted events against these labels.

    python -m ingest.gold    # check the labels; quotes too if $GENAI_DATA_DIR is set
"""

import math
import os
import re
import sys
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ValidationError

from store.prices import DATA_DIR_ENV
from store.schema import DividendKind, Event, EventType

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "data" / "gold"
FILINGS = "filings.csv"
EVENTS = "events.csv"
TEXT_DIR = Path("text/edgar")  # under $GENAI_DATA_DIR

STATUSES = ("todo", "labelled", "checked")
DONE = ("labelled", "checked")  # filings whose labels are scored

# Event fields a person labels; the extractor is scored on the same ones.
FIELDS = (
    "is_new_announcement",
    "announcement_date",
    "dividend_per_share",
    "dividend_kind",
    "record_date",
    "payable_date",
    "buyback_amount_usd",
    "buyback_shares",
    "is_new_program",
    "buyback_expiry",
)
DIVIDEND_FIELDS = ("dividend_per_share", "dividend_kind", "record_date", "payable_date")
BUYBACK_FIELDS = ("buyback_amount_usd", "buyback_shares", "is_new_program", "buyback_expiry")


class GoldEvent(BaseModel):
    """One row of data/gold/events.csv."""

    gold_id: str
    accession_no: str
    event_type: EventType
    is_new_announcement: bool
    announcement_date: date | None = None
    dividend_per_share: float | None = None
    dividend_kind: DividendKind | None = None
    record_date: date | None = None
    payable_date: date | None = None
    buyback_amount_usd: float | None = None
    buyback_shares: int | None = None
    is_new_program: bool | None = None
    buyback_expiry: date | None = None
    quote: str
    notes: str = ""


def _blank_to_none(row: dict) -> dict:
    return {k: (None if isinstance(v, str) and not v.strip() else v) for k, v in row.items()}


def load_filings(gold_dir: Path = GOLD_DIR) -> pd.DataFrame:
    """data/gold/filings.csv as text columns, blanks as ""."""
    return pd.read_csv(Path(gold_dir) / FILINGS, dtype=str, keep_default_na=False)


def load_events(gold_dir: Path = GOLD_DIR) -> list[GoldEvent]:
    """data/gold/events.csv as GoldEvents; ValueError names the first bad row."""
    rows = pd.read_csv(Path(gold_dir) / EVENTS, dtype=str, keep_default_na=False)
    events = []
    for i, row in enumerate(rows.to_dict("records"), start=2):  # line 1 is the header
        data = _blank_to_none(row)
        data["notes"] = data.get("notes") or ""
        try:
            events.append(GoldEvent.model_validate(data))
        except ValidationError as error:
            raise ValueError(f"{EVENTS} line {i}: {error}") from error
    return events


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _filing_text(data_dir: Path, ticker: str, accession_no: str) -> str | None:
    folder = data_dir / TEXT_DIR / ticker / accession_no
    if not folder.is_dir():
        return None
    return " ".join(p.read_text(errors="ignore") for p in sorted(folder.glob("*.txt")))


def check(
    filings: pd.DataFrame, events: list[GoldEvent], data_dir: Path | None = None
) -> list[str]:
    """Problems with the labels; empty when they are consistent.

    With data_dir (the shared Drive folder), each quote must occur in its filing's text,
    whitespace aside.
    """
    problems = []
    by_id = filings.set_index("gold_id")
    if not filings["gold_id"].is_unique:
        problems.append("filings.csv: gold_id values are not unique")
    for gold_id, status in filings[["gold_id", "status"]].itertuples(index=False):
        if status not in STATUSES:
            problems.append(f"{gold_id}: status {status!r} is not one of {STATUSES}")

    counts: dict[str, int] = {}
    for e in events:
        where = f"{e.gold_id} {e.event_type}"
        counts[e.gold_id] = counts.get(e.gold_id, 0) + 1
        if e.gold_id not in by_id.index:
            problems.append(f"{where}: gold_id is not in filings.csv")
            continue
        filing = by_id.loc[e.gold_id]
        if e.accession_no != filing["accession_no"]:
            problems.append(f"{where}: accession_no does not match filings.csv")
        if not _squash(e.quote):
            problems.append(f"{where}: quote is empty")
        unused = BUYBACK_FIELDS if e.event_type == "dividend" else DIVIDEND_FIELDS
        for field in unused:
            if getattr(e, field) is not None:
                problems.append(f"{where}: {field} belongs to the other event type")
        if e.event_type == "dividend":
            if e.dividend_per_share is None or e.dividend_kind is None:
                problems.append(f"{where}: needs dividend_per_share and dividend_kind")
            elif not 0.001 <= e.dividend_per_share <= 50:
                problems.append(f"{where}: dividend_per_share outside $0.001 to $50")
            if e.record_date and e.payable_date and e.record_date > e.payable_date:
                problems.append(f"{where}: record_date is after payable_date")
        else:
            if e.buyback_amount_usd is None and e.buyback_shares is None:
                problems.append(f"{where}: needs buyback_amount_usd or buyback_shares")
            if (e.buyback_amount_usd or 1) <= 0 or (e.buyback_shares or 1) <= 0:
                problems.append(f"{where}: buyback size must be positive")
            if e.is_new_program is None:
                problems.append(f"{where}: needs is_new_program")
        if data_dir is not None:
            text = _filing_text(Path(data_dir), filing["ticker"], e.accession_no)
            if text is None:
                problems.append(f"{where}: no text folder for the filing")
            elif _squash(e.quote) not in _squash(text):
                problems.append(f"{where}: quote is not copied exactly from the filing")

    for gold_id, status, n in filings[["gold_id", "status", "n_events"]].itertuples(index=False):
        if status not in DONE:
            continue
        if not n.strip().isdigit():
            problems.append(f"{gold_id}: {status} but n_events is not a whole number")
        elif int(n) != counts.get(gold_id, 0):
            problems.append(
                f"{gold_id}: n_events is {n} but events.csv has {counts.get(gold_id, 0)}"
            )
    return problems


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float | int) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    return a == b


def _agreement(gold: GoldEvent, predicted: BaseModel) -> int:
    return sum(_same(getattr(gold, f), getattr(predicted, f, None)) for f in FIELDS)


def score(
    filings: pd.DataFrame, events: list[GoldEvent], predicted: Iterable[Event | GoldEvent]
) -> dict:
    """How well predicted events match the gold labels, on labelled filings only.

    Events pair up within the same filing and event type, best field agreement first.
    precision and recall count events. field_accuracy is the share of paired events with
    that field right (blank on both sides counts as right), over pairs of the event type
    the field belongs to; None when there are none. filing_accuracy is the share of
    filings whose events all match on every field, including filings with no events.
    """
    done = filings[filings["status"].isin(DONE)]
    accessions = set(done["accession_no"])
    gold = [e for e in events if e.accession_no in accessions]
    pred = [p for p in predicted if p.accession_no in accessions]

    pairs = []
    perfect = {a: True for a in accessions}
    for accession in accessions:
        for event_type in ("dividend", "buyback"):
            g = [e for e in gold if e.accession_no == accession and e.event_type == event_type]
            p = [e for e in pred if e.accession_no == accession and e.event_type == event_type]
            options = sorted(
                ((_agreement(ge, pe), i, j) for i, ge in enumerate(g) for j, pe in enumerate(p)),
                reverse=True,
            )
            used_g, used_p = set(), set()
            for agree, i, j in options:
                if i in used_g or j in used_p:
                    continue
                used_g.add(i)
                used_p.add(j)
                pairs.append((g[i], p[j]))
                if agree < len(FIELDS):
                    perfect[accession] = False
            if len(g) != len(p):
                perfect[accession] = False

    unused = {"dividend": BUYBACK_FIELDS, "buyback": DIVIDEND_FIELDS}
    accuracy = {}
    for f in FIELDS:
        # Dividend fields are scored on dividend pairs only, buyback fields on buyback pairs.
        scored = [(g, p) for g, p in pairs if f not in unused[g.event_type]]
        right = sum(_same(getattr(g, f), getattr(p, f, None)) for g, p in scored)
        accuracy[f] = right / len(scored) if scored else None

    return {
        "n_filings": len(accessions),
        "n_gold": len(gold),
        "n_predicted": len(pred),
        "n_matched": len(pairs),
        "precision": len(pairs) / len(pred) if pred else 1.0,
        "recall": len(pairs) / len(gold) if gold else 1.0,
        "field_accuracy": accuracy,
        "filing_accuracy": sum(perfect.values()) / len(accessions) if accessions else None,
    }


def main() -> int:
    filings, events = load_filings(), load_events()
    value = os.environ.get(DATA_DIR_ENV)
    problems = check(filings, events, Path(value) if value else None)
    done = filings["status"].isin(DONE).sum()
    print(f"{len(filings)} filings ({done} labelled), {len(events)} events")
    if not value:
        print(f"Quotes not checked: set {DATA_DIR_ENV} to check them against the filings.")
    for problem in problems:
        print(f"  {problem}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
