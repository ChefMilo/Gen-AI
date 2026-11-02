"""Gold set: 60 hand-labelled filings for measuring event extraction (data/gold/README.md).

data/gold/filings.csv lists the filings; data/gold/events.csv holds one row per event a
person found in them. 15 filings (double = yes) are labelled again, independently, in
data/gold/events_second.csv. score() compares extracted events against the labels and
agreement() compares the two labellers.

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
from store.schema import DividendKind, EventType

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "data" / "gold"
FILINGS = "filings.csv"
EVENTS = "events.csv"
SECOND_EVENTS = "events_second.csv"
TEXT_DIR = Path("text/edgar")  # under $GENAI_DATA_DIR

STATUSES = ("todo", "labelled", "checked")
DONE = ("labelled", "checked")  # filings whose labels are scored
SECOND_STATUSES = ("", "todo", "labelled")

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
UNUSED = {"dividend": BUYBACK_FIELDS, "buyback": DIVIDEND_FIELDS}  # fields per event type


class GoldEvent(BaseModel):
    """One row of data/gold/events.csv (or events_second.csv)."""

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


def load_events(gold_dir: Path = GOLD_DIR, name: str = EVENTS) -> list[GoldEvent]:
    """events.csv (or name) as GoldEvents; ValueError names the first bad row."""
    rows = pd.read_csv(Path(gold_dir) / name, dtype=str, keep_default_na=False)
    events = []
    for i, row in enumerate(rows.to_dict("records"), start=2):  # line 1 is the header
        data = _blank_to_none(row)
        data["notes"] = data.get("notes") or ""
        try:
            events.append(GoldEvent.model_validate(data))
        except ValidationError as error:
            raise ValueError(f"{name} line {i}: {error}") from error
    return events


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _filing_text(data_dir: Path, ticker: str, accession_no: str) -> str | None:
    folder = data_dir / TEXT_DIR / ticker / accession_no
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.txt"))
    return " ".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


def _event_problems(e: GoldEvent, where: str) -> list[str]:
    problems = []
    if not _squash(e.quote):
        problems.append(f"{where}: quote is empty")
    for field in UNUSED[e.event_type]:
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
    return problems


def _check_events(
    events: list[GoldEvent], by_id: pd.DataFrame, file: str, data_dir: Path | None
) -> list[str]:
    problems = []
    for e in events:
        where = f"{file} {e.gold_id} {e.event_type}"
        if e.gold_id not in by_id.index:
            problems.append(f"{where}: gold_id is not in filings.csv")
            continue
        filing = by_id.loc[e.gold_id]
        if e.accession_no != filing["accession_no"]:
            problems.append(f"{where}: accession_no does not match filings.csv")
        problems += _event_problems(e, where)
        if data_dir is not None:
            text = _filing_text(Path(data_dir), filing["ticker"], e.accession_no)
            if text is None:
                problems.append(f"{where}: no text folder for the filing")
            elif _squash(e.quote) not in _squash(text):
                problems.append(f"{where}: quote is not copied exactly from the filing")
    return problems


def check(
    filings: pd.DataFrame,
    events: list[GoldEvent],
    data_dir: Path | None = None,
    second: list[GoldEvent] | None = None,
) -> list[str]:
    """Problems with the labels; empty when they are consistent.

    With data_dir (the shared Drive folder), each quote must occur in its filing's text,
    whitespace aside. second holds the independent labels of the double = yes filings.
    """
    problems = []
    if not filings["gold_id"].is_unique:
        problems.append("filings.csv: gold_id values are not unique")
    by_id = filings.drop_duplicates("gold_id").set_index("gold_id")
    for gold_id, status in filings[["gold_id", "status"]].itertuples(index=False):
        if status not in STATUSES:
            problems.append(f"{gold_id}: status {status!r} is not one of {STATUSES}")
    for gold_id, status in filings[["gold_id", "second_status"]].itertuples(index=False):
        if status not in SECOND_STATUSES:
            problems.append(f"{gold_id}: second_status {status!r} is not one of {SECOND_STATUSES}")

    problems += _check_events(events, by_id, EVENTS, data_dir)
    counts: dict[str, int] = {}
    for e in events:
        counts[e.gold_id] = counts.get(e.gold_id, 0) + 1
    for gold_id, status, n in filings[["gold_id", "status", "n_events"]].itertuples(index=False):
        if status not in DONE:
            continue
        if not n.strip().isdigit():
            problems.append(f"{gold_id}: {status} but n_events is not a whole number")
        elif int(n) != counts.get(gold_id, 0):
            problems.append(
                f"{gold_id}: n_events is {n} but events.csv has {counts.get(gold_id, 0)}"
            )

    second = second or []
    problems += _check_events(second, by_id, SECOND_EVENTS, data_dir)
    double = set(filings.loc[filings["double"] == "yes", "gold_id"])
    for e in second:
        if e.gold_id in by_id.index and e.gold_id not in double:
            problems.append(f"{SECOND_EVENTS} {e.gold_id}: filing is not marked double = yes")
    return problems


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float | int) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    return a == b


def _agreement(gold: GoldEvent, predicted: BaseModel) -> int:
    return sum(_same(getattr(gold, f), getattr(predicted, f, None)) for f in FIELDS)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(filings: pd.DataFrame, events: list[GoldEvent], predicted: Iterable[BaseModel]) -> dict:
    """How well predicted events (store.schema.Event rows) match the gold labels.

    Only labelled and checked filings count. Events pair up within the same filing and
    event type, best field agreement first.

    - precision, recall, f1: events found, counting pairs.
    - field_accuracy: share of pairs with the field right (blank on both sides counts as
      right), over pairs of the event type the field belongs to; None when there are none.
    - field_prf: per field, precision = right values / values predicted, recall = right
      values / values in the gold labels, f1; unpaired events count as misses.
    - filing_accuracy: share of filings whose events all match on every field, including
      filings with no events.
    - flagged_rate: share of predicted events with extraction_status "flagged".
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

    accuracy, prf = {}, {}
    for f in FIELDS:
        # Dividend fields are scored on dividend events only, buyback fields on buybacks.
        scored = [(g, p) for g, p in pairs if f not in UNUSED[g.event_type]]
        right = sum(_same(getattr(g, f), getattr(p, f, None)) for g, p in scored)
        accuracy[f] = _ratio(right, len(scored))
        hits = sum(
            getattr(g, f) is not None and _same(getattr(g, f), getattr(p, f, None))
            for g, p in scored
        )
        n_pred = sum(getattr(p, f, None) is not None for p in pred if f not in UNUSED[p.event_type])
        n_gold = sum(getattr(g, f) is not None for g in gold if f not in UNUSED[g.event_type])
        precision, recall = _ratio(hits, n_pred), _ratio(hits, n_gold)
        prf[f] = {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}

    precision = len(pairs) / len(pred) if pred else 1.0
    recall = len(pairs) / len(gold) if gold else 1.0
    statuses = [getattr(p, "extraction_status", None) for p in pred]
    return {
        "n_filings": len(accessions),
        "n_gold": len(gold),
        "n_predicted": len(pred),
        "n_matched": len(pairs),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "field_accuracy": accuracy,
        "field_prf": prf,
        "filing_accuracy": _ratio(sum(perfect.values()), len(accessions)),
        "flagged_rate": _ratio(statuses.count("flagged"), len(pred)) if pred else None,
    }


def agreement(filings: pd.DataFrame, events: list[GoldEvent], second: list[GoldEvent]) -> dict:
    """score() of the second labeller against the first, on the double-labelled filings.

    Counts only filings with double = yes that both labellers have finished.
    """
    both = filings[
        (filings["double"] == "yes")
        & filings["status"].isin(DONE)
        & (filings["second_status"] == "labelled")
    ]
    return score(both, events, second)


def main() -> int:
    filings, events = load_filings(), load_events()
    second = load_events(name=SECOND_EVENTS)
    value = os.environ.get(DATA_DIR_ENV)
    problems = check(filings, events, Path(value) if value else None, second)
    done = filings["status"].isin(DONE).sum()
    seconds = (filings["second_status"] == "labelled").sum()
    n_double = (filings["double"] == "yes").sum()
    print(f"{len(filings)} filings: {done} labelled, {seconds} of {n_double} second-labelled")
    print(f"{len(events)} events, {len(second)} second-labeller events")
    if not value:
        print(f"Quotes not checked: set {DATA_DIR_ENV} to check them against the filings.")
    for problem in problems:
        print(f"  {problem}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
