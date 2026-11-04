"""Hand-label a gold filing in the terminal (data/gold/README.md has the rules).

    python -m ingest.label G05              # label one filing
    python -m ingest.label next             # the next unlabelled filing
    python -m ingest.label next --range G01-G30
    python -m ingest.label next --second    # second labeller: the double = yes filings
    python -m ingest.label G05 --redo       # replace your earlier labels for G05

It shows the filing's sentences about dividends and buybacks, numbered, then asks for
each event field by field. A quote is given as sentence numbers ("12" or "12,13").
Answers are checked like `python -m ingest.gold`, then written to events.csv (or
events_second.csv) and filings.csv. The model's answers are never shown: the labels
must be your own reading.

Needs $GENAI_DATA_DIR (the filing text).
"""

import argparse
import os
import re
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from ingest import extract, gold
from store.prices import DATA_DIR_ENV

Ask = Callable[[str], str]  # prompt -> the person's answer (input() in the terminal)
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")
NAME_ENV = "GOLD_LABELLER"


# Showing the filing


def numbered_sentences(text: str) -> list[str]:
    """The filing's sentences, split as the extractor splits them (not trimmed)."""
    return extract._SENTENCE_END.split(extract.squash(text))


def relevant(sentences: list[str], window: int = 1) -> list[int]:
    """Indexes of sentences mentioning dividends or buybacks, plus `window` either side."""
    hits = [i for i, s in enumerate(sentences) if extract.KEYWORDS.search(s)]
    keep = {j for i in hits for j in range(i - window, i + window + 1)}
    return sorted(j for j in keep if 0 <= j < len(sentences))


def show(sentences: list[str], ids: list[int], out=print) -> None:
    last = None
    for i in ids:
        if last is not None and i != last + 1:
            out("   ...")
        mark = "*" if extract.KEYWORDS.search(sentences[i]) else " "
        out(f"{mark}[{i + 1}] {extract._trim(sentences[i])}")
        last = i


# Reading answers


class Retry(ValueError):
    """An answer that can't be read; the question is asked again."""


NOT_STATED = re.compile(r"\s*(-|none|na|n/a|nil|not stated|blank.*|skip)?\s*", re.I)


def parse_date(text: str) -> date | None:
    if NOT_STATED.fullmatch(text):
        return None
    text = re.sub(r"\s*,\s*", ", ", text.strip())
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise Retry("write the date as YYYY-MM-DD (or e.g. May 15, 2024)")


def parse_money(text: str) -> float | None:
    """'$0.25', '25 cents', '24.25c', '300m', '$1.5 billion', '300000000' -> a number."""
    if NOT_STATED.fullmatch(text):
        return None
    t = text.strip().lower().replace("$", "").replace(",", "").replace(" ", "")
    if not t:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(cents?|c|k|thousand|m|mn|million|b|bn|billion)?", t)
    if not m:
        raise Retry("write a number, e.g. 0.25, 25 cents, 300m or 1.5b")
    value = float(m.group(1))
    unit = m.group(2) or ""
    scale = {"c": 0.01, "cent": 0.01, "cents": 0.01, "k": 1e3, "thousand": 1e3, "m": 1e6}
    scale |= {"mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
    return value * scale.get(unit, 1)


def parse_yes_no(text: str, blank: bool | None = None) -> bool | None:
    t = text.strip().lower()
    if not t:
        if blank is None:
            raise Retry("answer y or n")
        return blank
    if t in ("y", "yes"):
        return True
    if t in ("n", "no"):
        return False
    raise Retry("answer y or n")


def parse_choice(text: str, options: dict[str, str]) -> str:
    t = text.strip().lower()
    for key, value in options.items():
        if t in (key, value):
            return value
    raise Retry(f"answer one of {', '.join(options)}")


def parse_quote(text: str, sentences: list[str]) -> str:
    """'12' or '12,13' or '12-14' -> those sentences joined; anything else is a typed quote."""
    t = text.strip()
    if re.fullmatch(r"\d+(\s*[-,]\s*\d+)*", t):
        ids: list[int] = []
        for part in re.split(r"\s*,\s*", t):
            a, _, b = part.partition("-")
            ids += list(range(int(a), int(b or a) + 1))
        if any(not 1 <= i <= len(sentences) for i in ids):
            raise Retry(f"sentence numbers run from 1 to {len(sentences)}")
        if any(len(sentences[i - 1]) > extract.LONG_SENTENCE for i in ids):
            raise Retry("that sentence is a long table: type the exact words instead")
        return " ".join(sentences[i - 1] for i in ids)
    if not t:
        raise Retry("give the sentence number(s) the event is stated in")
    return t


def ask_until(ask: Ask, prompt: str, parse: Callable[[str], object], out=print):
    while True:
        try:
            return parse(ask(prompt))
        except Retry as error:
            out(f"  ! {error}")


# One event


def ask_event(
    ask: Ask, sentences: list[str], gold_id: str, accession_no: str, out=print
) -> gold.GoldEvent:
    kind = ask_until(
        ask,
        "  type [d]ividend / [b]uyback: ",
        lambda t: parse_choice(t, {"d": "dividend", "b": "buyback"}),
        out,
    )
    row: dict = {"gold_id": gold_id, "accession_no": accession_no, "event_type": kind}
    row["is_new_announcement"] = ask_until(
        ask, "  new announcement in THIS filing? [y/n]: ", parse_yes_no, out
    )
    row["announcement_date"] = ask_until(
        ask, "  announcement date (blank = not stated): ", parse_date, out
    )
    if kind == "dividend":
        row["dividend_per_share"] = ask_until(
            ask, "  dividend per share in $ (e.g. 0.25): ", parse_money, out
        )
        row["dividend_kind"] = ask_until(
            ask,
            "  kind [r]egular / [s]pecial / [i]nitial: ",
            lambda t: parse_choice(t, {"r": "regular", "s": "special", "i": "initial"}),
            out,
        )
        row["record_date"] = ask_until(ask, "  record date (blank = not stated): ", parse_date, out)
        row["payable_date"] = ask_until(
            ask, "  payable date (blank = not stated): ", parse_date, out
        )
    else:
        row["buyback_amount_usd"] = ask_until(
            ask,
            "  amount NEWLY authorised in $ (e.g. 300m; blank if in shares): ",
            parse_money,
            out,
        )
        shares = ask_until(
            ask, "  shares authorised (e.g. 2.5m; blank if in $): ", parse_money, out
        )
        row["buyback_shares"] = None if shares is None else int(round(shares))
        row["is_new_program"] = ask_until(
            ask,
            "  new program? [y] new (or replacing one) / [n] increase or extension: ",
            parse_yes_no,
            out,
        )
        row["buyback_expiry"] = ask_until(
            ask, "  program end date (blank = not stated): ", parse_date, out
        )
    row["quote"] = ask_until(
        ask,
        "  quote: sentence number(s), e.g. 12 or 12,13: ",
        lambda t: parse_quote(t, sentences),
        out,
    )
    row["notes"] = ask("  notes (optional): ").strip()
    return gold.GoldEvent.model_validate(row)


def checked_event(ask: Ask, sentences, gold_id, accession_no, text: str, out=print):
    """ask_event until the event passes gold's checks (or the person accepts it as is)."""
    while True:
        try:
            event = ask_event(ask, sentences, gold_id, accession_no, out)
        except ValidationError as error:
            out(f"  ! {error.errors()[0]['msg']}; let's redo this event")
            continue
        problems = gold._event_problems(event, f"{gold_id} {event.event_type}")
        if gold._squash(event.quote) not in gold._squash(text):
            problems.append("the quote is not in the filing text")
        if not problems:
            return event
        for p in problems:
            out(f"  ! {p}")
        if ask_until(ask, "  keep it anyway? [y/N]: ", lambda t: parse_yes_no(t, False), out):
            return event


# Files


def _read(gold_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(gold_dir) / name, dtype=str, keep_default_na=False)


def _write(df: pd.DataFrame, gold_dir: Path, name: str) -> None:
    df.to_csv(Path(gold_dir) / name, index=False, lineterminator="\n")


def _as_csv_row(event: gold.GoldEvent, columns: list[str]) -> dict:
    values = event.model_dump()
    out = {}
    for c in columns:
        v = values.get(c)
        if v is None:
            out[c] = ""
        elif isinstance(v, bool):
            out[c] = "True" if v else "False"
        elif isinstance(v, float) and v.is_integer() and c == "buyback_amount_usd":
            out[c] = str(int(v))
        else:
            out[c] = str(v)
    return out


def save(gold_dir: Path, gold_id: str, events: list[gold.GoldEvent], by: str, second: bool):
    """Replace this filing's rows in the events file and mark it done in filings.csv."""
    name = gold.SECOND_EVENTS if second else gold.EVENTS
    rows = _read(gold_dir, name)
    columns = list(rows.columns)
    rows = rows[rows["gold_id"] != gold_id]
    new = pd.DataFrame([_as_csv_row(e, columns) for e in events], columns=columns)
    rows = pd.concat([rows, new], ignore_index=True)
    # sorted by gold_id (stable), so people labelling different ranges rarely clash in git
    rows = rows.sort_values("gold_id", kind="stable")
    _write(rows, gold_dir, name)

    filings = _read(gold_dir, gold.FILINGS)
    at = filings["gold_id"] == gold_id
    if second:
        filings.loc[at, ["second_status", "second_by"]] = ["labelled", by]
    else:
        filings.loc[at, ["status", "n_events", "labelled_by"]] = ["labelled", str(len(events)), by]
    _write(filings, gold_dir, gold.FILINGS)


def next_id(filings: pd.DataFrame, second: bool, span: tuple[str, str] | None) -> str | None:
    rows = filings
    if span:
        rows = rows[(rows["gold_id"] >= span[0]) & (rows["gold_id"] <= span[1])]
    if second:
        rows = rows[(rows["double"] == "yes") & (rows["second_status"] != "labelled")]
    else:
        rows = rows[~rows["status"].isin(gold.DONE)]
    return None if rows.empty else rows["gold_id"].iloc[0]


# The session


def _name(text: str) -> str:
    if not text.strip():
        raise Retry("type your name")
    return text.strip()


def _count(text: str) -> int:
    if not text.strip().isdigit():
        raise Retry("type a whole number")
    return int(text)


def label(
    gold_id: str,
    data_dir: Path,
    gold_dir: Path = gold.GOLD_DIR,
    ask: Ask = input,
    out=print,
    by: str = "",
    second: bool = False,
    redo: bool = False,
    show_all: bool = False,
) -> bool:
    """Label one filing; True if saved."""
    filings = _read(gold_dir, gold.FILINGS)
    match = filings[filings["gold_id"] == gold_id]
    if match.empty:
        out(f"{gold_id} is not in filings.csv")
        return False
    f = match.iloc[0]
    if second and f["double"] != "yes":
        out(f"{gold_id} is not one of the double-labelled filings (double = yes)")
        return False
    done = f["second_status"] == "labelled" if second else f["status"] in gold.DONE
    if done and not redo:
        out(f"{gold_id} is already labelled; add --redo to replace those labels")
        return False

    text = extract.filing_text(data_dir, f["ticker"], f["accession_no"])
    if not text.strip():
        out(f"No text for {f['ticker']} {f['accession_no']} under {data_dir}")
        return False
    sentences = numbered_sentences(text)
    ids = list(range(len(sentences))) if show_all else relevant(sentences)

    out("")
    out(
        f"=== {gold_id}  {f['ticker']}  {f['company_name']}  {f['form_type']} "
        f"items {f['items']}  filed {f['acceptance_datetime_et']}"
    )
    out(f"    {f['source_url']}")
    if not second:  # the second labeller reads the filing without the picking notes
        out(f"    picked for: {f['picked_for']}  (hint: {f['note']})")
    out(f"    {len(ids)} of {len(sentences)} sentences shown; * = mentions a dividend or buyback")
    out("")
    show(sentences, ids, out)
    out("")
    out("Rules: common stock only; past repurchase activity is not an event; a special paid")
    out("with the regular dividend is two events; a dividend increase = the NEW per-share amount;")
    out("a buyback increase = the ADDED dollar amount (total in notes).")
    out("Count each decision once, even if the 8-K and its press release both state it.")

    if not by:
        by = ask_until(ask, "Your name (for labelled_by): ", _name, out)
    n = ask_until(ask, "How many events in this filing? (0 = none): ", _count, out)
    events = []
    for k in range(n):
        out(f"\n-- event {k + 1} of {n}")
        events.append(checked_event(ask, sentences, gold_id, f["accession_no"], text, out))

    out(f"\n{gold_id}: {n} event(s)")
    for e in events:
        fields = {
            k: v
            for k, v in e.model_dump().items()
            if v not in (None, "") and k not in ("gold_id", "accession_no", "quote")
        }
        out(f"  {fields}")
    if not ask_until(ask, "Save? [Y/n]: ", lambda t: parse_yes_no(t, True), out):
        out("Not saved.")
        return False
    save(gold_dir, gold_id, events, by, second)
    out(f"Saved {gold_id} to {'events_second.csv' if second else 'events.csv'}.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("gold_id", help="a gold id such as G05, or 'next'")
    parser.add_argument("--range", help="with next: e.g. G01-G30")
    parser.add_argument("--second", action="store_true", help="second labeller (double = yes)")
    parser.add_argument("--redo", action="store_true", help="replace earlier labels")
    parser.add_argument("--all", action="store_true", help="show every sentence of the filing")
    parser.add_argument("--by", default=os.environ.get(NAME_ENV, ""), help="your name")
    args = parser.parse_args(argv)

    value = os.environ.get(DATA_DIR_ENV)
    if not value:
        print(f"Set {DATA_DIR_ENV} to the data folder first (it has text/edgar/).")
        return 1
    data_dir = Path(value)
    span = tuple(args.range.upper().split("-")) if args.range else None
    while True:
        gold_id = args.gold_id.upper()
        if gold_id == "NEXT":
            gold_id = next_id(gold.load_filings(), args.second, span)
            if gold_id is None:
                print("Nothing left to label in that range.")
                return 0
        saved = label(
            gold_id, data_dir, by=args.by, second=args.second, redo=args.redo, show_all=args.all
        )
        if args.gold_id.upper() != "NEXT" or not saved:
            return 0 if saved else 1
        if not parse_yes_no(input("\nNext filing? [Y/n]: "), True):
            return 0


if __name__ == "__main__":
    sys.exit(main())
