"""Build events.parquet from the candidate filings (docs/data_schema.md, "Event store").

    python -m ingest.run estimate        # filings, tokens and cost; no API calls
    python -m ingest.run gold            # the 60 gold filings, normal API; scores them
    python -m ingest.run submit          # all candidates to the Batch API (half price)
    python -m ingest.run status          # is the batch done?
    python -m ingest.run collect         # fetch results, retry, derive, save events.parquet

Add --limit N to try a few filings first. Replies are cached in results/extract/, so a
rerun only pays for what changed. Needs $GENAI_DATA_DIR and, except estimate, an
Anthropic key (.env).
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

from ingest import derive, extract, gold
from ingest.llm import LLM, REPO_ROOT, default_model
from store import events as store_events
from store.prices import UNIVERSE_QUALITY, USABLE, data_dir, trading_calendar
from store.schema import Event

OUT = REPO_ROOT / "results" / "extract"
CACHE = OUT / "llm_cache"
BATCH = OUT / "batch.json"
FILING_INDEX = Path("index/filing_index.parquet")
UNIVERSE = Path("universe.csv")
PRICES = Path("prices/prices.parquet")
WORKERS = 8
CHARS_PER_TOKEN = 4  # rough, for the estimate


def candidates(base: Path, limit: int | None = None) -> pd.DataFrame:
    """Dividend and buyback candidate filings of keep / keep_from companies."""
    index = pd.read_parquet(base / FILING_INDEX)
    quality = pd.read_csv(base / UNIVERSE_QUALITY)
    keep = set(quality.loc[quality["recommendation"].isin(USABLE), "ticker"])
    rows = index[(index["is_div_candidate"] | index["is_buyback_candidate"])]
    rows = rows[rows["ticker"].isin(keep)].sort_values(["acceptance_datetime_et", "ticker"])
    return rows.head(limit) if limit else rows


def _request(base: Path, row: dict) -> tuple[str, str, str]:
    """(custom_id, user message, excerpts) for one filing."""
    excerpts = extract.passages(extract.filing_text(base, row["ticker"], row["accession_no"]))
    filed = str(row["acceptance_datetime_et"])[:10]
    message = extract.user_message(row["company_name"], row["form_type"], filed, excerpts)
    return row["accession_no"], message, excerpts


def _requests(base: Path, rows: pd.DataFrame) -> list[tuple[str, str, str]]:
    records = rows.to_dict("records")
    with ThreadPoolExecutor(WORKERS) as pool:  # reading the text from Drive is the slow part
        return list(pool.map(lambda r: _request(base, r), records))


def _first_messages(requests):
    return [(cid, extract.SYSTEM, [{"role": "user", "content": m}]) for cid, m, _ in requests]


def run_extraction(base: Path, rows: pd.DataFrame, llm: LLM) -> tuple[pd.DataFrame, list[dict]]:
    """Events (Event columns, derived fields empty) and the filings that failed."""
    requests = {cid: (m, x) for cid, m, x in _requests(base, rows)}
    calendar = trading_calendar()
    now = datetime.now().replace(microsecond=0)
    version = f"{extract.EXTRACTOR_VERSION}+{llm.model}"

    def one(row: dict):
        message, excerpts = requests[row["accession_no"]]
        filed = datetime.fromisoformat(str(row["acceptance_datetime_et"])).date()
        if not excerpts.strip():
            return [], None
        try:
            found = extract.extract(llm.ask, message, excerpts, filed)
        except Exception as error:  # noqa: BLE001 - one bad filing must not stop the run
            return [], {
                "accession_no": row["accession_no"],
                "ticker": row["ticker"],
                "error": f"{type(error).__name__}: {error}"[:500],
            }
        return extract.to_events(found, row, calendar, now, version), None

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(one, rows.to_dict("records")))
    events = [e.model_dump() for found, _ in results for e in found]
    failures = [f for _, f in results if f]
    frame = pd.DataFrame(events, columns=list(Event.model_fields))
    return frame, failures


def _derive(base: Path, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    prices = pd.read_parquet(base / PRICES, columns=["ticker", "date", "mcap_usd", "mcap_bucket"])
    universe = pd.read_csv(base / UNIVERSE)
    return derive.derive(events, prices, universe)


def _report(events: pd.DataFrame, failures: list[dict], llm: LLM) -> dict:
    cost = llm.usage.cost(llm.model)
    usable = events[events["is_new_announcement"].astype(bool)] if len(events) else events
    report = {
        "model": llm.model,
        "events": len(events),
        "new_announcements": len(usable),
        "by_type": events["event_type"].value_counts().to_dict() if len(events) else {},
        "flagged": int((events["extraction_status"] == "flagged").sum()) if len(events) else 0,
        "failed_filings": len(failures),
        "usage": llm.usage.__dict__,
        "estimated_cost_usd": None if cost is None else round(cost, 2),
    }
    print(json.dumps(report, indent=2, default=str))
    return report


def cmd_estimate(base: Path, limit: int | None) -> None:
    rows = candidates(base, limit)
    requests = _requests(base, rows)
    system = len(extract.SYSTEM) / CHARS_PER_TOKEN
    tokens_in = sum(len(m) for _, m, _ in requests) / CHARS_PER_TOKEN + system * len(requests)
    tokens_out = 250 * len(requests)  # compact JSON, about one or two events per filing
    empty = sum(1 for _, _, x in requests if not x.strip())
    print(f"{len(requests)} filings ({empty} with no matching sentences)")
    print(f"~{tokens_in / 1e6:.1f}M input tokens, ~{tokens_out / 1e6:.1f}M output tokens")
    print("Cost (Sonnet 5.5, third-party prices; check the console):")
    full = (tokens_in * 2 + tokens_out * 10) / 1e6
    print(f"  normal API ~${full:.0f}, Batch API ~${full / 2:.0f}, plus ~10% for retries")


def cmd_gold(base: Path, limit: int | None) -> None:
    filings = gold.load_filings()
    index = pd.read_parquet(base / FILING_INDEX)
    rows = index[index["accession_no"].isin(filings["accession_no"])]
    rows = rows.head(limit) if limit else rows
    llm = LLM(CACHE)
    events, failures = run_extraction(base, rows, llm)
    events = _derive(base, events)
    OUT.mkdir(parents=True, exist_ok=True)
    store_events.save_events(events, OUT / "gold_events.parquet")
    _report(events, failures, llm)
    labels = gold.load_events()
    done = filings["status"].isin(gold.DONE).sum()
    if not done:
        print("No gold filings labelled yet: label some, then rerun to score (cached, free).")
        return
    predicted = [Event.model_validate(r) for r in _as_records(events)]
    result = gold.score(filings, labels, predicted)
    print(f"Score on {done} labelled gold filings:")
    print(json.dumps(result, indent=2, default=str))


def _as_records(events: pd.DataFrame) -> list[dict]:
    def clean(v):
        return None if (not isinstance(v, list | dict) and pd.isna(v)) else v

    return [{k: clean(v) for k, v in r.items()} for r in events.to_dict("records")]


def cmd_submit(base: Path, limit: int | None) -> None:
    rows = candidates(base, limit)
    requests = [r for r in _requests(base, rows) if r[2].strip()]
    llm = LLM(CACHE)
    batch_id = llm.submit_batch(_first_messages(requests))
    OUT.mkdir(parents=True, exist_ok=True)
    BATCH.write_text(json.dumps({"batch_id": batch_id, "limit": limit, "model": llm.model}))
    if batch_id is None:
        print("Every filing is already cached: run collect.")
    else:
        print(f"Submitted {len(requests)} filings as batch {batch_id} ({llm.model}).")
        print("Check with: python -m ingest.run status")


def cmd_status() -> None:
    state = json.loads(BATCH.read_text())
    print(json.dumps(LLM(CACHE, model=state["model"]).batch_status(state["batch_id"]), indent=2))


def cmd_collect(base: Path) -> None:
    state = json.loads(BATCH.read_text())
    llm = LLM(CACHE, model=state["model"])
    rows = candidates(base, state["limit"])
    if state["batch_id"]:
        status = llm.batch_status(state["batch_id"])
        if status["status"] != "ended":
            print(f"Batch not finished yet: {status}")
            return
        requests = [r for r in _requests(base, rows) if r[2].strip()]
        failed = llm.collect_batch(state["batch_id"], _first_messages(requests))
        print(f"Collected the batch; {len(failed)} requests failed and will be retried live.")
    events, failures = run_extraction(base, rows, llm)  # cached; only retries cost
    events = _derive(base, events)
    path = store_events.save_events(events, None if not state["limit"] else OUT / "events.parquet")
    report = _report(events, failures, llm)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
    pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)
    print(f"Saved {len(events)} events to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["estimate", "gold", "submit", "status", "collect"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N filings")
    args = parser.parse_args(argv)
    if args.command == "status":
        cmd_status()
        return 0
    base = data_dir()
    print(f"Data: {base}  Model: {default_model()}")
    {
        "estimate": lambda: cmd_estimate(base, args.limit),
        "gold": lambda: cmd_gold(base, args.limit),
        "submit": lambda: cmd_submit(base, args.limit),
        "collect": lambda: cmd_collect(base),
    }[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
