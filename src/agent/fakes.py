"""Local fakes for the gated tools, so the agent loop can run before the real layers land.

Built on Member 5's generators (src/universes/synthetic.py, fake_events.py), read-only.
Everything here is TEMPORARY and will be replaced by the real contracts:

    FakeSignalSpec, FakeCondition, FakeTextFilter  ->  Member 4's SignalSpec (src/dsl/)
    FakeBacktester                                ->  Member 4's backtest engine + harness
    FakeEventQuery                                ->  Member 1's event store
    FakeRetriever                                 ->  Member 2's retrieval and text features

The planted effect is local to this file and only exists to test the agent end to end;
real planted and placebo universes are Member 5's job. Fake ports raise ToolError for
failures the agent should see (budget used up, bad field values, ...).
TODO(member-3): when real ports land, map their error types to ToolError in the handlers.
"""

import math
import re
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from agent.interim.spec import DIRECTIONS, OPS_BY_KIND, make_spec_models, select
from agent.textmask import maybe_mask
from agent.tools import (
    RAG_SYSTEMS,
    Tool,
    ToolArgs,
    ToolError,
    check_hypothesis_id,
    record_signal,
    register,
)
from universes.fake_events import (
    BUYBACK_RATIONALE_LABELS,
    DIVIDEND_FRAMING_LABELS,
    make_fake_events,
)
from universes.synthetic import make_synthetic_prices

FAKE_START, FAKE_END = "2016-01-04", "2024-12-31"
# TODO(member-3): switch to the agreed research_end once configs/default.yaml has one.
FAKE_RESEARCH_END = date(2022, 12, 31)  # 2023-2024 is the fake holdout
N_TICKERS = 60
NOISE_SD = 0.04
PLANTED_EFFECT = 0.02
PLANTED_SPEC = {
    "event_type": "dividend",
    "conditions": [
        {"field": "dividend_change_pct", "op": ">", "value": 10},
        {"field": "mcap_bucket", "op": "==", "value": "small"},
    ],
    "effect_car_5d": PLANTED_EFFECT,
}
# Event has no sector column yet; the fakes assign one per ticker (real data: universe.csv).
FAKE_SECTORS = ("Industrials", "Financials", "Consumer Discretionary", "Utilities", "Materials")

# --- Fields and text features the agent may use (also shown in the task brief) -----------

# field -> (event type it applies to or None for both, kind, allowed values or None)
FIELDS: dict[str, tuple[str | None, str, tuple | None]] = {
    "dividend_per_share": ("dividend", "number", None),
    "dividend_change_pct": ("dividend", "number", None),
    "dividend_kind": ("dividend", "category", ("regular", "special", "initial")),
    "buyback_amount_usd": ("buyback", "number", None),
    "buyback_pct_mcap": ("buyback", "number", None),
    "is_new_program": ("buyback", "bool", None),
    "mcap_bucket": (None, "category", ("small", "mid", "large")),
    "sector": (None, "category", FAKE_SECTORS),
}
FIELD_NOTES = {
    "dividend_per_share": "USD per share",
    "dividend_change_pct": "% change vs the previous dividend",
    "buyback_amount_usd": "authorised amount, USD",
    "buyback_pct_mcap": "amount as % of market cap",
    "is_new_program": "false = increase or extension",
    "mcap_bucket": "market-cap third the day before the event",
}
TEXT_FEATURES: dict[str, tuple[str, tuple[str, ...]]] = {
    "dividend_framing": ("dividend", DIVIDEND_FRAMING_LABELS),
    "buyback_rationale": ("buyback", BUYBACK_RATIONALE_LABELS),
}
BREAKDOWNS = ("year", "sector", "mcap_bucket")
EVENT_TYPES = ("dividend", "buyback")
HOLDING_DAYS = (5,)  # the fakes only have 5-day CARs


# --- Data ---------------------------------------------------------------------------


def make_fake_universe(seed: int = 0, n_tickers: int = N_TICKERS) -> pd.DataFrame:
    """Usable fake events (new announcements that passed extraction) with the columns the
    fakes need: Event fields plus a derived sector, one column per text feature, car_5d."""
    tickers = [f"FK{i:03d}" for i in range(n_tickers)]
    prices = make_synthetic_prices(tickers, FAKE_START, FAKE_END, seed=seed)
    events = make_fake_events(prices, seed=seed)
    events = events[events["is_new_announcement"] & (events["extraction_status"] == "passed")]
    events = events.reset_index(drop=True)

    sector_of = {t: FAKE_SECTORS[i % len(FAKE_SECTORS)] for i, t in enumerate(tickers)}
    events["sector"] = events["ticker"].map(sector_of)
    for feature in TEXT_FEATURES:
        values = [tf.get(feature, {}) for tf in events["text_features"]]
        # object dtype keeps None for missing values (pandas 3 would store NaN in a str column)
        events[feature] = pd.Series([v.get("label") for v in values], dtype=object)
        events[f"{feature}_quote"] = pd.Series([v.get("quote") for v in values], dtype=object)

    rng = np.random.default_rng([seed, 42])
    noise = rng.normal(0.0, NOISE_SD, len(events))
    events["car_5d"] = noise + PLANTED_EFFECT * planted_mask(events)
    return events


def planted_mask(events: pd.DataFrame) -> pd.Series:
    return (
        (events["event_type"] == "dividend")
        & (events["dividend_change_pct"] > 10)
        & (events["mcap_bucket"] == "small")
    )


# --- Temporary stand-ins for Member 4's SignalSpec ------------------------------------


FakeCondition, FakeTextFilter, FakeSignalSpec = make_spec_models(
    FIELDS, TEXT_FEATURES, EVENT_TYPES, HOLDING_DAYS, prefix="Fake"
)


def event_study(cars: pd.Series) -> dict[str, Any]:
    """Mean, t-stat and a two-sided p-value (normal approximation; fake only)."""
    n = len(cars)
    if n < 2:
        return {"n_events": n, "mean_car": None, "t_stat": None, "p_value_raw": None,
                "hit_rate": None}
    mean, sd = float(cars.mean()), float(cars.std(ddof=1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
    return {
        "n_events": n,
        "mean_car": round(mean, 5),
        "t_stat": round(t, 3),
        "p_value_raw": round(math.erfc(abs(t) / math.sqrt(2)), 5),
        "hit_rate": round(float((cars > 0).mean()), 3),
    }


# --- Fake ports ------------------------------------------------------------------------


def _compact(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    return value.isoformat() if isinstance(value, date) else value


def _research_only(events: pd.DataFrame, research_end: date | None) -> pd.DataFrame:
    """Stand-in for the harness's time lockout: the fakes never show later events.
    None leaves the end open."""
    if research_end is None:
        return events.reset_index(drop=True)
    return events[events["first_tradeable_date"] <= research_end].reset_index(drop=True)


class FakeBacktester:
    def __init__(self, events: pd.DataFrame, budget: int, research_end: date = FAKE_RESEARCH_END):
        self.events = _research_only(events, research_end)
        self.budget = budget
        self.tests_used = 0
        self.specs: dict[str, FakeSignalSpec] = {}
        self.results: dict[str, pd.DataFrame] = {}

    def define(self, spec: dict[str, Any]) -> str:
        signal_id = f"sig-{len(self.specs) + 1}"
        self.specs[signal_id] = FakeSignalSpec.model_validate(spec)
        return signal_id

    def _spec(self, signal_id: str) -> FakeSignalSpec:
        if signal_id not in self.specs:
            known = ", ".join(self.specs) or "none yet"
            raise ToolError(f"unknown signal_id {signal_id!r}. Defined signals: {known}")
        return self.specs[signal_id]

    def run(self, signal_id: str) -> dict[str, Any]:
        spec = self._spec(signal_id)
        if self.tests_used >= self.budget:
            raise ToolError(f"testing budget exhausted ({self.budget} backtests used)")
        self.tests_used += 1
        selected = select(self.events, spec)
        sign = 1.0 if spec.direction == "long" else -1.0
        self.results[signal_id] = selected.assign(car=sign * selected["car_5d"])
        return {
            "signal_id": signal_id,
            **event_study(self.results[signal_id]["car"]),
            "event_ids": list(selected["event_id"]),  # all of them
            "tests_used": self.tests_used,
            "budget_remaining": self.budget - self.tests_used,
        }

    def breakdown(self, signal_id: str, by: str) -> dict[str, Any]:
        self._spec(signal_id)
        if signal_id not in self.results:
            raise ToolError(f"run_backtest {signal_id!r} before inspecting it")
        if by not in BREAKDOWNS:
            raise ToolError(f"by must be one of {list(BREAKDOWNS)}")
        rows = self.results[signal_id]
        keys = pd.to_datetime(rows["first_tradeable_date"]).dt.year if by == "year" else rows[by]
        groups = {str(k): event_study(g["car"]) for k, g in rows.groupby(keys)}
        return {"signal_id": signal_id, "by": by, "groups": groups}

    def signal_options(self) -> dict[str, Any]:
        return {
            "event_types": list(EVENT_TYPES),
            "operators": {kind: list(ops) for kind, ops in OPS_BY_KIND.items()},
            "holding_days": list(HOLDING_DAYS),
            "directions": list(DIRECTIONS),
            "breakdowns": list(BREAKDOWNS),
        }

    def budget_status(self) -> dict[str, int]:
        return {"budget": self.budget, "tests_used": self.tests_used}


class FakeEventQuery:
    def __init__(self, events: pd.DataFrame, research_end: date = FAKE_RESEARCH_END):
        self.events = _research_only(events, research_end)

    def query(self, event_type: str, conditions: list[dict], limit: int) -> dict[str, Any]:
        spec = FakeSignalSpec(event_type=event_type, conditions=conditions)
        rows = select(self.events, spec)
        cols = ["event_id", "ticker", "first_tradeable_date", *[c.field for c in spec.conditions]]
        cols = list(dict.fromkeys(cols))
        years = pd.to_datetime(rows["first_tradeable_date"]).dt.year.value_counts().sort_index()
        sample = rows[cols].head(limit)
        return {
            "n_events": len(rows),
            "by_year": {str(k): int(v) for k, v in years.items()},
            "sample_columns": cols,  # compact: one list per row, not one dict per row
            "sample_rows": [[_compact(v) for v in row] for row in sample.itertuples(index=False)],
        }

    def existing_event_ids(self, event_ids: list[str]) -> set[str]:
        return set(event_ids) & set(self.events["event_id"])

    def available_fields(self) -> dict[str, str]:
        out = {}
        for name, (applies_to, kind, allowed) in FIELDS.items():
            parts = [kind, f"{applies_to} events" if applies_to else "all events"]
            if allowed:
                parts.append("one of " + "/".join(allowed))
            if name in FIELD_NOTES:
                parts.append(FIELD_NOTES[name])
            out[name] = "; ".join(parts)
        return out


class FakeRetriever:
    def __init__(self, events: pd.DataFrame, research_end: date = FAKE_RESEARCH_END):
        self.events = _research_only(events, research_end)

    @staticmethod
    def _passages(row: pd.Series) -> list[str]:
        quotes = list((row["evidence"] or {}).values())
        quotes += [tf["quote"] for tf in (row["text_features"] or {}).values()]
        return quotes

    def search(self, query: str, filters: dict[str, Any], k: int) -> list[dict[str, Any]]:
        words = set(re.findall(r"[a-z]+", query.lower()))
        rows = self.events
        for key in ("ticker", "event_type"):
            if filters.get(key):
                rows = rows[rows[key] == filters[key]]
        hits = []
        for _, row in rows.iterrows():
            for text in self._passages(row):
                score = len(words & set(re.findall(r"[a-z]+", text.lower())))
                if score:
                    hits.append((score, row, text))
        hits.sort(key=lambda h: (-h[0], h[1]["first_tradeable_date"], h[1]["event_id"]))
        return [
            {
                "event_id": row["event_id"],
                "doc_ref": f"{row['ticker']}/{row['accession_no']}/ex99-1.txt",
                "date": row["first_tradeable_date"],
                "text": text,
            }
            for _, row, text in hits[:k]
        ]

    def text_feature(self, event_ids: list[str], feature: str) -> list[dict[str, Any]]:
        """Full per-event values; the tool handler shows the model a compact view."""
        if feature not in TEXT_FEATURES:
            raise ToolError(f"unknown feature {feature!r}. Available: {list(TEXT_FEATURES)}")
        rows = self.events[self.events["event_id"].isin(event_ids)]
        return [
            {"event_id": r["event_id"], "label": r[feature], "quote": r[f"{feature}_quote"]}
            for _, r in rows.iterrows()
            if r[feature] is not None
        ]

    def available_text_features(self) -> dict[str, list[str]]:
        return {name: list(labels) for name, (_, labels) in TEXT_FEATURES.items()}


# --- Smoke-test verdict ------------------------------------------------------------------
# Stand-in for Member 5's matcher (src/evaluation/): event-set Jaccard only. The real rule
# also requires the same event type, the right sign and passing the corrected threshold.


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def planted_event_ids(backtester: FakeBacktester) -> set[str]:
    """Planted events the agent could see (research period only)."""
    return set(backtester.events.loc[planted_mask(backtester.events), "event_id"])


def best_planted_overlap(backtester: FakeBacktester, claims) -> tuple[float, str | None]:
    """(best Jaccard, signal_id) over the claimed signals' full backtested event sets."""
    planted = planted_event_ids(backtester)
    best, best_id = 0.0, None
    for claim in claims:
        rows = backtester.results.get(claim.signal_id)
        if rows is not None and (score := jaccard(set(rows["event_id"]), planted)) > best:
            best, best_id = score, claim.signal_id
    return best, best_id


def make_fake_ports(seed: int = 0, budget: int = 15, research_end: date = FAKE_RESEARCH_END):
    """(backtester, events, retriever, universe_id) over one fake planted universe."""
    events = make_fake_universe(seed)
    return (
        FakeBacktester(events, budget, research_end),
        FakeEventQuery(events, research_end),
        FakeRetriever(events, research_end),
        f"fake_planted_seed{seed}",
    )


# --- Tool arguments and handlers ---------------------------------------------------------


class QueryEventsArgs(ToolArgs):
    event_type: Literal[EVENT_TYPES]  # type: ignore[valid-type]
    conditions: list[FakeCondition] = Field(default_factory=list, max_length=5)
    limit: int = Field(10, ge=0, le=20)


class DefineSignalArgs(ToolArgs):
    spec: FakeSignalSpec
    hypothesis_id: str | None = Field(None, description="The hypothesis this signal tests.")


class SignalIdArgs(ToolArgs):
    signal_id: str


class InspectResultsArgs(ToolArgs):
    signal_id: str
    by: Literal[BREAKDOWNS]  # type: ignore[valid-type]


class SearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    ticker: str | None = None
    event_type: Literal[EVENT_TYPES] | None = None  # type: ignore[valid-type]
    k: int = Field(5, ge=1, le=10)


class TextFeatureArgs(ToolArgs):
    event_ids: list[str] = Field(min_length=1, max_length=200)
    feature: Literal[tuple(TEXT_FEATURES)]  # type: ignore[valid-type]


def query_events(session, args: QueryEventsArgs) -> dict:
    conditions = [c.model_dump() for c in args.conditions]
    return session.events.query(args.event_type, conditions, args.limit)


def define_signal(session, args: DefineSignalArgs) -> dict:
    check_hypothesis_id(session, args.hypothesis_id)
    spec = args.spec.model_dump()
    signal_id = session.backtester.define(spec)
    record_signal(session, signal_id, spec, args.hypothesis_id)
    return {"signal_id": signal_id, "hypothesis_id": args.hypothesis_id}


MODEL_BACKTEST_KEYS = ("signal_id", "n_events", "mean_car", "t_stat", "p_value_raw",
                       "hit_rate", "tests_used", "budget_remaining")
EXAMPLE_EVENT_IDS = 3
TEXT_FEATURE_EXAMPLES = 5


def backtest_view(result: dict, min_events: int | None) -> dict:
    """What the model sees: stats and budget first, warnings, a few example event_ids."""
    view = {k: result.get(k) for k in MODEL_BACKTEST_KEYS}
    warnings = []
    if result["n_events"] < 2:
        warnings.append("too few events for statistics")
    if min_events and result["n_events"] < min_events:
        warnings.append(f"n_events below the minimum of {min_events}; cannot be a finding")
    view["warnings"] = warnings
    view["example_event_ids"] = result["event_ids"][:EXAMPLE_EVENT_IDS]
    return view


def run_backtest(session, args: SignalIdArgs) -> dict:
    result = session.backtester.run(args.signal_id)  # full result, every event_id
    session.tested_signals.add(args.signal_id)  # only reached on success
    session.signal_event_ids[args.signal_id] = list(result["event_ids"])
    session.call_extra["event_ids"] = list(result["event_ids"])  # trace only, not the model
    view = backtest_view(result, session.config.get("min_events"))
    summary = (f"{args.signal_id}: n={view['n_events']}, mean CAR={view['mean_car']}, "
               f"t={view['t_stat']}")
    data = {k: v for k, v in view.items() if k != "example_event_ids"}
    session.notebook.add("backtest_result", session.step, summary, data)
    return view


def inspect_results(session, args: InspectResultsArgs) -> dict:
    return session.backtester.breakdown(args.signal_id, args.by)


def _mask_flag(session) -> bool:
    return bool(session.agent_config.get("mask_dates_in_text", False))


def search_announcements(session, args: SearchArgs) -> dict:
    filters = {"ticker": args.ticker, "event_type": args.event_type}
    hits = session.retriever.search(args.query, filters, args.k)
    mask = _mask_flag(session)
    return {"results": [{**h, "text": maybe_mask(h["text"], mask)} for h in hits]}


def get_text_feature(session, args: TextFeatureArgs) -> dict:
    """Label counts over every requested event, plus a few examples with quotes."""
    mask = _mask_flag(session)
    values = [{**v, "quote": maybe_mask(v["quote"], mask)}
              for v in session.retriever.text_feature(args.event_ids, args.feature)]
    counts: dict[str, int] = {}
    for v in values:
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    return {"feature": args.feature, "n_events": len(values), "label_counts": counts,
            "examples": values[:TEXT_FEATURE_EXAMPLES]}


def register_fake_tools() -> None:
    """Register the six gated tools against whatever ports the session holds."""
    for tool in (
        Tool("query_events", QueryEventsArgs, query_events),
        Tool("define_signal", DefineSignalArgs, define_signal),
        Tool("run_backtest", SignalIdArgs, run_backtest),
        Tool("inspect_results", InspectResultsArgs, inspect_results),
        Tool("search_announcements", SearchArgs, search_announcements, systems=RAG_SYSTEMS),
        Tool("get_text_feature", TextFeatureArgs, get_text_feature, systems=RAG_SYSTEMS),
    ):
        register(tool)
