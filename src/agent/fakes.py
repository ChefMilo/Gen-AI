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
from pydantic import Field, model_validator

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
DIRECTIONS = ("long", "short")
OPS_BY_KIND = {
    "number": ("==", "!=", ">", ">=", "<", "<=", "in"),
    "category": ("==", "!=", "in"),
    "bool": ("==", "!="),
}


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
        events[feature] = [v.get("label") for v in values]
        events[f"{feature}_quote"] = [v.get("quote") for v in values]

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


class FakeCondition(ToolArgs):
    field: Literal[tuple(FIELDS)]  # type: ignore[valid-type]
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in"]
    value: bool | float | str | list[bool | float | str]

    @model_validator(mode="after")
    def _check_value(self) -> "FakeCondition":
        _, kind, allowed = FIELDS[self.field]
        values = self.value if isinstance(self.value, list) else [self.value]
        if (self.op == "in") != isinstance(self.value, list):
            raise ValueError("use a list value with 'in', and a single value otherwise")
        if self.op not in OPS_BY_KIND[kind]:
            ops = " ".join(OPS_BY_KIND[kind])
            raise ValueError(f"{self.field} is a {kind} field; use one of: {ops}")
        if allowed and any(v not in allowed for v in values):
            raise ValueError(f"{self.field} must be one of {list(allowed)}")
        return self


class FakeTextFilter(ToolArgs):
    feature: Literal[tuple(TEXT_FEATURES)]  # type: ignore[valid-type]
    equals: str

    @model_validator(mode="after")
    def _check_label(self) -> "FakeTextFilter":
        labels = TEXT_FEATURES[self.feature][1]
        if self.equals not in labels:
            raise ValueError(f"{self.feature} labels are {list(labels)}")
        return self


class FakeSignalSpec(ToolArgs):
    event_type: Literal[EVENT_TYPES]  # type: ignore[valid-type]
    conditions: list[FakeCondition] = Field(default_factory=list, max_length=5)
    text_filter: FakeTextFilter | None = None
    holding_days: Literal[HOLDING_DAYS] = HOLDING_DAYS[0]  # type: ignore[valid-type]
    direction: Literal[DIRECTIONS] = "long"  # type: ignore[valid-type]

    @model_validator(mode="after")
    def _fields_match_event_type(self) -> "FakeSignalSpec":
        for c in self.conditions:
            applies_to = FIELDS[c.field][0]
            if applies_to not in (None, self.event_type):
                raise ValueError(f"{c.field} only applies to {applies_to} events")
        if self.text_filter and TEXT_FEATURES[self.text_filter.feature][0] != self.event_type:
            raise ValueError(f"{self.text_filter.feature} only applies to "
                             f"{TEXT_FEATURES[self.text_filter.feature][0]} events")
        return self


def select(events: pd.DataFrame, spec: FakeSignalSpec) -> pd.DataFrame:
    mask = events["event_type"] == spec.event_type
    for c in spec.conditions:
        col = events[c.field]
        if c.op == "in":
            mask &= col.isin(c.value)
        else:
            mask &= {
                "==": col == c.value, "!=": col != c.value, ">": col > c.value,
                ">=": col >= c.value, "<": col < c.value, "<=": col <= c.value,
            }[c.op]
    if spec.text_filter:
        mask &= events[spec.text_filter.feature] == spec.text_filter.equals
    return events[mask]


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


def _research_only(events: pd.DataFrame, research_end: date) -> pd.DataFrame:
    """Stand-in for the harness's time lockout: the fakes never show later events."""
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
            "event_ids": list(selected["event_id"][:200]),
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
        years = pd.to_datetime(rows["first_tradeable_date"]).dt.year.value_counts().sort_index()
        return {
            "n_events": len(rows),
            "by_year": {str(k): int(v) for k, v in years.items()},
            "sample": rows[list(dict.fromkeys(cols))].head(limit).to_dict(orient="records"),
        }

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
    limit: int = Field(10, ge=0, le=50)


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
    k: int = Field(5, ge=1, le=20)


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


def run_backtest(session, args: SignalIdArgs) -> dict:
    result = session.backtester.run(args.signal_id)
    session.tested_signals.add(args.signal_id)  # only reached on success
    summary = (f"{args.signal_id}: n={result['n_events']}, mean CAR={result['mean_car']}, "
               f"t={result['t_stat']}")
    data = {k: v for k, v in result.items() if k != "event_ids"}
    session.notebook.add("backtest_result", session.step, summary, data)
    return result


def inspect_results(session, args: InspectResultsArgs) -> dict:
    return session.backtester.breakdown(args.signal_id, args.by)


def search_announcements(session, args: SearchArgs) -> dict:
    filters = {"ticker": args.ticker, "event_type": args.event_type}
    return {"results": session.retriever.search(args.query, filters, args.k)}


def get_text_feature(session, args: TextFeatureArgs) -> dict:
    return {"feature": args.feature,
            "values": session.retriever.text_feature(args.event_ids, args.feature)}


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
