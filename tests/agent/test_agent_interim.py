"""Interim ports over a universe view: R0 parity, budget, lockout, no leaks, harness."""

import json
from datetime import date

import pandas as pd
import pytest
from scipy import stats

import agent.session as session_module
from agent.finding import EvidenceQuote, Finding, FindingClaim
from agent.interim.backtest import INSUFFICIENT, TESTED, ZERO_VARIANCE, summarize_returns
from agent.interim.data import field_table, is_fake, research_events
from agent.interim.harness import claims_to_findings, corrected_p_values, holm_adjust
from agent.interim.ports import NO_RESEARCH_END, build_ports
from agent.session import new_session
from agent.tools import ToolError, dispatch, tools_for
from evaluation.reference_car import event_cars
from evaluation.systems import UniverseView, make_view
from universes.fake_events import make_fake_events
from universes.planted import PlantSpec, make_planted
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

START, END = date(2015, 1, 1), date(2020, 12, 31)
RAISE = {"event_type": "dividend",
         "conditions": [{"field": "dividend_change_pct", "op": ">", "value": 10}]}


@pytest.fixture(scope="module")
def base():
    tickers = [f"T{i:03d}" for i in range(40)]
    prices = make_synthetic_prices(tickers, START, END, seed=3)
    benchmarks = make_synthetic_benchmarks(START, END, seed=3)
    events = make_fake_events(prices, seed=3)
    spec = PlantSpec(spec_id="dev_raise_up200bp", event_type="dividend",
                     condition="dividend_change_pct > 10", effect=0.02, strength="strong",
                     complexity="single", feature_type="structured")
    planted, truth = make_planted(prices, events, spec, min_events=20, benchmarks=benchmarks)
    return planted, benchmarks, events, truth


@pytest.fixture
def view(base):
    prices, benchmarks, events, _ = base
    return make_view("u-test", prices, benchmarks, events)


def ports_for(view, registry, budget=15, with_retriever=False, mask_dates=True, **kw):
    return build_ports(view, budget=budget, holding_days=5, min_events=20,
                       with_retriever=with_retriever, mask_dates=mask_dates, **kw)


def session_for(ports, system="R2", mask_dates=True, monkeypatch=None):
    config = {"research_start": date(2015, 1, 1), "min_events": 20, "alpha": 0.05,
              "agent": {"model": "m", "tool_result_max_chars": 20000,
                        "mask_dates_in_text": mask_dates}}
    return new_session(system, config, "u-test", ports.backtester, ports.events,
                       retriever=ports.retriever, research_end=ports.research_end or END,
                       in_memory=True)


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(session_module, "code_version", lambda: None)


# --- R0 parity ------------------------------------------------------------------------
# A line-for-line copy of R0's dividend hypothesis (src/baselines/r0.py, PR #17).
# TODO(member-3): import baselines.r0.R0System instead once PR #17 merges.


def r0_dividend_increase(view, holding_days=5, min_events=20):
    events = view.events
    selected = events.loc[events["event_type"].eq("dividend")
                          & events["dividend_change_pct"].gt(10)].copy()
    cars = event_cars(view.prices, view.benchmarks, selected, holding_days)
    valid = cars["car"].notna() & ~cars["car"].isin([float("inf"), float("-inf")])
    cars = cars.loc[valid].copy()
    returns = cars["car"].astype(float)
    if len(returns) < max(min_events, 2) or returns.nunique() == 1:
        return None
    test = stats.ttest_1samp(returns, popmean=0.0)
    return {"event_ids": cars["event_id"].tolist(), "n_events": len(returns),
            "mean_car": float(returns.mean()), "t_stat": float(test.statistic),
            "p_value_raw": float(test.pvalue)}


def test_same_signal_gives_r0s_car_and_p(view, registry):
    ports = ports_for(view, registry)
    result = ports.backtester.run(ports.backtester.define(RAISE))
    r0 = r0_dividend_increase(view)
    assert result["status"] == TESTED
    assert result["event_ids"] == r0["event_ids"]
    assert result["n_events"] == r0["n_events"] > 20
    for key in ("mean_car", "t_stat", "p_value_raw"):
        assert result[key] == r0[key]  # exactly, not approximately


def test_mean_car_is_raw_and_direction_only_flips_the_position(view, registry):
    ports = ports_for(view, registry)
    long = ports.backtester.run(ports.backtester.define(RAISE))
    short = ports.backtester.run(ports.backtester.define({**RAISE, "direction": "short"}))
    assert short["mean_car"] == long["mean_car"] > 0  # planted +2%: the market went up
    assert short["position_mean_car"] == -long["position_mean_car"]
    assert short["t_stat"] == long["t_stat"] and short["p_value_raw"] == long["p_value_raw"]
    assert short["hit_rate"] == pytest.approx(1 - long["hit_rate"])


def test_summarize_returns_matches_r0_rules():
    assert summarize_returns(pd.DataFrame({"car": [0.01, 0.03] * 9}), 20)["status"] == INSUFFICIENT
    assert summarize_returns(pd.DataFrame({"car": [0.02] * 20}), 20)["status"] == ZERO_VARIANCE
    tested = summarize_returns(pd.DataFrame({"car": [0.01, 0.03] * 10}), 20)
    assert tested["status"] == TESTED and tested["t_stat"] == pytest.approx(76 ** 0.5)
    one = summarize_returns(pd.DataFrame({"car": [0.01]}), 0)
    assert one["status"] == INSUFFICIENT  # max(min_events, 2)


# --- Budget, options, breakdowns ------------------------------------------------------


def test_budget_counts_every_run_and_then_refuses(view, registry):
    ports = ports_for(view, registry, budget=2)
    bt = ports.backtester
    sig = bt.define(RAISE)
    bt.run(sig)
    bt.run(sig)  # a rerun is another test
    with pytest.raises(ToolError, match="budget exhausted"):
        bt.run(sig)
    assert bt.budget_status() == {"budget": 2, "tests_used": 2}
    assert len(bt.attempts) == 2


def test_signal_options_offer_only_what_the_data_has(view, registry):
    ports = ports_for(view, registry)
    options = ports.backtester.signal_options()
    assert options["holding_days"] == [5]
    assert options["breakdowns"] == ["year", "mcap_bucket"]  # no sector column
    assert "sector" not in ports.events.available_fields()
    with pytest.raises(ValueError, match="sector"):
        ports.backtester.define({"event_type": "dividend",
                                 "conditions": [{"field": "sector", "op": "==", "value": "x"}]})


def test_breakdown_groups_cover_the_measurable_events(view, registry):
    ports = ports_for(view, registry)
    bt = ports.backtester
    sig = bt.define(RAISE)
    result = bt.run(sig)
    for by in ("year", "mcap_bucket"):
        groups = bt.breakdown(sig, by)["groups"]
        assert sum(g["n_events"] for g in groups.values()) == result["n_events"]
    with pytest.raises(ToolError, match="by must be one of"):
        bt.breakdown(sig, "sector")


# --- Lockout --------------------------------------------------------------------------


def test_nothing_after_research_end_reaches_the_agent(base, registry):
    prices, benchmarks, events, _ = base
    end = date(2018, 6, 29)
    view = make_view("u-test", prices, benchmarks, events, research_start=date(2016, 1, 1),
                     research_end=end)
    ports = ports_for(view, registry, with_retriever=True)
    bt = ports.backtester
    assert pd.to_datetime(bt.prices["date"]).max() <= pd.Timestamp(end)
    assert pd.to_datetime(bt.benchmarks["date"]).max() <= pd.Timestamp(end)
    days = pd.to_datetime(ports.events.events["first_tradeable_date"])
    assert days.min() >= pd.Timestamp(2016, 1, 1) and days.max() <= pd.Timestamp(end)

    q = ports.events.query("dividend", [], 20)
    assert max(int(y) for y in q["by_year"]) <= 2018
    result = bt.run(bt.define(RAISE))
    ids = set(result["event_ids"])
    by_id = ports.events.events.set_index("event_id")["first_tradeable_date"]
    assert all(pd.Timestamp(by_id[i]) <= pd.Timestamp(end) for i in ids)
    hits = ports.retriever.search("dividend", {}, 5)
    assert hits and all(h["filing_date"] <= end.isoformat() for h in hits)
    assert ports.retriever_cutoff == end and ports.warnings == []


def test_windows_running_past_research_end_are_dropped_as_r0_does(base, registry):
    prices, benchmarks, events, _ = base
    end = date(2018, 6, 29)
    view = make_view("u-test", prices, benchmarks, events, research_end=end)
    ports = ports_for(view, registry)
    result = ports.backtester.run(ports.backtester.define({"event_type": "dividend"}))
    days = sorted(pd.to_datetime(ports.backtester.prices["date"]).unique())
    # A 5-day window from the 5th-last trading day still fits; later ones run past the end.
    fits, cut = days[-5], days[-4]
    rows = ports.events.events[ports.events.events["event_type"] == "dividend"]
    ftd = pd.to_datetime(rows["first_tradeable_date"])
    late = set(rows.loc[ftd >= cut, "event_id"])
    assert late, "fixture should have dividends in the last 4 trading days"
    assert not late & set(result["event_ids"])
    assert set(rows.loc[ftd == fits, "event_id"]) <= set(result["event_ids"])
    skipped = event_cars(ports.backtester.prices, ports.backtester.benchmarks,
                         rows[rows["event_id"].isin(late)], 5).attrs["skipped"]
    assert set(skipped["reason"]) == {"past_data"}


def test_open_end_is_recorded_and_cuts_nothing(view, registry):
    ports = ports_for(view, registry, with_retriever=True)
    assert ports.research_end is None and ports.warnings == [NO_RESEARCH_END]
    assert len(ports.backtester.prices) == len(view.prices)
    assert ports.retriever_cutoff == pd.to_datetime(
        view.events["first_tradeable_date"]).max().date()


# --- No leaks -------------------------------------------------------------------------

LEAKY = {"car_5d": 0.5, "injected_car": 0.02, "placebo": True,
         "original_first_tradeable_date": date(2001, 1, 1)}


def test_outcome_and_bookkeeping_columns_never_reach_the_model(view, registry):
    leaky = UniverseView(view.universe_id, view.prices, view.benchmarks,
                         view.events.assign(**LEAKY), None, None)
    ports = ports_for(leaky, registry, with_retriever=True)
    assert not set(LEAKY) & set(field_table(ports.events.events))
    s = session_for(ports, system="R3")
    outputs = []
    calls = [
        ("query_events", {"event_type": "dividend", "conditions": RAISE["conditions"],
                          "limit": 20}),
        ("define_signal", {"spec": RAISE}),
        ("run_backtest", {"signal_id": "sig-1"}),
        ("inspect_results", {"signal_id": "sig-1", "by": "year"}),
        ("inspect_results", {"signal_id": "sig-1", "by": "mcap_bucket"}),
        ("search_announcements", {"query": "dividend increase", "k": 5}),
    ]
    for name, args in calls:
        content, is_error = dispatch(s, name, args)
        assert not is_error, content
        outputs.append(content)
    ids = json.loads(outputs[2])["example_event_ids"]
    content, _ = dispatch(s, "get_text_feature",
                          {"event_ids": ids, "feature": "dividend_framing"})
    outputs.append(content)
    text = "\n".join(outputs)
    for name in [*LEAKY, "2001-01-01", "prices", "adj_close"]:
        assert name not in text


# --- Tools ----------------------------------------------------------------------------


def test_r2_and_r3_tool_sets(view, registry):
    ports_for(view, registry, with_retriever=True)
    r2 = {t.name for t in tools_for("R2")}
    r3 = {t.name for t in tools_for("R3")}
    assert r3 - r2 == {"search_announcements", "get_text_feature"}
    assert {"query_events", "define_signal", "run_backtest", "inspect_results",
            "record_hypothesis", "report_finding", "finish", "view_notebook"} == r2


def test_backtest_view_is_rounded_and_warns(view, registry):
    ports = ports_for(view, registry)
    s = session_for(ports)
    dispatch(s, "define_signal", {"spec": {"event_type": "dividend", "conditions": [
        {"field": "dividend_kind", "op": "==", "value": "initial"}]}})
    content, is_error = dispatch(s, "run_backtest", {"signal_id": "sig-1"})
    out = json.loads(content)
    assert not is_error and out["status"] == INSUFFICIENT
    assert any("cannot be a finding" in w for w in out["warnings"])
    assert len(out["example_event_ids"]) <= 3


def test_search_k_is_capped_at_five(view, registry):
    ports = ports_for(view, registry, with_retriever=True)
    s = session_for(ports, system="R3")
    content, is_error = dispatch(s, "search_announcements", {"query": "dividend", "k": 6})
    assert is_error and "less than or equal to 5" in content
    content, _ = dispatch(s, "search_announcements", {"query": "dividend"})
    assert len(json.loads(content)["results"]) == 3


# --- Harness: Holm and claims -> Findings ---------------------------------------------


def test_holm_matches_r0():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]) == []
    assert holm_adjust([0.02]) == pytest.approx([0.02])


def attempt(i, signal_id, status, p, mean=0.01, direction="long"):
    return {"test_index": i, "signal_id": signal_id, "status": status,
            "spec": {"event_type": "dividend", "conditions": [], "direction": direction,
                     "holding_days": 5},
            "n_events": 30 if status == TESTED else 5, "mean_car": mean,
            "t_stat": 3.0 if status == TESTED else None,
            "p_value_raw": p if status == TESTED else None,
            "event_ids": [f"e{i}-{k}" for k in range(30 if status == TESTED else 5)]}


def claim(signal_id):
    return FindingClaim(signal_id=signal_id, rationale="r",
                        evidence=[EvidenceQuote(event_id="e1-0", quote="q")], step=3)


def test_one_strong_signal_and_untestable_ones_get_r0s_corrected_p():
    attempts = [attempt(1, "sig-1", TESTED, 0.004), attempt(2, "sig-2", INSUFFICIENT, None),
                attempt(3, "sig-3", ZERO_VARIANCE, None), attempt(4, "sig-4", INSUFFICIENT, None)]
    # R0: untestable attempts stay in the family at p = 1
    r0 = holm_adjust([0.004, 1.0, 1.0, 1.0])
    assert corrected_p_values(attempts, correct=True) == r0
    findings, report = claims_to_findings([claim("sig-1")], attempts, session_id="s",
                                          system="R2", alpha=0.05)
    assert findings[0].p_value_corrected == pytest.approx(0.016) == r0[0]
    assert findings[0].passed_correction and report["tests_in_family"] == 4


def test_passed_correction_is_r0s_rule_at_the_boundary():
    attempts = [attempt(1, "sig-1", TESTED, 0.025), attempt(2, "sig-2", TESTED, 0.6)]
    findings, _ = claims_to_findings([claim("sig-1")], attempts, session_id="s",
                                     system="R2", alpha=0.05)
    assert findings[0].p_value_corrected == pytest.approx(0.05)
    assert findings[0].passed_correction  # <= alpha, as R0


def test_untestable_claims_are_not_returned_but_reported():
    attempts = [attempt(1, "sig-1", TESTED, 0.001), attempt(2, "sig-2", INSUFFICIENT, None)]
    findings, report = claims_to_findings([claim("sig-1"), claim("sig-2")], attempts,
                                          session_id="s", system="R2", alpha=0.05)
    assert [f.signal_id for f in findings] == ["s-sig-1"]
    assert [c["returned"] for c in report["claims"]] == [True, False]


def test_r3u_skips_the_correction():
    attempts = [attempt(i, f"sig-{i}", TESTED, 0.03) for i in range(1, 6)]
    findings, report = claims_to_findings([claim("sig-1")], attempts, session_id="s",
                                          system="R3-U", alpha=0.05, correct=False)
    assert findings[0].p_value_corrected == findings[0].p_value_raw == 0.03
    assert findings[0].passed_correction and report["correction"] == "none"


def test_every_finding_field_is_filled_and_effect_car_is_raw():
    attempts = [attempt(1, "sig-1", TESTED, 0.001, mean=-0.015, direction="short")]
    findings, report = claims_to_findings([claim("sig-1")], attempts, session_id="abc",
                                          system="R3", alpha=0.05)
    f = findings[0]
    assert Finding.model_validate(f.model_dump()) == f
    assert all(getattr(f, name) is not None for name in Finding.model_fields)
    assert f.effect_car == -0.015 and f.direction == "short"  # raw: the market fell
    assert (f.finding_id, f.session_id, f.signal_id) == ("abc-f0", "abc", "abc-sig-1")
    assert f.signal_spec.event_type == "dividend" and f.signal_spec.benchmark == "SPY"
    assert f.n_events == len(f.event_ids) == 30
    assert report["claims"][0]["spec"]["direction"] == "short"  # full spec kept


def test_session_claims_become_findings_end_to_end(view, registry):
    ports = ports_for(view, registry)
    s = session_for(ports)
    dispatch(s, "define_signal", {"spec": RAISE})
    content, _ = dispatch(s, "run_backtest", {"signal_id": "sig-1"})
    first = json.loads(content)["example_event_ids"][0]
    content, is_error = dispatch(s, "report_finding", {
        "signal_id": "sig-1", "rationale": "raises signal confidence",
        "evidence": [{"event_id": first, "quote": "raised"}]})
    assert not is_error, content
    findings, _ = claims_to_findings(s.claims, ports.backtester.attempts,
                                     session_id=s.session_id, system="R2", alpha=0.05)
    r0 = r0_dividend_increase(view)
    assert findings[0].event_ids == r0["event_ids"]
    assert findings[0].effect_car == r0["mean_car"]
    assert findings[0].p_value_corrected == r0["p_value_raw"]  # one test: Holm = raw
    assert findings[0].passed_correction


# --- Data helpers ---------------------------------------------------------------------


def test_research_events_keeps_usable_rows_and_flattens_text_features(base):
    _, _, events, _ = base
    rows = research_events(events, None, None)
    assert rows["is_new_announcement"].all() and (rows["extraction_status"] == "passed").all()
    assert len(rows) < len(events)  # flagged and not-new rows exist in the fake data
    assert {"dividend_framing", "dividend_framing_quote", "buyback_rationale"} <= set(rows)
    assert is_fake(rows)
