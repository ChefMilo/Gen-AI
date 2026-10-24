import json

import pytest

from agent import fakes
from agent.fakes import (
    FAKE_RESEARCH_END,
    PLANTED_SPEC,
    FakeBacktester,
    FakeEventQuery,
    FakeRetriever,
    make_fake_universe,
)
from agent.tools import dispatch, schemas_for
from store.schema import Event
from universes.fake_events import BUYBACK_RATIONALE_LABELS, DIVIDEND_FRAMING_LABELS

NULL_SPEC = {  # no planted effect: pure noise
    "event_type": "dividend",
    "conditions": [{"field": "mcap_bucket", "op": "==", "value": "large"}],
}


def call(s, name, **arguments):
    content, is_error = dispatch(s, name, arguments)
    return (content if is_error else json.loads(content)), is_error


# --- Universe ------------------------------------------------------------------------


def test_universe_keeps_only_usable_events(events):
    assert events["is_new_announcement"].all()
    assert (events["extraction_status"] == "passed").all()
    assert set(events["event_type"]) == {"dividend", "buyback"}


def test_universe_is_deterministic(events):
    again = make_fake_universe(seed=0)
    assert again["event_id"].equals(events["event_id"])
    assert again["car_5d"].equals(events["car_5d"])
    assert not make_fake_universe(seed=1)["car_5d"].equals(events["car_5d"])


def test_derived_columns(events):
    assert set(events["sector"]) == set(fakes.FAKE_SECTORS)
    assert events.groupby("ticker")["sector"].nunique().max() == 1
    dividends = events[events["event_type"] == "dividend"]
    buybacks = events[events["event_type"] == "buyback"]
    assert set(dividends["dividend_framing"]) <= set(DIVIDEND_FRAMING_LABELS)
    assert set(buybacks["buyback_rationale"]) <= set(BUYBACK_RATIONALE_LABELS)
    assert dividends["buyback_rationale"].isna().all()
    assert 0.035 < events["car_5d"].std() < 0.045


def test_fields_are_event_columns_and_labels_come_from_member_5(events):
    fields = FakeEventQuery(events).available_fields()
    assert set(fields) - {"sector"} <= set(Event.model_fields)
    assert {c["field"] for c in PLANTED_SPEC["conditions"]} <= set(fields)
    assert FakeRetriever(events).available_text_features() == {
        "dividend_framing": list(DIVIDEND_FRAMING_LABELS),
        "buyback_rationale": list(BUYBACK_RATIONALE_LABELS),
    }


def test_ports_never_show_events_after_research_end(events):
    assert events["first_tradeable_date"].max() > FAKE_RESEARCH_END  # holdout exists
    for port in (FakeBacktester(events, 1), FakeEventQuery(events), FakeRetriever(events)):
        assert port.events["first_tradeable_date"].max() <= FAKE_RESEARCH_END


# --- Planted vs null -------------------------------------------------------------------


def test_planted_signal_is_significant_and_positive(events):
    bt = FakeBacktester(events, budget=5)
    result = bt.run(bt.define({k: PLANTED_SPEC[k] for k in ("event_type", "conditions")}))
    assert result["n_events"] >= 20
    assert result["mean_car"] > 0.01
    assert result["t_stat"] > 2.5 and result["p_value_raw"] < 0.05


def test_null_signal_is_not_significant(events):
    bt = FakeBacktester(events, budget=5)
    result = bt.run(bt.define(NULL_SPEC))
    assert result["n_events"] > 100
    assert abs(result["t_stat"]) < 2 and result["p_value_raw"] > 0.05


def test_short_direction_flips_the_sign(events):
    bt = FakeBacktester(events, budget=5)
    spec = {k: PLANTED_SPEC[k] for k in ("event_type", "conditions")}
    long = bt.run(bt.define(spec))
    short = bt.run(bt.define({**spec, "direction": "short"}))
    assert short["mean_car"] == pytest.approx(-long["mean_car"])


# --- Through the dispatcher --------------------------------------------------------------


def test_full_research_path_links_hypothesis_to_claim(fake_session):
    s = fake_session()
    h, _ = call(s, "record_hypothesis", hypothesis="Small-cap dividend hikes",
                rationale="hikes signal confidence where information is scarce")
    spec = {k: PLANTED_SPEC[k] for k in ("event_type", "conditions")}
    defined, is_error = call(s, "define_signal", spec=spec, hypothesis_id=h["hypothesis_id"])
    assert not is_error and defined == {"signal_id": "sig-1", "hypothesis_id": "h1"}
    assert s.signals["sig-1"]["conditions"][0]["field"] == "dividend_change_pct"

    result, is_error = call(s, "run_backtest", signal_id="sig-1")
    assert not is_error and result["t_stat"] > 2.5 and result["budget_remaining"] == 14
    assert s.tested_signals == {"sig-1"}

    by_year, is_error = call(s, "inspect_results", signal_id="sig-1", by="year")
    assert not is_error and set(by_year["groups"]) <= {str(y) for y in range(2016, 2023)}

    claim, is_error = call(s, "report_finding", signal_id="sig-1", rationale="robust",
                           evidence=[{"event_id": result["event_ids"][0], "quote": "raised"}])
    assert not is_error and claim["accepted"]

    kinds = [e.kind for e in s.notebook.entries]
    assert kinds == ["hypothesis", "signal_defined", "backtest_result", "claim"]
    backtest = next(e for e in s.trace.events if e.get("name") == "run_backtest")
    assert backtest["hypothesis_id"] == "h1"
    totals = s.trace.totals()
    assert totals["backtests_linked_to_hypothesis"] == 1 and totals["backtests_unlinked"] == 0
    assert totals["hypotheses_recorded"] == 1 and totals["hypotheses_with_claim"] == 1


def test_define_signal_without_hypothesis_is_allowed_but_unlinked(fake_session):
    s = fake_session()
    call(s, "define_signal", spec=NULL_SPEC)
    call(s, "run_backtest", signal_id="sig-1")
    assert s.signal_hypotheses == {"sig-1": None}
    assert s.trace.totals()["backtests_unlinked"] == 1


def test_unknown_hypothesis_id_is_a_tool_error(fake_session):
    s = fake_session()
    content, is_error = call(s, "define_signal", spec=NULL_SPEC, hypothesis_id="h9")
    assert is_error and "unknown hypothesis_id 'h9'" in content
    assert s.signals == {} and s.invalid_calls == 1


def test_budget_exhaustion_is_returned_as_an_error(fake_session):
    s = fake_session(budget=2)
    call(s, "define_signal", spec=NULL_SPEC)
    call(s, "define_signal", spec={**NULL_SPEC, "direction": "short"})
    assert not call(s, "run_backtest", signal_id="sig-1")[1]
    assert not call(s, "run_backtest", signal_id="sig-2")[1]
    call(s, "define_signal", spec=NULL_SPEC)
    content, is_error = call(s, "run_backtest", signal_id="sig-3")
    assert is_error
    assert content == "Error in run_backtest: testing budget exhausted (2 backtests used)"
    assert s.tested_signals == {"sig-1", "sig-2"} and s.invalid_calls == 1


def test_failed_backtest_does_not_mark_signal_tested(fake_session):
    s = fake_session()
    content, is_error = call(s, "run_backtest", signal_id="sig-7")
    assert is_error and "unknown signal_id 'sig-7'" in content
    assert s.tested_signals == set() and s.backtester.tests_used == 0


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"event_type": "dividend", "conditions": [{"field": "mcap_bucket", "op": "==",
                                                     "value": "tiny"}]},
         "mcap_bucket must be one of ['small', 'mid', 'large']"),
        ({"event_type": "dividend", "conditions": [{"field": "sector", "op": ">", "value": 3}]},
         "sector is a category field; use one of: == != in"),
        ({"event_type": "buyback", "conditions": [{"field": "dividend_change_pct", "op": ">",
                                                    "value": 10}]},
         "dividend_change_pct only applies to dividend events"),
        ({"event_type": "dividend", "conditions": [{"field": "colour", "op": "==", "value": 1}]},
         "spec.conditions.0.field: Input should be"),
        ({"event_type": "dividend", "text_filter": {"feature": "dividend_framing",
                                                    "equals": "sustainable"}},
         "dividend_framing labels are ['fake_routine', 'fake_growth', 'fake_cautious']"),
    ],
)
def test_bad_specs_explain_themselves(fake_session, spec, message):
    content, is_error = call(fake_session(), "define_signal", spec=spec)
    assert is_error and message in content


def test_inspect_before_run_is_an_error(fake_session):
    s = fake_session()
    call(s, "define_signal", spec=NULL_SPEC)
    content, is_error = call(s, "inspect_results", signal_id="sig-1", by="sector")
    assert is_error and "run_backtest 'sig-1' before inspecting" in content


def test_query_events_counts_without_using_budget(fake_session):
    s = fake_session()
    result, is_error = call(s, "query_events", event_type="dividend",
                            conditions=PLANTED_SPEC["conditions"], limit=3)
    assert not is_error and result["n_events"] >= 20 and len(result["sample"]) == 3
    assert set(result["sample"][0]) == {"event_id", "ticker", "first_tradeable_date",
                                        "dividend_change_pct", "mcap_bucket"}
    assert s.backtester.tests_used == 0


def test_r2_cannot_use_retrieval_but_r3_can(fake_session):
    r2 = fake_session("R2")
    assert "search_announcements" not in [t["name"] for t in schemas_for("R2")]
    content, is_error = call(r2, "search_announcements", query="repurchase")
    assert is_error and content.startswith("Unknown tool")

    r3 = fake_session("R3")
    found, is_error = call(r3, "search_announcements", query="repurchase million", k=3)
    assert not is_error and len(found["results"]) == 3
    assert all(r["doc_ref"].endswith("/ex99-1.txt") for r in found["results"])
    ids = [found["results"][0]["event_id"]]
    feature, is_error = call(r3, "get_text_feature", event_ids=ids, feature="buyback_rationale")
    assert not is_error and feature["values"][0]["label"] in BUYBACK_RATIONALE_LABELS


def test_all_ten_tools_have_flat_schemas_once_fakes_are_registered(fake_session):
    names = [t["name"] for t in schemas_for("R3")]
    assert len(names) == 10
    assert all("$ref" not in json.dumps(t["input_schema"]) for t in schemas_for("R3"))


# --- No outcome leaks --------------------------------------------------------------------

OUTCOME_KEYS = {"car", "car_5d", "mean_car", "t_stat", "p_value_raw", "hit_rate", "ret",
                "adj_close", "close", "open", "high", "low", "raw_close", "volume"}


def _keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(v) for v in value))
    return set()


def test_condition_fields_exclude_outcomes():
    assert not OUTCOME_KEYS & set(fakes.FIELDS)


def test_only_backtest_tools_expose_outcomes(fake_session):
    s = fake_session("R3")
    conditions = PLANTED_SPEC["conditions"]
    outputs = {
        "record_hypothesis": call(s, "record_hypothesis", hypothesis="h", rationale="r"),
        "query_events": call(s, "query_events", event_type="dividend", conditions=conditions,
                             limit=50),
        "search_announcements": call(s, "search_announcements", query="dividend share", k=20),
        "define_signal": call(s, "define_signal",
                              spec={"event_type": "dividend", "conditions": conditions}),
    }
    ids = [r["event_id"] for r in outputs["search_announcements"][0]["results"]]
    outputs["get_text_feature"] = call(s, "get_text_feature", event_ids=ids,
                                       feature="dividend_framing")
    outputs["view_notebook"] = (dispatch(s, "view_notebook", {})[0], False)
    for name, (result, is_error) in outputs.items():
        assert not is_error, name
        if name == "view_notebook":
            assert not any(k in result for k in ("mean_car", "t_stat")), name  # nothing run yet
        else:
            assert not OUTCOME_KEYS & _keys(result), name

    result, _ = call(s, "run_backtest", signal_id="sig-1")
    assert {"mean_car", "t_stat"} <= set(result)
    groups, _ = call(s, "inspect_results", signal_id="sig-1", by="year")
    assert all("mean_car" in g for g in groups["groups"].values())


# --- Smoke-test matcher (stand-in for Member 5's) ------------------------------------------


def test_jaccard():
    assert fakes.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert fakes.jaccard(set(), set()) == 0.0


def test_best_planted_overlap(events):
    from agent.finding import FindingClaim

    bt = FakeBacktester(events, budget=5)
    planted = bt.define({k: PLANTED_SPEC[k] for k in ("event_type", "conditions")})
    null = bt.define(NULL_SPEC)
    untested = bt.define(NULL_SPEC)
    bt.run(planted)
    bt.run(null)

    def claim(sid):
        return FindingClaim(signal_id=sid, rationale="r", evidence=[], step=1)

    assert fakes.best_planted_overlap(bt, [claim(null), claim(planted)]) == (1.0, planted)
    # large caps never overlap the small-cap planted set; an untested claim has no events
    assert fakes.best_planted_overlap(bt, [claim(null), claim(untested)]) == (0.0, None)
    assert fakes.best_planted_overlap(bt, []) == (0.0, None)
