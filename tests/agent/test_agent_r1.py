"""R1, the single-shot baseline, with scripted FakeLLM replies (no real LLM)."""

import json
from datetime import date

import pytest

import agent.session as session_module
from agent.config import load_config
from agent.llm import FakeLLM, LLMResponse, Usage
from agent.prompts import SYSTEM_PROMPT
from baselines.r1_single_shot import R1System, parse_reply
from evaluation.systems import make_view
from universes.fake_events import make_fake_events
from universes.planted import PlantSpec, make_planted
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

UNIVERSE_ID = "dev_planted_seed4242_u07"
RAISE_SMALL = {"event_type": "dividend", "conditions": [
    {"field": "dividend_change_pct", "op": ">", "value": 10},
    {"field": "mcap_bucket", "op": "==", "value": "small"}]}
BROAD = {"event_type": "dividend"}
BUYBACKS = {"event_type": "buyback", "direction": "short"}


def item(spec, hypothesis="h", rationale="r"):
    return {"hypothesis": hypothesis, "rationale": rationale, "spec": spec}


def reply(payload) -> LLMResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return LLMResponse(text=text, stop_reason="end_turn",
                       usage=Usage(input_tokens=5, cache_creation_input_tokens=3000,
                                   output_tokens=800))


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(session_module, "code_version", lambda: None)


@pytest.fixture(scope="module")
def view():
    start, end, seed = date(2015, 1, 1), date(2020, 12, 31), 9821
    prices = make_synthetic_prices([f"T{i:03d}" for i in range(80)], start, end, seed=seed)
    benchmarks = make_synthetic_benchmarks(start, end, seed=seed)
    events = make_fake_events(prices, seed=seed)
    spec = PlantSpec(spec_id="r1_test", event_type="dividend",
                     condition='dividend_change_pct > 10 and mcap_bucket == "small"',
                     effect=0.03, strength="strong", complexity="conjunction",
                     feature_type="structured")
    planted, _ = make_planted(prices, events, spec, min_events=20, benchmarks=benchmarks)
    return make_view(UNIVERSE_ID, planted, benchmarks, events, date(2016, 1, 1),
                     date(2020, 12, 31))


def run_r1(view, replies, budget=30):
    llms = []

    def factory(agent, seed):
        llms.append(FakeLLM(replies))
        return llms[-1]

    config = load_config()
    config["agent"] |= {"model": "claude-sonnet-5-5", "trace_dir": None}
    run = R1System(config=config, client_factory=factory).run(view, budget, 3)
    return run, llms[0]


def test_valid_list_is_backtested_and_reported_like_r0(view, registry):
    run, llm = run_r1(view, [reply([item(RAISE_SMALL), item(BROAD), item(BUYBACKS)])])
    assert len(llm.calls) == 1 and llm.calls[0]["tools"] == []
    meta = run.metadata
    assert meta["backtests_used"] == 3 and meta["tests_in_family"] == 3
    assert meta["specs_proposed"] == 3 and meta["specs_invalid"] == 0
    assert meta["correction"] == "holm" and meta["end_reason"] == "finished"
    assert len(run.findings) == 3  # every tested signal, flagged (R0 convention)
    raise_small = run.findings[0]
    assert raise_small.system == "R1" and raise_small.passed_correction
    assert raise_small.direction == "long" and run.findings[2].direction == "short"
    assert raise_small.p_value_corrected == pytest.approx(
        min(1.0, 3 * raise_small.p_value_raw))  # the smallest of 3
    for row in meta["strict_correction"]:
        assert {"p_value_corrected_attempted", "p_value_corrected_strict"} <= set(row)
    assert meta["cost_usd"] == pytest.approx((5 * 2.0 + 3000 * 2.5 + 800 * 10.0) / 1e6)


def test_invalid_specs_are_skipped_and_counted(view, registry):
    bad_field = item({"event_type": "dividend",
                      "conditions": [{"field": "sector", "op": "==", "value": "x"}]})
    text = item({"event_type": "buyback",
                 "text_filter": {"feature": "buyback_rationale", "equals": "undervalued"}})
    replies = [reply([item(BROAD), bad_field, text, "not an object",
                      {"hypothesis": "no spec", "rationale": "r"}])]
    run, _ = run_r1(view, replies)
    assert run.metadata["specs_proposed"] == 5 and run.metadata["specs_invalid"] == 4
    assert run.metadata["invalid_calls"] == 4 and run.metadata["backtests_used"] == 1
    assert len(run.findings) == 1


def test_malformed_json_gets_one_retry(view, registry):
    run, llm = run_r1(view, [reply("I would test dividends."), reply([item(BROAD)])])
    assert len(llm.calls) == 2
    assert "not valid JSON" in llm.calls[1]["messages"][-1]["text"]
    assert run.metadata["parse_retries"] == 1 and len(run.findings) == 1


def test_two_malformed_replies_give_zero_findings(view, registry):
    run, llm = run_r1(view, [reply("{oops"), reply('[{"hypothesis": ')])
    assert len(llm.calls) == 2
    assert run.findings == [] and run.metadata["end_reason"] == "parse_failed"
    assert run.metadata["backtests_used"] == 0


def test_budget_is_respected(view, registry):
    specs = [item({"event_type": "dividend", "conditions": [
        {"field": "dividend_change_pct", "op": ">", "value": float(v)}]}) for v in (5, 10, 20, 30)]
    run, _ = run_r1(view, [reply(specs)], budget=2)
    assert run.metadata["backtests_used"] == 2 and run.metadata["specs_over_budget"] == 2
    assert run.metadata["tests_in_family"] == 2


def test_duplicate_specs_are_one_test_and_one_finding(view, registry):
    run, _ = run_r1(view, [reply([item(RAISE_SMALL), item({**RAISE_SMALL, "direction": "short"})])])
    assert run.metadata["backtests_used"] == 1 and run.metadata["duplicate_backtests"] == 1
    assert len(run.findings) == 1


def test_prompt_is_blind_and_r2_style(view, registry):
    _, llm = run_r1(view, [reply([])])
    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    brief = call["messages"][0]["text"]
    for leak in (UNIVERSE_ID, "planted", "placebo", "seed", "4242", "fake"):
        assert leak.lower() not in brief.lower()
    assert "Text features" not in brief and "buyback_rationale" not in brief
    for tool in ("query_events", "define_signal", "inspect_results", "record_hypothesis"):
        assert tool not in brief
    assert "Reply with JSON only" in brief and "at most 30 signals" in brief


def test_parse_reply_tolerates_fences_and_prose():
    assert parse_reply('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_reply('Here you go: [1, 2] done') == [1, 2]
    with pytest.raises(ValueError):
        parse_reply('{"a": 1}')


def test_untestable_signals_count_as_tests_but_are_not_findings(view, registry):
    initial = item({"event_type": "dividend", "conditions": [
        {"field": "dividend_kind", "op": "==", "value": "initial"}]})
    run, _ = run_r1(view, [reply([item(RAISE_SMALL), initial])])
    assert run.metadata["backtests_used"] == 2 and run.metadata["tests_in_family"] == 2
    assert len(run.findings) == 1  # as R0: untestable attempts stay in Holm at p = 1
    statuses = [c["status"] for c in run.metadata["harness"]["claims"]]
    assert statuses == ["tested", "insufficient_events"]
    assert run.findings[0].p_value_corrected == pytest.approx(
        min(1.0, 2 * run.findings[0].p_value_raw))
