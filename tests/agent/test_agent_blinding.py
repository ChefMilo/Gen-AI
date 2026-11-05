"""Nothing the model sees may reveal the universe: its id, seed or type.

A FakeLLM session runs every tool (successes, errors, the notebook, a nudge) on a
planted and on a placebo universe with telling ids, and every string the model received
(system prompt, tool schemas, brief, tool results, nudges) is checked.
"""

import json
import re
from datetime import date

import pytest

import agent.session as session_module
from agent.interim.ports import build_ports
from agent.llm import FakeLLM, LLMResponse, ToolCall, Usage
from agent.loop import run_session
from agent.session import new_session
from evaluation.systems import make_view
from universes.fake_events import make_fake_events
from universes.placebo import make_placebo
from universes.planted import PlantSpec, make_planted
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

SEED = 7319
FORBIDDEN = [r"planted", r"placebo", r"holdout", r"\bdev\b", r"fake", r"synthetic",
             r"real[-_ ]universe", r"\bseed", str(SEED)]
RAISE = [{"field": "dividend_change_pct", "op": ">", "value": 10}]


def turn(*calls, text=""):
    tool_calls = [ToolCall(id=f"t{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)]
    return LLMResponse(text=text, tool_calls=tool_calls,
                       stop_reason="tool_use" if tool_calls else "end_turn",
                       usage=Usage(input_tokens=10, output_tokens=5))


SCRIPT = [
    turn(("record_hypothesis", {"hypothesis": "Dividend raises", "rationale": "signalling"})),
    turn(("query_events", {"event_type": "dividend", "conditions": RAISE, "limit": 20}),
         ("query_events", {"event_type": "dividend",
                           "conditions": [{"field": "mcap_bucket", "op": "==", "value": "x"}]}),
         ("search_announcements", {"query": "dividend record date payable", "k": 5}),
         ("search_announcements", {"query": "repurchase undervalued", "k": 5})),
    turn(("define_signal", {"spec": {"event_type": "dividend", "conditions": RAISE},
                            "hypothesis_id": "h1"}),
         ("define_signal", {"spec": {"event_type": "dividend", "text_filter": {
             "feature": "dividend_framing", "equals": "nonsense"}}})),
    turn(("run_backtest", {"signal_id": "sig-1"}), ("run_backtest", {"signal_id": "sig-9"})),
    turn(("inspect_results", {"signal_id": "sig-1", "by": "year"}),
         ("inspect_results", {"signal_id": "sig-1", "by": "mcap_bucket"}),
         ("inspect_results", {"signal_id": "sig-1", "by": "sector"})),
    turn(("get_text_feature", {"event_ids": ["EVENT_IDS"], "feature": "dividend_framing"}),
         ("get_text_feature", {"event_ids": ["x"], "feature": "made_up"})),
    turn(text="Thinking about it."),  # no tool call: gets the nudge
    turn(("view_notebook", {}), ("buy_stock", {"ticker": "X"})),
    turn(("report_finding", {"signal_id": "sig-1", "rationale": "raises signal confidence",
                             "evidence": [{"event_id": "EVENT_ID", "quote": "raised"}]})),
    turn(("finish", {"reason": "done"})),
]


@pytest.fixture(scope="module")
def universes():
    tickers = [f"T{i:03d}" for i in range(30)]
    start, end = date(2015, 1, 1), date(2019, 12, 31)
    prices = make_synthetic_prices(tickers, start, end, seed=SEED)
    benchmarks = make_synthetic_benchmarks(start, end, seed=SEED)
    events = make_fake_events(prices, seed=SEED)
    spec = PlantSpec(spec_id=f"dev_planted_seed{SEED}", event_type="dividend",
                     condition="dividend_change_pct > 10", effect=0.02, strength="strong",
                     complexity="single", feature_type="structured", seed=SEED)
    planted_prices, _ = make_planted(prices, events, spec, min_events=20,
                                     benchmarks=benchmarks)
    placebo_events, _ = make_placebo(prices, events, seed=SEED)
    return {
        f"dev_planted_seed{SEED}_u01": make_view(f"dev_planted_seed{SEED}_u01",
                                                planted_prices, benchmarks, events),
        f"dev_placebo_holdout_seed{SEED}_u02": make_view(
            f"dev_placebo_holdout_seed{SEED}_u02", prices, benchmarks, placebo_events),
    }


def script_for(view_events):
    """SCRIPT with real event ids filled in."""
    ids = list(view_events.loc[view_events["event_type"] == "dividend", "event_id"][:3])
    out = []
    for r in SCRIPT:
        calls = []
        for c in r.tool_calls:
            args = json.loads(json.dumps(c.arguments).replace('["EVENT_IDS"]', json.dumps(ids))
                              .replace('"EVENT_ID"', json.dumps(ids[0])))
            calls.append(c.model_copy(update={"arguments": args}))
        out.append(r.model_copy(update={"tool_calls": calls}))
    return out


def everything_the_model_saw(llm: FakeLLM) -> str:
    parts = []
    for call in llm.calls:
        parts += [call["system"], json.dumps(call["tools"])]
        for m in call["messages"]:
            if m["role"] == "user":
                parts.append(m["text"])
            elif m["role"] == "tool_results":
                parts += [r["content"] for r in m["results"]]
    return "\n".join(parts)


@pytest.mark.parametrize("kind", ["planted", "placebo"])
def test_no_message_reveals_the_universe(universes, kind, registry, monkeypatch):
    monkeypatch.setattr(session_module, "code_version", lambda: None)
    universe_id = next(u for u in universes if kind in u)
    view = universes[universe_id]
    ports = build_ports(view, budget=5, holding_days=5, min_events=20, with_retriever=True,
                        mask_dates=True)
    config = {"research_start": date(2015, 1, 1), "min_events": 20,
              "agent": {"model": "m", "max_steps": 20, "tool_result_max_chars": 4000,
                        "mask_dates_in_text": True}}
    session = new_session("R3", config, universe_id, ports.backtester, ports.events,
                          retriever=ports.retriever, research_end=date(2019, 12, 31),
                          in_memory=True)
    llm = FakeLLM(script_for(ports.events.events))
    run_session(session, llm)

    assert session.trace.events[-1]["reason"] == "finished"
    assert session.trace.events[0]["universe_id"] == universe_id  # kept in the trace
    tool_events = [e for e in session.trace.events if e["kind"] == "tool_call"]
    assert any(e["is_error"] for e in tool_events)  # errors were exercised too
    assert any(e["kind"] == "nudge" for e in session.trace.events)
    seen = everything_the_model_saw(llm)
    assert universe_id not in seen
    found = [p for p in FORBIDDEN if re.search(p, seen, flags=re.IGNORECASE)]
    assert found == [], [m.group(0) for p in found
                         for m in [re.search(rf".{{0,60}}{p}.{{0,60}}", seen, re.I | re.S)]]
