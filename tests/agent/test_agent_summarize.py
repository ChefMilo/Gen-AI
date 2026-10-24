import json

from agent.summarize_trace import main, summarize

USAGE = {"input_tokens": 10, "output_tokens": 50, "cache_read_input_tokens": 0,
         "cache_creation_input_tokens": 0}
SPEC = {"event_type": "dividend", "direction": "long", "holding_days": 5,
        "conditions": [{"field": "dividend_change_pct", "op": ">", "value": 10},
                       {"field": "mcap_bucket", "op": "==", "value": "small"}]}
LONG_THINKING = "Small caps may react more to dividend news. " * 20


def tool(step, name, arguments, result, is_error=False, **extra):
    return {"kind": "tool_call", "step": step, "name": name, "arguments": arguments,
            "result": result, "is_error": is_error, "duration_ms": 1.0, **extra}


def turn(step, thinking="", text=""):
    return {"kind": "model_turn", "step": step, "text": text, "thinking": thinking,
            "tool_calls": [], "stop_reason": "tool_use", "usage": USAGE}


TRUNCATED_BACKTEST = ('{"signal_id": "sig-1", "n_events": 59, "mean_car": 0.017, "t_stat": 3.16, '
                      '"p_value_raw": 0.0016, "hit_rate": 0.66, "event_ids": ["e1", "e2", '
                      '"e3"[truncated]')
EVENTS = [
    {"kind": "session_start", "session_id": "abc123", "agent_system": "R2",
     "universe_id": "fake_planted_seed0", "settings": {"model": "m", "max_tokens": 16000,
                                                       "effort": None},
     "code_version": {"commit": "ca957c1", "dirty": False}, "prompt_version": "0123456789ab",
     "research_start": "2016-01-01", "research_end": "2022-12-31"},
    turn(1, thinking=LONG_THINKING),
    tool(1, "record_hypothesis", {"hypothesis": "Small-cap hikes", "rationale": "r"},
         '{"recorded": true, "hypothesis_id": "h1"}'),
    tool(1, "query_events", {"event_type": "dividend", "conditions": SPEC["conditions"]},
         '{"n_events": 59, "by_year": {}, "sample": []}'),
    turn(2),
    tool(2, "define_signal", {"spec": SPEC, "hypothesis_id": "h1"},
         '{"signal_id": "sig-1", "hypothesis_id": "h1"}'),
    tool(2, "run_backtest", {"signal_id": "sig-1"}, TRUNCATED_BACKTEST, hypothesis_id="h1"),
    turn(3),
    tool(3, "inspect_results", {"signal_id": "sig-1", "by": "mcap_bucket"},
         json.dumps({"signal_id": "sig-1", "by": "mcap_bucket",
                     "groups": {"small": {"n_events": 59, "mean_car": 0.017}}})),
    tool(3, "buy_stock", {"ticker": "X"}, "Unknown tool 'buy_stock'. Available tools: finish",
         is_error=True),
    {"kind": "nudge", "step": 3, "text": "You have not called finish."},
    turn(4),
    {"kind": "notebook", "entry_kind": "claim", "step": 4, "summary": "sig-1: robust",
     "data": {"signal_id": "sig-1", "hypothesis_id": "h1", "n_evidence": 2, "evidence": [
         {"event_id": "e1", "doc_ref": None, "quote": "raised the dividend", "grounded": True},
         {"event_id": "e2", "doc_ref": None, "quote": "made up", "grounded": False}]}},
    tool(4, "report_finding", {"signal_id": "sig-1", "rationale": "robust", "evidence": []},
         '{"accepted": true, "claims_so_far": 1}', evidence_grounded=[True, False]),
    tool(4, "finish", {"reason": "found it"}, '{"finished": true, "claims": 1}'),
    {"kind": "session_end", "reason": "finished",
     "totals": {"steps": 4, "evidence_quotes": 2, "evidence_quotes_grounded": 1}},
]


def test_header_shows_settings_code_and_prompt_version():
    out = summarize(EVENTS)
    assert "session abc123  system R2  universe fake_planted_seed0" in out
    assert "code ca957c1  prompt_version 0123456789ab  research 2016-01-01 to 2022-12-31" in out


def test_timeline_lines():
    out = summarize(EVENTS)
    assert "- record_hypothesis -> h1: Small-cap hikes" in out
    assert ('- query_events dividend: dividend_change_pct > 10, mcap_bucket == "small" '
            "-> n=59") in out
    assert ('- define_signal [h1] dividend: dividend_change_pct > 10, mcap_bucket == "small" '
            "| long 5d -> sig-1") in out
    assert "- inspect_results sig-1 by mcap_bucket -> small: n=59 car=0.017" in out
    assert "NUDGE: You have not called finish." in out
    assert "- finish: found it" in out


def test_truncated_backtest_still_shows_what_the_model_saw():
    out = summarize(EVENTS)
    assert ("- run_backtest sig-1 [h1] -> n_events=59 mean_car=0.017 t_stat=3.16 "
            "p_value_raw=0.0016 budget_remaining=(not shown)  [result truncated]") in out


def test_errors_are_shown_in_full():
    out = summarize(EVENTS)
    assert '- buy_stock {"ticker": "X"}' in out
    assert "ERROR: Unknown tool 'buy_stock'. Available tools: finish" in out


def test_thinking_only_with_flag_and_clipped():
    assert "thinking:" not in summarize(EVENTS)
    out = summarize(EVENTS, thinking=True)
    line = next(x for x in out.splitlines() if "thinking: Small caps" in x)
    assert line.endswith("...") and len(line) < 330
    assert "thinking: (no thinking text recorded)" in out


def test_totals_claims_with_grounding_and_end_reason():
    out = summarize(EVENTS)
    assert "  evidence_quotes_grounded: 1" in out
    assert "claims (1):" in out and "sig-1 [h1] step 4: sig-1: robust" in out
    assert '- [grounded] e1: "raised the dividend"' in out
    assert '- [NOT grounded] e2: "made up"' in out
    assert out.splitlines()[-1] == "end reason: finished"


def test_unfinished_trace_says_so():
    assert summarize(EVENTS[:5]).splitlines()[-1] == (
        "end reason: (no session_end: run did not finish)"
    )


def test_cli_reads_a_file_and_writes_nothing(tmp_path, capsys):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in EVENTS) + "\n", encoding="utf-8")
    before = path.read_bytes()
    assert main([str(path), "--thinking"]) == 0
    assert "end reason: finished" in capsys.readouterr().out
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.jsonl"]
