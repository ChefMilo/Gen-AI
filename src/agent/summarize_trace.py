"""Print a compact, read-only timeline of one session trace.

    python -m agent.summarize_trace results/traces/<session_id>.jsonl [--thinking]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from agent.trace import read_trace

THINKING_CHARS = 300
BACKTEST_KEYS = ("n_events", "mean_car", "t_stat", "p_value_raw", "budget_remaining")


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def parse_result(result: str) -> Any:
    """The tool result as JSON, or None if it is not JSON (e.g. truncated)."""
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return None


def pick(result: str, keys: tuple[str, ...]) -> dict:
    """Top-level numbers from a result, even when it was truncated."""
    parsed = parse_result(result)
    if isinstance(parsed, dict):
        return {k: parsed.get(k, "?") for k in keys}
    found = {}
    for k in keys:
        m = re.search(rf'"{k}": ([^,}}]+)', result)
        found[k] = m.group(1) if m else "(not shown)"
    return found


def conditions(spec: dict) -> str:
    parts = [f"{c['field']} {c['op']} {json.dumps(c['value'])}" for c in spec.get("conditions", [])]
    text = " AND ".join(parts) or "(no conditions)"
    if spec.get("text_filter"):
        tf = spec["text_filter"]
        text += f" AND text {tf['feature']} == {tf['equals']}"
    return text


def describe_call(e: dict) -> str:
    name, args, result = e["name"], e["arguments"], e["result"]
    if e["is_error"]:
        return f"{name} {json.dumps(args, default=str)}\n        ERROR: {result}"
    if not isinstance(args, dict):
        return f"{name} {clip(json.dumps(args, default=str), 120)}"
    out = parse_result(result)
    if name == "record_hypothesis":
        return f"record_hypothesis -> {out['hypothesis_id']}: {clip(args['hypothesis'], 140)}"
    if name == "query_events":
        spec = {"conditions": args.get("conditions", [])}
        n = out["n_events"] if isinstance(out, dict) else pick(result, ("n_events",))["n_events"]
        return f"query_events {args['event_type']}: {conditions(spec)} -> n={n}"
    if name == "define_signal":
        spec = args["spec"]
        sid = out["signal_id"] if isinstance(out, dict) else "?"
        return (f"define_signal [{args.get('hypothesis_id') or 'no hypothesis'}] "
                f"{spec['event_type']}: {conditions(spec)} | {spec.get('direction', 'long')} "
                f"{spec.get('holding_days', 5)}d -> {sid}")
    if name == "run_backtest":
        r = pick(result, BACKTEST_KEYS)
        note = "  [result truncated]" if result.endswith("[truncated]") else ""
        return (f"run_backtest {args['signal_id']} [{e.get('hypothesis_id') or 'unlinked'}] -> "
                + " ".join(f"{k}={v}" for k, v in r.items()) + note)
    if name == "inspect_results" and isinstance(out, dict):
        groups = "; ".join(f"{g}: n={v['n_events']} car={v['mean_car']}"
                           for g, v in out["groups"].items())
        return f"inspect_results {args['signal_id']} by {args['by']} -> {groups}"
    if name == "report_finding":
        return f"report_finding {args['signal_id']}: {clip(args['rationale'], 140)}"
    if name == "finish":
        return f"finish: {clip(args['reason'], 200)}"
    return f"{name} {clip(json.dumps(args, default=str), 120)} -> {clip(result, 120)}"


def summarize(events: list[dict], thinking: bool = False) -> str:
    start = next(e for e in events if e["kind"] == "session_start")
    settings = start.get("settings", {})
    code = start.get("code_version") or {}
    lines = [
        f"session {start['session_id']}  system {start.get('agent_system')}  "
        f"universe {start.get('universe_id')}",
        f"model {settings.get('model')}  max_tokens {settings.get('max_tokens')}  "
        f"effort {settings.get('effort')}",
        f"code {code.get('commit', '?')}{' (dirty)' if code.get('dirty') else ''}  "
        f"prompt_version {start.get('prompt_version') or '(not recorded)'}  "
        f"research {start.get('research_start')} to {start.get('research_end')}",
        "",
    ]
    for e in events:
        if e["kind"] == "model_turn":
            lines.append(f"[{e['step']:>2}] stop={e['stop_reason']}  "
                         f"out_tokens={e['usage']['output_tokens']}")
            if e.get("text"):
                lines.append(f"     text: {clip(e['text'], THINKING_CHARS)}")
            if thinking:
                text = e.get("thinking") or "(no thinking text recorded)"
                lines.append(f"     thinking: {clip(text, THINKING_CHARS)}")
        elif e["kind"] == "tool_call":
            lines.append(f"     - {describe_call(e)}")
        elif e["kind"] == "nudge":
            lines.append(f"     NUDGE: {e['text']}")

    end = next((e for e in reversed(events) if e["kind"] == "session_end"), None)
    lines += ["", "totals:"]
    for k, v in (end["totals"] if end else {}).items():
        lines.append(f"  {k}: {v}")

    claims = [e for e in events if e["kind"] == "notebook" and e["entry_kind"] == "claim"]
    lines += ["", f"claims ({len(claims)}):"]
    for c in claims:
        lines.append(f"  {c['data']['signal_id']} [{c['data'].get('hypothesis_id')}] "
                     f"step {c['step']}: {clip(c['summary'], 300)}")
        for q in c["data"].get("evidence", []):
            source = q.get("event_id") or q.get("doc_ref")
            mark = {True: "grounded", False: "NOT grounded"}.get(q.get("grounded"), "")
            method = q.get("grounding_method", "text")
            detail = ""
            if method == "event":
                detail = (f" (event_exists={q.get('event_exists')}, "
                          f"in_signal={q.get('in_signal')})")
            lines.append(f"    - [{mark}/{method}] {source}: \"{clip(q['quote'], 160)}\"{detail}")
        if "evidence" not in c["data"]:  # traces from before grounding was recorded
            report = next((e for e in events if e.get("name") == "report_finding"
                           and e["step"] == c["step"] and not e["is_error"]), None)
            for q in (report or {}).get("arguments", {}).get("evidence", []):
                source = q.get("event_id") or q.get("doc_ref")
                lines.append(f"    - [grounding not recorded] {source}: "
                             f"\"{clip(q['quote'], 160)}\"")
    lines += ["", f"end reason: {end['reason'] if end else '(no session_end: run did not finish)'}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agent.summarize_trace", description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--thinking", action="store_true", help="show thinking text per turn")
    args = p.parse_args(argv)
    print(summarize(read_trace(args.trace), thinking=args.thinking))
    return 0


if __name__ == "__main__":
    sys.exit(main())
