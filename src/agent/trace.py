"""Append-only JSONL trace of one research session: trace_dir/<session_id>.jsonl.

Event kinds: session_start, model_turn, tool_call, notebook, nudge, session_end.
Every line has seq, ts, session_id and kind. Raw provider blocks (signatures etc.)
are never written; only the readable thinking text is kept, in its own field.
"""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.llm import LLMResponse


class Trace:
    def __init__(self, session_id: str, trace_dir: str | Path | None = None):
        """trace_dir=None keeps events in memory only (tests, dry runs)."""
        self.session_id = session_id
        self.events: list[dict] = []
        self.path = Path(trace_dir) / f"{session_id}.jsonl" if trace_dir else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields: Any) -> dict:
        event = {
            "seq": len(self.events),
            "ts": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "kind": kind,
            **fields,
        }
        self.events.append(event)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        return event

    def session_start(
        self,
        config: dict,
        system: str,
        universe_id: str,
        model: str | None,
        max_tokens: int | None,
        effort: str | None,
        code_version: dict | None = None,
        thinking_display: str | None = None,
        mask_dates_in_text: bool | None = None,
        **extra: Any,
    ) -> dict:
        """system is the system prompt; extra holds session facts (research system, period)."""
        settings = {"model": model, "max_tokens": max_tokens, "effort": effort,
                    "thinking_display": thinking_display,
                    "mask_dates_in_text": mask_dates_in_text}
        return self.log(
            "session_start",
            config=config,
            system=system,
            universe_id=universe_id,
            settings=settings,
            code_version=code_version,
            **extra,
        )

    def model_turn(self, step: int, response: LLMResponse) -> dict:
        return self.log(
            "model_turn",
            step=step,
            text=response.text,
            thinking=response.thinking,
            tool_calls=[c.model_dump() for c in response.tool_calls],
            stop_reason=response.stop_reason,
            usage=response.usage.model_dump(),
        )

    def tool_call(
        self,
        step: int,
        name: str,
        arguments: dict,
        result: str,
        is_error: bool,
        duration_ms: float,
        **extra: Any,
    ) -> dict:
        """extra: per-tool measurements, e.g. hypothesis_id on run_backtest."""
        return self.log(
            "tool_call",
            step=step,
            name=name,
            arguments=arguments,
            result=result,
            is_error=is_error,
            duration_ms=round(duration_ms, 1),
            **extra,
        )

    def nudge(self, step: int, text: str) -> dict:
        return self.log("nudge", step=step, text=text)

    def session_end(self, reason: str) -> dict:
        return self.log("session_end", reason=reason, totals=self.totals())

    def totals(self) -> dict:
        turns = [e for e in self.events if e["kind"] == "model_turn"]
        calls = [e for e in self.events if e["kind"] == "tool_call"]
        backtests = [e for e in calls if e["name"] == "run_backtest" and not e["is_error"]]
        notes = [e for e in self.events if e["kind"] == "notebook"]
        claim_notes = [e for e in notes if e["entry_kind"] == "claim"]
        claimed = {e["data"].get("hypothesis_id") for e in claim_notes}
        quotes = [q for e in claim_notes for q in e["data"].get("evidence", [])]
        by_method = {m: [q for q in quotes if q.get("grounding_method") == m]
                     for m in ("event", "text")}
        return {
            "steps": len(turns),
            "tool_calls": len(calls),
            "invalid_calls": sum(e["is_error"] for e in calls),
            "input_tokens": sum(e["usage"]["input_tokens"] for e in turns),
            "cache_read_input_tokens": sum(e["usage"]["cache_read_input_tokens"] for e in turns),
            "cache_creation_input_tokens": sum(
                e["usage"]["cache_creation_input_tokens"] for e in turns
            ),
            "output_tokens": sum(e["usage"]["output_tokens"] for e in turns),
            "nudges": sum(e["kind"] == "nudge" for e in self.events),
            "stop_reasons": dict(Counter(e["stop_reason"] for e in turns)),
            # protocol adherence: successful backtests by whether their signal cites a hypothesis
            "backtests_linked_to_hypothesis": sum(e.get("hypothesis_id") is not None
                                                  for e in backtests),
            "backtests_unlinked": sum(e.get("hypothesis_id") is None for e in backtests),
            "hypotheses_recorded": sum(e["entry_kind"] == "hypothesis" for e in notes),
            "hypotheses_with_claim": len(claimed - {None}),
            # evidence grounding: quotes that are exact substrings of earlier tool results
            "evidence_quotes": len(quotes),
            "evidence_quotes_grounded": sum(q["grounded"] for q in quotes),
            "evidence_quotes_event": len(by_method["event"]),
            "evidence_quotes_event_grounded": sum(q["grounded"] for q in by_method["event"]),
            "evidence_quotes_text": len(by_method["text"]),
            "evidence_quotes_text_grounded": sum(q["grounded"] for q in by_method["text"]),
        }


def read_trace(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
