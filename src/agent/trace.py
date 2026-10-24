"""Append-only JSONL trace of one research session: trace_dir/<session_id>.jsonl.

Event kinds: session_start, model_turn, tool_call, notebook, session_end.
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
        **extra: Any,
    ) -> dict:
        """system is the system prompt; extra holds session facts (research system, period)."""
        settings = {"model": model, "max_tokens": max_tokens, "effort": effort}
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

    def session_end(self, reason: str) -> dict:
        return self.log("session_end", reason=reason, totals=self.totals())

    def totals(self) -> dict:
        turns = [e for e in self.events if e["kind"] == "model_turn"]
        calls = [e for e in self.events if e["kind"] == "tool_call"]
        backtests = [e for e in calls if e["name"] == "run_backtest" and not e["is_error"]]
        notes = [e for e in self.events if e["kind"] == "notebook"]
        claimed = {e["data"].get("hypothesis_id") for e in notes if e["entry_kind"] == "claim"}
        return {
            "steps": len(turns),
            "tool_calls": len(calls),
            "invalid_calls": sum(e["is_error"] for e in calls),
            "input_tokens": sum(e["usage"]["input_tokens"] for e in turns),
            "output_tokens": sum(e["usage"]["output_tokens"] for e in turns),
            "stop_reasons": dict(Counter(e["stop_reason"] for e in turns)),
            # protocol adherence: successful backtests by whether their signal cites a hypothesis
            "backtests_linked_to_hypothesis": sum(e.get("hypothesis_id") is not None
                                                  for e in backtests),
            "backtests_unlinked": sum(e.get("hypothesis_id") is None for e in backtests),
            "hypotheses_recorded": sum(e["entry_kind"] == "hypothesis" for e in notes),
            "hypotheses_with_claim": len(claimed - {None}),
        }


def read_trace(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
