"""Research notebook: the agent's own record of what it tried, which it can read back.

Entries come from tool handlers (hypotheses, signals, backtest results, claims), never
from the model's thinking or from raw provider blocks. Every entry is also traced.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from agent.trace import Trace

NotebookKind = Literal["hypothesis", "signal_defined", "backtest_result", "claim", "note"]

SUMMARY_MAX_CHARS = 200
FORBIDDEN_KEYS = {"thinking", "raw", "signature"}


def _forbidden_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = FORBIDDEN_KEYS & set(value)
        for v in value.values():
            found |= _forbidden_keys(v)
        return found
    if isinstance(value, list):
        return set().union(*(_forbidden_keys(v) for v in value))
    return set()


class NotebookEntry(BaseModel):
    kind: NotebookKind
    step: int
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def _no_model_internals(cls, data: dict[str, Any]) -> dict[str, Any]:
        if found := _forbidden_keys(data):
            raise ValueError(f"notebook data may not contain {sorted(found)}")
        return data


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


class Notebook:
    def __init__(self, trace: Trace | None = None):
        self.entries: list[NotebookEntry] = []
        self.trace = trace

    def add(
        self, kind: NotebookKind, step: int, summary: str, data: dict[str, Any] | None = None
    ) -> NotebookEntry:
        entry = NotebookEntry(
            kind=kind, step=step, summary=_clip(summary, SUMMARY_MAX_CHARS), data=data or {}
        )
        self.entries.append(entry)
        if self.trace:
            self.trace.log(
                "notebook", entry_kind=kind, step=step, summary=entry.summary, data=entry.data
            )
        return entry

    def render(self, max_entries: int | None = None, max_data_chars: int = 300) -> str:
        """Compact text for the model: one line per entry, data clipped."""
        if not self.entries:
            return "(notebook is empty)"
        shown = self.entries if max_entries is None else self.entries[-max_entries:]
        lines = []
        if len(shown) < len(self.entries):
            lines.append(f"({len(self.entries) - len(shown)} earlier entries not shown)")
        for e in shown:
            lines.append(f"[step {e.step}] {e.kind}: {e.summary}")
            if e.data:
                data = json.dumps(e.data, default=str, separators=(",", ":"))
                lines.append("    " + _clip(data, max_data_chars))
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps([e.model_dump() for e in self.entries], default=str)
