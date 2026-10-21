"""Agent tool signatures (interface 4 in docs/design.md section 7)."""

from typing import Any

from backtest.result import BacktestResult
from dsl.signal import SignalSpec

# TODO(member-3): placeholders only. Member 3 owns these signatures, agreed with
# Member 2 (retrieval tools) and Member 4 (signal and backtest tools). Names follow
# the tool table in docs/design.md section 3.3; argument and return types are open.


def search_announcements(query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    raise NotImplementedError


def query_events(filters: dict[str, Any]) -> list[dict[str, Any]]:
    raise NotImplementedError


def get_text_feature(event_ids: list[str], feature: str) -> dict[str, Any]:
    raise NotImplementedError


def define_signal(spec: SignalSpec) -> str:
    raise NotImplementedError


def run_backtest(signal_id: str) -> BacktestResult:
    raise NotImplementedError


def inspect_results(signal_id: str) -> dict[str, Any]:
    raise NotImplementedError


def report_finding(signal_id: str, rationale: str, evidence: list[str]) -> dict[str, Any]:
    raise NotImplementedError
