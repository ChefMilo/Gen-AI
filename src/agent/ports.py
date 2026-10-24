"""What the agent needs from other layers, as Protocols.

The agent codes against these; src/agent/fakes.py implements them for now, and the
real layers will plug in behind them.

TODO(member-3): tighten these types once the contracts land:
- Backtester: SignalSpec (src/dsl/signal.py) and BacktestResult (src/backtest/result.py)
  from Member 4 replace the dicts.
- Retriever: search result and text-feature formats from Member 2.
- EventQuery: the event loader from Member 1 (events.parquet, store.schema.Event).
"""

from typing import Any, Protocol


class Backtester(Protocol):
    def define(self, spec: dict[str, Any]) -> str:
        """Validate and register a signal; return its signal_id."""
        ...

    def run(self, signal_id: str) -> dict[str, Any]:
        """Run the event study; counts against the testing budget."""
        ...

    def breakdown(self, signal_id: str, by: str) -> dict[str, Any]:
        """Results of an already-run signal split by year, sector, mcap_bucket, ..."""
        ...

    def signal_options(self) -> dict[str, Any]:
        """What a signal may contain: event_types, operators (by field kind), holding_days,
        directions, breakdowns. The task brief is built from this, never hard-coded."""
        ...

    def budget_status(self) -> dict[str, int]:
        """{"budget": total backtests allowed, "tests_used": backtests run so far}"""
        ...


class Retriever(Protocol):
    def search(self, query: str, filters: dict[str, Any], k: int) -> list[dict[str, Any]]:
        """Passages matching the query, each with a doc reference."""
        ...

    def text_feature(self, event_ids: list[str], feature: str) -> list[dict[str, Any]]:
        """One {event_id, label, quote} per event."""
        ...

    def available_text_features(self) -> dict[str, list[str]]:
        """feature name -> allowed labels, so prompts never hard-code them."""
        ...


class EventQuery(Protocol):
    def query(
        self, event_type: str, conditions: list[dict[str, Any]], limit: int
    ) -> dict[str, Any]:
        """Count and sample events matching the conditions."""
        ...

    def available_fields(self) -> dict[str, str]:
        """field name -> short description, so prompts never hard-code them."""
        ...
