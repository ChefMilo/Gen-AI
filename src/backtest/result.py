"""Backtest result schema (interface 3 in docs/design.md section 7)."""

from pydantic import BaseModel


# TODO(member-4): placeholder only. Member 4 owns the full BacktestResult design
# (CAR, t-stat, hit rate, Sharpe, n events, breakdowns).
class BacktestResult(BaseModel):
    signal_id: str
