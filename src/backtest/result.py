"""Structured outputs from the event-study engine."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backtest.portfolio import PortfolioResult
from dsl.signal import SignalSpec


class EventReturn(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    event_id: str
    ticker: str
    first_tradeable_date: date
    car: float
    n_days: int


class SkippedEvent(BaseModel):
    event_id: str
    ticker: str
    first_tradeable_date: date
    reason: str


class BacktestResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    signal_id: str
    signal_spec: SignalSpec | None = None
    status: Literal[
        "not_run",
        "tested",
        "insufficient_events",
        "zero_variance",
        "invalid_statistics",
    ] = "not_run"

    n_selected: int = 0
    n_events: int = 0
    event_ids: list[str] = Field(default_factory=list)
    event_returns: list[EventReturn] = Field(default_factory=list)
    skipped_events: list[SkippedEvent] = Field(default_factory=list)

    mean_car: float | None = None
    directional_mean_car: float | None = None
    t_stat: float | None = None
    p_value_raw: float | None = Field(default=None, ge=0, le=1)
    hit_rate: float | None = Field(default=None, ge=0, le=1)

    portfolio: PortfolioResult | None = None
