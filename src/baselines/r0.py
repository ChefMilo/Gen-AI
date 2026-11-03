# R0: a deterministic baseline for event-driven research
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from scipy import stats

from agent.finding import Finding
from dsl.signal import SignalSpec
from evaluation.reference_car import event_cars

if TYPE_CHECKING:
    from evaluation.systems import SystemRun, UniverseView


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust p-values, preserving their original order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda i: p_values[i])
    adjusted = [1.0] * count
    previous = 0.0

    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        previous = max(previous, value)
        adjusted[index] = previous

    return adjusted


class R0System:
    name = "r0"

    def __init__(self, holding_days: int = 5, min_events: int = 20):
        self.holding_days = holding_days
        self.min_events = min_events

    def select_dividend_increases(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Select dividend increases strictly greater than 10 percent"""
        matches = events["event_type"].eq("dividend") & events["dividend_change_pct"].gt(10)

        return events.loc[matches].copy()

    def calculate_dividend_returns(
        self,
        view: UniverseView,
    ) -> pd.DataFrame:
        """Calculate CAR for dividend increases above 10 percent."""
        selected = self.select_dividend_increases(view.events)

        return event_cars(
            prices=view.prices,
            benchmarks=view.benchmarks,
            events=selected,
            holding_days=self.holding_days,
        )

    def summarize_returns(self, cars: pd.DataFrame) -> dict:
        """Summarize measurable returns and test their mean against zero."""
        returns = cars["car"].astype(float)

        # Exclude missing or infinite results.
        valid = returns.notna() & returns.ne(float("inf")) & returns.ne(float("-inf"))
        returns = returns.loc[valid]

        result = {
            "n_events": len(returns),
            "mean_car": float(returns.mean()) if len(returns) else None,
            "t_stat": None,
            "p_value_raw": None,
            "status": "insufficient_events",
        }

        if len(returns) < max(self.min_events, 2):
            return result

        # Identical returns have no sample variance, so we cannot
        # report a conventional finite t-statistic.
        if returns.nunique() == 1:
            result["status"] = "zero_variance"
            return result

        test = stats.ttest_1samp(returns, popmean=0.0)

        result.update(
            t_stat=float(test.statistic),
            p_value_raw=float(test.pvalue),
            status="tested",
        )
        return result

    def calculate_other_returns(
        self,
        view: UniverseView,
        event_type: str,
        field: str,
        value: str | float,
    ) -> pd.DataFrame:
        events = view.events
        values = events.get(
            field,
            pd.Series(index=events.index, dtype=object),
        )

        if field == "buyback_pct_mcap":
            matches = pd.to_numeric(values, errors="coerce").gt(value)
        else:
            matches = values.eq(value)

        selected = events.loc[events["event_type"].eq(event_type) & matches.fillna(False)].copy()

        return event_cars(
            view.prices,
            view.benchmarks,
            selected,
            self.holding_days,
        )

    def run(
        self,
        view: UniverseView,
        budget: int,
        seed: int,
    ) -> SystemRun:
        from evaluation.systems import SystemRun

        if budget < 0:
            raise ValueError("budget must be non-negative")

        sid = f"{self.name}-{view.universe_id}-{seed}"
        metadata = {
            "backtests_used": 0,
            "tool_calls": 0,
            "tokens": 0,
            "cost_usd": 0.0,
            "correction": "holm",
            "alpha": 0.05,
            "hypotheses": [],
        }

        # Fixed order; no selection based on observed performance.
        grid = [
            (
                "dividend_increase_gt10",
                "dividend",
                {"dividend_change_pct_gt": 10},
                lambda: self.calculate_dividend_returns(view),
            ),
            (
                "special_dividend",
                "dividend",
                {"dividend_kind": "special"},
                lambda: self.calculate_other_returns(view, "dividend", "dividend_kind", "special"),
            ),
            (
                "buyback_gt5pct",
                "buyback",
                {"buyback_pct_mcap_gt": 5},
                lambda: self.calculate_other_returns(view, "buyback", "buyback_pct_mcap", 5.0),
            ),
        ]

        attempts = []

        for hypothesis_id, event_type, filters, calculate in grid[:budget]:
            cars = calculate()
            valid = cars["car"].notna() & ~cars["car"].isin([float("inf"), float("-inf")])
            cars = cars.loc[valid].copy()
            summary = self.summarize_returns(cars)

            record = {
                "hypothesis_id": hypothesis_id,
                "event_type": event_type,
                "filters": filters,
                "holding_days": self.holding_days,
                "benchmark": "SPY",
                **summary,
            }
            metadata["hypotheses"].append(record)
            attempts.append((record, cars))

        metadata["backtests_used"] = len(attempts)

        # Unusable attempts remain in the correction family with p=1.
        p_values = [
            record["p_value_raw"] if record["status"] == "tested" else 1.0 for record, _ in attempts
        ]
        adjusted = holm_adjust(p_values)
        findings = []

        for index, ((record, cars), corrected) in enumerate(zip(attempts, adjusted, strict=True)):
            record["p_value_corrected"] = corrected
            record["passed_correction"] = record["status"] == "tested" and corrected <= 0.05

            if record["status"] != "tested":
                continue

            mean_car = record["mean_car"]
            findings.append(
                Finding(
                    finding_id=f"{sid}-f{index}",
                    session_id=sid,
                    system="R0",
                    signal_id=f"{sid}-s{index}",
                    signal_spec=SignalSpec(event_type=record["event_type"]),
                    event_type=record["event_type"],
                    event_ids=cars["event_id"].tolist(),
                    n_events=record["n_events"],
                    direction="long" if mean_car >= 0 else "short",
                    effect_car=mean_car,
                    t_stat=record["t_stat"],
                    p_value_raw=record["p_value_raw"],
                    p_value_corrected=corrected,
                    passed_correction=record["passed_correction"],
                    rationale=(
                        f"Fixed hypothesis {record['hypothesis_id']}: "
                        f"{record['filters']}; "
                        f"{self.holding_days}-day CAR against SPY."
                    ),
                    evidence=[],
                )
            )

        return SystemRun(
            findings=findings,
            session_id=sid,
            metadata=metadata,
        )
