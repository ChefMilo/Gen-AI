"""Daily equal-weight event portfolio with explicit timing and cash accounting.

Positions are economic notionals marked with adjusted total returns. Targets are
rounded in raw-close share lots at each close. This is a daily-rebalanced total-
return simulation, not a broker ledger of dividends, settlement and tax lots.
"""

import math
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from backtest.selection import select_events
from dsl.signal import SignalSpec


class PortfolioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    initial_capital: float = Field(default=1_000_000, gt=0)
    fee_bps: float = Field(default=5, ge=0, le=1000)
    slippage_bps: float = Field(default=5, ge=0, le=1000)
    annual_borrow_bps: float = Field(default=0, ge=0)
    lot_size: int = Field(default=1, strict=True, ge=1)
    annual_risk_free: float = Field(default=0, gt=-1)
    periods_per_year: int = Field(default=252, strict=True, ge=1)


class PortfolioResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    summary: dict
    daily: list[dict]
    trades: list[dict]
    skipped_events: list[dict]


def return_statistics(returns, annual_risk_free=0.0, periods_per_year=252) -> dict:
    values = np.asarray(returns, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Portfolio returns must be finite, nonempty and greater than -1")
    excess = values - ((1 + annual_risk_free) ** (1 / periods_per_year) - 1)
    sd = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    daily_sharpe = float(excess.mean() / sd) if sd > 0 else None
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {
        "total_return": float(wealth[-1] - 1),
        "annualized_return": float(wealth[-1] ** (periods_per_year / len(values)) - 1),
        "sharpe_daily": daily_sharpe,
        "sharpe_annualized": daily_sharpe * math.sqrt(periods_per_year)
        if daily_sharpe is not None
        else None,
        "max_drawdown": float(drawdown.min()),
        "n_days": len(values),
    }


def deflated_sharpe(
    returns, trial_sharpes: list[float], n_trials: int, config: PortfolioConfig
) -> dict:
    """Bailey/Lopez de Prado equations (expected max SR + probabilistic SR).

    SRs are DAILY, never annualized inside the formula. The supplied search
    dispersion must cover all completed trials, not only selected findings.
    n_trials is nominal; correlated trials/serial dependence remain limitations.
    """
    values = np.array(returns, dtype=float, copy=True)
    values -= (1 + config.annual_risk_free) ** (1 / config.periods_per_year) - 1
    available = np.asarray(trial_sharpes, dtype=float)
    if n_trials < 1 or n_trials < len(available):
        raise ValueError("n_trials must cover the supplied trial sharpes")
    if not np.isfinite(values).all() or not np.isfinite(available).all():
        raise ValueError("DSR inputs must be finite")
    if len(values) < 4 or float(values.std(ddof=1)) <= 0:
        return {"probability": None, "reason": "insufficient_nonconstant_history"}
    if n_trials > 1 and len(available) < 2:
        return {"probability": None, "reason": "need_search_sharpe_dispersion"}
    sr = float(values.mean() / values.std(ddof=1))
    threshold = 0.0
    if n_trials > 1:
        dispersion = float(available.std(ddof=1))
        gamma = 0.5772156649015329
        threshold = dispersion * (
            (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
            + gamma * stats.norm.ppf(1 - 1 / (n_trials * math.e))
        )
    skew = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    variance = 1 - skew * sr + (kurtosis - 1) * sr * sr / 4
    if not math.isfinite(variance) or variance <= 0:
        return {"probability": None, "reason": "invalid_sampling_variance"}
    probability = float(
        stats.norm.cdf((sr - threshold) * math.sqrt(len(values) - 1) / math.sqrt(variance))
    )
    return {
        "probability": probability,
        "daily_sharpe": sr,
        "benchmark_daily_sharpe": threshold,
        "nominal_trials": n_trials,
        "available_trial_sharpes": len(available),
        "assumption": "nominal trial count; no serial-dependence adjustment",
    }


def simulate_portfolio(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
    events: pd.DataFrame,
    spec: SignalSpec,
    config: PortfolioConfig,
    start: date,
    end: date,
) -> PortfolioResult:
    """Enter at event-day close; earn the NEXT holding_days daily returns.

    Active tickers share one gross exposure, at most 100% of equity. Overlapping
    events for the same ticker merge into one position. Cash earns zero;
    short collateral earns no interest. Cost inputs are assumptions, not quotes.
    """
    selected = select_events(events, spec)
    selected = selected.loc[
        pd.to_datetime(selected["first_tradeable_date"]).dt.date.between(start, end)
    ]
    bench = benchmarks.loc[benchmarks["ticker"].eq(spec.benchmark)].copy()
    bench["date"] = pd.to_datetime(bench["date"]).dt.normalize()
    bench = bench.loc[bench["date"].dt.date.between(start, end)].sort_values("date")
    if bench.empty or bench["date"].duplicated().any():
        raise ValueError("A unique benchmark trading calendar is required")
    calendar = pd.DatetimeIndex(bench["date"])
    price = prices.copy()
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    if price.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date prices")
    if "raw_close" not in price.columns:
        raise ValueError("Portfolio lot sizing requires raw_close")
    lookup = price.set_index(["ticker", "date"])
    active = [set() for _ in calendar]
    skipped = []
    for event in selected.to_dict(orient="records"):
        day = pd.Timestamp(event["first_tradeable_date"]).normalize()
        position = int(calendar.searchsorted(day))
        reason = None
        if position == len(calendar) or calendar[position] != day:
            reason = "no_entry_day"
        elif position + spec.holding_days >= len(calendar):
            reason = "exit_after_research_cutoff"
        elif (event["ticker"], day) not in lookup.index:
            reason = "no_entry_price"
        else:
            row = lookup.loc[(event["ticker"], day)]
            if (
                bool(row["is_bad_price"])
                or not np.isfinite(row["raw_close"])
                or row["raw_close"] <= 0
            ):
                reason = "bad_entry_price"
        if reason:
            skipped.append({"event_id": event["event_id"], "reason": reason})
            continue
        for i in range(position, position + spec.holding_days):
            active[i].add(event["ticker"])

    cash = config.initial_capital
    previous_nav = cash
    positions = {}
    sign = 1 if spec.direction == "long" else -1
    cost_rate = (config.fee_bps + config.slippage_bps) / 10000
    borrow_rate = config.annual_borrow_bps / 10000 / config.periods_per_year
    daily, trades = [], []
    benchmark_nav = config.initial_capital
    for i, day in enumerate(calendar):
        pnl = 0.0
        borrow = sum(max(-value, 0) for value in positions.values()) * borrow_rate
        for ticker in sorted(positions):
            if (ticker, day) not in lookup.index:
                raise ValueError(f"Missing held-position price: {ticker} {day.date()}")
            row = lookup.loc[(ticker, day)]
            ret = float(row["ret"])
            if bool(row["is_bad_price"]) or not math.isfinite(ret) or ret <= -1:
                raise ValueError(f"Invalid held-position return: {ticker} {day.date()}")
            pnl += positions[ticker] * ret
            positions[ticker] *= 1 + ret
        cash -= borrow
        nav = cash + sum(positions.values())
        if nav <= 0:
            raise ValueError("Portfolio equity exhausted")

        targets = {}
        # Cost <= rate * (old gross + target gross). Solve the cash
        # constraint A + rate * (old gross + A) <= NAV before lot rounding.
        old_gross = sum(abs(value) for value in positions.values())
        affordable = (nav - cost_rate * old_gross) / (1 + cost_rate)
        if affordable < 0:
            raise ValueError("Equity cannot fund liquidation costs")
        target_budget = min(nav / (1 + 2 * cost_rate), affordable)
        allocation = target_budget / len(active[i]) if active[i] else 0.0
        for ticker in sorted(active[i]):
            if (ticker, day) not in lookup.index:
                raise ValueError(f"Missing rebalance price: {ticker} {day.date()}")
            row = lookup.loc[(ticker, day)]
            raw = float(row["raw_close"])
            if bool(row["is_bad_price"]) or not math.isfinite(raw) or raw <= 0:
                raise ValueError(f"Invalid rebalance price: {ticker} {day.date()}")
            lots = math.floor(allocation / (raw * config.lot_size))
            targets[ticker] = sign * lots * config.lot_size * raw

        turnover = fees = slippage = 0.0
        for ticker in sorted(set(positions) | set(targets)):
            old = positions.get(ticker, 0.0)
            target = targets.get(ticker, 0.0)
            delta = target - old
            if abs(delta) <= 1e-9:
                continue
            notional = abs(delta)
            fee = notional * config.fee_bps / 10000
            slip = notional * config.slippage_bps / 10000
            cash -= delta + fee + slip
            turnover += notional
            fees += fee
            slippage += slip
            trades.append(
                {
                    "date": day.date().isoformat(),
                    "ticker": ticker,
                    "notional_change": delta,
                    "target_notional": target,
                    "fee": fee,
                    "slippage": slip,
                }
            )
        positions = {ticker: value for ticker, value in targets.items() if value}
        final_nav = cash + sum(positions.values())
        if final_nav <= 0 or (sign == 1 and cash < -1e-6):
            raise ValueError("Portfolio cash/equity constraint violated")
        benchmark_return = 0.0 if i == 0 else float(bench.iloc[i]["ret"])
        if not math.isfinite(benchmark_return) or benchmark_return <= -1:
            raise ValueError("Invalid benchmark return")
        benchmark_nav *= 1 + benchmark_return
        daily.append(
            {
                "date": day.date().isoformat(),
                "equity": final_nav,
                "cash": cash,
                "gross_exposure": sum(abs(value) for value in positions.values()),
                "active_tickers": len(positions),
                "gross_pnl": pnl,
                "fees": fees,
                "slippage": slippage,
                "borrow_cost": borrow,
                "turnover": turnover,
                "net_return": final_nav / previous_nav - 1,
                "benchmark_equity": benchmark_nav,
            }
        )
        previous_nav = final_nav
    if positions:
        raise RuntimeError("Positions must be closed by the end of the research window")
    summary = return_statistics(
        [row["net_return"] for row in daily], config.annual_risk_free, config.periods_per_year
    )
    summary.update(
        {
            "ending_equity": previous_nav,
            "fees": sum(row["fees"] for row in daily),
            "slippage": sum(row["slippage"] for row in daily),
            "borrow_cost": sum(row["borrow_cost"] for row in daily),
            "turnover": sum(row["turnover"] for row in daily),
            "benchmark_total_return": benchmark_nav / config.initial_capital - 1,
            "entry": "event-day close; returns start next trading day",
            "accounting": "daily-rebalanced total-return notionals; raw-close lot rounding",
        }
    )
    return PortfolioResult(summary=summary, daily=daily, trades=trades, skipped_events=skipped)
