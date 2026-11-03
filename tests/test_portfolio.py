import numpy as np
import pandas as pd
import pytest

from backtest.portfolio import (
    PortfolioConfig,
    deflated_sharpe,
    return_statistics,
    simulate_portfolio,
)
from dsl.signal import SignalSpec
from harness.backtester import HarnessConfig, ResearchBacktester


@pytest.fixture
def data():
    days = pd.bdate_range("2020-01-06", periods=5)
    prices = pd.DataFrame(
        {
            "ticker": ["A"] * 5,
            "date": days,
            "raw_close": [100.0] * 5,
            "ret": [0.5, 0.1, -0.1, 0.05, 0.0],
            "is_bad_price": [False] * 5,
        }
    )
    bench = pd.DataFrame({"ticker": ["SPY"] * 5, "date": days, "ret": [0.0] * 5})
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["A"],
            "event_type": ["dividend"],
            "first_tradeable_date": [days[0]],
            "acceptance_datetime": [days[0] + pd.Timedelta(hours=9)],
            "is_new_announcement": [True],
            "extraction_status": ["passed"],
        }
    )
    return prices, bench, events


def simulate(data, holding_days=2, direction="long", **config):
    return simulate_portfolio(
        *data,
        SignalSpec(event_type="dividend", holding_days=holding_days, direction=direction),
        PortfolioConfig(initial_capital=1000, fee_bps=0, slippage_bps=0, **config),
        pd.Timestamp("2020-01-06").date(),
        pd.Timestamp("2020-01-10").date(),
    )


def test_entry_excludes_event_day_and_exits_on_schedule(data):
    result = simulate(data)
    assert result.daily[0]["net_return"] == 0  # ignores the event-day +50%
    assert result.summary["ending_equity"] == pytest.approx(990)
    assert result.summary["total_return"] == pytest.approx(-0.01)
    assert result.daily[2]["active_tickers"] == 0
    assert result.daily[-1]["equity"] == pytest.approx(990)


def test_costs_are_charged_on_entry_and_exit(data):
    result = simulate_portfolio(
        *data,
        SignalSpec(event_type="dividend", holding_days=1),
        PortfolioConfig(initial_capital=10000, fee_bps=100, slippage_bps=0),
        pd.Timestamp("2020-01-06").date(),
        pd.Timestamp("2020-01-10").date(),
    )
    assert result.trades[0]["target_notional"] == 9800
    assert result.summary["fees"] == pytest.approx(98 + 107.8)
    assert result.summary["ending_equity"] == pytest.approx(10000 + 980 - 205.8)
    for day in result.daily:
        assert day["cash"] >= -1e-6
        assert day["gross_exposure"] <= day["equity"] + 1e-6


def test_short_loses_when_stock_rises_and_pays_borrow(data):
    result = simulate(data, holding_days=1, direction="short", annual_borrow_bps=252)
    assert result.summary["borrow_cost"] == pytest.approx(0.1)
    assert result.summary["ending_equity"] == pytest.approx(899.9)


def test_share_lot_rounding_can_leave_all_capital_in_cash(data):
    result = simulate(data, lot_size=100)
    assert result.trades == []
    assert result.summary["ending_equity"] == 1000
    assert result.summary["sharpe_annualized"] is None


def test_overlapping_events_do_not_double_exposure(data):
    prices, bench, events = data
    repeated = pd.concat([events, events.assign(event_id="e2")], ignore_index=True)
    assert simulate((prices, bench, repeated)).summary == simulate(data).summary


def test_future_missing_price_is_not_used_to_preselect_successful_trades(data):
    prices, bench, events = data
    with pytest.raises(ValueError, match="Missing held-position price"):
        simulate((prices.iloc[[0, 2, 3, 4]], bench, events))


def test_exit_cannot_cross_research_cutoff(data):
    prices, bench, events = data
    events = events.assign(first_tradeable_date=bench["date"].iloc[-1])
    result = simulate((prices, bench, events))
    assert result.trades == []
    assert result.skipped_events[0]["reason"] == "exit_after_research_cutoff"


def test_hand_calculated_drawdown_and_sharpe():
    returns = [0.1, -0.2, 0.1]
    result = return_statistics(returns)
    assert result["total_return"] == pytest.approx(1.1 * 0.8 * 1.1 - 1)
    assert result["max_drawdown"] == pytest.approx(-0.2)
    assert result["sharpe_annualized"] == pytest.approx(0, abs=1e-12)


def test_dsr_single_trial_and_more_search_penalty():
    values = np.array([-0.01, 0.01] * 20)
    original = values.copy()
    single = deflated_sharpe(values, [0.0], 1, PortfolioConfig())
    assert single["probability"] == pytest.approx(0.5)
    ten = deflated_sharpe(values, [-0.1, 0.1], 10, PortfolioConfig())
    hundred = deflated_sharpe(values, [-0.1, 0.1], 100, PortfolioConfig())
    assert hundred["probability"] < ten["probability"] < single["probability"]
    np.testing.assert_array_equal(values, original)


def test_harness_records_portfolio_and_final_dsr(data, tmp_path):
    harness = ResearchBacktester(
        *data,
        HarnessConfig(
            research_start="2020-01-06",
            research_end="2020-01-10",
            budget=2,
            min_events=2,
            portfolio=PortfolioConfig(),
        ),
        tmp_path / "audit.jsonl",
    )
    signal = harness.define({"event_type": "dividend", "holding_days": 2})
    result = harness.run(signal)
    assert result["portfolio"]["summary"]["entry"].startswith("event-day close")
    with pytest.raises(RuntimeError, match="finalization"):
        harness.portfolio_reports()
    harness.finalize([], "portfolio-session", "R2")
    assert "deflated_sharpe" in harness.portfolio_reports()[signal]


def test_full_rotation_reserves_costs_without_borrowing_cash(data):
    prices, bench, events = data
    prices = prices.assign(raw_close=0.01, ret=0.0)
    prices = pd.concat([prices, prices.assign(ticker="B")], ignore_index=True)
    later = events.assign(event_id="e2", ticker="B", first_tradeable_date=bench["date"].iloc[1])
    events = pd.concat([events, later], ignore_index=True)
    result = simulate_portfolio(
        prices,
        bench,
        events,
        SignalSpec(event_type="dividend", holding_days=1),
        PortfolioConfig(initial_capital=1_000_000, fee_bps=5, slippage_bps=5),
        bench["date"].iloc[0].date(),
        bench["date"].iloc[-1].date(),
    )
    assert min(row["cash"] for row in result.daily) >= -1e-6
    assert result.summary["fees"] > 0
    assert result.daily[1]["active_tickers"] == 1
    assert result.daily[2]["active_tickers"] == 0
