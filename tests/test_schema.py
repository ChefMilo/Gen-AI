from datetime import date, datetime

import pytest
from pydantic import ValidationError

from agent import tools
from backtest.result import BacktestResult
from dsl.signal import SignalSpec
from store import prices
from store.schema import Event


def make_event(**overrides):
    fields = dict(
        event_id="0001437749-25-015532-buyback-1",
        event_type="buyback",
        ticker="HSTM",
        cik="0001090012",
        company_name="HealthStream",
        accession_no="0001437749-25-015532",
        form_type="8-K",
        items="8.01",
        source_url="https://www.sec.gov/Archives/edgar/data/1090012/000143774925015532/",
        acceptance_datetime=datetime(2025, 5, 5, 16, 30),
        first_tradeable_date=date(2025, 5, 6),
        is_new_announcement=True,
        buyback_amount_usd=25_000_000,
        is_new_program=True,
        extraction_status="passed",
        extracted_at=datetime(2025, 10, 21, 12, 0),
        extractor_version="0.1",
    )
    fields.update(overrides)
    return Event(**fields)


def test_event_accepts_data_schema_example():
    event = make_event()
    assert event.first_tradeable_date == date(2025, 5, 6)
    assert event.dividend_per_share is None


@pytest.mark.parametrize("event_type", ["rights_issue", "bonus_issue", "consolidation"])
def test_event_rejects_non_edgar_event_types(event_type):
    with pytest.raises(ValidationError):
        make_event(event_type=event_type)


def test_event_text_feature_needs_label_and_quote():
    event = make_event(text_features={"buyback_rationale": {"label": "undervalued", "quote": "..."}})
    assert event.text_features["buyback_rationale"].label == "undervalued"
    with pytest.raises(ValidationError):
        make_event(text_features={"buyback_rationale": {"label": "undervalued"}})


def test_signal_spec_defaults_to_spy_and_rejects_sti():
    assert SignalSpec(event_type="dividend").benchmark == "SPY"
    assert SignalSpec(event_type="dividend", benchmark="IWM").benchmark == "IWM"
    with pytest.raises(ValidationError):
        SignalSpec(event_type="dividend", benchmark="STI")


def test_backtest_result_placeholder():
    assert BacktestResult(signal_id="s1").signal_id == "s1"


def test_tool_placeholders_raise():
    with pytest.raises(NotImplementedError):
        tools.run_backtest("s1")


def test_data_dir_requires_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(prices.DATA_DIR_ENV, raising=False)
    with pytest.raises(RuntimeError):
        prices.data_dir()
    monkeypatch.setenv(prices.DATA_DIR_ENV, str(tmp_path))
    assert prices.data_dir() == tmp_path
