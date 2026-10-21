"""One valid example of every interface model."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from agent.finding import EvidenceQuote, Finding
from backtest.result import BacktestResult
from dsl.signal import SignalSpec
from store.schema import Event


def example_event() -> Event:
    return Event(
        event_id="0001437749-25-015532-dividend-1",
        event_type="dividend",
        ticker="WSO",
        cik="0000105016",
        company_name="Watsco",
        accession_no="0001437749-25-015532",
        form_type="8-K",
        items="8.01,9.01",
        source_url="https://www.sec.gov/Archives/edgar/data/105016/000143774925015532/",
        acceptance_datetime=datetime(2025, 4, 1, 9, 15),
        first_tradeable_date=date(2025, 4, 1),
        is_new_announcement=True,
        dividend_per_share=0.10,
        dividend_kind="regular",
        record_date=date(2025, 4, 15),
        payable_date=date(2025, 4, 30),
        prev_dividend_per_share=0.09,
        dividend_change_pct=11.1,
        mcap_usd=1.2e9,
        mcap_bucket="mid",
        evidence={"dividend_per_share": "a quarterly dividend of 10 cents per share"},
        extraction_status="passed",
        extracted_at=datetime(2025, 10, 21, 12, 0),
        extractor_version="0.1",
    )


def example_signal_spec() -> SignalSpec:
    return SignalSpec(event_type="dividend")


def example_backtest_result() -> BacktestResult:
    return BacktestResult(signal_id="sig-001")


def example_finding() -> Finding:
    return Finding(
        finding_id="f-001",
        session_id="sess-001",
        system="R3",
        signal_id="sig-001",
        signal_spec=example_signal_spec(),
        event_type="dividend",
        event_ids=["0001437749-25-015532-dividend-1"],
        n_events=1,
        direction="long",
        effect_car=0.012,
        t_stat=2.4,
        p_value_raw=0.016,
        p_value_corrected=0.048,
        passed_correction=True,
        rationale="Dividend increases above 10% earn positive abnormal returns.",
        evidence=[
            EvidenceQuote(event_id="0001437749-25-015532-dividend-1", quote="10 cents per share"),
            EvidenceQuote(doc_ref="WSO/0001437749-25-015532/ex99-1.txt", quote="raised"),
        ],
    )


@pytest.mark.parametrize(
    "build",
    [example_event, example_signal_spec, example_backtest_result, example_finding],
)
def test_example_model_round_trips_through_json(build):
    model = build()
    assert type(model).model_validate_json(model.model_dump_json()) == model


def test_finding_requires_event_ids():
    data = example_finding().model_dump()
    del data["event_ids"]
    with pytest.raises(ValidationError):
        Finding.model_validate(data)


def test_evidence_needs_event_id_or_doc_ref():
    with pytest.raises(ValidationError):
        EvidenceQuote(quote="orphan quote")
