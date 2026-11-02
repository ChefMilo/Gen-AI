"""Event store schema: one row of events.parquet.

Source of truth is docs/data_schema.md ("Event store"). Where docs/design.md
disagrees (SGX, rights issues, STI, announce_datetime), data_schema.md wins.

Backtests use only events with is_new_announcement=True and
extraction_status="passed", dated by first_tradeable_date.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal["dividend", "buyback"]
Benchmark = Literal["SPY", "IWM", "MDY"]
DEFAULT_BENCHMARK: Benchmark = "SPY"
McapBucket = Literal["small", "mid", "large"]
DividendKind = Literal["regular", "special", "initial"]
ExtractionStatus = Literal["passed", "flagged"]


class TextFeature(BaseModel):
    """A Retrieval text feature: a label plus the exact quote behind it."""

    label: str
    quote: str


class Event(BaseModel):
    event_id: str  # accession no. + type + counter, e.g. 0001437749-25-015532-buyback-1
    event_type: EventType
    ticker: str
    cik: str
    company_name: str

    # Links back to the filing index
    accession_no: str
    form_type: str
    items: str
    source_url: str

    # Timing
    acceptance_datetime: datetime  # US Eastern
    first_tradeable_date: date  # the event date for backtests
    announcement_date: date | None = None
    is_new_announcement: bool

    # Dividend fields
    dividend_per_share: float | None = None  # USD
    dividend_kind: DividendKind | None = None
    record_date: date | None = None
    payable_date: date | None = None
    ex_date: date | None = None  # rarely stated; not extracted
    prev_dividend_per_share: float | None = None  # derived
    dividend_change_pct: float | None = None  # derived

    # Buyback fields
    buyback_amount_usd: float | None = None
    buyback_shares: int | None = None
    is_new_program: bool | None = None
    buyback_expiry: date | None = None
    buyback_pct_mcap: float | None = None  # derived

    # Derived as of the day before first_tradeable_date
    mcap_usd: float | None = None
    mcap_bucket: McapBucket | None = None
    sector: str | None = None  # from universe.csv

    text_features: dict[str, TextFeature] = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict)  # field name -> exact quote

    extraction_status: ExtractionStatus
    flags: list[str] = Field(default_factory=list)
    extracted_at: datetime
    extractor_version: str
