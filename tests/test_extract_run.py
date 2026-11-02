"""The extraction pipeline end to end, with a fake Anthropic client."""

import json
from datetime import date
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from ingest import llm as llm_module
from ingest import run
from store import events as store
from store import prices
from test_extract import BB, DIV, DIV_EVENT

TICKER, ACC = "ACME", "0000000001-24-000001"
GOOD = json.dumps({"events": [DIV_EVENT], "quotes": [DIV, BB]})


class FakeMessages:
    def __init__(self, replies):
        self.replies = replies
        self.sent = []
        self.batches = FakeBatches(self)

    def create(self, **params):
        self.sent.append(params)
        return _message(self.replies.pop(0))


def _message(text):
    usage = NS(model_dump=lambda: {"input_tokens": 1000, "output_tokens": 100})
    return NS(content=[NS(type="text", text=text)], usage=usage)


class FakeBatches:
    def __init__(self, messages):
        self.messages = messages
        self.submitted = []

    def create(self, requests):
        self.submitted = list(requests)
        return NS(id="batch-1")

    def retrieve(self, batch_id):
        counts = NS(model_dump=lambda: {"succeeded": len(self.submitted)})
        return NS(processing_status="ended", request_counts=counts)

    def results(self, batch_id):
        for r in self.submitted:
            result = NS(type="succeeded", message=_message(GOOD))
            yield NS(custom_id=r["custom_id"], result=result)


def fake_client(replies=()):
    return NS(messages=FakeMessages(list(replies)))


@pytest.fixture
def data(tmp_path, monkeypatch):
    folder = tmp_path / "text" / "edgar" / TICKER / ACC
    folder.mkdir(parents=True)
    (folder / "ex99.txt").write_text(f"Acme results. {DIV} {BB} Thanks.")
    (tmp_path / "index").mkdir()
    pd.DataFrame(
        [
            {
                "ticker": TICKER,
                "cik": "0000000001",
                "company_name": "Acme Corp",
                "accession_no": ACC,
                "form_type": "8-K",
                "items": "8.01",
                "source_url": "",
                "acceptance_datetime_et": "2024-05-02 09:00:00",
                "is_div_candidate": True,
                "is_buyback_candidate": True,
            }
        ]
    ).to_parquet(tmp_path / run.FILING_INDEX)
    (tmp_path / "prices").mkdir()
    pd.DataFrame({"date": ["2024-05-01", "2024-05-02", "2024-05-03"]}).to_csv(
        tmp_path / prices.TRADING_CALENDAR, index=False
    )
    pd.DataFrame(
        {
            "ticker": [TICKER],
            "date": pd.to_datetime(["2024-05-01"]),
            "mcap_usd": [1e9],
            "mcap_bucket": ["mid"],
        }
    ).to_parquet(tmp_path / run.PRICES)
    pd.DataFrame({"ticker": [TICKER], "sector": ["Industrials"]}).to_csv(
        tmp_path / run.UNIVERSE, index=False
    )
    pd.DataFrame(
        {"ticker": [TICKER], "recommendation": ["keep"], "usable_from": ["2016-01-01"]}
    ).to_csv(tmp_path / prices.UNIVERSE_QUALITY, index=False)
    monkeypatch.setenv(prices.DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(run, "OUT", tmp_path / "out")
    monkeypatch.setattr(run, "CACHE", tmp_path / "out" / "cache")
    monkeypatch.setattr(run, "BATCH", tmp_path / "out" / "batch.json")
    prices._read_calendar.cache_clear()
    return tmp_path


def test_llm_caches_replies(tmp_path):
    client = fake_client([GOOD])
    model = llm_module.LLM(tmp_path, model="claude-sonnet-5-5", client=client)
    messages = [{"role": "user", "content": "hi"}]
    assert model.ask("sys", messages) == GOOD
    assert model.ask("sys", messages) == GOOD  # no second API call: the reply list is empty
    assert model.usage.calls == 1 and model.usage.cached_calls == 1
    assert model.usage.cost("claude-sonnet-5-5") == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    assert client.messages.sent[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_run_extraction_builds_derived_events(data):
    model = llm_module.LLM(run.CACHE, model="m", client=fake_client([GOOD]))
    rows = run.candidates(data)
    events, failures = run.run_extraction(data, rows, model)
    assert failures == []
    events = run._derive(data, events)
    [e] = events.to_dict("records")
    assert e["event_id"] == f"{ACC}-dividend-1" and e["extraction_status"] == "passed"
    assert e["first_tradeable_date"] == date(2024, 5, 2)
    assert e["mcap_usd"] == 1e9 and e["sector"] == "Industrials"
    assert e["extractor_version"] == "extract-1.2+m"


def test_run_extraction_records_failed_filings(data):
    model = llm_module.LLM(run.CACHE, model="m", client=fake_client(["no", "still no"]))
    events, failures = run.run_extraction(data, run.candidates(data), model)
    assert events.empty and failures[0]["accession_no"] == ACC


def test_batch_submit_then_collect_saves_events(data, monkeypatch):
    client = fake_client()
    monkeypatch.setattr(llm_module.LLM, "_client", lambda self: client)
    run.cmd_submit(data, limit=None)
    assert json.loads(run.BATCH.read_text())["batch_id"] == "batch-1"
    assert client.messages.batches.submitted[0]["custom_id"] == ACC
    run.cmd_collect(data)
    assert client.messages.sent == []  # everything came from the batch
    saved = store.load_events()
    assert saved["event_id"].tolist() == [f"{ACC}-dividend-1"]
    report = json.loads((run.OUT / "report.json").read_text())
    assert report["events"] == 1 and report["failed_filings"] == 0
