import json
from datetime import date, datetime

import pytest

from ingest import extract

DIV = (
    "The Board of Directors declared a quarterly cash dividend of $0.25 per share, payable "
    "June 1, 2024 to holders of record on May 15, 2024."
)
BB = "The Board also approved a new $50 million share repurchase program."
TEXT = (
    "Acme Corp reported results. Revenue rose 5%. " + DIV + " " + BB + " Our outlook is good. "
    "Forward-looking statements about dividends involve risks. The weather was nice."
)
FILING = {
    "ticker": "ACME",
    "cik": "0000000001",
    "company_name": "Acme Corp",
    "accession_no": "0000000001-24-000001",
    "form_type": "8-K",
    "items": "8.01,9.01",
    "source_url": "https://www.sec.gov/x/",
    "acceptance_datetime_et": "2024-05-02 16:30:00",
}
CALENDAR = [date(2024, 5, 2), date(2024, 5, 3), date(2024, 5, 6)]


def reply(events, quotes=(DIV, BB)):
    return json.dumps({"events": events, "quotes": list(quotes)})


DIV_EVENT = {
    "type": "dividend",
    "new": True,
    "announced": None,
    "dps": 0.25,
    "kind": "regular",
    "record": "2024-05-15",
    "payable": "2024-06-01",
    "evidence": {"new": 0, "dps": 0, "kind": 0, "record": 0, "payable": 0},
}
BB_EVENT = {
    "type": "buyback",
    "new": True,
    "amount_usd": 50e6,
    "new_program": True,
    "evidence": {"new": 1, "amount_usd": 1, "new_program": 1},
}


def test_passages_keep_mentions_with_neighbours_and_skip_boilerplate():
    out = extract.passages(TEXT)
    assert DIV in out and BB in out
    assert "Revenue rose 5%." in out and "Our outlook is good." in out  # neighbours
    assert "Acme Corp reported results." not in out
    assert "Forward-looking" not in out  # boilerplate is not a mention...
    assert "weather" not in out


def test_passages_keep_decisions_first_when_over_the_cap():
    noise = " ".join(f"Dividends paid in year {y} were fine." for y in range(40))
    text = noise + " Quiet sentence. " + DIV
    out = extract.passages(text, window=0, max_chars=len(DIV) + 40)
    assert DIV in out


def test_passages_cut_tables_around_keywords():
    table = "x " * 2000 + "Cash dividends 0.25 0.25 " + "y " * 2000
    out = extract.passages(table)
    assert "Cash dividends 0.25" in out and len(out) < 1000


def test_check_reply_accepts_good_events():
    events = extract.check_reply(reply([DIV_EVENT, BB_EVENT]), TEXT, date(2024, 5, 2))
    div, bb = events
    assert div.problems == [] and bb.problems == []
    assert div.fields["dividend_per_share"] == 0.25 and div.fields["payable_date"] == date(
        2024, 6, 1
    )
    assert div.evidence["record_date"] == DIV
    assert bb.fields == {
        "buyback_amount_usd": 50e6,
        "buyback_shares": None,
        "is_new_program": True,
        "buyback_expiry": None,
        "is_new_announcement": True,
        "announcement_date": None,
    }


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"dps": 80.0}, "outside $0.001 to $50"),
        ({"record": "2024-07-01"}, "record date is after payable date"),
        ({"amount_usd": 5.0}, "amount_usd must be null"),
        ({"payable": "2024-04-01", "record": "2024-03-01"}, "paid before the filing date"),
        ({"evidence": {"new": 0, "dps": 0, "kind": 0}}, "no evidence quote for record"),
        ({"kind": None}, "needs dps and kind"),
    ],
)
def test_check_reply_finds_problems(change, problem):
    [event] = extract.check_reply(reply([DIV_EVENT | change]), TEXT, date(2024, 5, 2))
    assert any(problem in p for p in event.problems), event.problems


def test_check_reply_rejects_quotes_not_in_the_text():
    [event] = extract.check_reply(reply([BB_EVENT], quotes=("x", "We bought $9 billion.")), TEXT)
    assert any("not copied exactly" in p for p in event.problems)


def test_check_reply_needs_json():
    with pytest.raises(ValueError):
        extract.check_reply("Sorry, I can't help.", TEXT)


class Script:
    """A fake ask(): replies in order and records the messages it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


def test_extract_retries_once_with_the_problems():
    bad = DIV_EVENT | {"dps": 80.0}
    ask = Script(reply([bad]), reply([DIV_EVENT]))
    [event] = extract.extract(ask, "msg", TEXT)
    assert event.problems == []
    assert len(ask.calls) == 2
    assert "outside $0.001 to $50" in ask.calls[1][-1]["content"]


def test_extract_keeps_problem_events_after_the_retry():
    bad = DIV_EVENT | {"dps": 80.0}
    [event] = extract.extract(Script(reply([bad]), "not json"), "msg", TEXT)
    assert event.problems


def test_extract_raises_when_no_reply_is_readable():
    with pytest.raises(ValueError):
        extract.extract(Script("no", "still no"), "msg", TEXT)


def test_extract_no_retry_when_clean():
    ask = Script(reply([]))
    assert extract.extract(ask, "msg", TEXT) == []
    assert len(ask.calls) == 1


def test_to_events_numbers_events_and_dates_them():
    found = extract.check_reply(reply([DIV_EVENT, BB_EVENT, BB_EVENT | {"amount_usd": 1.0}]), TEXT)
    found[2].problems.append("made up")
    events = extract.to_events(found, FILING, CALENDAR, datetime(2024, 5, 3), "test")
    assert [e.event_id for e in events] == [
        "0000000001-24-000001-dividend-1",
        "0000000001-24-000001-buyback-1",
        "0000000001-24-000001-buyback-2",
    ]
    assert events[0].first_tradeable_date == date(2024, 5, 3)  # accepted after 16:00
    assert events[0].extraction_status == "passed"
    assert events[2].extraction_status == "flagged" and events[2].flags == ["check: made up"]


def test_judgement_fields_reuse_the_value_quote():
    event = DIV_EVENT | {"evidence": {"dps": 0, "record": 0, "payable": 0}}  # no new / kind
    [checked] = extract.check_reply(reply([event]), TEXT)
    assert checked.problems == []
    assert checked.evidence["is_new_announcement"] == DIV
    assert checked.evidence["dividend_kind"] == DIV


def test_reply_with_prose_after_the_json_is_read():
    text = (
        reply([DIV_EVENT])
        + "\n\nI left out the buyback because only past repurchases were reported."
    )
    [checked] = extract.check_reply(text, TEXT)
    assert checked.problems == []


def test_filing_text_reads_utf8(tmp_path):
    folder = tmp_path / extract.TEXT_DIR / "ACME" / "acc"
    folder.mkdir(parents=True)
    (folder / "ex99.txt").write_text("Acme\u2019s Board declared a dividend.", encoding="utf-8")
    assert extract.filing_text(tmp_path, "ACME", "acc") == "Acme\u2019s Board declared a dividend."
