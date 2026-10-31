from datetime import date, datetime

import pandas as pd
import pytest

from ingest import gold
from store.schema import Event

DIV_QUOTE = "declared a quarterly cash dividend of $0.10 per share"
BB_QUOTE = "authorized the repurchase of up to $25 million"


def _filings(**status):
    rows = [
        ("G01", "AAA", "0000000001-24-000001", "labelled", "2"),
        ("G02", "BBB", "0000000002-24-000002", "labelled", "0"),
        ("G03", "CCC", "0000000003-24-000003", "todo", ""),
    ]
    df = pd.DataFrame(rows, columns=["gold_id", "ticker", "accession_no", "status", "n_events"])
    for gold_id, value in status.items():
        df.loc[df["gold_id"] == gold_id, "status"] = value
    return df


def _events():
    return [
        gold.GoldEvent(
            gold_id="G01",
            accession_no="0000000001-24-000001",
            event_type="dividend",
            is_new_announcement=True,
            dividend_per_share=0.10,
            dividend_kind="regular",
            record_date=date(2024, 7, 1),
            payable_date=date(2024, 7, 15),
            quote=DIV_QUOTE,
        ),
        gold.GoldEvent(
            gold_id="G01",
            accession_no="0000000001-24-000001",
            event_type="buyback",
            is_new_announcement=True,
            buyback_amount_usd=25e6,
            is_new_program=True,
            quote=BB_QUOTE,
        ),
    ]


def _as_event(e: gold.GoldEvent, n: int = 1, **changes) -> Event:
    fields = {f: getattr(e, f) for f in gold.FIELDS} | changes
    return Event(
        event_id=f"{e.accession_no}-{e.event_type}-{n}",
        event_type=e.event_type,
        ticker="AAA",
        cik="0000000001",
        company_name="A Co",
        accession_no=e.accession_no,
        form_type="8-K",
        items="8.01",
        source_url="",
        acceptance_datetime=datetime(2024, 6, 20, 16, 30),
        first_tradeable_date=date(2024, 6, 21),
        extraction_status="passed",
        extracted_at=datetime(2024, 6, 22),
        extractor_version="test",
        **fields,
    )


def test_repo_gold_files_are_valid():
    filings = gold.load_filings()
    assert len(filings) == 30
    assert filings["accession_no"].is_unique
    assert gold.check(filings, gold.load_events()) == []


def test_load_events_reads_blanks_as_none(tmp_path):
    header = (gold.GOLD_DIR / gold.EVENTS).read_text().splitlines()[0]
    row = "G01,0000000001-24-000001,buyback,True,,,,,,,2500000,False,2025-04-30,a quote,"
    (tmp_path / gold.EVENTS).write_text(f"{header}\n{row}\n")
    [e] = gold.load_events(tmp_path)
    assert e.buyback_shares == 2_500_000 and e.buyback_amount_usd is None
    assert e.is_new_program is False and e.buyback_expiry == date(2025, 4, 30)
    assert e.notes == ""


def test_load_events_names_the_bad_line(tmp_path):
    header = (gold.GOLD_DIR / gold.EVENTS).read_text().splitlines()[0]
    row = "G01,0000000001-24-000001,rights_issue,True,,,,,,,,,,a quote,"
    (tmp_path / gold.EVENTS).write_text(f"{header}\n{row}\n")
    with pytest.raises(ValueError, match="line 2"):
        gold.load_events(tmp_path)


def test_check_passes_consistent_labels():
    assert gold.check(_filings(), _events()) == []


def test_check_finds_problems():
    bad = _events()
    bad[0] = bad[0].model_copy(
        update={"record_date": date(2024, 8, 1), "buyback_shares": 5, "quote": " "}
    )
    bad.append(bad[1].model_copy(update={"gold_id": "G99"}))
    problems = gold.check(_filings(G03="done"), bad)
    text = "\n".join(problems)
    assert "record_date is after payable_date" in text
    assert "buyback_shares belongs to the other event type" in text
    assert "quote is empty" in text
    assert "G99 buyback: gold_id is not in filings.csv" in text
    assert "G03: status 'done'" in text


def test_check_counts_events_per_labelled_filing():
    problems = gold.check(_filings(), _events()[:1])
    assert problems == ["G01: n_events is 2 but events.csv has 1"]


def test_check_quotes_against_filing_text(tmp_path):
    folder = tmp_path / gold.TEXT_DIR / "AAA" / "0000000001-24-000001"
    folder.mkdir(parents=True)
    wrapped = DIV_QUOTE.replace(" of ", " of\n")  # line breaks differ from the quote
    (folder / "ex99.txt").write_text(f"The Board\n  {wrapped}.")
    problems = gold.check(_filings(), _events(), data_dir=tmp_path)
    assert problems == ["G01 buyback: quote is not copied exactly from the filing"]


def test_score_perfect_extraction():
    events = _events()
    result = gold.score(_filings(), events, [_as_event(e) for e in events])
    assert result["precision"] == result["recall"] == 1.0
    assert result["filing_accuracy"] == 1.0
    assert set(result["field_accuracy"].values()) == {1.0}
    assert result["n_filings"] == 2  # G03 is still todo


def test_score_wrong_field_missed_event_and_false_positive():
    div, bb = _events()
    predicted = [
        _as_event(div, dividend_per_share=0.01),  # wrong amount
        _as_event(div, n=2, is_new_announcement=False, record_date=None),  # one too many
        _as_event(bb.model_copy(update={"accession_no": "0000000002-24-000002"})),  # no event
    ]
    result = gold.score(_filings(), [div, bb], predicted)
    assert result["n_matched"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1 / 2)
    assert result["field_accuracy"]["dividend_per_share"] == 0.0
    assert result["field_accuracy"]["is_new_announcement"] == 1.0  # the better pair was kept
    assert result["field_accuracy"]["buyback_amount_usd"] is None  # no buyback pairs
    assert result["filing_accuracy"] == 0.0


def test_score_ignores_todo_filings():
    div, _ = _events()
    stray = _as_event(div.model_copy(update={"accession_no": "0000000003-24-000003"}))
    result = gold.score(_filings(), [], [stray])
    assert result["n_predicted"] == 0 and result["precision"] == 1.0
