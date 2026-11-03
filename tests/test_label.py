import pandas as pd
import pytest

from ingest import gold, label

DIV = (
    "The Board of Directors declared a quarterly cash dividend of $0.25 per share, payable "
    "June 1, 2024 to holders of record on May 15, 2024."
)
BB = "The Board also approved a new $50 million share repurchase program."
TEXT = "Acme Corp reported results. Revenue rose 5%. " + DIV + " " + BB + " The weather was nice."
ACC = "0000000001-24-000001"
FILING_COLUMNS = (
    "gold_id,ticker,company_name,accession_no,form_type,items,acceptance_datetime_et,year,"
    "bucket,sector,source_url,picked_for,note,double,status,n_events,labelled_by,checked_by,"
    "second_status,second_by"
)
EVENT_COLUMNS = (
    "gold_id,accession_no,event_type,is_new_announcement,announcement_date,dividend_per_share,"
    "dividend_kind,record_date,payable_date,buyback_amount_usd,buyback_shares,is_new_program,"
    "buyback_expiry,quote,notes"
)


@pytest.fixture
def dirs(tmp_path):
    gold_dir, data_dir = tmp_path / "gold", tmp_path / "data"
    gold_dir.mkdir()
    common = "ACME,Acme Corp,{},8-K,8.01,2024-05-02 16:30:00,2024,mid,Energy,u,p,n"
    rows = [
        "G01," + common.format(ACC) + ",yes,todo,,,,,",
        "G02," + common.format("acc2") + ",,todo,,,,,",
    ]
    (gold_dir / "filings.csv").write_text("\n".join([FILING_COLUMNS, *rows]) + "\n")
    for name in (gold.EVENTS, gold.SECOND_EVENTS):
        (gold_dir / name).write_text(EVENT_COLUMNS + "\n")
    folder = data_dir / "text" / "edgar" / "ACME" / ACC
    folder.mkdir(parents=True)
    (folder / "ex99.txt").write_text(TEXT, encoding="utf-8")
    return gold_dir, data_dir


def answers(*items):
    queue = list(items)
    return lambda prompt: queue.pop(0)


def sentence_no(text):
    return str(label.numbered_sentences(TEXT).index(text) + 1)


DIVIDEND = ["d", "y", "", "25 cents", "r", "May 15, 2024", "2024-06-01", sentence_no(DIV), ""]
BUYBACK = ["b", "y", "", "$50 million", "", "y", "", sentence_no(BB), "board"]


def test_label_writes_events_and_marks_the_filing(dirs):
    gold_dir, data_dir = dirs
    ask = answers("Justin", "2", *DIVIDEND, *BUYBACK, "")
    shown = []
    assert label.label("G01", data_dir, gold_dir, ask=ask, out=shown.append)
    events = gold.load_events(gold_dir)
    assert [e.event_type for e in events] == ["dividend", "buyback"]
    assert events[0].dividend_per_share == 0.25 and events[0].quote == DIV
    assert events[1].buyback_amount_usd == 50e6 and events[1].notes == "board"
    filings = gold.load_filings(gold_dir).set_index("gold_id")
    assert filings.loc["G01", ["status", "n_events", "labelled_by"]].tolist() == [
        "labelled",
        "2",
        "Justin",
    ]
    assert gold.check(gold.load_filings(gold_dir), events, data_dir) == []
    assert any("[3]" in line for line in shown)


def test_second_labels_go_to_their_own_file(dirs):
    gold_dir, data_dir = dirs
    ask = answers("0", "y")
    assert label.label(
        "G01", data_dir, gold_dir, ask=ask, out=lambda s: None, by="Terry", second=True
    )
    filings = gold.load_filings(gold_dir).set_index("gold_id")
    assert filings.loc["G01", ["status", "second_status", "second_by"]].tolist() == [
        "todo",
        "labelled",
        "Terry",
    ]
    assert pd.read_csv(gold_dir / gold.EVENTS).empty


def test_redo_replaces_earlier_rows(dirs):
    gold_dir, data_dir = dirs
    label.label("G01", data_dir, gold_dir, ask=answers("1", *DIVIDEND, "y"), out=print, by="J")
    assert not label.label("G01", data_dir, gold_dir, ask=answers(), out=print, by="J")
    label.label(
        "G01", data_dir, gold_dir, ask=answers("1", *BUYBACK, "y"), out=print, by="J", redo=True
    )
    assert [e.event_type for e in gold.load_events(gold_dir)] == ["buyback"]


def test_bad_answers_are_asked_again(dirs):
    gold_dir, data_dir = dirs
    dividend = ["x", *DIVIDEND[:3], "lots", *DIVIDEND[3:5], "15/05/2024", *DIVIDEND[6:]]
    shown = []
    label.label(
        "G01", data_dir, gold_dir, ask=answers("1", *dividend, ""), out=shown.append, by="J"
    )
    assert gold.load_events(gold_dir)[0].record_date.isoformat() == "2024-05-15"
    assert sum(line.startswith("  !") for line in shown) == 2


def test_a_quote_not_in_the_filing_is_caught(dirs):
    gold_dir, data_dir = dirs
    typed = [*DIVIDEND[:7], "The Board declared $9.", "", "n", *DIVIDEND, "y"]
    shown = []
    label.label("G01", data_dir, gold_dir, ask=answers("1", *typed), out=shown.append, by="J")
    assert any("not in the filing" in line for line in shown)
    assert gold.load_events(gold_dir)[0].quote == DIV


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("$0.25", 0.25),
        ("24.25 cents", 0.2425),
        ("300m", 300e6),
        ("$1.5 billion", 1.5e9),
        ("2,500,000", 2.5e6),
        ("", None),
    ],
)
def test_parse_money(text, value):
    got = label.parse_money(text)
    assert got is None if value is None else got == pytest.approx(value)


def test_next_id_skips_done_and_respects_range(dirs):
    gold_dir, _ = dirs
    filings = gold.load_filings(gold_dir)
    assert label.next_id(filings, False, None) == "G01"
    assert label.next_id(filings, False, ("G02", "G30")) == "G02"
    filings.loc[0, "second_status"] = "labelled"
    assert label.next_id(filings, True, None) is None
