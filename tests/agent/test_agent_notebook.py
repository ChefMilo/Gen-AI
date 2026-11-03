import json

import pytest
from pydantic import ValidationError

from agent.notebook import SUMMARY_MAX_CHARS, Notebook
from agent.trace import Trace


def filled() -> Notebook:
    nb = Notebook()
    nb.add("hypothesis", 1, "Dividend hikes in small caps earn positive CAR",
           {"rationale": "signalling is stronger where information is scarce"})
    nb.add("signal_defined", 2, "sig-1: dividend, change > 10, small", {"signal_id": "sig-1"})
    nb.add("backtest_result", 3, "sig-1: n=84, CAR +2.1%, t=3.9", {"n_events": 84, "t_stat": 3.9})
    return nb


def test_render_is_one_line_per_entry_plus_data():
    text = filled().render()
    assert text.splitlines() == [
        "[step 1] hypothesis: Dividend hikes in small caps earn positive CAR",
        '    {"rationale":"signalling is stronger where information is scarce"}',
        "[step 2] signal_defined: sig-1: dividend, change > 10, small",
        '    {"signal_id":"sig-1"}',
        "[step 3] backtest_result: sig-1: n=84, CAR +2.1%, t=3.9",
        '    {"n_events":84,"t_stat":3.9}',
    ]


def test_render_truncates_long_data_and_summaries():
    nb = Notebook()
    nb.add("note", 1, "x" * 1000, {"event_ids": [f"e{i}" for i in range(500)]})
    entry_line, data_line = nb.render(max_data_chars=100).splitlines()
    assert data_line.endswith("...[truncated]") and len(data_line) < 130
    assert entry_line.endswith("...[truncated]")
    assert len(nb.entries[0].summary) == SUMMARY_MAX_CHARS + len("...[truncated]")


def test_render_last_n_entries():
    text = filled().render(max_entries=1)
    assert text.splitlines()[0] == "(2 earlier entries not shown)"
    assert "[step 3]" in text and "[step 1]" not in text


def test_empty_notebook():
    assert Notebook().render() == "(notebook is empty)"


def test_to_json_round_trips():
    data = json.loads(filled().to_json())
    assert [e["kind"] for e in data] == ["hypothesis", "signal_defined", "backtest_result"]
    assert data[2]["data"]["n_events"] == 84


def test_unknown_kind_is_rejected():
    with pytest.raises(ValidationError):
        Notebook().add("thought", 1, "no such kind")


@pytest.mark.parametrize(
    "data",
    [{"thinking": "..."}, {"raw": []}, {"result": {"blocks": [{"signature": "x"}]}}],
)
def test_thinking_and_raw_blocks_never_enter_the_notebook(data):
    nb = Notebook()
    with pytest.raises(ValidationError):
        nb.add("note", 1, "leak", data)
    assert nb.entries == []


def test_entries_are_written_to_the_trace():
    trace = Trace("s")
    nb = Notebook(trace)
    nb.add("note", 4, "checked budget", {"remaining": 10})
    assert trace.events[-1]["kind"] == "notebook"
    assert trace.events[-1]["entry_kind"] == "note"
    assert trace.events[-1]["summary"] == "checked budget"
    assert trace.events[-1]["data"] == {"remaining": 10}
