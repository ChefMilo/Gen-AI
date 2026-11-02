from datetime import date

import pandas as pd

from store import events as store
from store import prices
from store.schema import Event
from universes.fake_events import make_fake_events, rows_as_events
from universes.synthetic import make_synthetic_prices


def _fake(tmp_path, monkeypatch):
    px = make_synthetic_prices(["A", "B", "C"], "2023-01-01", "2024-12-31", seed=0)
    events = make_fake_events(px, seed=0)
    pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "recommendation": ["keep", "keep_from", "drop"],
            "usable_from": ["2016-01-01", "2024-01-01", None],
        }
    ).to_csv(tmp_path / prices.UNIVERSE_QUALITY, index=False)
    monkeypatch.setenv(prices.DATA_DIR_ENV, str(tmp_path))
    return events


def test_round_trip_keeps_every_event(tmp_path, monkeypatch):
    events = _fake(tmp_path, monkeypatch)
    path = store.save_events(events)
    assert path == tmp_path / store.EVENTS
    back = store.load_events(usable_only=False)
    assert list(back.columns) == list(Event.model_fields)
    assert len(back) == len(events)
    models = rows_as_events(back)  # validates every row
    original = {e.event_id: e for e in rows_as_events(events)}
    for m in models[:50]:
        assert m == original[m.event_id]


def test_usable_only_and_date_range(tmp_path, monkeypatch):
    events = _fake(tmp_path, monkeypatch)
    store.save_events(events)
    usable = store.load_events()
    assert usable["is_new_announcement"].all()
    assert (usable["extraction_status"] == "passed").all()
    assert "C" not in set(usable["ticker"])
    b = usable[usable["ticker"] == "B"]
    assert (pd.to_datetime(b["first_tradeable_date"]) >= "2024-01-01").all()
    window = store.load_events(start="2024-03-01", end=date(2024, 8, 31))
    days = pd.to_datetime(window["first_tradeable_date"])
    assert len(window) and days.min() >= pd.Timestamp("2024-03-01")
    assert days.max() <= pd.Timestamp("2024-08-31")
