import pandas as pd
import pytest
from pydantic import ValidationError

from backtest.selection import select_events
from dsl.signal import SignalSpec


def example_spec():
    return SignalSpec(
        event_type="dividend",
        conditions=[
            {
                "field": "dividend_change_pct",
                "op": ">",
                "value": 10,
            },
            {
                "field": "mcap_bucket",
                "op": "==",
                "value": "small",
            },
        ],
    )


def example_events():
    return pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d", "e", "f"],
            "event_type": ["dividend"] * 6,
            "is_new_announcement": [
                True, True, True, False, True, True
            ],
            "extraction_status": [
                "passed", "passed", "passed",
                "passed", "flagged", "passed",
            ],
            "dividend_change_pct": [
                15.0, 10.0, 20.0, 20.0, 20.0, None
            ],
            "mcap_bucket": [
                "small", "small", "large",
                "small", "small", "small",
            ],
        }
    )


def test_signal_round_trip():
    spec = example_spec()

    restored = SignalSpec.model_validate_json(
        spec.model_dump_json()
    )

    assert restored == spec
    assert spec.holding_days == 5
    assert spec.direction == "long"


def test_selection_and_input_unchanged():
    events = example_events()
    original = events.copy(deep=True)

    selected = select_events(events, example_spec())

    assert selected["event_id"].tolist() == ["a"]
    pd.testing.assert_frame_equal(events, original)


@pytest.mark.parametrize(
    "changes",
    [
        {"holding_days": 0},
        {"holding_days": True},
        {"holding_days": 61},
        {"direction": "sideways"},
        {"entry": "before_announcement"},
        {"unexpected": "argument"},
        {
            "conditions": [
                {
                    "field": "buyback_pct_mcap",
                    "op": ">",
                    "value": 5,
                }
            ]
        },
        {
            "conditions": [
                {
                    "field": "mcap_bucket",
                    "op": "==",
                    "value": "tiny",
                }
            ]
        },
        {
            "text_filter": {
                "feature": "buyback_rationale",
                "equals": "undervalued",
            }
        },
    ],
)
def test_invalid_signals_rejected(changes):
    inputs = {"event_type": "dividend", **changes}

    with pytest.raises(ValidationError):
        SignalSpec.model_validate(inputs)


def test_missing_value_does_not_match_not_equal():
    spec = SignalSpec(
        event_type="dividend",
        conditions=[
            {
                "field": "dividend_change_pct",
                "op": "!=",
                "value": 10,
            }
        ],
    )

    selected = select_events(example_events(), spec)

    assert selected["event_id"].tolist() == ["a", "c"]


def test_duplicate_event_ids_rejected():
    events = example_events()
    events.loc[1, "event_id"] = "a"

    with pytest.raises(ValueError, match="unique"):
        select_events(events, example_spec())


def test_text_filter():
    events = example_events()
    events["text_features"] = [
        {"dividend_framing": {"label": "confidence"}},
        {},
        {},
        {},
        {},
        {},
    ]
    spec = example_spec().model_copy(
        update={
            "text_filter": SignalSpec(
                event_type="dividend",
                text_filter={
                    "feature": "dividend_framing",
                    "equals": "confidence",
                },
            ).text_filter
        }
    )

    selected = select_events(events, spec)

    assert selected["event_id"].tolist() == ["a"]