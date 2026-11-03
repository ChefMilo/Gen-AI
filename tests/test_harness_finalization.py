import pytest

from harness.backtester import (
    GuardrailError,
    HarnessConfig,
    ResearchBacktester,
)
from harness.corrections import holm_adjust
from test_backtest_engine import sample_data as sample_data
from test_harness import harness_data as harness_data


def test_holm_known_answer():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    assert holm_adjust([0.02], family_size=5) == pytest.approx([0.10])

    assert holm_adjust([]) == []


def make_session(harness_data, tmp_path, budget=2):
    prices, benchmarks, events = harness_data

    return ResearchBacktester(
        prices,
        benchmarks,
        events,
        HarnessConfig(
            research_start="2020-01-06",
            research_end="2020-01-13",
            budget=budget,
        ),
        tmp_path / "finalization.jsonl",
    )


def test_unreported_tests_are_included(
    harness_data,
    tmp_path,
    monkeypatch,
):
    import harness.backtester as module

    original = module.run_event_study
    p_values = iter([0.04, 0.90])

    def controlled_result(*args, **kwargs):
        result = original(*args, **kwargs)
        result.p_value_raw = next(p_values)
        return result

    monkeypatch.setattr(module, "run_event_study", controlled_result)

    harness = make_session(harness_data, tmp_path)
    first = harness.define({"event_type": "dividend"})
    second = harness.define({"event_type": "dividend"})
    harness.run(first)
    harness.run(second)

    findings = harness.finalize(
        claims=[
            {
                "signal_id": first,
                "rationale": "Report only the better-looking result",
                "evidence": [],
            }
        ],
        session_id="session-test",
        system="R2",
    )

    assert len(findings) == 1
    assert findings[0].p_value_raw == 0.04
    assert findings[0].p_value_corrected == pytest.approx(0.08)
    assert not findings[0].passed_correction

    with pytest.raises(GuardrailError, match="closed"):
        harness.run(first)

    with pytest.raises(GuardrailError, match="closed"):
        harness.define({"event_type": "dividend"})


def test_untested_claim_rejected(harness_data, tmp_path):
    harness = make_session(harness_data, tmp_path)
    signal_id = harness.define({"event_type": "dividend"})

    with pytest.raises(GuardrailError, match="no completed backtest"):
        harness.finalize(
            claims=[
                {
                    "signal_id": signal_id,
                    "rationale": "Never tested",
                }
            ],
            session_id="session-test",
            system="R2",
        )


def test_insufficient_sample_cannot_become_finding(
    harness_data,
    tmp_path,
):
    prices, benchmarks, events = harness_data
    harness = make_session(
        (prices, benchmarks, events.iloc[:2]),
        tmp_path,
    )
    signal_id = harness.define({"event_type": "dividend"})
    harness.run(signal_id)

    findings = harness.finalize(
        claims=[
            {
                "signal_id": signal_id,
                "rationale": "Only two observations",
            }
        ],
        session_id="session-test",
        system="R2",
    )

    assert findings == []
