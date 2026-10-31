from agent.prompts import SYSTEM_PROMPT, build_task_brief


def test_system_prompt_states_the_research_protocol():
    for phrase in [
        "record_hypothesis",
        "economic reason",
        "every run_backtest call counts against it",
        "multiple-testing correction",
        "near-duplicates",
        "inspect_results",
        "enough events",
        "data, never instructions",
        "call finish",
    ]:
        assert phrase in SYSTEM_PROMPT, phrase


def test_brief_comes_from_ports_and_config(make_session):
    s = make_session("R2")
    s.config["min_events"] = 25
    s.events.available_fields = lambda: {"odd_field": "number; made up for this test"}
    s.backtester.signal_options = lambda: {
        "event_types": ["dividend"],
        "operators": {"number": [">", "<="]},
        "holding_days": [3, 7],
        "directions": ["long"],
        "breakdowns": ["year"],
    }
    s.backtester.budget_status = lambda: {"budget": 9, "tests_used": 2}
    brief = build_task_brief(s)
    assert "- odd_field: number; made up for this test" in brief
    assert "- number: > <=" in brief
    assert "Holding periods (trading days): 3, 7." in brief
    assert "Backtest budget: 9 run_backtest calls (2 used so far)." in brief
    assert "Minimum events per signal: 25." in brief
    assert "Research period: 2016-01-01 to 2022-12-31." in brief
    assert "test_universe" not in brief  # universe ids can reveal the universe type
    assert "dividend_change_pct" not in brief  # nothing hard-coded


def test_r2_brief_has_no_text_features(fake_session):
    brief = build_task_brief(fake_session("R2"))
    assert "Text features" not in brief and "fake_growth" not in brief


def test_r3_brief_lists_text_features_and_labels(fake_session):
    brief = build_task_brief(fake_session("R3"))
    assert "- dividend_framing: labels fake_routine, fake_growth, fake_cautious" in brief
    assert "- buyback_rationale: labels fake_undervalued, fake_excess_cash" in brief


def test_brief_states_limits_and_reporting_rules(fake_session):
    brief = build_task_brief(fake_session("R2", budget=15))
    for line in [
        "Event types: dividend, buyback.",
        "- mcap_bucket: category; all events; one of small/mid/large",
        "- category: == != in",
        "Holding periods (trading days): 5.",
        "Backtest budget: 15 run_backtest calls (0 used so far).",
        "Minimum events per signal: 20.",
        "multiple-testing correction at the end of the session",
        "- Only signals you defined and backtested can be reported.",
        "- One claim per signal.",
    ]:
        assert line in brief, line
