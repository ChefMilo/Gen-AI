from datetime import date
from typing import get_args

import pytest
from pydantic import ValidationError

import agent.session as session_module
from agent.finding import EvidenceQuote, FindingClaim, System
from agent.ports import Backtester, EventQuery, Retriever
from agent.session import AgentSystem, ResearchSession, new_session

CONFIG = {
    "research_start": date(2016, 1, 1),
    "research_end": None,  # as in configs/default.yaml today
    "agent": {"model": "m", "max_tokens": 16000, "effort": None, "trace_dir": None},
}


class StubBacktester:
    def define(self, spec):
        return "sig-1"

    def run(self, signal_id):
        return {}

    def breakdown(self, signal_id, by):
        return {}


class StubEvents:
    def query(self, event_type, conditions, limit):
        return {"n": 0}

    def available_fields(self):
        return {"mcap_bucket": "small / mid / large"}


class StubRetriever:
    def search(self, query, filters, k):
        return []

    def text_feature(self, event_ids, feature):
        return []

    def available_text_features(self):
        return {"dividend_framing": ["fake_routine"]}


@pytest.fixture(autouse=True)
def fixed_code_version(monkeypatch):
    monkeypatch.setattr(session_module, "code_version", lambda: {"commit": "abc", "dirty": True})


def make(system="R2", **kwargs) -> ResearchSession:
    kwargs.setdefault("research_end", "2022-12-31")
    return new_session(
        system, CONFIG, "fake_planted_seed0", StubBacktester(), StubEvents(), in_memory=True,
        **kwargs,
    )


def test_session_construction():
    s = make()
    assert s.system == "R2" and s.universe_id == "fake_planted_seed0"
    assert (s.research_start, s.research_end) == (date(2016, 1, 1), date(2022, 12, 31))
    assert len(s.session_id) == 32 and s.trace.session_id == s.session_id
    assert s.notebook.trace is s.trace
    assert (s.claims, s.signals, s.done, s.invalid_calls, s.step) == ([], {}, False, 0, 0)
    assert s.code_version == {"commit": "abc", "dirty": True}
    assert s.trace.path is None


def test_each_session_gets_a_new_id():
    assert make().session_id != make().session_id


def test_stubs_satisfy_the_ports():
    backtester: Backtester = StubBacktester()
    events: EventQuery = StubEvents()
    retriever: Retriever = StubRetriever()
    assert backtester.define({}) == "sig-1"
    assert "mcap_bucket" in events.available_fields()
    assert "dividend_framing" in retriever.available_text_features()


def test_agent_systems_are_finding_systems_without_r0():
    assert set(get_args(AgentSystem)) == set(get_args(System)) - {"R0"}


@pytest.mark.parametrize("system", ["R0", "R4", "r2"])
def test_unknown_or_non_agent_system_is_rejected(system):
    with pytest.raises(ValueError, match="system must be one of"):
        make(system)


def test_r3_needs_a_retriever():
    with pytest.raises(ValueError, match="retriever"):
        make("R3")
    assert make("R3-U", retriever=StubRetriever()).retriever is not None


def test_missing_research_end_explains_why():
    with pytest.raises(ValueError, match="not agreed"):
        make(research_end=None)


def test_research_dates_accept_strings_and_override_config():
    s = make(research_start="2018-01-01", research_end="2020-06-30")
    assert (s.research_start, s.research_end) == (date(2018, 1, 1), date(2020, 6, 30))


def test_research_end_must_follow_start():
    with pytest.raises(ValueError, match="after"):
        make(research_end="2015-12-31")


def test_trace_file_uses_configured_dir(tmp_path):
    config = {**CONFIG, "agent": {**CONFIG["agent"], "trace_dir": tmp_path}}
    s = new_session("R2", config, "u", StubBacktester(), StubEvents(), research_end="2022-12-31")
    assert s.trace.path == tmp_path / f"{s.session_id}.jsonl"


def test_log_start_records_settings_universe_period_and_code_version():
    s = make()
    event = s.log_start("SYSTEM PROMPT")
    assert event["kind"] == "session_start" and event["system"] == "SYSTEM PROMPT"
    assert event["universe_id"] == "fake_planted_seed0"
    assert event["agent_system"] == "R2"
    assert event["settings"] == {"model": "m", "max_tokens": 16000, "effort": None}
    assert event["code_version"] == {"commit": "abc", "dirty": True}
    assert (event["research_start"], event["research_end"]) == (date(2016, 1, 1),
                                                                 date(2022, 12, 31))


def test_finding_claim():
    claim = FindingClaim(
        signal_id="sig-1",
        rationale="Small-cap dividend hikes signal confidence.",
        evidence=[EvidenceQuote(event_id="e1", quote="raised the dividend 12%")],
        step=7,
    )
    assert FindingClaim.model_validate_json(claim.model_dump_json()) == claim


def test_finding_claim_reuses_evidence_rules():
    with pytest.raises(ValidationError, match="event_id or a doc_ref"):
        FindingClaim(signal_id="s", rationale="r", evidence=[{"quote": "orphan"}], step=1)
