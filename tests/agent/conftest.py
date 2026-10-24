"""Shared helpers for agent tests: stub ports and an in-memory session."""

from datetime import date

import pytest

import agent.session as session_module
import agent.tools as tools
from agent.session import new_session


class StubBacktester:
    def define(self, spec):
        return "sig-1"

    def run(self, signal_id):
        return {}

    def breakdown(self, signal_id, by):
        return {}

    def signal_options(self):
        return {"event_types": ["dividend"], "operators": {"number": [">"]},
                "holding_days": [5], "directions": ["long"], "breakdowns": ["year"]}

    def budget_status(self):
        return {"budget": 3, "tests_used": 0}


class StubEvents:
    def query(self, event_type, conditions, limit):
        return {"n": 0}

    def available_fields(self):
        return {}


class StubRetriever:
    def search(self, query, filters, k):
        return []

    def text_feature(self, event_ids, feature):
        return []

    def available_text_features(self):
        return {}


def agent_config(**overrides):
    agent = {"model": "m", "max_tokens": 16000, "effort": None, "tool_result_max_chars": 4000}
    return {"research_start": date(2016, 1, 1), "research_end": None, "agent": agent | overrides}


@pytest.fixture
def make_session(monkeypatch):
    monkeypatch.setattr(session_module, "code_version", lambda: None)

    def make(system="R2", **agent_overrides):
        retriever = StubRetriever() if system in {"R3", "R3-U"} else None
        return new_session(
            system,
            agent_config(**agent_overrides),
            "test_universe",
            StubBacktester(),
            StubEvents(),
            retriever=retriever,
            research_end="2022-12-31",
            in_memory=True,
        )

    return make


@pytest.fixture
def registry(monkeypatch):
    """A private copy of the tool registry, so tests can register tools freely."""
    monkeypatch.setattr(tools, "_TOOLS", dict(tools._TOOLS))
    return tools._TOOLS


# --- Fake ports on Member 5's synthetic data (built once per test run) -------------------


@pytest.fixture(scope="session")
def events():
    from agent.fakes import make_fake_universe

    return make_fake_universe(seed=0)


@pytest.fixture
def fake_session(events, registry, monkeypatch):
    """make(system, budget, **agent_overrides) -> session on fresh fake ports."""
    from agent.fakes import (
        FAKE_RESEARCH_END,
        FakeBacktester,
        FakeEventQuery,
        FakeRetriever,
        register_fake_tools,
    )

    monkeypatch.setattr(session_module, "code_version", lambda: None)
    register_fake_tools()

    def make(system="R2", budget=15, **agent_overrides):
        config = agent_config(tool_result_max_chars=20000, **agent_overrides)
        config["min_events"] = 20
        return new_session(
            system, config, "fake_planted_seed0",
            FakeBacktester(events, budget), FakeEventQuery(events),
            retriever=FakeRetriever(events) if system in {"R3", "R3-U"} else None,
            research_end=FAKE_RESEARCH_END, in_memory=True,
        )

    return make
