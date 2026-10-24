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
