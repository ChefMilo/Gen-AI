"""ingest.llm reads .env without loading it into os.environ (the live-call incident)."""

import os
import sys
import types

import pytest

import ingest.llm as llm_module

NAMES = ("ANTHROPIC_API_KEY", "EXTRACT_MODEL", "AGENT_MODEL")


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "# keys\nANTHROPIC_API_KEY=sk-test-not-real\nexport AGENT_MODEL='claude-test'\n"
        "EXTRACT_MODEL=\nOTHER=x # note\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_module, "DOTENV", path)
    return path


def test_default_model_reads_dotenv_without_touching_the_environment(env_file):
    assert llm_module.default_model() == "claude-test"  # EXTRACT_MODEL empty: skipped
    assert llm_module.env_or_dotenv("OTHER") == "x"
    assert not any(name in os.environ for name in NAMES)


def test_environment_wins_over_dotenv(env_file, monkeypatch):
    monkeypatch.setenv("EXTRACT_MODEL", "from-env")
    assert llm_module.default_model() == "from-env"


def test_client_gets_the_key_explicitly(env_file, monkeypatch):
    seen = {}
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda **kwargs: seen.update(kwargs) or "client"
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert llm_module.LLM(env_file.parent, model="m")._client() == "client"
    assert seen == {"api_key": "sk-test-not-real"}
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_no_dotenv_means_the_sdk_default(tmp_path, monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_module, "DOTENV", tmp_path / "missing.env")
    assert llm_module.default_model() == llm_module.DEFAULT_MODEL
    assert llm_module.env_or_dotenv("ANTHROPIC_API_KEY") is None
