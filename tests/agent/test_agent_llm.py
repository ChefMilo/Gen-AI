import json

import pytest

from agent.llm import (
    AnthropicClient,
    CachedClient,
    FakeLLM,
    FakeLLMExhausted,
    LLMResponse,
    ToolCall,
    Usage,
    assistant_message,
    build_anthropic_request,
    cache_key,
    from_anthropic_message,
    to_anthropic_messages,
    to_anthropic_tools,
)

TOOLS = [{"name": "finish", "description": "End.", "input_schema": {"type": "object"}}]


def call(name="finish", id="t1", **arguments):
    return ToolCall(id=id, name=name, arguments=arguments)


# --- FakeLLM ------------------------------------------------------------------


def test_fake_llm_returns_script_in_order_and_records_calls():
    first, second = LLMResponse(text="one"), LLMResponse(text="two")
    llm = FakeLLM([first, second])
    messages = [{"role": "user", "text": "hi"}]
    assert llm.complete("sys", messages, TOOLS) is first
    messages.append({"role": "user", "text": "again"})
    assert llm.complete("sys", messages, TOOLS) is second
    assert [len(c["messages"]) for c in llm.calls] == [1, 2]  # copies, not the live list
    assert llm.calls[0]["system"] == "sys" and llm.calls[0]["tools"] == TOOLS


def test_fake_llm_raises_clearly_when_script_runs_out():
    llm = FakeLLM([LLMResponse(text="only")])
    llm.complete("s", [], [])
    with pytest.raises(FakeLLMExhausted, match="1 responses; call 2"):
        llm.complete("s", [], [])


# --- CachedClient -------------------------------------------------------------


def test_cached_client_miss_then_hit(tmp_path):
    response = LLMResponse(text="x", tool_calls=[call(reason="done")], usage=Usage(input_tokens=5))
    inner = FakeLLM([response])
    cached = CachedClient(inner, tmp_path)
    messages = [{"role": "user", "text": "hi"}]

    assert cached.complete("sys", messages, TOOLS) == response
    assert cached.complete("sys", messages, TOOLS) == response  # FakeLLM would raise if called
    assert (cached.hits, cached.misses, len(inner.calls)) == (1, 1, 1)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cached_client_different_request_is_a_miss(tmp_path):
    inner = FakeLLM([LLMResponse(text="a"), LLMResponse(text="b")])
    cached = CachedClient(inner, tmp_path)
    assert cached.complete("sys", [{"role": "user", "text": "1"}], TOOLS).text == "a"
    assert cached.complete("sys", [{"role": "user", "text": "2"}], TOOLS).text == "b"
    assert cached.misses == 2


def test_cache_key_depends_on_every_part():
    base = ("m", "sys", [{"role": "user", "text": "hi"}], TOOLS)
    key = cache_key(*base)
    assert cache_key(*base) == key
    assert cache_key("other", *base[1:]) != key
    assert cache_key(base[0], "other", *base[2:]) != key
    assert cache_key(*base[:2], [{"role": "user", "text": "bye"}], TOOLS) != key
    assert cache_key(*base[:3], []) != key


def test_cache_is_not_shared_across_settings(tmp_path):
    first, second = FakeLLM([LLMResponse(text="a")]), FakeLLM([LLMResponse(text="b")])
    first.effort, second.effort = "low", "high"
    messages = [{"role": "user", "text": "hi"}]
    assert CachedClient(first, tmp_path).complete("s", messages, TOOLS).text == "a"
    assert CachedClient(second, tmp_path).complete("s", messages, TOOLS).text == "b"


def test_cache_key_accepts_tool_call_models():
    def history():
        return [assistant_message(LLMResponse(tool_calls=[call()]))]

    assert cache_key("m", "s", history(), []) == cache_key("m", "s", history(), [])


# --- Anthropic conversion (pure functions, no client) -------------------------


def test_neutral_messages_convert_to_anthropic_format():
    messages = [
        {"role": "user", "text": "brief"},
        assistant_message(LLMResponse(text="thinking aloud", tool_calls=[call("a", "t1", x=1)])),
        {
            "role": "tool_results",
            "results": [
                {"tool_call_id": "t1", "content": '{"ok": true}', "is_error": False},
                {"tool_call_id": "t2", "content": "bad", "is_error": True},
            ],
        },
    ]
    assert to_anthropic_messages(messages) == [
        {"role": "user", "content": "brief"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking aloud"},
                {"type": "tool_use", "id": "t1", "name": "a", "input": {"x": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": '{"ok": true}',
                 "is_error": False},
                {"type": "tool_result", "tool_use_id": "t2", "content": "bad", "is_error": True},
            ],
        },
    ]


def test_assistant_raw_content_is_replayed_verbatim():
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
    ]
    msg = {"role": "assistant", "text": "", "tool_calls": [call("a")], "raw": raw}
    assert to_anthropic_messages([msg]) == [{"role": "assistant", "content": raw}]


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        to_anthropic_messages([{"role": "system", "text": "x"}])


def test_tools_keep_only_api_fields():
    tools = [{**TOOLS[0], "systems": ["R2"]}]
    assert to_anthropic_tools(tools) == TOOLS


def test_anthropic_response_converts_to_llm_response():
    content = [
        {"type": "thinking", "thinking": "Small caps react more.", "signature": "sig"},
        {"type": "text", "text": "Testing "},
        {"type": "text", "text": "now."},
        {"type": "tool_use", "id": "t9", "name": "run_backtest", "input": {"signal_id": "s1"}},
    ]
    message = {
        "content": content,
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 100},
    }
    response = from_anthropic_message(message)
    assert response.text == "Testing now."
    assert response.thinking == "Small caps react more."
    assert response.tool_calls == [call("run_backtest", "t9", signal_id="s1")]
    assert response.stop_reason == "tool_use"
    assert response.usage == Usage(input_tokens=120, output_tokens=30, cache_read_input_tokens=100)
    assert response.usage.total_input_tokens == 220
    assert response.raw == content
    json.dumps(response.model_dump(mode="json"))  # cacheable


def test_anthropic_round_trip_keeps_tool_ids():
    message = {"content": [{"type": "tool_use", "id": "t1", "name": "a", "input": {}}],
               "stop_reason": "tool_use", "usage": {}}
    converted = to_anthropic_messages([assistant_message(from_anthropic_message(message))])
    assert converted[0]["content"][0]["id"] == "t1"


def test_anthropic_client_needs_a_model():
    with pytest.raises(ValueError, match="AGENT_MODEL"):
        AnthropicClient(model=None, max_tokens=100)  # fails before importing anthropic


def test_request_omits_optional_settings_when_unset():
    request = build_anthropic_request("m", 16000, None, "sys", [{"role": "user", "text": "hi"}], [])
    assert "output_config" not in request
    assert "tools" not in request
    assert request["model"] == "m" and request["max_tokens"] == 16000
    assert request["messages"] == [{"role": "user", "content": "hi"}]


def test_request_omits_thinking_when_display_unset():
    request = build_anthropic_request("m", 16000, None, "sys", [], [], None)
    assert "thinking" not in request


@pytest.mark.parametrize("display", ["summarized", "omitted"])
def test_request_passes_thinking_display_when_set(display):
    request = build_anthropic_request("m", 16000, None, "sys", [], [], display)
    assert request["thinking"] == {"type": "adaptive", "display": display}


def test_anthropic_client_rejects_unknown_thinking_display():
    with pytest.raises(ValueError, match="thinking_display"):
        AnthropicClient(model="m", max_tokens=100, thinking_display="loud")


def test_cache_key_includes_thinking_display(tmp_path):
    a, b = FakeLLM([LLMResponse(text="a")]), FakeLLM([LLMResponse(text="b")])
    a.thinking_display, b.thinking_display = "summarized", "omitted"
    messages = [{"role": "user", "text": "hi"}]
    assert CachedClient(a, tmp_path).complete("s", messages, TOOLS).text == "a"
    assert CachedClient(b, tmp_path).complete("s", messages, TOOLS).text == "b"


def test_request_passes_effort_and_tools_when_set():
    request = build_anthropic_request("m", 16000, "high", "sys", [], TOOLS)
    assert request["output_config"] == {"effort": "high"}
    assert request["tools"] == TOOLS


def test_anthropic_client_refuses_to_build_under_pytest(monkeypatch):
    from agent.llm import LiveLLMInTestError

    assert "PYTEST_CURRENT_TEST" in __import__("os").environ
    with pytest.raises(LiveLLMInTestError, match="FakeLLM"):
        AnthropicClient(model="m", max_tokens=100)


def test_live_client_in_tests_needs_the_explicit_opt_in(monkeypatch):
    """With AGENT_ALLOW_LIVE_LLM_IN_TESTS=1 the guard steps aside (a stub SDK here, so
    nothing is called)."""
    import sys
    import types

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: "stub-sdk-client"
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("AGENT_ALLOW_LIVE_LLM_IN_TESTS", "1")
    assert AnthropicClient(model="m", max_tokens=100)._client == "stub-sdk-client"
    monkeypatch.setenv("AGENT_ALLOW_LIVE_LLM_IN_TESTS", "yes")  # only "1" counts
    with pytest.raises(RuntimeError, match="refusing"):
        AnthropicClient(model="m", max_tokens=100)


def test_dotenv_leaks_cannot_reach_a_real_client(tmp_path, monkeypatch):
    """Regression: src/ingest/llm.py loads .env into os.environ for the rest of the pytest
    process; an agent test must still not get credentials or a live client."""
    import os

    from ingest.llm import load_dotenv

    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-test-not-real\nAGENT_MODEL=claude-test\n")
    for name in ("ANTHROPIC_API_KEY", "AGENT_MODEL"):  # so teardown removes the fake values
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    load_dotenv(env)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-not-real"  # the leak, reproduced
    import agent.system as system_module

    with pytest.raises(RuntimeError, match="real LLM client"):
        system_module.default_client({"model": "claude-test", "max_tokens": 10}, 0)
    with pytest.raises(RuntimeError, match="refusing"):
        AnthropicClient(model="claude-test", max_tokens=10)


def test_default_client_settings_come_from_dotenv_without_touching_the_environment(
        tmp_path, monkeypatch):
    """python -m evaluation.run never loads .env, so the agent's default client reads the
    model and key from it directly, and must not leave them in os.environ."""
    import os

    from agent.system import env_or_dotenv

    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-test-not-real\nAGENT_MODEL=claude-test\n")
    assert env_or_dotenv("AGENT_MODEL", env) == "claude-test"
    assert env_or_dotenv("ANTHROPIC_API_KEY", env) == "sk-test-not-real"
    assert "ANTHROPIC_API_KEY" not in os.environ and "AGENT_MODEL" not in os.environ
    monkeypatch.setenv("AGENT_MODEL", "from-env")  # the environment wins
    assert env_or_dotenv("AGENT_MODEL", env) == "from-env"
    assert env_or_dotenv("AGENT_MODEL", tmp_path / "missing.env") == "from-env"
    assert env_or_dotenv("NOT_THERE", env) is None


def test_anthropic_client_passes_an_explicit_key_to_the_sdk(monkeypatch):
    import sys
    import types

    seen = {}
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda **kwargs: seen.update(kwargs) or "stub"
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("AGENT_ALLOW_LIVE_LLM_IN_TESTS", "1")
    AnthropicClient(model="m", max_tokens=10, api_key="sk-explicit")
    assert seen == {"api_key": "sk-explicit"}
