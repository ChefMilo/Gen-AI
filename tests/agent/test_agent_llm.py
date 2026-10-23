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
        {"type": "thinking", "thinking": "", "signature": "sig"},
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
    assert response.tool_calls == [call("run_backtest", "t9", signal_id="s1")]
    assert response.stop_reason == "tool_use"
    assert response.usage == Usage(input_tokens=120, output_tokens=30)
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
