"""LLM clients behind one provider-neutral interface.

The loop only sees these neutral messages:

    {"role": "user", "text": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...], "raw": Any}
    {"role": "tool_results", "results": [{"tool_call_id": str, "content": str, "is_error": bool}]}

"raw" on an assistant message is the provider's own content (e.g. Anthropic content
blocks, including signed thinking blocks). A client may replay it verbatim; clients
that don't understand it ignore it. Tools are [{"name", "description", "input_schema"}].
"""

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


class LLMResponse(BaseModel):
    text: str = ""
    thinking: str = ""  # readable reasoning text, if the provider returns any
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = Field(default_factory=Usage)
    raw: Any = None  # provider content, JSON-serialisable


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def assistant_message(response: LLMResponse) -> dict:
    """The neutral assistant message to append after a model turn."""
    return {
        "role": "assistant",
        "text": response.text,
        "tool_calls": list(response.tool_calls),
        "raw": response.raw,
    }


# --- Fake (tests) -------------------------------------------------------------


class FakeLLMExhausted(RuntimeError):
    pass


class FakeLLM:
    """Returns scripted responses in order and records every call it receives."""

    model = "fake"

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls: list[dict] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append(
            {"system": system, "messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)}
        )
        if len(self.calls) > len(self.script):
            raise FakeLLMExhausted(
                f"FakeLLM script has {len(self.script)} responses; call {len(self.calls)} has none"
            )
        return self.script[len(self.calls) - 1]


# --- Cache --------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def cache_key(settings: Any, system: str, messages: list[dict], tools: list[dict]) -> str:
    """settings: the model and anything else that changes the answer (max_tokens, effort)."""
    payload = {"settings": settings, "system": system, "messages": messages, "tools": tools}
    text = json.dumps(payload, sort_keys=True, default=_jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedClient:
    """Wraps a client; identical requests are answered from JSON files in cache_dir."""

    SETTINGS = ("model", "max_tokens", "effort", "thinking_display")

    def __init__(self, inner: LLMClient, cache_dir: str | Path, seed: int | None = None):
        """seed (the runner's per-pair seed) joins the key, so repeated seeds on the same
        universe get fresh samples instead of cache replays; None keeps the old keys."""
        self.inner = inner
        self.model = getattr(inner, "model", None)
        self.settings = {name: getattr(inner, name, None) for name in self.SETTINGS}
        if seed is not None:
            self.settings["seed"] = seed
        self.cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        path = self.cache_dir / f"{cache_key(self.settings, system, messages, tools)}.json"
        if path.exists():
            self.hits += 1
            return LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        self.misses += 1
        response = self.inner.complete(system, messages, tools)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(response.model_dump_json(), encoding="utf-8")
        return response


# --- Anthropic ----------------------------------------------------------------


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Neutral messages -> Messages API messages."""
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            content = m.get("raw")
            if content is None:  # not produced by Anthropic (e.g. a scripted turn)
                content = [{"type": "text", "text": m["text"]}] if m.get("text") else []
                for call in m.get("tool_calls", []):
                    call = ToolCall.model_validate(call)
                    block = {"type": "tool_use", "id": call.id, "name": call.name}
                    content.append({**block, "input": call.arguments})
            if content:
                out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool_results":
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": r["tool_call_id"],
                    "content": r["content"],
                    "is_error": r["is_error"],
                }
                for r in m["results"]
            ]
            out.append({"role": "user", "content": blocks})
        else:
            raise ValueError(f"unknown message role: {m['role']}")
    return out


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in tools
    ]


def from_anthropic_message(message: dict) -> LLMResponse:
    """A Messages API response (as a JSON dict) -> LLMResponse."""
    content = message.get("content", [])
    text = "".join(b["text"] for b in content if b["type"] == "text")
    thinking = "".join(b.get("thinking", "") for b in content if b["type"] == "thinking")
    calls = [
        ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
        for b in content
        if b["type"] == "tool_use"
    ]
    usage = message.get("usage") or {}
    return LLMResponse(
        text=text,
        thinking=thinking,
        tool_calls=calls,
        stop_reason=message.get("stop_reason") or "",
        usage=Usage(
            input_tokens=usage.get("input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            cache_read_input_tokens=usage.get("cache_read_input_tokens") or 0,
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens") or 0,
        ),
        raw=content,
    )


THINKING_DISPLAYS = ("summarized", "omitted")  # anthropic.types.ThinkingConfigAdaptiveParam


def build_anthropic_request(
    model: str,
    max_tokens: int,
    effort: str | None,
    system: str,
    messages: list[dict],
    tools: list[dict],
    thinking_display: str | None = None,
) -> dict:
    """Keyword arguments for messages.create(). Optional settings are left out when unset."""
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": to_anthropic_messages(messages),
        "cache_control": {"type": "ephemeral"},  # the loop resends the whole history each step
    }
    if tools:
        request["tools"] = to_anthropic_tools(tools)
    if effort:
        request["output_config"] = {"effort": effort}
    if thinking_display:
        request["thinking"] = {"type": "adaptive", "display": thinking_display}
    return request


LIVE_IN_TESTS_ENV = "AGENT_ALLOW_LIVE_LLM_IN_TESTS"


class LiveLLMInTestError(RuntimeError):
    """A real LLM client was about to be built inside a test run."""


def refuse_live_llm_under_pytest() -> None:
    """Tests must never call a real LLM: refuse while pytest runs (PYTEST_CURRENT_TEST),
    unless AGENT_ALLOW_LIVE_LLM_IN_TESTS=1 says so explicitly."""
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get(LIVE_IN_TESTS_ENV) != "1":
        raise LiveLLMInTestError(
            "refusing to build a real AnthropicClient under pytest: tests must use FakeLLM "
            f"(set {LIVE_IN_TESTS_ENV}=1 only for a deliberate live test)"
        )


class AnthropicClient:
    """Claude via the Messages API. Settings come from config, never code."""

    def __init__(
        self,
        model: str | None,
        max_tokens: int,
        effort: str | None = None,
        thinking_display: str | None = None,
    ):
        if not model:
            raise ValueError("No model set: fill agent.model in the config or set $AGENT_MODEL")
        if thinking_display not in (None, *THINKING_DISPLAYS):
            raise ValueError(f"thinking_display must be one of {THINKING_DISPLAYS} or null")
        refuse_live_llm_under_pytest()
        import anthropic  # agent extra; imported here so tests never need it

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking_display = thinking_display
        self._client = anthropic.Anthropic()  # key from $ANTHROPIC_API_KEY

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        request = build_anthropic_request(
            self.model, self.max_tokens, self.effort, system, messages, tools,
            self.thinking_display,
        )
        message = self._client.messages.create(**request)
        return from_anthropic_message(message.model_dump(mode="json", exclude_none=True))
