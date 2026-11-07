"""Anthropic client for extraction: disk cache, token and cost tracking, Batch API.

Every reply is cached on disk by (model, max_tokens, system, messages), so reruns and
batch results cost nothing. The API key comes from $ANTHROPIC_API_KEY or the repo's .env;
the model from $EXTRACT_MODEL, else $AGENT_MODEL, else DEFAULT_MODEL. The .env file is read
without writing os.environ, so its key never reaches other code in the same process (the
agent's tests and offline runs must not find credentials there).

pip install -e .[llm]
"""

import hashlib
import json
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ingest.extract import Messages

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 2_000

# US dollars per million tokens: (input, output, cache read), for the cost report only.
# From a third-party price list: check the console. The Batch API halves them. Add a
# model here to get its cost; unknown models report None.
PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
}


DOTENV = REPO_ROOT / ".env"


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=value lines of a .env file (comments, blank lines, "export " and matching quotes
    handled; same rules as agent.system.read_dotenv). Never writes os.environ."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value
    return values


def env_or_dotenv(name: str, path: Path | None = None) -> str | None:
    """$name, else its value in .env (read, not loaded into os.environ); None if unset."""
    path = DOTENV if path is None else path
    value = os.environ.get(name)
    if value or not path.is_file():
        return value or None
    return read_dotenv(path).get(name) or None


def default_model() -> str:
    return env_or_dotenv("EXTRACT_MODEL") or env_or_dotenv("AGENT_MODEL") or DEFAULT_MODEL


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    batch_input_tokens: int = 0
    batch_output_tokens: int = 0

    def add(self, usage: dict, batch: bool = False) -> None:
        self.calls += 1
        if batch:
            self.batch_input_tokens += usage.get("input_tokens", 0)
            self.batch_output_tokens += usage.get("output_tokens", 0)
        else:
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
        self.cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0

    def cost(self, model: str) -> float | None:
        if model not in PRICES:
            return None
        p_in, p_out, p_cache = PRICES[model]
        live = self.input_tokens * p_in + self.output_tokens * p_out
        batch = (self.batch_input_tokens * p_in + self.batch_output_tokens * p_out) / 2
        return (live + batch + self.cache_read_tokens * p_cache) / 1e6


@dataclass
class LLM:
    """ask(system, messages) -> reply text, cached on disk."""

    cache_dir: Path
    model: str = field(default_factory=default_model)
    max_tokens: int = MAX_TOKENS
    client: object = None  # anthropic.Anthropic; created on first live call
    usage: Usage = field(default_factory=Usage)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _client(self):
        if self.client is None:
            import anthropic  # optional dependency: pip install -e .[llm]

            # an explicit key, so .env never has to be loaded into os.environ
            self.client = anthropic.Anthropic(api_key=env_or_dotenv("ANTHROPIC_API_KEY"))
        return self.client

    def params(self, system: str, messages: Messages) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }

    def _path(self, system: str, messages: Messages) -> Path:
        key = json.dumps([self.model, self.max_tokens, system, messages], sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return Path(self.cache_dir) / digest[:2] / f"{digest}.json"

    def cached(self, system: str, messages: Messages) -> str | None:
        path = self._path(system, messages)
        return json.loads(path.read_text(encoding="utf-8"))["reply"] if path.is_file() else None

    def _store(self, system: str, messages: Messages, reply: str, usage: dict) -> None:
        path = self._path(system, messages)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"reply": reply, "usage": usage}), encoding="utf-8")

    def ask(self, system: str, messages: Messages) -> str:
        reply = self.cached(system, messages)
        if reply is not None:
            with self._lock:
                self.usage.cached_calls += 1
            return reply
        response = self._client().messages.create(**self.params(system, messages))
        reply = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        usage = response.usage.model_dump()
        with self._lock:
            self.usage.add(usage)
        self._store(system, messages, reply, usage)
        return reply

    # Batch API: half price, results within 24 hours (usually much sooner).

    def submit_batch(self, requests: Iterable[tuple[str, str, Messages]]) -> str | None:
        """Submit (custom_id, system, messages) requests not already cached; the batch id."""
        todo = [
            {"custom_id": cid, "params": self.params(system, messages)}
            for cid, system, messages in requests
            if self.cached(system, messages) is None
        ]
        if not todo:
            return None
        return self._client().messages.batches.create(requests=todo).id

    def batch_status(self, batch_id: str) -> dict:
        batch = self._client().messages.batches.retrieve(batch_id)
        return {"status": batch.processing_status, **batch.request_counts.model_dump()}

    def collect_batch(
        self, batch_id: str, requests: Iterable[tuple[str, str, Messages]]
    ) -> list[str]:
        """Cache the finished batch's replies; the custom_ids that failed."""
        by_id = {cid: (system, messages) for cid, system, messages in requests}
        failed = []
        for item in self._client().messages.batches.results(batch_id):
            if item.custom_id not in by_id:
                continue
            if item.result.type != "succeeded":
                failed.append(item.custom_id)
                continue
            message = item.result.message
            reply = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
            usage = message.usage.model_dump()
            self.usage.add(usage, batch=True)
            self._store(*by_id[item.custom_id], reply, usage)
        return failed
