"""The agent as a research system in the team's experiment runner (evaluation.systems).

AgentResearchSystem implements ResearchSystem: run(view, budget, seed) -> SystemRun. Per
universe it builds the interim ports from the view (agent.interim), a session over the
view's research period, the LLM client from config, runs the loop, turns the claims into
Findings (agent.interim.harness) and returns them with the session metadata.

R2System, R3System and R3USystem take no arguments, as evaluation.systems.make_systems
builds them; agent.systems registers them as "r2", "r3" and "r3-u".

research_end None (the split is not agreed yet) is an open end on planted and placebo
universes. A view does not say which type it is, by design, so the caller may pass a
private universe_id -> universe_type map (agent.experiment does, from the plan). It is
used for one thing only: refusing a real universe without research_end, and refusing
every universe without research_end when there is no map. It never reaches the model,
the trace or the metadata.
"""

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agent.config import MODEL_ENV, REPO_ROOT, load_config
from agent.interim.harness import claims_to_findings
from agent.interim.ports import build_ports
from agent.interim.retrieval import RagFactory, shared_index_retriever
from agent.llm import AnthropicClient, CachedClient, LLMClient
from agent.loop import run_session
from agent.session import new_session
from evaluation.systems import SystemRun
from universes.planted import DEFAULT_HOLDING_DAYS

ClientFactory = Callable[[dict, int], LLMClient]  # (agent config, pair seed) -> client
OPEN_END_TYPES = frozenset({"planted", "placebo"})
TOKEN_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                "output_tokens")
PRICE_FIELDS = {"input_tokens": "input", "cache_read_input_tokens": "cache_read",
                "cache_creation_input_tokens": "cache_write", "output_tokens": "output"}


class ResearchEndNotSet(RuntimeError):
    pass


DOTENV = REPO_ROOT / ".env"


def env_or_dotenv(name: str, path: Path = DOTENV) -> str | None:
    """$name, else its value in the repo's .env, read WITHOUT writing os.environ (so the
    key never leaks into the rest of the process). python -m evaluation.run does not load
    .env itself."""
    value = os.environ.get(name)
    if value or not path.is_file():
        return value or None
    return read_dotenv(path).get(name) or None


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=value lines of a .env file (comments, blank lines, "export " and matching
    quotes handled), with no dependency, so it works where the agent extra is absent."""
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


def default_client(agent: dict, seed: int) -> LLMClient:
    """AnthropicClient from config (model: agent.model, else $AGENT_MODEL or .env), behind
    the response cache keyed by the pair seed. The key comes from $ANTHROPIC_API_KEY or
    .env."""
    client = AnthropicClient(agent.get("model") or env_or_dotenv(MODEL_ENV),
                             agent["max_tokens"], agent.get("effort"),
                             agent.get("thinking_display"),
                             api_key=env_or_dotenv("ANTHROPIC_API_KEY"))
    cache_dir = agent.get("cache_dir")
    return CachedClient(client, cache_dir, seed=seed) if cache_dir else client


def cost_usd(model: str | None, tokens: Mapping[str, int], pricing: Mapping) -> float | None:
    """From agent.pricing (USD per million tokens); None if the model is not priced."""
    prices = (pricing or {}).get(model)
    if not prices:
        return None
    return sum(tokens[f] * float(prices[p]) for f, p in PRICE_FIELDS.items()) / 1_000_000


def _tool_events(session, name: str) -> list[dict]:
    return [e for e in session.trace.events if e["kind"] == "tool_call" and e["name"] == name]


def breakdown_groups_viewed(session) -> int:
    """Subgroup results the agent saw through inspect_results (item 15 sensitivity)."""
    total = 0
    for e in _tool_events(session, "inspect_results"):
        if not e["is_error"]:
            try:
                total += len(json.loads(e["result"]).get("groups", {}))
            except ValueError:  # truncated result
                pass
    return total


def check_research_end(view, universe_types: Mapping[str, str] | None) -> None:
    """Refuse an open research_end unless the universe is known to be planted or placebo."""
    if view.research_end is not None:
        return
    if universe_types is None:
        raise ResearchEndNotSet(
            "research_end is not set and the universe type is unknown, so this run "
            "could touch holdout data. The team must set the research/holdout split "
            "(research_end in configs/default.yaml), or run through agent.experiment."
        )
    if universe_types.get(view.universe_id) not in OPEN_END_TYPES:
        raise ResearchEndNotSet(
            f"research_end is not set: refusing to run on real universe "
            f"{view.universe_id!r}. The team must set the research/holdout split "
            "(research_end in configs/default.yaml) before agents run on real data."
        )


def run_metadata(session, ports, report: dict, client, agent: dict, budget: int, seed: int,
                 view, started: float, mask: bool) -> dict[str, Any]:
    """SystemRun.metadata for a session over interim ports (agent systems and R1)."""
    start_event = session.trace.events[0]
    end_event = session.trace.events[-1]
    totals = end_event["totals"]
    tokens = {f: int(totals[f]) for f in TOKEN_FIELDS}
    model = agent.get("model") or getattr(client, "model", None)
    metadata = {
        "backtests_used": ports.backtester.tests_used,
        "tool_calls": totals["tool_calls"],
        "tokens": sum(tokens.values()),
        "cost_usd": cost_usd(model, tokens, agent.get("pricing")),
        "wall_time_s": time.perf_counter() - started,
        "trace_path": str(session.trace.path) if session.trace.path else None,
        "model": model,
        "prompt_version": start_event.get("prompt_version"),
        "end_reason": end_event["reason"],
        "invalid_calls": totals["invalid_calls"],
        "inspect_results_calls": len(_tool_events(session, "inspect_results")),
        "breakdown_groups_viewed": breakdown_groups_viewed(session),
        "duplicate_backtests": ports.backtester.duplicates,
        "strict_family_size": report["strict"]["family_size"],
        "findings_passed_strict": report["strict"]["findings_passed_strict"],
        "findings_passed_attempted": report["strict"]["findings_passed_attempted"],
        # Per finding: the official value and the two sensitivity checks. Member 5's
        # uniform rule needs only Finding.p_value_raw and the plan's budget.
        "strict_correction": [
            {k: c.get(k) for k in ("finding_id", "signal_id", "p_value_corrected",
                                   "passed_correction", "p_value_corrected_attempted",
                                   "passed_correction_attempted",
                                   "p_value_corrected_strict", "passed_correction_strict")}
            for c in report["claims"] if c["returned"]],
        "token_breakdown": tokens,
        "steps": totals["steps"],
        "budget": budget,
        "correction": report["correction"],  # "holm" (tests run) or "none" (R3-U)
        "tests_in_family": report["tests_in_family"],
        "seed": seed,
        "universe_id": view.universe_id,
        "research_start": str(session.research_start),
        "research_end": None if ports.research_end is None else str(ports.research_end),
        "warnings": list(ports.warnings),
        "mask_dates_in_text": mask,
        "retriever": ports.retriever_kind,
        "signal_specs": {sid: spec.model_dump(mode="json")
                         for sid, spec in ports.backtester.specs.items()},
        "harness": report,
        "agent_totals": totals,
    }
    return metadata


class AgentResearchSystem:
    """One agent system (R2, R3 or R3-U) for the experiment runner."""

    def __init__(
        self,
        system: str,
        config: dict | None = None,
        holding_days: int = DEFAULT_HOLDING_DAYS,
        min_events: int | None = None,
        client_factory: ClientFactory | None = None,
        universe_types: Mapping[str, str] | None = None,
        rag_factory: RagFactory = shared_index_retriever,
    ):
        if system not in ("R2", "R3", "R3-U"):
            raise ValueError(f"system must be R2, R3 or R3-U, got {system!r}")
        self.system = system
        self.name = system.lower()
        self.config = config if config is not None else load_config()
        self.holding_days = holding_days
        self.min_events = min_events if min_events is not None else self.config["min_events"]
        self.client_factory = client_factory  # None: default_client, looked up at run time
        self._universe_types = dict(universe_types) if universe_types is not None else None
        self.rag_factory = rag_factory

    def _check_research_end(self, view) -> None:
        check_research_end(view, self._universe_types)

    def run(self, view, budget: int, seed: int) -> SystemRun:
        started = time.perf_counter()
        self._check_research_end(view)
        agent = self.config["agent"]
        mask = bool(agent.get("mask_dates_in_text", False))
        ports = build_ports(view, budget=budget, holding_days=self.holding_days,
                            min_events=self.min_events,
                            with_retriever=self.system in ("R3", "R3-U"), mask_dates=mask,
                            rag_factory=self.rag_factory)
        config = {**self.config, "min_events": self.min_events}
        session = new_session(
            self.system, config, view.universe_id, ports.backtester, ports.events,
            retriever=ports.retriever,
            research_start=ports.research_start or self.config.get("research_start"),
            research_end=ports.research_end, open_end=True,
        )
        client = (self.client_factory or default_client)(agent, seed)
        run_session(session, client, seed=seed, warnings=ports.warnings)

        findings, report = claims_to_findings(
            session.claims, ports.backtester.attempts, session_id=session.session_id,
            system=self.system, alpha=float(self.config.get("alpha", 0.05)),
            correct=self.system != "R3-U", aliases=ports.backtester.aliases,
            specs={sid: s.model_dump(mode="json") for sid, s in ports.backtester.specs.items()},
            viewed_groups=ports.backtester.viewed_groups,
        )
        metadata = run_metadata(session, ports, report, client, agent, budget, seed, view,
                                started, mask)
        return SystemRun(findings=findings, session_id=session.session_id, metadata=metadata)


class R2System(AgentResearchSystem):
    def __init__(self, **kwargs: Any):
        super().__init__("R2", **kwargs)


class R3System(AgentResearchSystem):
    def __init__(self, **kwargs: Any):
        super().__init__("R3", **kwargs)


# R3-U uses R3's exact prompt and the plan's budget on purpose; only the harness differs:
# no multiple-testing correction.
class R3USystem(AgentResearchSystem):
    def __init__(self, **kwargs: Any):
        super().__init__("R3-U", **kwargs)


# Runner names -> factories. agent.systems registers these (the only registration path).
AGENT_FACTORIES: dict[str, type[AgentResearchSystem]] = {
    "r2": R2System, "r3": R3System, "r3-u": R3USystem,
}
