"""R1: the single-shot LLM baseline (registered as "r1" by agent.systems).

One LLM call, no tools and no feedback. The model gets the agents' system prompt
(research protocol) and an R2-style brief (structured fields only, no text features) that
asks for a JSON list of up to <budget> signals, each with a hypothesis, a rationale and a
spec. The interim harness (agent.interim) then backtests every valid spec in order, each
counting toward the budget; invalid specs are skipped and counted. Malformed JSON gets one
retry with the error; a second failure gives zero findings.

Findings follow R0: every tested signal is returned with its passed_correction flag (Holm
over every backtest run, as for R2 and R3), and untestable signals are reported in the
metadata only. The client, response cache, pricing, trace format and metadata are the
agent systems' (agent.system).
"""

import json
import re
import time
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from agent import system as agent_system
from agent.config import load_config
from agent.finding import FindingClaim
from agent.interim.harness import claims_to_findings
from agent.interim.ports import build_ports
from agent.llm import LLMResponse, assistant_message
from agent.prompts import SYSTEM_PROMPT, build_task_brief, prompt_version
from agent.session import new_session
from agent.tools import ToolError
from evaluation.systems import SystemRun
from universes.planted import DEFAULT_HOLDING_DAYS

SYSTEM = "R1"
MAX_ATTEMPTS = 2  # the first call plus one retry for malformed JSON
RETRY = "Your reply was not valid JSON ({error}). Reply again with the JSON list only."
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class R1Error(ValueError):
    """A reply that is not a JSON list."""


def parse_reply(text: str) -> list:
    """The reply's JSON list; code fences and text around the list are tolerated."""
    text = _FENCE.sub("", text.strip())
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise R1Error("no JSON list found")
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise R1Error(f"invalid JSON: {e.msg} at position {e.pos}") from e
    if not isinstance(value, list):
        raise R1Error("the reply is not a list")
    return value


def check_item(item: Any, signal_spec: type) -> dict[str, Any]:
    """{hypothesis, rationale, spec} with a valid structured spec; ValueError if not."""
    if not isinstance(item, Mapping):
        raise ValueError("not an object")
    hypothesis, rationale = item.get("hypothesis"), item.get("rationale")
    if not (isinstance(hypothesis, str) and hypothesis.strip()):
        raise ValueError("missing hypothesis")
    if not (isinstance(rationale, str) and rationale.strip()):
        raise ValueError("missing rationale")
    spec = item.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError("missing spec")
    if spec.get("text_filter") is not None:
        raise ValueError("text filters are not available to R1")
    try:
        valid = signal_spec.model_validate(dict(spec))
    except ValidationError as e:
        raise ValueError(f"invalid spec ({e.error_count()} errors)") from e
    return {"hypothesis": hypothesis.strip(), "rationale": rationale.strip(),
            "spec": valid.model_dump(mode="json", exclude_none=True)}


class R1System:
    """Single-shot baseline for the experiment runner (ResearchSystem)."""

    name = "r1"

    def __init__(
        self,
        config: dict | None = None,
        holding_days: int = DEFAULT_HOLDING_DAYS,
        min_events: int | None = None,
        client_factory: agent_system.ClientFactory | None = None,
        universe_types: Mapping[str, str] | None = None,
    ):
        self.config = config if config is not None else load_config()
        self.holding_days = holding_days
        self.min_events = min_events if min_events is not None else self.config["min_events"]
        self.client_factory = client_factory  # None: agent.system.default_client
        self._universe_types = dict(universe_types) if universe_types is not None else None

    def _ask(self, session, client, message: str) -> tuple[list | None, int, str]:
        """(items or None, retries used, end reason)."""
        messages = [{"role": "user", "text": message}]
        for attempt in range(MAX_ATTEMPTS):
            session.step += 1
            response: LLMResponse = client.complete(SYSTEM_PROMPT, messages, [])
            session.trace.model_turn(session.step, response)
            if response.stop_reason == "refusal":
                return None, attempt, "refusal"
            try:
                return parse_reply(response.text), attempt, "finished"
            except R1Error as e:
                session.trace.log("r1_parse_error", step=session.step, error=str(e))
                messages += [assistant_message(response),
                             {"role": "user", "text": RETRY.format(error=e)}]
        return None, MAX_ATTEMPTS - 1, "parse_failed"

    def run(self, view, budget: int, seed: int) -> SystemRun:
        started = time.perf_counter()
        agent_system.check_research_end(view, self._universe_types)
        agent = self.config["agent"]
        mask = bool(agent.get("mask_dates_in_text", False))
        ports = build_ports(view, budget=budget, holding_days=self.holding_days,
                            min_events=self.min_events, with_retriever=False, mask_dates=mask)
        config = {**self.config, "min_events": self.min_events}
        session = new_session(
            SYSTEM, config, view.universe_id, ports.backtester, ports.events,
            research_start=ports.research_start or self.config.get("research_start"),
            research_end=ports.research_end, open_end=True,
        )
        brief = build_task_brief(session, single_shot=True)
        session.log_start(SYSTEM_PROMPT, prompt_version(SYSTEM_PROMPT, brief, []), seed=seed,
                          warnings=ports.warnings)
        client = (self.client_factory or agent_system.default_client)(agent, seed)
        items, retries, reason = self._ask(session, client, brief)

        bt = ports.backtester
        proposed, invalid, over_budget = len(items or []), 0, 0
        claims = []
        for index, item in enumerate(items or []):
            try:
                checked = check_item(item, ports.models.signal_spec)
            except ValueError as e:
                invalid += 1
                session.trace.log("r1_spec", index=index, valid=False, error=str(e))
                continue
            signal_id = bt.define(checked["spec"])
            try:
                result = bt.run(signal_id)
            except ToolError as e:  # budget used up
                over_budget += 1
                session.trace.log("r1_spec", index=index, valid=True, signal_id=signal_id,
                                  error=str(e))
                continue
            session.trace.log("r1_spec", index=index, valid=True, signal_id=signal_id,
                              status=result["status"], duplicate_of=result["duplicate_of"],
                              hypothesis=checked["hypothesis"], event_ids=result["event_ids"])
            if result["duplicate_of"] is None:  # one claim per underlying test
                claims.append(FindingClaim(
                    signal_id=signal_id, step=session.step, evidence=[],
                    rationale=f"{checked['hypothesis']}. {checked['rationale']}"))
        session.trace.session_end(reason)

        findings, report = claims_to_findings(
            claims, bt.attempts, session_id=session.session_id, system=SYSTEM,
            alpha=float(self.config.get("alpha", 0.05)), aliases=bt.aliases,
            specs={sid: s.model_dump(mode="json") for sid, s in bt.specs.items()},
            viewed_groups=bt.viewed_groups,
        )
        metadata = agent_system.run_metadata(session, ports, report, client, agent, budget,
                                             seed, view, started, mask)
        metadata.update({"specs_proposed": proposed, "specs_invalid": invalid,
                         "specs_over_budget": over_budget, "parse_retries": retries,
                         "invalid_calls": invalid})
        return SystemRun(findings=findings, session_id=session.session_id, metadata=metadata)
