"""Run one research session on the fakes.

    python -m agent.run_session --system R2 --budget 15 --seed 0 --dry-run
    python -m agent.run_session --system R2 --budget 15 --seed 0
    python -m agent.run_session --system R2 --fake-llm script.json --no-cache

--fake-llm takes a JSON list of LLMResponse objects, e.g.
    [{"tool_calls": [{"id": "t1", "name": "finish", "arguments": {"reason": "done"}}],
      "stop_reason": "tool_use"}]
"""

import argparse
import json
import sys
from pathlib import Path

from agent.config import load_config
from agent.fakes import (
    FAKE_RESEARCH_END,
    best_planted_overlap,
    make_fake_ports,
    register_fake_tools,
)
from agent.llm import AnthropicClient, CachedClient, FakeLLM, LLMClient, LLMResponse
from agent.loop import run_session
from agent.prompts import SYSTEM_PROMPT, build_task_brief
from agent.session import ResearchSession, new_session
from agent.tools import RAG_SYSTEMS, schemas_for


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m agent.run_session", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=["R2", "R3", "R3-U"], required=True)
    p.add_argument("--budget", type=int, help="backtests allowed (default: backtest_budget)")
    p.add_argument("--seed", type=int, default=0, help="fake universe seed")
    p.add_argument("--fake-llm", type=Path, help="scripted responses (JSON); no API call")
    p.add_argument("--no-cache", action="store_true", help="do not use the LLM response cache")
    p.add_argument("--dry-run", action="store_true",
                   help="print the system prompt, task brief and tool schemas, then exit")
    p.add_argument("--model", help="model for this run only (overrides AGENT_MODEL)")
    p.add_argument("--config", type=Path, help="config file (default: configs/default.yaml)")
    return p.parse_args(argv)


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # agent extra
    except ImportError:
        return
    load_dotenv()


def build_client(args: argparse.Namespace, agent: dict) -> LLMClient:
    if args.fake_llm:
        script = json.loads(args.fake_llm.read_text(encoding="utf-8"))
        return FakeLLM([LLMResponse.model_validate(r) for r in script])
    client: LLMClient = AnthropicClient(
        agent["model"], agent["max_tokens"], agent.get("effort"), agent.get("thinking_display")
    )
    return client if args.no_cache else CachedClient(client, agent["cache_dir"])


def print_dry_run(session: ResearchSession) -> None:
    print("=== SYSTEM PROMPT ===", SYSTEM_PROMPT, sep="\n")
    print("=== TASK BRIEF ===", build_task_brief(session), "", sep="\n")
    print("=== TOOL SCHEMAS ===", json.dumps(schemas_for(session.system), indent=2), sep="\n")


def print_report(session: ResearchSession, client: LLMClient, jaccard_min: float) -> None:
    end = session.trace.events[-1]
    t = end["totals"]
    print(f"end reason:    {end['reason']}")
    print(f"steps:         {t['steps']}   tool calls: {t['tool_calls']}   "
          f"invalid calls: {t['invalid_calls']}   nudges: {t['nudges']}")
    print(f"input tokens:  {t['input_tokens']} uncached, {t['cache_read_input_tokens']} cache "
          f"read, {t['cache_creation_input_tokens']} cache write;  output: {t['output_tokens']}")
    if isinstance(client, CachedClient):
        print(f"llm cache:     {client.hits} hits, {client.misses} misses")
    print(f"claims:        {len(session.claims)}")
    for c in session.claims:
        print(f"  - {c.signal_id} (step {c.step}): {c.rationale}")
    print(f"trace:         {session.trace.path or '(in memory)'}")

    best, signal_id = best_planted_overlap(session.backtester, session.claims)
    verdict = "PASS" if best >= jaccard_min else "FAIL"
    print(f"smoke test:    {verdict}  best Jaccard vs planted = {best:.2f}"
          f"{f' ({signal_id})' if signal_id else ''}, threshold {jaccard_min}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv_if_available()
    config = load_config(args.config)
    agent = config["agent"]
    if args.model:
        agent["model"] = args.model

    register_fake_tools()
    budget = args.budget if args.budget is not None else config.get("backtest_budget", 30)
    backtester, events, retriever, universe_id = make_fake_ports(args.seed, budget)
    session = new_session(
        args.system, config, universe_id, backtester, events,
        retriever=retriever if args.system in RAG_SYSTEMS else None,
        research_end=FAKE_RESEARCH_END,  # TODO(member-3): drop once the team agrees the split
        in_memory=args.dry_run,
    )
    if args.dry_run:
        print_dry_run(session)
        return 0

    client = build_client(args, agent)
    run_session(session, client)
    print_report(session, client, config.get("jaccard_min", 0.6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
