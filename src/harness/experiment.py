"""ResearchSystem adapter for Member 5's runner, using the real Person 4 ports."""

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from agent.loop import run_session
from harness.agent_adapter import finalize_agent_session, register_research_tools
from harness.agent_session import make_research_session
from harness.backtester import HarnessConfig, ResearchBacktester
from harness.limits import ModelLimits, ResourceLimitedClient


class AgentResearchSystem:
    """Inject client/retriever factories; never pass planted truths to either.

    This adapter is programmatic: it does not register fake clients as production
    R3 defaults or change the evaluation CLI's existing system registry.
    """

    def __init__(
        self,
        name: str,
        client_factory: Callable,
        output_dir: Path,
        retriever_factory: Callable | None = None,
        portfolio=None,
        model_limits: ModelLimits | None = None,
        agent_config: dict | None = None,
        min_events: int = 20,
        alpha: float = 0.05,
        correction_family: str = "budget",
        client_kind: str = "unspecified",
    ):
        if name not in {"R2", "R3", "R3-U"}:
            raise ValueError("AgentResearchSystem supports R2, R3, R3-U")
        if (name != "R2") != (retriever_factory is not None):
            raise ValueError("R3/R3-U require a retriever factory; R2 does not use one")
        self.name, self.client_factory = name, client_factory
        self.output_dir, self.retriever_factory = Path(output_dir), retriever_factory
        self.portfolio, self.model_limits = portfolio, model_limits or ModelLimits()
        self.agent_config = agent_config or {}
        self.min_events, self.alpha = min_events, alpha
        self.correction_family, self.client_kind = correction_family, client_kind

    def run(self, view, budget: int, seed: int):
        from evaluation.systems import SystemRun, make_view

        started = time.perf_counter()

        if view.research_start is None or view.research_end is None:
            raise ValueError(
                "Agree explicit research_start/research_end before an agent experiment"
            )
        folder = self.output_dir / self.name / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        config = HarnessConfig(
            research_start=view.research_start,
            research_end=view.research_end,
            budget=budget,
            min_events=self.min_events,
            alpha=self.alpha,
            correction_family=self.correction_family,
            mode="unguarded" if self.name == "R3-U" else "guarded",
            portfolio=self.portfolio,
        )
        backtester = ResearchBacktester(
            view.prices, view.benchmarks, view.events, config, folder / "audit.jsonl"
        )
        # Factories receive the same visible view/seed; no universe record or truth.
        import pandas as pd

        safe_view = make_view(
            view.universe_id,
            view.prices.loc[pd.to_datetime(view.prices["date"]).dt.date.le(config.research_end)],
            view.benchmarks.loc[
                pd.to_datetime(view.benchmarks["date"]).dt.date.le(config.research_end)
            ],
            backtester.research_events(),
            config.research_start,
            config.research_end,
        )
        client = self.client_factory(safe_view, seed)
        if not isinstance(client, ResourceLimitedClient):
            raise TypeError("client_factory must return ResourceLimitedClient with explicit limits")
        if client.limits != self.model_limits:
            raise ValueError("Client and agent model resource settings must match")
        if client.calls:
            raise ValueError("Each research session requires a fresh client budget")
        backend = self.retriever_factory(safe_view, config) if self.retriever_factory else None
        settings = {
            **self.agent_config,
            "model": client.model,
            "max_tokens": self.model_limits.response_output_limit,
            "max_steps": self.model_limits.max_calls,
            "max_session_input_tokens": self.model_limits.max_input_tokens,
        }
        session = make_research_session(
            backtester,
            system=self.name,
            universe_id=view.universe_id,
            agent_config=settings,
            retrieval_backend=backend,
            trace_dir=folder / "traces",
        )
        client.audit = backtester._record
        register_research_tools()
        try:
            run_session(session, client)
            findings = finalize_agent_session(session)
        except Exception as error:
            (folder / "error.json").write_text(
                json.dumps(
                    {"error": str(error), "universe_id": view.universe_id, **client.metadata()},
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            raise
        metadata = {
            "backtests_used": backtester.budget_status()["tests_used"],
            "tool_calls": session.trace.totals()["tool_calls"],
            "tokens": client.input_tokens + client.output_tokens,
            "cost_usd": client.estimated_cost_usd
            if client.rates
            else (0.0 if self.client_kind == "scripted" else None),
            "wall_time_s": time.perf_counter() - started,
            "trace_path": str(session.trace.path),
            "audit_path": str(folder / "audit.jsonl"),
            "client_kind": self.client_kind,
            "model_limits": client.limits.model_dump(mode="json"),
            "token_rates": client.rates.model_dump(mode="json") if client.rates else None,
            "seed": seed,
            **client.metadata(),
            "portfolio_reports": backtester.portfolio_reports(),
        }
        result = SystemRun(findings=findings, session_id=session.session_id, metadata=metadata)
        (folder / "run.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result
