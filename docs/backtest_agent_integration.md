# Backtest/agent integration

The real agent loop can now use Member 4's event-study engine and guardrail
harness through `harness.agent_adapter`. No LLM provider is required for the
scripted integration tests.

## How the pieces fit

1. The session controller constructs `ResearchBacktester` with a fixed
   `HarnessConfig`, data frames, and a new audit file.
2. It calls `harness.agent_session.make_research_session` with that backtester.
   The factory builds matching research-period event-query and retrieval ports
   without executing Git provenance commands.
3. After any temporary tool registration, it calls `register_research_tools()`.
   This registers all six real-port tools. The smaller `register_backtest_tools()`
   remains available for backtest-only integration.
4. `agent.loop.run_session` drives the model/tool loop. The adapter records
   complete measurable event IDs for scoring but shows the model a compact
   statistical summary.
5. After a successful loop termination (including resource caps), the controller
   calls `finalize_agent_session(session)`. The harness computes corrected
   findings from all attempted tests and closes the session. Internal-error
   termination cannot be finalized as a successful research session.

Intentional `GuardrailError` rejections become recoverable agent `ToolError`
responses. Unexpected implementation errors propagate instead of being hidden.
The input schema comes from the real `SignalSpec`, not `FakeSignalSpec`.
R2 cannot define text-filter signals.

## Verified without provider calls

The scripted integration fixture has 20 dividend events with known mean
five-day abnormal return of 2%. It exercises hypothesis recording, signal
definition, the real return calculator, a rejected second backtest when the
budget is one, descriptive breakdown, a claim, and harness finalization.
Additional tests cover invalid input, R2 text-filter rejection, unexpected
errors, premature finalization, step/input-token caps, and compact tool outputs.

The fixture deliberately uses a minimal controlled event port and empty evidence.
It proves backtest/control integration, not event-query isolation, quote
faithfulness, model discovery skill, or trading profitability.

## Running the focused test

From the repository root, in the virtual environment:

```powershell
python -m pytest tests/test_backtest_agent_integration.py -v --basetemp=C:\GenAI-TestTemp\pytest-real-harness-agent
```

Only use a disposable directory as `--basetemp`: pytest clears it. These tests
disable the agent's Git-provenance hook. The existing production session factory
and some other repository tests still invoke Git; do not run those without
authorization when Git commands are prohibited. No changes to Git or commits are
needed to use this adapter.

## Integration boundaries and remaining work

- `agent.run_session` still launches temporary ports. This change does not
  silently replace its CLI or wire in a shared financial dataset.
- `register_backtest_tools` still replaces only backtest tools. The new
  `register_research_tools` and session factory also bind the real data ports;
  see the shared-data-port checks below.
- `new_session` normally records Git provenance. A caller must explicitly arrange
  authorized provenance handling before running a production session.
- Guarded/unguarded descriptions now follow the actual harness settings.
  Empirical R3-U evaluation still needs matched model/dataset runs.
- The tool registry is process-global. Register tools before sequential sessions;
  concurrent experiments need an isolated registry design.
- Shared-port sessions now check supplied quotes against event sources and
  retrieved passages. These checks do not establish semantic faithfulness.
- Step/input-token cap tests establish existing loop behavior. They do not
  establish a monetary cost cap or prevent tokens already spent on a model call.
- The reference event study uses additive adjusted daily stock return minus
  benchmark return, starting on `first_tradeable_date`. Intraday announcements
  can make part of the event-day return untradeable. The portfolio view now models costs and compounded
  NAV with Sharpe/drawdown; dependence-aware inference remains separate work.
  See the Person 4 handoff for accounting assumptions.
- A real paired R3/R3-U study across planted/placebo universes is still needed.
  The unit-tested mode switch does not establish empirical false-discovery rates.


## Shared research data ports

`harness.agent_session.make_research_session` now constructs real structured
event queries and, for R3/R3-U, a `ResearchRetriever` around the supplied retrieval
backend. Both use the backtester's immutable configuration and isolated event
snapshot. The factory does not execute Git provenance commands or model calls.
Use `register_research_tools()` after any temporary registrations to expose all
six real-port tools, then run the agent loop and call `finalize_agent_session`.

```python
from harness.agent_adapter import register_research_tools, finalize_agent_session
from harness.agent_session import make_research_session
from agent.loop import run_session

# backtester: an already-created ResearchBacktester with a new audit file
# backend: e.g. rag.retriever.Retriever, configured with the exact research_end
# client: a scripted client or a separately configured provider client
session = make_research_session(
    backtester, system="R3", universe_id="experiment-universe",
    retrieval_backend=backend,
    agent_config={"max_steps": 60, "max_session_input_tokens": 100000},
)
register_research_tools()
run_session(session, client)
findings = finalize_agent_session(session)
```

Structured queries expose approved fields only and exclude future, pre-research,
flagged and old-announcement events. Text features use this same event snapshot.
Document searches may include pre-research historical documents for context, but
never documents after research_end. The wrapper clamps requests, checks every
returned filing date, and records served passages for source-quote checking.
It relies on the backend's source metadata being accurate.

Supplied event evidence must quote stored source text for an event in the
signal's measurable event set. Supplied document evidence must quote a passage
served in the same session. Empty evidence remains permitted for structured-only
systems; substring checks verify source presence, not whether a quote supports
the economic reasoning. Rejected evidence is recorded in the audit.

R3-U task briefs now correctly describe disabled budget/correction controls.
The step/input-token caps and research cutoff remain enabled. Announcements are
still untrusted data. Tests include embedded instructions and a scripted attempt
to change the budget; this checks the tool boundary, not general LLM resistance
to prompt injection.

The new tests use the actual hybrid-index/retriever implementation with a tiny
deterministic embedder. They do not download models or run a real LLM study.
The Person 4 handoff now describes the experiment adapter, portfolio accounting,
projected model-cost controls, once-per-ledger holdout evaluator and offline
comparison command. Real-dataset configuration and empirical live-model
R3/R3-U evaluation remain pending team decisions and spending approval.
