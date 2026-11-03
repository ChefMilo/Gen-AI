# Person 4: backtesting and guardrails

## Your role in plain language

You own the calculator and the referee. The research agent proposes an idea;
your engine measures its historical result; your harness decides what it may
access, how many tests it may run, and whether its final claim passes correction.
The evaluation layer separately checks findings against hidden planted truths.

The local implementation is usable without Git commands or a paid model. A
scripted comparison verifies the complete pipeline. It is not a substitute for
the team's real-model experiments or an agreed real-data holdout period.

## What is implemented

| Responsibility | Implementation |
|---|---|
| Validated structured signals | `dsl.condition`, `dsl.signal`; bounded operators/fields, numeric/category/bool conditions, text filters, holding period and direction |
| Eligible-event selection | `backtest.selection`; new passed announcements only, missing values excluded, duplicate IDs rejected |
| Event study | `backtest.engine`; raw additive stock-minus-benchmark CAR, sample t-test, hit rate, measurable IDs and skip reasons |
| Portfolio strategy view | `backtest.portfolio`; daily equity/cash, costs, lot sizing, long/short exposure, turnover, Sharpe and drawdown |
| Time lockout | `harness.backtester`, `data_access`; shared dates across backtests, queries and retrieval, acceptance-time checks |
| Testing budget | Reserved before computation; failed executions count; invalid definitions and unknown IDs do not execute tests |
| Multiple-testing correction | Session-wide Holm; configurable attempted-family or full-budget family; unreported attempts included |
| Final portfolio search diagnostic | Deflated Sharpe computed from all available trial Sharpes at finalization, including unreported trials |
| Minimum events | At least 20 measurable events by default; insufficient/nonconstant statistics cannot become accepted findings |
| Audit and source checks | JSONL requests/results/rejections/finalization; supplied quotes checked against approved event sources or passages served in-session |
| Agent integration | Six real tools, matching task brief, recoverable guardrail errors, finalization outside the model's tool list |
| Step/token/cost controls | Agent loop caps plus `ResourceLimitedClient` preflight call/token/cost reservations; explicit prices required for money limits |
| Guardrail ablation | R3-U disables budget and Holm only; time/source/minimum-event/resource controls stay enabled |
| One-time holdout evaluation | `harness.holdout`; freezes signals before data loading, preserves failed reservations, refuses reuse of a named ledger |
| Evaluation integration | `harness.experiment.AgentResearchSystem` implements the runner interface using injected model/retrieval factories |

## Event study versus portfolio returns

These deliberately answer different questions.

**Event study:** sum `stock.ret - benchmark.ret` on event day 0 through day h-1.
This preserves the project's reference CAR definition and planted-effect scoring.
The mean effect keeps the stock-return sign even when the proposed trade is short.
The t-test is two-sided. Bad prices, missing benchmark observations, incomplete
windows, missing trading days and nonfinite returns are not treated as zero.

**Portfolio:** enter at the close of the first allowed event date, earn returns on
the next h trading days, and exit at close. Thus an intraday announcement never
earns the pre-announcement part of the event-day return. An exit beyond the known
research cutoff is not opened. A missing held-position return raises an error;
the engine does not select only trades whose future price history looks clean.

The portfolio is an illustrative daily-rebalanced TOTAL-RETURN NOTIONAL model.
It marks signed exposures using adjusted total returns (including reinvestment),
then rounds each new target using raw-close share lots. It is not a broker ledger
of individual dividend cash flows, tax lots, settlement, margin or stock lending.
Do not add dividend cash again on top of adjusted returns.

At each close, active tickers share gross exposure of at most one times equity.
Overlapping events in the same ticker merge into one exposure; they do not double
leverage. Fee/slippage cash reserves include closing old and opening new positions.
Cash earns zero, shorts retain collateral without interest, and the configurable
borrow fee is charged on prior short notional. Short-price increases cause losses.
There is no liquidity/market-impact model beyond the configured slippage rate.

Default demonstration assumptions are USD 1m capital, 5 bps fees plus 5 bps
slippage on EACH traded dollar, one-share lots, zero risk-free rate and zero
borrow cost. These are example parameters, not empirically calibrated estimates.
Change them and report sensitivity before interpreting real performance.

Sharpe uses the sample standard deviation of daily excess returns, annualized
with sqrt(252), including cash days. Drawdown is a negative peak-to-trough value
(e.g. -0.10 means a 10% drawdown). Zero-variance histories have no finite Sharpe.

Deflated Sharpe uses daily (not annualized) Sharpe, skewness, kurtosis, sample
length and search dispersion. Nominal trial counts are reported explicitly;
correlated hypotheses and serial dependence are NOT fully corrected. If there
is insufficient history or search dispersion, it returns an unavailable reason.
DSR is a reported diagnostic, not an additional hidden discovery threshold;
Holm-corrected event-study significance remains the evaluator's discovery gate.
Reference: [Bailey and Lopez de Prado, The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

## Run the reproducible offline comparison

In an activated virtual environment, from the repository root:

```powershell
python -m harness.offline --config configs/person4_offline.yaml --out-dir results/person4-offline-new-run
```

Choose a NEW directory: existing outputs are never silently overwritten. This
command makes no Git or paid API calls. It uses 10 planted and 10 placebo
universes by default, with both arms receiving identical universe views, seeds
and scripted requests. R3 permits three backtests; R3-U attempts the whole
eight-signal stress grid. Near duplicates intentionally stress the corrections.
The synthetic fixture has clean price flags by construction; invalid-price
behavior is tested independently. Dates in this config are demonstration dates,
not the project's real research/holdout decision.

Outputs include `REPORT.md`, configuration, source hashes, dataset design,
per-universe results, pooled summaries, paired comparisons and per-session
findings/traces/audits. `run.json` includes daily portfolio paths/trades and
final portfolio/DSR diagnostics for all completed signals. Bootstrap intervals
in this demonstration describe scripted behavior, not LLM discovery quality.

The run under `results/person4-offline-20261101-verified` is the checked local
example. An earlier `person4-offline-20261101` directory is an incomplete
debugging run, retained separately; it is not the final report.

## Use a real research system later

`AgentResearchSystem` accepts two factories: a fresh bounded model client and a
retrieval backend. They receive a visible universe view and seed, never planted
truths or universe-type labels. Both factories receive cutoff-restricted data.
Use the same client configuration and seed policy in paired R3/R3-U runs.
The existing evaluation CLI's default registry is unchanged: use this adapter
programmatically with Member 5's runner or add an explicitly configured factory.
The existing runner invokes Git to check ignored output; this task did not run
it or silently disable that protection.

For a live Anthropic client, `harness.live.make_limited_anthropic_client` requires
an explicit model, response limit, cost rates and estimated USD budget. It
disables SDK retries. The key stays in the normal environment variable and is
never written into configuration, traces or reports. No live calls were made
for the offline deliverable.

The limiter reserves estimated input plus maximum output before each provider
call. Failed calls keep their reservation and stop further calls. Reported
cached/uncached input and output usage are priced separately. These are rate-
based estimates, not authoritative billing. Token counts can differ from actual
usage, so provider/account spending limits are needed for a hard money cap.
See [provider token-counting documentation](https://platform.claude.com/docs/en/build-with-claude/token-counting).

## Holdout procedure: do not guess the split

The shared config deliberately still has `research_end: null` and
`holdout_start: null`. The team must decide those dates and identify the shared
event, price and filing/index files before the real study.

After research, freeze the significant findings. Construct `HoldoutConfig`
with the agreed nonoverlapping dates, choose a canonical evaluation ledger path,
and call `evaluate_holdout_once(findings, config, ledger_path, loader)`. The
loader receives holdout start/end and returns `(prices, benchmarks, events)`.
The ledger records the frozen definition hash BEFORE invoking that loader.
Failed attempts remain recorded, and a second call with the same ledger fails.
The team controls the canonical ledger; arbitrary Python/filesystem access can
create another path, so this is an application rule rather than OS isolation.
Finding objects do not carry their original research dates: the controller must
check this configuration against the original audited research run.

Holdout results report raw effect retention, sign agreement, measurable sample
size and correction across the fixed holdout family. No agent gets holdout data,
and no new hypothesis is selected using those results.

## What the tests prove, and what they do not

Checks cover known return arithmetic, costs on both trade sides, a full-rotation
cash edge case, share-lot rounding, short losses/borrow charges, cutoff exits,
missing held prices, overlapping exposures, Sharpe/drawdown, nominal-trial DSR,
preflight resource rejection, failed-call reservations, source quote rejection,
retrieval/query date isolation and complete scripted sessions. Git subprocesses
are blocked during the focused verification.

Embedded announcement instructions and scripted attempts to change the budget
test the tool boundary. They do not prove that a live LLM resists every prompt
injection or interprets evidence faithfully. Source-substring checks verify
presence, not economic reasoning. Empty evidence is still permitted.

Event t-tests assume independent observations; overlapping/correlated events
and adaptive hypothesis selection can invalidate nominal p-value calibration.
Holm and budgets reduce some snooping risks but do not erase them. Real-data
coverage, survivorship bias and extraction quality remain report limitations.

## Verified local result (1 November 2026)

The final focused verification passed **133 tests**, with no failures, errors or
skips. Lint/format checks passed. Git subprocesses were blocked and none were
attempted. This is a focused result, not a rerun of every legacy repository test.

The checked offline output contains 20 universes (10 planted, 10 placebo), 40
paired sessions and zero paid API calls:

| Scripted arm | Planted recall | Precision | Mean placebo false discoveries |
|---|---:|---:|---:|
| R3 | 0.400 | 0.400 | 0.000 |
| R3-U | 0.600 | 0.250 | 0.100 |

In this predefined stress script, the unguarded arm tested more ideas and found
more planted patterns, but had lower precision and one significant placebo claim
across ten placebo universes. This is only pipeline evidence; the tiny synthetic
script does not establish the behavior of a real LLM agent.

## Team handoff and remaining empirical work

The code deliverable and offline verification can be completed without spending
API credits. The live-model result is a separate, pending experiment:

1. Agree research/holdout dates and the actual dataset/index locations.
2. Review interface defaults with Members 1, 2, 3 and 5.
3. Approve a small model/provider/token/USD budget; start with one pilot session.
4. If the pilot is correct, run matched R3/R3-U universes under the same model
   and resource settings. Preserve all failures and attempted hypotheses.
5. Evaluate recall/false discoveries with Member 5, then test frozen discoveries
   once on the real holdout. Report null/negative results too.

No Git commands or commits were run during this implementation. The full legacy
suite includes Git-dependent tests; the focused no-Git verification is the
current evidence. The user's earlier 476-pass full-suite result predates these
last additions and must not be presented as a new full-suite run.

## A short explanation for your presentation

"I built the event-study and portfolio calculators, then placed a harness around
the agent's tools. It fixes the dates, limits backtests and model resources,
records every attempt, checks supplied sources and applies correction at the
end. I verified it with known-answer tests and matched offline stress runs.
The portfolio includes fees, slippage, lot rounding, short costs, Sharpe and
drawdown. The team's real-model comparison remains pending the data split and
approved API budget; I am not presenting scripted results as LLM performance."
