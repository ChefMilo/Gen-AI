# Gen-AI

Agentic event-driven alpha research with planted-signal evaluation. See `docs/design.md` for the architecture and `docs/data_schema.md` for the data (which takes precedence where the two disagree).

## What this is

LLM research agents look for event-driven trading signals in SEC 8-K dividend and buyback announcements. Each system is scored on synthetic universes with planted signals (recall, precision) and on placebo universes with none (false discoveries).

| System | What it is |
|---|---|
| R0 | Fixed grid of structured hypotheses, Holm-corrected; no LLM |
| R1 | One LLM call proposes signals; the harness tests them; no feedback |
| R2 | Agent with backtest tools, structured fields only |
| R3 | R2 plus retrieval and text features |
| R3-U | R3 with the same prompt and backtest budget, but no multiple-testing correction |

## Setup

Python 3.11 or later.

```
pip install -e ".[dev,agent]"   # add rag / llm for retrieval and extraction (below)
python -m pytest -q
```

The LLM systems read `AGENT_MODEL` (the final run used `claude-sonnet-5-5`) and `ANTHROPIC_API_KEY` from the environment, else from `.env` in the repo root. `.env` is git-ignored: never commit it.

No test calls a live API: `AnthropicClient` refuses to start under pytest unless `AGENT_ALLOW_LIVE_LLM_IN_TESTS=1`, and the agent tests remove credentials.

**Data.** The data is not in git. It lives in the team's shared Drive folder `Gen AI/Data/` (ask a team member for access). Copy it locally, or use Google Drive for desktop, and point `GENAI_DATA_DIR` at it:

```
export GENAI_DATA_DIR="/path/to/Gen AI/Data"        # bash
set "GENAI_DATA_DIR=C:\path\to\Gen AI\Data"          # Windows cmd
$env:GENAI_DATA_DIR = "C:\path\to\Gen AI\Data"       # PowerShell
```

The final plan runs on synthetic data and doesn't need it. Extraction, retrieval and real prices do.

## Layout and owners

Code lives in `src/`, one package per layer, with the owner named in each `__init__.py`.

| Package | Owner |
|---|---|
| `src/ingest`, `src/store` | Justin (Member 1): EDGAR 8-Ks, LLM extraction, event and price stores |
| `src/rag` | Sean (Member 2): chunk index, hybrid retrieval, text features |
| `src/agent`, `src/baselines/r1_single_shot.py` | Terry (Member 3): agent loop, tools, prompts, R1 |
| `src/dsl`, `src/backtest`, `src/harness` | Ryan (Member 4): signal DSL, event study, budget and corrections |
| `src/universes`, `src/evaluation`, `src/baselines/r0.py` | Collin (Member 5): planted and placebo universes, scoring, R0 |

## Retrieval index

`src/rag/` searches the 8-K text. It needs the filings at `$GENAI_DATA_DIR/text/edgar/<TICKER>/<accession_no>/<document>.txt`, next to `index/` and `universe_quality.csv`.

```
pip install -e .[rag]   # sentence-transformers; the embedding model downloads on first use
```

**Use the shared index.** Download `rag_index/` from the shared Drive folder into `results/rag_index/` (about 500 MB, git-ignored). Loading it takes about 30 seconds:

```python
from datetime import date
from rag.build import load_index
from rag.config import load_rag_config
from rag.retriever import Retriever

retriever = Retriever(load_index(load_rag_config()), events=[], cutoff=date(2021, 12, 31))
retriever.search("special one-time cash dividend", {"event_type": "dividend"}, k=3)
```

`events` takes `store.schema.Event` rows and is only needed for `text_feature()`. Nothing filed after `cutoff` is ever returned.

**Or rebuild it** after the filings or the `rag` settings in `configs/default.yaml` change. On a MacBook this takes about 50 minutes. Keep it plugged in with the lid open; `caffeinate -i` stops it sleeping mid-build:

```
caffeinate -i python -m rag.build
```

An index only loads with the embedding model it was built with.

## Event extraction

`src/ingest/` builds `events/events.parquet` on the shared Drive from the candidate filings, with Claude via the Anthropic API. Put `ANTHROPIC_API_KEY` in `.env`; `EXTRACT_MODEL` picks the model (default: `AGENT_MODEL`, else `claude-sonnet-5-5`).

```
pip install -e .[llm]
python -m ingest.run estimate      # filings, tokens and cost; no API calls
python -m ingest.run gold          # the gold filings (data/gold/), scored against the labels
python -m ingest.run submit        # all candidates via the Batch API (half price)
python -m ingest.run status
python -m ingest.run collect       # fetch, retry, derive, save events.parquet
```

`--limit N` runs the first N filings only (saved under `results/extract/`, not to Drive). Replies are cached in `results/extract/`, so reruns are free.


## Final experiment plan

`configs/final.yaml` (`final-v1`) defines the frozen final evaluation. Its planted grid is built in code (`src/evaluation/final_grid.py`) from the config. There are 4 strengths × single/conjunction × structured/text, and each (complexity, feature type) cell has one base planted at all four strengths: 4 bases × 4 strengths = 16 specs. Two strengths of the same base never share a universe. Each planted and placebo universe redraws the synthetic prices' idiosyncratic noise from its own seed (`data.replicates: resample_noise`), so a spec's repeated plants are independent draws. Plan construction raises if a seed or spec (compared by canonical signature) is shared with the dev configs, or if a final base shares more than `dev_guard.max_containment` (5%) of its targets with an agent dev spec, or reaches `jaccard_min` with one (PR #18's `configs/experiment_agent_dev.yaml`; skipped with a warning when that file is absent).

The current bases:

| Complexity | Feature type | Base | Targets |
|---|---|---|---|
| single | structured | `dividend_large`: dividends of large caps | 1041 |
| single | text | `dividend_framing_growth`: dividends framed as `fake_growth` | 788 |
| conjunction | structured | `dividend_raise_small`: dividend raises > 10%, small caps | 131 |
| conjunction | text | `buyback_rationale_undervalued_small`: buybacks framed as `fake_undervalued`, small caps | 77 |

Event type is not balanced: both structured bases are dividends, because no one-feature structured buyback base keeps clear of PR #18's dev specs. `dividend_framing_growth` overlaps the other dividend bases, so it only shares a universe with the buyback base. The planner searches for an assignment that gives every spec the same number of appearances, so each of the 16 specs appears in 2 planted universes. When no such assignment exists, it keeps the greedy one and logs the specs that got fewer.

Everything below runs offline, with no LLM calls:

```bash
python -m evaluation.run --config configs/final.yaml --dry-run --systems r0  # build + check the frozen hash
python scripts/oracle_check.py      # results/final-v1/oracle.csv: is each planted cell findable?
python scripts/estimate_cost.py     # sessions x tokens x price per LLM system
```

`configs/final.plan.sha256` holds the hash of the canonical `plan.json`. The runner warns when a `final-v1` plan differs from it, and `--freeze-plan` re-records it (do that only on purpose).

The final grid uses the fake event data's text labels (`fake_growth`, `fake_undervalued`) and is frozen.

Scoring reports two rules:
- **Uniform (the primary rule):** the `*_uniform` columns. It is Holm over max(`backtest_budget`, the session's number of findings) at `alpha` from `configs/default.yaml`, recomputed by the evaluator from each finding's `p_value_raw` (`evaluation.metrics.recorrect`).
- **Official:** each system's own `passed_correction`. This is what `evaluation.run` prints.

## Running the final evaluation

```
python -m agent.freeze --config configs/final.yaml --check     # every session's prompt is frozen
python -m evaluation.run --config configs/final.yaml            # 5 systems x 27 universes
python -m evaluation.summarize --results results/final-v1       # rewrite summary/ from runs/
python -m agent.pilot_summary results/final-v1                  # LLM systems under every rule
```

- **`evaluation.run`:** writes `results/final-v1/` (`plan.json`, `universes/`, `runs/`, `summary/`). The agents' traces go to `results/traces/`.
- **Resuming:** rerunning resumes. A pair is skipped when its `metadata.json` exists for the same universe, and a failed pair is retried. `--no-resume` reruns finished pairs.
- **`agent.pilot_summary`:** prints, for each system, recall, precision and placebo false discoveries under four rules: official, uniform, attempted (Holm over the tests run) and strict (attempted plus every breakdown group viewed). It also prints tokens, cost and end reasons.

**Reproducibility.**
- **Plan:** canonical hash `0f12f4a7d484f2018518bf0474993daacea96db8f62375936c9141cf3f86ee69` (`configs/final.plan.sha256`). The runner warns when a plan differs.
- **Code and prompts:** tag `prompt-freeze-v1` (`b725f95`). The final-v1 run used `1993258`, the merge of PR #29, whose tree is identical. The LLM systems refuse a prompt version that isn't in `agent.frozen_prompt_versions` (`configs/default.yaml`).
- **Seeds:** every universe's seed is in `plan.json`, and each system × universe pair's seed is derived from `base_seed`, the system name and the universe id.
- **Response cache:** LLM replies are cached in `results/llm_cache/`. The key is the full request, the model settings and the pair's seed.
  - The cache is shared on the team Drive alongside the results.
  - Rerunning with it replays every call exactly, at no API cost.
  - A replay still needs the `agent` extra, `AGENT_MODEL=claude-sonnet-5-5` and an API key set, because the client is built before the cache is checked.

## Summaries and figures

A run writes `results/<id>/summary/`: the result tables as CSV, plus `index.json` (plan hash, systems, universe counts, environment). A summary always covers every system with runs on disk for that plan, so a run with `--systems` never leaves a partial one. Read its CSVs with `evaluation.summary.read_summary`: pandas' defaults read the system name `null` as missing.

Both commands below run offline and run no system:

```bash
python -m evaluation.summarize --results results/<id>     # re-score the runs on disk, rewrite summary/
python scripts/make_figures.py --results results/<id>     # PNG + SVG in results/<id>/figures/
```

`make_figures.py` reads only `summary/`. Its options:

- `--oracle results/<id>/oracle.csv` adds the oracle's findable rate as a dashed line. `scripts/oracle_check.py` writes that file, and only for plans built from a `grid:` block. An `oracle.csv` from another plan is left out, with a note on the figure.
- `--rule uniform` draws the figures under the uniform rule, as `<name>_uniform`. The default is `--rule own`.
- `--input-price` and `--output-price` (USD per million tokens) estimate the cost of systems with no `cost_usd`, and the axis says so.

| Figure | Shows |
|---|---|
| `detection_curve` | Recall vs strength per system; placebo false discoveries per test |
| `detection_by_cell` | Recall vs strength, by complexity × feature type |
| `recall_by_snr` | Recall per system across SNR bins |
| `r2_vs_r3_text` | R2 vs R3 on text truths: truth-level paired difference, CI and exact McNemar p, with the universe-level test beside it |
| `r3_vs_r3u` | Recall, precision and placebo false discoveries per test, with paired differences |
| `cost_vs_recall` | Recall vs mean cost per session |

A system with no data is skipped, with a note on the figure and a log line.

On the final plan each planted universe holds one feature type and each spec appears in 2 universes. A feature type's comparison therefore rests on 8 universes, and each point in `detection_by_cell` on 2 truths.

## Dev pilots

Agent pilots run on dev universes (dev plant specs and seeds, kept apart from the final plan):

```
python -m agent.experiment --config configs/experiment_agent_dev.yaml --systems r1,r2,r3,r3-u --budget 30 --out results/<id>
python -m agent.pilot_summary results/<id>
```

- **`--systems`:** comma-separated.
- **`--budget`:** overrides the config's backtest budget.
- **Traces:** go to `results/<id>/traces/`.
- **`--allow-unfrozen`:** lets a pilot try a prompt that isn't in `agent.frozen_prompt_versions`. It is for development only. `agent.experiment` is the only entry point that accepts it, and the final run (`evaluation.run`) refuses an unfrozen prompt.
