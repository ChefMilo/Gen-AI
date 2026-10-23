# Skeleton status: setup/skeleton (2026-10-21)

Recon only. Nothing was changed except this file, which is uncommitted.

## Summary

The previous session finished every requested step (a–f). Its work is in commit `5cff9b1`, which is pushed and is part of open PR #1. CI on the PR passed. The only thing left is a stale `eval/` mention in `docs/design.md:282`. That line is documentation and no import uses it.

| Step | Status |
|---|---|
| a. Finding model, `report_finding -> Finding` | DONE |
| b. `eval` → `evaluation` rename | DONE (one stale mention in the docs) |
| c. `configs/default.yaml` | DONE |
| d. Ruff config and CI workflow | DONE |
| e. Test with one valid example of each model | DONE |
| f. Commit, push and PR with a placeholders table | DONE |

## 1. Git state

- Current branch: `setup/skeleton`.
- `git status`: clean. There are no uncommitted or untracked files, apart from ignored caches and build output (`.pytest_cache/`, `.ruff_cache/`, `build/`, `__pycache__/`, `src/gen_ai_event_research.egg-info/`).
- `git log --oneline -15 --all --graph` (after `git fetch --all`):
  ```
  * 5cff9b1 Add Finding placeholder, run config, ruff and CI; rename eval to evaluation
  * 6b3208a Add project skeleton, event store schema and interface placeholders
  * acf5ffd Initial commit
  ```
- `origin/setup/skeleton` exists and points at `5cff9b1`. The local branch is 0 ahead and 0 behind it. `main` and `origin/main` are both at `acf5ffd`.
- PR: `gh pr list --state all` shows **#1 "Project skeleton, event store schema and interface placeholders"**. It is OPEN, from `setup/skeleton` into `main`: https://github.com/ChefMilo/Gen-AI/pull/1. The CI check `test` ran on 2026-10-21 at 06:06Z and its result was **SUCCESS**.

## 2. Step checks

### a. Finding model — DONE

- `src/agent/finding.py` exists. The TODO is at `src/agent/finding.py:28`: `# TODO(member-3): owner confirms. Placeholder agreed with Member 5 (Evaluation)...`
- `class Finding(BaseModel)` is at `src/agent/finding.py:30`. All the expected fields are present under the expected names, and none are missing or renamed:

| Field | Line | Type |
|---|---|---|
| finding_id | 31 | `str` |
| session_id | 32 | `str` |
| system | 33 | `Literal["R0","R1","R2","R3","R3-U"]` (`System`, line 10) |
| signal_id | 34 | `str` |
| signal_spec | 35 | `SignalSpec` |
| event_type | 36 | `EventType` (`"dividend" \| "buyback"`) |
| event_ids | 37 | `list[str]`, required (no default) |
| n_events | 38 | `int` |
| direction | 39 | `Literal["long","short"]` (`Direction`, line 11) |
| effect_car | 40 | `float` |
| t_stat | 41 | `float` |
| p_value_raw | 42 | `float` |
| p_value_corrected | 43 | `float` |
| passed_correction | 44 | `bool` |
| rationale | 45 | `str` |
| evidence | 46 | `list[EvidenceQuote]` |

- The file also adds `EvidenceQuote` (`src/agent/finding.py:14`). Its fields are `event_id | None`, `doc_ref | None` and `quote`, and a validator (lines 21–25) requires either `event_id` or `doc_ref`.
- `report_finding` returns `Finding`: `src/agent/tools.py:38`, `def report_finding(signal_id: str, rationale: str, evidence: list[EvidenceQuote]) -> Finding:`.

### b. Package rename — DONE

- `src/eval/` no longer exists. The commit deleted `src/eval/__init__.py`.
- `src/evaluation/__init__.py` exists, with the docstring "Owner: Member 5 (Evaluation)".
- No `.py`, `.toml` or `.yml` file references `eval` as a module.
- One stale mention: `docs/design.md:282` still shows `└── eval/ # scoring, matching, statistics, RAG eval` in the layout tree. It is documentation only and does not affect code.

### c. configs/default.yaml — DONE

| Key | Expected | Actual (line) |
|---|---|---|
| data_dir | from `GENAI_DATA_DIR` | `${GENAI_DATA_DIR}` (3) |
| research_start | 2016-01-01 | `2016-01-01` (5) |
| research_end | placeholder | `null  # TODO: agree the research/holdout split` (6) |
| holdout_start | placeholder | `null  # TODO: agree the research/holdout split` (7) |
| backtest_budget | 30 | `30` (9) |
| min_events | 20 | `20` (10) |
| jaccard_min | 0.6 | `0.6` (11) |
| alpha | 0.05 | `0.05` (12) |
| seed | 0 | `0` (13) |

PyYAML does not expand `${GENAI_DATA_DIR}`. Whatever code loads this config has to substitute it itself. `src/store/prices.py:17` `data_dir()` already reads the same environment variable.

### d. Ruff and CI — DONE

- `pyproject.toml:19-25`: `[tool.ruff]` sets `line-length = 100`, `target-version = "py311"` and `src = ["src","tests"]`. `[tool.ruff.lint]` sets `select = ["E","F","I","B","UP"]`. `ruff>=0.6` is in the `dev` extras (line 13).
- `.github/workflows/ci.yml`: triggers on `pull_request` (line 4) and uses Python `"3.11"` (line 13). It runs `pip install -e ".[dev]"` (14), `ruff check .` (15) and `pytest -q` (16).

### e. Model example test — DONE

`tests/test_models.py` builds `Event` (lines 14–40), `SignalSpec` (43–44), `BacktestResult` (47–48) and `Finding` (51–72). The test at lines 75–81 round-trips each one through JSON. There are also negative tests:
- `event_ids` is required (lines 84–88).
- `EvidenceQuote` needs an `event_id` or a `doc_ref` (lines 91–93).

### f. Commit, push, PR — DONE

- Commit `5cff9b1` holds this work. It is pushed to `origin/setup/skeleton`.
- PR #1 is open. Its description has a **"Placeholders for owners to confirm or replace"** table:
  - `SignalSpec`: Member 4
  - `BacktestResult`: Member 4
  - the 7 tool signatures: Member 3, agreed with Members 2 and 4
  - `Finding` and `EvidenceQuote`: Member 3
  - the `prices.py` helpers: Member 1
  - `research_end` and `holdout_start`: Team

## 3. Lint and tests

- `ruff check .`: the bare `ruff` command is not on PATH in this shell (`ruff: command not found`). Running it through Python works: `python -m ruff check .` gives `All checks passed!` (ruff 0.16.8).
- `pytest -q`: `15 passed in 6.14s`.
- The local interpreter is Python 3.12.5. CI runs 3.11, and the CI run on PR #1 passed.

## 4. Contract map (Pydantic models in src/)

| File | Class | Owner (from TODO) | Fields |
|---|---|---|---|
| `src/store/schema.py:30` | `Event` | No TODO. The package docstring names Member 1 (Data). Source of truth is `docs/data_schema.md` | event_id, event_type, ticker, cik, company_name, accession_no, form_type, items, source_url, acceptance_datetime, first_tradeable_date, announcement_date?, is_new_announcement, dividend_per_share?, dividend_kind?, record_date?, payable_date?, ex_date?, prev_dividend_per_share?, dividend_change_pct?, buyback_amount_usd?, buyback_shares?, is_new_program?, buyback_expiry?, buyback_pct_mcap?, mcap_usd?, mcap_bucket?, text_features (dict[str, TextFeature]), evidence (dict[str, str]), extraction_status, flags, extracted_at, extractor_version |
| `src/store/schema.py:23` | `TextFeature` | No TODO. Store package (Member 1); its values come from Retrieval (Member 2) | label, quote |
| `src/dsl/signal.py:12` | `SignalSpec` | TODO(member-4) (line 8) | event_type, benchmark (default `SPY`) |
| `src/backtest/result.py:8` | `BacktestResult` | TODO(member-4) (line 6) | signal_id |
| `src/agent/finding.py:30` | `Finding` | TODO(member-3) (line 28), agreed with Member 5 | finding_id, session_id, system, signal_id, signal_spec, event_type, event_ids, n_events, direction, effect_car, t_stat, p_value_raw, p_value_corrected, passed_correction, rationale, evidence |
| `src/agent/finding.py:14` | `EvidenceQuote` | Covered by the TODO(member-3) on `Finding` | event_id?, doc_ref?, quote (needs event_id or doc_ref) |

`?` = optional, defaults to `None`.

Shared types in `src/store/schema.py:15-20`:
- `EventType`: dividend | buyback
- `Benchmark`: SPY | IWM | MDY
- `DEFAULT_BENCHMARK`: SPY
- `McapBucket`
- `DividendKind`
- `ExtractionStatus`

Interfaces that are functions rather than models:
- `src/agent/tools.py`: 7 tool stubs, TODO(member-3) at line 9.
- `src/store/prices.py`: `load_prices`, `load_benchmarks` and `first_tradeable_date`, TODO(member-1) at line 27. `data_dir()` is already implemented.

## 5. Next actions

Steps a–f are all complete. What remains:

1. Optional: change `eval/` to `evaluation/` in `docs/design.md:282` with a small follow-up commit on `setup/skeleton`.
2. Ask the owners to review PR #1 and confirm or replace their placeholders:
   - Member 4: `SignalSpec`, `BacktestResult`
   - Member 3: `Finding` and the tool signatures
   - Member 1: the `prices.py` helpers
3. Merge PR #1 into `main` once it is approved. CI is already green.
4. Team: agree the research/holdout split, then fill in `research_end` and `holdout_start` in `configs/default.yaml`.
5. Optional, local only: install ruff on PATH (`pip install -e ".[dev]"`) so that the bare `ruff check .` works the way it does in CI.
