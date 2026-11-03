# Gen-AI

Agentic event-driven alpha research with planted-signal evaluation. See `docs/design.md` for the architecture and `docs/data_schema.md` for the data (which takes precedence where the two disagree).

## Setup

```
pip install -e .[dev]
export GENAI_DATA_DIR="/path/to/Gen AI/Data"   # local copy of the shared Drive folder
python -m pytest -q
```

Code lives in `src/` (one package per layer, owners named in each `__init__.py`).

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

`configs/final.yaml` (`final-v1`) defines the frozen final evaluation. Its planted grid is built in code (`src/evaluation/final_grid.py`) from the config. There are 4 strengths × single/conjunction × structured/text, and each (complexity, feature type) cell has one base planted at all four strengths: 4 bases × 4 strengths = 16 specs. Two strengths of the same base never share a universe. Each planted and placebo universe redraws the synthetic prices' idiosyncratic noise from its own seed (`data.replicates: resample_noise`), so a spec's repeated plants are independent draws. Plan construction raises if a seed or spec (compared by canonical signature) is shared with the dev configs. Everything below runs offline, with no LLM calls:

```bash
python -m evaluation.run --config configs/final.yaml --dry-run --systems r0  # build + check the frozen hash
python scripts/oracle_check.py      # results/final-v1/oracle.csv: is each planted cell findable?
python scripts/estimate_cost.py     # sessions x tokens x price per LLM system
```

`configs/final.plan.sha256` holds the hash of the canonical `plan.json`. The runner warns when a `final-v1` plan differs from it, and `--freeze-plan` re-records it (do that only on purpose).

**The specs are not final yet.** The text bases use the fake event data's labels (`fake_growth`, `fake_excess_cash`), because the final text-feature label set is still being agreed. Once it is, update the grid's text bases to the agreed labels, re-run the oracle check, and re-freeze the plan with `--freeze-plan`. Until then the frozen hash pins the current spec set only.

Scoring reports two rules. The primary rule is each system's own `passed_correction`. The secondary rule (`*_uniform` columns) is a uniform re-correction: Holm over `backtest_budget` at `alpha` from `configs/default.yaml`, recomputed by the evaluator from each finding's `p_value_raw`.
