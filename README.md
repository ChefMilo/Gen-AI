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
