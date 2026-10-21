# Gen-AI

Agentic event-driven alpha research with planted-signal evaluation. See `docs/design.md` for the architecture and `docs/data_schema.md` for the data (which takes precedence where the two disagree).

## Setup

```
pip install -e .[dev]
export GENAI_DATA_DIR="/path/to/Gen AI/Data"   # local copy of the shared Drive folder
python -m pytest -q
```

Code lives in `src/` (one package per layer, owners named in each `__init__.py`).
