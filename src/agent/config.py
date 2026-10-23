"""Run config: configs/default.yaml with environment overrides."""

import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
MODEL_ENV = "AGENT_MODEL"


def _expand_env(value: Any) -> Any:
    """Replace ${VAR} in strings; a variable that is not set becomes None."""
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        return None if "${" in expanded else expanded
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config, expand ${VAR}s, and fill agent.model from $AGENT_MODEL if null."""
    import yaml  # agent extra

    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        config = _expand_env(yaml.safe_load(f))
    agent = config.setdefault("agent", {})
    if not agent.get("model"):
        agent["model"] = os.environ.get(MODEL_ENV) or None
    return config
