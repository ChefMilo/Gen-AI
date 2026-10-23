"""Run config: configs/default.yaml with environment overrides."""

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
MODEL_ENV = "AGENT_MODEL"
AGENT_PATH_KEYS = ("cache_dir", "trace_dir")


def _expand_env(value: Any) -> Any:
    """Replace ${VAR} in strings; a variable that is not set becomes None."""
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        return None if "${" in expanded else expanded
    return value


def resolve_path(path: str | Path) -> Path:
    """Relative paths are relative to the repo root, not the working directory."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config, expand ${VAR}s, fill agent.model from $AGENT_MODEL if null,
    and resolve the agent's output directories against the repo root."""
    import yaml

    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        config = _expand_env(yaml.safe_load(f))
    agent = config.setdefault("agent", {})
    if not agent.get("model"):
        agent["model"] = os.environ.get(MODEL_ENV) or None
    for key in AGENT_PATH_KEYS:
        if agent.get(key):
            agent[key] = resolve_path(agent[key])
    return config
