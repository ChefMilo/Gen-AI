"""Retrieval settings: the rag section of configs/default.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel

from rag.text_features import FeatureSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class RagConfig(BaseModel):
    index_dir: Path  # relative to the repo root
    embedding_model: str
    query_prefix: str = ""
    chunk_max_chars: int
    candidates: int
    rrf_k: int
    passages_per_feature: int
    text_features: dict[str, FeatureSpec]


def load_rag_config(path: Path = DEFAULT_CONFIG_PATH) -> RagConfig:
    with open(path, encoding="utf-8") as f:
        config = RagConfig.model_validate(yaml.safe_load(f)["rag"])
    return config.model_copy(update={"index_dir": REPO_ROOT / config.index_dir})
