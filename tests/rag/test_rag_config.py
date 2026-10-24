from rag.config import REPO_ROOT, load_rag_config


def test_default_config_parses():
    config = load_rag_config()
    assert config.index_dir == REPO_ROOT / "results" / "rag_index"
    assert {spec.event_type for spec in config.text_features.values()} == {"dividend", "buyback"}
    assert all(len(spec.labels) >= 2 for spec in config.text_features.values())
