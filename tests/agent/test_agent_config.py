from agent.config import MODEL_ENV, load_config


def test_default_config_has_agent_section(monkeypatch):
    monkeypatch.delenv(MODEL_ENV, raising=False)
    agent = load_config()["agent"]
    assert agent["model"] is None
    assert agent["max_steps"] == 60
    assert agent["max_tokens"] == 16000
    assert agent["effort"] is None
    assert agent["tool_result_max_chars"] == 4000
    assert agent["cache_dir"] == "results/llm_cache"
    assert agent["trace_dir"] == "results/traces"


def test_model_comes_from_env_when_unset(monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "some-model")
    assert load_config()["agent"]["model"] == "some-model"


def test_model_in_yaml_wins_over_env(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("agent:\n  model: yaml-model\n")
    monkeypatch.setenv(MODEL_ENV, "env-model")
    assert load_config(path)["agent"]["model"] == "yaml-model"


def test_env_vars_expand_and_unset_ones_become_none(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("data_dir: ${GENAI_DATA_DIR}\nother: ${NOT_SET_ANYWHERE_123}\n")
    monkeypatch.setenv("GENAI_DATA_DIR", "/data")
    config = load_config(path)
    assert config["data_dir"] == "/data"
    assert config["other"] is None
