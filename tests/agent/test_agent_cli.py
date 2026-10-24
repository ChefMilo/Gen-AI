import json

import pytest
import yaml

import agent.run_session as cli
import agent.session as session_module
from agent.config import DEFAULT_CONFIG_PATH
from agent.fakes import PLANTED_SPEC
from agent.trace import read_trace

SPEC = {k: PLANTED_SPEC[k] for k in ("event_type", "conditions")}
NULL_SPEC = {"event_type": "dividend",
             "conditions": [{"field": "mcap_bucket", "op": "==", "value": "large"}]}


def step(name, arguments, i=0):
    return {"tool_calls": [{"id": f"t{i}", "name": name, "arguments": arguments}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 10}}


def script(spec):
    return [
        step("record_hypothesis", {"hypothesis": "h", "rationale": "r"}),
        step("define_signal", {"spec": spec, "hypothesis_id": "h1"}),
        step("run_backtest", {"signal_id": "sig-1"}),
        step("report_finding", {"signal_id": "sig-1", "rationale": "r",
                                "evidence": [{"event_id": "e1", "quote": "q"}]}),
        step("finish", {"reason": "done"}),
    ]


@pytest.fixture
def config_path(tmp_path, registry, monkeypatch):
    """The team config with agent outputs redirected to tmp_path."""
    monkeypatch.setattr(session_module, "code_version", lambda: None)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["agent"]["trace_dir"] = str(tmp_path / "traces")
    config["agent"]["cache_dir"] = str(tmp_path / "cache")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def no_api(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no API client may be built in this test")

    monkeypatch.setattr(cli, "AnthropicClient", refuse)


def run_cli(config_path, *args):
    return cli.main(["--config", str(config_path), *args])


def test_dry_run_prints_prompt_brief_and_schemas_and_makes_no_call(
    config_path, no_api, capsys, tmp_path
):
    assert run_cli(config_path, "--system", "R2", "--budget", "15", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "=== SYSTEM PROMPT ===" in out and "=== TASK BRIEF ===" in out
    assert "Backtest budget: 15 run_backtest calls" in out
    schemas = json.loads(out.split("=== TOOL SCHEMAS ===\n", 1)[1])
    assert "search_announcements" not in {t["name"] for t in schemas}
    assert not (tmp_path / "traces").exists()  # dry runs write no trace


def test_scripted_run_claiming_the_planted_signal_passes(config_path, no_api, capsys, tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps(script(SPEC)), encoding="utf-8")
    assert run_cli(config_path, "--system", "R2", "--budget", "15", "--fake-llm", str(path),
                   "--model", "override-model") == 0
    out = capsys.readouterr().out
    assert "end reason:    finished" in out
    assert "claims:        1" in out
    assert "smoke test:    PASS  best Jaccard vs planted = 1.00 (sig-1), threshold 0.6" in out

    traces = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(traces) == 1 and str(traces[0]) in out
    start = read_trace(traces[0])[0]
    assert start["settings"]["model"] == "override-model"
    assert start["universe_id"] == "fake_planted_seed0"


def test_scripted_run_claiming_a_null_signal_fails(config_path, no_api, capsys, tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps(script(NULL_SPEC)), encoding="utf-8")
    run_cli(config_path, "--system", "R2", "--fake-llm", str(path))
    assert "smoke test:    FAIL  best Jaccard vs planted = 0.00" in capsys.readouterr().out


def test_real_run_builds_a_cached_anthropic_client(config_path, monkeypatch):
    built = {}

    class FakeAnthropic:
        def __init__(self, model, max_tokens, effort):
            built.update(model=model, max_tokens=max_tokens, effort=effort)

    monkeypatch.setattr(cli, "AnthropicClient", FakeAnthropic)
    args = cli.parse_args(["--system", "R2", "--model", "m"])
    config = cli.load_config(config_path)
    config["agent"]["model"] = args.model
    client = cli.build_client(args, config["agent"])
    assert isinstance(client, cli.CachedClient)
    assert built == {"model": "m", "max_tokens": 16000, "effort": None}
    assert not isinstance(cli.build_client(cli.parse_args(["--system", "R2", "--no-cache"]),
                                           config["agent"]), cli.CachedClient)
