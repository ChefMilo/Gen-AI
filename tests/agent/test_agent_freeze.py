"""The prompt freeze: frozen_prompt_versions in the agent config, the refusal, and the
--allow-unfrozen flag (agent.experiment only). No LLM calls."""

import json
from pathlib import Path

import pytest
import yaml

import agent.session as session_module
from agent.config import REPO_ROOT, load_config
from agent.freeze import frozen_yaml, plan_prompt_versions, unfrozen
from agent.llm import FakeLLM, LLMResponse
from agent.system import AgentResearchSystem, PromptNotFrozen, check_frozen
from baselines.r1_single_shot import R1System

FINAL = Path(REPO_ROOT, "configs", "final.yaml")
FROZEN = {"r1": ["114ad3973638"], "r2": ["acbabba4fe3e"], "r3": ["363f54bdad43"],
          "r3-u": ["363f54bdad43"]}


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(session_module, "code_version", lambda: None)


def test_config_records_the_frozen_versions():
    agent = load_config()["agent"]
    assert agent["frozen_prompt_versions"] == FROZEN
    assert agent["max_session_input_tokens"] == 600_000


def test_check_frozen_refuses_unless_allowed():
    agent = {"frozen_prompt_versions": FROZEN}
    assert check_frozen("r3", "363f54bdad43", agent, allow_unfrozen=False) is True
    with pytest.raises(PromptNotFrozen, match="r3: prompt_version abc .*--allow-unfrozen"):
        check_frozen("r3", "abc", agent, allow_unfrozen=False)
    assert check_frozen("r3", "abc", agent, allow_unfrozen=True) is False
    with pytest.raises(PromptNotFrozen):  # a frozen version of another system does not count
        check_frozen("r2", "363f54bdad43", agent, allow_unfrozen=False)
    with pytest.raises(PromptNotFrozen):  # nothing frozen
        check_frozen("r2", "acbabba4fe3e", {}, allow_unfrozen=False)


@pytest.fixture(scope="module")
def final_versions():
    """Two final-plan universes, computed as a run would. Importing agent.systems registers
    systems with the runner, so the runner's registry and the module are restored after."""
    import sys

    import agent.tools as tools
    from evaluation import systems as runner_systems

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tools, "_TOOLS", dict(tools._TOOLS))  # sessions register interim tools
        mp.setattr(runner_systems, "SYSTEMS", dict(runner_systems.SYSTEMS))
        mp.setattr(runner_systems, "SYSTEM_SOURCES", dict(runner_systems.SYSTEM_SOURCES))
        mp.delitem(sys.modules, "agent.systems", raising=False)
        versions = plan_prompt_versions(FINAL, universe_ids=["final-v1-u00", "final-v1-u01"])
        sys.modules.pop("agent.systems", None)
    return versions


def test_final_plan_sessions_pass_the_freeze(final_versions, registry):
    assert set(final_versions) == {"r1", "r2", "r3", "r3-u"}
    assert all(len(v) == 2 for v in final_versions.values())
    assert unfrozen(final_versions, load_config()["agent"]) == []
    assert yaml.safe_load(frozen_yaml(final_versions)) == {"frozen_prompt_versions": FROZEN}


def test_an_unfrozen_session_is_refused_before_any_client(final_versions, registry):
    """A frozen-looking session whose config no longer lists its version is refused, for
    the agents and for R1, before a client is built; with allow_unfrozen it runs."""
    from evaluation.plan import load_experiment_config, plan_experiment
    from evaluation.run import load_base_data
    from evaluation.systems import make_view
    from evaluation.universe_store import build_universe

    cfg = load_experiment_config(FINAL).resolved()
    data = load_base_data(cfg)
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    entry = plan.entry("final-v1-u00")
    universe = build_universe(entry, data)
    view = make_view(entry.universe_id, universe.prices, universe.benchmarks, universe.events,
                     cfg.research_start, cfg.research_end)
    finish = LLMResponse(tool_calls=[{"id": "t", "name": "finish",
                                      "arguments": {"reason": "done"}}], stop_reason="tool_use")
    made = []

    def factory(agent, seed):
        made.append(seed)
        return FakeLLM([finish, LLMResponse(text="[]")])

    config = load_config()
    config["agent"] |= {"trace_dir": None, "model": "m"}
    run = AgentResearchSystem("R2", config=config, client_factory=factory).run(view, 30, 1)
    assert run.metadata["prompt_frozen"] is True  # the frozen final-plan prompt runs

    thawed = json.loads(json.dumps(config, default=str))
    thawed["agent"]["frozen_prompt_versions"] = {"r2": ["000000000000"], "r1": []}
    for system in (AgentResearchSystem("R2", config=thawed, client_factory=factory),
                   R1System(config=thawed, client_factory=factory)):
        with pytest.raises(PromptNotFrozen):
            system.run(view, 30, 1)
    assert len(made) == 1  # only the frozen run built a client
    allowed = AgentResearchSystem("R2", config=thawed, client_factory=factory,
                                  allow_unfrozen=True).run(view, 30, 1)
    assert allowed.metadata["prompt_frozen"] is False


def test_only_agent_experiment_can_allow_unfrozen(monkeypatch):
    import sys

    import agent.experiment as experiment
    from evaluation import run as runner
    from evaluation import systems as runner_systems

    # main() imports agent.systems, which registers with the runner: keep that private
    monkeypatch.setattr(runner_systems, "SYSTEMS", dict(runner_systems.SYSTEMS))
    monkeypatch.setattr(runner_systems, "SYSTEM_SOURCES", dict(runner_systems.SYSTEM_SOURCES))
    monkeypatch.delitem(sys.modules, "agent.systems", raising=False)

    seen = {}

    def capture(cfg, names, out_dir, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(experiment, "run_dev_experiment", capture)
    with pytest.raises(SystemExit):
        experiment.main(["--config", str(REPO_ROOT / "configs" / "experiment_agent_dev.yaml"),
                         "--systems", "r2", "--allow-unfrozen"])
    assert seen["allow_unfrozen"] is True
    with pytest.raises(SystemExit) as exit_:  # the runner has no such flag
        runner.main(["--config", str(FINAL), "--allow-unfrozen"])
    assert exit_.value.code == 2
    sys.modules.pop("agent.systems", None)
