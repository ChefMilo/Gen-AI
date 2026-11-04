"""configs/final.yaml's systems block resolves the agent systems from agent.systems, the
way python -m evaluation.run does it (resolve_system_names, then make_systems)."""

import importlib
import sys
from pathlib import Path

import pytest

from agent.config import REPO_ROOT
from agent.system import AgentResearchSystem, R2System, R3System, R3USystem
from baselines.r0 import R0System
from evaluation import systems as runner_systems
from evaluation.plan import load_experiment_config


@pytest.fixture
def fresh_registry(monkeypatch):
    """The runner's registry without the agent systems, and agent.systems not imported,
    so resolving the config has to import it."""
    monkeypatch.setattr(runner_systems, "SYSTEMS", {
        k: v for k, v in runner_systems.SYSTEMS.items() if k not in ("r2", "r3", "r3-u")})
    monkeypatch.setattr(runner_systems, "SYSTEM_SOURCES", dict(runner_systems.SYSTEM_SOURCES))
    monkeypatch.delitem(sys.modules, "agent.systems", raising=False)
    return runner_systems.SYSTEMS


def final_config():
    return load_experiment_config(Path(REPO_ROOT, "configs", "final.yaml")).resolved()


def test_final_config_lists_the_agent_module():
    cfg = final_config()
    assert cfg.system_modules == ["agent.systems"]
    assert cfg.system_names == ["r0", "r1", "r2", "r3", "r3-u"]


def test_final_systems_block_resolves_r0_and_the_agent_systems(fresh_registry):
    cfg = final_config()
    names = [n for n in cfg.system_names if n != "r1"]  # r1: Stage C
    runner_systems.resolve_system_names(names, cfg.system_modules)
    assert "agent.systems" in sys.modules
    for name in ("r2", "r3", "r3-u"):
        assert runner_systems.SYSTEM_SOURCES[name] == "agent.systems"
    made = runner_systems.make_systems(names, cfg.holding_days)  # builds, calls no LLM
    assert [s.name for s in made] == ["r0", "r2", "r3", "r3-u"]
    assert isinstance(made[0], R0System)
    assert [type(s) for s in made[1:]] == [R2System, R3System, R3USystem]
    assert all(isinstance(s, AgentResearchSystem) and s.holding_days == cfg.holding_days
               and s.client_factory is None for s in made[1:])
    assert all(isinstance(s, runner_systems.ResearchSystem) for s in made)


def test_registering_twice_is_a_no_op_and_there_is_one_path(fresh_registry):
    module = importlib.import_module("agent.systems")
    module.register()
    module.register()
    assert {fresh_registry[n] for n in ("r2", "r3", "r3-u")} == {R2System, R3System, R3USystem}
    import agent.experiment as experiment

    assert not hasattr(experiment, "register_agent_systems")  # no legacy path left
