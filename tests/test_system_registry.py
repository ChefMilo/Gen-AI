"""The system registry and the systems.modules config hook (evaluation.systems)."""

import logging
import sys
import textwrap
import types

import pytest
import yaml

from evaluation import run as run_module
from evaluation import systems as registry
from evaluation.plan import ExperimentConfig, SystemsConfig
from evaluation.systems import (
    LEGACY_SYSTEM_MODULE,
    NullSystem,
    SystemConfigError,
    make_systems,
    register_system,
    resolve_system_names,
)

MODULE = "registry_test_systems"


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    """Restore SYSTEMS and SYSTEM_SOURCES after each test; no real agent.experiment."""
    monkeypatch.setattr(registry, "SYSTEMS", dict(registry.SYSTEMS))
    monkeypatch.setattr(registry, "SYSTEM_SOURCES", dict(registry.SYSTEM_SOURCES))
    monkeypatch.setitem(sys.modules, LEGACY_SYSTEM_MODULE, None)  # import fails cleanly
    monkeypatch.delitem(sys.modules, MODULE, raising=False)


@pytest.fixture
def system_module(tmp_path, monkeypatch):
    """An importable module that registers "dummy" (taking holding_days only)."""
    (tmp_path / f"{MODULE}.py").write_text(textwrap.dedent("""
        from evaluation.systems import NullSystem, register_system

        class DummySystem(NullSystem):
            name = "dummy"

            def __init__(self, holding_days=5):
                self.holding_days = holding_days

        register_system("dummy", DummySystem)
    """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return MODULE


def write_config(tmp_path, systems) -> str:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({"experiment_id": "t", "systems": systems}),
                    encoding="utf-8")
    return str(path)


def test_list_form_still_parses():
    cfg = ExperimentConfig(experiment_id="t", systems=["null", "random"])
    assert cfg.system_names == ["null", "random"] and cfg.system_modules == []
    assert cfg.resolved().system_names == ["null", "random"]


def test_mapping_form_parses_and_survives_resolved():
    cfg = ExperimentConfig.model_validate(
        {"experiment_id": "t", "systems": {"names": ["dummy"], "modules": [MODULE]}})
    assert isinstance(cfg.systems, SystemsConfig)
    cfg = cfg.resolved()
    assert cfg.system_names == ["dummy"] and cfg.system_modules == [MODULE]
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(
            {"experiment_id": "t", "systems": {"names": [], "module": [MODULE]}})


def test_config_with_one_module(system_module, caplog):
    cfg = ExperimentConfig.model_validate(
        {"experiment_id": "t", "systems": {"names": ["null", "dummy"],
                                           "modules": [system_module]}})
    assert "dummy" not in registry.SYSTEMS
    with caplog.at_level(logging.WARNING):
        resolve_system_names(cfg.system_names, cfg.system_modules)
    assert registry.SYSTEM_SOURCES["dummy"] == MODULE
    null, dummy = make_systems(cfg.system_names, holding_days=7, min_events=30)
    assert isinstance(null, NullSystem)
    assert dummy.name == "dummy" and dummy.holding_days == 7  # min_events not passed
    assert "DEPRECATED" not in caplog.text
    resolve_system_names(["dummy"], [system_module])  # importing again is harmless


def test_missing_module_is_named():
    with pytest.raises(SystemConfigError, match="'no_such_systems_module'"):
        resolve_system_names(["null"], ["no_such_systems_module"])


def test_module_that_fails_on_import_is_named(tmp_path, monkeypatch):
    (tmp_path / "broken_systems.py").write_text("raise RuntimeError('boom')\n",
                                                encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemConfigError, match=r"'broken_systems'.*RuntimeError: boom"):
        resolve_system_names(["null"], ["broken_systems"])
    monkeypatch.delitem(sys.modules, "broken_systems", raising=False)


def test_unregistered_name_lists_available(system_module):
    with pytest.raises(SystemConfigError) as err:
        resolve_system_names(["null", "nope"], [system_module])
    message = str(err.value)
    assert "['nope']" in message
    for name in ("null", "random", "r0", "dummy"):
        assert repr(name) in message


def test_cli_reports_config_errors_before_loading_data(tmp_path, monkeypatch, capsys):
    def no_run(*args, **kwargs):
        raise AssertionError("must fail before running")

    monkeypatch.setattr(run_module, "run_experiment", no_run)
    path = write_config(tmp_path, {"names": ["null"], "modules": ["no_such_systems_module"]})
    with pytest.raises(SystemExit) as exit_:
        run_module.main(["--config", path])
    assert exit_.value.code == 2
    assert "no_such_systems_module" in capsys.readouterr().err

    path = write_config(tmp_path, ["null", "nope"])
    with pytest.raises(SystemExit):
        run_module.main(["--config", path])
    err = capsys.readouterr().err
    assert "nope" in err and "registered:" in err


def test_cli_uses_config_module(tmp_path, monkeypatch, system_module):
    seen = {}

    def fake_run(cfg, systems, out_dir, resume, dry_run):
        seen["names"] = [s.name for s in systems]
        raise SystemExit(0)

    monkeypatch.setattr(run_module, "run_experiment", fake_run)
    path = write_config(tmp_path, {"names": ["dummy", "null"], "modules": [system_module]})
    with pytest.raises(SystemExit):
        run_module.main(["--config", path, "--out-dir", str(tmp_path / "out")])
    assert seen["names"] == ["dummy", "null"]


def legacy_module() -> types.ModuleType:
    """A stand-in for agent.experiment, which updates SYSTEMS directly."""

    class LegacySystem(NullSystem):
        name = "legacy"

    module = types.ModuleType(LEGACY_SYSTEM_MODULE)
    module.register_agent_systems = lambda: registry.SYSTEMS.update({"legacy": LegacySystem})
    return module


def test_fallback_to_agent_experiment_warns(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, LEGACY_SYSTEM_MODULE, legacy_module())
    with caplog.at_level(logging.WARNING, logger="evaluation.systems"):
        resolve_system_names(["null", "legacy"])
    assert registry.SYSTEM_SOURCES["legacy"] == LEGACY_SYSTEM_MODULE
    [record] = [r for r in caplog.records if "DEPRECATED" in r.getMessage()]
    assert "'legacy'" in record.getMessage() and LEGACY_SYSTEM_MODULE in record.getMessage()
    [system] = make_systems(["legacy"])
    assert system.name == "legacy"


def test_fallback_not_used_when_modules_cover_the_names(system_module, monkeypatch, caplog):
    called = []
    module = legacy_module()
    module.register_agent_systems = lambda: called.append(True)
    monkeypatch.setitem(sys.modules, LEGACY_SYSTEM_MODULE, module)
    with caplog.at_level(logging.WARNING):
        resolve_system_names(["dummy"], [system_module])
    assert not called and "DEPRECATED" not in caplog.text


def test_register_conflicting_factory_raises():
    class Other(NullSystem):
        pass

    register_system("null", NullSystem)  # same factory: no-op
    with pytest.raises(ValueError, match="already registered"):
        register_system("null", Other)
