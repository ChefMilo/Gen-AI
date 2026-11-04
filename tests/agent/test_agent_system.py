"""The agent systems inside the team's experiment runner, driven by FakeLLM scripts."""

import json
from datetime import date

import pytest

import agent.session as session_module
import agent.tools as tools
from agent.config import load_config
from agent.experiment import (
    DevExperimentConfig,
    build_dev_plan,
    check_dev_only,
    run_dev_experiment,
)
from agent.finding import Finding
from agent.llm import CachedClient, FakeLLM, LLMResponse, ToolCall, Usage
from agent.system import (
    AgentResearchSystem,
    R2System,
    ResearchEndNotSet,
    cost_usd,
)
from evaluation import systems as runner_systems
from evaluation.run import load_base_data
from evaluation.systems import ResearchSystem, make_view

CONDITION = [{"field": "dividend_change_pct", "op": ">", "value": 10}]
EXPECTED_METADATA = {
    "backtests_used", "tool_calls", "tokens", "cost_usd", "wall_time_s", "trace_path",
    "model", "prompt_version", "end_reason", "invalid_calls", "inspect_results_calls",
    "breakdown_groups_viewed", "signal_specs", "token_breakdown", "research_end",
    "warnings", "mask_dates_in_text", "seed", "budget", "harness", "duplicate_backtests",
    "strict_family_size", "findings_passed_strict", "strict_correction",
}
PRICING = {"test-model": {"input": 3.0, "cache_read": 0.3, "cache_write": 3.75, "output": 15.0}}

DEV = {
    "experiment": {
        "experiment_id": "dev_test", "n_planted": 0, "n_placebo": 0, "include_real": False,
        "base_seed": 9500, "holding_days": 5, "backtest_budget": 15, "plant_min_events": 20,
        "n_boot": 20,
        "data": {"source": "synthetic", "n_tickers": 40, "start": "2015-01-01",
                 "end": "2020-12-31", "seed": 9500, "events_seed": 9500},
    },
    "dev": {
        "planted": [{"seed": 9501, "specs": [{
            "spec_id": "dev_test_raise_up300bp", "event_type": "dividend",
            "condition": "dividend_change_pct > 10", "effect": 0.03, "direction": 1,
            "strength": "strong", "complexity": "single", "feature_type": "structured"}]}],
        "placebo": [{"seed": 9601}],
    },
}


def turn(*calls):
    tool_calls = [ToolCall(id=f"t{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)]
    return LLMResponse(tool_calls=tool_calls, stop_reason="tool_use",
                       usage=Usage(input_tokens=100, cache_read_input_tokens=1000,
                                   cache_creation_input_tokens=200, output_tokens=50))


SCRIPT = [
    turn(("record_hypothesis", {"hypothesis": "Dividend raises", "rationale": "signalling"})),
    turn(("define_signal", {"spec": {"event_type": "dividend", "conditions": CONDITION},
                            "hypothesis_id": "h1"})),
    turn(("run_backtest", {"signal_id": "sig-1"})),
    turn(("inspect_results", {"signal_id": "sig-1", "by": "year"})),
    turn(("report_finding", {"signal_id": "sig-1", "rationale": "raises signal confidence",
                             "evidence": [{"doc_ref": "x", "quote": "raised"}]})),
    turn(("finish", {"reason": "done"})),
]


class Clients:
    """client_factory that hands every session a fresh scripted FakeLLM and keeps them."""

    def __init__(self, script=SCRIPT):
        self.script, self.made = script, []

    def __call__(self, agent, seed):
        llm = FakeLLM(self.script)
        self.made.append((seed, llm))
        return llm


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(session_module, "code_version", lambda: None)


@pytest.fixture(scope="module")
def dev():
    cfg = DevExperimentConfig.model_validate(DEV)
    return cfg, load_base_data(cfg.experiment.resolved())


def agent_config(**agent):
    config = load_config()
    config["agent"] |= {"model": "test-model", "pricing": PRICING, **agent}
    return config


@pytest.fixture(scope="module")
def r2_run(dev, tmp_path_factory):
    """One real-runner experiment: r2 on a planted and a placebo dev universe."""
    cfg, data = dev
    clients = Clients()
    out = tmp_path_factory.mktemp("dev_run")
    saved = dict(tools._TOOLS)  # sessions register interim tools; restore them afterwards
    try:
        outcome = run_dev_experiment(cfg, ["r2"], out, data=data,
                                     agent_config=agent_config(), client_factory=clients)
    finally:
        tools._TOOLS.clear()
        tools._TOOLS.update(saved)
    return outcome, out, clients


def register_agent_systems():
    """Register through agent.systems, imported here (inside a test with runner_registry)
    so that collecting this file never touches the runner's real registry."""
    import importlib

    importlib.import_module("agent.systems").register()


@pytest.fixture
def runner_registry(monkeypatch):
    """A private copy of the runner's system registry."""
    monkeypatch.setattr(runner_systems, "SYSTEMS", dict(runner_systems.SYSTEMS))
    monkeypatch.setattr(runner_systems, "SYSTEM_SOURCES", dict(runner_systems.SYSTEM_SOURCES))
    return runner_systems.SYSTEMS


def run_meta(out, system, universe_id):
    path = out / "runs" / system / universe_id / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_scripted_r2_finding_is_recovered_by_the_matcher(r2_run):
    outcome, _, _ = r2_run
    assert {p.status for p in outcome.pairs} == {"ran"}
    rows = outcome.tables.per_universe.set_index("universe_type")
    assert rows.loc["planted", "n_matched"] == 1 and rows.loc["planted", "recall"] == 1.0


def test_placebo_findings_are_counted(r2_run):
    outcome, out, _ = r2_run
    placebo = outcome.plan.by_type("placebo")[0].universe_id
    findings = [json.loads(line) for line in
                (out / "runs" / "r2" / placebo / "findings.jsonl").read_text().splitlines()]
    assert len(findings) == 1
    row = outcome.tables.per_universe.set_index("universe_type").loc["placebo"]
    assert row["false_discoveries"] == sum(f["passed_correction"] for f in findings)
    assert row["n_findings"] == 1


def test_metadata_has_every_expected_key_and_no_universe_type(r2_run):
    outcome, out, _ = r2_run
    for entry in outcome.plan.universes:
        meta = run_meta(out, "r2", entry.universe_id)["metadata"]
        assert EXPECTED_METADATA <= set(meta)
        assert meta["backtests_used"] == 1 and meta["end_reason"] == "finished"
        assert meta["tokens"] == 6 * (100 + 1000 + 200 + 50)
        assert meta["token_breakdown"]["cache_read_input_tokens"] == 6000
        assert meta["cost_usd"] == pytest.approx(
            (600 * 3 + 6000 * 0.3 + 1200 * 3.75 + 300 * 15) / 1e6)
        assert meta["inspect_results_calls"] == 1 and meta["breakdown_groups_viewed"] >= 1
        assert meta["signal_specs"]["sig-1"]["conditions"] == CONDITION
        assert meta["warnings"] == [] and meta["research_end"] == "2022-12-31"
        text = json.dumps(meta)
        assert "planted" not in text and "placebo" not in text
        assert (out / "traces").exists() and meta["trace_path"].startswith(str(out))


def test_the_model_never_sees_the_universe(r2_run):
    outcome, _, clients = r2_run
    ids = [e.universe_id for e in outcome.plan.universes]
    for _, llm in clients.made:
        seen = json.dumps(llm.calls, default=str)
        assert not [i for i in ids if i in seen]
        assert "planted" not in seen and "placebo" not in seen


def test_each_pair_gets_its_own_seed(r2_run):
    _, _, clients = r2_run
    seeds = [seed for seed, _ in clients.made]
    assert len(set(seeds)) == len(seeds) == 2


# --- R3, R3-U and the research_end guard --------------------------------------------


@pytest.fixture(scope="module")
def planted_view(dev):
    cfg, data = dev
    plan = build_dev_plan(cfg, data)
    from evaluation.universe_store import build_universe

    entry = plan.by_type("planted")[0]
    universe = build_universe(entry, data)
    view = make_view(entry.universe_id, universe.prices, universe.benchmarks, universe.events,
                     date(2016, 1, 1), None)
    return view, {e.universe_id: e.universe_type for e in plan.universes}


def test_r3_gets_the_retrieval_tools_and_r2_does_not(planted_view, registry, tmp_path):
    view, types = planted_view
    tools = {}
    for name in ("R2", "R3"):
        clients = Clients()
        system = AgentResearchSystem(name, config=agent_config(trace_dir=None),
                                     client_factory=clients, universe_types=types)
        run = system.run(view, 15, 1)
        tools[name] = {t["name"] for t in clients.made[0][1].calls[0]["tools"]}
        assert run.metadata["retriever"] == (None if name == "R2" else "fake")
    assert tools["R3"] - tools["R2"] == {"search_announcements", "get_text_feature"}


def test_r3u_uses_the_plan_budget_and_no_correction(planted_view, registry):
    view, types = planted_view
    system = AgentResearchSystem("R3-U", config=agent_config(trace_dir=None),
                                 client_factory=Clients(), universe_types=types)
    run = system.run(view, 15, 1)
    assert run.metadata["budget"] == 15 and run.metadata["harness"]["correction"] == "none"
    f = run.findings[0]
    assert f.system == "R3-U" and f.p_value_corrected == f.p_value_raw


def test_real_universe_without_research_end_is_refused(planted_view, registry):
    view, _ = planted_view
    real = AgentResearchSystem("R2", config=agent_config(trace_dir=None),
                               client_factory=Clients(), universe_types={view.universe_id: "real"})
    with pytest.raises(ResearchEndNotSet, match="research/holdout split"):
        real.run(view, 15, 1)
    no_map = AgentResearchSystem("R2", config=agent_config(trace_dir=None),
                                 client_factory=Clients())
    with pytest.raises(ResearchEndNotSet, match="universe type is unknown"):
        no_map.run(view, 15, 1)
    ended = make_view(view.universe_id, view.prices, view.benchmarks, view.events,
                      date(2016, 1, 1), date(2019, 12, 31))
    run = real.run(ended, 15, 1)  # a set research_end runs anywhere
    assert run.metadata["research_end"] == "2019-12-31" and run.metadata["warnings"] == []


def test_runner_built_systems_run_now_that_research_end_is_set(dev, tmp_path, registry,
                                                               runner_registry, monkeypatch):
    """research_end is set in configs/default.yaml, so systems built by the runner (no
    universe-type map) run; the guard only refuses an open end (tests above)."""
    import agent.system as system_module

    cfg, data = dev
    register_agent_systems()
    monkeypatch.setattr(system_module, "default_client", Clients())
    systems = runner_systems.make_systems(["r2"])
    from evaluation.run import run_systems
    from evaluation.universe_store import build_universes

    plan = build_dev_plan(cfg, data)
    assert plan.config.research_end == date(2022, 12, 31)
    build_universes(tmp_path, plan.universes, plan.fingerprint, data)
    statuses = run_systems(plan, systems, data, tmp_path)
    assert {s.status for s in statuses} == {"ran"}


def test_the_default_client_is_never_real_in_tests(planted_view, registry):
    """The autouse fixture replaces default_client; a system without a client_factory
    must hit it, not the network."""
    view, types = planted_view
    ended = make_view(view.universe_id, view.prices, view.benchmarks, view.events,
                      date(2016, 1, 1), date(2019, 12, 31))
    system = AgentResearchSystem("R2", config=agent_config(trace_dir=None),
                                 universe_types=types)
    with pytest.raises(RuntimeError, match="real LLM client was requested in a test"):
        system.run(ended, 15, 1)


# --- Registration, dev guard, cache, cost ---------------------------------------------


def test_agent_systems_register_with_the_runner_names(runner_registry):
    register_agent_systems()
    made = runner_systems.make_systems(["r2", "r3", "r3-u"])
    assert [s.name for s in made] == ["r2", "r3", "r3-u"]
    assert all(isinstance(s, ResearchSystem) for s in made)
    assert isinstance(made[0], R2System)


def test_dev_specs_and_seeds_must_not_be_the_final_ones(dev):
    cfg, data = dev
    grid_spec = {**DEV["dev"]["planted"][0]["specs"][0], "effect": 0.02, "direction": -1,
                 "condition": "dividend_change_pct > 10"}
    bad = DevExperimentConfig.model_validate(
        {**DEV, "dev": {**DEV["dev"], "planted": [{"seed": 9501, "specs": [grid_spec]}]}})
    with pytest.raises(ValueError, match="final spec grid"):
        check_dev_only(bad, data.events)
    bad_seed = DevExperimentConfig.model_validate(
        {**DEV, "dev": {**DEV["dev"], "placebo": [{"seed": 1003}]}})
    with pytest.raises(ValueError, match="seeds"):
        check_dev_only(bad_seed, data.events)
    check_dev_only(cfg, data.events)  # the dev config itself passes


def test_dev_universe_ids_are_prefixed_and_opaque(dev):
    cfg, data = dev
    plan = build_dev_plan(cfg, data)
    assert [u.universe_id for u in plan.universes] == ["dev_u01", "dev_u02"]
    assert {u.seed for u in plan.universes} == {9501, 9601}


def test_cache_key_includes_the_seed(tmp_path):
    response = LLMResponse(text="hi")
    inner = FakeLLM([response] * 3)
    args = ("system", [{"role": "user", "text": "x"}], [])
    CachedClient(inner, tmp_path, seed=1).complete(*args)
    again = CachedClient(inner, tmp_path, seed=1)
    again.complete(*args)
    other = CachedClient(inner, tmp_path, seed=2)
    other.complete(*args)
    assert (again.hits, other.misses) == (1, 1) and len(inner.calls) == 2
    assert "seed" not in CachedClient(inner, tmp_path).settings  # old keys unchanged


def test_cost_needs_a_priced_model():
    tokens = {"input_tokens": 1_000_000, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "output_tokens": 1_000_000}
    assert cost_usd("test-model", tokens, PRICING) == pytest.approx(18.0)
    assert cost_usd("other", tokens, PRICING) is None
    assert cost_usd("test-model", tokens, {}) is None


def test_the_shipped_dev_config_is_dev_only():
    from pathlib import Path

    from agent.config import REPO_ROOT
    from agent.experiment import load_dev_config

    cfg = load_dev_config(Path(REPO_ROOT, "configs", "experiment_agent_dev.yaml"))
    assert (len(cfg.dev.planted), len(cfg.dev.placebo)) == (2, 3)
    data = load_base_data(cfg.experiment.resolved())
    check_dev_only(cfg, data.events)  # no final-grid spec, no final-plan seed
    plan = build_dev_plan(cfg, data)
    assert [u.universe_id for u in plan.universes] == [f"dev_u0{k}" for k in range(1, 6)]


# --- Strict correction (item 15): measured, not enforced ------------------------------

BROAD = [{"field": "dividend_change_pct", "op": ">", "value": 10}]
NARROW = [*BROAD, {"field": "mcap_bucket", "op": "==", "value": "small"}]
BREAKDOWN_PATH = [
    turn(("record_hypothesis", {"hypothesis": "Dividend raises", "rationale": "signalling"})),
    turn(("define_signal", {"spec": {"event_type": "dividend", "conditions": BROAD}})),
    turn(("run_backtest", {"signal_id": "sig-1"})),  # broad: fails
    turn(("inspect_results", {"signal_id": "sig-1", "by": "mcap_bucket"})),  # 3 groups seen
    turn(("record_hypothesis", {"hypothesis": "Small caps drive it", "rationale": "coverage"})),
    turn(("define_signal", {"spec": {"event_type": "dividend", "conditions": NARROW},
                            "hypothesis_id": "h2"})),
    turn(("run_backtest", {"signal_id": "sig-2"})),  # narrower: passes the official Holm
    turn(("report_finding", {"signal_id": "sig-2", "rationale": "small caps",
                             "evidence": [{"doc_ref": "x", "quote": "raised"}]})),
    turn(("finish", {"reason": "done"})),
]


@pytest.fixture(scope="module")
def breakdown_view():
    """A small universe (found by search) where the dev_u04 path plays out: the broad
    test fails, the small-cap breakdown stands out, the narrower test passes Holm over
    2 tests and fails it once the 3 viewed groups are counted."""
    from universes.fake_events import make_fake_events
    from universes.planted import PlantSpec, make_planted
    from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

    start, end, seed = date(2015, 1, 1), date(2020, 12, 31), 9709
    prices = make_synthetic_prices([f"T{i:03d}" for i in range(40)], start, end, seed=seed)
    benchmarks = make_synthetic_benchmarks(start, end, seed=seed)
    events = make_fake_events(prices, seed=seed)
    spec = PlantSpec(spec_id="dev_test_small_raise", event_type="dividend",
                     condition='dividend_change_pct > 10 and mcap_bucket == "small"',
                     effect=0.01, strength="moderate", complexity="conjunction",
                     feature_type="structured")
    planted, _ = make_planted(prices, events, spec, min_events=20, benchmarks=benchmarks)
    return make_view("dev_u99", planted, benchmarks, events, date(2016, 1, 1), None)


def test_breakdown_path_passes_officially_and_fails_strict(breakdown_view, registry):
    system = AgentResearchSystem("R2", config=agent_config(trace_dir=None),
                                 client_factory=Clients(BREAKDOWN_PATH),
                                 universe_types={"dev_u99": "planted"})
    run = system.run(breakdown_view, 15, 1)
    meta = run.metadata
    [finding] = run.findings
    assert finding.passed_correction  # official: Holm over the 2 backtests
    assert finding.p_value_corrected == pytest.approx(2 * finding.p_value_raw)
    [strict] = meta["strict_correction"]
    assert strict["passed_correction"] and not strict["passed_correction_strict"]
    assert strict["p_value_corrected_strict"] > 0.05 > strict["p_value_corrected"]
    assert meta["strict_family_size"] == 5 and meta["breakdown_groups_viewed"] == 3
    assert meta["findings_passed_strict"] == 0 and meta["duplicate_backtests"] == 0
    assert "p_value_corrected_strict" not in Finding.model_fields  # Finding is unchanged


def test_summary_prints_tokens_by_type_and_cost(r2_run):
    from agent.experiment import report

    outcome, out, _ = r2_run
    text = report(outcome, out)
    line = next(x for x in text.splitlines() if x.startswith("tokens r2"))
    assert "(2 sessions)" in line and "cache_read_input_tokens 12,000" in line
    assert "total 16,200" in line and "cost $" in line


def test_r3_and_r3u_see_byte_identical_prompts(planted_view, registry):
    """R3-U is a harness-only ablation: same system prompt, brief and tool schemas as R3
    (so the same prompt_version); only the correction differs."""
    view, types = planted_view
    seen, versions = {}, {}
    for name in ("R3", "R3-U"):
        clients = Clients()
        run = AgentResearchSystem(name, config=agent_config(trace_dir=None),
                                  client_factory=clients, universe_types=types).run(view, 15, 1)
        first = clients.made[0][1].calls[0]
        brief = first["messages"][0]["text"]
        seen[name] = (first["system"], brief, json.dumps(first["tools"], sort_keys=True))
        versions[name] = run.metadata["prompt_version"]
        for text in (first["system"], brief):
            assert "not enforced" not in text and "disabled" not in text
    assert seen["R3"] == seen["R3-U"]
    assert versions["R3"] == versions["R3-U"]


@pytest.mark.parametrize("name", ["R2", "R3", "R3-U"])
def test_runs_carry_what_the_uniform_and_sensitivity_scores_need(name, planted_view,
                                                                 registry):
    """Member 5's uniform rule needs only each Finding's p_value_raw plus the plan's budget;
    the attempted and strict sensitivity values are in the metadata, for every system."""
    from evaluation.metrics import UniformCorrection, recorrect

    view, types = planted_view
    run = AgentResearchSystem(name, config=agent_config(trace_dir=None),
                              client_factory=Clients(), universe_types=types).run(view, 30, 1)
    [finding] = run.findings
    [uniform] = recorrect(run.findings, UniformCorrection(budget=30, alpha=0.05))
    assert uniform.p_value_corrected == pytest.approx(min(1.0, 30 * finding.p_value_raw))
    [row] = run.metadata["strict_correction"]
    assert row["p_value_corrected_attempted"] == pytest.approx(finding.p_value_raw)  # 1 test
    assert {"passed_correction_attempted", "p_value_corrected_strict",
            "passed_correction_strict"} <= set(row)
    assert run.metadata["correction"] == ("none" if name == "R3-U" else "holm")
    assert run.metadata["budget"] == 30 and run.metadata["tests_in_family"] == 1
