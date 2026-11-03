"""The final grid (evaluation.final_grid), its dev overlap guard, the frozen plan hash,
the oracle check and the cost estimate. All offline."""

import logging

import pytest
import yaml

from backtest.selection import select_events
from evaluation.cost import LLM_FREE_SYSTEMS, cost_table, sessions_per_system
from evaluation.final_grid import (
    DevGuardConfig,
    GridConfig,
    base_report,
    build_grid,
    dev_overlap,
    dev_usage,
    grid_specs,
    spec_signature,
)
from evaluation.oracle import COLUMNS, findable_rates, monotone_violations, oracle_table
from evaluation.plan import (
    REPO_ROOT,
    DataConfig,
    ExperimentConfig,
    _target_key,
    load_experiment_config,
    plan_canonical_json,
    plan_experiment,
    plan_sha256,
)
from evaluation.run import check_frozen_plan, run_experiment
from experiment_helpers import small_data
from universes.planted import select_targets

FINAL = REPO_ROOT / "configs" / "final.yaml"
PLAN_DUMP = REPO_ROOT / "results" / "ci" / "final-plan.json"  # git-ignored


@pytest.fixture(scope="module")
def final_cfg():
    return load_experiment_config(FINAL)


@pytest.fixture(scope="module")
def data():
    return small_data()


def small_grid(final_cfg, n: int = 1) -> GridConfig:
    """The final grid cut to its n strongest strengths (fits the small test data)."""
    g = final_cfg.grid
    return GridConfig(strengths=g.strengths[-n:], directions=g.directions[-n:],
                      bases=g.bases)


def guarded(tmp_path, final_cfg, dev: dict | None = None, **overrides) -> ExperimentConfig:
    """A tiny guarded config on the small data: no planted universes, so only the guard
    can fail, and optionally a dev config written to tmp_path."""
    paths = []
    if dev is not None:
        path = tmp_path / "dev.yaml"
        path.write_text(yaml.safe_dump(dev), encoding="utf-8")
        paths.append(str(path))
    values = {
        "experiment_id": "guard", "n_planted": 0, "n_placebo": 2, "base_seed": 10000,
        "plant_min_events": 5, "systems": ["null"], "grid": small_grid(final_cfg, 2),
        "dev_guard": DevGuardConfig(min_seed=10000, dev_configs=paths),
    }
    return ExperimentConfig(**{**values, **overrides})


# --- configs/final.yaml -------------------------------------------------------------


def test_final_config(final_cfg):
    raw = yaml.safe_load(FINAL.read_text(encoding="utf-8"))
    assert "backtest_budget" not in raw and "alpha" not in raw  # from default.yaml
    assert final_cfg.experiment_id == "final-v1" and final_cfg.base_seed == 10000
    assert final_cfg.n_placebo == 10 and final_cfg.include_real
    assert final_cfg.system_names == ["r0", "r1", "r2", "r3", "r3-u"]
    assert final_cfg.system_modules == ["agent.systems"]
    assert final_cfg.dev_guard.min_seed == 10000
    assert final_cfg.resolved().backtest_budget is not None


def test_grid_is_four_bases_at_four_strengths(final_cfg):
    cells = build_grid(final_cfg.grid, holding_days=5)
    plants = [c.plant for c in cells]
    assert len(plants) == 16
    assert len({p.spec_id for p in plants}) == 16
    assert len({spec_signature(p) for p in plants}) == 16
    assert {(p.strength, p.complexity, p.feature_type) for p in plants} == {
        (s, c, f) for s in ("weak", "moderate", "strong", "very_strong")
        for c in ("single", "conjunction") for f in ("structured", "text")}
    # One base per (complexity, feature type): strength never changes the targets.
    by_cell = {}
    for p in plants:
        by_cell.setdefault((p.complexity, p.feature_type), set()).add(_target_key(p))
    assert len(by_cell) == 4 and all(len(keys) == 1 for keys in by_cell.values())
    assert build_grid(final_cfg.grid, holding_days=5) == cells  # deterministic


def test_plant_and_signal_select_the_same_events(final_cfg, data):
    for cell in build_grid(final_cfg.grid, holding_days=5):
        planted = set(select_targets(data.events, cell.plant)["event_id"])
        signal = set(select_events(data.events, cell.signal)["event_id"])
        assert planted == signal and planted, cell.plant.spec_id
        assert cell.signal.direction == ("long" if cell.plant.direction > 0 else "short")


def test_grid_validation(final_cfg):
    raw = final_cfg.grid.model_dump()
    raw["strengths"][0] = 0.03
    with pytest.raises(ValueError, match="no strength label"):
        GridConfig.model_validate(raw)
    raw = final_cfg.grid.model_dump()
    raw["bases"]["single"]["structured"] = raw["bases"]["conjunction"]["structured"] | {
        "name": "two_features"}
    with pytest.raises(ValueError, match="exactly one feature"):
        GridConfig.model_validate(raw)
    raw = final_cfg.grid.model_dump()
    raw["bases"]["conjunction"]["text"]["conditions"] = []
    with pytest.raises(ValueError, match="two features or more"):
        GridConfig.model_validate(raw)


def test_grid_specs_refuse_short_leaky_and_unknown_bases(final_cfg, data):
    with pytest.raises(ValueError, match="fewer than plant_min_events"):
        grid_specs(final_cfg.grid, data.events, min_events=10_000, holding_days=5)
    raw = final_cfg.grid.model_dump()
    raw["bases"]["single"]["text"]["text_filter"]["equals"] = "confidence"
    with pytest.raises(ValueError, match="not one of"):
        grid_specs(GridConfig.model_validate(raw), data.events, 1, 5)
    # Most dividends are framed as routine: the event type alone would match the plant.
    raw = final_cfg.grid.model_dump()
    raw["bases"]["single"]["text"]["text_filter"]["equals"] = "fake_routine"
    with pytest.raises(ValueError, match="event type alone"):
        grid_specs(GridConfig.model_validate(raw), data.events, 1, 5, jaccard_min=0.6)
    grid_specs(final_cfg.grid, data.events, 1, 5, jaccard_min=0.6)  # the final bases pass


def test_base_report(final_cfg, data):
    report = base_report(final_cfg.grid, data.events)
    assert len(report) == 4
    conj = report[report["complexity"] == "conjunction"]
    assert (conj["small_share"] == 1.0).all()
    assert (report["max_part_jaccard"] < 0.6).all()


def test_signature_ignores_names_and_condition_order(final_cfg):
    plant = next(c.plant for c in build_grid(final_cfg.grid, 5)
                 if c.plant.condition and " and " in c.plant.condition)
    assert " and " in plant.condition
    first, second = plant.condition.split(" and ")
    renamed = plant.model_copy(update={"spec_id": "other", "condition": f"{second}  and {first}"})
    assert spec_signature(renamed) == spec_signature(plant)
    assert spec_signature(plant.model_copy(update={"direction": -plant.direction})) != \
        spec_signature(plant)


# --- Dev overlap guard ------------------------------------------------------------------


def test_guard_passes_on_disjoint_dev(tmp_path, final_cfg, data):
    cfg = guarded(tmp_path, final_cfg, dev={"experiment_id": "dev", "base_seed": 5000,
                                            "n_planted": 2, "n_placebo": 2})
    plan_experiment(data.prices, data.events, cfg, data.benchmarks)


def test_guard_counts_dev_placebo_seeds(tmp_path, final_cfg, data):
    """A dev plan at base_seed 9000 puts its placebos at 10000+ (base + 1000)."""
    cfg = guarded(tmp_path, final_cfg, dev={"experiment_id": "dev", "base_seed": 9000,
                                            "n_planted": 2, "n_placebo": 2})
    with pytest.raises(ValueError, match=r"dev configs use seeds >= .*\[10000, 10001\]"):
        plan_experiment(data.prices, data.events, cfg, data.benchmarks)


def test_guard_rejects_final_seeds_below_min_seed(tmp_path, final_cfg, data):
    cfg = guarded(tmp_path, final_cfg, base_seed=9999)
    with pytest.raises(ValueError, match="below dev_guard.min_seed"):
        plan_experiment(data.prices, data.events, cfg, data.benchmarks)


def test_guard_rejects_dev_seeds_in_the_final_range(tmp_path, final_cfg, data):
    dev = {"experiment": {"experiment_id": "dev", "base_seed": 9000, "n_planted": 0,
                          "n_placebo": 0},
           "dev": {"planted": [], "placebo": [{"seed": 11000}]}}
    cfg = guarded(tmp_path, final_cfg, dev=dev)
    with pytest.raises(ValueError, match="dev configs use seeds >= "):
        plan_experiment(data.prices, data.events, cfg, data.benchmarks)


def test_guard_rejects_a_dev_spec_with_another_name(tmp_path, final_cfg, data):
    final_spec = build_grid(small_grid(final_cfg, 2), 5)[0].plant
    copy = final_spec.model_dump(exclude={"seed", "holding_days"})
    copy["spec_id"] = "dev_innocent_name"
    dev = {"experiment": {"experiment_id": "dev", "base_seed": 9000, "n_planted": 0,
                          "n_placebo": 0},
           "dev": {"planted": [{"seed": 9001, "specs": [copy]}], "placebo": []}}
    cfg = guarded(tmp_path, final_cfg, dev=dev)
    with pytest.raises(ValueError, match=f"{final_spec.spec_id} = .*dev_innocent_name"):
        plan_experiment(data.prices, data.events, cfg, data.benchmarks)


# PR #18's configs/experiment_agent_dev.yaml dev specs (agent/system branch), inline
# because the file is not on main.
PR18_DEV_SPECS = [
    {"spec_id": "dev_new_program_large_up150bp", "event_type": "buyback",
     "condition": 'is_new_program == True and mcap_bucket == "large"', "effect": 0.015,
     "direction": 1, "strength": "moderate", "complexity": "conjunction",
     "feature_type": "structured"},
    {"spec_id": "dev_rationale_excess_cash_down150bp", "event_type": "buyback",
     "text_feature": {"feature": "buyback_rationale", "equals": "fake_excess_cash"},
     "effect": 0.015, "direction": -1, "strength": "moderate", "complexity": "single",
     "feature_type": "text"},
]


def agent_dev(specs: list[dict]) -> dict:
    """An agent.experiment {experiment, dev} config planting these specs at seed 9001."""
    return {"experiment": {"experiment_id": "dev", "base_seed": 9000, "n_planted": 0,
                           "n_placebo": 0, "holding_days": 5},
            "dev": {"planted": [{"seed": 9001, "specs": specs}], "placebo": []}}


def dev_spec(base, spec_id: str, condition: str | None = None, **update) -> dict:
    """A dev spec on a final base's events, different in effect (so not the same
    signature), with condition replacing the base's."""
    spec = base.model_dump(exclude={"seed", "holding_days"})
    spec.update(spec_id=spec_id, effect=0.015, strength="moderate", **update)
    if condition is not None:
        spec["condition"] = condition
    return spec


def final_base(final_cfg, name: str):
    return next(c.plant for c in build_grid(small_grid(final_cfg, 2), 5)
                if c.plant.spec_id.startswith(name))


def test_final_grid_does_not_overlap_pr18_dev_specs(tmp_path, final_cfg, data):
    cfg = guarded(tmp_path, final_cfg, dev=agent_dev(PR18_DEV_SPECS))
    plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    usage = dev_usage(cfg.dev_guard.dev_configs, data.events)
    assert len(usage.agent_specs) == 2
    report = dev_overlap([c.plant for c in build_grid(final_cfg.grid, 5)], usage.agent_specs,
                         data.events)
    assert len(report) == 4 * 2 and (report["n_variants"] == 4).all()
    assert (report["n_shared"] == 0).all()


def test_guard_rejects_a_final_base_inside_a_dev_spec(tmp_path, final_cfg, data):
    """The excess-cash case: a small-cap base nested in an all-caps dev spec has a low
    Jaccard but containment 1."""
    base = final_base(final_cfg, "buyback_rationale_undervalued_small")
    broader = dev_spec(base, "dev_undervalued_all_caps", condition="")
    cfg = guarded(tmp_path, final_cfg, dev=agent_dev([broader]))
    with pytest.raises(ValueError, match=rf"{base.spec_id} vs .*dev_undervalued_all_caps: "
                                         r"\d+/\d+ shared \(containment 1\.00, Jaccard 0\.[0-5]"):
        plan_experiment(data.prices, data.events, cfg, data.benchmarks)


def test_guard_rejects_a_dev_spec_inside_a_final_base(tmp_path, final_cfg, data):
    base = final_base(final_cfg, "dividend_large")
    narrower = dev_spec(base, "dev_large_raises",
                        condition=f"{base.condition} and dividend_change_pct > 0")
    cfg = guarded(tmp_path, final_cfg, dev=agent_dev([narrower]))
    with pytest.raises(ValueError, match=f"{base.spec_id} vs .*dev_large_raises"):
        plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    # Below max_containment (and jaccard_min) it passes.
    loose = cfg.dev_guard.model_copy(update={"max_containment": 0.9})
    plan_experiment(data.prices, data.events, cfg.model_copy(update={"dev_guard": loose}),
                    data.benchmarks)


def test_guard_rejects_a_dev_spec_at_jaccard_min(tmp_path, final_cfg, data):
    """With containment allowed, the Jaccard check still catches the same targets under
    another effect (a different signature)."""
    base = final_base(final_cfg, "dividend_raise_small")
    same = dev_spec(base, "dev_raise_small_again")
    cfg = guarded(tmp_path, final_cfg, dev=agent_dev([same]))
    cfg = cfg.model_copy(update={"dev_guard": cfg.dev_guard.model_copy(
        update={"max_containment": 1.0})})
    with pytest.raises(ValueError, match=r"Jaccard >= jaccard_min=.*"
                                         r"dev_raise_small_again: .*Jaccard 1\.00"):
        plan_experiment(data.prices, data.events, cfg, data.benchmarks)


def test_final_grid_is_disjoint_from_the_fake_v1_grid(final_cfg, data):
    """configs/final.yaml guards against configs/experiment.yaml's default grid: seeds
    and signatures only (its specs are runner-test specs, not agent dev specs)."""
    cfg = final_cfg.model_copy(update={"n_planted": 0, "n_placebo": 1, "plant_min_events": 1})
    plan_experiment(data.prices, data.events, cfg, data.benchmarks)


# --- Frozen plan hash -------------------------------------------------------------------


def test_plan_hash_and_frozen_check(tmp_path, final_cfg, data, caplog):
    cfg = guarded(tmp_path, final_cfg)
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    again = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    assert plan_sha256(plan) == plan_sha256(again)

    frozen = tmp_path / "plan.sha256"
    frozen.write_text(f"{plan_sha256(plan)}  guard/plan.json\n", encoding="utf-8")
    plan = plan.model_copy(update={"config": plan.config.model_copy(
        update={"plan_sha256_file": str(frozen)})})
    with caplog.at_level(logging.WARNING, logger="evaluation.run"):
        check_frozen_plan(plan)  # the hash covers the config, so it changed
    assert "PLAN CHANGED" in caplog.text
    caplog.clear()
    frozen.write_text(plan_sha256(plan) + "\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="evaluation.run"):
        assert check_frozen_plan(plan) == plan_sha256(plan)
    assert "PLAN CHANGED" not in caplog.text


# --- Oracle and cost --------------------------------------------------------------------


def test_oracle_table_on_a_small_plan(tmp_path, final_cfg, data):
    cfg = ExperimentConfig(
        experiment_id="oracle", n_planted=4, n_placebo=1, include_real=False,
        plant_min_events=5, base_seed=10000, systems=["null"], backtest_budget=30,
        grid=small_grid(final_cfg, 1), data=DataConfig(replicates="resample_noise"),
    )
    out_dir = tmp_path / "results"
    (tmp_path / ".gitignore").write_text("results/\n", encoding="utf-8")
    outcome = run_experiment(cfg, [], out_dir, data=data, dry_run=True)
    table = oracle_table(outcome.plan, data, out_dir, alpha=0.05, min_events=5)
    assert list(table.columns) == COLUMNS
    assert len(table) == sum(len(u.specs) for u in outcome.plan.by_type("planted"))
    assert set(table["spec_id"]) == {c.plant.spec_id for c in build_grid(cfg.grid, 5)}
    assert (table["p_corrected"] >= table["p_raw"]).all()
    assert (table["p_corrected"] == (table["p_raw"] * 30).clip(upper=1)).all()
    assert table.loc[table["complexity"] == "single", "findable"].all()  # 4% is findable
    assert monotone_violations(table).empty  # one strength: nothing to violate
    rates = findable_rates(table)
    assert rates["n"].sum() == len(table)


def test_cost_table_skips_llm_free_systems(final_cfg, data):
    cfg = final_cfg.model_copy(update={"n_planted": 0, "n_placebo": 3, "plant_min_events": 1})
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    sessions = sessions_per_system(plan, cfg.system_names)
    assert sessions == {name: 4 for name in cfg.system_names}  # 3 placebo + 1 real
    assert "r0" in LLM_FREE_SYSTEMS
    table = cost_table(sessions, {n: (1000, 100) for n in sessions}, 2.0, 10.0)
    assert list(table["system"]) == ["r1", "r2", "r3", "r3-u", "TOTAL"]
    row = table.iloc[0]
    assert row["tokens"] == 4 * 1100
    assert row["usd"] == pytest.approx(4 * (1000 * 2 + 100 * 10) / 1e6)
    assert table.iloc[-1]["usd"] == pytest.approx(4 * row["usd"])


def test_frozen_final_plan_hash_reproduces(final_cfg):
    """configs/final.plan.sha256 matches the plan configs/final.yaml builds today. If this
    fails, the final plan changed: re-freeze only on purpose (--freeze-plan)."""
    from evaluation.plan import frozen_plan_sha256
    from evaluation.run import load_base_data

    cfg = final_cfg.resolved()
    data = load_base_data(cfg)
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    actual, frozen = plan_sha256(plan), frozen_plan_sha256(cfg)
    if actual != frozen:
        # For diffing against another machine's build (CI uploads this file).
        PLAN_DUMP.parent.mkdir(parents=True, exist_ok=True)
        PLAN_DUMP.write_text(plan_canonical_json(plan), encoding="utf-8")
    assert actual == frozen, f"canonical plan JSON written to {PLAN_DUMP}"
