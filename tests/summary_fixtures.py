"""Synthetic summary tables shaped like the final plan: R2, R3, R3-U, and planted universes
that each hold ONE feature type, with one single and one conjunction truth.

MATCHED gives (R2, R3, R3-U) found per (universe, spec); the hand-computed values in
tests/test_figures.py and tests/test_group_comparisons.py come from it.
"""

import pandas as pd

from evaluation.metrics import TRUTH_COLUMNS, MetricsTables
from evaluation.oracle import COLUMNS as ORACLE_COLUMNS
from evaluation.stats import summary_frames

SYSTEMS = ["r2", "r3", "r3-u"]
# Text universes t1-t4 and structured universes s1-s4, in two strengths per universe.
TRUTHS = {
    ("t1", "text_single_weak"): ("single", "text", "weak", 1.5),
    ("t1", "text_conj_strong"): ("conjunction", "text", "strong", 9.0),
    ("t2", "text_single_strong"): ("single", "text", "strong", 18.0),
    ("t2", "text_conj_weak"): ("conjunction", "text", "weak", 2.5),
    ("t3", "text_single_weak"): ("single", "text", "weak", 1.0),
    ("t3", "text_conj_strong"): ("conjunction", "text", "strong", 6.0),
    ("t4", "text_single_strong"): ("single", "text", "strong", 20.0),
    ("t4", "text_conj_weak"): ("conjunction", "text", "weak", 3.0),
    ("s1", "struct_single_weak"): ("single", "structured", "weak", 1.2),
    ("s1", "struct_conj_strong"): ("conjunction", "structured", "strong", 10.0),
    ("s2", "struct_single_strong"): ("single", "structured", "strong", 17.0),
    ("s2", "struct_conj_weak"): ("conjunction", "structured", "weak", 2.0),
    ("s3", "struct_single_weak"): ("single", "structured", "weak", 1.8),
    ("s3", "struct_conj_strong"): ("conjunction", "structured", "strong", 7.0),
    ("s4", "struct_single_strong"): ("single", "structured", "strong", 16.0),
    ("s4", "struct_conj_weak"): ("conjunction", "structured", "weak", 2.2),
}
# Same keys as TRUTHS: found by (r2, r3, r3-u).
MATCHED = {
    ("t1", "text_single_weak"): (1, 1, 1),
    ("t1", "text_conj_strong"): (0, 1, 1),
    ("t2", "text_single_strong"): (0, 1, 1),
    ("t2", "text_conj_weak"): (0, 1, 1),
    ("t3", "text_single_weak"): (1, 0, 1),
    ("t3", "text_conj_strong"): (0, 1, 1),
    ("t4", "text_single_strong"): (0, 0, 0),
    ("t4", "text_conj_weak"): (1, 1, 1),
    ("s1", "struct_single_weak"): (1, 1, 1),
    ("s1", "struct_conj_strong"): (1, 1, 1),
    ("s2", "struct_single_strong"): (1, 1, 1),
    ("s2", "struct_conj_weak"): (0, 0, 1),
    ("s3", "struct_single_weak"): (0, 0, 0),
    ("s3", "struct_conj_strong"): (1, 1, 1),
    ("s4", "struct_single_strong"): (1, 1, 1),
    ("s4", "struct_conj_weak"): (0, 1, 1),
}
# Found under the system's own rule but not under the uniform one.
LOST_UNDER_UNIFORM = {("r3", "t1", "text_single_weak")}
# Placebo universe -> system -> (false discoveries, uniform false discoveries, backtests).
PLACEBO = {
    "p1": {"r2": (1, 0, 20), "r3": (1, 0, 30), "r3-u": (3, 1, 100)},
    "p2": {"r2": (0, 0, 20), "r3": (0, 0, 30), "r3-u": (1, 0, 100)},
}
EXTRA_FINDINGS = {"r2": 0, "r3": 1, "r3-u": 4}  # significant unmatched findings per universe
TOKENS = {"r2": (40_000, 10_000), "r3": (60_000, 15_000), "r3-u": (120_000, 30_000)}


def _truth_rows(systems: list[str]) -> pd.DataFrame:
    rows = []
    for key, (complexity, feature_type, strength, snr) in TRUTHS.items():
        universe_id, spec_id = key
        for k, system in enumerate(SYSTEMS):
            if system not in systems:
                continue
            matched = bool(MATCHED[key][k])
            rows.append({
                "system": system, "universe_id": universe_id, "spec_id": spec_id,
                "strength": strength, "complexity": complexity, "feature_type": feature_type,
                "event_type": "buyback" if feature_type == "text" else "dividend",
                "direction": "up", "snr": snr, "matched": matched,
                "matched_uniform": matched and (system, *key) not in LOST_UNDER_UNIFORM,
            })
    return pd.DataFrame(rows, columns=TRUTH_COLUMNS)


def _planted_rows(truths: pd.DataFrame) -> list[dict]:
    rows = []
    for (system, universe_id), group in truths.groupby(["system", "universe_id"], sort=False):
        n, matched = len(group), int(group["matched"].sum())
        uniform = int(group["matched_uniform"].sum())
        extra = EXTRA_FINDINGS[system]
        rows.append({
            "system": system, "universe_id": universe_id, "universe_type": "planted",
            "n_truths": n, "n_matched": matched, "recall": matched / n,
            "n_significant": matched + extra, "false_discoveries": float("nan"),
            "effect_error_sum": 0.0, "effect_error_abs_sum": 0.0,
            "n_matched_uniform": uniform, "recall_uniform": uniform / n,
            "n_significant_uniform": uniform + extra, "false_discoveries_uniform": float("nan"),
            **_efficiency(system),
        })
    return rows


def _efficiency(system: str) -> dict:
    budget = next(iter(PLACEBO.values()))[system][2]
    inputs, outputs = TOKENS[system]
    return {"backtests_used": budget, "tool_calls": 2 * budget, "tokens": inputs + outputs,
            "input_tokens": inputs, "output_tokens": outputs}


def _placebo_rows(systems: list[str]) -> list[dict]:
    rows = []
    for universe_id, per_system in PLACEBO.items():
        for system, (fd, fd_uniform, backtests) in per_system.items():
            if system not in systems:
                continue
            rows.append({
                "system": system, "universe_id": universe_id, "universe_type": "placebo",
                "n_truths": 0, "n_matched": 0, "recall": float("nan"), "n_significant": fd,
                "false_discoveries": fd, "effect_error_sum": 0.0, "effect_error_abs_sum": 0.0,
                "n_matched_uniform": 0, "recall_uniform": float("nan"),
                "n_significant_uniform": fd_uniform, "false_discoveries_uniform": fd_uniform,
                **_efficiency(system), "backtests_used": backtests,
            })
    return rows


def fixture_tables(systems: list[str] | None = None) -> MetricsTables:
    systems = SYSTEMS if systems is None else systems
    truths = _truth_rows(systems)
    rows = _planted_rows(truths) + _placebo_rows(systems)
    per_universe = pd.DataFrame(rows)
    per_universe["cost_usd"] = float("nan")
    return MetricsTables(per_universe, pd.DataFrame(), truths)


def write_fixture_summary(results, systems: list[str] | None = None, n_boot: int = 200) -> None:
    """results/summary/*.csv for these systems, written the way the runner writes them."""
    tables = fixture_tables(systems)
    summary = results / "summary"
    summary.mkdir(parents=True)
    frames = {"per_universe": tables.per_universe, "truths": tables.truths,
              **summary_frames(tables, seed=0, n_boot=n_boot)}
    for name, frame in frames.items():
        frame.to_csv(summary / f"{name}.csv", index=False)


def oracle_frame() -> pd.DataFrame:
    """Findable: every strong truth and the weak single text truths; other weak ones not."""
    rows = []
    for (universe_id, spec_id), (complexity, feature_type, strength, _) in TRUTHS.items():
        findable = strength == "strong" or (complexity, feature_type) == ("single", "text")
        rows.append({
            "spec_id": spec_id, "universe_id": universe_id, "strength": strength,
            "complexity": complexity, "feature_type": feature_type, "n_events": 100, "t": 4.0,
            "p_raw": 0.001, "p_corrected": 0.03, "findable": findable,
        })
    return pd.DataFrame(rows, columns=ORACLE_COLUMNS)
