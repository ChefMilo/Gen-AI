"""Discovery, generalisation and efficiency metrics (docs/design.md section 3, "Metrics").

evaluate() scores one system's findings on one universe; metrics_table() turns a list
of those results into one row per (system, universe_id), the recall breakdowns, and one
row per planted truth (for statistics by group, evaluation.stats).

Definitions, per (system, universe):
- recall: matched truths / truths (planted only);
- precision: matched findings / significant findings (planted only), with
  n_unmatched = significant findings that did not match (wrong sign, duplicates,
  no truth nearby), reported alongside;
- false_discoveries: significant findings (placebo only; every one is false);
- effect_error: reported effect_car - true_effect, over matches. effect_car is the raw
  mean event CAR, signed as computed, so a negative planted effect gives a negative
  effect_car whatever Finding.direction says (direction is a separate field).
  true_effect is the mean injected_car over the finding's matched measurable targets
  (direction * effect if the truth has none) plus, when the truth has a baseline, their mean
  baseline_car (the CAR they had before planting); nominal_effect_error:
  effect_car - direction * effect, alongside.
  effect_error_sum and effect_error_abs_sum (0 with no matches) let statistics pool
  the error over all matches rather than average per-universe means.

Two scoring rules (both reported when evaluate() gets a UniformCorrection):
- primary: each finding's own passed_correction, i.e. the system's own family
  (R0: its 3 hypotheses; the harness: its budget; R3-U: none);
- secondary (*_uniform columns): one rule for every system. The evaluator recomputes
  Holm (harness.corrections.holm_adjust) over the session's findings' p_value_raw, in a
  family of max(backtest_budget, n findings) with the unreported slots at p = 1, and
  passed = p_corrected <= alpha (configs/default.yaml). The evaluator only sees reported
  findings, so a session that ran (and hid) more significant tests is judged as if it
  had not: the uniform rule is a lower bound on the true multiplicity, never stricter.
"""

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from agent.finding import Finding
from evaluation.matching import Match, MatchResult, match_findings, sign
from evaluation.universe_record import UniverseRecord
from harness.corrections import holm_adjust

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

GROUPS = ["overall", "strength", "complexity", "feature_type", "event_type", "direction"]
EFFICIENCY_FIELDS = ["backtests_used", "tool_calls", "tokens", "cost_usd"]
# Split of tokens, from the session metadata's token_breakdown when it has one.
TOKEN_FIELDS = ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                "output_tokens"]


class UniformCorrection(BaseModel):
    """The secondary scoring rule: Holm over a budget-sized family at alpha."""

    budget: int
    alpha: float


def uniform_correction(budget: int | None = None,
                       path: Path = DEFAULT_CONFIG_PATH) -> UniformCorrection:
    """backtest_budget (unless given) and alpha from configs/default.yaml."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return UniformCorrection(budget=int(cfg["backtest_budget"] if budget is None else budget),
                             alpha=float(cfg["alpha"]))


def recorrect(findings: list[Finding], rule: UniformCorrection) -> list[Finding]:
    """Copies of findings with p_value_corrected and passed_correction from the uniform
    rule (module docstring). A non-finite p_value_raw counts as 1."""
    if not findings:
        return []
    p_raw = [f.p_value_raw if math.isfinite(f.p_value_raw) else 1.0 for f in findings]
    p_raw = [min(max(p, 0.0), 1.0) for p in p_raw]
    adjusted = holm_adjust(p_raw, family_size=max(rule.budget, len(findings)))
    return [f.model_copy(update={"p_value_corrected": a, "passed_correction": a <= rule.alpha})
            for f, a in zip(findings, adjusted, strict=True)]


class UniverseResult(BaseModel):
    """One system's findings on one universe, scored."""

    system: str
    record: UniverseRecord
    findings: list[Finding]
    match: MatchResult  # primary: the system's own passed_correction
    match_uniform: MatchResult | None = None  # secondary: the uniform re-correction
    metadata: dict[str, Any] = {}  # session metadata, e.g. tool_calls, tokens, cost_usd


def evaluate(
    system: str,
    record: UniverseRecord,
    findings: list[Finding],
    jaccard_min: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    uniform: UniformCorrection | None = None,
) -> UniverseResult:
    """Score findings; with uniform, also under the uniform re-correction."""
    return UniverseResult(
        system=system,
        record=record,
        findings=findings,
        match=match_findings(findings, record, jaccard_min),
        match_uniform=(None if uniform is None
                       else match_findings(recorrect(findings, uniform), record, jaccard_min)),
        metadata=dict(metadata or {}),
    )


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def effect_errors(matches: list[Match]) -> dict[str, float]:
    """Mean and mean absolute effect_error and nominal_effect_error; NaN with no matches."""
    out = {}
    for name in ("effect_error", "nominal_effect_error"):
        errors = np.array([getattr(m, name) for m in matches], dtype=float)
        out[f"{name}_mean"] = float(errors.mean()) if len(errors) else float("nan")
        out[f"{name}_mean_abs"] = float(np.abs(errors).mean()) if len(errors) else float("nan")
    return out


def precision(match: MatchResult) -> dict[str, float | int]:
    """Matched / significant findings, and the significant findings left unmatched."""
    n_significant = match.n_significant
    n_matched = len(match.matches)
    return {
        "precision": _ratio(n_matched, n_significant),
        "n_unmatched": n_significant - n_matched,
    }


def efficiency(findings: list[Finding], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """EFFICIENCY_FIELDS found in the session metadata, else on the findings.

    Fields that are absent are left out. On findings, a field is read as a
    session-level value: the first one that is not None. TOKEN_FIELDS come from
    metadata["token_breakdown"], when it is there.
    """
    out = {}
    for field in EFFICIENCY_FIELDS:
        value = metadata.get(field)
        if value is None:
            value = next(
                (v for f in findings if (v := getattr(f, field, None)) is not None), None
            )
        if value is not None:
            out[field] = value
    breakdown = metadata.get("token_breakdown") or {}
    out.update({field: breakdown[field] for field in TOKEN_FIELDS if field in breakdown})
    return out


def universe_row(result: UniverseResult) -> dict[str, Any]:
    match = result.match
    planted = match.universe_type == "planted"
    nan = float("nan")
    row = {
        "system": result.system,
        "universe_id": match.universe_id,
        "universe_type": match.universe_type,
        "seed": result.record.seed,
        "n_findings": len(match.outcomes),
        "n_significant": match.n_significant,
        "n_truths": match.n_truths,
        "n_matched": len(match.matches),
        "recall": _ratio(len(match.matches), match.n_truths) if planted else nan,
        "precision": nan,
        "n_unmatched": nan,
        "n_near_miss": len(match.with_status("near_miss")),
        "n_duplicate": len(match.with_status("duplicate")),
        "false_discoveries": (
            len(match.with_status("false_discovery"))
            if match.universe_type == "placebo" else nan
        ),
        **effect_errors(match.matches),
        "effect_error_sum": float(sum(m.effect_error for m in match.matches)),
        "effect_error_abs_sum": float(sum(abs(m.effect_error) for m in match.matches)),
    }
    if planted:
        row.update(precision(match))
    row.update(uniform_columns(result))
    row.update(efficiency(result.findings, result.metadata))
    return row


UNIFORM_COLUMNS = ["n_significant_uniform", "n_matched_uniform", "recall_uniform",
                   "precision_uniform", "false_discoveries_uniform"]


def uniform_columns(result: UniverseResult) -> dict[str, float]:
    """The secondary rule's counts and rates; NaN when it was not computed."""
    out = dict.fromkeys(UNIFORM_COLUMNS, float("nan"))
    match = result.match_uniform
    if match is None:
        return out
    out["n_significant_uniform"] = match.n_significant
    out["n_matched_uniform"] = len(match.matches)
    if match.universe_type == "planted":
        out["recall_uniform"] = _ratio(len(match.matches), match.n_truths)
        out["precision_uniform"] = _ratio(len(match.matches), match.n_significant)
    if match.universe_type == "placebo":
        out["false_discoveries_uniform"] = len(match.with_status("false_discovery"))
    return out


def _truth_labels(truth) -> dict[str, str]:
    spec = truth.spec
    return {
        "overall": "all",
        "strength": spec.strength,
        "complexity": spec.complexity,
        "feature_type": spec.feature_type,
        "event_type": spec.event_type,
        "direction": "up" if spec.direction > 0 else "down",
    }


TRUTH_COLUMNS = ["system", "universe_id", "spec_id", *GROUPS[1:], "snr", "matched",
                 "matched_uniform"]


def _matched_uniform(result: UniverseResult) -> set[str] | None:
    if result.match_uniform is None:
        return None
    return {m.spec_id for m in result.match_uniform.matches}


def truth_rows(results: Iterable[UniverseResult]) -> pd.DataFrame:
    """One row per (system, universe_id, planted truth): group labels, snr (NaN if
    unknown) and matched."""
    rows = []
    for result in results:
        if result.match.universe_type != "planted":
            continue
        matched = {m.spec_id for m in result.match.matches}
        uniform = _matched_uniform(result)
        for truth in result.record.truths:
            labels = _truth_labels(truth)
            rows.append({
                "system": result.system, "universe_id": result.match.universe_id,
                "spec_id": truth.spec.spec_id, **{g: labels[g] for g in GROUPS[1:]},
                "snr": float("nan") if truth.snr is None else truth.snr,
                "matched": truth.spec.spec_id in matched,
                "matched_uniform": (None if uniform is None
                                    else truth.spec.spec_id in uniform),
            })
    return pd.DataFrame(rows, columns=TRUTH_COLUMNS)


def recall_breakdowns(results: Iterable[UniverseResult]) -> pd.DataFrame:
    """Planted recall per system, pooled over universes, overall and by each of GROUPS."""
    rows = []
    for result in results:
        if result.match.universe_type != "planted":
            continue
        matched = {m.spec_id for m in result.match.matches}
        uniform = _matched_uniform(result)
        for truth in result.record.truths:
            labels = _truth_labels(truth)
            for group in GROUPS:
                rows.append({
                    "system": result.system, "group": group, "value": labels[group],
                    "matched": truth.spec.spec_id in matched,
                    "matched_uniform": (float("nan") if uniform is None
                                        else float(truth.spec.spec_id in uniform)),
                })
    columns = ["system", "group", "value", "n_truths", "n_matched", "recall",
               "n_matched_uniform", "recall_uniform"]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    keys = ["system", "group", "value"]
    out = (
        df.groupby(keys, sort=False)["matched"]
        .agg(n_truths="size", n_matched="sum")
        .reset_index()
    )
    out["n_matched"] = out["n_matched"].astype(int)
    out["recall"] = out["n_matched"] / out["n_truths"]
    # NaN when any result in the group lacks the uniform rule.
    uniform = (df.groupby(keys, sort=False)["matched_uniform"]
               .agg(lambda v: v.sum() if v.notna().all() else float("nan"))
               .rename("n_matched_uniform").reset_index())
    out = out.merge(uniform, on=keys, how="left")
    out["recall_uniform"] = out["n_matched_uniform"] / out["n_truths"]
    return out[columns]


def placebo_summary(results: Iterable[UniverseResult]) -> pd.DataFrame:
    """Per system: placebo universes, share with >= 1 false discovery, mean count."""
    rows = [
        {"system": r.system, "false_discoveries": len(r.match.with_status("false_discovery"))}
        for r in results
        if r.match.universe_type == "placebo"
    ]
    columns = ["system", "n_universes", "share_with_false_discovery", "mean_false_discoveries"]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    return (
        df.groupby("system", sort=False)["false_discoveries"]
        .agg(
            n_universes="size",
            share_with_false_discovery=lambda s: float((s >= 1).mean()),
            mean_false_discoveries="mean",
        )
        .reset_index()[columns]
    )


class MetricsTables(NamedTuple):
    per_universe: pd.DataFrame  # one row per (system, universe_id)
    breakdowns: pd.DataFrame  # recall by group, per system
    truths: pd.DataFrame  # one row per (system, universe_id, planted truth), TRUTH_COLUMNS


def metrics_table(results: list[UniverseResult]) -> MetricsTables:
    keys = [(r.system, r.match.universe_id) for r in results]
    if len(keys) != len(set(keys)):
        raise ValueError("more than one result for the same (system, universe_id)")
    per_universe = pd.DataFrame([universe_row(r) for r in results])
    return MetricsTables(per_universe, recall_breakdowns(results), truth_rows(results))


def _car_frame(values, column: str) -> pd.DataFrame:
    """finding_id -> CAR from Findings, dicts, a {finding_id: car} mapping or a DataFrame.

    Dicts and DataFrames give the CAR as effect_car or car.
    """
    if isinstance(values, pd.DataFrame):
        df = values
    elif isinstance(values, Mapping):
        df = pd.DataFrame({"finding_id": list(values), "effect_car": list(values.values())})
    else:
        df = pd.DataFrame(
            [v.model_dump(include={"finding_id", "effect_car"}) if isinstance(v, BaseModel)
             else dict(v) for v in values]
        )
    if df.empty:
        return pd.DataFrame(columns=["finding_id", column])
    car = "effect_car" if "effect_car" in df.columns else "car"
    return df[["finding_id", car]].rename(columns={car: column})


def retention(in_sample_findings, holdout_results) -> pd.DataFrame:
    """Per in-sample finding: in-sample and holdout CAR, sign agreement, holdout/in-sample.

    holdout_car is NaN for a finding with no holdout result; sign_agree is then NaN
    too. ratio is NaN when the in-sample CAR is zero.
    """
    df = _car_frame(in_sample_findings, "in_sample_car").merge(
        _car_frame(holdout_results, "holdout_car"), on="finding_id", how="left"
    )
    ins = df["in_sample_car"].astype(float)
    out = df["holdout_car"].astype(float)
    df["in_sample_car"], df["holdout_car"] = ins, out
    agree = [sign(a) == sign(b) for a, b in zip(ins, out, strict=True)]
    df["sign_agree"] = pd.Series(agree, index=df.index, dtype=float).where(out.notna())
    df["ratio"] = (out / ins.where(ins != 0)).astype(float)
    return df


def retention_summary(table: pd.DataFrame) -> dict[str, float | int]:
    """Over findings with a holdout result: share agreeing in sign, mean and median ratio."""
    scored = table[table["holdout_car"].notna()]
    return {
        "n_findings": len(table),
        "n_with_holdout": len(scored),
        "sign_agreement": float(scored["sign_agree"].mean()) if len(scored) else float("nan"),
        "mean_ratio": float(scored["ratio"].mean()) if len(scored) else float("nan"),
        "median_ratio": float(scored["ratio"].median()) if len(scored) else float("nan"),
    }
