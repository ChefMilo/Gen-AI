"""Bootstrap confidence intervals and paired system comparisons (docs/design.md section 3,
"Statistics").

Inputs are evaluation.metrics.metrics_table outputs: per_universe (one row per
(system, universe_id)) and truths (one row per planted truth).

Universes are the unit of resampling: bootstrap_ci draws universe ids with
replacement and evaluates the statistic on all rows of the drawn universes (a universe
drawn twice counts twice). Metrics are pooled over universes, e.g. recall =
sum(matched truths) / sum(truths), not the mean of per-universe ratios. CIs are
percentile intervals; replicates where a pooled ratio has a zero denominator are left
out of the percentiles.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from evaluation.metrics import MetricsTables

DEFAULT_N_BOOT = 10_000
DEFAULT_ALPHA = 0.05
STRENGTH_ORDER = ["very_weak", "weak", "moderate", "strong", "very_strong"]
SNR_BINS = [(2.0, "<2"), (4.0, "2-4"), (8.0, "4-8"), (16.0, "8-16"), (float("inf"), ">=16")]
SNR_BIN_ORDER = [label for _, label in SNR_BINS]
CURVE_GROUPS = ["strength", "feature_type", "complexity", "snr_bin"]
COMPARISON_GROUPS = ["feature_type", "strength", "complexity"]
COMPLEXITY_ORDER = ["single", "conjunction"]
FEATURE_TYPE_ORDER = ["structured", "text"]
_VALUE_ORDER = {"strength": STRENGTH_ORDER, "snr_bin": SNR_BIN_ORDER,
                "complexity": COMPLEXITY_ORDER, "feature_type": FEATURE_TYPE_ORDER}
# Recall metric -> the truths column it counts.
_RECALL_COLUMNS = {"recall": "matched", "recall_uniform": "matched_uniform"}


@dataclass(frozen=True)
class PooledRatio:
    """sum(numerator) / sum(denominator) over rows; NaN when the denominator sums to 0.

    numerator and denominator are column names, or functions of the rows returning a
    Series. bootstrap_ci evaluates it on all replicates at once instead of row by row.
    """

    numerator: str | Callable[[pd.DataFrame], pd.Series]
    denominator: str | Callable[[pd.DataFrame], pd.Series]

    def _column(self, df: pd.DataFrame, which) -> np.ndarray:
        values = df[which] if isinstance(which, str) else which(df)
        return np.asarray(values, dtype=float)

    def parts(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self._column(df, self.numerator), self._column(df, self.denominator)

    def __call__(self, df: pd.DataFrame) -> float:
        num, den = self.parts(df)
        return float(num.sum() / den.sum()) if den.sum() else float("nan")


def _ones(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def _has_false_discovery(df: pd.DataFrame) -> pd.Series:
    return (df["false_discoveries"] >= 1).astype(float)


def _has_false_discovery_uniform(df: pd.DataFrame) -> pd.Series:
    """1 or 0 per universe; NaN where the uniform rule was not computed."""
    count = df["false_discoveries_uniform"].astype(float)
    return (count >= 1).astype(float).where(count.notna())


# metric -> (universe type it is computed on, pooled statistic over per_universe rows)
POOLED_METRICS: dict[str, tuple[str, PooledRatio]] = {
    "recall": ("planted", PooledRatio("n_matched", "n_truths")),
    "precision": ("planted", PooledRatio("n_matched", "n_significant")),
    "effect_error_mean": ("planted", PooledRatio("effect_error_sum", "n_matched")),
    "effect_error_mean_abs": ("planted", PooledRatio("effect_error_abs_sum", "n_matched")),
    "mean_false_discoveries": ("placebo", PooledRatio("false_discoveries", _ones)),
    "share_with_false_discovery": ("placebo", PooledRatio(_has_false_discovery, _ones)),
    # False discoveries over the backtests the sessions ran; NaN for a system that ran none.
    "false_discoveries_per_test": ("placebo",
                                   PooledRatio("false_discoveries", "backtests_used")),
    # Secondary rule (evaluation.metrics, uniform re-correction); NaN when not computed.
    "recall_uniform": ("planted", PooledRatio("n_matched_uniform", "n_truths")),
    "precision_uniform": ("planted",
                          PooledRatio("n_matched_uniform", "n_significant_uniform")),
    "mean_false_discoveries_uniform": ("placebo",
                                       PooledRatio("false_discoveries_uniform", _ones)),
    "share_with_false_discovery_uniform": ("placebo",
                                           PooledRatio(_has_false_discovery_uniform, _ones)),
    "false_discoveries_per_test_uniform": ("placebo",
                                           PooledRatio("false_discoveries_uniform",
                                                       "backtests_used")),
}


def _draws(n_ids: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_ids) positions of universe ids drawn with replacement."""
    return np.random.default_rng(seed).integers(0, n_ids, size=(n_boot, n_ids))


def _counts(draws: np.ndarray, n_ids: int) -> np.ndarray:
    """(n_boot, n_ids) how often each universe id was drawn in each replicate."""
    counts = np.zeros((draws.shape[0], n_ids))
    np.add.at(counts, (np.arange(draws.shape[0])[:, None], draws), 1)
    return counts


def _per_id_sums(df: pd.DataFrame, statistic: PooledRatio, by: str, ids: np.ndarray):
    """Numerator and denominator summed per universe id, in the order of ids."""
    num, den = statistic.parts(df)
    sums = pd.DataFrame({"id": df[by].to_numpy(), "num": num, "den": den})
    sums = sums.groupby("id").sum().reindex(ids, fill_value=0.0)
    return sums["num"].to_numpy(), sums["den"].to_numpy()


def _ratio_replicates(counts: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    top, bottom = counts @ num, counts @ den
    out = np.full(len(counts), np.nan)
    np.divide(top, bottom, out=out, where=bottom != 0)
    return out


def _percentiles(replicates: np.ndarray, alpha: float) -> tuple[float, float]:
    finite = replicates[~np.isnan(replicates)]
    if not len(finite):
        return float("nan"), float("nan")
    lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_ci(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    by: str = "universe_id",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> dict[str, float | int]:
    """estimate, percentile CI (lo, hi) and n_universes, resampling values of `by`.

    statistic(rows) -> float. A PooledRatio is evaluated on all replicates at once;
    any other function is called once per replicate on the resampled rows. Both give
    the same result for the same seed.
    """
    ids = np.array(sorted(df[by].unique()))
    if not len(ids):
        nan = float("nan")
        return {"estimate": nan, "lo": nan, "hi": nan, "n_universes": 0}
    draws = _draws(len(ids), n_boot, seed)
    if isinstance(statistic, PooledRatio):
        num, den = _per_id_sums(df, statistic, by, ids)
        replicates = _ratio_replicates(_counts(draws, len(ids)), num, den)
    else:
        positions = df.groupby(by).indices
        rows = [positions[i] for i in ids]
        replicates = np.array([
            statistic(df.iloc[np.concatenate([rows[k] for k in draw])]) for draw in draws
        ], dtype=float)
    lo, hi = _percentiles(replicates, alpha)
    return {"estimate": float(statistic(df)), "lo": lo, "hi": hi, "n_universes": len(ids)}


def _rows(tables: MetricsTables, system: str, universe_type: str | None = None):
    df = tables.per_universe
    if df.empty:
        return df
    df = df[df["system"] == system]
    return df if universe_type is None else df[df["universe_type"] == universe_type]


def _systems(tables: MetricsTables, systems: Sequence[str] | None) -> list[str]:
    if systems is not None:
        return list(systems)
    df = tables.per_universe
    return [] if df.empty else list(dict.fromkeys(df["system"]))


def _available(df: pd.DataFrame, statistic: PooledRatio) -> bool:
    """False if a column the statistic reads is missing or all NaN (e.g. the uniform
    rule was not computed), so it reports NaN instead of a ratio of empty sums."""
    if df.empty:
        return True
    try:
        numerator, denominator = statistic.parts(df)
    except KeyError:
        return False
    return not (np.isnan(numerator).all() or np.isnan(denominator).all())


def system_summary(
    tables: MetricsTables,
    systems: Sequence[str] | None = None,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """One row per system: n_universes per type, and each POOLED_METRICS metric with
    its CI (columns <metric>, <metric>_lo, <metric>_hi). systems defaults to all, in
    order of appearance."""
    rows = []
    for system in _systems(tables, systems):
        row = {"system": system}
        for universe_type in ("planted", "placebo", "real"):
            rows_t = _rows(tables, system, universe_type)
            row[f"n_{universe_type}"] = 0 if rows_t.empty else rows_t["universe_id"].nunique()
        for metric, (universe_type, statistic) in POOLED_METRICS.items():
            df = _rows(tables, system, universe_type)
            if not _available(df, statistic):
                nan = float("nan")
                row.update({metric: nan, f"{metric}_lo": nan, f"{metric}_hi": nan})
                continue
            ci = bootstrap_ci(df, statistic, n_boot=n_boot, alpha=alpha, seed=seed)
            row.update({metric: ci["estimate"], f"{metric}_lo": ci["lo"],
                        f"{metric}_hi": ci["hi"]})
        rows.append(row)
    return pd.DataFrame(rows)


def snr_bin(snr: float) -> str | None:
    """SNR_BINS label of snr (upper bound exclusive); None for NaN."""
    if snr is None or np.isnan(snr):
        return None
    return next(label for upper, label in SNR_BINS if snr < upper)


def _matched_float(truths: pd.DataFrame, column: str) -> pd.Series:
    """A truths matched column as 0.0/1.0, NaN where it was not computed (or is absent)."""
    if column not in truths.columns:
        return pd.Series(np.nan, index=truths.index, dtype=float)
    return truths[column].astype(float)


def _per_universe_counts(truths: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """n_truths, n_matched and n_matched_uniform per (keys, universe_id). n_matched_uniform
    is NaN for a universe where the uniform rule was not computed."""
    rows = truths.assign(_own=_matched_float(truths, "matched"),
                         _uniform=_matched_float(truths, "matched_uniform"))
    grouped = rows.groupby([*keys, "universe_id"], sort=False)
    out = grouped.agg(n_truths=("_own", "size"), n_matched=("_own", "sum"))
    out["n_matched_uniform"] = grouped["_uniform"].sum(min_count=1)
    return out.reset_index()


def _recall_rows(
    truths: pd.DataFrame, keys: list[str], ids: list[str], seed: int, n_boot: int, alpha: float,
) -> list[dict]:
    """Recall with CI, own and uniform, per value of keys, resampling all of ids."""
    counts = _per_universe_counts(truths, keys)
    out = []
    for values, group in counts.groupby(keys):
        per_universe = (group.set_index("universe_id")[["n_truths", "n_matched",
                                                        "n_matched_uniform"]]
                        .reindex(ids, fill_value=0))
        frame = per_universe.reset_index(names="universe_id")
        own = bootstrap_ci(frame, PooledRatio("n_matched", "n_truths"),
                           n_boot=n_boot, alpha=alpha, seed=seed)
        nan = float("nan")
        uniform = {"estimate": nan, "lo": nan, "hi": nan}
        computed = group["n_matched_uniform"].notna().all()
        if computed:
            uniform = bootstrap_ci(frame, PooledRatio("n_matched_uniform", "n_truths"),
                                   n_boot=n_boot, alpha=alpha, seed=seed)
        out.append({
            **dict(zip(keys, values, strict=True)),
            "n_truths": int(per_universe["n_truths"].sum()),
            "n_matched": int(per_universe["n_matched"].sum()),
            "recall": own["estimate"], "lo": own["lo"], "hi": own["hi"],
            "n_universes": own["n_universes"],
            "n_matched_uniform": (per_universe["n_matched_uniform"].sum() if computed else nan),
            "recall_uniform": uniform["estimate"],
            "lo_uniform": uniform["lo"], "hi_uniform": uniform["hi"],
        })
    return out


RECALL_COLUMNS = ["n_truths", "n_matched", "recall", "lo", "hi", "n_universes",
                  "n_matched_uniform", "recall_uniform", "lo_uniform", "hi_uniform"]
CELL_KEYS = ["complexity", "feature_type", "strength"]


def _planted_ids(tables: MetricsTables, system: str) -> list[str]:
    planted = _rows(tables, system, "planted")
    return sorted(planted["universe_id"].unique()) if not planted.empty else []


def detection_cells_data(
    tables: MetricsTables,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Recall with CI per (system, complexity, feature_type, strength), own and uniform,
    resampling as detection_curve_data does. Sorted by system, then each key's order."""
    columns = ["system", *CELL_KEYS, *RECALL_COLUMNS]
    out = []
    for system in _systems(tables, None):
        mine = tables.truths[tables.truths["system"] == system]
        rows = _recall_rows(mine, CELL_KEYS, _planted_ids(tables, system), seed, n_boot, alpha)
        out.extend({"system": system, **row} for row in rows)
    df = pd.DataFrame(out, columns=columns)
    if df.empty:
        return df
    ranks = {k: df[k].map({v: i for i, v in enumerate(_VALUE_ORDER[k])}) for k in CELL_KEYS}
    order = df.assign(**{f"_{k}": r for k, r in ranks.items()})
    return (order.sort_values(["system", *[f"_{k}" for k in CELL_KEYS]])
            [columns].reset_index(drop=True))


def detection_curve_data(
    tables: MetricsTables,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Recall with CI per (system, group, value) for group in CURVE_GROUPS, under the
    system's own rule (recall, lo, hi) and the uniform rule (*_uniform; NaN if not computed).

    snr_bin bins the truths' snr (SNR_BINS); truths with no snr are left out of it.
    Each system resamples its planted universe ids (all of them, so a universe with no
    truth in a group still counts as drawn). Rows are sorted by system, group, then
    value, with strength in STRENGTH_ORDER and snr_bin in SNR_BIN_ORDER.
    """
    columns = ["system", "group", "value", *RECALL_COLUMNS]
    truths = tables.truths
    snr = truths["snr"] if "snr" in truths.columns else pd.Series(np.nan, index=truths.index)
    truths = truths.assign(snr_bin=snr.astype(float).map(snr_bin))
    out = []
    for system in _systems(tables, None):
        ids = _planted_ids(tables, system)
        mine = truths[truths["system"] == system]
        for group in CURVE_GROUPS:
            for row in _recall_rows(mine, [group], ids, seed, n_boot, alpha):
                value = row.pop(group)
                out.append({"system": system, "group": group, "value": value, **row})
    df = pd.DataFrame(out, columns=columns)
    if df.empty:
        return df
    ranks = {g: {v: i for i, v in enumerate(vs)} for g, vs in _VALUE_ORDER.items()}
    order = [ranks[g].get(v, len(ranks[g])) if g in ranks else 0
             for g, v in zip(df["group"], df["value"], strict=True)]
    group_rank = df["group"].map({g: i for i, g in enumerate(CURVE_GROUPS)})
    df = df.assign(_g=group_rank, _o=order).sort_values(["system", "_g", "_o", "value"])
    return df.drop(columns=["_g", "_o"]).reset_index(drop=True)


@dataclass
class PairedComparison:
    system_a: str
    system_b: str
    metric: str
    estimate: float  # pooled metric of a minus pooled metric of b
    lo: float
    hi: float
    n_universes: int
    differences: pd.DataFrame  # universe_id, a, b, diff (per-universe metric values)
    n_pairs: int  # universes with a defined difference, used by the test
    test: str | None  # "wilcoxon", "sign", or None with no pairs
    p_value: float


def _signed_rank_test(diffs: np.ndarray) -> tuple[str | None, float]:
    """Wilcoxon signed-rank p-value; exact sign test (p = 1) when every difference is 0."""
    if not len(diffs):
        return None, float("nan")
    if not np.any(diffs):
        # No positive or negative differences: the sign test sees 0 successes in 0
        # trials, which is no evidence of a difference.
        return "sign", 1.0
    return "wilcoxon", float(sp_stats.wilcoxon(diffs).pvalue)


def paired_comparison(
    tables: MetricsTables,
    system_a: str,
    system_b: str,
    metric: str,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> PairedComparison:
    """system_a minus system_b on one POOLED_METRICS metric, over the same universes.

    Both systems must have run on the same universe ids (any type), each with the same
    universe type. The bootstrap draws universe ids once per replicate and applies the
    draw to both systems. The signed-rank test uses per-universe metric values; a
    universe where either is undefined (e.g. precision with no significant findings)
    is left out of the test but stays in the pooled estimate.
    """
    if metric not in POOLED_METRICS:
        raise ValueError(f"unknown metric {metric!r}; one of {sorted(POOLED_METRICS)}")
    universe_type, statistic = POOLED_METRICS[metric]
    all_a, all_b = _rows(tables, system_a), _rows(tables, system_b)
    types_a = dict(zip(all_a.get("universe_id", []), all_a.get("universe_type", []),
                       strict=True))
    types_b = dict(zip(all_b.get("universe_id", []), all_b.get("universe_type", []),
                       strict=True))
    if types_a.keys() != types_b.keys():
        only_a = sorted(types_a.keys() - types_b.keys())
        only_b = sorted(types_b.keys() - types_a.keys())
        raise ValueError(
            f"{system_a} and {system_b} ran on different universes: "
            f"only {system_a}: {only_a}; only {system_b}: {only_b}"
        )
    differing = sorted(u for u in types_a if types_a[u] != types_b[u])
    if differing:
        raise ValueError(f"universe types differ between systems for: {differing}")

    ids = np.array(sorted(u for u, t in types_a.items() if t == universe_type))
    a, b = all_a[all_a["universe_id"].isin(ids)], all_b[all_b["universe_id"].isin(ids)]
    if not (_available(a, statistic) and _available(b, statistic)):
        nan = float("nan")
        return PairedComparison(
            system_a=system_a, system_b=system_b, metric=metric, estimate=nan, lo=nan, hi=nan,
            n_universes=len(ids), n_pairs=0, test=None, p_value=nan,
            differences=pd.DataFrame(columns=["universe_id", "a", "b", "diff"]),
        )
    per_universe = pd.DataFrame({
        "universe_id": ids,
        "a": [statistic(a[a["universe_id"] == u]) for u in ids],
        "b": [statistic(b[b["universe_id"] == u]) for u in ids],
    })
    per_universe["diff"] = per_universe["a"] - per_universe["b"]
    diffs = per_universe["diff"].dropna().to_numpy()
    test, p_value = _signed_rank_test(diffs)

    nan = float("nan")
    estimate, lo, hi = nan, nan, nan
    if len(ids):
        counts = _counts(_draws(len(ids), n_boot, seed), len(ids))
        rep_a = _ratio_replicates(counts, *_per_id_sums(a, statistic, "universe_id", ids))
        rep_b = _ratio_replicates(counts, *_per_id_sums(b, statistic, "universe_id", ids))
        estimate = statistic(a) - statistic(b)
        lo, hi = _percentiles(rep_a - rep_b, alpha)
    return PairedComparison(
        system_a=system_a, system_b=system_b, metric=metric, estimate=estimate, lo=lo,
        hi=hi, n_universes=len(ids), differences=per_universe, n_pairs=len(diffs),
        test=test, p_value=p_value,
    )


COMPARISON_COLUMNS = ["system_a", "system_b", "metric", "estimate", "lo", "hi",
                      "n_universes", "n_pairs", "test", "p_value"]


def compare_systems(
    tables: MetricsTables,
    systems: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """paired_comparison for every pair of systems (in order) and metric, one row each.

    Each pair is compared on the universes both systems have rows for, so a pair that
    failed for one system (evaluation.run) drops that universe from that comparison
    only. metrics defaults to every POOLED_METRICS metric.
    """
    names = _systems(tables, systems)
    metrics = list(POOLED_METRICS) if metrics is None else list(metrics)
    df = tables.per_universe
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = set(df.loc[df["system"] == a, "universe_id"]) & set(
                df.loc[df["system"] == b, "universe_id"])
            pair = tables._replace(per_universe=df[df["system"].isin([a, b])
                                                   & df["universe_id"].isin(common)])
            for metric in metrics:
                c = paired_comparison(pair, a, b, metric, seed=seed, n_boot=n_boot,
                                      alpha=alpha)
                rows.append({"system_a": a, "system_b": b, "metric": metric,
                             "estimate": c.estimate, "lo": c.lo, "hi": c.hi,
                             "n_universes": c.n_universes, "n_pairs": c.n_pairs,
                             "test": c.test, "p_value": c.p_value})
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _group_values(truths: pd.DataFrame, group: str) -> list[str]:
    """The group's values in the truths, in _VALUE_ORDER where there is one."""
    present = set(truths[group].dropna())
    order = _VALUE_ORDER.get(group, [])
    return [v for v in order if v in present] + sorted(present - set(order))


def _system_pairs(tables: MetricsTables, systems: Sequence[str] | None) -> list[tuple[str, str]]:
    names = _systems(tables, systems)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]


def group_comparison(
    tables: MetricsTables,
    system_a: str,
    system_b: str,
    group: str,
    value: str,
    metric: str = "recall",
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> PairedComparison:
    """paired_comparison of recall on one group's truths (e.g. feature_type == "text").

    Truths are aggregated per system per universe, over only the universes where both
    systems have truths of that group. On the final plan each planted universe holds one
    feature type, so a feature type's comparison has half the planted universes.
    Bootstrap draws and the Wilcoxon/sign test are paired_comparison's.
    """
    if metric not in _RECALL_COLUMNS:
        raise ValueError(f"unknown metric {metric!r}; one of {sorted(_RECALL_COLUMNS)}")
    truths = tables.truths
    rows = truths[truths[group] == value]
    a, b = rows[rows["system"] == system_a], rows[rows["system"] == system_b]
    common = set(a["universe_id"]) & set(b["universe_id"])
    pair = pd.concat([a, b])
    counts = _per_universe_counts(pair[pair["universe_id"].isin(common)], ["system"])
    counts["universe_type"] = "planted"
    return paired_comparison(tables._replace(per_universe=counts), system_a, system_b, metric,
                             seed=seed, n_boot=n_boot, alpha=alpha)


GROUP_COMPARISON_COLUMNS = ["system_a", "system_b", "group", "value", "metric", "recall_a",
                            "recall_b", "estimate", "lo", "hi", "n_universes", "n_pairs", "test",
                            "p_value"]


def group_comparisons(
    tables: MetricsTables,
    systems: Sequence[str] | None = None,
    groups: Sequence[str] = tuple(COMPARISON_GROUPS),
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """group_comparison per system pair, group value and recall metric, one row each.

    recall_a and recall_b are the systems' pooled recalls over those universes, and
    estimate is recall_a - recall_b.
    """
    rows = []
    for a, b in _system_pairs(tables, systems):
        for group in groups:
            for value in _group_values(tables.truths, group):
                for metric in _RECALL_COLUMNS:
                    c = group_comparison(tables, a, b, group, value, metric, seed=seed,
                                         n_boot=n_boot, alpha=alpha)
                    recall_a, recall_b = _pooled_recalls(tables, a, b, group, value, metric)
                    rows.append({
                        "system_a": a, "system_b": b, "group": group, "value": value,
                        "metric": metric, "recall_a": recall_a, "recall_b": recall_b,
                        "estimate": c.estimate, "lo": c.lo, "hi": c.hi,
                        "n_universes": c.n_universes, "n_pairs": c.n_pairs, "test": c.test,
                        "p_value": c.p_value,
                    })
    return pd.DataFrame(rows, columns=GROUP_COMPARISON_COLUMNS)


def _paired_truths(tables: MetricsTables, a: str, b: str, group: str, value: str,
                   column: str) -> pd.DataFrame:
    """One row per truth both systems scored: universe_id, a, b (0.0/1.0, or NaN if the
    rule was not computed)."""
    truths = tables.truths
    rows = truths[truths[group] == value]
    keys = ["universe_id", "spec_id"]

    def side(system):
        mine = rows[rows["system"] == system]
        return mine[keys].assign(matched=_matched_float(mine, column))

    return side(a).merge(side(b), on=keys, suffixes=("_a", "_b")).rename(
        columns={"matched_a": "a", "matched_b": "b"})


def _pooled_recalls(tables: MetricsTables, a: str, b: str, group: str, value: str,
                    metric: str) -> tuple[float, float]:
    paired = _paired_truths(tables, a, b, group, value, _RECALL_COLUMNS[metric])
    if paired.empty:
        return float("nan"), float("nan")
    return float(paired["a"].mean()), float(paired["b"].mean())


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value: binomial(a_only + b_only, 1/2) on the discordant
    truths. 1 when there are none."""
    n = a_only + b_only
    return float(sp_stats.binomtest(min(a_only, b_only), n, 0.5).pvalue) if n else 1.0


def truth_comparison(
    tables: MetricsTables,
    system_a: str,
    system_b: str,
    group: str,
    value: str,
    metric: str = "recall",
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Truth-level paired test on one group's truths, paired by (universe_id, spec_id).

    Counts truths found by A only, B only, both and neither; gives each system's recall
    and recall_a - recall_b (estimate), each with a percentile CI from one set of universe
    draws (paired_comparison's, so a shared seed gives group_comparison's CI on the
    difference); and the exact McNemar p-value on the discordant truths. McNemar treats
    truths as independent, though truths of one universe share a session, so read it
    with the universe-resampled CI and group_comparison's universe-level test.
    """
    if metric not in _RECALL_COLUMNS:
        raise ValueError(f"unknown metric {metric!r}; one of {sorted(_RECALL_COLUMNS)}")
    paired = _paired_truths(tables, system_a, system_b, group, value, _RECALL_COLUMNS[metric])
    nan = float("nan")
    row = {
        "system_a": system_a, "system_b": system_b, "group": group, "value": value,
        "metric": metric, "n_truths": len(paired), "n_universes": paired["universe_id"].nunique(),
        "a_only": nan, "b_only": nan, "both": nan, "neither": nan,
        "recall_a": nan, "lo_a": nan, "hi_a": nan, "recall_b": nan, "lo_b": nan, "hi_b": nan,
        "estimate": nan, "lo": nan, "hi": nan, "p_value": nan,
    }
    if paired.empty or paired[["a", "b"]].isna().any(axis=None):
        return row
    found_a, found_b = paired["a"] == 1, paired["b"] == 1
    a_only, b_only = int((found_a & ~found_b).sum()), int((~found_a & found_b).sum())
    per_universe = paired.groupby("universe_id").agg(a=("a", "sum"), b=("b", "sum"),
                                                     n=("a", "size")).sort_index()
    counts = _counts(_draws(len(per_universe), n_boot, seed), len(per_universe))
    den = per_universe["n"].to_numpy(dtype=float)
    rep_a = _ratio_replicates(counts, per_universe["a"].to_numpy(dtype=float), den)
    rep_b = _ratio_replicates(counts, per_universe["b"].to_numpy(dtype=float), den)
    recall_a, recall_b = float(paired["a"].mean()), float(paired["b"].mean())
    lo_a, hi_a = _percentiles(rep_a, alpha)
    lo_b, hi_b = _percentiles(rep_b, alpha)
    lo, hi = _percentiles(rep_a - rep_b, alpha)
    row.update({
        "a_only": a_only, "b_only": b_only, "both": int((found_a & found_b).sum()),
        "neither": int((~found_a & ~found_b).sum()),
        "recall_a": recall_a, "lo_a": lo_a, "hi_a": hi_a,
        "recall_b": recall_b, "lo_b": lo_b, "hi_b": hi_b,
        "estimate": recall_a - recall_b, "lo": lo, "hi": hi,
        "p_value": mcnemar_exact(a_only, b_only),
    })
    return row


TRUTH_COMPARISON_COLUMNS = ["system_a", "system_b", "group", "value", "metric", "n_truths",
                            "n_universes", "a_only", "b_only", "both", "neither", "recall_a",
                            "lo_a", "hi_a", "recall_b", "lo_b", "hi_b", "estimate", "lo", "hi",
                            "p_value"]


def truth_comparisons(
    tables: MetricsTables,
    systems: Sequence[str] | None = None,
    groups: Sequence[str] = tuple(COMPARISON_GROUPS),
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """truth_comparison per system pair, group value and recall metric, one row each."""
    rows = [
        truth_comparison(tables, a, b, group, value, metric, seed=seed, n_boot=n_boot,
                         alpha=alpha)
        for a, b in _system_pairs(tables, systems)
        for group in groups
        for value in _group_values(tables.truths, group)
        for metric in _RECALL_COLUMNS
    ]
    return pd.DataFrame(rows, columns=TRUTH_COMPARISON_COLUMNS)


def summary_frames(tables: MetricsTables, seed: int = 0,
                   n_boot: int = DEFAULT_N_BOOT) -> dict[str, pd.DataFrame]:
    """Every statistics table in summary/, by file name without .csv."""
    return {
        "system_summary": system_summary(tables, seed=seed, n_boot=n_boot),
        "detection_curve": detection_curve_data(tables, seed=seed, n_boot=n_boot),
        "detection_cells": detection_cells_data(tables, seed=seed, n_boot=n_boot),
        "comparisons": compare_systems(tables, seed=seed, n_boot=n_boot),
        "group_comparisons": group_comparisons(tables, seed=seed, n_boot=n_boot),
        "truth_comparisons": truth_comparisons(tables, seed=seed, n_boot=n_boot),
    }
