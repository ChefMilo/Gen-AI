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
STRENGTH_ORDER = ["very_weak", "weak", "moderate", "strong"]
SNR_BINS = [(1.0, "<1"), (2.0, "1-2"), (3.0, "2-3"), (4.0, "3-4"), (float("inf"), ">=4")]
SNR_BIN_ORDER = [label for _, label in SNR_BINS]
CURVE_GROUPS = ["strength", "feature_type", "complexity", "snr_bin"]
_VALUE_ORDER = {"strength": STRENGTH_ORDER, "snr_bin": SNR_BIN_ORDER}


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


# metric -> (universe type it is computed on, pooled statistic over per_universe rows)
POOLED_METRICS: dict[str, tuple[str, PooledRatio]] = {
    "recall": ("planted", PooledRatio("n_matched", "n_truths")),
    "precision": ("planted", PooledRatio("n_matched", "n_significant")),
    "effect_error_mean": ("planted", PooledRatio("effect_error_sum", "n_matched")),
    "effect_error_mean_abs": ("planted", PooledRatio("effect_error_abs_sum", "n_matched")),
    "mean_false_discoveries": ("placebo", PooledRatio("false_discoveries", _ones)),
    "share_with_false_discovery": ("placebo", PooledRatio(_has_false_discovery, _ones)),
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
            ci = bootstrap_ci(_rows(tables, system, universe_type), statistic,
                              n_boot=n_boot, alpha=alpha, seed=seed)
            row.update({metric: ci["estimate"], f"{metric}_lo": ci["lo"],
                        f"{metric}_hi": ci["hi"]})
        rows.append(row)
    return pd.DataFrame(rows)


def snr_bin(snr: float) -> str | None:
    """SNR_BINS label of snr (upper bound exclusive); None for NaN."""
    if snr is None or np.isnan(snr):
        return None
    return next(label for upper, label in SNR_BINS if snr < upper)


def detection_curve_data(
    tables: MetricsTables,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Recall with CI per (system, group, value) for group in CURVE_GROUPS.

    snr_bin bins the truths' snr (SNR_BINS); truths with no snr are left out of it.
    Each system resamples its planted universe ids (all of them, so a universe with no
    truth in a group still counts as drawn). Rows are sorted by system, group, then
    value, with strength in STRENGTH_ORDER and snr_bin in SNR_BIN_ORDER.
    """
    columns = ["system", "group", "value", "n_truths", "n_matched", "recall", "lo", "hi",
               "n_universes"]
    truths = tables.truths
    snr = truths["snr"] if "snr" in truths.columns else pd.Series(np.nan, index=truths.index)
    truths = truths.assign(snr_bin=snr.astype(float).map(snr_bin))
    out = []
    for system in _systems(tables, None):
        planted = _rows(tables, system, "planted")
        ids = sorted(planted["universe_id"].unique()) if not planted.empty else []
        mine = truths[truths["system"] == system]
        for group in CURVE_GROUPS:
            counts = (
                mine.assign(matched=mine["matched"].astype(int))
                .groupby([group, "universe_id"])["matched"]
                .agg(n_truths="size", n_matched="sum")
                .reset_index()
            )
            for value, per_universe in counts.groupby(group):
                per_universe = (per_universe.set_index("universe_id")[["n_truths", "n_matched"]]
                                .reindex(ids, fill_value=0))
                ci = bootstrap_ci(per_universe.reset_index(names="universe_id"),
                                  PooledRatio("n_matched", "n_truths"),
                                  n_boot=n_boot, alpha=alpha, seed=seed)
                out.append({
                    "system": system, "group": group, "value": value,
                    "n_truths": int(per_universe["n_truths"].sum()),
                    "n_matched": int(per_universe["n_matched"].sum()),
                    "recall": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"],
                    "n_universes": ci["n_universes"],
                })
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
