"""Result figures, from summary/ only (plus an optional oracle.csv). No system is run.

Every figure function takes a FigureData (load_figure_data) and rule="own" (each system's
own correction) or "uniform" (the evaluator's re-correction), and returns a
matplotlib.figure.Figure. Figures use no pyplot and no global state, so the app can call
them directly; save with fig.savefig.

A system with no data is skipped, with a note on the figure and a log line. Colours,
markers and order are fixed per system, so a system looks the same in every figure
whichever others ran; stubs and any other system are grey.
"""

import logging
import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from evaluation.stats import (
    COMPLEXITY_ORDER,
    FEATURE_TYPE_ORDER,
    SNR_BIN_ORDER,
    STRENGTH_ORDER,
)
from evaluation.summary import SUMMARY_DIR, SYSTEM_ORDER, ordered_systems, read_summary

log = logging.getLogger(__name__)

RULES = ("own", "uniform")
RULE_LABELS = {"own": "each system's own correction",
               "uniform": "uniform Holm re-correction over the budget"}
TABLES = ["system_summary", "detection_curve", "detection_cells", "comparisons",
          "group_comparisons", "truth_comparisons", "per_universe", "truths"]

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLOURS = {"r0": "#2a78d6", "r1": "#eb6834", "r2": "#1baf7a", "r3": "#eda100", "r3-u": "#e87ba4"}
MARKERS = {"r0": "o", "r1": "s", "r2": "^", "r3": "D", "r3-u": "v", "null": "X", "random": "P"}
OTHER_MARKER = "*"
STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.titleweight": "semibold",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}
SNR_LABELS = {"<2": "<2", "2-4": "2–4", "4-8": "4–8", "8-16": "8–16", ">=16": "≥16"}
STRENGTH_TEXT = {s: s.replace("_", " ") for s in STRENGTH_ORDER}


@dataclass(frozen=True)
class FigureData:
    """The summary tables a figure reads. A table that is absent is an empty frame."""

    tables: dict[str, pd.DataFrame]
    oracle: pd.DataFrame | None = None
    oracle_note: str | None = None

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())


def checked_oracle(oracle: pd.DataFrame | None,
                   truths: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """oracle.csv, or None and a note if it does not cover this plan's planted truths (it
    was computed on another plan, so its findable rates say nothing about these)."""
    if oracle is None or truths.empty:
        return oracle, None
    keys = ["universe_id", "spec_id"]
    wanted = set(map(tuple, truths[keys].to_numpy()))
    have = set(map(tuple, oracle[keys].to_numpy()))
    if wanted <= have:
        return oracle, None
    note = (f"Oracle line left out: oracle.csv lacks {len(wanted - have)} of "
            f"{len(wanted)} planted truths (computed on another plan).")
    log.warning(note)
    return None, note


def load_figure_data(results: str | Path, oracle: str | Path | None = None) -> FigureData:
    """The tables of results/<id>/summary/ and, given its path, an oracle.csv."""
    summary = Path(results) / SUMMARY_DIR
    tables = {name: read_summary(summary, name) for name in TABLES
              if (summary / f"{name}.csv").exists()}
    frame = None
    if oracle is not None:
        path = Path(oracle)
        frame = read_summary(path.parent, path.stem)
    frame, note = checked_oracle(frame, tables.get("truths", pd.DataFrame()))
    return FigureData(tables, frame, note)


# --- small helpers ------------------------------------------------------------------


def _label(system: str) -> str:
    return system.upper() if system in SYSTEM_ORDER else system


def _colour(system: str) -> str:
    return COLOURS.get(system, MUTED)


def _marker(system: str) -> str:
    return MARKERS.get(system, OTHER_MARKER)


def _p(value: float) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _signed(value: float) -> str:
    """value with its sign to 3 places, never "-0.000"."""
    return f"{round(value, 3) + 0.0:+.3f}"


def _interval(row: pd.Series) -> str:
    return f"{_signed(row['estimate'])}, CI [{_signed(row['lo'])}, {_signed(row['hi'])}]"


def _annotate(ax, x: float, y: float, text: str, **kwargs) -> None:
    """Text on an axes, on the surface colour so grid and zero lines do not cross it."""
    ax.text(x, y, text, transform=ax.transAxes, parse_math=False,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}, **kwargs)


def _where(frame: pd.DataFrame, **equals) -> pd.DataFrame:
    """Rows where each column equals the value; empty if a column is absent."""
    if any(column not in frame.columns for column in equals):
        return frame.iloc[0:0]
    for column, value in equals.items():
        frame = frame[frame[column] == value]
    return frame


def _metric(name: str, rule: str) -> str:
    return name if rule == "own" else f"{name}_uniform"


def _curve_columns(rule: str) -> tuple[str, str, str]:
    if rule == "own":
        return "recall", "lo", "hi"
    return "recall_uniform", "lo_uniform", "hi_uniform"


def _present(frame: pd.DataFrame) -> list[str]:
    return ordered_systems(frame["system"]) if "system" in frame.columns else []


def _missing(figure: str, present: list[str], notes: list[str],
             expected: list[str] | None = None) -> None:
    gone = [s for s in (expected or SYSTEM_ORDER) if s not in present]
    if gone:
        names = ", ".join(_label(s) for s in gone)
        log.warning("%s: no data for %s, skipped", figure, names)
        notes.append(f"No data for {names}; skipped.")


def _finish(fig: Figure, title: str, rule: str, notes: list[str]) -> Figure:
    """Title (naming the uniform rule when it is used), then the rule and notes under the
    plots."""
    suffix = " (uniform rule)" if rule == "uniform" else ""
    fig.suptitle(f"{title}{suffix}", x=0.01, ha="left", fontsize=12, fontweight="semibold",
                 color=INK)
    width, height = fig.get_size_inches()
    all_notes = [f"Scored with {RULE_LABELS[rule]}.", *notes]
    lines = [line for note in all_notes for line in textwrap.wrap(note, int(width * 17))]
    reserve = (0.16 * len(lines) + 0.08) / height
    fig.get_layout_engine().set(rect=(0, reserve, 1, 1 - reserve))
    fig.text(0.01, 0.015, "\n".join(lines), fontsize=8, color=INK_2, va="bottom", ha="left",
             parse_math=False)
    return fig


def _new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), layout="constrained")


def _message(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK_2,
            wrap=True)


def _system_handles(systems: list[str]) -> list[Line2D]:
    return [Line2D([], [], color=_colour(s), marker=_marker(s), linewidth=2, markersize=6,
                   markeredgecolor=SURFACE, label=_label(s)) for s in systems]


def _oracle_handle() -> Line2D:
    return Line2D([], [], color=INK, linestyle="--", linewidth=1.5,
                  label="Oracle: findable rate")


def _legend(fig: Figure, handles: list) -> None:
    if handles:
        fig.legend(handles=handles, loc="outside upper right", ncols=min(len(handles), 7))


def _lines(ax, frame: pd.DataFrame, x_values: list[str], x_column: str, systems: list[str],
           rule: str) -> None:
    """One line with a CI band per system over ordered categories; gaps where a category
    has no truths. Lines are dodged slightly so that equal values stay visible."""
    y_col, lo_col, hi_col = _curve_columns(rule)
    spread = 0.03 * (len(systems) - 1)
    for k, system in enumerate(systems):
        rows = _where(frame, system=system).set_index(x_column).reindex(x_values)
        dodge = (k / (len(systems) - 1) - 0.5) * 2 * spread if len(systems) > 1 else 0.0
        x = np.arange(len(x_values)) + dodge
        y = rows[y_col].to_numpy(dtype=float)
        ax.fill_between(x, rows[lo_col].to_numpy(dtype=float), rows[hi_col].to_numpy(dtype=float),
                        color=_colour(system), alpha=0.13, linewidth=0)
        ax.plot(x, y, color=_colour(system), linewidth=2, marker=_marker(system), markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)


def _recall_axis(ax, labels: list[str], xlabel: str) -> None:
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.35, max(len(labels) - 1, 0) + 0.35)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Recall")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)


def _oracle_rates(oracle: pd.DataFrame | None, **cell) -> pd.Series:
    """Findable rate per strength, within a cell if given."""
    rows = pd.DataFrame() if oracle is None else _where(oracle, **cell)
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.assign(found=rows["findable"].astype(float)).groupby("strength")["found"].mean()


def _plot_oracle(ax, rates: pd.Series, strengths: list[str]) -> None:
    y = rates.reindex(strengths).to_numpy(dtype=float)
    ax.plot(np.arange(len(strengths)), y, color=INK, linestyle="--", linewidth=1.5, zorder=4)


def _dots(ax, rows: list[dict], xlabel: str | None = None) -> None:
    """Horizontal point-and-CI per row ({label, system, value, lo, hi}), each labelled."""
    for k, row in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([row["lo"], row["hi"]], [y, y], color=_colour(row["system"]), linewidth=2,
                solid_capstyle="round")
        ax.plot([row["value"]], [y], color=_colour(row["system"]),
                marker=_marker(row["system"]), markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{row['value']:.3f}", (row["hi"], y), xytext=(7, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(rows)), [r["label"] for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.22)
    if xlabel:
        ax.set_xlabel(xlabel)


def _summary_dots(summary: pd.DataFrame, metric: str, systems: list[str]) -> list[dict]:
    """Rows for _dots from system_summary; systems with no value are left out."""
    rows = []
    for system in systems:
        row = _where(summary, system=system)
        if row.empty or metric not in row.columns or pd.isna(row.iloc[0][metric]):
            continue
        row = row.iloc[0]
        rows.append({"label": _label(system), "system": system, "value": row[metric],
                     "lo": row[f"{metric}_lo"], "hi": row[f"{metric}_hi"]})
    return rows


def _pair_row(frame: pd.DataFrame, first: str, second: str, **equals) -> pd.Series | None:
    """The row comparing two systems, oriented as first minus second whichever way it
    was written. None if there is no row or its estimate is missing."""
    columns = {"system_a": first, "system_b": second}
    direct = _where(frame, **columns, **equals)
    if not direct.empty:
        row = direct.iloc[0]
    else:
        swapped = _where(frame, system_a=second, system_b=first, **equals)
        if swapped.empty:
            return None
        row = swapped.iloc[0].copy()
        row["estimate"], row["lo"], row["hi"] = -row["estimate"], -row["hi"], -row["lo"]
        for left, right in (("a_only", "b_only"), ("recall_a", "recall_b"), ("lo_a", "lo_b"),
                            ("hi_a", "hi_b")):
            if left in row.index:
                row[left], row[right] = row[right], row[left]
    return None if pd.isna(row["estimate"]) else row


def _difference(ax, row: pd.Series | None, label: str, xlabel: str | None = None) -> None:
    """One paired difference with its CI, against a zero line."""
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks([0], [label])
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", visible=False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if row is None:
        ax.text(0.5, 0.75, "no data", transform=ax.transAxes, ha="center", color=INK_2)
        return
    ax.plot([row["lo"], row["hi"]], [0, 0], color=INK_2, linewidth=2, solid_capstyle="round")
    ax.plot([row["estimate"]], [0], color=INK, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    ax.margins(x=0.15)


# --- figures -------------------------------------------------------------------------


def _figure(fn: Callable) -> Callable:
    """Check the rule and apply the shared style while the figure is built."""

    @wraps(fn)
    def wrapper(data: FigureData, rule: str = "own", **kwargs) -> Figure:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}; one of {RULES}")
        with mpl.rc_context(STYLE):
            return fn(data, rule, **kwargs)

    return wrapper


def _strengths(frame: pd.DataFrame, column: str) -> list[str]:
    present = set(frame[column]) if column in frame.columns else set()
    return [s for s in STRENGTH_ORDER if s in present]


@_figure
def detection_curve(data: FigureData, rule: str = "own") -> Figure:
    """Recall vs strength per system, and placebo false discoveries per test."""
    fig = _new_figure(10, 4.6)
    curve_ax, fd_ax = fig.subplots(1, 2, width_ratios=[2.2, 1])
    curve = _where(data["detection_curve"], group="strength")
    systems, notes = _present(curve), []
    _missing("detection_curve", systems, notes)
    strengths = _strengths(curve, "value")
    _lines(curve_ax, curve, strengths, "value", systems, rule)
    _recall_axis(curve_ax, [STRENGTH_TEXT[s] for s in strengths], "Planted strength")
    curve_ax.set_title("Planted signals recovered")
    handles = list(_system_handles(systems))
    rates = _oracle_rates(data.oracle)
    if not rates.empty and strengths:
        _plot_oracle(curve_ax, rates, strengths)
        handles.append(_oracle_handle())
    if data.oracle_note:
        notes.append(data.oracle_note)

    metric = _metric("false_discoveries_per_test", rule)
    summary = data["system_summary"]
    listed = _present(summary)
    rows = _summary_dots(summary, metric, listed)
    fd_ax.set_title("Placebo false discoveries per test")
    if rows:
        _dots(fd_ax, rows, "False discoveries / backtests")
    else:
        _message(fd_ax, "No placebo data")
    no_tests = [_label(s) for s in listed if s not in {r["system"] for r in rows}]
    if no_tests:
        notes.append(f"No false-discovery rate for {', '.join(no_tests)} (no backtests).")
    _legend(fig, handles)
    return _finish(fig, "Detection by planted strength", rule, notes)


@_figure
def detection_by_cell(data: FigureData, rule: str = "own") -> Figure:
    """Recall vs strength per system, faceted by complexity x feature type."""
    cells = data["detection_cells"]
    fig = _new_figure(10, 7)
    axes = fig.subplots(len(COMPLEXITY_ORDER), len(FEATURE_TYPE_ORDER), sharey=True)
    systems, notes = _present(cells), []
    _missing("detection_by_cell", systems, notes)
    strengths = _strengths(cells, "strength")
    has_oracle = False
    for i, complexity in enumerate(COMPLEXITY_ORDER):
        for j, feature_type in enumerate(FEATURE_TYPE_ORDER):
            ax = axes[i, j]
            cell = _where(cells, complexity=complexity, feature_type=feature_type)
            ax.set_title(f"{complexity.capitalize()}, {feature_type}")
            if cell.empty:
                _message(ax, "No truths in this cell")
                continue
            _lines(ax, cell, strengths, "strength", _present(cell), rule)
            n = cell.groupby("strength")["n_truths"].max().reindex(strengths)
            labels = [f"{STRENGTH_TEXT[s]}\nn={int(v)}" if pd.notna(v) else STRENGTH_TEXT[s]
                      for s, v in n.items()]
            _recall_axis(ax, labels, "Planted strength (n: truths per system)" if i else "")
            rates = _oracle_rates(data.oracle, complexity=complexity, feature_type=feature_type)
            if not rates.empty:
                _plot_oracle(ax, rates, strengths)
                has_oracle = True
    if data.oracle_note:
        notes.append(data.oracle_note)
    handles = _system_handles(systems) + ([_oracle_handle()] if has_oracle else [])
    _legend(fig, handles)
    return _finish(fig, "Detection by complexity and feature type", rule, notes)


@_figure
def recall_by_snr(data: FigureData, rule: str = "own") -> Figure:
    """Recall per system across the SNR bins."""
    fig = _new_figure(7.5, 4.6)
    ax = fig.subplots()
    curve = _where(data["detection_curve"], group="snr_bin")
    systems, notes = _present(curve), []
    _missing("recall_by_snr", systems, notes)
    _lines(ax, curve, SNR_BIN_ORDER, "value", systems, rule)
    n = (curve.groupby("value")["n_truths"].max().reindex(SNR_BIN_ORDER)
         if not curve.empty else pd.Series(np.nan, index=SNR_BIN_ORDER))
    labels = [f"{SNR_LABELS[b]}\nn={int(v)}" if pd.notna(v) else SNR_LABELS[b]
              for b, v in n.items()]
    _recall_axis(ax, labels, "Planted signal-to-noise ratio (n: truths per system)")
    _legend(fig, _system_handles(systems))
    return _finish(fig, "Recall by signal-to-noise ratio", rule, notes)


@_figure
def r2_vs_r3_text(data: FigureData, rule: str = "own") -> Figure:
    """Recall on text truths, R2 and R3, with the truth-level paired test (main result)
    and the universe-level test as a secondary annotation."""
    first, second = "r3", "r2"
    fig = _new_figure(10, 4.4)
    recall_ax, diff_ax = fig.subplots(1, 2, width_ratios=[1, 1.5])
    notes: list[str] = []
    metric = _metric("recall", rule)
    filters = {"group": "feature_type", "value": "text", "metric": metric}
    truth = _pair_row(data["truth_comparisons"], first, second, **filters)
    universe = _pair_row(data["group_comparisons"], first, second, **filters)
    recall_ax.set_title("Recall on text truths")
    diff_ax.set_title("Paired difference, R3 − R2")
    if truth is None:
        _missing("r2_vs_r3_text", _present(data["truths"]), notes, [second, first])
        if not notes:
            notes.append(f"No text-truth comparison under the {rule} rule.")
            log.warning("r2_vs_r3_text: no text-truth comparison under the %s rule", rule)
        _message(recall_ax, "Needs R2 and R3")
        _message(diff_ax, "Needs R2 and R3")
        return _finish(fig, "R2 vs R3 on text signals", rule, notes)

    _dots(recall_ax, [
        {"label": "R2", "system": "r2", "value": truth["recall_b"], "lo": truth["lo_b"],
         "hi": truth["hi_b"]},
        {"label": "R3", "system": "r3", "value": truth["recall_a"], "lo": truth["lo_a"],
         "hi": truth["hi_a"]},
    ], "Recall")
    _difference(diff_ax, truth, "R3 − R2", "Recall difference, 95% CI (universes resampled)")
    diff_ax.set_ylim(-1.5, 0.8)
    n_truths = int(truth["n_truths"])
    _annotate(
        diff_ax, 0.02, 0.36,
        f"Truth level: {_interval(truth)}\n"
        f"Exact McNemar p = {_p(truth['p_value'])}, n = {n_truths} truths\n"
        f"R3 only {int(truth['a_only'])}, R2 only {int(truth['b_only'])}, "
        f"both {int(truth['both'])}, neither {int(truth['neither'])}",
        va="top", color=INK,
    )
    if universe is not None:
        test = {"wilcoxon": "Wilcoxon", "sign": "sign test"}.get(universe["test"], "test")
        _annotate(diff_ax, 0.02, 0.09,
                  f"Universe level: {test} p = {_p(universe['p_value'])}, "
                  f"n = {int(universe['n_universes'])} universes",
                  va="top", color=INK_2, fontsize=8)
    notes.append("McNemar treats truths as independent, though truths of one universe share a "
                 "session; the CI resamples whole universes.")
    return _finish(fig, "R2 vs R3 on text signals", rule, notes)


@_figure
def r3_vs_r3u(data: FigureData, rule: str = "own") -> Figure:
    """Recall, precision and placebo false discoveries per test, R3 vs R3-U, with paired
    differences."""
    panels = [("recall", "Recall"), ("precision", "Precision"),
              ("false_discoveries_per_test", "Placebo FD per test")]
    fig = _new_figure(11, 4.8)
    axes = fig.subplots(2, 3, height_ratios=[1.3, 1])
    summary, comparisons = data["system_summary"], data["comparisons"]
    present, notes = _present(summary), []
    wanted = ["r3", "r3-u"]
    _missing("r3_vs_r3u", present, notes, wanted)
    for k, (name, title) in enumerate(panels):
        metric = _metric(name, rule)
        top, bottom = axes[0, k], axes[1, k]
        top.set_title(title)
        rows = _summary_dots(summary, metric, [s for s in wanted if s in present])
        if rows:
            _dots(top, rows)
        else:
            _message(top, "No data")
        row = _pair_row(comparisons, "r3", "r3-u", metric=metric)
        _difference(bottom, row, "R3 − R3-U")
        if row is not None:
            _annotate(bottom, 0.02, 0.06,
                      f"{_interval(row)}\n"
                      f"p = {_p(row['p_value'])}, n = {int(row['n_universes'])} universes",
                      va="bottom", color=INK, fontsize=8)
    return _finish(fig, "R3 vs R3-U", rule, notes)


def cost_per_session(per_universe: pd.DataFrame, input_price: float | None = None,
                     output_price: float | None = None) -> pd.DataFrame:
    """Mean cost per session per system: system, cost, estimated, split.

    cost_usd where a system has it. A system with none is estimated from tokens at the
    prices (USD per million tokens), if both are given: input_tokens and both cache token
    counts at the input price and output_tokens at the output price (split True), or,
    with only the total, all tokens at the input price (split False). cost is NaN when
    neither is possible.
    """
    columns = ["system", "cost", "estimated", "split"]
    if "system" not in per_universe.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for system in _present(per_universe):
        mine = _where(per_universe, system=system)
        actual = (mine["cost_usd"].astype(float) if "cost_usd" in mine.columns
                  else pd.Series(np.nan, index=mine.index))
        row = {"system": system, "cost": actual.mean(), "estimated": False, "split": False}
        if actual.isna().all() and input_price is not None and output_price is not None:
            inputs = [c for c in ("input_tokens", "cache_read_input_tokens",
                                  "cache_creation_input_tokens") if c in mine.columns]
            if inputs and "output_tokens" in mine.columns:
                cost = (mine[inputs].astype(float).sum(axis=1) * input_price
                        + mine["output_tokens"].astype(float) * output_price) / 1e6
                row.update(cost=cost.mean(), estimated=True, split=True)
            elif "tokens" in mine.columns and mine["tokens"].notna().any():
                cost = mine["tokens"].astype(float) * input_price / 1e6
                row.update(cost=cost.mean(), estimated=True)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


@_figure
def cost_vs_recall(data: FigureData, rule: str = "own", input_price: float | None = None,
                   output_price: float | None = None) -> Figure:
    """Recall vs mean cost per session, one labelled point per system."""
    fig = _new_figure(7.5, 4.8)
    ax = fig.subplots()
    costs = cost_per_session(data["per_universe"], input_price, output_price)
    summary = data["system_summary"]
    metric = _metric("recall", rule)
    present, notes = _present(summary), []
    _missing("cost_vs_recall", present, notes)
    estimated = bool(costs["estimated"].any()) if not costs.empty else False
    skipped = []
    labels: dict[tuple[float, float], list[str]] = {}  # systems at one point share a label
    for row in _summary_dots(summary, metric, present):
        cost = _where(costs, system=row["system"])
        if cost.empty or pd.isna(cost.iloc[0]["cost"]):
            skipped.append(row["label"])
            continue
        x = cost.iloc[0]["cost"]
        ax.plot([x, x], [row["lo"], row["hi"]], color=_colour(row["system"]), linewidth=2,
                solid_capstyle="round")
        ax.plot([x], [row["value"]], color=_colour(row["system"]),
                marker=_marker(row["system"]), markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)
        labels.setdefault((x, row["value"]), []).append(row["label"])
    for point, names in labels.items():
        ax.annotate(", ".join(names), point, xytext=(7, 4), textcoords="offset points",
                    color=INK, fontsize=9)
    if skipped:
        names = ", ".join(skipped)
        log.warning("cost_vs_recall: no cost for %s, skipped", names)
        notes.append(f"No cost for {names}: cost_usd is empty and token prices were not given.")
    ax.set_ylim(-0.03, 1.03)
    ax.margins(x=0.15)
    plotted = costs["cost"].dropna() if not costs.empty else pd.Series(dtype=float)
    if len(plotted) and plotted.max() > 0:  # cost axis from zero, with room for a free system
        ax.set_xlim(left=-0.04 * plotted.max())
    elif len(plotted):  # every system is free: no scale to show, and no negative costs
        ax.set_xlim(-0.2, 1)
        ax.set_xticks([0], ["$0"], parse_math=False)
    ax.set_ylabel("Recall (95% CI)")
    ax.set_xlabel("Mean cost per session, USD (estimated from tokens)" if estimated
                  else "Mean cost per session, USD")
    if estimated:
        priced = costs[costs["estimated"]]
        note = f"Estimated at ${input_price:g} input and ${output_price:g} output per million "
        if priced["split"].all():
            note += "tokens; cache tokens counted at the input price."
        else:
            note += "tokens; sessions with only a token total are all at the input price."
        notes.append(note)
    return _finish(fig, "Recall against cost", rule, notes)


# name -> function, in reading order; each is saved as <name>.png and <name>.svg.
FIGURES: dict[str, Callable[..., Figure]] = {
    "detection_curve": detection_curve,
    "detection_by_cell": detection_by_cell,
    "recall_by_snr": recall_by_snr,
    "r2_vs_r3_text": r2_vs_r3_text,
    "r3_vs_r3u": r3_vs_r3u,
    "cost_vs_recall": cost_vs_recall,
}
