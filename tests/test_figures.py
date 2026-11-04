"""Figures from summary tables shaped like the final plan (tests/summary_fixtures.py)."""

import importlib.util
import io
import logging
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.text import Text

from evaluation.figures import (
    FIGURES,
    FigureData,
    checked_oracle,
    cost_per_session,
    cost_vs_recall,
    detection_by_cell,
    detection_curve,
    load_figure_data,
    r2_vs_r3_text,
    r3_vs_r3u,
    recall_by_snr,
)
from summary_fixtures import oracle_frame, write_fixture_summary


def all_text(figure) -> str:
    return "\n".join(t.get_text() for t in figure.findobj(Text))


def renders(figure) -> None:
    for fmt in ("png", "svg"):
        buffer = io.BytesIO()
        figure.savefig(buffer, format=fmt, dpi=60)
        assert buffer.tell() > 0


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    path = tmp_path_factory.mktemp("results")
    write_fixture_summary(path)
    return path


@pytest.fixture(scope="module")
def data(results):
    return load_figure_data(results)


@pytest.fixture
def with_oracle(data):
    return FigureData(data.tables, oracle_frame())


@pytest.mark.parametrize("rule", ["own", "uniform"])
@pytest.mark.parametrize("name", list(FIGURES))
def test_every_figure_renders(data, name, rule):
    renders(FIGURES[name](data, rule=rule))


@pytest.mark.parametrize("name", list(FIGURES))
def test_every_figure_renders_with_no_data(name):
    renders(FIGURES[name](FigureData({})))


@pytest.mark.parametrize("name", list(FIGURES))
def test_every_figure_skips_missing_systems(tmp_path, name, caplog):
    write_fixture_summary(tmp_path, systems=["r3"], n_boot=20)
    with caplog.at_level(logging.WARNING, logger="evaluation.figures"):
        figure = FIGURES[name](load_figure_data(tmp_path))
    renders(figure)
    assert "No data for" in all_text(figure) or "Needs R2 and R3" in all_text(figure)
    assert "skipped" in caplog.text


def test_unknown_rule_raises(data):
    with pytest.raises(ValueError, match="unknown rule"):
        detection_curve(data, rule="strict")


def test_oracle_line_is_drawn_only_when_given(data, with_oracle):
    def oracle_lines(figure, ax_index=0):
        return [ln for ln in figure.axes[ax_index].get_lines() if ln.get_linestyle() == "--"]

    plain = detection_curve(data)
    assert not oracle_lines(plain) and "Oracle" not in all_text(plain)
    figure = detection_curve(with_oracle)
    line = oracle_lines(figure)[0]
    # weak: only t1's and t3's single text truths, 2 of 8; strong: all 8.
    assert list(line.get_ydata()) == pytest.approx([2 / 8, 1.0])
    assert "Oracle: findable rate" in all_text(figure)


def test_oracle_line_per_cell(with_oracle):
    figure = detection_by_cell(with_oracle)
    assert len(figure.axes) == 4
    by_title = {ax.get_title(loc="left"): ax for ax in figure.axes}
    line = next(ln for ln in by_title["Single, text"].get_lines() if ln.get_linestyle() == "--")
    assert list(line.get_ydata()) == pytest.approx([1.0, 1.0])
    line = next(ln for ln in by_title["Single, structured"].get_lines()
                if ln.get_linestyle() == "--")
    assert list(line.get_ydata()) == pytest.approx([0.0, 1.0])


def test_stale_oracle_is_left_out_with_a_note(results, tmp_path, caplog):
    stale = oracle_frame().assign(universe_id="another-plan")
    stale.to_csv(tmp_path / "oracle.csv", index=False)
    with caplog.at_level(logging.WARNING, logger="evaluation.figures"):
        data = load_figure_data(results, tmp_path / "oracle.csv")
    assert data.oracle is None
    assert "computed on another plan" in caplog.text
    figure = detection_curve(data)
    assert "Oracle line left out" in all_text(figure)


def test_checked_oracle_accepts_a_superset(data):
    oracle, note = checked_oracle(oracle_frame(), data["truths"].iloc[:4])
    assert oracle is not None and note is None


def test_detection_curve_shows_the_false_discovery_rate(data):
    text = all_text(detection_curve(data))
    assert "0.017" in text  # r3: 1 / 60
    assert "0.020" in text  # r3-u: 4 / 200
    assert "Placebo false discoveries per test" in text


def test_recall_by_snr_uses_the_new_bins(data):
    text = all_text(recall_by_snr(data))
    for label in ("<2", "2–4", "4–8", "8–16", "≥16"):
        assert label in text
    # snr < 2 in the fixture: 1.5, 1.0, 1.2, 1.8 (and none else).
    assert "<2\nn=4" in text


def test_r2_vs_r3_text_reports_truth_and_universe_level(data):
    text = all_text(r2_vs_r3_text(data))
    # R3 minus R2 on 8 text truths: 6/8 - 3/8.
    assert "Truth level: +0.375" in text
    assert "Exact McNemar p = 0.375, n = 8 truths" in text
    assert "R3 only 4, R2 only 1, both 2, neither 1" in text
    assert "n = 4 universes" in text


def test_r2_vs_r3_text_uniform_rule(data):
    text = all_text(r2_vs_r3_text(data, rule="uniform"))
    assert "uniform Holm" in text
    assert "R3 only 4, R2 only 2, both 1, neither 1" in text  # r3 loses t1's weak truth


def test_r3_vs_r3u_panels(data):
    figure = r3_vs_r3u(data)
    text = all_text(figure)
    for title in ("Recall", "Precision", "Placebo FD per test"):
        assert title in text
    assert "R3-U" in text
    assert "n = 8 universes" in text and "n = 2 universes" in text


def test_cost_is_estimated_from_tokens_and_labelled(data):
    costs = cost_per_session(data["per_universe"], input_price=3.0, output_price=15.0)
    r3 = costs.set_index("system").loc["r3"]
    assert r3["cost"] == pytest.approx((60_000 * 3 + 15_000 * 15) / 1e6)
    assert bool(r3["estimated"]) and bool(r3["split"])
    figure = cost_vs_recall(data, input_price=3.0, output_price=15.0)
    assert "estimated from tokens" in figure.axes[0].get_xlabel()
    assert "$3 input and $15 output" in all_text(figure)


def test_cost_with_only_a_token_total_uses_the_input_price():
    frame = pd.DataFrame({"system": ["r3"], "tokens": [1_000_000], "cost_usd": [None]})
    row = cost_per_session(frame, input_price=2.0, output_price=10.0).iloc[0]
    assert row["cost"] == pytest.approx(2.0)
    assert bool(row["estimated"]) and not bool(row["split"])


def test_recorded_cost_is_not_estimated():
    frame = pd.DataFrame({"system": ["r3", "r3"], "cost_usd": [1.0, 3.0], "tokens": [1, 1]})
    row = cost_per_session(frame, input_price=2.0, output_price=10.0).iloc[0]
    assert row["cost"] == 2.0 and not bool(row["estimated"])


def test_cost_without_prices_is_a_note_not_an_error(data, caplog):
    with caplog.at_level(logging.WARNING, logger="evaluation.figures"):
        figure = cost_vs_recall(data)
    assert "token prices were not given" in all_text(figure)
    assert "estimated" not in figure.axes[0].get_xlabel()
    assert "no cost for" in caplog.text


def test_free_systems_share_a_label_and_no_negative_cost_is_shown(data):
    per_universe = data["per_universe"].assign(cost_usd=0.0)
    summary = data["system_summary"].assign(recall=0.5)
    figure = cost_vs_recall(FigureData({**data.tables, "per_universe": per_universe,
                                        "system_summary": summary}))
    assert "R2, R3, R3-U" in all_text(figure)
    assert [t.get_text() for t in figure.axes[0].get_xticklabels()] == ["$0"]
    renders(figure)


def test_stubs_are_grey_and_core_colours_are_fixed(tmp_path):
    write_fixture_summary(tmp_path, systems=["r2", "r3"], n_boot=20)
    data = load_figure_data(tmp_path)
    curve = data.tables["detection_curve"].replace({"system": {"r2": "null"}})
    grey = detection_curve(FigureData({**data.tables, "detection_curve": curve}))
    colours = {ln.get_color() for ln in grey.axes[0].get_lines()}
    assert colours == {"#898781", "#eda100"}  # stub grey, r3's yellow


def test_make_figures_script_writes_png_and_svg(results, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    oracle_frame().to_csv(tmp_path / "oracle.csv", index=False)
    args = ["--results", str(results), "--oracle", str(tmp_path / "oracle.csv"),
            "--input-price", "3", "--output-price", "15"]
    assert module.main(args) == 0
    written = {p.name for p in (results / "figures").iterdir()}
    assert written == {f"{name}.{ext}" for name in FIGURES for ext in ("png", "svg")}
    assert module.main([*args, "--rule", "uniform"]) == 0
    assert (results / "figures" / "detection_curve_uniform.png").exists()
