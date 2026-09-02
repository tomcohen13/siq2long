"""Interval arithmetic and the aggregation behind the accuracy figure."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # no display in CI

from plots import PAPER_RC, accuracy_table, plot_model_accuracy, save_figure
from stats import wilson_interval


# --- wilson_interval ---------------------------------------------------------

def test_interval_brackets_the_estimate():
    low, high = wilson_interval(262, 852)
    assert low < 262 / 852 < high


def test_interval_stays_inside_unit_range_at_the_extremes():
    """The normal approximation would run past 0 and 1 here; Wilson must not."""
    assert wilson_interval(0, 20)[0] >= 0.0
    assert wilson_interval(20, 20)[1] <= 1.0


def test_interval_narrows_as_n_grows():
    widths = [wilson_interval(round(0.3 * n), n) for n in (16, 100, 852)]
    widths = [high - low for low, high in widths]
    assert widths == sorted(widths, reverse=True)


def test_llava_interval_clears_the_chance_floor():
    """The paper claims above-chance; this is the arithmetic behind it."""
    low, _ = wilson_interval(262, 852)
    assert low > 0.25


def test_qwen_and_internvl_intervals_overlap():
    """0.668 vs 0.658 is not a ranking -- guard against the figure implying one."""
    qwen = wilson_interval(569, 852)
    intern = wilson_interval(561, 852)
    assert qwen[0] < intern[1] and intern[0] < qwen[1]


@pytest.mark.parametrize("successes, n", [(0, 0), (-1, 10), (11, 10)])
def test_rejects_impossible_inputs(successes, n):
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


# --- accuracy_table ----------------------------------------------------------

def _long_frame():
    counts = {"a": 6, "b": 3}
    return pd.DataFrame(
        [{"model": m, "correct": i < k} for m, k in counts.items() for i in range(10)]
    )


def test_table_counts_and_accuracy():
    table = accuracy_table(_long_frame())
    assert table.loc["a", "n"] == 10 and table.loc["a", "correct"] == 6
    assert table.loc["a", "accuracy"] == pytest.approx(0.6)


def test_table_is_sorted_descending():
    assert list(accuracy_table(_long_frame()).index) == ["a", "b"]


def test_table_drops_unlabeled_rows():
    """Unlabeled rows must leave the denominator, not count as wrong."""
    df = pd.DataFrame([
        {"model": "a", "correct": True},
        {"model": "a", "correct": False},
        {"model": "a", "correct": None},
    ])
    table = accuracy_table(df)
    assert table.loc["a", "n"] == 2
    assert table.loc["a", "accuracy"] == pytest.approx(0.5)


def test_table_requires_expected_columns():
    with pytest.raises(KeyError):
        accuracy_table(pd.DataFrame({"model": ["a"], "score": [1]}))


def test_table_groups_by_any_column():
    df = pd.DataFrame([
        {"condition": "tx", "correct": True},
        {"condition": "notx", "correct": False},
    ])
    assert set(accuracy_table(df, by="condition").index) == {"tx", "notx"}


# --- figure ------------------------------------------------------------------

def test_figure_has_one_bar_per_group_and_a_chance_line():
    fig, ax = plot_model_accuracy(_long_frame())
    assert len(ax.patches) == 2
    # the chance rule plus the x=0 baseline
    assert len([line for line in ax.lines if line.get_linestyle() != "-"]) == 1
    matplotlib.pyplot.close(fig)


def test_chance_line_can_be_omitted():
    fig, ax = plot_model_accuracy(_long_frame(), chance=None)
    assert all(line.get_linestyle() == "-" for line in ax.lines)
    matplotlib.pyplot.close(fig)


def test_bars_start_at_zero():
    """A truncated baseline would exaggerate differences between models."""
    fig, ax = plot_model_accuracy(_long_frame())
    assert ax.get_xlim()[0] == 0
    matplotlib.pyplot.close(fig)


def test_dark_mode_uses_its_own_surface():
    light, ax_l = plot_model_accuracy(_long_frame())
    dark, ax_d = plot_model_accuracy(_long_frame(), dark=True)
    assert ax_l.get_facecolor() != ax_d.get_facecolor()
    matplotlib.pyplot.close(light)
    matplotlib.pyplot.close(dark)
