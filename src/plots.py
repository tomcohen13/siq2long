"""
Figures for the writeup.

One hue, not one per model: the model names on the axis already carry identity, so
colouring each bar differently would be decoration. What changes how this chart
reads is the chance floor and the confidence intervals -- a 4-way multiple choice
question has a 25% floor, and an accuracy without an interval invites readers to
rank models on differences that are noise.
"""

from dataclasses import dataclass
from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import PercentFormatter

from stats import wilson_interval


#: Font handling for print. Matplotlib defaults to Type 3 fonts in PDF/EPS, which
#: camera-ready checkers at NeurIPS and the CVF venues reject; 42 selects TrueType.
#: `svg.fonttype: none` keeps SVG text as text rather than outlines, so it stays
#: searchable and editable.
PAPER_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def save_figure(fig, path: str | FilePath, dpi: int = 300) -> FilePath:
    """
    Write a figure for print: vector when the suffix allows, TrueType fonts embedded.

    Prefer `.pdf` for LaTeX -- text stays selectable and nothing resamples. Use `.png`
    only for slides or a README, where `dpi` matters; at 300 it is sharp on paper but
    still a raster.
    """
    path = FilePath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor=fig.get_facecolor(),  # else savefig reverts to opaque white
        )
    return path


@dataclass(frozen=True)
class Theme:
    """Chart surface and ink. Dark is its own selected set, not an inverted light."""

    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    baseline: str
    series: str
    series_dark: str


LIGHT = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    series="#2a78d6",       # sequential blue, step 450
    series_dark="#184f95",  # step 600, for the interval whiskers
)

DARK = Theme(
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    series="#3987e5",       # step 400
    series_dark="#86b6ef",  # step 250, lighter so whiskers read on a dark fill
)


def accuracy_table(results: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    """
    Per-group accuracy with Wilson intervals, sorted descending.

    Expects long-form rows (one per question) with a boolean `correct`, because the
    interval needs `n` -- a pre-aggregated mean can't produce one. Unlabeled rows
    (`correct` is NA) are dropped rather than counted wrong.
    """
    for column in (by, "correct"):
        if column not in results.columns:
            raise KeyError(f"results needs a {column!r} column, got {list(results.columns)}")

    scored = results.dropna(subset=["correct"])
    grouped = scored.groupby(by)["correct"]

    table = pd.DataFrame({"n": grouped.size(), "correct": grouped.sum().astype(int)})
    table["accuracy"] = table["correct"] / table["n"]
    bounds = [wilson_interval(int(c), int(n)) for c, n in zip(table["correct"], table["n"])]
    table["ci_low"] = [low for low, _ in bounds]
    table["ci_high"] = [high for _, high in bounds]
    return table.sort_values("accuracy", ascending=False)


def _rounded_bar(ax, x_end: float, y_center: float, height: float, radius_px: float = 4.0):
    """
    Bar path: square where it meets the baseline, rounded at the data end.

    The radius is specified in pixels and converted to data units per axis, so the
    corner stays circular on screen instead of stretching with the aspect ratio.
    """
    ax.figure.canvas.draw()
    box = ax.get_window_extent()
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    r_x = radius_px / box.width * (x_hi - x_lo)
    r_y = radius_px / box.height * (y_hi - y_lo)

    r_x = min(r_x, abs(x_end) / 2)
    r_y = min(r_y, height / 2)
    y0, y1 = y_center - height / 2, y_center + height / 2

    vertices = [
        (0, y0),
        (x_end - r_x, y0),
        (x_end, y0), (x_end, y0 + r_y),      # quadratic corner
        (x_end, y1 - r_y),
        (x_end, y1), (x_end - r_x, y1),      # quadratic corner
        (0, y1),
        (0, y0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    return Path(vertices, codes)


def plot_model_accuracy(
    results: pd.DataFrame,
    *,
    by: str = "model",
    chance: float | None = 0.25,
    title: str = "Zero-shot accuracy on Social-IQ 2.0 oracle clips",
    subtitle: str | None = None,
    dark: bool = False,
    figsize: tuple[float, float] = (8.2, 3.9),
    bar_height: float = 0.32,
):
    """
    Horizontal accuracy bars with 95% Wilson intervals and the chance floor.

    Args:
        results: long-form results, one row per question, with `by` and `correct`.
            Concatenate the per-model JSONLs and add a `model` column.
        by: column to group on -- `model`, or e.g. `condition` for tx vs notx.
        chance: reference line for random guessing; None to omit. 0.25 for 4 options.
        subtitle: defaults to a note on sample size, which matters when runs skipped
            videos and n differs between models.

    Returns:
        (fig, ax). Save with `fig.savefig(path, dpi=300)`; keep `figsize` as-is so
        the rounded corners stay 4px.
    """
    theme = DARK if dark else LIGHT
    table = accuracy_table(results, by=by)

    # Plotted bottom-up, so reversing puts the strongest model at the top.
    ordered = table.iloc[::-1]
    positions = range(len(ordered))

    fig, ax = plt.subplots(figsize=figsize, facecolor=theme.surface)
    ax.set_facecolor(theme.surface)

    upper = max(ordered["ci_high"].max(), chance or 0)
    ax.set_xlim(0, min(1.0, upper + 0.07))
    ax.set_ylim(-0.6, len(ordered) - 0.4)  # equal air above the top bar and below the last

    # Grid first and recessive: solid hairlines, one step off the surface.
    ax.xaxis.grid(True, color=theme.grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if chance is not None:
        # Dashed *because* it is a threshold, not a grid -- the one place dashing
        # carries meaning rather than noise.
        ax.axvline(chance, color=theme.ink_muted, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            f"chance  {chance:.0%}",
            xy=(chance, len(ordered) - 0.55),
            xytext=(4, 0),
            textcoords="offset points",
            color=theme.ink_muted,
            fontsize=8.5,
            va="center",
        )

    for y, (_, row) in zip(positions, ordered.iterrows()):
        ax.add_patch(
            PathPatch(
                _rounded_bar(ax, row["accuracy"], y, bar_height),
                facecolor=theme.series,
                edgecolor="none",  # gaps and rings separate marks, never strokes
                zorder=2,
            )
        )
        # Caps sized in points, not data units, so they stay even as the axis rescales.
        # No surface ring here: the interval's lower half necessarily lies inside its own
        # bar, and a surface-coloured halo there cuts a white notch through the fill. A
        # darker step of the same hue reads against both the fill and the surface.
        ax.plot(
            [row["ci_low"], row["ci_high"]], [y, y],
            color=theme.series_dark, linewidth=1.5,
            marker="|", markersize=5.5, markeredgewidth=1.5,
            solid_capstyle="butt", zorder=3,
        )
        ax.annotate(
            f"{row['accuracy']:.1%}",
            xy=(row["ci_high"], y),
            xytext=(7, 0),
            textcoords="offset points",
            color=theme.ink,
            fontsize=10,
            va="center",
        )

    ax.axvline(0, color=theme.baseline, linewidth=1.0, zorder=2)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(ordered.index, color=theme.ink, fontsize=10.5)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(axis="x", colors=theme.ink_muted, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)

    counts = table["n"]
    if subtitle is None:
        span = f"n = {counts.iloc[0]:,}" if counts.nunique() == 1 else f"n = {counts.min():,}–{counts.max():,}"
        subtitle = f"{span} questions per model · bars show 95% Wilson intervals"

    ax.set_title(title, color=theme.ink, fontsize=12.5, loc="left", pad=22, fontweight="bold")
    ax.annotate(
        subtitle,
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
        color=theme.ink_secondary, fontsize=9.5, va="bottom",
    )

    fig.tight_layout()
    return fig, ax


#: Categorical hues for the three chunk representations, assigned in this fixed order and
#: never cycled. Okabe-Ito blue/vermillion/green: validated for CVD separation (worst
#: adjacent pair dE 11.0 deutan, 25.8 normal vision) against the light surface.
REPRESENTATION_HUES = {
    "video": "#0072B2",
    "transcript": "#D55E00",
    "fused": "#009E73",
}


def plot_top1_by_pool_size(
    strata: dict[tuple[str, str], list[dict]],
    path: str | FilePath,
    title: str = "",
    theme: Theme = LIGHT,
) -> FilePath:
    """
    Top-1 against how many chunks the oracle competed against, faceted by query form.

    Two panels rather than six lines on one axis: the question-only and question+options
    conditions differ by more than the representations do, and overlaying them would hide
    that. The panels share a y-axis so the gap between them is readable as a distance.

    The dashed line is the random floor for each band -- `mean(1/pool)`, not a constant,
    since a 2-chunk video is a coin flip and a 20-chunk one is not. It is the only
    reference that makes the series interpretable, so it is drawn on both panels.
    """
    queries = sorted({q for _, q in strata}, key=len)  # "question" before "question+options"
    bands = [row["band"] for row in next(iter(strata.values()))]
    x = range(len(bands))

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, len(queries), figsize=(9.0, 3.6), sharey=True, facecolor=theme.surface
        )
        for ax, query in zip(axes, queries):
            ax.set_facecolor(theme.surface)
            floor = [row["random"] for row in strata[(next(iter(REPRESENTATION_HUES)), query)]]
            ax.plot(x, floor, ls=(0, (4, 3)), lw=1.5, color=theme.ink_muted, zorder=1,
                    label="random")

            for rep, hue in REPRESENTATION_HUES.items():
                rows = strata[(rep, query)]
                y = [r["top1"] for r in rows]
                err = [[r["top1"] - r["lo"] for r in rows], [r["hi"] - r["top1"] for r in rows]]
                ax.errorbar(x, y, yerr=err, color=hue, lw=2.0, marker="o", markersize=5,
                            capsize=0, elinewidth=1.2, zorder=3, label=rep,
                            markeredgecolor=theme.surface, markeredgewidth=1.2)

            ax.set_title(query, color=theme.ink, fontsize=10.5, loc="left", pad=8)
            ax.set_xticks(list(x), bands)
            ax.set_xlabel("chunks in the video", color=theme.ink_secondary, fontsize=9.5)
            ax.grid(axis="y", color=theme.grid, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(theme.baseline)
            ax.tick_params(colors=theme.ink_secondary, labelsize=9)

        axes[0].set_ylabel("top-1 accuracy", color=theme.ink_secondary, fontsize=9.5)
        axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axes[0].set_ylim(0, None)
        # Legend once, outside the data: identity is never colour-alone.
        axes[-1].legend(frameon=False, fontsize=9, labelcolor=theme.ink_secondary,
                        loc="upper right", handlelength=1.6)

        n = sum(row["n"] for row in strata[("video", queries[0])])
        fig.tight_layout()
        # Stacked above the axes rather than inside the figure box: `save_figure` crops
        # with bbox_inches="tight", so anything past y=1 is kept and nothing collides
        # with the panel titles.
        if title:
            fig.text(0.0, 1.10, title, color=theme.ink, fontsize=12, ha="left",
                     va="bottom", fontweight="bold")
        fig.text(0.0, 1.02, f"n = {n:,} questions · bars show 95% Wilson intervals",
                 color=theme.ink_secondary, fontsize=9.5, ha="left", va="bottom")
    return save_figure(fig, path)
