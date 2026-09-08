"""Figures for the writeup."""

from dataclasses import dataclass
from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

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

#: Categorical hues for encoders, keyed by cache stem, assigned in this fixed order and
#: never cycled. Okabe-Ito blue/vermillion/green plus a violet: validated for CVD
#: separation (worst adjacent dE 11.0 deutan, 25.8 normal) and >= 3:1 contrast on the light
#: surface. A fifth encoder folds into "other" or gets its own facet rather than a new hue.
ENCODER_HUES = {
    "xclip": "#0072B2",
    "pe-video": "#D55E00",
    "bge": "#009E73",
    # "bm25": "#7B52AB",
}

#: Shape carries the pool, hue carries the encoder, so the two variables never compete for
#: the same channel. Filled circle for the task, filled diamond for the control.
POOL_MARKERS = {"within": "o", "cross": "D"}



def plot_pool_contrast(
    within: dict[str, dict[tuple[str, str], float]],
    cross: dict[str, dict[tuple[str, str], float]],
    random_floor: dict[str, float],
    path: str | FilePath,
    title: str = "",
    theme: Theme = LIGHT,
) -> FilePath:
    """
    Top-1 with same-video distractors versus other-video distractors, one panel per encoder.

    A dumbbell, because the quantity of interest is a *change* between two conditions on
    the same configuration: one row per (representation, query), an open marker for the
    within-video pool and a filled one for the cross-video pool, joined by a connector in
    the representation's hue. Grouped bars would encode the same two numbers while making
    the reader compute the difference; the connector *is* the difference, and its length
    is the only thing the reader has to look at.

    Encoders are faceted rather than interleaved: rows stay in one place across panels, so
    "does this encoder fail the same way" is a vertical comparison at a fixed height. The
    x-axis is shared, so connector lengths are comparable between panels too.

    Pool sizes are matched between conditions, so the random floor is one line per panel
    and every horizontal distance on the chart is interpretable.

    Args:
        within: `{encoder: {(representation, query): top1}}` against each video's own chunks.
        cross: the same keys, against a same-sized pool drawn from other videos.
        random_floor: `{encoder: mean(1/pool)}`, identical across conditions by construction.
        path: where to write the figure.
    """
    encoders = [e for e in within if e in cross]
    if not encoders:
        raise ValueError("within and cross share no encoder")

    rows = sorted(
        {k for e in encoders for k in within[e] if k in cross[e]},
        # Grouped by representation in the fixed hue order, question before question+options.
        key=lambda k: (list(REPRESENTATION_HUES).index(k[0]), len(k[1])),
    )
    if not rows:
        raise ValueError("within and cross share no configuration")

    hi_x = max(max(cross[e].values()) for e in encoders)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, len(encoders), sharex=True, sharey=True,
            figsize=(3.6 * len(encoders) + 1.6, 0.46 * len(rows) + 2.0),
            facecolor=theme.surface,
        )
        # subplots returns a bare Axes for a single panel and an array otherwise.
        axes = axes if len(encoders) > 1 else [axes]

        for ax, encoder in zip(axes, encoders):
            ax.set_facecolor(theme.surface)
            floor = random_floor[encoder]
            ax.axvline(floor, ls=(0, (4, 3)), lw=1.4, color=theme.ink_muted, zorder=1)

            for y, key in enumerate(reversed(rows)):
                if key not in within[encoder] or key not in cross[encoder]:
                    continue
                colour = REPRESENTATION_HUES[key[0]]
                a, b = within[encoder][key], cross[encoder][key]
                ax.plot([a, b], [y, y], lw=2, color=colour, zorder=2, solid_capstyle="round")
                ax.plot(a, y, "o", ms=8, color=theme.surface, markeredgecolor=colour,
                        markeredgewidth=2, zorder=3)
                ax.plot(b, y, "o", ms=8, color=colour, markeredgecolor=theme.surface,
                        markeredgewidth=1.2, zorder=4)
                ax.annotate(f"+{100 * (b - a):.1f}", xy=(max(a, b), y), xytext=(9, 0),
                            textcoords="offset points", ha="left", va="center",
                            color=theme.ink_secondary, fontsize=8.5)

            ax.set_title(encoder, color=theme.ink, fontsize=10.5, fontweight="bold", pad=8)
            ax.set_xlabel("top-1 oracle retrieval", color=theme.ink_secondary, fontsize=9.5)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_locator(MultipleLocator(0.10))
            ax.grid(axis="x", color=theme.grid, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(theme.baseline)
            ax.tick_params(colors=theme.ink_secondary, labelsize=9, length=0)

        labels = [f"{rep} · {'question' if q == 'question' else 'question + options'}"
                  for rep, q in reversed(rows)]
        axes[0].set_yticks(range(len(rows)), labels, fontsize=9.5)
        axes[0].set_ylim(-0.7, len(rows) - 0.3)
        axes[0].set_xlim(0.10, max(hi_x + 0.12, 0.60))

        # Shape carries the condition, hue carries the representation, and the row label
        # names the representation in ink -- so identity is never colour alone.
        marker = dict(marker="o", ls="none", ms=8, color=theme.ink_secondary)
        fig.legend(
            handles=[
                Line2D([], [], **marker | {"mfc": theme.surface, "mew": 2},
                       label="distractors from the same video"),
                Line2D([], [], **marker | {"mew": 1.2, "mec": theme.surface},
                       label="distractors from other videos"),
                Line2D([], [], ls=(0, (4, 3)), lw=1.4, color=theme.ink_muted,
                       label="random chunk"),
            ],
            loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, frameon=False,
            handletextpad=0.5, columnspacing=1.6, fontsize=9.5,
            labelcolor=theme.ink_secondary,
        )
        fig.tight_layout()
        if title:
            fig.text(0.0, 1.075, title, color=theme.ink, fontsize=12, ha="left",
                     va="bottom", fontweight="bold")
    return save_figure(fig, path)


def plot_pool_contrast_overlay(
    within: dict[str, dict[tuple[str, str], float]],
    cross: dict[str, dict[tuple[str, str], float]],
    random_floor: float,
    path: str | FilePath,
    title: str = "",
    theme: Theme = LIGHT,
) -> FilePath:
    """
    Every encoder's within- and cross-video top-1 on one shared axis.

    The paper version of `plot_pool_contrast`. That function facets by encoder and hues by
    representation, which is the right shape for reading one encoder in detail. This one
    puts all encoders on a single x-axis so the comparison that matters -- *does this
    encoder fail the same way* -- is a horizontal distance rather than a jump between
    panels, and it survives being printed at column width.

    Encoding: **hue is the encoder, shape is the pool.** Representation stays in the row
    label, where it costs no channel. Marks are dodged vertically within each row so
    nothing overlaps, and the connector line is gone: with two shapes the pairing is
    already visible, and six connectors on one axis read as a barcode.

    Args:
        within: `{encoder: {(representation, query): top1}}` against each video's own chunks.
        cross: the same keys, against a same-sized pool drawn from other videos.
        random_floor: `mean(1/pool)`, shared by both conditions and every encoder because
            pool sizes are matched.
        path: where to write the figure.
    """
    encoders = [e for e in within if e in cross]
    if not encoders:
        raise ValueError("within and cross share no encoder")
    unknown = [e for e in encoders if e not in ENCODER_HUES]
    if unknown:
        raise ValueError(f"no hue assigned for {unknown}; add them to ENCODER_HUES")

    rows = sorted(
        {k for e in encoders for k in within[e] if k in cross[e]},
        # Grouped by representation in the fixed hue order, question before question+options.
        key=lambda k: (list(REPRESENTATION_HUES).index(k[0]), len(k[1])),
    )
    if not rows:
        raise ValueError("within and cross share no configuration")

    # The dodge is in data units, so it only reads as "one group" if the row pitch stays
    # tight: a tall row turns the same offset into a visible gap and the marks scatter.
    span = 0.30
    offsets = [0.0] if len(encoders) == 1 else [
        span * (i / (len(encoders) - 1) - 0.5) for i in range(len(encoders))
    ]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.0, 0.52 * len(rows) + 1.7), facecolor=theme.surface)
        ax.set_facecolor(theme.surface)

        for y in range(len(rows)):
            # A faint band per row keeps the dodged marks legible as one group.
            if y % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color=theme.grid, alpha=0.35, lw=0, zorder=0)

        ax.axvline(random_floor, ls=(0, (4, 3)), lw=1.4, color=theme.ink_muted, zorder=1)

        for y, key in enumerate(reversed(rows)):
            for encoder, dy in zip(encoders, offsets):
                colour = ENCODER_HUES[encoder]
                present = {p: s[encoder][key] for p, s in (("within", within), ("cross", cross))
                           if key in s[encoder]}
                # A hairline, not a rule: shape and hue both carry identity, and neither
                # says "these two marks are one measurement". Without it 24 marks on one
                # axis read as a scatter. Kept faint so the marks stay the figure.
                if len(present) == 2:
                    ax.plot(list(present.values()), [y + dy] * 2, lw=1.2, color=colour,
                            alpha=0.45, solid_capstyle="round", zorder=2)
                for pool_type, value in present.items():
                    ax.plot(
                        value, y + dy, POOL_MARKERS[pool_type],
                        ms=6.5 if pool_type == "cross" else 7.5, color=colour,
                        markeredgecolor=theme.surface, markeredgewidth=1.4, zorder=3,
                    )

        labels = [f"{rep} · {'question' if q == 'question' else 'question + options'}"
                  for rep, q in reversed(rows)]
        ax.set_yticks(range(len(rows)), labels, fontsize=9.5)
        ax.set_ylim(-0.5, len(rows) - 0.5)
        ax.set_xlabel("top-1 oracle retrieval", color=theme.ink_secondary, fontsize=9.5)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.xaxis.set_major_locator(MultipleLocator(0.10))
        ax.set_xlim(0.08, max(max(v for e in encoders for v in cross[e].values()) + 0.05, 0.60))
        ax.grid(axis="x", color=theme.grid, lw=0.8, zorder=1)
        ax.set_axisbelow(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme.baseline)
        ax.tick_params(colors=theme.ink_secondary, labelsize=9, length=0)

        shapes = [
            Line2D([], [], marker=POOL_MARKERS[p], ls="none", ms=7,
                   color=theme.ink_secondary, markeredgecolor=theme.surface,
                   markeredgewidth=1.2, label=name)
            for p, name in (("within", "same video"), ("cross", "other videos"))
        ]
        # Swatches, not markers: a third mark shape in the legend would read as a third
        # pool condition.
        hues = [Patch(facecolor=ENCODER_HUES[e], edgecolor="none", label=e) for e in encoders]
        floor = [Line2D([], [], ls=(0, (4, 3)), lw=1.4, color=theme.ink_muted,
                        label=f"random {random_floor:.1%}")]
        fig.legend(
            handles=shapes + hues + floor,
            loc="lower left", bbox_to_anchor=(0.0, 1.005),
            ncol=len(shapes) + len(hues) + 1, frameon=False,
            handletextpad=0.5, columnspacing=1.5, fontsize=9.5,
            labelcolor=theme.ink_secondary,
        )
        fig.tight_layout()
        if title:
            fig.text(0.0, 1.085, title, color=theme.ink, fontsize=12, ha="left",
                     va="bottom", fontweight="bold")
    return save_figure(fig, path)


def plot_pool_by_condition(
    within: dict[str, dict[tuple[str, str], float]],
    cross: dict[str, dict[tuple[str, str], float]],
    random_floor: float,
    path: str | FilePath,
    title: str = "",
    theme: Theme = LIGHT,
) -> FilePath:
    """
    Top-1 per retrieval configuration, both pools, both encoders, on one accuracy axis.

    The main-paper figure, and deliberately the same grammar as `plot_condition_accuracy`:
    accuracy on y, conditions on x, hue for the model. Two figures that read identically
    cost the reader one orientation instead of two.

    Chosen over a within-vs-cross scatter against the identity line. The scatter states the
    claim more compactly, but only the *contrast*; this carries the contrast, the absolute
    levels, and the chance floor as a single horizontal rule -- which is the only way the
    reader sees that `transcript · question` falls **below** chance against its own video's
    chunks.

    Encoding: hue is the encoder, `POOL_MARKERS` shape is the pool, and a hairline joins
    the pair so four marks per condition read as two measurements rather than a scatter.

    Args:
        within: `{encoder: {(representation, query): top1}}` against each video's own chunks.
        cross: the same keys, against a same-sized pool drawn from other videos.
        random_floor: `mean(1/pool)`, identical across pools and encoders by construction.
        path: where to write the figure.
    """
    encoders = [e for e in within if e in cross]
    if not encoders:
        raise ValueError("within and cross share no encoder")
    unknown = [e for e in encoders if e not in ENCODER_HUES]
    if unknown:
        raise ValueError(f"no hue assigned for {unknown}; add them to ENCODER_HUES")

    conditions = sorted(
        {k for e in encoders for k in within[e] if k in cross[e]},
        key=lambda k: (list(REPRESENTATION_HUES).index(k[0]), len(k[1])),
    )
    if not conditions:
        raise ValueError("within and cross share no configuration")

    span = 0.30
    offsets = [0.0] if len(encoders) == 1 else [
        span * (i / (len(encoders) - 1) - 0.5) for i in range(len(encoders))
    ]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=theme.surface)
        ax.set_facecolor(theme.surface)

        ax.axhline(random_floor, ls=(0, (4, 3)), lw=1.4, color=theme.ink_muted, zorder=1)
        # Below the line, not above: the space above it at the right edge holds marks.
        ax.annotate(f"random {random_floor:.1%}", xy=(len(conditions) - 0.55, random_floor),
                    xytext=(0, -5), textcoords="offset points", ha="right", va="top",
                    color=theme.ink_muted, fontsize=8.5)

        # Representations are grouped on x; a separator makes the grouping structural
        # rather than something the reader infers from the labels.
        for i in range(1, len(conditions)):
            if conditions[i][0] != conditions[i - 1][0]:
                ax.axvline(i - 0.5, color=theme.grid, lw=1.0, zorder=0)

        for x, key in enumerate(conditions):
            for encoder, dx in zip(encoders, offsets):
                colour = ENCODER_HUES[encoder]
                present = {p: s[encoder][key] for p, s in (("within", within), ("cross", cross))
                           if key in s[encoder]}
                if len(present) == 2:
                    ax.plot([x + dx] * 2, list(present.values()), lw=1.2, color=colour,
                            alpha=0.45, solid_capstyle="round", zorder=2)
                for pool_type, value in present.items():
                    ax.plot(
                        x + dx, value, POOL_MARKERS[pool_type],
                        ms=6.5 if pool_type == "cross" else 7.5, color=colour,
                        markeredgecolor=theme.surface, markeredgewidth=1.4, zorder=3,
                    )

        ax.set_xticks(
            range(len(conditions)),
            ["question" if q == "question" else "+ options" for _, q in conditions],
            fontsize=9.5,
        )
        ax.set_xlim(-0.6, len(conditions) - 0.4)
        # Representation named once under its own pair, below the query labels.
        for rep in dict.fromkeys(r for r, _ in conditions):
            xs = [i for i, (r, _) in enumerate(conditions) if r == rep]
            ax.annotate(
                rep, xy=(sum(xs) / len(xs), 0.0), xycoords=("data", "axes fraction"),
                xytext=(0, -26), textcoords="offset points", ha="center", va="top",
                color=theme.ink, fontsize=10.5, fontweight="bold", annotation_clip=False,
            )

        ax.set_ylabel("top-1 oracle retrieval", color=theme.ink_secondary, fontsize=9.5)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_locator(MultipleLocator(0.10))
        ax.grid(axis="y", color=theme.grid, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(theme.baseline)
        ax.tick_params(colors=theme.ink_secondary, labelsize=9, length=0)

        fig.legend(
            handles=[
                *(Line2D([], [], marker=POOL_MARKERS[p], ls="none", ms=7,
                         color=theme.ink_secondary, markeredgecolor=theme.surface,
                         markeredgewidth=1.2, label=name)
                  for p, name in (("within", "same video"), ("cross", "other videos"))),
                *(Patch(facecolor=ENCODER_HUES[e], edgecolor="none", label=e)
                  for e in encoders),
            ],
            loc="lower left", bbox_to_anchor=(0.0, 1.005),
            ncol=len(encoders) + 2, frameon=False,
            handletextpad=0.5, columnspacing=1.5, fontsize=9.5,
            labelcolor=theme.ink_secondary,
        )
        fig.tight_layout()
        if title:
            fig.text(0.0, 1.085, title, color=theme.ink, fontsize=12, ha="left",
                     va="bottom", fontweight="bold")
    return save_figure(fig, path)


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


#: Hues for retrievers, assigned in this fixed order and never cycled. Validated for CVD
#: separation (worst adjacent pair dE 21.9 protan, 31.2 normal) against the light surface.
#: Only retrievers get a hue -- `random` and `oracle` are the floor and ceiling the
#: measured system sits between, so they read as references in recessive ink instead.
#: The same encoder hues under the display names the end-to-end figure labels its
#: conditions with ("top1 (x-clip)"), so one encoder is one colour across every figure.
RETRIEVER_HUES = {
    "x-clip": ENCODER_HUES["xclip"],
    # The trained head sits beside the encoder it wraps, so the pair reads as one comparison.
    # Kept short because these become x tick labels, four to a group.
    "adapted": "#CC79A7",
    "pe-video": ENCODER_HUES["pe-video"],
}


#: The three kinds of chunk in a pool, and the ink each gets. The oracle is the subject, so
#: it takes the strong colour; uninformative chunks are what it loses to.
CHUNK_KINDS = {
    "the oracle": ENCODER_HUES["xclip"],
    "uninformative chunks": "#CC79A7",
    "other chunks": "#898781",
}


def plot_chunk_ordering(
    positions: dict[tuple[str, str], dict[str, float]],
    path: str | FilePath,
    criterion: str,
    n_questions: int,
    title: str = "",
    theme: Theme = LIGHT,
) -> FilePath:
    """
    Where each kind of chunk lands in its pool, as the query changes.

    One panel per backbone, two positions on x: the bare question, then the question with
    its answer options. Three lines, one per kind of chunk. The claim is that the oracle
    line and the uninformative line **cross** -- an interrogative query ranks chunks that
    cannot be evidence above the one that is, and adding declarative text reverses it.

    y is mean normalized rank inverted, so higher on the page means ranked earlier, which
    is the direction a reader assumes without being told.

    Args:
        positions: `{(backbone, query_form): {kind: mean position}}`, positions in [0, 1]
            with 0 = ranked first. Kinds are the keys of `CHUNK_KINDS`.
        criterion: what made a chunk uninformative, stated on the figure. The whole result
            turns on this threshold, so it belongs in the subtitle rather than the caption.
        n_questions: for the subtitle.
        title: drawn above the axes.
    """
    backbones = list(dict.fromkeys(b for b, _ in positions))
    forms = list(dict.fromkeys(f for _, f in positions))

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, len(backbones), figsize=(2.9 * len(backbones) + 0.6, 3.5),
            facecolor=theme.surface, sharey=True,
        )
        axes = axes if len(backbones) > 1 else [axes]

        for ax, backbone in zip(axes, backbones):
            ax.set_facecolor(theme.surface)
            for kind, colour in CHUNK_KINDS.items():
                ys = [positions[(backbone, form)][kind] for form in forms]
                ax.plot(range(len(forms)), ys, color=colour, lw=2.0, zorder=3,
                        marker="o", markersize=7, markeredgecolor=theme.surface, mew=1.5)
                # Both ends carry their value: the reader is comparing where lines start
                # against where they finish, and the crossing is the whole point.
                for x, y, dx, ha in ((0, ys[0], -8, "right"), (len(forms) - 1, ys[-1], 8, "left")):
                    ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(dx, 0),
                                textcoords="offset points", color=colour, fontsize=9,
                                va="center", ha=ha, fontweight="bold")

            ax.set_xticks(range(len(forms)), forms, fontsize=9.5)
            # Room at both ends for the endpoint labels.
            ax.set_xlim(-0.35, len(forms) - 0.55)
            ax.set_title(backbone, color=theme.ink, fontsize=10.5, fontweight="bold", pad=8)
            ax.grid(axis="y", color=theme.grid, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "bottom"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color(theme.baseline)
            ax.tick_params(colors=theme.ink_secondary, labelsize=9, length=0)

        # 0 at the top: ranked first should sit high on the page. The numbers alone leave
        # the reader to work out which direction is good, so both ends say so in words.
        axes[0].set_ylim(0.68, 0.18)
        axes[0].set_ylabel("where the encoder puts it", color=theme.ink_secondary, fontsize=9.5)
        for frac, text, va in ((0.99, "ranked first", "top"), (0.01, "ranked last", "bottom")):
            axes[0].annotate(text, xy=(0, frac), xycoords="axes fraction",
                             xytext=(-46, 0), textcoords="offset points", rotation=90,
                             ha="center", va=va, color=theme.ink_muted, fontsize=8.5)

        fig.tight_layout()
        # Below the panels, in a row: inside an axis it would sit on top of the lines.
        fig.legend(
            handles=[Line2D([], [], color=c, lw=2.0, marker="o", markersize=6, label=k)
                     for k, c in CHUNK_KINDS.items()],
            loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=len(CHUNK_KINDS),
            frameon=False, fontsize=9.5, labelcolor=theme.ink_secondary,
        )
        if title:
            fig.text(0.0, 1.06, title, color=theme.ink, fontsize=12, ha="left",
                     va="bottom", fontweight="bold")
        fig.text(0.0, 0.99, f"uninformative: {criterion} · n = {n_questions:,} questions",
                 color=theme.ink_secondary, fontsize=9.5, ha="left", va="bottom")
    return save_figure(fig, path)


def plot_condition_accuracy(
    counts: dict[tuple[str, str], tuple[int, int]],
    path: str | FilePath,
    chance: float = 0.25,
    title: str = "",
    theme: Theme = LIGHT,
) -> FilePath:
    """
    QA accuracy per model under each retrieval condition, with 95% Wilson intervals.

    Dots and intervals rather than bars: chance is 25%, so a bar growing from zero would
    encode a quantity nobody cares about and invite the eye to compare areas. Dots carry
    no area, which also makes the truncated accuracy axis honest -- necessary when every
    result sits between 0.5 and 0.7 and the differences that matter are a few points wide.

    Accuracy on y, conditions grouped by model on x, so comparing models is one horizontal
    scan against a shared scale. `oracle` is the ceiling a perfect retriever reaches and
    `random` the floor of picking blind; both are drawn in recessive ink because they are
    references, not systems, and the shaded span between them is the headroom on offer.
    The figure's claim is where the coloured retriever dot falls inside that span.

    Args:
        counts: `{(model, condition): (hits, n)}`. Conditions are "oracle", "random", or
            "top1 (<retriever>)"; the retriever name selects the hue.
        chance: the multiple-choice floor, reported in the subtitle.
    """
    models = list(dict.fromkeys(model for model, _ in counts))
    order = ["random"] + [f"top1 ({r})" for r in RETRIEVER_HUES] + ["oracle"]
    present = [c for c in order if any(cond == c for _, cond in counts)]

    def hue(condition: str) -> str:
        for name, colour in RETRIEVER_HUES.items():
            if condition == f"top1 ({name})":
                return colour
        return theme.ink if condition == "oracle" else theme.ink_muted

    def short(condition: str) -> str:
        return condition.replace("top1 (", "").rstrip(")") if "top1" in condition else condition

    bounds = [wilson_interval(h, n) for h, n in counts.values()]
    lo_y = min(low for low, _ in bounds) - 0.02
    hi_y = max(high for _, high in bounds) + 0.03

    with plt.rc_context(PAPER_RC):
        width = max(3.4, 0.62 * len(models) * len(present) + 1.0)
        fig, ax = plt.subplots(figsize=(width, 3.4), facecolor=theme.surface)
        ax.set_facecolor(theme.surface)
        # Chance is 25% and every condition clears it by 30 points, so plotting it would
        # spend most of the axis on empty space and squeeze the 5-point effect that is the
        # actual subject. The floor goes in the subtitle instead.
        if lo_y <= chance <= hi_y:
            ax.axhline(chance, ls=(0, (4, 3)), lw=1.4, color=theme.ink_muted, zorder=1)

        ticks, labels = [], []
        col = 0.0
        for model in models:
            group = [c for c in present if (model, c) in counts]
            span = {c: counts[(model, c)][0] / counts[(model, c)][1]
                    for c in ("random", "oracle") if (model, c) in counts}
            if len(span) == 2:
                # The headroom a retriever could win, as a vertical span behind the group.
                ax.bar(col + (len(group) - 1) / 2, span["oracle"] - span["random"],
                       bottom=span["random"], width=len(group) - 0.15,
                       color=theme.grid, zorder=0)

            for cond in group:
                hits, n = counts[(model, cond)]
                value = hits / n
                low, high = wilson_interval(hits, n)
                colour = hue(cond)
                ax.plot([col, col], [low, high], lw=1.6, color=colour, zorder=3)
                ax.plot(col, value, "o", ms=7, color=colour, zorder=4,
                        markeredgecolor=theme.surface, markeredgewidth=1.2)
                ax.annotate(
                    f"{value:.1%}", xy=(col, high), xytext=(0, 6),
                    textcoords="offset points", ha="center",
                    color=theme.ink_secondary, fontsize=9,
                )
                ticks.append(col)
                labels.append(short(cond))
                col += 1
            # Model name centred under its own group, below the condition labels.
            ax.annotate(
                model, xy=(col - (len(group) + 1) / 2, 0.0), xycoords=("data", "axes fraction"),
                xytext=(0, -22), textcoords="offset points", ha="center", va="top",
                color=theme.ink, fontsize=10.5, fontweight="bold", annotation_clip=False,
            )
            col += 0.9

        ax.set_xticks(ticks, labels, fontsize=9.5)
        ax.set_xlim(-0.8, col - 0.3)
        ax.set_ylabel("QA accuracy", color=theme.ink_secondary, fontsize=9.5)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        # Whole 5-point steps. The default locator lands on 2.5-point steps, which round
        # to an uneven 52/55/57/60/62 ladder and make the axis look mis-drawn.
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_ylim(lo_y, hi_y)
        ax.grid(axis="y", color=theme.grid, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(theme.baseline)
        ax.tick_params(colors=theme.ink_secondary, labelsize=9, length=0)

        total = max(n for _, n in counts.values())
        fig.tight_layout()
        if title:
            fig.text(0.0, 1.10, title, color=theme.ink, fontsize=12, ha="left",
                     va="bottom", fontweight="bold")
        fig.text(0.0, 1.00, f"n = {total:,} questions · 95% Wilson intervals · "
                            f"band spans random to oracle · chance {chance:.0%}",
                 color=theme.ink_secondary, fontsize=9.5, ha="left", va="bottom")
    return save_figure(fig, path)
