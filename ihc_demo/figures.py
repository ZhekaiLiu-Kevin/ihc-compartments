"""Figures for the synthetic brightfield immunohistochemistry demonstration."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator


INK = "#172e3e"
MUTED = "#536572"
GRID = "#dce3e5"
TEAL = "#087f81"
OCHRE = "#b65c17"
BLUE = "#3c6098"

# A sequential scale with a light-to-dark luminance gradient for optical density.
DAB_CMAP = LinearSegmentedColormap.from_list(
    "ihc_dab_od", ["#fffcf2", "#e3bf84", "#b78243", "#754323", "#351b0e"]
)

PHENOTYPES = (
    ("nuclear-enriched", BLUE, "s"),
    ("balanced", "#6c7656", "D"),
    ("ring-enriched", OCHRE, "o"),
)


def _zoom_window(labels):
    """Choose a populated central crop using the label mask alone."""
    height, width = labels.shape
    size = max(1, round(min(height, width) * 0.36))
    best_count = -1
    best_y = best_x = 0
    # Try the centered window first. Ties keep the first window.
    for fy, fx in ((0.5, 0.5), (0.35, 0.35), (0.35, 0.65),
                   (0.65, 0.35), (0.65, 0.65)):
        y = int(round((height - size) * fy))
        x = int(round((width - size) * fx))
        labels_in_crop = np.unique(labels[y:y + size, x:x + size])
        count = np.count_nonzero(labels_in_crop)
        if count > best_count:
            best_count = count
            best_y, best_x = y, x
    return best_y, best_x, size


def _inner_outline(labels):
    """One-pixel inner edges for each nucleus, including touching labels."""
    padded = np.pad(labels, 1, mode="constant")
    edges = (
        (labels != padded[:-2, 1:-1]) | (labels != padded[2:, 1:-1])
        | (labels != padded[1:-1, :-2]) | (labels != padded[1:-1, 2:])
    )
    return edges & (labels > 0)


def _panel_title(ax, text):
    ax.set_title(text, loc="left", size=13, weight="semibold", color=INK, pad=10)


def overview_figure(rgb, labels, rings, dab_od, cells):
    """Display the analysis field, compartment masks, and optical density.

    The two enlarged panels show identical pixels. The ring tint spans every
    measured ring pixel, while a one-pixel outline locates each nuclear mask.
    """
    height, width = labels.shape
    y, x, size = _zoom_window(labels)
    crop = np.s_[y:y + size, x:x + size]
    overlay = np.asarray(rgb, dtype=np.float64) / 255.0
    ring_pixels = rings > 0
    overlay[ring_pixels] = 0.48 * overlay[ring_pixels] + 0.52 * np.array(
        [0.05, 0.66, 0.66]
    )
    overlay[_inner_outline(labels)] = [0.95, 0.52, 0.05]

    fig = plt.figure(figsize=(12, 5.2), facecolor="white")
    fig.text(0.033, 0.934, "Measuring nuclear and perinuclear DAB",
             fontsize=21, fontweight="semibold", color=INK)
    fig.text(0.034, 0.874,
             f"Simulated IHC field  ·  {len(cells)} nuclei  ·  {width} × {height} pixels",
             fontsize=11, color=MUTED)
    axes = [fig.add_axes([left, 0.172, 0.291, 0.573])
            for left in (0.033, 0.354, 0.675)]
    ax_full, ax_overlay, ax_dab = axes
    ax_full.imshow(rgb, interpolation="nearest")
    # Double outline remains legible against both dark and pale backgrounds.
    for color, weight in (("white", 3.5), (INK, 1.2)):
        ax_full.add_patch(Rectangle((x - 0.5, y - 0.5), size, size,
                                    linewidth=weight, edgecolor=color, fill=False))
    ax_overlay.imshow(overlay[crop], interpolation="nearest")
    max_od = max(0.1, np.ceil(10 * np.nanmax(dab_od[crop])) / 10.0)
    image = ax_dab.imshow(dab_od[crop], cmap=DAB_CMAP, vmin=0, vmax=max_od,
                          interpolation="nearest")
    for ax, title in zip(axes, ("A   Full field", "B   Compartment masks", "C   DAB intensity")):
        _panel_title(ax, title)
        ax.set_axis_off()

    ax_full.text(0, -0.08, "Frame enlarged in B and C.", transform=ax_full.transAxes,
                 va="top", color=MUTED, fontsize=9)
    ax_overlay.legend(
        handles=[Line2D([0], [0], color="#e57e0e", lw=2.5, label="Nuclear edge"),
                 Patch(facecolor=TEAL, alpha=0.72, label="Perinuclear mask")],
        loc="upper left", bbox_to_anchor=(0, -0.045), borderaxespad=0,
        frameon=False, fontsize=9, ncol=2, handlelength=1.1, columnspacing=0.9,
    )
    box = ax_dab.get_position()
    cb_ax = fig.add_axes([box.x0, box.y0 - 0.051, box.width * 0.75, 0.015])
    bar = fig.colorbar(image, cax=cb_ax, orientation="horizontal",
                       ticks=[0, max_od / 2, max_od])
    bar.outline.set_visible(False)
    cb_ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=2, pad=2)
    cb_ax.set_xlabel("DAB optical density", color=MUTED, fontsize=9, labelpad=2)
    return fig


def _style_plot(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#bac7cd")
    ax.tick_params(colors=MUTED, length=3, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)


def qc_figure(cells, truth, sensitivity):
    """Compare compartment estimates with simulation targets and sweep ring size.

    Scatter includes all finite pairs. Ribbons show interquartile ranges across
    cells with a finite ratio in each phenotype, computed separately per scale.
    """
    data = cells.merge(truth, on="cell_id", validate="one_to_one")
    fig = plt.figure(figsize=(11.7, 5.4), facecolor="white")
    fig.text(0.065, 0.932, "Compartment recovery and ring-width sensitivity",
             fontsize=20, fontweight="semibold", color=INK)
    fig.text(0.066, 0.867, "Synthetic ground truth and within-field sensitivity analysis",
             color=MUTED, fontsize=11)
    ax_recovery = fig.add_axes([0.087, 0.245, 0.365, 0.488])
    ax_sensitivity = fig.add_axes([0.575, 0.245, 0.377, 0.488])
    for ax in (ax_recovery, ax_sensitivity):
        _style_plot(ax)
    _panel_title(ax_recovery, "A   Target vs. measured stain")
    _panel_title(ax_sensitivity, "B   Effect of the ring definition")

    upper = 0.1
    for prefix, name, color, marker in (
        ("nuclear", "Nucleus", BLUE, "o"), ("ring", "Perinuclear ring", OCHRE, "^")
    ):
        target = data[f"target_{prefix}_dab_od"].to_numpy(dtype=float)
        measured = data[f"{prefix}_mean_dab_od"].to_numpy(dtype=float)
        valid = np.isfinite(target) & np.isfinite(measured)
        if valid.any():
            upper = max(upper, target[valid].max(), measured[valid].max())
        ax_recovery.scatter(target[valid], measured[valid], marker=marker, s=24,
                            alpha=0.7, color=color, edgecolors="white",
                            linewidths=0.35, label=name, zorder=3)
    upper = np.ceil((upper * 1.03) * 10) / 10
    ax_recovery.plot([0, upper], [0, upper], color=MUTED, lw=1.1,
                     linestyle=(0, (4, 3)), label="Identity", zorder=2)
    ax_recovery.set(xlim=(0, upper), ylim=(0, upper),
                    xlabel="Simulated target DAB (OD)", ylabel="Measured mean DAB (OD)")
    ax_recovery.xaxis.set_major_locator(MaxNLocator(5))
    ax_recovery.yaxis.set_major_locator(MaxNLocator(5))
    ax_recovery.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.24), ncol=3,
                       frameon=False, fontsize=8.7, columnspacing=1.0, handlelength=1.4)

    ax_sensitivity.axhline(1, color=MUTED, lw=0.9, ls="--", zorder=1)
    for name, color, marker in PHENOTYPES:
        selection = sensitivity.loc[
            (sensitivity["phenotype"] == name)
            & np.isfinite(sensitivity["ring_to_nucleus_ratio"])
        ]
        if selection.empty:
            continue
        grouped = selection.groupby("outer_scale")["ring_to_nucleus_ratio"]
        q = grouped.quantile([0.25, 0.5, 0.75]).unstack().sort_index()
        scales = q.index.to_numpy(dtype=float)
        ax_sensitivity.fill_between(scales, q[0.25].to_numpy(), q[0.75].to_numpy(),
                                    alpha=0.15, linewidth=0, color=color)
        ax_sensitivity.plot(scales, q[0.5].to_numpy(), label=name.capitalize(),
                            marker=marker, markersize=4, lw=1.75, color=color)
    ax_sensitivity.set_xlabel("Outer radius / nuclear radius")
    ax_sensitivity.set_ylabel("Perinuclear / nuclear DAB")
    ax_sensitivity.set_ylim(bottom=0)
    scales = np.sort(sensitivity["outer_scale"].dropna().unique())
    if 0 < len(scales) <= 9:
        ax_sensitivity.set_xticks(scales)
    ax_sensitivity.yaxis.set_major_locator(MaxNLocator(5))
    ax_sensitivity.margins(x=0.035)
    if ax_sensitivity.get_legend_handles_labels()[0]:
        ax_sensitivity.legend(loc="upper left", bbox_to_anchor=(-0.035, -0.24),
                              frameon=False, fontsize=8.4, ncol=3, columnspacing=0.8,
                              handlelength=1.25)
    fig.text(0.066, 0.035, "A  One point per cell and compartment.   "
             "B  Lines show cell medians. Bands span the middle 50% of cells.",
             color=MUTED, fontsize=9)
    return fig
