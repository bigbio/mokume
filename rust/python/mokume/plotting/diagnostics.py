"""Abundance ranks and multi-contrast summaries without extra dependencies."""

from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from mokume.plotting._common import static_style, style_axes
from mokume.plotting.style import (
    CONDITION_COLORS,
    INACTIVE_COLOR,
    SAMPLE_POINT_COLOR,
    SINGLE_SERIES_COLOR,
    TEXT_COLOR,
)


@static_style()
def plot_abundance_rank(log2_values, highlight_genes=None):
    """Rank proteins by observed mean log2 abundance."""
    abundance = log2_values.mean(axis=1).dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ranks = np.arange(1, len(abundance) + 1)
    ax.plot(ranks, abundance, color=SINGLE_SERIES_COLOR, linewidth=2)
    for protein in highlight_genes or []:
        if protein in abundance.index:
            rank = abundance.index.get_loc(protein) + 1
            ax.scatter(
                rank,
                abundance[protein],
                color=SINGLE_SERIES_COLOR,
                s=36,
                edgecolors=TEXT_COLOR,
                linewidths=0.6,
                zorder=3,
            )
            ax.annotate(
                protein,
                (rank, abundance[protein]),
                xytext=(5, 5),
                textcoords="offset points",
                color=TEXT_COLOR,
            )
    ax.set(xlabel="Protein rank", ylabel="Mean log2 intensity", title="Abundance rank")
    style_axes(ax)
    fig.tight_layout()
    return fig


def upset_counts(sets):
    """Count exclusive membership patterns, as in a standard UpSet plot."""
    return Counter(
        tuple(name for name, members in sets.items() if protein in members)
        for protein in set().union(*sets.values())
    )


@static_style()
def plot_upset(sets, output_file):
    """Show intersection sizes, set sizes and the membership matrix."""
    counts = upset_counts(sets)
    intersections = sorted(counts, key=lambda key: (-counts[key], key))
    names = list(sets)
    fig = plt.figure(figsize=(max(4.5, len(intersections) * 0.32), 3.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[2, 1])
    top, left, matrix = (
        fig.add_subplot(grid[position]) for position in ((0, 1), (1, 0), (1, 1))
    )
    bars = top.bar(
        range(len(intersections)),
        [counts[key] for key in intersections],
        color=SINGLE_SERIES_COLOR,
        width=0.7,
        linewidth=0,
    )
    top.bar_label(bars, color=TEXT_COLOR)
    top.margins(y=0.12)
    left.barh(
        range(len(names)),
        [len(sets[name]) for name in names],
        color=SINGLE_SERIES_COLOR,
        height=0.7,
        linewidth=0,
    )
    _intersection_matrix(matrix, names, intersections)
    top.set(
        ylabel="Intersection size",
        title="Significant proteins across contrasts",
        xticks=[],
    )
    left.set(xlabel="Set size", yticks=range(len(names)), yticklabels=names)
    matrix.set(xticks=[], yticks=range(len(names)), yticklabels=[])
    matrix.tick_params(left=False)
    left.invert_xaxis()
    for ax in (top, left, matrix):
        style_axes(ax)
    matrix.set_ylim(left.get_ylim())
    # The first count is wider than its bar; a half-step keeps it off the y axis.
    low, high = matrix.get_xlim()
    top.set_xlim(low - 0.5, high)
    matrix.set_xlim(low - 0.5, high)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def _intersection_matrix(ax, names, intersections):
    """Draw active membership points over a light inactive grid."""
    for x, intersection in enumerate(intersections):
        enabled = [index for index, name in enumerate(names) if name in intersection]
        ax.scatter(
            [x] * len(names),
            range(len(names)),
            s=64,
            color=INACTIVE_COLOR,
            linewidths=0,
        )
        ax.plot([x] * len(enabled), enabled, color=SAMPLE_POINT_COLOR, linewidth=2)
        ax.scatter(
            [x] * len(enabled),
            enabled,
            s=64,
            color=SAMPLE_POINT_COLOR,
            linewidths=0,
            zorder=3,
        )


def _fold_change_points(ax, tables, protein, color, offset):
    """Use filled and open dots for significant and nonsignificant values."""
    for index, table in enumerate(tables.values()):
        row = table.set_index("ProteinName").reindex([protein]).iloc[0]
        significant = row["significance"] in ("UP", "DOWN")
        ax.scatter(
            index + offset,
            row["log2FC"],
            s=24,
            alpha=0.85,
            linewidths=0 if significant else 1,
            edgecolors=color,
            facecolors=color if significant else "none",
            zorder=3,
        )


def _legends_below(fig, groups):
    """Stack frameless legends under the settled axes."""
    renderer = fig.canvas.get_renderer()
    top = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes if ax.get_visible())
    for handles in groups:
        legend = fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, top / fig.bbox.height),
            ncol=min(3, len(handles)),
            frameon=False,
            labelcolor=TEXT_COLOR,
        )
        top = legend.get_window_extent(renderer).y0


def _dot_handle(label, color, filled=True):
    """Build a dot-only legend entry matching the plotted dots."""
    return Line2D(
        [],
        [],
        linestyle="",
        marker="o",
        markersize=5,
        alpha=0.85,
        markeredgewidth=0 if filled else 1,
        markeredgecolor=color,
        markerfacecolor=color if filled else "none",
        label=label,
    )


def _contrast_axis(ax, names):
    """Mark zero and label the unordered contrasts."""
    rotation = 30 if max(map(len, names)) > 8 else 0
    ax.axhline(0, color=SAMPLE_POINT_COLOR, linewidth=0.6)
    ax.set_xticks(
        range(len(names)),
        names,
        rotation=rotation,
        ha="right" if rotation else "center",
    )
    ax.set_xlim(-0.5, len(names) - 0.5)
    style_axes(ax)


def _overlay_proteins(ax, tables, proteins):
    """Dodge up to six colored proteins within each contrast."""
    spacing = min(0.12, 0.6 / len(proteins))
    for index, (protein, color) in enumerate(zip(proteins, CONDITION_COLORS)):
        offset = (index - (len(proteins) - 1) / 2) * spacing
        _fold_change_points(ax, tables, protein, color, offset)
    ax.set_title("log2FC across contrasts")
    ax.set_ylabel("log2 fold change")
    return [
        _dot_handle(protein, color)
        for protein, color in zip(proteins, CONDITION_COLORS)
    ]


def _protein_panels(fig, axes, tables, proteins):
    """Give each protein its own panel and hide the unused ones."""
    for ax, protein in zip(axes.flat, proteins):
        _fold_change_points(ax, tables, protein, SINGLE_SERIES_COLOR, 0)
        ax.set_title(protein)
    for ax in list(axes.flat)[len(proteins) :]:
        ax.set_visible(False)
    fig.supylabel(
        "log2 fold change", color=TEXT_COLOR, fontsize=plt.rcParams["axes.labelsize"]
    )


@static_style()
def plot_fold_changes(tables, proteins, output_file):
    """Compare fold changes, switching to small multiples above six proteins.

    Contrasts are unordered, so markers are dodged within each contrast rather
    than joined. Confidence intervals are omitted: DE tables carry no standard
    errors.
    """
    multiple = len(proteins) > 6
    rows = int(np.ceil(len(proteins) / 3)) if multiple else 1
    fig, axes = plt.subplots(
        rows,
        3 if multiple else 1,
        figsize=(7.2, rows * 2.4) if multiple else (3.5, 3),
        squeeze=False,
        sharey=True,
    )
    if multiple:
        _protein_panels(fig, axes, tables, proteins)
        groups = []
    else:
        groups = [_overlay_proteins(axes[0, 0], tables, proteins)]
    for ax in axes.flat:
        _contrast_axis(ax, list(tables))
    fig.tight_layout()
    groups.append(
        [
            _dot_handle(label, SAMPLE_POINT_COLOR, filled)
            for label, filled in (("Significant", True), ("Not significant", False))
        ]
    )
    _legends_below(fig, groups)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
