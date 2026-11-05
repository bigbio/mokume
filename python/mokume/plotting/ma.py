"""MA plots from DE log2 group means or the corresponding protein matrix."""

import pandas as pd
from matplotlib import pyplot as plt

from mokume.plotting._common import log2_matrix, static_style, style_axes
from mokume.plotting.style import (
    category_colors,
    category_order,
    MISSING_COLOR,
    SAMPLE_POINT_COLOR,
    SIGNIFICANCE_COLORS,
    TEXT_COLOR,
    WHITE,
)


def read_protein_groups(path):
    """Read the explicit protein-to-group mapping used for spike-in MA plots."""
    groups = pd.read_csv(path, sep="\t").set_index("ProteinName")["group"]
    if groups.index.has_duplicates:
        raise ValueError("Protein groups must contain unique ProteinName values")
    return groups


def ma_values(de, protein_df, conditions, sample_to_condition):
    """Return A in log2 units; DE mean columns already contain log2 means."""
    columns = [f"mean_{condition}" for condition in conditions]
    if all(column in de for column in columns):
        return de[columns].mean(axis=1)
    matrix = log2_matrix(protein_df)
    means = pd.DataFrame(
        {
            condition: matrix[
                [
                    sample
                    for sample in matrix
                    if sample_to_condition.get(sample) == condition
                ]
            ].mean(axis=1)
            for condition in conditions
        }
    )
    return de["ProteinName"].map(means.mean(axis=1))


@static_style()
def plot_ma(de, protein_df, sample_to_condition, output_file, options):
    """Draw significance or composition groups with optional expected ratios."""
    abundance = ma_values(de, protein_df, options["conditions"], sample_to_condition)
    groups = options.get("groups")
    labels, names = _group_labels(de, groups)
    # Significance classes keep their fixed order; user groups sort like other categories.
    if groups is not None and len(names) > 3:
        _plot_facets(de, abundance, labels, output_file, options)
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    _plot_groups(ax, de, abundance, labels, names, options)
    for value in (-options["log2fc"], options["log2fc"]):
        ax.axhline(value, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0, color=SAMPLE_POINT_COLOR, linewidth=0.6)
    ax.set(
        xlabel="Mean log2 intensity (A)",
        ylabel="log2 fold change (M)",
        title=options["title"] + (f" ({names[0]})" if len(names) == 1 else ""),
    )
    if len(names) > 1:
        legend = ax.legend(
            labelcolor=TEXT_COLOR,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(len(names), 3),
            frameon=False,
            markerscale=2.5,
        )
        # Density marks are translucent; legend keys show the full group color.
        for handle in legend.legend_handles:
            handle.set_alpha(1)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def _group_labels(de, groups):
    """Put neutral significance points first and retain unmapped proteins."""
    if groups is None:
        labels = de["significance"]
        names = [
            name
            for name in ("Unchanged", "NotTested", "UP", "DOWN")
            if name in set(labels)
        ]
    else:
        labels = pd.Series(
            groups.reindex(de["ProteinName"]).fillna("Unassigned").to_numpy(),
            index=de.index,
        )
        names = category_order(labels)
    return labels, names


def _plot_groups(ax, de, abundance, labels, names, options):
    """Draw each composition/significance group and its expected-ratio line."""
    groups = options.get("groups")
    colors = (
        SIGNIFICANCE_COLORS
        if groups is None
        else dict(zip(names, category_colors(len(names))))
    )
    for name in names:
        color = colors[name]
        subset = labels == name
        _group_points(ax, abundance[subset], de.loc[subset, "log2FC"], color, name)
        if name in options.get("expected", {}):
            value = options["expected"][name]
            ax.axhline(value, color=color, linestyle="--", linewidth=2)
    _label_expected(ax, names, options.get("expected", {}))


def _group_points(ax, abundance, fold_changes, color, name=None):
    """Use the same density marks in grouped and faceted MA plots."""
    ax.scatter(
        abundance,
        fold_changes,
        color=color,
        marker="o",
        s=8,
        alpha=0.6,
        linewidths=0,
        label=name,
        rasterized=True,
    )


def _plot_facets(de, abundance, labels, output_file, options):
    """Emphasize one composition group per panel against the other proteins."""
    names = category_order(labels)
    rows = (len(names) + 2) // 3
    fig, axes = plt.subplots(
        rows, 3, figsize=(7.2, rows * 2.4), sharex=True, sharey=True, squeeze=False
    )
    for index, (name, color) in enumerate(zip(names, category_colors(len(names)))):
        ax = axes.flat[index]
        subset = labels == name
        _group_points(ax, abundance[~subset], de.loc[~subset, "log2FC"], MISSING_COLOR)
        _group_points(ax, abundance[subset], de.loc[subset, "log2FC"], color)
        for value in (-options["log2fc"], options["log2fc"]):
            ax.axhline(value, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(0, color=SAMPLE_POINT_COLOR, linewidth=0.6)
        if name in options.get("expected", {}):
            ax.axhline(
                options["expected"][name], color=color, linestyle="--", linewidth=2
            )
        _label_expected(ax, [name], options.get("expected", {}))
        ax.set_title(name)
        style_axes(ax)
        ax.tick_params(labelbottom=index + 3 >= len(names))
    for ax in list(axes.flat)[len(names) :]:
        fig.delaxes(ax)
    fig.suptitle(options["title"])
    fig.supxlabel("Mean log2 intensity (A)", fontsize=10)
    fig.supylabel("log2 fold change (M)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def _label_expected(ax, names, expected):
    """Write one label per expected ratio so groups sharing it do not overprint."""
    shared = {}
    for name in names:
        if name in expected:
            shared.setdefault(expected[name], []).append(name)
    for value, members in shared.items():
        ax.text(
            0.01,
            value,
            f"{', '.join(members)}: {value:g}",
            color=TEXT_COLOR,
            transform=ax.get_yaxis_transform(),
            va="bottom",
            fontsize=8,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.85, "pad": 1},
        )
