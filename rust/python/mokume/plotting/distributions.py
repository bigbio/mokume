"""Condition-ordered density and box plots for quantification QC reports."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mokume.plotting._common import ordered_samples, static_style, style_axes
from mokume.plotting.style import (
    category_colors,
    category_order,
    SAMPLE_POINT_COLOR,
    TEXT_COLOR,
)


def distribution_groups(dataset, class_field):
    """Return sample order and colors; absent condition sets are neutral."""
    conditions = (
        dataset.drop_duplicates(class_field)
        .set_index(class_field)["Condition"]
        .to_dict()
        if "Condition" in dataset
        else {}
    )
    order = ordered_samples(dataset[class_field].unique(), conditions)
    names = category_order(conditions.values())
    palette = dict(zip(names, category_colors(len(names))))
    colors = {
        sample: palette.get(conditions.get(sample), SAMPLE_POINT_COLOR)
        for sample in order
    }
    return order, colors, palette


def _distribution_data(dataset, field, class_field, log2):
    """Prepare finite values without changing pandas or numpy global options."""
    frame = dataset[[field, class_field]].copy()
    if log2:
        frame[field] = np.log2(frame[field].where(frame[field] > 0))
    return frame.loc[np.isfinite(frame[field])]


def _condition_legend(ax, palette, filled=False):
    """Describe conditions rather than repeating a legend entry for each sample."""
    if len(palette) > 1:
        ax.legend(
            handles=[
                Patch(facecolor=color, alpha=0.5, label=name)
                if filled
                else Line2D([], [], color=color, linewidth=2, label=name)
                for name, color in palette.items()
            ],
            labelcolor=TEXT_COLOR,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )


@static_style()
def plot_distributions(
    dataset: pd.DataFrame,
    field: str,
    class_field: str,
    title: str = "",
    log2: bool = True,
    width: float = 10,
) -> Figure:
    """
    Plot one density curve per class, colored by its condition.

    Samples use the dataset's ``Condition`` colors; the palette expands for
    additional conditions. Without conditions, curves are neutral.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame to plot.
    field : str
        Field that would be used in the dataframe to plot the density.
    class_field : str
        Field to group the densities into classes.
    title : str, optional
        Title of the plot.
    log2 : bool, optional
        Log the intensity values.
    width : float, optional
        Size of the plot.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    order, colors, palette = distribution_groups(dataset, class_field)
    frame = _distribution_data(dataset, field, class_field, log2)
    fig, ax = plt.subplots(figsize=(min(width, 7.2), 3.2))
    for sample in order:
        sns.kdeplot(
            frame.loc[frame[class_field] == sample, field],
            color=colors[sample],
            linewidth=1,
            ax=ax,
        )
    _condition_legend(ax, palette)
    ax.set(
        title=f"{title} ({len(order)} samples)",
        xlabel=f"{'log2 ' if log2 else ''}{field}",
        ylabel="Density",
    )
    style_axes(ax)
    fig.tight_layout()
    return fig


@static_style()
def plot_box_plot(
    dataset: pd.DataFrame,
    field: str,
    class_field: str,
    log2: bool = False,
    width: float = 10,
    rotation: int = 30,
    title: str = "",
    violin: bool = False,
) -> Figure:
    """
    Plot a box plot of a value field per class, ordered by condition.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataframe with peptide intensities.
    field : str
        Intensity field.
    class_field : str
        Class to group the peptides.
    log2 : bool, optional
        Transform peptide intensities to log scale.
    width : float, optional
        Size of the plot.
    rotation : int, optional
        Rotation of the x-axis labels when samples are many or names are long.
    title : str, optional
        Title of the box plot.
    violin : bool, optional
        Draw violins instead of boxes.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    order, colors, palette = distribution_groups(dataset, class_field)
    values = _distribution_data(dataset, field, class_field, log2).groupby(
        class_field, observed=True
    )[field]
    order = [sample for sample in order if sample in values.groups]
    fig, ax = plt.subplots(figsize=(min(width, 7.2), 3.4))
    _box_bodies(
        ax,
        [values.get_group(sample).to_numpy() for sample in order],
        [colors[sample] for sample in order],
        violin,
    )
    rotation = (
        rotation
        if len(order) > 10 or max(map(len, map(str, order)), default=0) > 12
        else 0
    )
    ax.set_xticks(
        range(1, len(order) + 1),
        order,
        rotation=rotation,
        ha="right" if rotation else "center",
    )
    ax.set(
        title=f"{title} ({len(order)} samples)",
        xlabel="Sample",
        ylabel=f"{'log2 ' if log2 else ''}{field}",
    )
    _condition_legend(ax, palette, filled=True)
    style_axes(ax)
    fig.tight_layout()
    return fig


def _box_bodies(ax, values, colors, violin):
    """Draw translucent box/violin bodies with neutral medians and outliers."""
    if not values:
        return
    if violin:
        bodies = ax.violinplot(values, showmedians=True)["bodies"]
    else:
        boxes = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.7,
            medianprops={"color": TEXT_COLOR},
            flierprops={
                "marker": ".",
                "markersize": 3,
                "markerfacecolor": SAMPLE_POINT_COLOR,
                "markeredgecolor": SAMPLE_POINT_COLOR,
            },
        )
        bodies = boxes["boxes"]
        for points in boxes["fliers"]:
            points.set_rasterized(True)
    for body, color in zip(bodies, colors):
        body.set_facecolor(color)
        body.set_alpha(0.5)
