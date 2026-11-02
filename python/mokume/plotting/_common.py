"""Local plot styling and matrix preparation shared by diagnostic plots."""

from contextlib import contextmanager
from textwrap import fill

import numpy as np
import pandas as pd

from mokume.plotting.style import (
    category_order,
    MATPLOTLIB_FONTS,
    MATPLOTLIB_STYLE,
    SAMPLE_POINT_COLOR,
    TEXT_COLOR,
)


@contextmanager
def static_style():
    """Scope static-plot defaults to creation and export, restoring caller settings.

    Prefer an installed Latin sans-serif, followed by the caller's installed
    font families for glyphs such as Chinese sample names. Import matplotlib
    only when plotting, since matrix helpers also serve HTML-only reports.
    """
    from matplotlib import font_manager, rc_context, rcParams

    installed = {
        font.name.casefold(): font.name for font in font_manager.fontManager.ttflist
    }
    primary = next(
        (name for name in MATPLOTLIB_FONTS if name.casefold() in installed), None
    )
    configured = []
    for family in rcParams["font.family"]:
        configured.extend(rcParams.get(f"font.{family}", [family]))
    families = [primary] if primary else []
    families.extend(
        installed[name.casefold()]
        for name in configured
        if name.casefold() in installed
    )
    with rc_context({**MATPLOTLIB_STYLE, "font.family": list(dict.fromkeys(families))}):
        yield


def style_axes(ax):
    """Style axes locally; static_style scopes defaults without global changes."""
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR, direction="out")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(TEXT_COLOR)
        ax.spines[side].set_linewidth(0.8)


def short_protein_name(name):
    """A protein group's first member, as its entry name when written db|accession|entry."""
    return str(name).split(";", maxsplit=1)[0].split("|")[-1]


def volcano_label_rows(frame, labelled, count=10):
    """The ``count`` changed proteins with the lowest p-values, besides labelled ones."""
    changed = frame[
        frame["significance"].isin(("UP", "DOWN"))
        & ~frame["ProteinName"].isin(labelled)
    ]
    order = changed.assign(size=-changed["log2FC"].abs()).sort_values(
        ["pvalue", "size"], kind="stable"
    )
    return changed.loc[order.index[:count]]


def nearest_anchor(points):
    """Choose the observed point nearest the coordinate-wise group median."""
    points = np.asarray(points)
    return points[np.argmin(np.linalg.norm(points - np.median(points, axis=0), axis=1))]


def _overlap_count(box, boxes):
    """Count overlapping display-space rectangles."""
    rectangles = np.asarray(boxes).reshape(-1, 4)
    return np.count_nonzero(
        (box[..., 0] < rectangles[:, 2])
        & (box[..., 2] > rectangles[:, 0])
        & (box[..., 1] < rectangles[:, 3])
        & (box[..., 3] > rectangles[:, 1])
    )


def _label_candidates(size, center, bounds, radii, directions=8):
    """Try evenly spaced directions at increasing distances within the label bounds."""
    half = np.asarray(size) / 2
    for radius in radii:
        for angle in np.arange(directions) * 2 * np.pi / directions:
            direction = np.array([np.cos(angle), np.sin(angle)])
            point = center + direction * (radius + half)
            point = np.clip(point, bounds[:2] + half, bounds[2:] - half)
            yield point, np.r_[point - half, point + half]


def _label_position(size, center, placed, layout, fit="required"):
    """Prefer nearby clear positions; far or compromised labels get a leader.

    With ``fit="optional"`` a label is dropped (None) when every position would
    cover another label; with ``fit="near"`` when no position beside its anchor
    is clear of labels, points and leaders, and whose own leader misses every label.
    """
    bounds, points, scale = layout["bounds"], layout["points"], layout["scale"]
    candidates = list(
        _label_candidates(size, center, bounds, np.arange(4, 25, 4) * scale)
    )
    available = [
        (point, box) for point, box in candidates if not _overlap_count(box, placed)
    ]
    clear = [
        (point, box) for point, box in available if not _overlap_count(box, points)
    ]
    if fit == "near":
        clear = _clear_of_leaders(clear, center, placed, layout)
    if clear:
        point, box = clear[0]
        # As in ggrepel, a label pushed past the nearest rings keeps a leader.
        gap = np.linalg.norm(center - np.clip(center, box[:2], box[2:]))
        return point, box, gap > 8 * scale
    if fit == "near":
        return None
    if available:
        point, box = min(available, key=lambda item: _overlap_count(item[1], points))
        return point, box, True
    # Prefer a longer leader to overlap, keeping a nearby fallback for dense plots.
    radii = np.arange(28 * scale, np.linalg.norm(bounds[2:] - bounds[:2]), 4 * scale)
    far = next(
        (
            (point, box)
            for point, box in _label_candidates(size, center, bounds, radii)
            if not _overlap_count(box, placed)
        ),
        None,
    )
    if far is None and fit != "required":
        return None
    point, box = far or min(
        candidates, key=lambda item: np.linalg.norm(item[0] - center)
    )
    return point, box, True


def _leader_boxes(start, end, scale):
    """Small squares along a leader, so later labels and leaders can avoid it."""
    samples = np.linspace(start, end, max(2, int(np.linalg.norm(end - start) / scale)))
    return np.hstack((samples - scale, samples + scale))


def _clear_of_leaders(spots, center, placed, layout):
    """Keep spots clear of earlier leaders whose own leader misses every label."""
    return [
        (point, box)
        for point, box in spots
        if not _overlap_count(box, layout["leaders"])
        and not _overlap_count(
            _leader_boxes(center, point, layout["scale"])[4:, None], placed
        )
    ]


def _clear_label_position(size, center, placed, layout):
    """Prefer the nearest spot clear of points whose leader misses other labels.

    Leaders block whole directions in a dense plot, so the search fans out wider.
    Crossing another leader is the lesser fault; running through text is not. A
    leader's first points are skipped: an anchor lying on an earlier leader would
    otherwise block every direction.
    """
    bounds, scale = layout["bounds"], layout["scale"]
    occupied = [*placed, *layout["leaders"]]
    radii = np.arange(4 * scale, np.linalg.norm(bounds[2:] - bounds[:2]), 4 * scale)
    spots = [
        (point, box)
        for point, box in _label_candidates(size, center, bounds, radii, 32)
        if not _overlap_count(box, occupied)
        and not _overlap_count(box, layout["points"])
    ]
    for blocked in (occupied, placed):
        for point, box in spots:
            leader = _leader_boxes(center, point, scale)
            if not _overlap_count(leader[4:, None], blocked):
                gap = np.linalg.norm(center - np.clip(center, box[:2], box[2:]))
                return point, box, gap > 8 * scale
    return _label_position(size, center, occupied, layout)


def _place_one_label(ax, label, anchor, placed, layout, fit="required"):
    """Measure and position one label in the already settled figure layout."""
    text = ax.annotate(
        label,
        anchor,
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=layout["fontsize"],
        arrowprops={"arrowstyle": "-", "color": SAMPLE_POINT_COLOR, "lw": 0.5},
        zorder=5,
    )
    extent = text.get_window_extent(ax.figure.canvas.get_renderer())
    center = ax.transData.transform(anchor)
    size = (extent.width + 2 * layout["scale"], extent.height + 2 * layout["scale"])
    position = (
        _clear_label_position(size, center, placed, layout)
        if fit == "clear"
        else _label_position(size, center, placed, layout, fit)
    )
    if position is None:
        text.remove()
        return
    point, box, leader = position
    text.xyann = (point - center) / layout["scale"]
    text.arrow_patch.set_visible(leader)
    placed.append(box)
    if leader and fit in ("clear", "near"):
        layout["leaders"].extend(_leader_boxes(center, point, layout["scale"]))


def _point_boxes(ax, points, scale):
    """Display-space squares around data points that labels should not cover."""
    if points is None:
        points = [
            xy for collection in ax.collections for xy in collection.get_offsets()
        ]
    centers = ax.transData.transform(np.asarray(points).reshape(-1, 2))
    return [np.r_[center - 4 * scale, center + 4 * scale] for center in centers]


def _label_layout(ax, points, fontsize):
    """Settle the layout, then collect what labels must stay inside and avoid."""
    ax.figure.canvas.draw()
    scale = ax.figure.dpi / 72
    bounds = ax.get_window_extent().extents + np.array([2, 2, -2, -2]) * scale
    layout = {
        "bounds": bounds,
        "scale": scale,
        "fontsize": fontsize,
        "points": _point_boxes(ax, points, scale),
        "leaders": [],
    }
    placed = [
        text.get_window_extent().extents
        for text in [
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ]
    ]
    return layout, placed


def place_clear_labels(ax, anchors):
    """Label anchors inside dense point clouds from clear spots beside the points."""
    layout, placed = _label_layout(ax, None, 7)
    for label in sorted(anchors, key=lambda name: -len(str(name))):
        _place_one_label(ax, label, anchors[label], placed, layout, "clear")


def place_labels(
    ax, anchors, points=None, fontsize=7, optional=None, fit="optional", avoid=None
):
    """Keep neutral labels inside their axes, avoiding overlaps when space allows.

    ``optional`` labels follow in their given order and are left out where they
    would cover another label; with ``fit="near"`` they also need a spot beside
    their anchor clear of every drawn point, and required labels look further
    for such a spot. ``avoid`` adds data-space points that are not drawn as
    markers, such as samples along a reference line.
    """
    layout, placed = _label_layout(ax, points, fontsize)
    guides = [] if avoid is None else _point_boxes(ax, avoid, layout["scale"])
    layout["points"] = [*layout["points"], *guides]
    required = "clear" if fit == "near" else "required"
    for label in sorted(anchors, key=lambda name: -len(str(name))):
        _place_one_label(ax, label, anchors[label], placed, layout, required)
    if fit == "near":
        layout = {
            **layout,
            "points": [*_point_boxes(ax, None, layout["scale"]), *guides],
        }
    for label, anchor in (optional or {}).items():
        _place_one_label(ax, label, anchor, placed, layout, fit)


def label_categories(ax, frame, coordinates, group_column):
    """Label each category at its median-nearest sample, in legend order."""
    groups = frame.groupby(group_column)
    anchors = {
        fill(str(group), 28): nearest_anchor(
            groups.get_group(group)[list(coordinates)].to_numpy()
        )
        for group in category_order(frame[group_column])
    }
    place_labels(ax, {}, optional=anchors, fit="near")


def confidence_ellipse(points, level=0.95, segments=72):
    """Outline of a group's normal data ellipse, as ggplot2 stat_ellipse(type="norm").

    The covariance is scaled by 2 F(level; 2, n - 1), whose quantile has a
    closed form for two numerator degrees of freedom.
    """
    points = np.asarray(points, dtype=float)
    dof = len(points) - 1
    radius = np.sqrt(dof * ((1 - level) ** (-2 / dof) - 1))
    values, vectors = np.linalg.eigh(np.cov(points.T))
    angle = np.linspace(0, 2 * np.pi, segments + 1)
    circle = np.column_stack((np.cos(angle), np.sin(angle)))
    return (
        points.mean(axis=0)
        + radius * circle * np.sqrt(np.clip(values, 0, None)) @ vectors.T
    )


def group_ellipses(frame, coordinates, group_column):
    """95% ellipses of the groups with three or more samples, for twelve groups or fewer."""
    if frame[group_column].nunique() > 12:
        return {}
    return {
        group: confidence_ellipse(part[list(coordinates)].to_numpy())
        for group, part in frame.groupby(group_column)
        if len(part) >= 3
    }


def draw_group_ellipses(ax, frame, coordinates, group_column, palette):
    """Outline each group's ellipse in its color, beneath the sample points."""
    for group, outline in group_ellipses(frame, coordinates, group_column).items():
        ax.plot(*outline.T, color=palette[group], linewidth=0.8, alpha=0.8, zorder=0.5)


def expand_category_axes(ax):
    """Give crowded embeddings a page-width canvas and 6.4 by 4.5 inch axes."""
    ax.figure.set_layout_engine(None)
    ax.figure.set_size_inches(7.2, 5.5)
    ax.set_position([0.65 / 7.2, 0.55 / 5.5, 6.4 / 7.2, 4.5 / 5.5])


def _legend_columns(fig, handles, labels, title):
    """Use the largest column count whose rendered width fits 7.2 inches."""
    renderer = fig.canvas.get_renderer()
    labels = [fill(str(label), 38) for label in labels]
    options = {
        "title": title,
        "loc": "lower center",
        "bbox_to_anchor": (0.5, 0.02),
        "frameon": False,
        "labelcolor": TEXT_COLOR,
        "columnspacing": 1,
        "handletextpad": 0.4,
    }
    for columns in range(len(labels), 1, -1):
        legend = fig.legend(handles, labels, ncol=columns, **options)
        if legend.get_window_extent(renderer).width / fig.dpi <= 6.9:
            return legend
        legend.remove()
    return fig.legend(handles, labels, ncol=1, **options)


def bottom_legend(ax, handles=None, title=None):
    """Grow the canvas below a legend while retaining all axes' physical sizes."""
    fig = ax.figure
    fig.canvas.draw()
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        labels = [handle.get_label() for handle in handles]
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    legend = _legend_columns(fig, handles, labels, title)
    bounds = legend.get_window_extent(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    positions = [
        axis.get_position().bounds * np.array([width, height, width, height])
        for axis in fig.axes
    ]
    extra = bounds.height / fig.dpi + 0.35
    new_width = max(width, bounds.width / fig.dpi + 0.3)
    fig.set_layout_engine(None)
    fig.set_size_inches(new_width, height + extra)
    # The anchor is a fraction of the figure, which just grew.
    legend.set_bbox_to_anchor((0.5, 0.14 / (height + extra)), fig.transFigure)
    for axis, position in zip(fig.axes, positions):
        position[:2] += ((new_width - width) / 2, extra)
        axis.set_position(
            position / [new_width, height + extra, new_width, height + extra]
        )


def panel_label(ax, letter):
    """Keep a lowercase bold panel letter on the title line, left of the axes."""
    from matplotlib import rcParams

    ax.annotate(
        letter.lower(),
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-5, rcParams["axes.titlepad"]),
        textcoords="offset points",
        ha="right",
        va="baseline",
        fontweight="bold",
        color=TEXT_COLOR,
    )


def read_frame(source, sep=","):
    """Accept an in-memory table or a table path."""
    return source if isinstance(source, pd.DataFrame) else pd.read_csv(source, sep=sep)


def log2_matrix(frame, is_log2=False):
    """Index a wide protein table and mask nonpositive linear intensities."""
    values = frame.set_index(frame.columns[0])
    return values if is_log2 else np.log2(values.where(values > 0))


def ordered_samples(samples, conditions):
    """Order samples by condition, as in the legend, and then by sample name."""
    order = {
        condition: index
        for index, condition in enumerate(category_order(conditions.values()))
    }
    return sorted(
        samples, key=lambda sample: (order.get(conditions.get(sample), -1), sample)
    )


def finite_list(values):
    """Return compact, standards-compliant numeric JSON values."""
    return [round(float(value), 2) if np.isfinite(value) else None for value in values]
