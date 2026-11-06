"""Tissue atlas and dendrograms sharing proteomic-group colors."""

from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.hierarchy import dendrogram as scipy_dendro
from scipy.spatial.distance import pdist

from mokume.plotting._common import (
    bottom_legend,
    nearest_anchor,
    panel_label,
    place_clear_labels as _label_groups,
    static_style,
    style_axes,
)
from mokume.plotting.style import (
    category_colors,
    SAMPLE_POINT_COLOR,
    TEXT_COLOR,
    WHITE,
)

logger = logging.getLogger(__name__)


class _SaveOpts(NamedTuple):
    """Bundle the figure-output options shared by every atlas saver."""

    dpi: int
    save_pdf: bool


def _name_cluster(members: list[str], cluster_id: int, used_names: set[str]) -> str:
    """Name a cluster only when one organ family has a strict majority."""
    family, count = Counter(_organ_key(tissue) for tissue in members).most_common(1)[0]
    name = family.capitalize()
    if count > len(members) / 2 and name not in used_names:
        return name
    return f"Group {cluster_id}"


def _compute_tissue_groups(
    adata: ad.AnnData,
    max_clusters: int = 10,
) -> tuple[dict[str, list[str]], dict[str, str], np.ndarray]:
    """Compute data-driven tissue groups from hierarchical linkage.

    Returns
    -------
    proteomic_groups : dict[str, list[str]]
    tissue_to_group : dict[str, str]
    z_linkage : np.ndarray
        Ward linkage matrix (reused by the plotting function).
    """
    tissues = adata.obs["tissue"].values
    unique_tissues = sorted(np.unique(tissues))
    tissue_means = np.array(
        [adata.X[tissues == t].mean(axis=0) for t in unique_tissues]
    )

    tissue_means = np.nan_to_num(tissue_means, nan=0.0)
    dist = pdist(tissue_means, metric="correlation")
    finite_mask = np.isfinite(dist)
    if not finite_mask.all():
        dist[~finite_mask] = np.nanmax(dist[finite_mask]) if finite_mask.any() else 1.0

    z_linkage = linkage(dist, method="ward")
    n_groups = min(max_clusters, len(unique_tissues))
    cut_labels = fcluster(z_linkage, t=n_groups, criterion="maxclust")

    raw_groups: dict[int, list[str]] = {}
    for i, tissue in enumerate(unique_tissues):
        raw_groups.setdefault(cut_labels[i], []).append(tissue)

    proteomic_groups: dict[str, list[str]] = {}
    used_names: set[str] = set()
    for cl, members in sorted(raw_groups.items()):
        name = _name_cluster(members, cl, used_names)
        used_names.add(name)
        proteomic_groups[name] = members

    tissue_to_group: dict[str, str] = {
        tissue: group_name
        for group_name, members in proteomic_groups.items()
        for tissue in members
    }
    return proteomic_groups, tissue_to_group, z_linkage


# Keyword order matters: specific organ families precede broader matches.
_ORGAN_ALIASES: list[tuple[str, str]] = [
    ("lymph node", "lymph node"),
    ("bone marrow", "bone marrow"),
    ("bone", "bone"),
    ("brain", "brain"),
    ("oral cavity", "oral cavity"),
    ("mouth", "oral cavity"),
    ("tonsil", "oral cavity"),
    ("lung", "lung"),
    ("pleura", "lung"),
    ("salivary gland", "salivary gland"),
    ("small intestine", "intestine"),
    ("large intestine", "intestine"),
    ("ileocecal", "intestine"),
    ("cecum", "intestine"),
    ("intestine", "intestine"),
    ("rectum", "rectum"),
    ("esophag", "esophagus"),
    ("ovar", "ovary"),
    ("fallopian", "uterus"),
    ("endometri", "uterus"),
    ("uter", "uterus"),
    ("cervix", "uterus"),
    ("vulva", "vulva"),
    ("vagina", "vagina"),
    ("renal pelvis", "kidney"),
    ("pararenal", "kidney"),
    ("kidney", "kidney"),
    ("liver", "liver"),
    ("pancrea", "pancreas"),
    ("colon", "colon"),
    ("pyloric", "stomach"),
    ("paragastric", "stomach"),
    ("gastric", "stomach"),
    ("stomach", "stomach"),
    ("breast", "breast"),
    ("prostate", "prostate"),
    ("thyroid", "thyroid"),
    ("bladder", "bladder"),
    ("muscle", "muscle"),
    ("skin", "skin"),
    ("epidermis", "skin"),
    ("hypoderm", "skin"),
    ("pharynx", "pharynx"),
    ("spleen", "spleen"),
    ("adrenal", "adrenal gland"),
    ("testis", "testis"),
    ("placenta", "placenta"),
    ("omentum", "omentum"),
    ("retroperiton", "peritoneum"),
    ("peritone", "peritoneum"),
    ("pericard", "heart"),
    ("heart", "heart"),
    ("cerebrospinal", "nervous system"),
    ("spinal", "nervous system"),
    ("nervous", "nervous system"),
    ("orbital", "eye"),
    ("eye", "eye"),
    ("synovi", "joint"),
    ("cartil", "joint"),
    ("head and neck", "head and neck"),
    ("neck", "head and neck"),
    ("larynx", "larynx"),
    ("blood", "blood"),
    ("ascites", "ascites"),
]


def _organ_key(tissue: str) -> str:
    """Map a tissue label to its organ family for majority-based cluster naming.

    Falls back to the text before the first comma (e.g. "colon, sigmoid" ->
    "colon") when no keyword matches.
    """
    low = tissue.lower()
    for keyword, canonical in _ORGAN_ALIASES:
        if keyword in low:
            return canonical
    return tissue.split(",")[0].strip()


def _group_styles(proteomic_groups: dict) -> dict[str, str]:
    """Assign one shared palette color to each proteomic group."""
    return dict(zip(proteomic_groups, category_colors(len(proteomic_groups))))


def _sample_points(ax, x, y, color, **kwargs):
    """Keep atlas samples and dendrogram group marks visually identical."""
    ax.scatter(
        x,
        y,
        color=color,
        marker="o",
        s=20,
        alpha=0.85,
        linewidths=0,
        **kwargs,
    )


def _draw_tsne_panel(ax, adata, proteomic_groups, group_styles) -> dict:
    """Draw existing t-SNE coordinates with the proteomic-group encoding."""
    tsne = adata.obsm["X_tsne"]
    tissues = adata.obs["tissue"].values
    anchors = {}
    for group, members in proteomic_groups.items():
        points = tsne[np.isin(tissues, members)]
        color = group_styles[group]
        _sample_points(
            ax,
            points[:, 0],
            points[:, 1],
            color,
            zorder=3,
        )
        # Anchor on a real member: a split group's mean can sit in empty space.
        anchors[group] = nearest_anchor(points)
    ax.set_title("Tissue atlas", loc="left")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.margins(0.18)
    style_axes(ax)
    return anchors


def _group_legend(ax, group_styles, **kwargs) -> None:
    """Show one neutral-text legend entry for each proteomic group."""
    if len(group_styles) < 2:
        return
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=np.sqrt(20),
            alpha=0.85,
            markerfacecolor=color,
            markeredgewidth=0,
            label=group,
        )
        for group, color in group_styles.items()
    ]
    if len(group_styles) > 12:
        bottom_legend(ax, handles, title="Proteomic group")
        return
    ax.legend(
        handles=handles,
        title="Proteomic group",
        labelcolor=TEXT_COLOR,
        frameon=False,
        **kwargs,
    )


def _draw_dendrogram_panel(
    ax, z_linkage, unique_tissues, tissue_to_group, group_styles
) -> None:
    """Use neutral leaf labels and the same group markers as the atlas."""
    scipy_dendro(
        z_linkage,
        labels=unique_tissues,
        ax=ax,
        orientation="right",
        leaf_font_size=8,
        above_threshold_color=SAMPLE_POINT_COLOR,
        color_threshold=0,
    )
    for collection in ax.collections:
        collection.set_linewidth(1)
    style_axes(ax)
    marker_transform = ax.get_yaxis_transform() + ScaledTranslation(
        -12 / 72, 0, ax.figure.dpi_scale_trans
    )
    for label in ax.get_yticklabels():
        label.set_color(TEXT_COLOR)
        color = group_styles[tissue_to_group[label.get_text()]]
        _sample_points(
            ax,
            0,
            label.get_position()[1],
            color,
            transform=marker_transform,
            clip_on=False,
            zorder=4,
        )
    ax.tick_params(axis="y", length=0, pad=25)
    ax.set_title("Tissue dendrogram", loc="left")
    ax.set_xlabel("Ward distance (correlation)")


def _fit_dendrogram(ax):
    """Fit leaf labels to their rendered row height, growing below 5 pt."""
    fig = ax.figure
    fig.canvas.draw()
    labels = ax.get_yticklabels()
    row_height = ax.get_window_extent().height * 72 / fig.dpi / len(labels)
    if row_height < 5 / 0.7:
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height + len(labels) * (5 / 0.7 - row_height) / 72)
        fig.canvas.draw()
        row_height = ax.get_window_extent().height * 72 / fig.dpi / len(labels)
    plt.setp(labels, fontsize=max(5, min(8, row_height * 0.7)))


def _save_figure(fig, out_dir: Path, name: str, opts: _SaveOpts) -> None:
    """Save each atlas view with consistent output options."""
    fig.savefig(
        out_dir / f"{name}.png", dpi=opts.dpi, bbox_inches="tight", facecolor=WHITE
    )
    if opts.save_pdf:
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    logger.info("Saved %s.png", name)


def _save_individual_atlas(out_dir, adata, proteomic_groups, group_styles, opts):
    """Render a standalone atlas with a compact group legend."""
    fig, ax = plt.subplots(figsize=(4.5, 3.6), layout="constrained")
    if len(group_styles) <= 12:
        _group_legend(ax, group_styles, loc="center left", bbox_to_anchor=(1.02, 0.5))
    anchors = _draw_tsne_panel(ax, adata, proteomic_groups, group_styles)
    if len(group_styles) > 12:
        _group_legend(ax, group_styles)
    _label_groups(ax, anchors)
    _save_figure(fig, out_dir, "tissue_atlas", opts)


def _save_individual_dendrogram(out_dir, tree, group_styles, opts):
    """Render a standalone dendrogram with a group legend."""
    z_linkage, tissues, tissue_to_group = tree
    longest = max(map(len, tissues))
    fig, ax = plt.subplots(
        figsize=(max(3.5, 3.2 + longest * 0.06), max(3, len(tissues) * 0.16)),
        layout="constrained",
    )
    _group_legend(ax, group_styles, loc="upper left", bbox_to_anchor=(1.02, 1))
    _draw_dendrogram_panel(ax, z_linkage, tissues, tissue_to_group, group_styles)
    _fit_dendrogram(ax)
    _save_figure(fig, out_dir, "tissue_dendrogram", opts)


@static_style()
def plot_slide_atlas_dendrogram(
    adata: ad.AnnData,
    out_dir: Path,
    *,
    dpi: int = 300,
    save_pdf: bool = True,
) -> None:
    """Plot combined and standalone views of the same proteomic groups."""
    if "X_tsne" not in adata.obsm:
        logger.warning("No t-SNE embedding, skipping slide figure")
        return
    tissues = adata.obs["tissue"].values
    unique_tissues = sorted(np.unique(tissues))
    if len(unique_tissues) < 2:
        logger.warning(
            "Atlas dendrogram requires at least two tissues; found %d, "
            "skipping slide figure",
            len(unique_tissues),
        )
        return
    groups, tissue_to_group, z_linkage = _compute_tissue_groups(adata)
    styles = _group_styles(groups)
    adata.uns["proteomic_groups"] = groups
    adata.uns["tissue_colors"] = {
        tissue: styles[group] for tissue, group in tissue_to_group.items()
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            9.6 if len(unique_tissues) > _TALL_TISSUES else 7.2,
            max(3.6, len(unique_tissues) * 0.16),
        ),
        gridspec_kw={
            "width_ratios": [1.6 if len(unique_tissues) > _TALL_TISSUES else 1.3, 1]
        },
        layout="constrained",
    )
    _draw_slide_panels(
        axes, adata, groups, styles, (z_linkage, unique_tissues, tissue_to_group)
    )
    opts = _SaveOpts(dpi, save_pdf)
    _save_figure(fig, out_dir, "slide_atlas_dendrogram", opts)
    _save_individual_atlas(out_dir, adata, groups, styles, opts)
    _save_individual_dendrogram(
        out_dir, (z_linkage, unique_tissues, tissue_to_group), styles, opts
    )


_TALL_TISSUES = 36


def _draw_slide_panels(axes, adata, groups, styles, tree):
    """Lay out the atlas and tree before positioning their labels."""
    if len(styles) <= 12:
        _group_legend(
            axes[0], styles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3
        )
    _draw_dendrogram_panel(axes[1], *tree, styles)
    anchors = _draw_tsne_panel(axes[0], adata, groups, styles)
    if len(tree[1]) > _TALL_TISSUES:
        # The tree needs the height; a square atlas stays on top instead of stretching.
        axes[0].set_box_aspect(1)
        axes[0].set_anchor("N")
    _fit_dendrogram(axes[1])
    if len(styles) > 12:
        _group_legend(axes[0], styles)
    _label_groups(axes[0], anchors)
    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter)
