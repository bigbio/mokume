"""
Tissue atlas slide figure: t-SNE + dendrogram in one 16:9 panel.
"""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.hierarchy import dendrogram as scipy_dendro
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

_SMALL_THRESHOLD = 8


class _SaveOpts(NamedTuple):
    """Bundle the figure-output options shared by every atlas saver."""

    dpi: int
    save_pdf: bool


_GROUP_PALETTE = {
    "Muscle/Cardiac": {"base": "#D32F2F", "cmap": matplotlib.colormaps["Reds"]},
    "Neural": {"base": "#1565C0", "cmap": matplotlib.colormaps["Blues"]},
    "Hepato-renal": {"base": "#8D6E63", "cmap": matplotlib.colormaps["copper"]},
    "Immune/Gut-associated": {
        "base": "#2E7D32",
        "cmap": matplotlib.colormaps["Greens"],
    },
    "Gonadal/Reproductive": {
        "base": "#7B1FA2",
        "cmap": matplotlib.colormaps["Purples"],
    },
    "Oropharyngeal": {"base": "#00838F", "cmap": matplotlib.colormaps["GnBu"]},
    "Pituitary": {"base": "#F57F17", "cmap": matplotlib.colormaps["YlOrBr"]},
    "Mucosal/Epithelial": {"base": "#E65100", "cmap": matplotlib.colormaps["Oranges"]},
    "Secretory/Glandular": {"base": "#558B2F", "cmap": matplotlib.colormaps["YlGn"]},
    "Stromal/Connective": {"base": "#78909C", "cmap": matplotlib.colormaps["cool"]},
}

_GROUP_HEURISTICS: list[tuple[set[str], str]] = [
    ({"heart", "muscle"}, "Muscle/Cardiac"),
    ({"brain", "eye", "spinal cord"}, "Neural"),
    ({"kidney", "liver"}, "Hepato-renal"),
    ({"intestine", "lymph node", "spleen"}, "Immune/Gut-associated"),
    ({"ovary", "testis", "uterus"}, "Gonadal/Reproductive"),
    ({"salivary gland", "tonsil"}, "Oropharyngeal"),
    ({"pituitary gland"}, "Pituitary"),
]


def _despine(ax: matplotlib.axes.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_convex_hull(ax, points, color, label=None, pad=2.5):
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
    except (ValueError, RuntimeError):
        return
    hull_pts = points[hull.vertices]
    hull_pts = np.vstack([hull_pts, hull_pts[0]])
    centroid = points.mean(axis=0)
    norms = np.linalg.norm(hull_pts - centroid, axis=1, keepdims=True).clip(1e-6)
    expanded = centroid + (hull_pts - centroid) * (1 + pad / norms)
    ax.plot(
        expanded[:, 0],
        expanded[:, 1],
        color=color,
        linestyle="--",
        linewidth=1.8,
        alpha=0.7,
        zorder=1,
    )
    if label:
        top_idx = np.argmax(expanded[:, 1])
        ax.text(
            expanded[top_idx, 0],
            expanded[top_idx, 1] + 1.5,
            label,
            fontsize=7,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.85,
                "linewidth": 0.8,
            },
        )


def _draw_ellipse(ax, points, color, label=None):
    if len(points) < 2:
        return
    cx, cy = points.mean(axis=0)
    sx = max(points[:, 0].std() * 2.5, 3)
    sy = max(points[:, 1].std() * 2.5, 3)
    ellipse = Ellipse(
        (cx, cy),
        width=sx * 2,
        height=sy * 2,
        fill=False,
        edgecolor=color,
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        zorder=1,
    )
    ax.add_patch(ellipse)
    if label:
        ax.text(
            cx,
            cy + sy + 1.5,
            label,
            fontsize=6.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.8,
                "linewidth": 0.6,
            },
        )


def _name_cluster(
    members: list[str],
    cluster_id: int,
    used_names: set[str],
) -> str:
    """Assign a human-readable name to a cluster using heuristics."""
    member_set = set(members)
    # Exact subset match first
    for pattern, pname in _GROUP_HEURISTICS:
        if pattern <= member_set and pname not in used_names:
            return pname
    # Partial overlap fallback
    for pattern, pname in _GROUP_HEURISTICS:
        if pattern & member_set and pname not in used_names:
            return pname
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
    for cl, members in raw_groups.items():
        name = _name_cluster(members, cl, used_names)
        used_names.add(name)
        proteomic_groups[name] = members

    tissue_to_group: dict[str, str] = {
        tissue: group_name
        for group_name, members in proteomic_groups.items()
        for tissue in members
    }
    return proteomic_groups, tissue_to_group, z_linkage


# Same-organ-across-groups colour unification: every sub-region of one organ
# (all the brain lobes, every lymph-node station, each bone site) shares a single
# hue and differs only by shade, so the atlas reads as organ blocks instead of
# 137 unrelated categorical colours. The proteomic-group hulls and dendrogram
# still encode the clustering; the dot colour now reads as the organ of origin.
# Keyword order matters — the most specific match wins, so "bone marrow" precedes
# "bone" and "lymph node" precedes every organ that can carry a nodal metastasis.
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

_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def _organ_key(tissue: str) -> str:
    """Map a tissue label to its organ family for colour grouping.

    Falls back to the text before the first comma (e.g. "colon, sigmoid" ->
    "colon") when no keyword matches.
    """
    low = tissue.lower()
    for keyword, canonical in _ORGAN_ALIASES:
        if keyword in low:
            return canonical
    return tissue.split(",")[0].strip()


def _organ_families(tissues) -> dict[str, list[str]]:
    """Group unique tissue labels by organ family, ordered by family size.

    ``tissues`` may be the raw per-sample array, so labels are de-duplicated
    first; otherwise every sample would add a repeated legend row.
    """
    families: dict[str, list[str]] = {}
    seen: set[str] = set()
    for tissue in tissues:
        label = str(tissue)
        if label in seen:
            continue
        seen.add(label)
        families.setdefault(_organ_key(label), []).append(label)
    for members in families.values():
        members.sort()
    return dict(sorted(families.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def _build_tissue_colors(
    proteomic_groups: dict[str, list[str]],
) -> dict[str, tuple]:
    """Colour every tissue by its organ family, shading sub-regions by lightness.

    Tissues of the same organ (brain lobes, lymph-node stations, bone sites)
    share one hue and differ only in lightness/saturation, so the atlas reads as
    organ blocks rather than 137 unrelated categorical colours. Hues are spread
    by the golden-ratio conjugate with the largest families first, giving the
    dominant organs the most separated hues.
    """
    all_tissues = [t for members in proteomic_groups.values() for t in members]
    families = _organ_families(all_tissues)

    colours: dict[str, tuple] = {}
    for family_index, members in enumerate(families.values()):
        hue = (family_index * _GOLDEN_RATIO_CONJUGATE) % 1.0
        count = len(members)
        if count == 1:
            lightness_values = [0.52]
            saturation_values = [0.62]
        else:
            lightness_values = list(np.linspace(0.40, 0.72, count))
            saturation_values = list(np.linspace(0.74, 0.48, count))
        for member_index, tissue in enumerate(members):
            rgb = colorsys.hls_to_rgb(
                hue,
                lightness_values[member_index],
                saturation_values[member_index],
            )
            colours[tissue] = (*rgb, 1.0)
    return colours


def _draw_tsne_panel(
    ax,
    tsne_emb,
    tissues,
    tissue_order,
    tissue_colors,
    proteomic_groups,
) -> None:
    """Scatter t-SNE points and draw group hulls / ellipses."""
    for t in tissue_order:
        mask = tissues == t
        ax.scatter(
            tsne_emb[mask, 0],
            tsne_emb[mask, 1],
            c=[tissue_colors.get(t, "#999")],
            s=45,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )

    for group_name, members in proteomic_groups.items():
        present = [t for t in members if (tissues == t).sum() > 0]
        if not present:
            continue
        group_mask = np.isin(tissues, present)
        group_pts = tsne_emb[group_mask]
        color = _GROUP_PALETTE.get(group_name, {"base": "#999"})["base"]
        if group_mask.sum() >= _SMALL_THRESHOLD:
            _draw_convex_hull(ax, group_pts, color, label=group_name, pad=2.5)
        else:
            _draw_ellipse(ax, group_pts, color, label=group_name)


def _build_legend_elements(
    tissues,
    proteomic_groups,
    tissue_colors,
) -> list[Line2D]:
    """Build legend handles grouped by proteomic group.

    The swatch colour for each tissue still comes from the organ-family palette
    (``tissue_colors``), so the legend groups by proteomic cluster while the
    colours read as organ of origin.
    """
    elements: list[Line2D] = []
    for group_name, members in proteomic_groups.items():
        elements.append(
            Line2D(
                [0],
                [0],
                marker="none",
                label=f"  {group_name}",
                color="none",
                markerfacecolor="none",
            )
        )
        for t in members:
            n = int((tissues == t).sum())
            if n > 0:
                elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=tissue_colors.get(t, "#999"),
                        markeredgecolor="white",
                        markeredgewidth=0.3,
                        markersize=8,
                        label=f"    {t} ({n})",
                    )
                )
    return elements


def _draw_dendrogram_panel(
    ax,
    z_linkage,
    unique_tissues,
    tissue_to_group,
) -> None:
    """Render dendrogram and style labels by group color."""
    scipy_dendro(
        z_linkage,
        labels=unique_tissues,
        ax=ax,
        orientation="right",
        leaf_font_size=8,
        above_threshold_color="#BDBDBD",
        color_threshold=0,
    )
    for coll in ax.collections:
        coll.set_linewidth(0.8)
    for line in ax.lines:
        line.set_linewidth(0.8)
    for lbl in ax.get_yticklabels():
        t = lbl.get_text()
        grp = tissue_to_group.get(t, "")
        color = _GROUP_PALETTE.get(grp, {"base": "#333"})["base"]
        lbl.set_color(color)
        lbl.set_fontweight("bold")
        lbl.set_fontsize(8)
    ax.set_title(
        "B  Tissue Dendrogram", fontsize=16, fontweight="bold", loc="left", pad=8
    )
    ax.set_xlabel("Ward distance (correlation)", fontsize=9, labelpad=5)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)


def _save_atlas_figures(
    fig,
    out_dir: Path,
    opts: _SaveOpts,
) -> None:
    """Save the combined slide figure (t-SNE + dendrogram)."""
    fig.savefig(
        out_dir / "slide_atlas_dendrogram.png",
        dpi=opts.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    if opts.save_pdf:
        fig.savefig(
            out_dir / "slide_atlas_dendrogram.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    logger.info("Saved slide_atlas_dendrogram.png")


def _save_individual_atlas(
    out_dir: Path,
    adata: ad.AnnData,
    tissue_order: list[str],
    tissue_colors: dict,
    proteomic_groups: dict,
    opts: _SaveOpts,
) -> None:
    """Render and save ``tissue_atlas.png`` as a standalone figure.

    Avoids the unreliable ``bbox_inches=ax.get_tightbbox()`` crop, which used
    to leak into the neighboring dendrogram axes when the legend extended
    beyond the t-SNE axes.
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    tissues = adata.obs["tissue"].values
    _draw_tsne_panel(
        ax,
        adata.obsm["X_tsne"],
        tissues,
        tissue_order,
        tissue_colors,
        proteomic_groups,
    )
    legend_elements = _build_legend_elements(tissues, proteomic_groups, tissue_colors)
    # Spread the (often 100+) group + tissue entries across several columns so the
    # legend stays a compact block no taller than the plot rather than one tall
    # column that forces the whole figure into a thin vertical strip.
    legend_ncol = max(1, min(4, (len(legend_elements) + 49) // 50))
    leg = ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        ncol=legend_ncol,
        columnspacing=1.0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#ddd",
        handletextpad=0.5,
        labelspacing=0.2,
        borderpad=0.6,
        title="Proteomic group",
        title_fontsize=8,
    )
    leg.set_alignment("left")
    _configure_tsne_axes(ax, adata)
    fig.savefig(
        out_dir / "tissue_atlas.png",
        dpi=opts.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    if opts.save_pdf:
        fig.savefig(
            out_dir / "tissue_atlas.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    logger.info("Saved tissue_atlas.png")


def _save_individual_dendrogram(
    out_dir: Path,
    z_linkage,
    unique_tissues: list[str],
    tissue_to_group: dict,
    opts: _SaveOpts,
) -> None:
    """Render and save ``tissue_dendrogram.png`` as a standalone figure."""
    fig, ax = plt.subplots(figsize=(7, 9))
    _draw_dendrogram_panel(ax, z_linkage, unique_tissues, tissue_to_group)
    fig.savefig(
        out_dir / "tissue_dendrogram.png",
        dpi=opts.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    if opts.save_pdf:
        fig.savefig(
            out_dir / "tissue_dendrogram.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    logger.info("Saved tissue_dendrogram.png")


def _store_group_metadata(
    adata: ad.AnnData,
    proteomic_groups: dict,
    tissue_colors: dict,
) -> None:
    """Store proteomic groups and hex tissue colors in adata.uns."""
    adata.uns["proteomic_groups"] = proteomic_groups
    adata.uns["tissue_colors"] = {
        t: matplotlib.colors.rgb2hex(c[:3]) if isinstance(c, tuple) else c
        for t, c in tissue_colors.items()
    }


def _build_tissue_order(
    proteomic_groups: dict,
    tissues: np.ndarray,
) -> list[str]:
    """Ordered tissue list from proteomic groups, filtered to present tissues."""
    return [
        t
        for members in proteomic_groups.values()
        for t in members
        if (tissues == t).sum() > 0
    ]


def _configure_tsne_axes(ax, adata: ad.AnnData) -> None:
    """Apply title, labels, aspect, despine, and stats box to the t-SNE axes."""
    ax.set_title("A  Tissue Atlas", fontsize=16, fontweight="bold", loc="left", pad=8)
    ax.set_aspect("equal")
    _despine(ax)
    ax.set_xlabel("t-SNE 1", fontsize=10, color="#888", labelpad=5)
    ax.set_ylabel("t-SNE 2", fontsize=10, color="#888", labelpad=5)

    metrics = adata.uns.get("embedding_metrics", {})
    pca_var = metrics.get("pca_var_explained")
    pca_str = f"{pca_var:.3f}" if pca_var is not None else "N/A"
    stats = (
        f"{adata.n_vars:,} proteins | {adata.n_obs} samples"
        f" | {adata.obs['tissue'].nunique()} tissues"
        f" | PCA variance = {pca_str}"
    )
    ax.text(
        0.01,
        0.01,
        stats,
        transform=ax.transAxes,
        fontsize=8.5,
        fontfamily="monospace",
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#F5F5F5",
            "edgecolor": "#BDBDBD",
            "alpha": 0.95,
            "linewidth": 1,
        },
    )


def _compute_fig_width(proteomic_groups: dict, tissues: np.ndarray) -> float:
    """Estimate figure width from number of legend entries."""
    n_legend = sum(
        1 + sum(1 for t in members if (tissues == t).sum() > 0)
        for members in proteomic_groups.values()
    )
    return max(20, 14 + n_legend * 0.35)


def plot_slide_atlas_dendrogram(
    adata: ad.AnnData,
    out_dir: Path,
    *,
    dpi: int = 300,
    save_pdf: bool = True,
) -> None:
    """Plot the combined tissue atlas and dendrogram figure."""
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

    proteomic_groups, tissue_to_group, z_linkage = _compute_tissue_groups(adata)
    tissue_colors = _build_tissue_colors(proteomic_groups)
    _store_group_metadata(adata, proteomic_groups, tissue_colors)
    tissue_order = _build_tissue_order(proteomic_groups, tissues)

    fig = plt.figure(figsize=(_compute_fig_width(proteomic_groups, tissues), 11))
    gs = GridSpec(
        1,
        2,
        width_ratios=[1.4, 0.8],
        wspace=0.01,
        left=0.01,
        right=0.99,
        bottom=0.05,
        top=0.90,
    )

    ax_tsne = fig.add_subplot(gs[0])
    _draw_tsne_panel(
        ax_tsne,
        adata.obsm["X_tsne"],
        tissues,
        tissue_order,
        tissue_colors,
        proteomic_groups,
    )

    legend_elements = _build_legend_elements(tissues, proteomic_groups, tissue_colors)
    leg = ax_tsne.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        frameon=True,
        framealpha=0.95,
        edgecolor="#ddd",
        handletextpad=0.5,
        labelspacing=0.2,
        borderpad=0.6,
        title="Proteomic group",
        title_fontsize=8,
    )
    leg.set_alignment("left")
    _configure_tsne_axes(ax_tsne, adata)

    ax_dend = fig.add_subplot(gs[1])
    _draw_dendrogram_panel(ax_dend, z_linkage, unique_tissues, tissue_to_group)

    save_opts = _SaveOpts(dpi=dpi, save_pdf=save_pdf)
    _save_atlas_figures(fig, out_dir, save_opts)
    _save_individual_atlas(
        out_dir,
        adata,
        tissue_order,
        tissue_colors,
        proteomic_groups,
        save_opts,
    )
    _save_individual_dendrogram(
        out_dir,
        z_linkage,
        unique_tissues,
        tissue_to_group,
        save_opts,
    )
