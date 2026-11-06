"""
Tissue marker analysis plots: heatmap, marker t-SNE, dotplot.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from textwrap import fill

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import scanpy as sc

from mokume.imputation.censored import impute_mindet
from mokume.plotting._common import static_style, style_axes
from mokume.plotting.style import (
    GRID_COLOR,
    MISSING_COLOR,
    SEQUENTIAL_COLORS,
    TEXT_COLOR,
)

logger = logging.getLogger(__name__)

_EXPR_CMAP = LinearSegmentedColormap.from_list("marker_expression", SEQUENTIAL_COLORS)
_MARKER_FDR = 0.05
_MARKER_MIN_LOG2FC = 0.5


def compute_markers(adata: ad.AnnData, min_group_size: int = 2) -> ad.AnnData:
    """Run Wilcoxon rank-sum test for tissue markers.

    Adds ``tissue_markers`` key to ``adata.uns``.
    Skips gracefully when too many tissues have < *min_group_size* samples
    (Wilcoxon requires ≥ 2 per group).
    """

    # Check group sizes — Wilcoxon needs ≥ 2 samples per group
    tissue_counts = adata.obs["tissue"].value_counts()
    valid = tissue_counts[tissue_counts >= min_group_size]
    if len(valid) < 2:
        n_single = (tissue_counts < min_group_size).sum()
        logger.warning(
            "Wilcoxon markers skipped: %d / %d tissues have < %d samples",
            n_single,
            len(tissue_counts),
            min_group_size,
        )
        return adata

    # Subset to tissues with enough samples
    if len(valid) < len(tissue_counts):
        dropped = tissue_counts[tissue_counts < min_group_size]
        logger.info(
            "Wilcoxon: using %d / %d tissues (dropped %d with < %d samples)",
            len(valid),
            len(tissue_counts),
            len(dropped),
            min_group_size,
        )
        mask = adata.obs["tissue"].isin(valid.index)
        adata_sub = adata[mask].copy()
    else:
        adata_sub = adata

    # Scanpy cannot rank matrices containing NaN. Use the same scale-aware,
    # per-sample MNAR assumption as the default TissueMap embedding rather than
    # treating missing log2 intensities as the arbitrary numeric value zero.
    x_backup = adata_sub.X.copy()
    protein_by_sample = pd.DataFrame(np.asarray(adata_sub.X).T)
    adata_sub.X = impute_mindet(protein_by_sample).T.to_numpy(dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", PerformanceWarning)
        sc.tl.rank_genes_groups(
            adata_sub,
            groupby="tissue",
            method="wilcoxon",
            key_added="tissue_markers",
            use_raw=False,
        )
    adata_sub.X = x_backup

    # Copy results back to original adata
    adata.uns["tissue_markers"] = adata_sub.uns["tissue_markers"]
    logger.info("Wilcoxon rank_genes_groups done")
    return adata


def _significant_markers(result: pd.DataFrame) -> pd.DataFrame:
    """Return positively enriched markers passing the display thresholds."""
    return result[
        (result["pvals_adj"] < _MARKER_FDR)
        & (result["logfoldchanges"] > _MARKER_MIN_LOG2FC)
    ]


def _collect_top_markers(
    adata: ad.AnnData,
    tissue_order: list[str],
    n_top: int,
) -> list[tuple[str, str]]:
    """Collect top *n_top* unique markers per tissue from Wilcoxon results."""
    heat_proteins: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tissue in tissue_order:
        try:
            result = sc.get.rank_genes_groups_df(
                adata, group=tissue, key="tissue_markers"
            )
        except (KeyError, ValueError):
            continue
        result = _significant_markers(result)
        count = 0
        for _, row in result.iterrows():
            prot = row["names"]
            if prot not in seen and count < n_top:
                heat_proteins.append((prot, tissue))
                seen.add(prot)
                count += 1
    return heat_proteins


def _compute_zscore_matrix(
    tissue_means: pd.DataFrame,
    available_proteins: list[str],
    available_tissues: list[str],
) -> np.ndarray:
    """Compute row-wise z-scored matrix clipped to [-3, 3]."""
    heat_mat = tissue_means.loc[available_proteins, available_tissues].values.astype(
        float
    )
    row_mean = np.nanmean(heat_mat, axis=1, keepdims=True)
    row_std = np.nanstd(heat_mat, axis=1, keepdims=True)
    row_std[row_std == 0] = 1
    heat_z = (heat_mat - row_mean) / row_std
    heat_z = np.clip(heat_z, -3, 3)
    return np.nan_to_num(heat_z, nan=0.0)


def _protein_label(protein: str) -> str:
    """Show the entry name for a three-field UniProt identifier."""
    fields = protein.split("|")
    if len(fields) == 3 and "_" in fields[2]:
        return fields[2].rsplit("_", 1)[0]
    return protein


def _draw_heatmap(ax, prepared, tissue_colors, title) -> None:
    """Render row z-scores centered on zero with neutral tissue labels."""
    protein_names, protein_tissues, tissues, proteins, heat_z = prepared
    im = ax.imshow(
        heat_z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest"
    )
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels([_protein_label(p) for p in proteins], fontsize=7)
    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels(
        tissues,
        rotation=90 if len(tissues) > 30 else 60,
        ha="center" if len(tissues) > 30 else "right",
        fontsize=8,
    )
    for i, tissue in enumerate(tissues):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, -1.5),
                1,
                1,
                color=tissue_colors.get(tissue, MISSING_COLOR),
                linewidth=0,
                clip_on=False,
            )
        )
    prev_tissue = protein_tissues[0] if protein_tissues else None
    for i, protein in enumerate(proteins):
        tissue = protein_tissues[protein_names.index(protein)]
        if tissue != prev_tissue:
            ax.axhline(y=i - 0.5, color=GRID_COLOR, linewidth=0.5)
            prev_tissue = tissue
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.3, aspect=20, pad=0.02)
    cbar.set_label("Z-score (mean tissue expression)", color=TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    ax.set_title(title, pad=20)
    style_axes(ax)


def _compute_tissue_means(
    adata: ad.AnnData,
    tissue_order: list[str],
) -> pd.DataFrame:
    """Per-tissue mean expression for each protein."""
    tissues = adata.obs["tissue"].values
    tissue_means = pd.DataFrame(index=adata.var.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in tissue_order:
            mask = tissues == t
            if mask.sum() > 0:
                tissue_means[t] = np.nanmean(adata.X[mask, :], axis=0)
    return tissue_means


def _prepare_heatmap_data(
    heat_proteins: list[tuple[str, str]],
    tissue_means: pd.DataFrame,
    tissue_order: list[str],
) -> tuple[list[str], list[str], list[str], list[str], np.ndarray] | None:
    """Filter markers to available proteins/tissues and compute z-scores.

    Returns None if no valid data remains.
    """
    protein_names, protein_tissues = zip(*heat_proteins)
    cols_set, idx_set = set(tissue_means.columns), set(tissue_means.index)
    available_tissues = [t for t in tissue_order if t in cols_set]
    available_proteins = [p for p in protein_names if p in idx_set]
    if not (available_proteins and available_tissues):
        return None
    heat_z = _compute_zscore_matrix(tissue_means, available_proteins, available_tissues)
    return (
        list(protein_names),
        list(protein_tissues),
        available_tissues,
        available_proteins,
        heat_z,
    )


@static_style()
def plot_marker_heatmap(
    adata: ad.AnnData,
    tissue_order: list[str],
    out_dir: Path,
    *,
    n_top: int = 5,
    dpi: int = 300,
    save_pdf: bool = True,
) -> None:
    """Top *n_top* Wilcoxon markers per tissue, z-scored heatmap."""
    if "tissue_markers" not in adata.uns:
        logger.warning("No tissue_markers in adata.uns, skipping heatmap")
        return

    tissue_colors = adata.uns.get("tissue_colors", {})
    tissue_means = _compute_tissue_means(adata, tissue_order)

    heat_proteins = _collect_top_markers(adata, tissue_order, n_top)
    if not heat_proteins:
        logger.warning("No markers found for heatmap")
        return

    prepared = _prepare_heatmap_data(heat_proteins, tissue_means, tissue_order)
    if prepared is None:
        return
    available_tissues, available_proteins = prepared[2:4]
    fig_height = max(6, len(available_proteins) * 0.13)
    fig, ax = plt.subplots(
        figsize=(max(4.5, len(available_tissues) * 0.14 + 1.5), fig_height)
    )
    heatmap_title = (
        f"Top {n_top} marker proteins per tissue (Wilcoxon, "
        f"{len(available_tissues)} tissues)\n"
        f"{adata.n_vars:,} proteins, {adata.n_obs} samples"
    )
    _draw_heatmap(ax, prepared, tissue_colors, heatmap_title)

    plt.tight_layout()
    fig.savefig(out_dir / "marker_heatmap.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_dir / "marker_heatmap.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved marker_heatmap.png")


def _select_showcase_markers(
    adata: ad.AnnData,
    tissue_order: list[str],
    n_showcase: int,
) -> list[tuple[str, str]]:
    """Pick one top marker per tissue (up to *n_showcase*)."""
    showcase: list[tuple[str, str]] = []
    seen_prots: set[str] = set()
    for t in tissue_order:
        try:
            result = sc.get.rank_genes_groups_df(adata, group=t, key="tissue_markers")
        except (KeyError, ValueError):
            continue
        result = _significant_markers(result)
        for _, row in result.iterrows():
            prot = row["names"]
            if prot not in seen_prots and prot in adata.var.index:
                showcase.append((prot, t))
                seen_prots.add(prot)
                break
        if len(showcase) >= n_showcase:
            break
    return showcase


def _draw_nan_points(ax, tsne: np.ndarray, nan_mask: np.ndarray) -> None:
    """Keep missing samples visible beneath the relative-abundance layer."""
    if nan_mask.any():
        ax.scatter(
            tsne[nan_mask, 0],
            tsne[nan_mask, 1],
            c=MISSING_COLOR,
            s=12,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )


def _draw_detected_points(ax, tsne, expr_all, nan_mask) -> None:
    """Scale detected values by their protein's 1st and 99th percentiles."""
    valid_idx = np.where(~nan_mask)[0]
    if valid_idx.size == 0:
        return
    expr_valid = expr_all[valid_idx]
    low, high = np.percentile(expr_valid, [1, 99])
    relative = Normalize(vmin=low, vmax=high, clip=True)(expr_valid)
    order = np.argsort(expr_valid)
    sorted_idx = valid_idx[order]
    ax.scatter(
        tsne[sorted_idx, 0],
        tsne[sorted_idx, 1],
        c=relative[order],
        cmap=_EXPR_CMAP,
        vmin=0,
        vmax=1,
        s=12,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )


def _render_marker_subplot(ax, tsne, adata, prot, tissue, var_index_map):
    """Plot one protein's relative abundance without altering expression."""
    expr_all = np.asarray(adata.X[:, var_index_map[prot]]).ravel()
    nan_mask = np.isnan(expr_all)
    _draw_nan_points(ax, tsne, nan_mask)
    _draw_detected_points(ax, tsne, expr_all, nan_mask)
    detected = f"{int((~nan_mask).sum())}/{len(expr_all)} samples"
    if "max_ts" in adata.var.columns:
        detected = f"TS={adata.var.loc[prot, 'max_ts']:.2f}, {detected}"
    # Panels sit side by side, so long tissue names wrap instead of meeting their neighbors.
    ax.set_title(
        f"{_protein_label(prot)}\n{fill(f'{tissue} marker', 26)}\n{detected}",
        pad=4,
        fontsize=7,
    )
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.margins(0.12)
    style_axes(ax)


def _create_subplot_grid(
    n_items: int,
    n_cols: int = 4,
) -> tuple[plt.Figure, list]:
    """Create a subplot grid and return (fig, flat_axes_list)."""
    n_rows = (n_items + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.8 * n_cols, 2.2 * n_rows), layout="constrained"
    )
    flat = np.asarray(axes).flatten().tolist()
    for ax in flat[n_items:]:
        ax.set_visible(False)
    return fig, flat


@static_style()
def plot_marker_tsne(
    adata: ad.AnnData,
    tissue_order: list[str],
    out_dir: Path,
    *,
    n_showcase: int = 8,
    dpi: int = 300,
    save_pdf: bool = True,
) -> None:
    """Show relative abundance within each showcase protein on one shared scale."""
    if "X_tsne" not in adata.obsm:
        logger.warning("Missing t-SNE, skipping marker_tsne")
        return
    if "tissue_markers" not in adata.uns:
        logger.warning("Missing markers, skipping marker_tsne")
        return

    showcase = _select_showcase_markers(adata, tissue_order, n_showcase)
    if not showcase:
        return

    tsne = adata.obsm["X_tsne"]
    fig, flat_axes = _create_subplot_grid(len(showcase))

    var_index_map = {name: i for i, name in enumerate(adata.var.index)}
    for idx, (prot, tissue) in enumerate(showcase):
        _render_marker_subplot(flat_axes[idx], tsne, adata, prot, tissue, var_index_map)

    fig.suptitle(
        "t-SNE — Relative abundance within each protein",
        color=TEXT_COLOR,
    )
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=_EXPR_CMAP),
        ax=flat_axes[: len(showcase)],
        shrink=0.6,
        pad=0.02,
    )
    cbar.set_label(
        "Relative abundance within each protein (1st–99th percentile)",
        color=TEXT_COLOR,
    )
    cbar.ax.tick_params(colors=TEXT_COLOR)
    fig.savefig(out_dir / "marker_tsne.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_dir / "marker_tsne.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved marker_tsne.png")


def _collect_dotplot_genes(
    adata: ad.AnnData,
    available: list[str],
    n_top: int,
) -> list[str]:
    """Collect top marker genes for dotplot (max 60)."""
    genes: list[str] = []
    seen: set[str] = set()
    max_total = n_top * len(available)
    for tissue in available:
        try:
            result = sc.get.rank_genes_groups_df(
                adata, group=tissue, key="tissue_markers"
            )
        except (KeyError, ValueError):
            continue
        result = _significant_markers(result)
        count = 0
        for _, row in result.iterrows():
            g = row["names"]
            if g not in seen and g in adata.var.index:
                genes.append(g)
                seen.add(g)
                count += 1
                if count >= n_top:
                    break
            if len(genes) >= max_total:
                break
    return genes[:60]


def _render_dotplot(adata, genes, out_dir, dpi) -> None:
    """Render Scanpy summaries with the shared scale and sample terminology."""
    dp = sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby="tissue",
        standard_scale="var",
        cmap=_EXPR_CMAP,
        show=False,
        return_fig=True,
        figsize=(
            max(7.2, len(genes) * 0.16 + 2.4),
            max(3.6, adata.obs["tissue"].nunique() * 0.16 + 1.5),
        ),
    )
    dp.style(dot_edge_lw=0, smallest_dot=8, largest_dot=60)
    dp.legend(
        size_title="Fraction of samples\nin tissue (%)",
        colorbar_title="Mean relative abundance\nwithin each protein",
    )
    axes = dp.get_axes()
    main = axes["mainplot_ax"]
    main.set_xticks(
        main.get_xticks(),
        [_protein_label(t.get_text()) for t in main.get_xticklabels()],
    )
    for ax in axes.values():
        style_axes(ax)
    for dots in axes["mainplot_ax"].collections:
        dots.set_rasterized(True)
    for dots in axes["size_legend_ax"].collections:
        dots.set_facecolor(MISSING_COLOR)
        dots.set_linewidth(0)
    dp.fig.savefig(out_dir / "marker_dotplot.png", dpi=dpi, bbox_inches="tight")
    plt.close(dp.fig)


@static_style()
def plot_dotplot(
    adata: ad.AnnData,
    tissue_order: list[str],
    out_dir: Path,
    *,
    n_top: int = 3,
    dpi: int = 300,
) -> None:
    """Scanpy dotplot of top markers per tissue."""
    if "tissue_markers" not in adata.uns:
        return

    available = [t for t in tissue_order if t in adata.obs["tissue"].values]
    dotplot_genes = _collect_dotplot_genes(adata, available, n_top)
    if not dotplot_genes:
        return

    x_backup = adata.X.copy()
    tissue_backup = adata.obs["tissue"].copy()
    try:
        adata.X = np.nan_to_num(adata.X, nan=0.0).astype(np.float32)
        adata.obs["tissue"] = pd.Categorical(
            adata.obs["tissue"],
            categories=available,
            ordered=True,
        )
        _render_dotplot(adata, dotplot_genes, out_dir, dpi)
        logger.info("Saved marker_dotplot.png")
    except (ValueError, TypeError, RuntimeError, KeyError) as exc:
        logger.error("Dotplot failed: %s", exc)
    finally:
        adata.X = x_backup
        adata.obs["tissue"] = tissue_backup


def save_markers_csv(
    adata: ad.AnnData,
    tissue_order: list[str],
    out_dir: Path,
    *,
    n_top: int = 10,
) -> None:
    """Save top markers per tissue to CSV."""
    if "tissue_markers" not in adata.uns:
        return

    records: list[dict] = []
    for tissue in tissue_order:
        try:
            result = sc.get.rank_genes_groups_df(
                adata, group=tissue, key="tissue_markers"
            )
        except (KeyError, ValueError):
            continue
        top = _significant_markers(result).head(n_top)
        for _, row in top.iterrows():
            records.append(
                {
                    "tissue": tissue,
                    "protein": row["names"],
                    "score": round(row["scores"], 2),
                    "logfoldchange": round(row["logfoldchanges"], 3),
                    "pval_adj": f"{row['pvals_adj']:.2e}",
                }
            )

    if records:
        df = pd.DataFrame(records)
        csv_path = out_dir / "tissue_markers_top10.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %s (%d markers)", csv_path.name, len(df))
