#!/usr/bin/env python3
"""Render two PXD030304 showcase figures from a Rust DirectLFQ matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from threadpoolctl import threadpool_limits

from mokume.tissuemap.config import (
    EmbeddingConfig,
    FilteringConfig,
    TissueSpecificityConfig,
)
from mokume.tissuemap.embedding import embed
from mokume.tissuemap.plotting.markers import compute_markers
from mokume.tissuemap.preprocessing import canonicalize_tissue
from mokume.tissuemap.protein_selection import filter_proteins
from mokume.tissuemap.tissue_specificity import compute_ts_scores


TISSUE_COL = "characteristics[organism part]"


def parse_args() -> argparse.Namespace:
    """Parse the Rust matrix, SDRF, and rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protein_matrix", type=Path)
    parser.add_argument("sdrf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--min-tissue-samples", type=int, default=5)
    return parser.parse_args()


def load_inputs(
    matrix_path: Path, sdrf_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align the log2 protein matrix with cell-line metadata."""
    proteins = pd.read_csv(matrix_path).set_index("ProteinName")
    proteins = proteins.apply(pd.to_numeric, errors="coerce")
    log2_matrix = np.log2(proteins.where(proteins > 0))

    sdrf = pd.read_csv(sdrf_path, sep="\t", dtype=str)
    run_counts = sdrf.groupby("source name")["comment[data file]"].nunique()
    metadata = sdrf.drop_duplicates("source name").set_index("source name")
    missing_samples = log2_matrix.columns.difference(metadata.index)
    if not missing_samples.empty:
        raise ValueError(f"SDRF lacks {len(missing_samples)} matrix samples")
    metadata = metadata.loc[log2_matrix.columns].copy()
    metadata["tissue"] = metadata[TISSUE_COL].map(canonicalize_tissue)
    metadata["run_count"] = run_counts.loc[metadata.index].astype(int)
    if metadata["tissue"].isna().any():
        raise ValueError("one or more matrix samples have no tissue annotation")
    return log2_matrix, metadata


def build_anndata(
    matrix: pd.DataFrame, metadata: pd.DataFrame, threads: int
) -> ad.AnnData:
    """Build the filtered AnnData object and compute stable embeddings."""
    filtered = filter_proteins(matrix, FilteringConfig(max_nan_frac=0.95))
    adata = ad.AnnData(
        X=filtered.T.to_numpy(dtype=np.float32),
        obs=metadata.copy(),
        var=pd.DataFrame(index=filtered.index),
    )
    adata.var.index.name = "protein"
    adata.layers["log2_corrected"] = adata.X.copy()
    config = EmbeddingConfig(
        pca_components=50,
        tsne_perplexity=15.0,
        random_state=42,
        imputation_method="mindet",
    )
    return embed(adata, config, n_jobs=threads)


def display_tissues(metadata: pd.DataFrame, count: int = 11) -> pd.Series:
    """Keep the most represented tissues and group the remainder as Other."""
    common = metadata["tissue"].value_counts().head(count).index
    return metadata["tissue"].where(metadata["tissue"].isin(common), "Other")


def tissue_palette(labels: pd.Series) -> dict[str, tuple[float, float, float]]:
    """Assign a stable categorical palette in frequency order."""
    names = labels.value_counts().index.tolist()
    colors = sns.color_palette("tab20", n_colors=len(names))
    return dict(zip(names, colors, strict=True))


def scatter_embedding(
    ax: plt.Axes,
    coordinates: np.ndarray,
    labels: pd.Series,
    title: str,
    axis_labels: tuple[str, str],
) -> None:
    """Plot one tissue-colored sample embedding."""
    palette = tissue_palette(labels)
    order = labels.value_counts().index
    for label in order:
        mask = labels.to_numpy() == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=palette[label],
            s=25,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.25,
            label=f"{label} ({int(mask.sum())})",
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="best",
        handletextpad=0.2,
        columnspacing=0.6,
    )


def plot_tissue_counts(ax: plt.Axes, metadata: pd.DataFrame) -> None:
    """Plot counts for the most represented tissues."""
    counts = metadata["tissue"].value_counts().head(20).sort_values()
    ax.barh(counts.index, counts.values, color="#457B9D")
    for row, value in enumerate(counts.values):
        ax.text(value + 1, row, str(value), va="center", fontsize=8)
    ax.set_xlabel("Cell lines")
    ax.set_title("C  Most represented tissues", loc="left", fontweight="bold")


def plot_runs_and_detection(
    ax: plt.Axes, matrix: pd.DataFrame, metadata: pd.DataFrame
) -> None:
    """Relate technical replication to detected-protein depth."""
    detected = matrix.notna().sum(axis=0).loc[metadata.index]
    jitter = np.random.default_rng(42).normal(0, 0.06, len(metadata))
    ax.scatter(
        metadata["run_count"] + jitter,
        detected,
        color="#6A4C93",
        s=18,
        alpha=0.45,
        linewidth=0,
    )
    medians = detected.groupby(metadata["run_count"]).median()
    ax.plot(medians.index, medians.values, color="#D62828", marker="o", linewidth=2)
    ax.set_xlabel("Technical runs per cell line")
    ax.set_ylabel("Detected proteins")
    ax.set_title("D  Replication and protein detection", loc="left", fontweight="bold")


def plot_detection_by_tissue(
    ax: plt.Axes, matrix: pd.DataFrame, metadata: pd.DataFrame
) -> None:
    """Compare detected-protein depth across major tissues."""
    top = metadata["tissue"].value_counts().head(10).index
    frame = metadata.loc[metadata["tissue"].isin(top), ["tissue"]].copy()
    frame["detected"] = matrix.notna().sum(axis=0).loc[frame.index]
    order = frame.groupby("tissue")["detected"].median().sort_values().index
    sns.boxplot(
        data=frame,
        x="detected",
        y="tissue",
        order=order,
        color="#90BE6D",
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=frame,
        x="detected",
        y="tissue",
        order=order,
        color="#355070",
        size=2,
        alpha=0.45,
        ax=ax,
    )
    ax.set_xlabel("Detected proteins per cell line")
    ax.set_ylabel("")
    ax.set_title("E  Detection across major tissues", loc="left", fontweight="bold")


def plot_scree(ax: plt.Axes, adata: ad.AnnData) -> None:
    """Plot component-wise and cumulative PCA variance."""
    variance = np.asarray(adata.uns["pca_variance_ratio"])[:20] * 100
    ax.bar(np.arange(1, len(variance) + 1), variance, color="#F4A261")
    ax.plot(np.arange(1, len(variance) + 1), np.cumsum(variance), color="#264653")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("F  PCA variance profile", loc="left", fontweight="bold")


def render_overview(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    adata: ad.AnnData,
    output: Path,
) -> None:
    """Render the six-panel cell-line atlas overview."""
    labels = display_tissues(metadata)
    variance = np.asarray(adata.uns["pca_variance_ratio"]) * 100
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), constrained_layout=True)
    scatter_embedding(
        axes[0, 0],
        adata.obsm["X_pca"],
        labels,
        "A  PCA by tissue of origin",
        (f"PC1 ({variance[0]:.1f}%)", f"PC2 ({variance[1]:.1f}%)"),
    )
    scatter_embedding(
        axes[0, 1],
        adata.obsm["X_tsne"],
        labels,
        "B  t-SNE by tissue of origin",
        ("t-SNE 1", "t-SNE 2"),
    )
    plot_tissue_counts(axes[0, 2], metadata)
    plot_runs_and_detection(axes[1, 0], matrix, metadata)
    plot_detection_by_tissue(axes[1, 1], matrix, metadata)
    plot_scree(axes[1, 2], adata)
    fig.suptitle(
        "PXD030304 cancer cell-line atlas — Rust Mokume overview",
        fontsize=22,
        fontweight="bold",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def marker_table(adata: ad.AnnData, tissues: list[str]) -> pd.DataFrame:
    """Select up to two nonredundant Wilcoxon markers per tissue."""
    records = []
    seen: set[str] = set()
    for tissue in tissues:
        result = sc.get.rank_genes_groups_df(adata, group=tissue, key="tissue_markers")
        candidates = result[
            (result["pvals_adj"] < 0.05) & (result["logfoldchanges"] > 0.5)
        ]
        kept = 0
        for row in candidates.itertuples(index=False):
            protein = str(row.names)
            if protein in seen:
                continue
            records.append({"tissue": tissue, "protein": protein})
            seen.add(protein)
            kept += 1
            if kept == 2:
                break
    return pd.DataFrame(records)


def tissue_mean_expression(adata: ad.AnnData, tissues: list[str]) -> np.ndarray:
    """Compute missing-aware mean protein expression for each tissue."""
    tissue_means = []
    for tissue in tissues:
        mask = adata.obs["tissue"].to_numpy() == tissue
        tissue_values = np.asarray(adata.X[mask], dtype=float)
        observed = np.isfinite(tissue_values).sum(axis=0)
        totals = np.nansum(tissue_values, axis=0)
        tissue_means.append(
            np.divide(
                totals,
                observed,
                out=np.full(tissue_values.shape[1], np.nan),
                where=observed > 0,
            )
        )
    return np.vstack(tissue_means).T


def plot_marker_heatmap(
    ax: plt.Axes, adata: ad.AnnData, markers: pd.DataFrame, tissues: list[str]
) -> None:
    """Plot tissue-mean z-scores for selected marker proteins."""
    protein_index = {protein: index for index, protein in enumerate(adata.var_names)}
    means = tissue_mean_expression(adata, tissues)
    values = means[[protein_index[protein] for protein in markers["protein"]]]
    row_mean = np.nanmean(values, axis=1, keepdims=True)
    row_std = np.nanstd(values, axis=1, keepdims=True)
    z_scores = (values - row_mean) / np.where(row_std == 0, 1, row_std)
    image = ax.imshow(z_scores, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels(tissues, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(markers)))
    ax.set_yticklabels(markers["protein"], fontsize=6)
    ax.set_title("A  Top Wilcoxon markers", loc="left", fontweight="bold")
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Tissue-mean z-score")


def plot_ts_distribution(ax: plt.Axes, ts_scores: pd.DataFrame) -> None:
    """Plot AdaTiSS scores, thresholds, and enrichment categories."""
    values = ts_scores["max_ts"].dropna()
    sns.histplot(values, bins=50, stat="density", color="#7189BF", ax=ax)
    thresholds = {
        "enriched": ts_scores.attrs["ts_enriched_threshold"],
        "specific": ts_scores.attrs["ts_specific_threshold"],
    }
    for label, value in thresholds.items():
        ax.axvline(value, linestyle="--", linewidth=1.5, label=f"{label}: {value:.2f}")
    categories = ts_scores["enrichment_category"].value_counts()
    ax.text(
        0.98,
        0.96,
        "\n".join(f"{name}: {value:,}" for name, value in categories.items()),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("AdaTiSS maximum tissue-specificity score")
    ax.set_ylabel("Density")
    ax.set_title("B  AdaTiSS score distribution", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)


def plot_specific_counts(ax: plt.Axes, ts_scores: pd.DataFrame) -> None:
    """Plot tissue-specific protein counts by assigned tissue."""
    specific = ts_scores[ts_scores["enrichment_category"] == "tissue-specific"]
    counts = specific["max_tissue"].value_counts().head(20).sort_values()
    ax.barh(counts.index, counts.values, color="#E76F51")
    for row, value in enumerate(counts.values):
        ax.text(value + 0.2, row, str(value), va="center", fontsize=8)
    ax.set_xlabel("Tissue-specific proteins")
    ax.set_title("C  Specific proteins by tissue", loc="left", fontweight="bold")


def plot_marker_embedding(
    ax: plt.Axes,
    adata: ad.AnnData,
    protein: str,
    tissue: str,
    panel: str,
) -> None:
    """Plot one marker's expression across the t-SNE embedding."""
    index = int(adata.var_names.get_loc(protein))
    expression = np.asarray(adata.X[:, index]).ravel()
    missing = np.isnan(expression)
    coordinates = adata.obsm["X_tsne"]
    ax.scatter(
        coordinates[missing, 0],
        coordinates[missing, 1],
        color="#D0D0D0",
        s=14,
        alpha=0.5,
        linewidth=0,
    )
    valid = ~missing
    order = np.argsort(expression[valid])
    selected = np.flatnonzero(valid)[order]
    points = ax.scatter(
        coordinates[selected, 0],
        coordinates[selected, 1],
        c=expression[selected],
        cmap="plasma",
        s=20,
        alpha=0.85,
        linewidth=0,
    )
    ax.figure.colorbar(points, ax=ax, pad=0.01, label="log2 DirectLFQ")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{panel}  {tissue} marker\n{protein}", loc="left", fontweight="bold")


def biology_inputs(
    adata: ad.AnnData, min_tissue_samples: int
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute robust-tissue markers and AdaTiSS scores."""
    counts = adata.obs["tissue"].value_counts()
    robust = counts[counts >= min_tissue_samples].index
    biology = adata[adata.obs["tissue"].isin(robust)].copy()
    compute_markers(biology, min_group_size=min_tissue_samples)
    top_tissues = biology.obs["tissue"].value_counts().head(12).index.tolist()
    markers = marker_table(biology, top_tissues)
    ts_scores = compute_ts_scores(
        np.asarray(biology.X),
        biology.obs["tissue"].to_numpy(),
        np.asarray(biology.var_names),
        TissueSpecificityConfig(use_pure_mad=True),
    )
    return biology, markers, ts_scores, top_tissues


def render_biology(
    adata: ad.AnnData,
    markers: pd.DataFrame,
    ts_scores: pd.DataFrame,
    tissues: list[str],
    output: Path,
) -> None:
    """Render the six-panel marker and tissue-specificity showcase."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), constrained_layout=True)
    plot_marker_heatmap(axes[0, 0], adata, markers, tissues)
    plot_ts_distribution(axes[0, 1], ts_scores)
    plot_specific_counts(axes[0, 2], ts_scores)
    showcase = markers.drop_duplicates("tissue").head(3)
    for panel, (ax, row) in zip(
        "DEF", zip(axes[1], showcase.itertuples()), strict=True
    ):
        plot_marker_embedding(ax, adata, row.protein, row.tissue, panel)
    fig.suptitle(
        "PXD030304 cancer cell-line atlas — tissue specificity and markers",
        fontsize=22,
        fontweight="bold",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def write_summary(
    output: Path,
    matrix: pd.DataFrame,
    adata: ad.AnnData,
    biology: ad.AnnData,
    ts_scores: pd.DataFrame,
) -> None:
    """Write compact provenance-neutral dataset and analysis statistics."""
    categories = ts_scores["enrichment_category"].value_counts().to_dict()
    payload = {
        "proteins": int(matrix.shape[0]),
        "embedding_proteins": int(adata.uns["embedding_metrics"]["n_proteins_used"]),
        "samples": int(matrix.shape[1]),
        "tissues": int(adata.obs["tissue"].nunique()),
        "missing_fraction": float(matrix.isna().to_numpy().mean()),
        "pc1_variance": float(adata.uns["pca_variance_ratio"][0]),
        "pc2_variance": float(adata.uns["pca_variance_ratio"][1]),
        "biology_samples": int(biology.n_obs),
        "biology_tissues": int(biology.obs["tissue"].nunique()),
        "ts_enriched_threshold": float(ts_scores.attrs["ts_enriched_threshold"]),
        "ts_specific_threshold": float(ts_scores.attrs["ts_specific_threshold"]),
        "enrichment_categories": {key: int(value) for key, value in categories.items()},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    """Render all PXD030304 figures and summary artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    with threadpool_limits(limits=args.threads):
        matrix, metadata = load_inputs(args.protein_matrix, args.sdrf)
        adata = build_anndata(matrix, metadata, args.threads)
        overview = args.output_dir / "pxd030304_rust_overview.png"
        render_overview(matrix, metadata, adata, overview)
        biology, markers, ts_scores, tissues = biology_inputs(
            adata, args.min_tissue_samples
        )
        biology_figure = args.output_dir / "pxd030304_rust_biology.png"
        render_biology(biology, markers, ts_scores, tissues, biology_figure)
        markers.to_csv(args.output_dir / "pxd030304_markers.csv", index=False)
        write_summary(
            args.output_dir / "pxd030304_summary.json",
            matrix,
            adata,
            biology,
            ts_scores,
        )
    print(
        f"wrote {overview} and {biology_figure}: "
        f"{matrix.shape[0]:,} proteins x {matrix.shape[1]:,} cell lines"
    )


if __name__ == "__main__":
    main()
