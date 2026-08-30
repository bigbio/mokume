#!/usr/bin/env python3
"""Render the six-panel PDC000125 showcase from Rust kernel tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


CONDITION_COL = "factor value[condition]"
CONDITION_COLORS = {
    "Primary Tumor": "#2A9D8F",
    "Solid Tissue Normal": "#F9844A",
}


def parse_args() -> argparse.Namespace:
    """Parse the Rust outputs and destination figure directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protein_matrix", type=Path)
    parser.add_argument("sdrf", type=Path)
    parser.add_argument("de_result", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def load_inputs(
    matrix_path: Path, sdrf_path: Path, de_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and align the protein matrix, SDRF, and DE result table."""
    proteins = pd.read_csv(matrix_path).set_index("ProteinName")
    proteins = proteins.apply(pd.to_numeric, errors="coerce")
    log2_matrix = np.log2(proteins.where(proteins > 0))

    metadata = pd.read_csv(sdrf_path, sep="\t", dtype=str)
    metadata = metadata.drop_duplicates("source name").set_index("source name")
    metadata = metadata.loc[log2_matrix.columns].copy()
    metadata["plex"] = (
        metadata.index.to_series()
        .str.extract(r"PDC000125-p(\d{2})_", expand=False)
        .astype(int)
    )
    if set(metadata[CONDITION_COL]) != set(CONDITION_COLORS):
        raise ValueError("matrix samples do not resolve to both UCEC conditions")

    de_result = pd.read_csv(de_path, float_precision="round_trip")
    de_result = de_result.set_index("ProteinName", drop=False)
    return log2_matrix, metadata, de_result


def pca_coordinates(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute sample PCA coordinates and explained-variance percentages."""
    complete = matrix.dropna(axis=0)
    if complete.shape[0] < 2:
        raise ValueError("PCA requires at least two complete proteins")
    n_components = min(20, complete.shape[0], complete.shape[1] - 1)
    model = PCA(n_components=n_components)
    coordinates = model.fit_transform(complete.T)
    frame = pd.DataFrame(
        coordinates[:, :2], index=matrix.columns, columns=["PC1", "PC2"]
    )
    return frame, model.explained_variance_ratio_ * 100


def plot_pca_condition(
    ax: plt.Axes,
    pca: pd.DataFrame,
    variance: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    """Plot sample PCA colored by tumor or normal condition."""
    for condition, color in CONDITION_COLORS.items():
        samples = metadata.index[metadata[CONDITION_COL] == condition]
        ax.scatter(
            pca.loc[samples, "PC1"],
            pca.loc[samples, "PC2"],
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            label=f"{condition} (n={len(samples)})",
        )
    ax.set_title("A  PCA by condition", loc="left", fontweight="bold")
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}%)")
    ax.legend(frameon=False, fontsize=9)


def plot_pca_plex(
    ax: plt.Axes,
    pca: pd.DataFrame,
    variance: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    """Plot sample PCA colored by TMT plex number."""
    points = ax.scatter(
        pca["PC1"],
        pca["PC2"],
        c=metadata.loc[pca.index, "plex"],
        cmap="turbo",
        vmin=1,
        vmax=17,
        s=42,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
    )
    colorbar = ax.figure.colorbar(points, ax=ax, pad=0.01)
    colorbar.set_label("TMT plex")
    colorbar.set_ticks([1, 5, 9, 13, 17])
    ax.set_title("B  PCA by TMT plex", loc="left", fontweight="bold")
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}%)")


def plot_cohort_composition(ax: plt.Axes, metadata: pd.DataFrame) -> None:
    """Plot tumor and normal sample counts for every TMT plex."""
    counts = metadata.groupby(["plex", CONDITION_COL]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CONDITION_COLORS, fill_value=0)
    bottom = np.zeros(len(counts))
    for condition, color in CONDITION_COLORS.items():
        values = counts[condition].to_numpy()
        ax.bar(counts.index, values, bottom=bottom, color=color, label=condition)
        bottom += values
    ax.set_xticks(range(1, 18))
    ax.set_xlabel("TMT plex")
    ax.set_ylabel("Biological samples")
    ax.set_title("C  Cohort composition", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)


def plot_missingness(
    ax: plt.Axes, matrix: pd.DataFrame, metadata: pd.DataFrame
) -> None:
    """Plot sample-level protein missingness across TMT plexes."""
    frame = metadata[["plex", CONDITION_COL]].copy()
    frame["missing"] = matrix.isna().mean(axis=0).mul(100).loc[frame.index]
    sns.scatterplot(
        data=frame,
        x="plex",
        y="missing",
        hue=CONDITION_COL,
        palette=CONDITION_COLORS,
        s=38,
        alpha=0.85,
        ax=ax,
        legend=False,
    )
    ax.set_xticks(range(1, 18))
    ax.set_xlabel("TMT plex")
    ax.set_ylabel("Missing protein values per sample (%)")
    ax.set_title("D  Sample completeness", loc="left", fontweight="bold")


def plot_detection_by_condition(
    ax: plt.Axes, matrix: pd.DataFrame, metadata: pd.DataFrame
) -> None:
    """Compare detected-protein counts between tumor and normal samples."""
    frame = metadata[[CONDITION_COL]].copy()
    frame["detected"] = matrix.notna().sum(axis=0).loc[frame.index]
    order = list(CONDITION_COLORS)
    sns.boxplot(
        data=frame,
        x=CONDITION_COL,
        y="detected",
        hue=CONDITION_COL,
        order=order,
        palette=CONDITION_COLORS,
        legend=False,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=frame,
        x=CONDITION_COL,
        y="detected",
        hue=CONDITION_COL,
        order=order,
        palette=CONDITION_COLORS,
        legend=False,
        size=3,
        alpha=0.55,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Detected proteins per sample")
    ax.set_title("E  Detection by condition", loc="left", fontweight="bold")


def plot_scree(ax: plt.Axes, variance: np.ndarray) -> None:
    """Plot component-wise and cumulative PCA variance."""
    components = np.arange(1, len(variance) + 1)
    ax.bar(components, variance, color="#F4A261")
    ax.plot(components, np.cumsum(variance), color="#264653")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("F  PCA variance profile", loc="left", fontweight="bold")


def significance_colors(de_result: pd.DataFrame) -> pd.Series:
    """Map differential-expression calls to plot colors."""
    colors = pd.Series("#B8BEC3", index=de_result.index)
    colors.loc[de_result["significance"] == "UP"] = "#E74C3C"
    colors.loc[de_result["significance"] == "DOWN"] = "#3498DB"
    return colors


def plot_volcano(ax: plt.Axes, de_result: pd.DataFrame) -> None:
    """Plot effect size against adjusted significance."""
    colors = significance_colors(de_result)
    y_value = -np.log10(de_result["adj_pvalue"])
    ax.scatter(de_result["log2FC"], y_value, c=colors, s=13, alpha=0.7, linewidth=0)
    ax.axvline(-0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(0.05), color="#777777", linestyle="--", linewidth=0.8)
    counts = de_result["significance"].value_counts()
    ax.text(
        0.02,
        0.98,
        f"UP {counts.get('UP', 0):,}  |  DOWN {counts.get('DOWN', 0):,}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10(BH adjusted p-value)")
    ax.set_title("B  Tumor vs normal volcano", loc="left", fontweight="bold")


def plot_ma(ax: plt.Axes, de_result: pd.DataFrame) -> None:
    """Plot differential effect size against mean abundance."""
    colors = significance_colors(de_result)
    ax.scatter(
        de_result["AveExpr"],
        de_result["log2FC"],
        c=colors,
        s=13,
        alpha=0.7,
        linewidth=0,
    )
    ax.axhline(-0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Average log2 expression")
    ax.set_ylabel("log2 fold change")
    ax.set_title("C  Mean-abundance effect sizes", loc="left", fontweight="bold")


def top_de_proteins(de_result: pd.DataFrame, count: int = 12) -> list[str]:
    """Select the strongest significant up- and down-regulated proteins."""
    significant = de_result[de_result["adj_pvalue"] < 0.05]
    up = significant.nlargest(count, "log2FC")["ProteinName"].tolist()
    down = significant.nsmallest(count, "log2FC")["ProteinName"].tolist()
    return up + down


def plot_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    de_result: pd.DataFrame,
) -> None:
    """Plot standardized profiles for the strongest DE proteins."""
    proteins = [name for name in top_de_proteins(de_result) if name in matrix.index]
    sample_order = metadata.sort_values([CONDITION_COL, "plex"]).index
    values = matrix.loc[proteins, sample_order].copy()
    values = values.apply(lambda row: row.fillna(row.median()), axis=1)
    values = values.sub(values.mean(axis=1), axis=0)
    scale = values.std(axis=1).replace(0, 1)
    values = values.div(scale, axis=0).clip(-3, 3)
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=7)
    ax.set_xticks([])
    ax.set_xlabel("Samples ordered by condition, then plex")
    ax.set_title(
        "A  Strongest DE protein profiles",
        loc="left",
        fontweight="bold",
        pad=24,
    )
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Protein-wise z-score")

    condition_codes = metadata.loc[sample_order, CONDITION_COL].map(
        {"Primary Tumor": 0, "Solid Tissue Normal": 1}
    )
    inset = ax.inset_axes([0, 1.01, 1, 0.018])
    inset.imshow(
        np.asarray(condition_codes)[None, :],
        aspect="auto",
        cmap=ListedColormap(list(CONDITION_COLORS.values())),
        vmin=0,
        vmax=1,
    )
    inset.set_axis_off()


def representative_de_proteins(de_result: pd.DataFrame) -> pd.DataFrame:
    """Select three nonredundant proteins for condition-level panels."""
    significant = de_result[de_result["adj_pvalue"] < 0.05]
    selected = pd.concat(
        [
            significant.nlargest(1, "log2FC"),
            significant.nsmallest(1, "log2FC"),
            significant.nsmallest(5, "adj_pvalue"),
        ]
    )
    return selected.drop_duplicates("ProteinName").head(3)


def plot_protein_distribution(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    protein: pd.Series,
    panel: str,
) -> None:
    """Plot one representative protein across tumor and normal samples."""
    protein_name = str(protein["ProteinName"])
    frame = metadata[[CONDITION_COL]].copy()
    frame["expression"] = matrix.loc[protein_name, frame.index]
    frame = frame.dropna(subset=["expression"])
    order = list(CONDITION_COLORS)
    sns.boxplot(
        data=frame,
        x=CONDITION_COL,
        y="expression",
        hue=CONDITION_COL,
        order=order,
        palette=CONDITION_COLORS,
        legend=False,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=frame,
        x=CONDITION_COL,
        y="expression",
        hue=CONDITION_COL,
        order=order,
        palette=CONDITION_COLORS,
        legend=False,
        size=3,
        alpha=0.5,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("log2 protein abundance")
    ax.set_title(
        f"{panel}  {protein_name}\nlog2FC {protein['log2FC']:.2f}",
        loc="left",
        fontweight="bold",
    )


def render_overview(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    output: Path,
) -> None:
    """Render the six-panel cohort and computational overview."""
    pca, variance = pca_coordinates(matrix)
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), constrained_layout=True)
    plot_pca_condition(axes[0, 0], pca, variance, metadata)
    plot_pca_plex(axes[0, 1], pca, variance, metadata)
    plot_cohort_composition(axes[0, 2], metadata)
    plot_missingness(axes[1, 0], matrix, metadata)
    plot_detection_by_condition(axes[1, 1], matrix, metadata)
    plot_scree(axes[1, 2], variance)
    fig.suptitle(
        "PDC000125 CPTAC UCEC — Rust Mokume overview",
        fontsize=22,
        fontweight="bold",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_biology(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    de_result: pd.DataFrame,
    output: Path,
) -> None:
    """Render the six-panel differential-expression showcase."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), constrained_layout=True)
    plot_heatmap(axes[0, 0], matrix, metadata, de_result)
    plot_volcano(axes[0, 1], de_result)
    plot_ma(axes[0, 2], de_result)
    representatives = representative_de_proteins(de_result)
    for panel, (ax, (_, protein)) in zip(
        "DEF", zip(axes[1], representatives.iterrows()), strict=True
    ):
        plot_protein_distribution(ax, matrix, metadata, protein, panel)
    fig.suptitle(
        "PDC000125 CPTAC UCEC — differential expression",
        fontsize=22,
        fontweight="bold",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Load Rust outputs and render both CPTAC UCEC showcase figures."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    with threadpool_limits(limits=args.threads):
        matrix, metadata, de_result = load_inputs(
            args.protein_matrix, args.sdrf, args.de_result
        )
        overview = args.output_dir / "cptac_ucec_overview.png"
        biology = args.output_dir / "cptac_ucec_biology.png"
        render_overview(matrix, metadata, overview)
        render_biology(matrix, metadata, de_result, biology)
    print(
        f"wrote {overview} and {biology}: {matrix.shape[0]:,} proteins, "
        f"{matrix.shape[1]:,} samples, {len(de_result):,} DE rows"
    )


if __name__ == "__main__":
    main()
