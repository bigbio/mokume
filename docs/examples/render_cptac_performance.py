#!/usr/bin/env python3
"""Render PDC000125 computational-QC panels from current Rust outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from mokume.plotting._common import panel_label, static_style, style_axes
from mokume.plotting.style import (
    SAMPLE_POINT_COLOR,
    SEQUENTIAL_COLORS,
    SINGLE_SERIES_COLOR,
    TEXT_COLOR,
    WHITE,
    category_colors,
)


CONDITION_COL = "factor value[condition]"
CONDITION_COLORS = dict(
    zip(("Primary Tumor", "Solid Tissue Normal"), category_colors(2), strict=True)
)
STAGE_COLORS = category_colors(3)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL_COLORS)
TIMING_FILES = {
    "raw_intensity": "Raw intensity",
    "globalmedian": "+ GlobalMedian",
    "globalmedian_irs": "+ IRS",
    "ratio": "Ratio",
}


def parse_args() -> argparse.Namespace:
    """Parse input matrices, timing records, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_matrix", type=Path)
    parser.add_argument("raw_matrix", type=Path)
    parser.add_argument("globalmedian_matrix", type=Path)
    parser.add_argument("irs_matrix", type=Path)
    parser.add_argument("ratio_matrix", type=Path)
    parser.add_argument("sdrf", type=Path)
    parser.add_argument("timing_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def read_matrix(path: Path, *, linear: bool) -> pd.DataFrame:
    """Read a Rust-written protein matrix under its scale contract."""
    matrix = pd.read_csv(path).set_index("ProteinName")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if linear:
        matrix = matrix.where(matrix > 0)
    return matrix


def load_timings(directory: Path) -> pd.DataFrame:
    """Load GNU time records for the four profiled workflows."""
    records = []
    for key, display_name in TIMING_FILES.items():
        timing_path = directory / f"{key}.time"
        fields = timing_path.read_text(encoding="utf-8").strip().split(",")
        if len(fields) != 3 or fields[0] != key:
            raise ValueError(f"invalid timing record: {timing_path}")
        records.append(
            {
                "workflow": display_name,
                "elapsed_seconds": float(fields[1]),
                "peak_gib": float(fields[2]) / 1024**2,
            }
        )
    return pd.DataFrame(records)


def load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Load and validate the matrices, SDRF metadata, and timings."""
    matrices = {
        "final": read_matrix(args.final_matrix, linear=True),
        "raw": read_matrix(args.raw_matrix, linear=True),
        "globalmedian": read_matrix(args.globalmedian_matrix, linear=True),
        "irs": read_matrix(args.irs_matrix, linear=True),
        "ratio": read_matrix(args.ratio_matrix, linear=False),
    }
    metadata = pd.read_csv(args.sdrf, sep="\t", dtype=str)
    metadata = metadata.drop_duplicates("source name").set_index("source name")
    metadata["plex"] = (
        metadata.index.to_series()
        .str.extract(r"PDC000125-p(\d{2})_", expand=False)
        .astype(int)
    )
    validate_inputs(matrices, metadata)
    return matrices, metadata, load_timings(args.timing_dir)


def validate_inputs(matrices: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> None:
    """Require aligned axes and agreement with the final IRS matrix."""
    intermediate = matrices["raw"]
    for name in ("globalmedian", "irs"):
        if not intermediate.index.equals(matrices[name].index) or not (
            intermediate.columns.equals(matrices[name].columns)
        ):
            raise ValueError(f"{name} matrix does not match raw matrix axes")

    final = matrices["final"]
    biological = metadata.index[metadata[CONDITION_COL].isin(CONDITION_COLORS)]
    if set(final.columns) != set(biological):
        raise ValueError("final matrix does not contain the 153 biological samples")
    if set(matrices["ratio"].columns) != set(biological):
        raise ValueError("ratio matrix does not contain the same biological samples")
    if not final.index.isin(intermediate.index).all():
        raise ValueError("final proteins are not a subset of intermediate outputs")

    final_values = final.to_numpy(dtype=float)
    irs_values = matrices["irs"].loc[final.index, final.columns].to_numpy(dtype=float)
    if not np.array_equal(np.isfinite(final_values), np.isfinite(irs_values)):
        raise ValueError("final and retained-reference IRS missing-value masks differ")
    finite = np.isfinite(final_values)
    scale = np.maximum(np.abs(final_values[finite]), np.abs(irs_values[finite]))
    scale = np.maximum(scale, 1.0)
    max_relative_delta = np.max(
        np.abs(final_values[finite] - irs_values[finite]) / scale
    )
    if max_relative_delta > 1e-12:
        raise ValueError(
            "final matrix is not numerically consistent with the IRS intermediate"
        )


def ordered_biological_metadata(
    matrix: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Order biological samples by condition and then TMT plex."""
    selected = metadata.loc[matrix.columns].copy()
    selected["condition_order"] = selected[CONDITION_COL].map(
        {"Primary Tumor": 0, "Solid Tissue Normal": 1}
    )
    return selected.sort_values(["condition_order", "plex"])


def plot_sample_correlation(
    ax: plt.Axes, matrix: pd.DataFrame, metadata: pd.DataFrame
) -> plt.Axes:
    """Plot sample correlations and return the strip that carries the title."""
    ordered = ordered_biological_metadata(matrix, metadata)
    correlation = np.log2(matrix.loc[:, ordered.index]).corr()
    off_diagonal = correlation.to_numpy()[
        np.triu_indices_from(correlation.to_numpy(), k=1)
    ]
    lower = np.floor(np.nanquantile(off_diagonal, 0.01) * 20) / 20
    image = ax.imshow(
        correlation,
        aspect="auto",
        cmap=SEQUENTIAL_CMAP,
        vmin=lower,
        vmax=1,
    )
    tumor_count = int((ordered[CONDITION_COL] == "Primary Tumor").sum())
    ax.axhline(tumor_count - 0.5, color=WHITE, linewidth=0.8)
    ax.axvline(tumor_count - 0.5, color=WHITE, linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{len(ordered)} samples ordered by condition, then plex")
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Pearson r", color=TEXT_COLOR)

    condition_codes = ordered[CONDITION_COL].map(
        {"Primary Tumor": 0, "Solid Tissue Normal": 1}
    )
    strip = ax.inset_axes([0, 1.01, 1, 0.025])
    strip.imshow(
        condition_codes.to_numpy()[None, :],
        aspect="auto",
        cmap=ListedColormap(list(CONDITION_COLORS.values())),
        vmin=0,
        vmax=1,
    )
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    strip.set_title("Sample correlation", loc="left")
    return strip


def stage_quantiles(
    matrix: pd.DataFrame, proteins: pd.Index, samples: pd.Index
) -> pd.DataFrame:
    """Compute per-channel protein-intensity quantiles for one stage."""
    values = np.log2(matrix.loc[proteins, samples])
    return values.quantile([0.1, 0.25, 0.5, 0.75, 0.9])


def draw_quantile_stage(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    color: str,
    order: pd.Index,
) -> None:
    """Draw one normalization-stage quantile ribbon."""
    ordered = frame.loc[:, order]
    x_values = np.arange(1, len(order) + 1)
    ax.fill_between(
        x_values, ordered.loc[0.1], ordered.loc[0.9], color=color, alpha=0.13, lw=0
    )
    ax.fill_between(
        x_values, ordered.loc[0.25], ordered.loc[0.75], color=color, alpha=0.3, lw=0
    )
    ax.plot(x_values, ordered.loc[0.5], color=color, linewidth=1)
    ax.set_title(title, loc="left")
    ax.set_xlim(1, len(order))
    ax.set_xticks([1, len(order)])
    ax.set_xlabel("Channel")
    style_axes(ax)


def plot_normalization_trajectory(
    axes: list[plt.Axes], matrices: dict[str, pd.DataFrame]
) -> None:
    """Plot sample-intensity distributions across normalization stages."""
    stages = [
        ("raw", "Raw intensity", STAGE_COLORS[0]),
        ("globalmedian", "GlobalMedian", STAGE_COLORS[1]),
        ("irs", "+ IRS", STAGE_COLORS[2]),
    ]
    proteins = matrices["final"].index
    samples = matrices["raw"].columns
    quantiles = {
        key: stage_quantiles(matrices[key], proteins, samples) for key, _, _ in stages
    }
    order = quantiles["raw"].loc[0.5].sort_values().index
    for ax, (key, title, color) in zip(axes, stages, strict=True):
        draw_quantile_stage(ax, quantiles[key], title, color, order)
    axes[0].set_ylabel("log2 protein intensity")


def pca_coordinates(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit sample PCA and return two coordinates plus explained variance."""
    model = PCA(n_components=2)
    coordinates = model.fit_transform(matrix.T)
    frame = pd.DataFrame(coordinates, index=matrix.columns, columns=["PC1", "PC2"])
    return frame, model.explained_variance_ratio_ * 100


def draw_pca_stage(
    ax: plt.Axes,
    title: str,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    """Draw one plex-coloured PCA stage."""
    coordinates, variance = pca_coordinates(matrix)
    plexes = metadata.loc[coordinates.index, "plex"]
    for plex, color in zip(sorted(plexes.unique()), category_colors(17), strict=True):
        ax.scatter(
            coordinates.loc[plexes == plex, "PC1"],
            coordinates.loc[plexes == plex, "PC2"],
            s=8,
            color=color,
            linewidths=0,
            alpha=0.85,
            label=str(plex),
        )
    ax.set_title(title, loc="left")
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}%)")
    style_axes(ax)


def plot_irs_pca(
    axes: list[plt.Axes], matrices: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> None:
    """Plot matched sample PCA views before and after IRS."""
    proteins = matrices["final"].index
    samples = matrices["final"].columns
    before = np.log2(matrices["globalmedian"].loc[proteins, samples])
    after = np.log2(matrices["irs"].loc[proteins, samples])
    complete = before.notna().all(axis=1) & after.notna().all(axis=1)
    views = [
        ("PCA before IRS", before.loc[complete]),
        ("PCA after IRS", after.loc[complete]),
    ]
    for ax, (title, matrix) in zip(axes, views, strict=True):
        draw_pca_stage(ax, title, matrix, metadata)
    axes[0].figure.legend(
        *axes[0].get_legend_handles_labels(),
        title=f"TMT plex (same {int(complete.sum()):,} complete proteins in both fits)",
        ncol=9,
        loc="outside lower right",
        frameon=False,
        labelcolor=TEXT_COLOR,
        columnspacing=0.6,
        handletextpad=0.1,
    )


def pooled_reference_cv(
    matrix: pd.DataFrame, proteins: pd.Index, references: pd.Index
) -> pd.Series:
    """Calculate protein CV across sufficiently observed pooled references."""
    values = matrix.loc[proteins, references]
    values = values.loc[values.notna().sum(axis=1) >= 12]
    return values.std(axis=1, ddof=1).div(values.mean(axis=1)).mul(100)


def plot_cv_reproducibility(
    ax: plt.Axes, matrices: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> None:
    """Plot pooled-reference CV before and after IRS alignment."""
    references = metadata.index[metadata[CONDITION_COL] == "Reference"]
    proteins = matrices["final"].index
    before = pooled_reference_cv(matrices["globalmedian"], proteins, references)
    after = pooled_reference_cv(matrices["irs"], proteins, references)
    shared = before.index.intersection(after.index)
    series = [
        ("Before IRS", before.loc[shared], STAGE_COLORS[1]),
        ("After IRS", after.loc[shared], STAGE_COLORS[2]),
    ]
    for name, values, color in series:
        ordered = np.sort(values.to_numpy())
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100
        ax.plot(
            ordered,
            cumulative,
            color=color,
            linewidth=1.5,
            label=f"{name}: median {np.median(ordered):.1f}%",
        )
    upper = np.nanquantile(before.loc[shared], 0.99) * 1.05
    ax.set_xlim(-upper * 0.03, upper)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Pooled-reference protein CV (%)")
    ax.set_ylabel("Cumulative proteins (%)")
    ax.set_title("Pooled-reference alignment", loc="left")
    ax.legend(
        title=f"n={len(shared):,} (≥12/17 references)",
        frameon=False,
        loc="lower right",
        labelcolor=TEXT_COLOR,
    )
    style_axes(ax)


def quantification_effects(
    matrices: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> pd.DataFrame:
    """Calculate matched tumor-normal effects from intensity and ratio paths."""
    final = matrices["final"]
    ratio = matrices["ratio"]
    proteins = final.index.intersection(ratio.index)
    samples = final.columns
    tumor = metadata.index[metadata[CONDITION_COL] == "Primary Tumor"].intersection(
        samples
    )
    normal = metadata.index[
        metadata[CONDITION_COL] == "Solid Tissue Normal"
    ].intersection(samples)
    intensity = np.log2(final.loc[proteins, samples])
    effects = pd.DataFrame(
        {
            "intensity": intensity[tumor].mean(axis=1) - intensity[normal].mean(axis=1),
            "ratio": ratio.loc[proteins, tumor].mean(axis=1)
            - ratio.loc[proteins, normal].mean(axis=1),
        }
    )
    return effects.dropna()


def plot_method_concordance(
    ax: plt.Axes, matrices: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> None:
    """Plot concordance between independent Rust TMT quantification paths."""
    effects = quantification_effects(matrices, metadata)
    pearson = pearsonr(effects["intensity"], effects["ratio"]).statistic
    spearman = spearmanr(effects["intensity"], effects["ratio"]).statistic
    points = ax.hexbin(
        effects["intensity"],
        effects["ratio"],
        gridsize=45,
        mincnt=1,
        bins="log",
        cmap=SEQUENTIAL_CMAP,
    )
    lower = min(effects.min())
    upper = max(effects.max())
    ax.plot(
        [lower, upper],
        [lower, upper],
        color=SAMPLE_POINT_COLOR,
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Intensity + GlobalMedian + IRS effect\n(mean log2 tumor - normal)")
    ax.set_ylabel("Reporter-ratio effect\n(mean log2 tumor - normal)")
    ax.set_title("Quantification concordance", loc="left")
    ax.text(
        0.02,
        0.98,
        f"n={len(effects):,} proteins\nPearson {pearson:.3f}\nSpearman {spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        color=TEXT_COLOR,
    )
    colorbar = ax.figure.colorbar(points, ax=ax, pad=0.01)
    colorbar.set_label("Protein density (log)", color=TEXT_COLOR)
    style_axes(ax)


def plot_execution_profile(axes: list[plt.Axes], timings: pd.DataFrame) -> None:
    """Plot elapsed time and peak RSS side by side on their own axes."""
    positions = np.arange(len(timings))
    panels = (
        ("elapsed_seconds", "Elapsed wall time (s)", "{:.1f}", "Run time, 24 threads"),
        ("peak_gib", "Peak RSS (GiB)", "{:.2f}", "Peak memory"),
    )
    for ax, (column, xlabel, template, title) in zip(axes, panels, strict=True):
        bars = ax.barh(
            positions, timings[column], color=SINGLE_SERIES_COLOR, linewidth=0
        )
        ax.bar_label(
            bars,
            [template.format(value) for value in timings[column]],
            padding=2,
            color=TEXT_COLOR,
        )
        ax.set_yticks(positions, timings["workflow"] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlim(0, timings[column].max() * 1.35)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        style_axes(ax)


def render(
    matrices: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    timings: pd.DataFrame,
    output: Path,
) -> None:
    """Render the computational-QC panels to one PNG."""
    figure = plt.figure(figsize=(12.6, 7.6), layout="constrained")
    top, bottom = figure.subfigures(2, 1)
    upper = top.subplots(
        1, 6, width_ratios=[2.6, 1, 1, 1, 1.4, 1.4], gridspec_kw={"wspace": 0.04}
    )
    lower = bottom.subplots(
        1, 4, width_ratios=[2.2, 2.6, 1.2, 1.2], gridspec_kw={"wspace": 0.04}
    )
    upper[2].sharey(upper[1])
    upper[3].sharey(upper[1])
    strip = plot_sample_correlation(upper[0], matrices["final"], metadata)
    plot_normalization_trajectory(upper[1:4], matrices)
    plot_irs_pca(upper[4:6], matrices, metadata)
    plot_cv_reproducibility(lower[0], matrices, metadata)
    plot_method_concordance(lower[1], matrices, metadata)
    plot_execution_profile(lower[2:4], timings)
    labelled = [strip, upper[1], upper[4], lower[0], lower[1], lower[2]]
    for letter, ax in zip("abcdef", labelled, strict=True):
        panel_label(ax, letter)
    figure.savefig(output, dpi=300, facecolor=WHITE)
    plt.close(figure)


def main() -> None:
    """Load current Rust outputs and render the documented figure."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with static_style(), threadpool_limits(limits=args.threads):
        matrices, metadata, timings = load_inputs(args)
        render(matrices, metadata, timings, args.output)
    effects = quantification_effects(matrices, metadata)
    print(
        f"wrote {args.output}: {matrices['final'].shape[0]:,} final proteins, "
        f"{matrices['final'].shape[1]:,} biological samples, "
        f"{len(effects):,} shared method effects"
    )


if __name__ == "__main__":
    main()
