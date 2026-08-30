#!/usr/bin/env python3
"""Render PDC000125 computational-QC panels from current Rust outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


CONDITION_COL = "factor value[condition]"
CONDITION_COLORS = {
    "Primary Tumor": "#2A9D8F",
    "Solid Tissue Normal": "#F9844A",
}
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
) -> None:
    """Plot the sample correlation matrix with condition annotation."""
    ordered = ordered_biological_metadata(matrix, metadata)
    correlation = np.log2(matrix.loc[:, ordered.index]).corr()
    off_diagonal = correlation.to_numpy()[
        np.triu_indices_from(correlation.to_numpy(), k=1)
    ]
    lower = np.floor(np.nanquantile(off_diagonal, 0.01) * 20) / 20
    image = ax.imshow(
        correlation,
        aspect="auto",
        cmap="viridis",
        vmin=lower,
        vmax=1,
    )
    tumor_count = int((ordered[CONDITION_COL] == "Primary Tumor").sum())
    ax.axhline(tumor_count - 0.5, color="white", linewidth=0.8)
    ax.axvline(tumor_count - 0.5, color="white", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{len(ordered)} samples ordered by condition, then plex")
    ax.set_title("A  Sample correlation", loc="left", fontweight="bold")
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Pearson r")

    condition_codes = ordered[CONDITION_COL].map(
        {"Primary Tumor": 0, "Solid Tissue Normal": 1}
    )
    inset = ax.inset_axes([0, 1.01, 1, 0.018])
    inset.imshow(
        condition_codes.to_numpy()[None, :],
        aspect="auto",
        cmap=ListedColormap(list(CONDITION_COLORS.values())),
        vmin=0,
        vmax=1,
    )
    inset.set_axis_off()


def stage_quantiles(
    matrix: pd.DataFrame, proteins: pd.Index, samples: pd.Index
) -> pd.DataFrame:
    """Compute per-channel protein-intensity quantiles for one stage."""
    values = np.log2(matrix.loc[proteins, samples])
    return values.quantile([0.1, 0.25, 0.5, 0.75, 0.9])


def draw_quantile_stage(
    ax: plt.Axes,
    frame: pd.DataFrame,
    details: tuple[int, str, str],
    order: pd.Index,
    y_limits: tuple[float, float],
) -> None:
    """Draw one normalization-stage quantile ribbon in an inset axis."""
    position, title, color = details
    inset = ax.inset_axes([0.02 + position * 0.325, 0.12, 0.295, 0.78])
    ordered = frame.loc[:, order]
    x_values = np.arange(1, len(order) + 1)
    inset.fill_between(
        x_values, ordered.loc[0.1], ordered.loc[0.9], color=color, alpha=0.13
    )
    inset.fill_between(
        x_values, ordered.loc[0.25], ordered.loc[0.75], color=color, alpha=0.3
    )
    inset.plot(x_values, ordered.loc[0.5], color=color, linewidth=1.1)
    inset.set_title(title, fontsize=10)
    inset.set_xlim(1, len(order))
    inset.set_ylim(*y_limits)
    inset.set_xticks([1, len(order)])
    inset.set_xlabel("Channel order", fontsize=9)
    if position == 0:
        inset.set_ylabel("log2 protein intensity", fontsize=9)
    else:
        inset.tick_params(labelleft=False)


def plot_normalization_trajectory(
    ax: plt.Axes, matrices: dict[str, pd.DataFrame]
) -> None:
    """Plot sample-intensity distributions across normalization stages."""
    stages = [
        ("raw", "Raw", "#264653"),
        ("globalmedian", "GlobalMedian", "#2A9D8F"),
        ("irs", "+ IRS", "#E76F51"),
    ]
    proteins = matrices["final"].index
    samples = matrices["raw"].columns
    quantiles = {
        key: stage_quantiles(matrices[key], proteins, samples) for key, _, _ in stages
    }
    order = quantiles["raw"].loc[0.5].sort_values().index
    limits = [
        quantiles[key].loc[level, order].to_numpy()
        for key, _, _ in stages
        for level in (0.1, 0.9)
    ]
    y_min = min(np.nanmin(values) for values in limits)
    y_max = max(np.nanmax(values) for values in limits)

    ax.set_axis_off()
    ax.text(
        0,
        1.03,
        "B  Sample intensity distributions",
        transform=ax.transAxes,
        fontweight="bold",
    )
    for position, (key, title, color) in enumerate(stages):
        draw_quantile_stage(
            ax,
            quantiles[key],
            (position, title, color),
            order,
            (y_min, y_max),
        )


def pca_coordinates(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit sample PCA and return two coordinates plus explained variance."""
    model = PCA(n_components=2)
    coordinates = model.fit_transform(matrix.T)
    frame = pd.DataFrame(coordinates, index=matrix.columns, columns=["PC1", "PC2"])
    return frame, model.explained_variance_ratio_ * 100


def draw_pca_stage(
    ax: plt.Axes,
    position: int,
    title: str,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
):
    """Draw one plex-coloured PCA stage and return its point collection."""
    inset = ax.inset_axes([0.02 + position * 0.45, 0.12, 0.4, 0.78])
    coordinates, variance = pca_coordinates(matrix)
    points = inset.scatter(
        coordinates["PC1"],
        coordinates["PC2"],
        c=metadata.loc[coordinates.index, "plex"],
        cmap="turbo",
        vmin=1,
        vmax=17,
        s=18,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.9,
    )
    inset.set_title(title, fontsize=10)
    inset.set_xlabel(f"PC1 ({variance[0]:.1f}%)", fontsize=9)
    inset.set_ylabel(f"PC2 ({variance[1]:.1f}%)", fontsize=9)
    return points


def plot_irs_pca(
    ax: plt.Axes, matrices: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> None:
    """Plot matched sample PCA views before and after IRS."""
    proteins = matrices["final"].index
    samples = matrices["final"].columns
    before = np.log2(matrices["globalmedian"].loc[proteins, samples])
    after = np.log2(matrices["irs"].loc[proteins, samples])
    complete = before.notna().all(axis=1) & after.notna().all(axis=1)
    views = [
        ("Before IRS", before.loc[complete]),
        ("After IRS", after.loc[complete]),
    ]

    ax.set_axis_off()
    ax.text(
        0,
        1.03,
        "C  IRS effect on sample structure",
        transform=ax.transAxes,
        fontweight="bold",
    )
    point_sets = []
    for position, (title, matrix) in enumerate(views):
        point_sets.append(draw_pca_stage(ax, position, title, matrix, metadata))
    color_axis = ax.inset_axes([0.93, 0.18, 0.018, 0.66])
    colorbar = ax.figure.colorbar(point_sets[-1], cax=color_axis)
    colorbar.set_label("TMT plex", fontsize=9)
    colorbar.set_ticks([1, 5, 9, 13, 17])
    ax.text(
        0.02,
        0.01,
        f"Same {int(complete.sum()):,} complete proteins in both PCA fits",
        transform=ax.transAxes,
        fontsize=9,
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
        ("Before IRS", before.loc[shared], "#264653"),
        ("After IRS", after.loc[shared], "#E76F51"),
    ]
    for name, values, color in series:
        ordered = np.sort(values.to_numpy())
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100
        ax.plot(
            ordered,
            cumulative,
            color=color,
            linewidth=2,
            label=f"{name}: median {np.median(ordered):.1f}%",
        )
    upper = np.nanquantile(before.loc[shared], 0.99) * 1.05
    ax.set_xlim(-upper * 0.03, upper)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Pooled-reference protein CV (%)")
    ax.set_ylabel("Cumulative proteins (%)")
    ax.set_title("D  Pooled-reference alignment", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(
        0.02,
        0.98,
        f"n={len(shared):,}; observed in at least 12/17 references",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )


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
        cmap="viridis",
    )
    lower = min(effects.min())
    upper = max(effects.max())
    ax.plot([lower, upper], [lower, upper], color="#777777", linestyle="--")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Intensity + GlobalMedian + IRS effect (mean log2 tumor - normal)")
    ax.set_ylabel("Reporter-ratio effect (mean log2 tumor - normal)")
    ax.set_title("E  Quantification concordance", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.98,
        f"n={len(effects):,} proteins\nPearson {pearson:.3f}  |  Spearman {spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    colorbar = ax.figure.colorbar(points, ax=ax, pad=0.01)
    colorbar.set_label("Protein density (log)")


def plot_execution_profile(ax: plt.Axes, timings: pd.DataFrame) -> None:
    """Plot elapsed time and peak RSS from the local single-run profile."""
    positions = np.arange(len(timings))
    bars = ax.barh(
        positions,
        timings["elapsed_seconds"],
        color=["#264653", "#2A9D8F", "#E76F51", "#F4A261"],
        alpha=0.85,
    )
    ax.set_yticks(positions, timings["workflow"])
    ax.invert_yaxis()
    ax.set_xlabel("Elapsed wall time (seconds)")
    ax.set_title("F  Local execution profile", loc="left", fontweight="bold")
    for rectangle, value in zip(bars, timings["elapsed_seconds"], strict=True):
        ax.text(
            rectangle.get_width() + 1,
            rectangle.get_y() + rectangle.get_height() / 2,
            f"{value:.1f}s",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, timings["elapsed_seconds"].max() * 1.22)

    memory_axis = ax.twiny()
    memory_axis.scatter(
        timings["peak_gib"],
        positions,
        marker="D",
        s=46,
        color="#8E44AD",
        label="Peak RSS",
        zorder=4,
    )
    memory_axis.set_xlabel("Peak RSS (GiB)", color="#8E44AD")
    memory_axis.tick_params(axis="x", colors="#8E44AD")
    memory_axis.set_xlim(0, timings["peak_gib"].max() * 1.25)
    ax.text(
        0.01,
        0.01,
        "Single local run; 4.13M QPX rows; 24 threads",
        transform=ax.transAxes,
        fontsize=9,
    )


def render(
    matrices: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    timings: pd.DataFrame,
    output: Path,
) -> None:
    """Render the six computational-QC panels to one PNG."""
    figure, axes = plt.subplots(2, 3, figsize=(24, 15), constrained_layout=True)
    plot_sample_correlation(axes[0, 0], matrices["final"], metadata)
    plot_normalization_trajectory(axes[0, 1], matrices)
    plot_irs_pca(axes[0, 2], matrices, metadata)
    plot_cv_reproducibility(axes[1, 0], matrices, metadata)
    plot_method_concordance(axes[1, 1], matrices, metadata)
    plot_execution_profile(axes[1, 2], timings)
    figure.suptitle(
        "PDC000125 CPTAC UCEC — Rust Mokume computational QC and performance",
        fontsize=22,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    """Load current Rust outputs and render the documented figure."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    with threadpool_limits(limits=args.threads):
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
