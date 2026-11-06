"""
AdaTiSS tissue specificity distribution and per-tissue bar chart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy
from scipy.stats import norm as _norm

from mokume.plotting._common import panel_label, static_style, style_axes
from mokume.plotting.style import (
    CONDITION_COLORS,
    SAMPLE_POINT_COLOR,
    SINGLE_SERIES_COLOR,
    TEXT_COLOR,
    WHITE,
)

if TYPE_CHECKING:
    from mokume.tissuemap.tissue_specificity import GMMThresholds

logger = logging.getLogger(__name__)


def _overlay_gmm_curves(ax, gmm, x_upper: float) -> None:
    """Draw GMM density curves and their compact legend below the panel."""
    x_grid = np.linspace(0, x_upper, 500)
    p_bg = gmm.bg_weight * _norm.pdf(x_grid, gmm.bg_mean, gmm.bg_std)
    p_sp = gmm.sp_weight * _norm.pdf(x_grid, gmm.sp_mean, gmm.sp_std)
    ax.plot(
        x_grid,
        p_bg,
        "-",
        color=CONDITION_COLORS[1],
        lw=2,
        zorder=3,
        label=f"Background (\u03bc={gmm.bg_mean:.2f})",
    )
    ax.plot(
        x_grid,
        p_sp,
        "-",
        color=CONDITION_COLORS[2],
        lw=2,
        zorder=3,
        label=f"Specific (\u03bc={gmm.sp_mean:.2f})",
    )
    ax.plot(
        x_grid,
        p_bg + p_sp,
        "-",
        color=TEXT_COLOR,
        lw=5,
        alpha=0.25,
        zorder=2,
        label="GMM mixture",
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, y=-34, units="points"),
        borderaxespad=0,
        fontsize=8,
        labelcolor=TEXT_COLOR,
    )


def _draw_thresholds(ax, categories, thresholds) -> None:
    """Join each neutral dashed threshold line to its label with an elbow."""
    enriched, specific, housekeeping = thresholds
    n_specific = int((categories == "tissue-specific").sum())
    n_enriched = int(categories.isin(["tissue-specific", "tissue-enriched"]).sum())
    labels = [
        (specific, f"Specific ≥ {specific:.2f} (n={n_specific})"),
        (enriched, f"Enriched ≥ {enriched:.2f} (n={n_enriched})"),
    ]
    if housekeeping is not None:
        n_hk = int((categories == "house-keeping").sum())
        labels.append(
            (housekeeping, f"House-keeping |TS| < {housekeeping:.2f} (n={n_hk})")
        )
    # Labels share one column right of the highest threshold. Each line rises to
    # its own row, the leftmost to the top, so no leader crosses another line.
    column = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, x=6, units="points")
    for row, (value, label) in enumerate(sorted(labels)):
        ax.annotate(
            label,
            (value, 0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(max(labels)[0], 0.95 - row * 0.1),
            textcoords=column,
            va="center",
            color=TEXT_COLOR,
            arrowprops={
                "arrowstyle": "-",
                "connectionstyle": "angle,angleA=0,angleB=90,rad=0",
                "relpos": (0, 0.5),
                "shrinkA": 2,
                "shrinkB": 0,
                "color": SAMPLE_POINT_COLOR,
                "linestyle": "--",
                "linewidth": 1,
            },
        )


def _plot_histogram_panel(
    ax, max_ts, categories, ts_enriched, ts_specific, ts_housekeeping, gmm
) -> None:
    """Panel A: TS histogram with direct threshold labels and optional GMM."""
    p99 = float(max_ts.quantile(0.99)) if len(max_ts) > 0 else ts_specific * 3
    x_upper = max(p99, ts_specific * 2)
    n_outliers = int((max_ts > x_upper).sum())
    use_density = gmm is not None
    heights, _, _ = ax.hist(
        max_ts.clip(upper=x_upper).values,
        bins=50,
        color=SINGLE_SERIES_COLOR,
        edgecolor=WHITE,
        linewidth=0.5,
        alpha=0.6,
        density=use_density,
    )
    _draw_thresholds(ax, categories, (ts_enriched, ts_specific, ts_housekeeping))
    if gmm is not None:
        _overlay_gmm_curves(ax, gmm, x_upper)

    left_bound = float(max_ts.min()) - 0.5 if len(max_ts) > 0 else 0.0
    ax.set_xlim(left=left_bound, right=x_upper * 1.05)
    if n_outliers > 0:
        ax.annotate(
            f"{n_outliers} > {x_upper:.1f}",
            (x_upper, heights[-1]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=TEXT_COLOR,
            style="italic",
        )
    ax.set_xlabel("AdaTiSS max TS score")
    ax.set_ylabel("Density" if use_density else "Number of proteins")
    ax.set_title(
        "AdaTiSS tissue specificity score distribution",
        loc="left",
    )
    style_axes(ax)


def _compute_specific_counts(
    ts_df: pd.DataFrame,
    unique_tissues: list[str],
) -> pd.Series:
    """Count tissue-specific proteins per tissue."""
    categories = ts_df.get("enrichment_category", pd.Series(dtype=str))
    specific_mask = categories == "tissue-specific"
    if specific_mask.any():
        return (
            ts_df.loc[specific_mask, "max_tissue"]
            .value_counts()
            .reindex(unique_tissues)
            .fillna(0)
            .astype(int)
        )
    return pd.Series(0, index=unique_tissues)


_BAR_PANEL_TOP_N = 25


def _plot_bar_panel(
    ax,
    specific_per_tissue: pd.Series,
    ts_specific: float,
    top_n: int = _BAR_PANEL_TOP_N,
) -> None:
    """Panel B: horizontal bar chart of tissue-specific protein counts.

    Only tissues that actually carry tissue-specific proteins are shown, capped
    at the ``top_n`` highest counts so the y-axis labels stay legible (a 137-row
    bar chart overlaps its own tissue names).
    """
    nonzero = specific_per_tissue[specific_per_tissue > 0]
    ranked = (nonzero if not nonzero.empty else specific_per_tissue).nlargest(top_n)
    spt = ranked.iloc[::-1]
    ax.barh(
        range(len(spt)),
        spt.values,
        color=SINGLE_SERIES_COLOR,
        height=0.7,
        linewidth=0,
    )
    ax.set_yticks(range(len(spt)))
    ax.set_yticklabels(spt.index, fontsize=8)
    ax.set_xlabel(
        f"Number of tissue-specific proteins\n(TS ≥ {ts_specific:.2f})",
    )
    title = "Tissue-specific proteins per tissue"
    if len(nonzero) > top_n:
        title += f" (top {top_n})"
    ax.set_title(
        title,
        loc="left",
    )
    style_axes(ax)
    for i, v in enumerate(spt.values):
        if v > 0:
            ax.text(v + 0.5, i, str(v), va="center", fontsize=6, color=TEXT_COLOR)


@static_style()
def plot_ts_distribution(
    ts_df: pd.DataFrame,
    unique_tissues: list[str],
    out_dir: Path,
    *,
    ts_enriched: float = 2.5,
    ts_specific: float = 4.0,
    ts_housekeeping: float | None = None,
    dpi: int = 300,
    save_pdf: bool = True,
) -> None:
    """Two-panel figure: TS score histogram + per-tissue specific proteins.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Output of :func:`compute_ts_scores` with ``max_ts`` and
        ``enrichment_category`` columns.  GMM fit is read from
        ``ts_df.attrs["gmm"]`` if available.
    unique_tissues : list[str]
        Tissue names (column order).
    out_dir : Path
        Directory to save plots.
    """
    if "max_ts" not in ts_df.columns:
        logger.warning("No max_ts in ts_df, skipping TS distribution plot")
        return

    max_ts = ts_df["max_ts"].dropna()
    categories = ts_df.get("enrichment_category", pd.Series(dtype=str))
    gmm: GMMThresholds | None = ts_df.attrs.get("gmm")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            7.2,
            max(3.2, min(len(unique_tissues), _BAR_PANEL_TOP_N) * 0.14 + 1.4),
        ),
    )
    _plot_histogram_panel(
        axes[0],
        max_ts,
        categories,
        ts_enriched,
        ts_specific,
        ts_housekeeping,
        gmm,
    )
    _plot_bar_panel(
        axes[1],
        _compute_specific_counts(ts_df, unique_tissues),
        ts_specific,
    )
    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter)

    plt.tight_layout()
    fig.savefig(out_dir / "ts_distribution.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_dir / "ts_distribution.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved ts_distribution.png")


@static_style()
def plot_specific_per_tissue(
    ts_df: pd.DataFrame,
    unique_tissues: list[str],
    out_dir: Path,
    *,
    ts_specific: float = 4.0,
    dpi: int = 300,
    save_pdf: bool = True,
) -> None:
    """Standalone per-tissue bar chart of tissue-specific protein counts.

    This is a simpler standalone version of Panel B above, useful when
    the combined figure is not needed.
    """
    specific_per_tissue = _compute_specific_counts(ts_df, unique_tissues)

    fig, ax = plt.subplots(figsize=(3.5, max(2.6, len(unique_tissues) * 0.16)))

    spt = specific_per_tissue.iloc[::-1]
    ax.barh(
        range(len(spt)),
        spt.values,
        color=SINGLE_SERIES_COLOR,
        height=0.7,
        linewidth=0,
    )
    ax.set_yticks(range(len(spt)))
    ax.set_yticklabels(spt.index, fontsize=8)
    ax.set_xlabel(
        f"Number of tissue-specific proteins\n(TS ≥ {ts_specific:.2f})",
    )
    ax.set_title(
        "Tissue-specific proteins per tissue (AdaTiSS)",
    )
    style_axes(ax)

    for i, v in enumerate(spt.values):
        if v > 0:
            ax.text(v + 0.5, i, str(v), va="center", fontsize=7, color=TEXT_COLOR)

    plt.tight_layout()
    fig.savefig(
        out_dir / "specific_per_tissue.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    if save_pdf:
        fig.savefig(
            out_dir / "specific_per_tissue.pdf",
            bbox_inches="tight",
        )
    plt.close()
    logger.info("Saved specific_per_tissue.png")
