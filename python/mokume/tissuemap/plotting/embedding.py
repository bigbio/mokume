"""Embedding diagnostic plots: PCA scree and cumulative variance."""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from mokume.plotting._common import panel_label, static_style, style_axes
from mokume.plotting.style import (
    SAMPLE_POINT_COLOR,
    SINGLE_SERIES_COLOR,
    TEXT_COLOR,
)

logger = logging.getLogger(__name__)


def _plot_scree_panel(ax, var_ratio: np.ndarray, n_pcs: int) -> None:
    """Panel A: scree bar chart."""
    ax.bar(
        range(1, n_pcs + 1),
        var_ratio * 100,
        color=SINGLE_SERIES_COLOR,
        width=0.7,
        linewidth=0,
    )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree plot", loc="left")
    ax.set_xlim(0.5, n_pcs + 0.5)
    ax.margins(y=0.15)
    style_axes(ax)
    # One block in the empty upper right; per-bar labels collide on narrow bars.
    ax.text(
        0.97,
        0.95,
        "\n".join(
            f"PC{i + 1}  {var_ratio[i] * 100:.1f}%" for i in range(min(3, n_pcs))
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=TEXT_COLOR,
    )


def _plot_cumulative_panel(ax, cumvar: np.ndarray, n_pcs: int) -> None:
    """Panel B: cumulative variance curve with neutral threshold annotations."""
    ax.plot(
        range(1, n_pcs + 1),
        cumvar * 100,
        "o-",
        color=SINGLE_SERIES_COLOR,
        markersize=4,
        markeredgewidth=0,
        linewidth=2,
    )
    for threshold in (50, 70):
        ax.axhline(threshold, color=SAMPLE_POINT_COLOR, linestyle="--", linewidth=1)
        # Label each threshold at the end the curve stays farther from.
        at_start = abs(cumvar[0] * 100 - threshold) > abs(cumvar[-1] * 100 - threshold)
        ax.text(
            1 if at_start else n_pcs,
            threshold + 1.5,
            f"{threshold}%",
            ha="left" if at_start else "right",
            color=TEXT_COLOR,
        )
    # The rising curve leaves the lower right empty; callouts there crossed the thresholds.
    ax.text(
        0.97,
        0.05,
        "\n".join(
            f"PC1-{n_pc}  {cumvar[n_pc - 1] * 100:.1f}%"
            for n_pc in dict.fromkeys([2, 10, 20, n_pcs])
            if n_pc <= n_pcs
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=TEXT_COLOR,
    )
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Cumulative variance", loc="left")
    ax.set_xlim(0.5, n_pcs + 0.5)
    ax.set_ylim(0, min(max(cumvar[-1] * 100 + 15, 78), 105))
    style_axes(ax)


@static_style()
def plot_pca_scree(
    adata: ad.AnnData,
    out_dir: Path,
    *,
    dpi: int = 300,
    save_pdf: bool = True,
) -> None:
    """Plot PCA variance without changing the stored component statistics.

    Requires ``adata.uns["pca_variance_ratio"]``, set by the embedding stage.
    """
    var_ratio = adata.uns.get("pca_variance_ratio")
    if var_ratio is None:
        logger.warning("No pca_variance_ratio in adata.uns, skipping scree plot")
        return

    var_ratio = np.asarray(var_ratio, dtype=float)
    n_pcs = len(var_ratio)
    cumvar = np.cumsum(var_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    _plot_scree_panel(axes[0], var_ratio, n_pcs)
    _plot_cumulative_panel(axes[1], cumvar, n_pcs)
    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scree.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_dir / "pca_scree.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved pca_scree.png")
