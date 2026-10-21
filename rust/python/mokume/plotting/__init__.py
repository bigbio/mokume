"""
Plotting utilities for the mokume package.

This module provides optional plotting functionality for QC reports and
visualizations. Install the matching distribution extra with
``pip install mokume-py[plotting]`` (pure Python) or
``pip install mokume[analysis]`` (Rust wheel).
"""

import importlib.util
from typing import TYPE_CHECKING

_PLOTTING_AVAILABLE = None


def is_plotting_available() -> bool:
    """
    Check if plotting dependencies (matplotlib, seaborn) are available.

    Returns
    -------
    bool
        True if matplotlib and seaborn are installed, False otherwise.
    """
    global _PLOTTING_AVAILABLE
    if _PLOTTING_AVAILABLE is None:
        _PLOTTING_AVAILABLE = (
            importlib.util.find_spec("matplotlib") is not None
            and importlib.util.find_spec("seaborn") is not None
        )
    return _PLOTTING_AVAILABLE


def _require_plotting():
    """Raise an ImportError with helpful message if plotting is not available."""
    if not is_plotting_available():
        raise ImportError(
            "Plotting dependencies (matplotlib, seaborn) are not installed. "
            "Install the matching extra with: pip install mokume-py[plotting] "
            "(pure Python) or pip install mokume[analysis] (Rust wheel)"
        )


# Lazy imports - only import when actually used
def __getattr__(name):
    if name in (
        "plot_distributions",
        "plot_box_plot",
    ):
        _require_plotting()
        from mokume.plotting.distributions import plot_distributions, plot_box_plot

        return {
            "plot_distributions": plot_distributions,
            "plot_box_plot": plot_box_plot,
        }[name]

    if name in (
        "plot_pca",
        "compute_pca_with_plot",
        "plot_tsne",
    ):
        _require_plotting()
        from mokume.plotting.pca import plot_pca, compute_pca_with_plot, plot_tsne

        return {
            "plot_pca": plot_pca,
            "compute_pca_with_plot": compute_pca_with_plot,
            "plot_tsne": plot_tsne,
        }[name]

    if name in (
        "plot_volcano",
        "plot_heatmap",
        "plot_pca_conditions",
    ):
        _require_plotting()
        from mokume.plotting.differential_expression import (
            plot_volcano,
            plot_heatmap,
            plot_pca_conditions,
        )

        return {
            "plot_volcano": plot_volcano,
            "plot_heatmap": plot_heatmap,
            "plot_pca_conditions": plot_pca_conditions,
        }[name]

    if name == "PdfPages":
        _require_plotting()
        from matplotlib.backends.backend_pdf import PdfPages

        return PdfPages

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For type checking and IDE support
if TYPE_CHECKING:
    from mokume.plotting.distributions import plot_distributions, plot_box_plot
    from mokume.plotting.pca import plot_pca, compute_pca_with_plot, plot_tsne
    from mokume.plotting.differential_expression import (
        plot_volcano,
        plot_heatmap,
        plot_pca_conditions,
    )
    from matplotlib.backends.backend_pdf import PdfPages


__all__ = [
    "is_plotting_available",
    "plot_distributions",
    "plot_box_plot",
    "plot_pca",
    "compute_pca_with_plot",
    "plot_tsne",
    "plot_volcano",
    "plot_heatmap",
    "plot_pca_conditions",
    "PdfPages",
]
