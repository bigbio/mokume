"""
Interactive report generation for mokume DE results.

This module provides optional interactive HTML reports using plotly. Install
the matching distribution extra with ``pip install mokume-py[reports]`` (pure
Python) or ``pip install mokume[analysis]`` (Rust wheel).
"""

import importlib.util
from typing import TYPE_CHECKING

_INTERACTIVE_AVAILABLE = None


def is_interactive_available() -> bool:
    """
    Check if interactive report dependencies (plotly) are available.

    Returns
    -------
    bool
        True if plotly is installed, False otherwise.
    """
    global _INTERACTIVE_AVAILABLE
    if _INTERACTIVE_AVAILABLE is None:
        _INTERACTIVE_AVAILABLE = importlib.util.find_spec("plotly") is not None
    return _INTERACTIVE_AVAILABLE


def _require_interactive():
    """Raise an ImportError with helpful message if plotly is not available."""
    if not is_interactive_available():
        raise ImportError(
            "Interactive report dependencies (plotly) are not installed. "
            "Install the matching extra with: pip install mokume-py[reports] "
            "(pure Python) or pip install mokume[analysis] (Rust wheel)"
        )


def __getattr__(name):
    if name == "generate_de_report":
        _require_interactive()
        from mokume.reports.interactive import generate_de_report

        return generate_de_report

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from mokume.reports.interactive import generate_de_report


__all__ = [
    "is_interactive_available",
    "generate_de_report",
]
