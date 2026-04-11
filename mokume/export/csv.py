"""
CSV / TSV export utilities for mokume data.

Provides functions for writing protein matrices and long-format data
to CSV/TSV files.
"""

import logging
from typing import TYPE_CHECKING, Optional

import pandas as pd

from mokume.core.constants import PROTEIN_NAME, SAMPLE_ID

if TYPE_CHECKING:
    from mokume.core.dataset import QpxDataset

logger = logging.getLogger(__name__)


def to_wide_csv(
    df: pd.DataFrame,
    output_path: str,
    protein_col: str = PROTEIN_NAME,
    sample_col: str = SAMPLE_ID,
    value_col: str = "Intensity",
    sep: str = ",",
) -> None:
    """Export a long-format DataFrame as a wide protein x sample CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with protein, sample, and intensity columns.
    output_path : str
        Output file path.
    protein_col : str
        Column for protein identifiers (becomes row index).
    sample_col : str
        Column for sample identifiers (becomes column headers).
    value_col : str
        Column containing values for matrix cells.
    sep : str
        Delimiter. Use '\\t' for TSV.
    """
    wide = df.pivot_table(
        index=protein_col,
        columns=sample_col,
        values=value_col,
        aggfunc="first",
    )
    wide.to_csv(output_path, sep=sep)
    logger.info(
        "Exported wide matrix (%d proteins x %d samples) to %s",
        wide.shape[0],
        wide.shape[1],
        output_path,
    )


def to_long_csv(
    df: pd.DataFrame,
    output_path: str,
    sep: str = ",",
    columns: Optional[list] = None,
) -> None:
    """Export a DataFrame to CSV in long format.

    Parameters
    ----------
    df : pd.DataFrame
        Data to export.
    output_path : str
        Output file path.
    sep : str
        Delimiter. Use '\\t' for TSV.
    columns : list, optional
        Subset of columns to export. If None, all columns are exported.
    """
    out = df[columns] if columns else df
    out.to_csv(output_path, sep=sep, index=False)
    logger.info("Exported %d rows to %s", len(out), output_path)


def dataset_to_csv(
    dataset: "QpxDataset",
    output_path: str,
    level: str = "proteins",
    wide: bool = True,
    value_col: str = "Intensity",
    sep: str = ",",
) -> None:
    """Export a QpxDataset level to CSV.

    Parameters
    ----------
    dataset : QpxDataset
        The dataset to export.
    output_path : str
        Output file path.
    level : str
        Data level to export.
    wide : bool
        If True, export as wide protein x sample matrix.
        If False, export as long format.
    value_col : str
        Column for values (used when wide=True).
    sep : str
        Delimiter.
    """
    df = dataset.get_level(level)
    if df is None:
        raise ValueError(f"Data level '{level}' is not populated")

    if wide:
        to_wide_csv(df, output_path, value_col=value_col, sep=sep)
    else:
        to_long_csv(df, output_path, sep=sep)
