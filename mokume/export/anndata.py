"""
AnnData export for mokume data.

Provides functions for converting DataFrames and QpxDatasets to AnnData
objects. AnnData is an export-only format — used for downstream analysis
with scanpy, scvi-tools, etc.

Requires: pip install mokume[anndata]
"""

from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from mokume.core.constants import PROTEIN_NAME, SAMPLE_ID

if TYPE_CHECKING:
    import anndata as ad

    from mokume.core.dataset import QpxDataset


def to_anndata(
    df: pd.DataFrame,
    obs_col: str = SAMPLE_ID,
    var_col: str = PROTEIN_NAME,
    value_col: str = "Intensity",
    layer_cols: Optional[List[str]] = None,
    obs_metadata_cols: Optional[List[str]] = None,
    var_metadata_cols: Optional[List[str]] = None,
) -> "ad.AnnData":
    """Create an AnnData object from a long-format DataFrame.

    This wraps the original ``mokume.io.parquet.create_anndata`` function
    with better default column names for the standard mokume workflow.

    Parameters
    ----------
    df : pd.DataFrame
        Input data in long format.
    obs_col : str
        Column for observations (samples). Default: SampleID.
    var_col : str
        Column for variables (proteins). Default: ProteinName.
    value_col : str
        Column for the main X matrix values.
    layer_cols : list[str], optional
        Additional columns to include as AnnData layers.
    obs_metadata_cols : list[str], optional
        Columns to include as observation metadata.
    var_metadata_cols : list[str], optional
        Columns to include as variable metadata.

    Returns
    -------
    anndata.AnnData
        AnnData object with X matrix, obs, var, and optional layers.
    """
    from mokume.io.parquet import create_anndata

    return create_anndata(
        df=df,
        obs_col=obs_col,
        var_col=var_col,
        value_col=value_col,
        layer_cols=layer_cols,
        obs_metadata_cols=obs_metadata_cols,
        var_metadata_cols=var_metadata_cols,
    )


def dataset_to_anndata(
    dataset: "QpxDataset",
    level: str = "proteins",
    value_col: str = "Intensity",
    layer_names: Optional[List[str]] = None,
) -> "ad.AnnData":
    """Export a QpxDataset level as AnnData.

    Convenience wrapper around ``QpxDataset.to_anndata()``.

    Parameters
    ----------
    dataset : QpxDataset
        The dataset to export.
    level : str
        Data level to export.
    value_col : str
        Column for the main X matrix values.
    layer_names : list[str], optional
        Names of dataset layers to include.

    Returns
    -------
    anndata.AnnData
    """
    return dataset.to_anndata(
        level=level,
        value_col=value_col,
        layer_names=layer_names,
    )
