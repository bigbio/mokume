"""
Schema definitions for the qpx data format.

This module centralizes column name constants and schema validation
for the qpx parquet format used throughout mokume. It is the single
source of truth for column names at each data level.

Column constants are re-exported here from constants.py for backward
compatibility. New code should import from this module.

Data levels
-----------
- features: Raw feature-level data from qpx parquet
- peptides: Assembled peptide-level data after normalization
- proteins: Protein-level quantification results
- psms: PSM-level data (used by ratio quantification)
"""

from typing import Dict, FrozenSet, List

import pandas as pd

# Re-export column constants from the canonical location
from mokume.core.constants import (
    BIOREPLICATE,
    CHANNEL,
    CONDITION,
    FRACTION,
    INTENSITY,
    NORM_INTENSITY,
    PARQUET_COLUMNS,
    PEPTIDE_CANONICAL,
    PEPTIDE_CHARGE,
    PEPTIDE_SEQUENCE,
    PROTEIN_NAME,
    REFERENCE,
    RUN,
    SAMPLE_ID,
    TECHREPLICATE,
    parquet_map,
)


# --- Schema definitions per data level ---

FEATURE_REQUIRED_COLS: FrozenSet[str] = frozenset({
    PROTEIN_NAME,
    PEPTIDE_SEQUENCE,
    SAMPLE_ID,
    INTENSITY,
})

PEPTIDE_REQUIRED_COLS: FrozenSet[str] = frozenset({
    PROTEIN_NAME,
    PEPTIDE_CANONICAL,
    SAMPLE_ID,
    NORM_INTENSITY,
})

PROTEIN_REQUIRED_COLS: FrozenSet[str] = frozenset({
    PROTEIN_NAME,
    SAMPLE_ID,
})

PSM_REQUIRED_COLS: FrozenSet[str] = frozenset({
    PROTEIN_NAME,
    PEPTIDE_SEQUENCE,
    PEPTIDE_CHARGE,
    SAMPLE_ID,
    INTENSITY,
})

_LEVEL_SCHEMAS: Dict[str, FrozenSet[str]] = {
    "features": FEATURE_REQUIRED_COLS,
    "peptides": PEPTIDE_REQUIRED_COLS,
    "proteins": PROTEIN_REQUIRED_COLS,
    "psms": PSM_REQUIRED_COLS,
}


def validate_schema(df: pd.DataFrame, level: str) -> List[str]:
    """Validate that a DataFrame has the required columns for a data level.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    level : str
        Data level: one of "features", "peptides", "proteins", "psms".

    Returns
    -------
    list[str]
        List of error messages. Empty if the DataFrame is valid.

    Examples
    --------
    >>> errors = validate_schema(my_df, "peptides")
    >>> if errors:
    ...     raise ValueError(f"Schema errors: {errors}")
    """
    required = _LEVEL_SCHEMAS.get(level)
    if required is None:
        return [f"Unknown data level: '{level}'. Available: {list(_LEVEL_SCHEMAS.keys())}"]

    missing = required - set(df.columns)
    return [
        f"Missing required column '{col}' for {level} level"
        for col in sorted(missing)
    ]


def available_levels() -> List[str]:
    """Return the list of available data levels.

    Returns
    -------
    list[str]
    """
    return list(_LEVEL_SCHEMAS.keys())
