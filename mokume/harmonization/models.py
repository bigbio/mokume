"""
Batch correction configuration and enums for the mokume package.

This module provides configuration classes and enums for batch effect
correction using ComBat (via inmoose).

Key Concepts:
- Batch: Technical variation to REMOVE (e.g., different runs, labs, processing days)
- Covariates: Biological variables to PRESERVE (e.g., sex, tissue from SDRF characteristics)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class BatchDetectionMethod(Enum):
    """
    Methods for detecting/assigning batch labels.

    Attributes
    ----------
    SAMPLE_PREFIX : str
        Extract batch from sample name prefix (e.g., PXD001-S1 → batch=PXD001).
    RUN_NAME : str
        Use run/reference file name as batch identifier.
    FRACTION : str
        Treat each fraction as a separate batch.
    TECHREPLICATE : str
        Treat each technical replicate as a separate batch.
    EXPLICIT_COLUMN : str
        Use values from a user-specified column.
    """

    SAMPLE_PREFIX = "sample_prefix"
    RUN_NAME = "run"
    FRACTION = "fraction"
    TECHREPLICATE = "techreplicate"
    EXPLICIT_COLUMN = "column"

    @classmethod
    def from_str(cls, name: str) -> "BatchDetectionMethod":
        """
        Convert a string to a BatchDetectionMethod.

        Parameters
        ----------
        name : str
            The name of the batch detection method.

        Returns
        -------
        BatchDetectionMethod
            The batch detection method enum value.

        Raises
        ------
        ValueError
            If the name does not match any method.
        """
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == name_lower:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown batch detection method: {name}. Valid options: {valid}")


@dataclass
class BatchCorrectionConfig:
    """
    Configuration for batch effect correction.

    This configuration controls how batch effects are detected and corrected
    using the ComBat algorithm (via inmoose).

    Attributes
    ----------
    enabled : bool
        Whether to apply batch correction. Default False.
    batch_method : BatchDetectionMethod
        How to detect/assign batch labels. Default SAMPLE_PREFIX.
    batch_column : str, optional
        Column name for explicit batch assignment (when batch_method=EXPLICIT_COLUMN).
    covariate_columns : List[str]
        SDRF columns to use as covariates (biological signal to preserve).
        Example: ["characteristics[sex]", "characteristics[organism part]"]
    parametric : bool
        Use parametric empirical Bayes estimation. Default True.
        Set False for non-parametric estimation.
    mean_only : bool
        Only adjust batch means, not individual effects. Default False.
    ref_batch : int, optional
        Batch ID to use as reference (all other batches adjusted to this one).

    Examples
    --------
    >>> config = BatchCorrectionConfig(
    ...     enabled=True,
    ...     batch_method=BatchDetectionMethod.SAMPLE_PREFIX,
    ...     covariate_columns=["characteristics[sex]", "characteristics[tissue]"],
    ... )
    """

    enabled: bool = False

    # Batch detection
    batch_method: BatchDetectionMethod = BatchDetectionMethod.SAMPLE_PREFIX
    batch_column: Optional[str] = None

    # Covariates from SDRF (biological signal to preserve)
    covariate_columns: List[str] = field(default_factory=list)

    # ComBat parameters
    parametric: bool = True
    mean_only: bool = False
    ref_batch: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.batch_method, str):
            self.batch_method = BatchDetectionMethod.from_str(self.batch_method)

        if self.batch_method == BatchDetectionMethod.EXPLICIT_COLUMN and not self.batch_column:
            raise ValueError(
                "batch_column must be specified when batch_method is EXPLICIT_COLUMN"
            )
