"""
Base class for batch correction methods.

This module provides an abstract base class for batch effect correction
algorithms. Implementations should register with the PluginRegistry.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class BatchCorrector(ABC):
    """Base class for batch effect correction methods.

    Batch correctors remove systematic technical variation (batch effects)
    while preserving biological signal. They operate on protein-level
    wide-format data (proteins x samples).

    Subclasses should register with::

        from mokume.core.registry import PluginRegistry

        @PluginRegistry.register("harmonization", "combat")
        class ComBatCorrector(BatchCorrector):
            ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name."""

    @abstractmethod
    def correct(
        self,
        df: pd.DataFrame,
        batch: List[int],
        covariates: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply batch correction to a protein intensity matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Wide-format protein intensity matrix (proteins x samples).
            Index is protein identifiers, columns are sample identifiers.
        batch : list[int]
            Batch assignment for each sample (column).
        covariates : list[list[int]], optional
            Biological covariates to preserve. Each inner list is a
            covariate with one value per sample.
        **kwargs
            Method-specific parameters.

        Returns
        -------
        pd.DataFrame
            Batch-corrected protein intensity matrix, same shape as input.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
