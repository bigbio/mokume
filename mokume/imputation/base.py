"""
Base class for imputation methods.

This module provides an abstract base class for missing value imputation
algorithms. Implementations should register with the PluginRegistry.
"""

from abc import ABC, abstractmethod

import pandas as pd


class ImputationMethod(ABC):
    """Base class for missing value imputation methods.

    Imputation methods fill in missing (NaN) values in wide-format
    numeric matrices. They can be applied at any data level (features,
    peptides, proteins).

    Subclasses should register with::

        from mokume.core.registry import PluginRegistry

        @PluginRegistry.register("imputation", "knn")
        class KNNImputation(ImputationMethod):
            ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name."""

    @abstractmethod
    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Impute missing values in a wide-format matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Wide-format numeric matrix (observations x variables).
            Missing values are represented as NaN.
        **kwargs
            Method-specific parameters.

        Returns
        -------
        pd.DataFrame
            Matrix with missing values imputed, same shape as input.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
