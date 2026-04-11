"""
Simple imputation methods (mean, median, most_frequent, constant).

Wraps sklearn.impute.SimpleImputer and registers each strategy
with the PluginRegistry.
"""

import pandas as pd
from sklearn.impute import SimpleImputer

from mokume.core.registry import PluginRegistry
from mokume.imputation.base import ImputationMethod


class _SimpleImputation(ImputationMethod):
    """Shared base for sklearn SimpleImputer strategies."""

    _strategy: str = ""

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        imputer = SimpleImputer(strategy=self._strategy, fill_value=self.fill_value)
        imputed = imputer.fit_transform(df)
        return pd.DataFrame(imputed, columns=df.columns, index=df.index)


@PluginRegistry.register("imputation", "mean")
class MeanImputation(_SimpleImputation):
    """Impute missing values with the column mean.

    Each column's NaN values are replaced by the mean of the
    non-missing values in that column.
    """

    _strategy = "mean"

    @property
    def name(self) -> str:
        return "Mean"


@PluginRegistry.register("imputation", "median")
class MedianImputation(_SimpleImputation):
    """Impute missing values with the column median.

    Each column's NaN values are replaced by the median of the
    non-missing values in that column.
    """

    _strategy = "median"

    @property
    def name(self) -> str:
        return "Median"


@PluginRegistry.register("imputation", "most_frequent")
class MostFrequentImputation(_SimpleImputation):
    """Impute missing values with the most frequent value.

    Each column's NaN values are replaced by the most frequent value
    in that column.
    """

    _strategy = "most_frequent"

    @property
    def name(self) -> str:
        return "MostFrequent"


@PluginRegistry.register("imputation", "constant")
class ConstantImputation(_SimpleImputation):
    """Impute missing values with a constant.

    Parameters
    ----------
    fill_value : float
        The value to fill NaN entries with. Default 0.0.
    """

    _strategy = "constant"

    @property
    def name(self) -> str:
        return "Constant"

    def __repr__(self) -> str:
        return f"ConstantImputation(fill_value={self.fill_value})"
