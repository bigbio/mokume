"""
Concrete feature-level (within-run) normalization methods.

Each class registers with the PluginRegistry and implements
``transform_series()`` — the per-run normalization math.
The run-balancing orchestration is handled by the base class.
"""

import pandas as pd

from mokume.core.registry import PluginRegistry
from mokume.normalization.base import FeatureNormalizer


@PluginRegistry.register("normalization.feature", "none")
class NoneFeatureNormalizer(FeatureNormalizer):
    """No-op normalizer — returns data unchanged."""

    @property
    def name(self) -> str:
        return "none"

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series

    def normalize(self, df, intensity_col=None, group_col=None, sample_col=None):
        return df


@PluginRegistry.register("normalization.feature", "mean")
class MeanFeatureNormalizer(FeatureNormalizer):
    """Mean normalization: intensity / mean(intensity)."""

    @property
    def name(self) -> str:
        return "mean"

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series / series.mean()


@PluginRegistry.register("normalization.feature", "median")
class MedianFeatureNormalizer(FeatureNormalizer):
    """Median normalization: intensity / median(intensity)."""

    @property
    def name(self) -> str:
        return "median"

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series / series.median()


@PluginRegistry.register("normalization.feature", "max")
class MaxFeatureNormalizer(FeatureNormalizer):
    """Max normalization: intensity / max(intensity)."""

    @property
    def name(self) -> str:
        return "max"

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series / series.max()


@PluginRegistry.register("normalization.feature", "global")
class GlobalFeatureNormalizer(FeatureNormalizer):
    """Global normalization: intensity / sum(intensity)."""

    @property
    def name(self) -> str:
        return "global"

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series / series.sum()


@PluginRegistry.register("normalization.feature", "max_min")
class MaxMinFeatureNormalizer(FeatureNormalizer):
    """Max-Min normalization: (intensity - min) / (max - min)."""

    @property
    def name(self) -> str:
        return "max_min"

    def transform_series(self, series: pd.Series) -> pd.Series:
        min_val = series.min()
        return (series - min_val) / (series.max() - min_val)


@PluginRegistry.register("normalization.feature", "iqr")
class IQRFeatureNormalizer(FeatureNormalizer):
    """IQR normalization: mean of 25th and 75th quantiles."""

    @property
    def name(self) -> str:
        return "iqr"

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series.quantile([0.75, 0.25], interpolation="linear").mean()
