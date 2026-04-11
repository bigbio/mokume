"""
Base classes for normalization methods.

This module provides abstract base classes for feature-level and
sample-level normalization methods. Implementations should register
with the PluginRegistry for automatic discovery.

Feature normalizers operate within a single run/sample to correct
for technical variation in intensity measurements.

Sample normalizers operate across samples to make them comparable.
Some (TMM, IRS, hierarchical) need the full dataset; others
(globalMedian, conditionMedian) adjust each sample independently.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from mokume.core.constants import NORM_INTENSITY, SAMPLE_ID, TECHREPLICATE


class FeatureNormalizer(ABC):
    """Base class for feature-level (within-run) normalization.

    Feature normalizers correct for technical variation within a single
    MS run. They operate on intensity values grouped by run or sample.

    Subclasses must implement ``transform_series()`` with the per-run
    normalization math. The concrete ``normalize()`` method handles the
    orchestration: iterating samples → runs, calling ``transform_series()``
    per run, computing per-run metrics, and balancing runs within a sample.

    Subclasses should register with::

        from mokume.core.registry import PluginRegistry

        @PluginRegistry.register("normalization.feature", "median")
        class MedianFeatureNormalizer(FeatureNormalizer):
            ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable normalizer name."""

    @abstractmethod
    def transform_series(self, series: pd.Series) -> pd.Series:
        """Apply normalization to a single intensity series (one run).

        Parameters
        ----------
        series : pd.Series
            Raw intensity values for a single run.

        Returns
        -------
        pd.Series
            Transformed metric (e.g., series / series.median()).
        """

    def normalize(
        self,
        df: pd.DataFrame,
        intensity_col: str = NORM_INTENSITY,
        group_col: str = TECHREPLICATE,
        sample_col: str = SAMPLE_ID,
    ) -> pd.DataFrame:
        """Normalize intensities across runs within each sample.

        Iterates over samples, applies ``transform_series()`` per run to
        compute a per-run metric, then scales each run so that its metric
        equals the sample-average metric.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-level DataFrame.
        intensity_col : str
            Column containing intensities to normalize.
        group_col : str
            Column defining run groups within a sample (e.g., TechReplicate).
        sample_col : str
            Column identifying samples.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalized intensities.
        """
        samples = df[sample_col].unique()
        for sample in samples:
            runs = df.loc[df[sample_col] == sample, group_col].unique().tolist()
            if len(runs) <= 1:
                continue

            sample_mask = df[sample_col] == sample
            sample_df = df.loc[sample_mask]

            # Compute per-run metric
            run_metrics = {}
            total_metric = 0
            for run in runs:
                run = str(run)
                run_series = sample_df.loc[
                    sample_df[group_col] == run, intensity_col
                ]
                metric = self.transform_series(run_series)
                run_metrics[run] = metric
                total_metric += metric

            sample_avg_metric = total_metric / len(runs)

            # Scale each run
            for run in runs:
                run = str(run)
                mask = (df[sample_col] == sample) & (df[group_col] == run)
                run_intensity = df.loc[mask, intensity_col]
                df.loc[mask, intensity_col] = run_intensity / (
                    run_metrics[run] / sample_avg_metric
                )

        return df

    def __call__(
        self,
        df: pd.DataFrame,
        technical_replicates: int,
        intensity_col: str = NORM_INTENSITY,
        group_col: str = TECHREPLICATE,
        sample_col: str = SAMPLE_ID,
    ) -> pd.DataFrame:
        """Callable interface matching the old enum signature."""
        if technical_replicates <= 1:
            return df
        return self.normalize(df, intensity_col, group_col, sample_col)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SampleNormalizer(ABC):
    """Base class for sample-level (across-sample) normalization.

    Sample normalizers make samples comparable by adjusting for
    systematic differences in total intensity, loading, etc.

    Some normalizers (TMM, IRS, hierarchical) need the full dataset
    to compute normalization factors. Others (globalMedian,
    conditionMedian) adjust each sample using pre-computed statistics.

    Subclasses should register with::

        from mokume.core.registry import PluginRegistry

        @PluginRegistry.register("normalization.sample", "tmm")
        class TMMSampleNormalizer(SampleNormalizer):
            ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable normalizer name."""

    @property
    def is_dataset_level(self) -> bool:
        """Whether this normalizer needs the full dataset at once.

        True for: TMM, IRS, hierarchical (need cross-sample statistics).
        False for: globalMedian, conditionMedian (per-sample adjustment).

        Returns
        -------
        bool
        """
        return False

    @abstractmethod
    def normalize(
        self,
        df: pd.DataFrame,
        intensity_col: str,
        sample_col: str,
        condition_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply normalization.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide/protein intensities.
        intensity_col : str
            Column containing intensities.
        sample_col : str
            Column identifying samples.
        condition_col : str, optional
            Column identifying conditions/groups.
        **kwargs
            Method-specific parameters.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalized intensities.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
