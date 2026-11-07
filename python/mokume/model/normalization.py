"""
Normalization method enumerations for the mokume package.

This module provides enumerations for feature-level and peptide-level
normalization methods, with registration of normalization functions.
"""

from typing import Callable
from enum import Enum, auto

import pandas as pd

from mokume.core.constants import CONDITION, NORM_INTENSITY, SAMPLE_ID, TECHREPLICATE

_method_registry: dict[
    "FeatureNormalizationMethod", Callable[[pd.Series], pd.Series]
] = {}


class FeatureNormalizationMethod(Enum):
    """
    Enumeration of feature-level normalization methods.

    This enum provides functionality to register custom normalization functions
    and apply them to replicate data. Supports normalization across multiple
    runs and samples.

    Attributes
    ----------
    NONE : auto
        No normalization.
    Mean : auto
        Mean normalization.
    Median : auto
        Median normalization.
    Max : auto
        Max normalization.
    Global : auto
        Global normalization.
    Max_Min : auto
        Max-Min normalization.
    IQR : auto
        Inter-quartile range normalization.
    """

    NONE = auto()

    Mean = auto()
    Median = auto()
    Max = auto()
    Global = auto()
    Max_Min = auto()
    IQR = auto()

    @classmethod
    def from_str(cls, name: str) -> "FeatureNormalizationMethod":
        """
        Get the normalization method from a string.

        Parameters
        ----------
        name : str
            The name of the normalization method.

        Returns
        -------
        FeatureNormalizationMethod
            The normalization method.

        Raises
        ------
        KeyError
            If the name does not match any normalization method.
        """
        if name is None:
            return cls.NONE
        name_ = name.lower()
        for k, v in cls._member_map_.items():
            if k.lower() == name_:
                return v
        raise KeyError(name)

    def register_replicate_fn(
        self, fn: Callable[[pd.Series], pd.Series]
    ) -> Callable[[pd.Series], pd.Series]:
        """
        Register a custom normalization function for replicate intensities.

        Parameters
        ----------
        fn : Callable[[pd.Series], pd.Series]
            A function that takes a pandas Series and returns a normalized pandas Series.

        Returns
        -------
        Callable[[pd.Series], pd.Series]
            The registered normalization function.
        """
        _method_registry[self] = fn
        return fn

    def normalize_replicates(self, df: pd.DataFrame, *args, **kwargs):
        """
        Normalize the replicate intensities using a registered normalization function.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing replicate intensity data.
        *args
            Additional positional arguments for the normalization function.
        **kwargs
            Additional keyword arguments for the normalization function.

        Returns
        -------
        pd.Series
            The normalized replicate intensities.
        """
        fn = _method_registry[self]
        return fn(df, *args, **kwargs)

    def normalize_sample(
        self, df, runs: list[str]
    ) -> tuple[dict[str, pd.Series], float]:
        """
        Normalize replicate intensities for a given sample across multiple runs.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing replicate intensity data.
        runs : list[str]
            A list of run identifiers for the sample.

        Returns
        -------
        tuple[dict[str, pd.Series], float]
            A dictionary mapping each run to its normalized replicate intensities
            and the average metric across all runs.
        """
        map_ = {}
        total = 0
        for run in runs:
            values = df.loc[df[TECHREPLICATE] == run, NORM_INTENSITY]
            run_m = self._replicate_metric(values)
            map_[run] = run_m
            total += run_m
        sample_average_metric = total / len(runs)
        return map_, sample_average_metric

    def _replicate_metric(self, values: pd.Series) -> float:
        """Return one run's scale factor for cross-replicate normalization.

        ``normalize_runs`` divides each run's intensities by this value relative
        to the sample average, so it has to be a single positive number. The
        registered replicate functions return a rescaled *Series* instead, which
        silently turned ``total`` into NaN via index misalignment and blanked the
        whole sample.
        """
        if self == FeatureNormalizationMethod.Mean:
            return values.mean()
        if self == FeatureNormalizationMethod.Median:
            return values.median()
        if self == FeatureNormalizationMethod.Max:
            return values.max()
        if self == FeatureNormalizationMethod.Global:
            return values.sum()
        if self == FeatureNormalizationMethod.IQR:
            return self.normalize_replicates(values)
        raise NotImplementedError(
            f"{self.name} rescales each run's values and has no single scale "
            "factor, so it cannot normalize across technical replicates. Use "
            "mean, median, max, global or iqr for --run-normalization when a "
            "sample has more than one run."
        )

    def normalize_runs(self, df: pd.DataFrame, technical_replicates: int):
        """
        Normalize the intensities of runs in the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing replicate intensity data.
        technical_replicates : int
            The number of technical replicates for each sample.

        Returns
        -------
        pd.DataFrame
            The DataFrame with normalized replicate intensities.
        """
        if technical_replicates > 1:
            samples = df[SAMPLE_ID].unique()
            for sample in samples:
                runs = df.loc[df[SAMPLE_ID] == sample, TECHREPLICATE].unique().tolist()
                if len(runs) > 1:
                    sample_df = df.loc[df[SAMPLE_ID] == sample, :]

                    replicate_metric_map, sample_average_metric = self.normalize_sample(
                        sample_df, runs
                    )

                    # For each replicate in each sample, normalize the per-replicate
                    # intensity by a replicate-level statistic, relative to the sample
                    # average over that replicate statistic.
                    for run in runs:
                        run_intensity = df.loc[
                            (df[SAMPLE_ID] == sample) & (df[TECHREPLICATE] == run),
                            NORM_INTENSITY,
                        ]
                        df.loc[
                            (df[SAMPLE_ID] == sample) & (df[TECHREPLICATE] == run),
                            NORM_INTENSITY,
                        ] = run_intensity / (
                            replicate_metric_map[run] / sample_average_metric
                        )
            return df
        else:
            return df

    def __call__(self, df: pd.DataFrame, technical_replicates: int):
        return self.normalize_runs(df, technical_replicates)


@FeatureNormalizationMethod.NONE.register_replicate_fn
def no_normalization(df, *args, **kwargs):
    """No normalization is performed on the data."""
    return df


@FeatureNormalizationMethod.Mean.register_replicate_fn
def mean_normalize(df, *args, **kwargs):
    """Mean normalization of the data."""
    return df / df.mean()


@FeatureNormalizationMethod.Median.register_replicate_fn
def median_normalize(df, *args, **kwargs):
    """Median normalization of the data."""
    return df / df.median()


@FeatureNormalizationMethod.Max.register_replicate_fn
def max_normalize(df, *args, **kwargs):
    """Max normalization of the data."""
    return df / df.max()


@FeatureNormalizationMethod.Global.register_replicate_fn
def global_normalize(df, *args, **kwargs):
    """Global normalization of the data."""
    return df / df.sum()


@FeatureNormalizationMethod.Max_Min.register_replicate_fn
def max_min_normalize(df, *args, **kwargs):
    """Max-Min normalization of the data."""
    min_ = df.min()
    return (df - min_) / (df.max() - min_)


@FeatureNormalizationMethod.IQR.register_replicate_fn
def iqr_normalization(df, *args, **kwargs):
    """IQR normalization of the data."""
    return df.quantile([0.75, 0.25], interpolation="linear").mean()


_peptide_method_registry = {}


class PeptideNormalizationMethod(Enum):
    """
    Enumeration for peptide/sample normalization methods.

    Also known as SampleNormalizationMethod (preferred name going forward).

    Provides functionality to register and apply normalization functions
    to peptide data across samples.

    Attributes
    ----------
    NONE : auto
        No normalization.
    GlobalMedian : auto
        Normalization using global median (sample median / global median).
    ConditionMedian : auto
        Normalization using condition-specific median.
    Hierarchical : auto
        DirectLFQ-style hierarchical clustering-based normalization.
        Uses HierarchicalSampleNormalizer from mokume.normalization.hierarchical.
    """

    NONE = auto()

    GlobalMedian = auto()
    ConditionMedian = auto()
    Hierarchical = auto()  # DirectLFQ-style, native mokume implementation
    Quantile = auto()  # Quantile normalization
    MedianCenter = auto()  # Sample-level median centering
    MeanCenter = auto()  # Sample-level mean centering
    Rlr = auto()  # Robust Linear Regression (NormalyzerDE)
    Loess = auto()  # LOESS regression on MA plot (Yang 2002)
    TMM = auto()  # Trimmed Mean of M-values (Robinson & Oshlack 2010)

    @classmethod
    def from_str(cls, name: str) -> "PeptideNormalizationMethod":
        """
        Convert a string to a PeptideNormalizationMethod.

        Parameters
        ----------
        name : str
            The name of the normalization method.

        Returns
        -------
        PeptideNormalizationMethod
            The normalization method.

        Raises
        ------
        KeyError
            If the name does not match any normalization method.
        """
        name_ = name.lower()
        for k, v in cls._member_map_.items():
            if k.lower() == name_:
                return v
        raise KeyError(name)

    def register_replicate_fn(
        self, fn: Callable[[pd.DataFrame, str, dict], pd.DataFrame]
    ) -> Callable[[pd.DataFrame, str, dict], pd.DataFrame]:
        """
        Register a function for a specific normalization method.

        Parameters
        ----------
        fn : Callable[[pd.DataFrame, str, dict], pd.DataFrame]
            The normalization function.

        Returns
        -------
        Callable[[pd.DataFrame, str, dict], pd.DataFrame]
            The normalization function.
        """
        _peptide_method_registry[self] = fn
        return fn

    def normalize_sample(self, dataset_df: pd.DataFrame, sample: str, med_map: dict):
        """
        Apply the registered normalization function to a sample.

        Parameters
        ----------
        dataset_df : pd.DataFrame
            The DataFrame containing peptide intensity data.
        sample : str
            The sample identifier.
        med_map : dict
            The median map.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the normalized peptide intensity data.
        """
        fn = _peptide_method_registry[self]
        return fn(dataset_df, sample, med_map)

    @property
    def is_dataset_level(self) -> bool:
        """Whether this method operates on the full dataset rather than per-sample.

        Hierarchical normalization requires the complete dataset and is
        applied after all samples are loaded, unlike GlobalMedian/ConditionMedian
        which operate per-sample during loading.
        """
        return self in (
            PeptideNormalizationMethod.Hierarchical,
            PeptideNormalizationMethod.Quantile,
            PeptideNormalizationMethod.MedianCenter,
            PeptideNormalizationMethod.MeanCenter,
            PeptideNormalizationMethod.Rlr,
            PeptideNormalizationMethod.Loess,
            PeptideNormalizationMethod.TMM,
        )

    def __call__(self, dataset_df: pd.DataFrame, sample: str, med_map: dict):
        """Invoke the normalize_sample method."""
        return self.normalize_sample(dataset_df, sample, med_map)


# Global-median follows each sample's detection depth and match-between-runs
# transfers, so these aggregations align shared peptide species instead (as the
# Rust CLI does).
HIERARCHICAL_DEFAULT_METHODS = frozenset({"maxlfq", "sum", "median"})


def default_normalization_methods(quant_method: str) -> tuple[str, str]:
    """Default run and sample normalization names of a quantification method.

    DirectLFQ and ratio normalize internally and peptide_count uses no
    intensities, so both default to ``none``. MaxLFQ, ``top<N>``, sum and
    median default to hierarchical sample normalization, the other intensity
    methods to global-median, all after median run normalization.
    """
    method = quant_method.lower()
    if method in {"directlfq", "ratio", "peptide_count"}:
        return "none", "none"
    if method in HIERARCHICAL_DEFAULT_METHODS or method.startswith("top"):
        return "median", "hierarchical"
    return "median", "globalmedian"


def parse_normalization_methods(
    run_method: str,
    sample_method: str,
) -> tuple[FeatureNormalizationMethod, PeptideNormalizationMethod]:
    """Parse run and sample normalization names with user-facing errors."""
    try:
        run = FeatureNormalizationMethod.from_str(run_method)
    except (AttributeError, KeyError):
        raise ValueError(f"Unknown run normalization method: {run_method!r}") from None
    try:
        sample = PeptideNormalizationMethod.from_str(sample_method)
    except (AttributeError, KeyError):
        raise ValueError(
            f"Unknown sample normalization method: {sample_method!r}"
        ) from None
    return run, sample


@PeptideNormalizationMethod.GlobalMedian.register_replicate_fn
def global_median(dataset_df, sample: str, med_map: dict):
    """Global median normalization of the data."""
    dataset_df.loc[:, NORM_INTENSITY] = dataset_df[NORM_INTENSITY] / med_map[sample]
    return dataset_df


@PeptideNormalizationMethod.ConditionMedian.register_replicate_fn
def condition_median(dataset_df, sample: str, med_map: dict):
    """Condition median normalization of the data."""
    con = dataset_df[CONDITION].unique()[0]
    dataset_df.loc[:, NORM_INTENSITY] = (
        dataset_df[NORM_INTENSITY] / med_map[con][sample]
    )
    return dataset_df


@PeptideNormalizationMethod.NONE.register_replicate_fn
def peptide_no_normalization(dataset_df, sample, med_map):
    """No normalization is performed on the data."""
    return dataset_df


@PeptideNormalizationMethod.Hierarchical.register_replicate_fn
def hierarchical_normalization(dataset_df, sample, med_map):
    """Placeholder: Hierarchical normalization is applied at dataset level."""
    return dataset_df


@PeptideNormalizationMethod.Quantile.register_replicate_fn
def quantile_normalization(dataset_df, sample, med_map):
    """Placeholder: quantile normalization is applied at dataset level."""
    return dataset_df


@PeptideNormalizationMethod.MedianCenter.register_replicate_fn
def median_center_normalization(dataset_df, sample, med_map):
    """Placeholder: sample-level median centering applied at dataset level."""
    return dataset_df


@PeptideNormalizationMethod.MeanCenter.register_replicate_fn
def mean_center_normalization(dataset_df, sample, med_map):
    """Placeholder: sample-level mean centering applied at dataset level."""
    return dataset_df


@PeptideNormalizationMethod.Rlr.register_replicate_fn
def rlr_normalization(dataset_df, sample, med_map):
    """Placeholder: RLR normalization is applied at dataset level."""
    return dataset_df


@PeptideNormalizationMethod.Loess.register_replicate_fn
def loess_normalization(dataset_df, sample, med_map):
    """Placeholder: LOESS normalization is applied at dataset level."""
    return dataset_df


@PeptideNormalizationMethod.TMM.register_replicate_fn
def tmm_normalization(dataset_df, sample, med_map):
    """Placeholder: TMM normalization is applied at dataset level."""
    return dataset_df


# ============================================================================
# Backward-compatible aliases (preferred names going forward)
# ============================================================================

# RunNormalizationMethod is the preferred name for FeatureNormalizationMethod
# (normalizes technical replicates/runs within each sample)
RunNormalizationMethod = FeatureNormalizationMethod

# SampleNormalizationMethod is the preferred name for PeptideNormalizationMethod
# (normalizes samples relative to each other)
SampleNormalizationMethod = PeptideNormalizationMethod
