"""
Concrete sample-level (across-sample) normalization methods.

Per-sample normalizers (is_dataset_level=False):
    none, globalmedian, conditionmedian

Dataset-level normalizers (is_dataset_level=True):
    hierarchical, tmm, irs

Dataset-level normalizers are thin adapters that pivot long → wide,
delegate to the existing fit/transform classes, and melt back.
"""

from typing import Optional

import numpy as np
import pandas as pd

from mokume.core.registry import PluginRegistry
from mokume.core.constants import (
    CONDITION,
    NORM_INTENSITY,
    PROTEIN_NAME,
    PEPTIDE_CANONICAL,
    SAMPLE_ID,
)
from mokume.core.logger import get_logger
from mokume.normalization.base import SampleNormalizer

logger = get_logger("mokume.normalization.sample_normalizers")

# ---------------------------------------------------------------------------
# Per-sample normalizers
# ---------------------------------------------------------------------------


@PluginRegistry.register("normalization.sample", "none")
class NoneSampleNormalizer(SampleNormalizer):
    """No-op normalizer — returns data unchanged."""

    @property
    def name(self) -> str:
        return "none"

    def normalize(self, df, intensity_col=NORM_INTENSITY, sample_col=SAMPLE_ID,
                  condition_col=None, **kwargs):
        return df


@PluginRegistry.register("normalization.sample", "globalmedian")
class GlobalMedianNormalizer(SampleNormalizer):
    """Normalize each sample by its median relative to the global median.

    Expects ``med_map`` (dict: sample → median ratio) and ``sample`` (str)
    in kwargs.
    """

    @property
    def name(self) -> str:
        return "globalmedian"

    def normalize(self, df, intensity_col=NORM_INTENSITY, sample_col=SAMPLE_ID,
                  condition_col=None, **kwargs):
        med_map = kwargs.get("med_map", {})
        sample = kwargs.get("sample")
        if sample is not None and sample in med_map:
            df.loc[:, intensity_col] = df[intensity_col] / med_map[sample]
        return df


@PluginRegistry.register("normalization.sample", "conditionmedian")
class ConditionMedianNormalizer(SampleNormalizer):
    """Normalize each sample by its condition-specific median ratio.

    Expects ``med_map`` (dict: condition → {sample → ratio}) and
    ``sample`` (str) in kwargs. The condition is read from ``condition_col``.
    """

    @property
    def name(self) -> str:
        return "conditionmedian"

    def normalize(self, df, intensity_col=NORM_INTENSITY, sample_col=SAMPLE_ID,
                  condition_col=None, **kwargs):
        med_map = kwargs.get("med_map", {})
        sample = kwargs.get("sample")
        if condition_col is None:
            condition_col = CONDITION
        if sample is not None and med_map:
            con = df[condition_col].unique()[0]
            if con in med_map and sample in med_map[con]:
                df.loc[:, intensity_col] = df[intensity_col] / med_map[con][sample]
        return df


# ---------------------------------------------------------------------------
# Dataset-level normalizers
# ---------------------------------------------------------------------------


@PluginRegistry.register("normalization.sample", "hierarchical")
class HierarchicalSampleNormalizerPlugin(SampleNormalizer):
    """Adapter for HierarchicalSampleNormalizer (DirectLFQ-style).

    Operates on the full dataset: long → wide → fit_transform → long.

    Parameters
    ----------
    num_samples_quadratic : int
        Use quadratic optimization for datasets with fewer samples.
    selected_proteins : list[str], optional
        Proteins to use for computing normalization factors.
    """

    def __init__(
        self,
        num_samples_quadratic: int = 50,
        selected_proteins: Optional[list] = None,
    ):
        self.num_samples_quadratic = num_samples_quadratic
        self.selected_proteins = selected_proteins

    @property
    def name(self) -> str:
        return "hierarchical"

    @property
    def is_dataset_level(self) -> bool:
        return True

    def normalize(self, df, intensity_col=NORM_INTENSITY, sample_col=SAMPLE_ID,
                  condition_col=None, **kwargs):
        from mokume.normalization.hierarchical import HierarchicalSampleNormalizer

        protein_col = kwargs.get("protein_col", PROTEIN_NAME)
        peptide_col = kwargs.get("peptide_col", PEPTIDE_CANONICAL)

        logger.info("Applying hierarchical sample normalization...")

        # Long → wide
        wide = df.pivot_table(
            index=[protein_col, peptide_col],
            columns=sample_col,
            values=intensity_col,
            aggfunc="sum",
        )
        wide = wide.replace(0, np.nan)
        wide_log2 = np.log2(wide)

        normalizer = HierarchicalSampleNormalizer(
            num_samples_quadratic=self.num_samples_quadratic,
            selected_proteins=self.selected_proteins,
        )
        normalized_log2 = normalizer.fit_transform(wide_log2)

        # Back to linear scale
        normalized_wide = 2 ** normalized_log2

        # Wide → long
        result = normalized_wide.reset_index().melt(
            id_vars=[protein_col, peptide_col],
            var_name=sample_col,
            value_name=intensity_col,
        )
        result = result.dropna(subset=[intensity_col])

        logger.info(f"Hierarchical normalization complete: {len(result)} rows")
        return result


@PluginRegistry.register("normalization.sample", "tmm")
class TMMSampleNormalizerPlugin(SampleNormalizer):
    """Adapter for TMMNormalizer.

    Operates on the full dataset: long → wide → fit_transform → long.
    """

    @property
    def name(self) -> str:
        return "tmm"

    @property
    def is_dataset_level(self) -> bool:
        return True

    def normalize(self, df, intensity_col=NORM_INTENSITY, sample_col=SAMPLE_ID,
                  condition_col=None, **kwargs):
        from mokume.normalization.tmm import TMMNormalizer

        protein_col = kwargs.get("protein_col", PROTEIN_NAME)
        peptide_col = kwargs.get("peptide_col", PEPTIDE_CANONICAL)

        logger.info("Applying TMM sample normalization...")

        # Long → wide
        wide = df.pivot_table(
            index=[protein_col, peptide_col],
            columns=sample_col,
            values=intensity_col,
            aggfunc="sum",
        )

        normalizer = TMMNormalizer()
        normalized_wide = normalizer.fit_transform(wide)

        # Wide → long
        result = normalized_wide.reset_index().melt(
            id_vars=[protein_col, peptide_col],
            var_name=sample_col,
            value_name=intensity_col,
        )
        result = result.dropna(subset=[intensity_col])

        logger.info(f"TMM normalization complete: {len(result)} rows")
        return result


@PluginRegistry.register("normalization.sample", "irs")
class IRSSampleNormalizerPlugin(SampleNormalizer):
    """Adapter for IRSNormalizer (Internal Reference Scaling).

    Operates on the full dataset in wide format. Requires
    ``reference_samples`` and ``sample_to_plex`` in kwargs.
    """

    def __init__(self, reference_samples=None, stat="median"):
        self.reference_samples = reference_samples or []
        self.stat = stat

    @property
    def name(self) -> str:
        return "irs"

    @property
    def is_dataset_level(self) -> bool:
        return True

    def normalize(self, df, intensity_col=NORM_INTENSITY, sample_col=SAMPLE_ID,
                  condition_col=None, **kwargs):
        from mokume.normalization.irs import IRSNormalizer

        sample_to_plex = kwargs.get("sample_to_plex", {})
        reference_samples = kwargs.get("reference_samples", self.reference_samples)
        stat = kwargs.get("stat", self.stat)

        if not reference_samples:
            logger.warning("No reference samples provided for IRS, skipping")
            return df

        normalizer = IRSNormalizer(reference_samples=reference_samples, stat=stat)
        return normalizer.fit_transform(df, sample_to_plex)
