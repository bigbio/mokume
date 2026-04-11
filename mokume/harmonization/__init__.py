"""
Batch effect correction (harmonization) for the mokume package.

This package consolidates all batch-correction-related code:
- Base class for batch correctors (plugin ABC)
- ComBat implementation (via inmoose)
- Configuration and enums
- Core correction functions
"""

from mokume.harmonization.base import BatchCorrector
from mokume.harmonization.combat import ComBatCorrector
from mokume.harmonization.models import BatchDetectionMethod, BatchCorrectionConfig
from mokume.harmonization.correction import (
    apply_batch_correction,
    compute_pca,
    detect_batches,
    extract_covariates_from_sdrf,
    get_batch_info_from_sample_names,
    is_batch_correction_available,
    is_inmoose_available,
    iterative_outlier_removal,
    remove_single_sample_batches,
    TooFewSamplesInBatch,
)

__all__ = [
    # Base
    "BatchCorrector",
    # Implementations
    "ComBatCorrector",
    # Config / enums
    "BatchDetectionMethod",
    "BatchCorrectionConfig",
    # Functions
    "apply_batch_correction",
    "compute_pca",
    "detect_batches",
    "extract_covariates_from_sdrf",
    "get_batch_info_from_sample_names",
    "is_batch_correction_available",
    "is_inmoose_available",
    "iterative_outlier_removal",
    "remove_single_sample_batches",
    "TooFewSamplesInBatch",
]
