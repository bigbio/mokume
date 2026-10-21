"""Canonical method contract for agentic configuration generation."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

# Acquisition types that knowledge records apply to; LFQ means label-free DDA.
DATA_TYPES: tuple[str, ...] = ("LFQ", "DIA", "TMT")
DE_METHODS: tuple[str, ...] = (
    "limrots",
    "limma",
    "deqms",
    "proda",
    "rots",
    "ensemble",
)
FDR_METHODS: tuple[str, ...] = ("bh", "ihw", "bky", "storey")
NORMALIZATION_METHODS: tuple[str, ...] = (
    "none",
    "median",
    "quantile",
    "mean",
    "rlr",
    "loess",
)
IMPUTATION_METHODS: tuple[str, ...] = (
    "none",
    "minprob",
    "mindet",
    "knn",
    "missforest",
    "seqknn",
    "qrilc",
    "impseq",
    "impseqrob",
    "bpca",
    "gms",
)
QUANTIFICATION_METHODS: tuple[str, ...] = (
    "directlfq",
    "pibaq",
    "maxlfq",
    "sum",
    "median",
    "ratio",
    "abd",
    "intensity",
    "peptide_count",
    "spectral_count",
    "top3",
)
ENSEMBLE_PRESETS: tuple[str, ...] = (
    "none",
    "limma,deqms,proda",
    "limma,rots,deqms",
    "limma,rots,deqms,proda",
)
CONFIDENCE_LEVELS: tuple[str, ...] = ("low", "moderate", "high")
GENERATED_CONFIG_FIELDS: tuple[str, ...] = (
    "name",
    "de_method",
    "fdr_method",
    "normalization",
    "imputation",
    "ensemble",
    "ensemble_k",
    "log2fc_threshold",
    "reasoning",
    "expected_outcome",
)
GENERATED_BLOCK_FIELDS: tuple[str, ...] = (
    "configs",
    "evidence_refs",
    "confidence",
    "limitations",
    "abstain_reason",
)
EXECUTABLE_AXES: tuple[str, ...] = (
    "normalization",
    "imputation",
    "de_method",
    "fdr_method",
    "log2fc_threshold",
    "ensemble",
    "ensemble_k",
)
FROZEN_AXES: tuple[str, ...] = (
    "quantification",
    "run_normalization",
    "sample_normalization",
    "peptide_filters",
    "protein_filters",
    "batch_correction",
    "irs",
)


def method_contract() -> dict[str, Any]:
    """Return the runtime method contract exposed to policy and the LLM."""
    return {
        "de_method": list(DE_METHODS),
        "fdr_method": list(FDR_METHODS),
        "fdr_method_by_de_method": {
            method: ["bh"] if method in {"rots", "limrots"} else list(FDR_METHODS)
            for method in DE_METHODS
        },
        "normalization": list(NORMALIZATION_METHODS),
        "imputation": list(IMPUTATION_METHODS),
        "quantification": [*QUANTIFICATION_METHODS, "top<N>"],
        "ensemble": list(ENSEMBLE_PRESETS),
        "log2fc_threshold": {"number": [0.0, 10.0], "sentinels": ["auto"]},
        "ensemble_k": {"ensemble": [1, 5], "non_ensemble": None},
        "generated_config_fields": list(GENERATED_CONFIG_FIELDS),
        "generated_block_fields": list(GENERATED_BLOCK_FIELDS),
        "executable_axes": list(EXECUTABLE_AXES),
        "frozen_axes": list(FROZEN_AXES),
    }


def is_supported_quantification(value: str) -> bool:
    """Return whether a canonical quantification name is understood by Mokume."""
    return value in QUANTIFICATION_METHODS or bool(re.fullmatch(r"top[1-9]\d*", value))


def requires_peptide_counts(de_method: str, ensemble: str) -> bool:
    """Return whether a differential-expression setting requires peptide counts."""
    return de_method == "deqms" or (
        de_method == "ensemble" and "deqms" in ensemble.split(",")
    )


def _validate_de_settings(
    de_method: str,
    fdr_method: str,
    ensemble: str,
    ensemble_k: Any,
) -> None:
    if de_method in {"rots", "limrots"} and fdr_method != "bh":
        raise ValueError(
            f"de_method={de_method!r} requires fdr_method='bh' because standalone "
            "ROTS-family methods preserve their native permutation FDR"
        )
    if de_method == "ensemble":
        if ensemble == "none":
            raise ValueError("de_method=ensemble requires an ensemble preset")
        if isinstance(ensemble_k, bool) or not isinstance(ensemble_k, int):
            raise ValueError("de_method=ensemble requires an integer ensemble_k")
        if not 1 <= ensemble_k <= 5:
            raise ValueError("ensemble_k must be between 1 and 5")
        member_count = len(ensemble.split(","))
        if ensemble_k > member_count:
            raise ValueError("ensemble_k cannot exceed the ensemble member count")
    else:
        if ensemble != "none":
            raise ValueError("ensemble must be 'none' unless de_method=ensemble")
        if ensemble_k is not None:
            raise ValueError("ensemble_k must be null unless de_method=ensemble")


def _validate_log2fc_threshold(gate: Any) -> None:
    if isinstance(gate, str):
        if gate.lower() != "auto":
            raise ValueError("log2fc_threshold string must be 'auto'")
    elif isinstance(gate, bool) or not isinstance(gate, (int, float)):
        raise ValueError("log2fc_threshold must be a number or 'auto'")
    elif not 0 <= float(gate) <= 10:
        raise ValueError("log2fc_threshold must be between 0 and 10")


def validate_config_values(item: Mapping[str, Any]) -> None:
    """Reject a generated configuration that violates the runtime contract."""
    _require_member(item, "de_method", DE_METHODS)
    _require_member(item, "fdr_method", FDR_METHODS)
    _require_member(item, "normalization", NORMALIZATION_METHODS)
    _require_member(item, "imputation", IMPUTATION_METHODS)
    _require_member(item, "ensemble", ENSEMBLE_PRESETS)
    _validate_de_settings(
        item["de_method"],
        item["fdr_method"],
        item["ensemble"],
        item["ensemble_k"],
    )
    _validate_log2fc_threshold(item["log2fc_threshold"])


def _require_member(
    item: Mapping[str, Any],
    field: str,
    allowed: tuple[str, ...],
) -> None:
    """Validate one enumerated configuration field."""
    value = item.get(field)
    if value not in allowed:
        raise ValueError(f"Unsupported {field}={value!r}; choose from {allowed}")
