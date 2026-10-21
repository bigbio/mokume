"""Maximal peptide ratio extraction from Cox et al. (2014), Eq. 3/Fig. 2.

This module reconstructs protein profiles from pairwise peptide-species ratios.
Normalization is an explicit upstream step; fraction-aware delayed normalization
is not performed here. Optional large-ratio stabilization follows Eq. 5.
Source: https://doi.org/10.1074/mcp.M113.031591
"""

import math
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mokume.core.constants import (
    NORM_INTENSITY,
    PEPTIDE_CANONICAL,
    PROTEIN_NAME,
    SAMPLE_ID,
)
from mokume.core.logger import get_logger
from mokume.core.registry import PluginRegistry
from mokume.quantification.base import ProteinQuantificationMethod

logger = get_logger("mokume.quantification.maxlfq")


def _identifier_sort_key(value: object) -> tuple[str, str, object]:
    value_type = type(value)
    return value_type.__module__, value_type.__qualname__, value


def _ratio_system(log_matrix: np.ndarray, min_count: int, stabilize: bool):
    """Build Eq. 3's unweighted graph Laplacian and right-hand side."""
    n_samples = log_matrix.shape[1]
    laplacian = np.zeros((n_samples, n_samples))
    rhs = np.zeros(n_samples)
    stabilized_pairs = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            valid = np.isfinite(log_matrix[:, i]) & np.isfinite(log_matrix[:, j])
            if np.count_nonzero(valid) < min_count:
                continue
            ratio = np.median(log_matrix[valid, j] - log_matrix[valid, i])
            if stabilize:
                ratio, applied = _stabilized_ratio(log_matrix[:, [i, j]], valid, ratio)
                stabilized_pairs += int(applied)
            laplacian[i, i] += 1
            laplacian[j, j] += 1
            laplacian[i, j] = laplacian[j, i] = -1
            rhs[i] -= ratio
            rhs[j] += ratio
    return laplacian, rhs, stabilized_pairs


def _stabilized_ratio(log_pair, shared_mask, median_ratio):
    """Apply Cox Eq. 5 in log2 space after the minimum shared count is met."""
    counts = np.isfinite(log_pair).sum(axis=0)
    x = counts.max() / np.count_nonzero(shared_mask)
    weight = np.clip((x - 2.5) / 2.5, 0.0, 1.0)
    if weight == 0:
        return median_ratio, False
    sum_ratio = _log_total(log_pair[:, 1]) - _log_total(log_pair[:, 0])
    return (1 - weight) * median_ratio + weight * sum_ratio, True


def _components(laplacian: np.ndarray):
    """Find connected sample components; isolated samples remain unquantified."""
    seen = set()
    for start in range(len(laplacian)):
        if start in seen or laplacian[start, start] == 0:
            continue
        pending = [start]
        seen.add(start)
        component = []
        while pending:
            i = pending.pop()
            component.append(i)
            for j in np.flatnonzero(laplacian[i] < 0):
                if j not in seen:
                    seen.add(j)
                    pending.append(j)
        yield sorted(component)


def _log_total(log_values: np.ndarray) -> float:
    values = np.asarray(log_values).ravel()
    values = values[np.isfinite(values)]
    maximum = values.max()
    return float(maximum + math.log2(math.fsum(np.exp2(values - maximum))))


def _maxlfq_solve_protein(
    peptide_matrix: np.ndarray,
    peptide_ids: Optional[np.ndarray] = None,
    min_ratio_count: int = 2,
    stabilize: bool = False,
) -> np.ndarray:
    """Solve a positive linear species-by-sample matrix using Cox et al. Eq. 3.

    Each valid sample-pair edge has equal weight. Medians are computed in log2
    space, including the midpoint convention for an even number of ratios.
    Samples without a valid edge return zero (Fig. 2D). Disconnected components
    are independently anchored to their observed intensity totals, then the
    whole supported profile is rescaled to the total input intensity. No ratio
    between disconnected components is identifiable from shared species.

    ``peptide_ids`` is accepted for compatibility; no reference peptide is used.
    """
    _ = peptide_ids
    return _solve_protein(peptide_matrix, min_ratio_count, stabilize)[0]


def _solve_protein(peptide_matrix, min_ratio_count, stabilize):
    """Return protein quantities and the number of stabilized sample pairs."""
    if not isinstance(stabilize, bool):
        raise ValueError("MaxLFQ stabilize must be a boolean")
    if (
        isinstance(min_ratio_count, bool)
        or not isinstance(min_ratio_count, (int, np.integer))
        or min_ratio_count < 1
    ):
        raise ValueError("MaxLFQ minimum ratio count must be a positive integer")
    matrix = np.asarray(peptide_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("MaxLFQ input must be a species-by-sample matrix")
    log_matrix = np.full(matrix.shape, np.nan)
    valid = np.isfinite(matrix) & (matrix > 0)
    np.log2(matrix, out=log_matrix, where=valid)
    laplacian, rhs, stabilized_pairs = _ratio_system(
        log_matrix, min_ratio_count, stabilize
    )
    log_profile = np.full(matrix.shape[1], -np.inf)
    components = list(_components(laplacian))
    for component in components:
        free = component[1:]
        relative = np.r_[0.0, np.linalg.solve(laplacian[np.ix_(free, free)], rhs[free])]
        log_profile[component] = (
            relative + _log_total(log_matrix[:, component]) - _log_total(relative)
        )
    if components:
        log_profile += _log_total(log_matrix) - _log_total(log_profile)
    if len(components) > 1:
        logger.warning(
            "MaxLFQ has %d disconnected sample components; "
            "between-component ratios are not identifiable",
            len(components),
        )
    with np.errstate(over="ignore", under="ignore"):
        result = np.exp2(log_profile)
    if np.any(~np.isfinite(result)) or np.any(np.isfinite(log_profile) & (result == 0)):
        raise ValueError("MaxLFQ protein intensity is outside the finite float64 range")
    return result, stabilized_pairs


def _process_protein(
    protein: str,
    protein_data: pd.DataFrame,
    columns: tuple[str, str, str],
    samples: list,
    ratio_options: dict,
) -> tuple[list, int]:
    """Sum duplicate species observations, then solve a single protein."""
    peptide_column, intensity_column, sample_column = columns
    peptides = sorted(protein_data[peptide_column].unique(), key=_identifier_sort_key)
    pivot = protein_data.pivot_table(
        index=peptide_column,
        columns=sample_column,
        values=intensity_column,
        aggfunc=math.fsum,
        observed=True,
    ).reindex(index=peptides, columns=samples)
    intensities, stabilized_pairs = _solve_protein(pivot.to_numpy(), **ratio_options)
    records = [
        {"protein": protein, "sample": sample, "intensity": intensity}
        for sample, intensity in zip(samples, intensities)
        if intensity > 0
    ]
    return records, stabilized_pairs


def _prepare_input(peptide_df, columns, run_column):
    """Filter invalid rows and retain the supplied species and sample/run axes."""
    protein_column, peptide_column, intensity_column, sample_column = columns
    frame = peptide_df.copy()
    frame = frame[np.isfinite(frame[intensity_column]) & (frame[intensity_column] > 0)]
    required = [protein_column, peptide_column, sample_column]
    if run_column is not None:
        required.append(run_column)
    frame = frame.dropna(subset=required)
    if {"PeptideSequence", "PrecursorCharge"}.issubset(frame.columns):
        frame = frame.dropna(subset=["PeptideSequence", "PrecursorCharge"])
        frame["PrecursorCharge"] = pd.to_numeric(frame["PrecursorCharge"])
        charge = frame["PrecursorCharge"]
        frame = frame[np.isfinite(charge) & (charge > 0) & (charge % 1 == 0)].copy()
        frame["_maxlfq_species"] = list(
            zip(frame["PeptideSequence"], frame["PrecursorCharge"])
        )
        peptide_column = "_maxlfq_species"
    if run_column is not None:
        frame["_maxlfq_sample_run"] = list(zip(frame[sample_column], frame[run_column]))
        axis_column = "_maxlfq_sample_run"
    else:
        axis_column = sample_column
    return frame, peptide_column, axis_column


def _format_results(all_results, protein_column, sample_column, run_column):
    """Restore caller identifiers and optional run columns in positive output."""
    result = pd.DataFrame(
        [row for group, _ in all_results for row in group],
        columns=["protein", "sample", "intensity"],
    ).rename(
        columns={
            "protein": protein_column,
            "sample": sample_column,
            "intensity": "Intensity",
        }
    )
    if run_column is not None:
        pairs = result[sample_column].tolist()
        result[sample_column] = [pair[0] for pair in pairs]
        result[run_column] = [pair[1] for pair in pairs]
        result = result[[protein_column, sample_column, run_column, "Intensity"]]
    return result


@PluginRegistry.register("quantification", "maxlfq")
@dataclass
class MaxLFQQuantification(ProteinQuantificationMethod):
    """Independent MaxLFQ ratio solver for prepared linear intensities.

    ``min_ratio_count`` is the number of shared species required for EACH sample
    pair (default 2). ``min_peptides`` is its legacy argument alias. Proteins or
    samples without a valid ratio are not replaced by median aggregation.
    ``force_builtin`` is retained as a no-op for compatibility. MaxLFQ never
    dispatches to DirectLFQ. ``threads`` follows joblib's worker-count semantics.
    ``stabilize`` enables Eq. 5's overlap-dependent large-ratio stabilization.
    """

    min_peptides: int = 2
    threads: int = -1
    verbose: int = 0
    force_builtin: bool = False
    min_ratio_count: Optional[int] = field(default=None, kw_only=True)
    stabilize: bool = field(default=False, kw_only=True)
    using_directlfq: ClassVar[bool] = False

    def __post_init__(self):
        self.min_ratio_count = (
            self.min_peptides if self.min_ratio_count is None else self.min_ratio_count
        )
        _maxlfq_solve_protein(
            np.empty((0, 0)),
            min_ratio_count=self.min_ratio_count,
            stabilize=self.stabilize,
        )
        self.min_peptides = self.min_ratio_count

    @property
    def name(self) -> str:
        return "MaxLFQ"

    def quantify(
        self,
        peptide_df: pd.DataFrame,
        protein_column: str = PROTEIN_NAME,
        peptide_column: str = PEPTIDE_CANONICAL,
        intensity_column: str = NORM_INTENSITY,
        sample_column: str = SAMPLE_ID,
        run_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Quantify sample or (sample, run) profiles, preserving species identity.

        When PeptideSequence and PrecursorCharge are available, use their pair
        as the species identifier. Otherwise use the supplied peptide column;
        a canonical-only input cannot recover discarded charge/modification data.
        Missing and invalid intensities are excluded; empty outputs keep headers.
        """
        frame, peptide_column, axis_column = _prepare_input(
            peptide_df,
            (protein_column, peptide_column, intensity_column, sample_column),
            run_column,
        )
        samples = sorted(frame[axis_column].unique(), key=_identifier_sort_key)
        groups = frame.groupby(protein_column, observed=True)
        all_results = Parallel(n_jobs=self.threads, verbose=self.verbose)(
            delayed(_process_protein)(
                protein,
                group,
                (peptide_column, intensity_column, axis_column),
                samples,
                {
                    "min_ratio_count": self.min_ratio_count,
                    "stabilize": self.stabilize,
                },
            )
            for protein, group in groups
        )
        logger.info(
            "MaxLFQ min_ratio_count=%d stabilize=%s: %d protein groups and %d "
            "sample pairs have positive stabilization weights",
            self.min_ratio_count,
            self.stabilize,
            sum(count > 0 for _, count in all_results),
            sum(count for _, count in all_results),
        )
        return _format_results(all_results, protein_column, sample_column, run_column)
