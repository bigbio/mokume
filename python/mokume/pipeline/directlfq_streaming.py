"""Streaming DirectLFQ protein estimation helpers."""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, TextIO

import numpy as np
import pandas as pd

from mokume.quantification.precision import profile_residuals

LOGGER = logging.getLogger(__name__)


def iter_protein_groups(normed_df: pd.DataFrame) -> Iterable[tuple[int, pd.DataFrame]]:
    """Yield DirectLFQ protein groups without materialising all groups at once."""
    protein_names = normed_df.index.get_level_values(0).to_numpy()
    seen_proteins = set()
    last_protein = object()
    groups_are_contiguous = True
    for protein_name in protein_names:
        if protein_name != last_protein:
            if protein_name in seen_proteins:
                groups_are_contiguous = False
                break
            seen_proteins.add(protein_name)
            last_protein = protein_name

    if not groups_are_contiguous:
        LOGGER.info("Sorting DirectLFQ frame by protein/ion before streaming")
        normed_df = normed_df.sort_index()
        protein_names = normed_df.index.get_level_values(0).to_numpy()

    ion_names = normed_df.index.get_level_values(1).to_numpy()
    normed_array = normed_df.to_numpy()

    switch_indices = np.where(protein_names[:-1] != protein_names[1:])[0] + 1
    starts = np.insert(switch_indices, 0, 0)
    stops = np.append(switch_indices, len(protein_names))

    for group_idx, (start, stop) in enumerate(zip(starts, stops)):
        index = pd.MultiIndex.from_arrays(
            [protein_names[start:stop], ion_names[start:stop]],
            names=normed_df.index.names,
        )
        yield group_idx, pd.DataFrame(normed_array[start:stop], index=index)


def _estimate_protein_group(args):
    """Worker: estimate one protein and optionally return normalized ions.

    The ordinary path still returns only the protein name and profile. Ion
    frames cross the process boundary only when the caller requested export;
    with ``include_residuals`` the worker also returns the shifted ions'
    residuals around the profile.
    """
    (
        group_idx,
        protein_df,
        num_samples_quadratic,
        min_nonan,
        include_ions,
        include_residuals,
    ) = args
    from directlfq.protein_intensity_estimation import (
        calculate_peptide_and_protein_intensities,
    )

    protein_profile, shifted_peptides = calculate_peptide_and_protein_intensities(
        group_idx,
        protein_df,
        num_samples_quadratic,
        min_nonan,
    )
    protein_name = shifted_peptides.index.get_level_values(0)[0]
    fit = (
        profile_residuals(shifted_peptides.to_numpy(), protein_profile)
        if include_residuals and protein_profile is not None
        else None
    )
    return (
        protein_name,
        protein_profile,
        shifted_peptides if include_ions else None,
        fit,
    )


@contextmanager
def _ion_export_file(
    path: Optional[str], sample_columns: Iterable[str]
) -> Iterator[Optional[TextIO]]:
    """Yield an atomic CSV target, or ``None`` when export is disabled."""
    if path is None:
        yield None
        return

    target = Path(path)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        pd.DataFrame(columns=["protein", "ion", *sample_columns]).to_csv(
            handle, index=False
        )
        yield handle
        handle.close()
        os.replace(temporary, target)
    except BaseException:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise


def _write_ion_results(results: Iterable[tuple], output: Optional[TextIO]) -> None:
    """Append DirectLFQ-normalized ion rows in linear intensity space."""
    if output is None:
        return
    for _, _, shifted_peptides, _ in results:
        if shifted_peptides is None:
            continue
        values = np.nan_to_num(2 ** shifted_peptides.to_numpy())
        ion_df = pd.DataFrame(values, columns=shifted_peptides.columns)
        ion_df.insert(0, "ion", shifted_peptides.index.get_level_values(1))
        ion_df.insert(0, "protein", shifted_peptides.index.get_level_values(0))
        ion_df.to_csv(output, index=False, header=False)


def _protein_profiles_to_frame(
    protein_ids: list[str],
    profile_rows: list[np.ndarray],
    sample_columns: Iterable[str],
) -> pd.DataFrame:
    """Build the linear protein matrix from streamed DirectLFQ profiles."""
    lfq_config = importlib.import_module("directlfq.config")

    if not profile_rows:
        return pd.DataFrame(columns=[lfq_config.PROTEIN_ID, *sample_columns])
    protein_index = pd.Index(protein_ids, name=lfq_config.PROTEIN_ID)
    protein_df = 2 ** pd.DataFrame(
        profile_rows,
        index=protein_index,
        columns=sample_columns,
    )
    return protein_df.replace(np.nan, 0).reset_index()


def estimate_protein_intensities_streamed(
    normed_df: pd.DataFrame,
    min_nonan: int,
    num_samples_quadratic: int,
    num_cores: Optional[int] = None,
    export_ions: Optional[str] = None,
    residuals: Optional[list] = None,
) -> pd.DataFrame:
    """Estimate proteins and optionally stream normalized ions to CSV.

    This mirrors the external directlfq package's protein estimation path, but
    keeps only final protein profiles in memory. When ``export_ions`` is set,
    each bounded result batch is converted back to linear intensity space and
    appended to an atomic CSV output instead of retaining the full ion matrix.
    When ``residuals`` is a list, each estimated protein's
    ``(protein, profile_residuals)`` pair is appended to it.

    When ``num_cores > 1`` the per-protein estimation runs across a process pool.
    Unlike directlfq's own multiprocessing, proteins are streamed from
    :func:`iter_protein_groups` and dispatched in bounded batches (so at most a
    few cores' worth of protein frames are materialised at once) and workers
    return normalized ions only when export is requested. Output row order is
    preserved, so results are identical to the sequential path regardless of
    core count.
    """
    protein_ids: list[str] = []
    profile_rows: list[np.ndarray] = []

    LOGGER.info("%d lfq-groups total", normed_df.index.get_level_values(0).nunique())

    with _ion_export_file(export_ions, normed_df.columns) as ion_output:
        options = (
            num_samples_quadratic,
            min_nonan,
            ion_output is not None,
            residuals is not None,
        )
        if num_cores is not None and num_cores > 1:
            LOGGER.info(
                "Estimating DirectLFQ proteins in parallel (%d cores)", num_cores
            )
            batch_size = num_cores * 4
            with mp.Pool(processes=num_cores) as pool:
                batch: list[tuple] = []
                for group_idx, protein_df in iter_protein_groups(normed_df):
                    batch.append((group_idx, protein_df, *options))
                    if len(batch) >= batch_size:
                        results = pool.map(_estimate_protein_group, batch)
                        _append_results(results, protein_ids, profile_rows, residuals)
                        _write_ion_results(results, ion_output)
                        batch = []
                if batch:
                    results = pool.map(_estimate_protein_group, batch)
                    _append_results(results, protein_ids, profile_rows, residuals)
                    _write_ion_results(results, ion_output)
        else:
            for group_idx, protein_df in iter_protein_groups(normed_df):
                results = [_estimate_protein_group((group_idx, protein_df, *options))]
                _append_results(results, protein_ids, profile_rows, residuals)
                _write_ion_results(results, ion_output)

    return _protein_profiles_to_frame(protein_ids, profile_rows, normed_df.columns)


def _append_results(
    results, protein_ids: list, profile_rows: list, residuals: Optional[list]
) -> None:
    """Append non-empty (name, profile) worker results, preserving order."""
    for protein_name, protein_profile, _, fit in results:
        if protein_profile is not None:
            protein_ids.append(protein_name)
            profile_rows.append(protein_profile)
        if fit is not None:
            residuals.append((protein_name, fit))
