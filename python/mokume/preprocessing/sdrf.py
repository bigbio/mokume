"""
SDRF analysis utilities.

This module provides functions for analyzing SDRF (Sample and Data Relationship Format)
files to determine quantification details for proteomics experiments.
"""

from typing import Optional

import pandas as pd

from mokume.model.labeling import QuantificationCategory, IsobaricLabel
from mokume.core.constants import CHANNEL, load_sdrf
from mokume.core.logger import get_logger

logger = get_logger("mokume.preprocessing.sdrf")


def analyse_sdrf(
    sdrf_path: str,
) -> tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]:
    """
    Analyzes an SDRF file to determine quantification details.

    Parameters
    ----------
    sdrf_path : str
        The file path to the SDRF file.

    Returns
    -------
    tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]
        A tuple containing the number of technical repetitions, the quantification category,
        a list of unique sample names, and the isobaric label scheme if applicable.
    """
    sdrf_df = load_sdrf(sdrf_path)

    labels = set(sdrf_df["comment[label]"])
    # Determine label type
    label, channel_set = QuantificationCategory.classify(labels)
    if label in (QuantificationCategory.TMT, QuantificationCategory.ITRAQ):
        choice_df = (
            pd.DataFrame.from_dict(
                channel_set.channels(), orient="index", columns=[CHANNEL]
            )
            .reset_index()
            .rename(columns={"index": "comment[label]"})
        )
        sdrf_df = sdrf_df.merge(choice_df, on="comment[label]", how="left")
    sample_names = sdrf_df["source name"].unique().tolist()
    technical_repetitions = len(sdrf_df["comment[technical replicate]"].unique())
    return technical_repetitions, label, sample_names, channel_set


# Number of source names, and of data files per name, spelled out in one
# shared-source-name warning before the rest is summarised as a count. Kept in
# step with ``SHARED_SOURCE_NAME_LIST_LIMIT`` in ``mokume-io/src/sdrf.rs``.
_SHARED_SOURCE_NAME_LIST_LIMIT = 10
_SHARED_SOURCE_FILE_LIST_LIMIT = 5


def _sdrf_cell(value) -> Optional[str]:
    """Return a trimmed SDRF cell as text, or ``None`` when it is empty."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    return text or None


def _technical_replicate(value) -> Optional[int]:
    """Parse ``comment[technical replicate]`` as a positive int, like the Rust reader."""
    text = _sdrf_cell(value)
    if text is None or not text.isdigit() or int(text) == 0:
        return None
    return int(text)


def shared_source_names(sdrf_df: pd.DataFrame) -> list[dict]:
    """
    List SDRF source names that are mapped to more than one data file.

    Runs that share a source name are combined into one sample, so
    normalization and cross-sample quantification see one column for them.
    That is how technical replicates and fractions are meant to be combined,
    but the same pattern also follows from a copy-paste error in the SDRF.

    Parameters
    ----------
    sdrf_df : pd.DataFrame
        SDRF table with lower-cased headers, as returned by ``load_sdrf``.

    Returns
    -------
    list[dict]
        One entry per shared source name, in SDRF order, with keys
        ``source_name``, ``data_files`` (distinct, in SDRF order) and
        ``same_replicate_and_fraction`` (two of the data files carry the same
        technical replicate and fraction identifier, identical or absent).
    """
    if (
        "source name" not in sdrf_df.columns
        or "comment[data file]" not in sdrf_df.columns
    ):
        return []
    missing = pd.Series(None, index=sdrf_df.index, dtype=object)
    technical = sdrf_df.get("comment[technical replicate]", missing)
    fraction = sdrf_df.get("comment[fraction identifier]", missing)

    groups: dict[str, list[tuple[str, tuple]]] = {}
    for name, data_file, techrep, frac in zip(
        sdrf_df["source name"], sdrf_df["comment[data file]"], technical, fraction
    ):
        name, data_file = _sdrf_cell(name), _sdrf_cell(data_file)
        if data_file is None:
            continue
        design = (_technical_replicate(techrep), _sdrf_cell(frac))
        groups.setdefault(name or data_file, []).append((data_file, design))

    shared = []
    for name, rows in groups.items():
        data_files = list(dict.fromkeys(data_file for data_file, _ in rows))
        if len(data_files) < 2:
            continue
        files_by_design: dict[tuple, set[str]] = {}
        for data_file, design in rows:
            files_by_design.setdefault(design, set()).add(data_file)
        shared.append(
            {
                "source_name": name,
                "data_files": data_files,
                "same_replicate_and_fraction": any(
                    len(files) > 1 for files in files_by_design.values()
                ),
            }
        )
    return shared


def _describe_shared_source_names(shared: list[dict]) -> str:
    parts = []
    for entry in shared[:_SHARED_SOURCE_NAME_LIST_LIMIT]:
        data_files = entry["data_files"]
        files = ", ".join(data_files[:_SHARED_SOURCE_FILE_LIST_LIMIT])
        if len(data_files) > _SHARED_SOURCE_FILE_LIST_LIMIT:
            files += f", ... {len(data_files) - _SHARED_SOURCE_FILE_LIST_LIMIT} more"
        parts.append(f"`{entry['source_name']}` ({files})")
    if len(shared) > _SHARED_SOURCE_NAME_LIST_LIMIT:
        parts.append(f"... {len(shared) - _SHARED_SOURCE_NAME_LIST_LIMIT} more")
    return "; ".join(parts)


def shared_source_name_warnings(sdrf_df: pd.DataFrame) -> list[str]:
    """
    Build the warnings for source names mapped to more than one data file.

    Mirrors ``SdrfTable::shared_source_name_warnings`` in the Rust kernel: one
    message for names whose runs share technical replicate and fraction (a
    likely duplicated name), one for the remaining names (replicates or
    fractions).

    Parameters
    ----------
    sdrf_df : pd.DataFrame
        SDRF table with lower-cased headers, as returned by ``load_sdrf``.

    Returns
    -------
    list[str]
        Warning messages; empty when every source name has a single data file.
    """
    shared = shared_source_names(sdrf_df)
    duplicated = [entry for entry in shared if entry["same_replicate_and_fraction"]]
    replicated = [entry for entry in shared if not entry["same_replicate_and_fraction"]]
    warnings = []
    if duplicated:
        names = _describe_shared_source_names(duplicated)
        warnings.append(
            "SDRF source names reused across data files with the same technical "
            f"replicate and fraction identifier: {names}. "
            "These runs are combined into one sample, which also changes the "
            "normalization of all other samples. Identical replicate and fraction "
            "values usually indicate a duplicated source name rather than technical "
            "replicates; check the SDRF `source name` column."
        )
    if replicated:
        warnings.append(
            "SDRF source names mapped to more than one data file: "
            f"{_describe_shared_source_names(replicated)}. Runs that share a source "
            "name are combined into one sample (technical replicates or fractions), "
            "which also affects the normalization of all other samples."
        )
    return warnings


def warn_shared_source_names(sdrf_df: pd.DataFrame) -> None:
    """Log :func:`shared_source_name_warnings` as warnings; the mapping is unchanged."""
    for message in shared_source_name_warnings(sdrf_df):
        logger.warning(message)
