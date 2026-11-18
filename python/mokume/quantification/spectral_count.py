"""Count-based protein quantification methods.

``SpectralCountQuantification`` counts distinct (run, scan, sequence) keys, whereas
``PeptideCountQuantification`` counts distinct canonical peptide sequences.
Keeping both contracts explicit prevents feature-level input from being
misreported as a spectral count.

References
----------
- Liu H, Sadygov RG, Yates JR. A model for random sampling and
  estimation of relative protein abundance in shotgun proteomics.
  Anal Chem. 2004;76(14):4193-201.
"""

from typing import Optional

import pandas as pd

from mokume.quantification.base import ProteinQuantificationMethod
from mokume.core.registry import PluginRegistry
from mokume.core.constants import (
    PROTEIN_NAME,
    PEPTIDE_CANONICAL,
    NORM_INTENSITY,
    SAMPLE_ID,
    RUN,
)


def _rank_one(scores) -> bool:
    """Keep rank one, or rows for which the producer supplied no rank."""
    if scores is None:
        return True
    if isinstance(scores, dict):
        return "rank" not in scores or scores["rank"] == 1
    return all(
        score["score_value"] == 1 for score in scores if score["score_name"] == "rank"
    )


def _spectrum_rows(
    frame: pd.DataFrame, group_cols: list[str], sequence: str
) -> pd.DataFrame:
    """Deduplicate complete spectrum keys and reject conflicting assignments."""
    run = (
        group_cols[2]
        if len(group_cols) == 3
        else ("run_file_name" if "run_file_name" in frame else RUN)
    )
    if run not in frame or "scan" not in frame:
        raise ValueError(
            "spectral_count requires run_file_name (or Run) and scan columns"
        )
    if "additional_scores" in frame:
        frame = frame.loc[frame["additional_scores"].map(_rank_one)]
    frame = frame.assign(_spectrum_scan=frame["scan"].map(tuple))
    keys = [run, "_spectrum_scan", sequence]
    assignments = frame.groupby(keys, dropna=False)[group_cols[:2]].nunique(
        dropna=False
    )
    if assignments.gt(1).any().any():
        raise ValueError(
            "PSMs for one (run, scan, sequence) have conflicting "
            "protein-group or sample assignments"
        )
    return frame.drop_duplicates(keys)


@PluginRegistry.register("quantification", "spectral_count")
class SpectralCountQuantification(ProteinQuantificationMethod):
    """
    Spectral count quantification.

    Protein abundance is the number of independent spectrum/sequence pairs
    per protein per sample, after excluding ranks other than one.
    """

    @property
    def name(self) -> str:
        return "SpectralCount"

    def quantify(
        self,
        peptide_df: pd.DataFrame,
        protein_column: str = PROTEIN_NAME,
        peptide_column: str = PEPTIDE_CANONICAL,
        intensity_column: str = NORM_INTENSITY,
        sample_column: str = SAMPLE_ID,
        run_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Count spectra per protein per sample.

        Parameters
        ----------
        peptide_df : pd.DataFrame
            PSM rows with a run column and a ``scan`` integer array. Repeated
            (run, complete scan array, peptide sequence) keys count once.
        protein_column : str
            Column name for protein identifiers.
        peptide_column : str
            Column containing the peptide sequence used in the spectrum key.
        intensity_column : str
            Unused (kept for API compatibility).
        sample_column : str
            Column name for sample identifiers.
        run_column : str, optional
            If provided, counting is done per run instead of per sample.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: *protein_column*, *sample_column*,
            (optionally *run_column*), and ``Intensity`` (the count).
        """
        if run_column is not None and run_column in peptide_df.columns:
            group_cols = [protein_column, sample_column, run_column]
        else:
            group_cols = [protein_column, sample_column]

        return self._aggregate(peptide_df, group_cols, peptide_column)

    @staticmethod
    def _aggregate(
        peptide_df: pd.DataFrame,
        group_cols: list[str],
        peptide_column: str,
    ) -> pd.DataFrame:
        """Count independent spectrum/sequence pairs per protein and sample or run."""
        rows = _spectrum_rows(peptide_df, group_cols, peptide_column)
        return rows.groupby(group_cols).size().reset_index(name="Intensity")


@PluginRegistry.register("quantification", "peptide_count")
class PeptideCountQuantification(SpectralCountQuantification):
    """Count distinct canonical peptides per protein and sample."""

    @property
    def name(self) -> str:
        return "PeptideCount"

    @staticmethod
    def _aggregate(
        peptide_df: pd.DataFrame,
        group_cols: list[str],
        peptide_column: str,
    ) -> pd.DataFrame:
        """Count unique peptide sequences per protein and sample or run."""
        return (
            peptide_df.groupby(group_cols, dropna=False)[peptide_column]
            .nunique()
            .reset_index(name="Intensity")
        )
