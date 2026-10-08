"""
Protein-level preprocessing filters.
"""

from typing import Tuple, List

import pandas as pd

from mokume.core.logger import get_logger
from mokume.core.constants import PROTEIN_NAME, PEPTIDE_CANONICAL
from mokume.core.contaminants import contaminant_group_mask
from mokume.preprocessing.filters.base import BaseFilter, FilterResult
from mokume.preprocessing.filters.enums import FilterLevel, RazorPeptideHandling


logger = get_logger("mokume.preprocessing.filters.protein")
PROTEIN_QVALUE = "pg_global_qvalue"


class ContaminantFilter(BaseFilter):
    """Filter contaminant and decoy proteins."""

    def __init__(
        self,
        patterns: List[str],
        remove_decoys: bool = True,
        protein_column: str = PROTEIN_NAME,
    ):
        """
        Initialize the filter.

        Parameters
        ----------
        patterns : list[str]
            List of patterns identifying contaminant proteins.
        remove_decoys : bool, optional
            Whether to remove decoy proteins.
        protein_column : str, optional
            Column name containing protein identifiers.
        """
        self.patterns = patterns
        self.remove_decoys = remove_decoys
        self.protein_column = protein_column

    @property
    def name(self) -> str:
        return "ContaminantFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PROTEIN

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)

        if self.protein_column not in df.columns:
            logger.warning(
                "%s: Protein column '%s' not found, skipping filter",
                self.name,
                self.protein_column,
            )
            return df, self._create_result(input_count, input_count)

        active_patterns = [
            pattern
            for pattern in self.patterns
            if self.remove_decoys or pattern.upper() != "DECOY"
        ]
        if not active_patterns:
            return df, self._create_result(
                input_count,
                input_count,
                {"patterns": [], "remove_decoys": self.remove_decoys},
            )

        mask = ~contaminant_group_mask(df[self.protein_column], active_patterns)
        filtered_df = df[mask].copy()

        output_count = len(filtered_df)

        logger.debug(
            "%s: Removed %d entries matching contaminant patterns",
            self.name,
            input_count - output_count,
        )

        return filtered_df, self._create_result(
            input_count,
            output_count,
            {"patterns": active_patterns, "remove_decoys": self.remove_decoys},
        )


class MinPeptideFilter(BaseFilter):
    """Filter proteins by minimum number of peptides."""

    def __init__(
        self,
        min_peptides: int = 1,
        min_unique_peptides: int = 2,
        protein_column: str = PROTEIN_NAME,
        peptide_column: str = PEPTIDE_CANONICAL,
    ):
        """
        Initialize the filter.

        Parameters
        ----------
        min_peptides : int, optional
            Minimum total peptides per protein.
        min_unique_peptides : int, optional
            Minimum unique peptides per protein.
        protein_column : str, optional
            Column name containing protein identifiers.
        peptide_column : str, optional
            Column name containing peptide sequences.
        """
        self.min_peptides = min_peptides
        self.min_unique_peptides = min_unique_peptides
        self.protein_column = protein_column
        self.peptide_column = peptide_column

    @property
    def name(self) -> str:
        return "MinPeptideFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PROTEIN

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)

        if self.protein_column not in df.columns:
            logger.warning(
                "%s: Protein column '%s' not found, skipping filter",
                self.name,
                self.protein_column,
            )
            return df, self._create_result(input_count, input_count)

        # Count peptides per protein
        if self.peptide_column in df.columns:
            peptide_counts = (
                df.groupby(self.protein_column)[self.peptide_column]
                .nunique()
                .reset_index()
            )
            peptide_counts.columns = [self.protein_column, "unique_peptide_count"]

            # Every retained row is a unique-peptide row in the standard
            # features2peptides path, so both thresholds apply to the distinct
            # peptide count and the stricter threshold wins.
            minimum_count = max(self.min_peptides, self.min_unique_peptides)
            passing_proteins = peptide_counts[
                peptide_counts["unique_peptide_count"] >= minimum_count
            ][self.protein_column]

            filtered_df = df[df[self.protein_column].isin(passing_proteins)].copy()
        else:
            # Fall back to counting rows per protein
            protein_counts = df[self.protein_column].value_counts()
            passing_proteins = protein_counts[protein_counts >= self.min_peptides].index

            filtered_df = df[df[self.protein_column].isin(passing_proteins)].copy()

        output_count = len(filtered_df)

        logger.debug(
            "%s: Removed %d entries below peptide thresholds (%d total, %d unique)",
            self.name,
            input_count - output_count,
            self.min_peptides,
            self.min_unique_peptides,
        )

        return filtered_df, self._create_result(
            input_count,
            output_count,
            {
                "min_peptides": self.min_peptides,
                "min_unique_peptides": self.min_unique_peptides,
            },
        )


class ProteinFDRFilter(BaseFilter):
    """Keep protein groups whose minimum QPX protein q-value passes."""

    def __init__(
        self,
        fdr_threshold: float,
        fdr_column: str = PROTEIN_QVALUE,
        protein_column: str = PROTEIN_NAME,
    ) -> None:
        self.fdr_threshold = fdr_threshold
        self.fdr_column = fdr_column
        self.protein_column = protein_column

    @property
    def name(self) -> str:
        return "ProteinFDRFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PROTEIN

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)
        if self.fdr_column not in df.columns or not df[self.fdr_column].notna().any():
            raise ValueError(
                "protein FDR filtering requires a populated QPX "
                f"'{self.fdr_column}' column"
            )
        protein_fdr = df.groupby(self.protein_column)[self.fdr_column].min()
        passing = protein_fdr[protein_fdr <= self.fdr_threshold].index
        filtered_df = df[df[self.protein_column].isin(passing)].copy()
        return filtered_df, self._create_result(
            input_count,
            len(filtered_df),
            {"fdr_threshold": self.fdr_threshold},
        )


class RazorPeptideFilter(BaseFilter):
    """Handle razor (shared) peptides."""

    def __init__(
        self,
        handling: str = "keep",
        protein_column: str = PROTEIN_NAME,
        peptide_column: str = PEPTIDE_CANONICAL,
    ):
        """
        Initialize the filter.

        Parameters
        ----------
        handling : str, optional
            How to handle razor peptides: 'keep', 'remove', 'assign_to_top'.
        protein_column : str, optional
            Column name containing protein identifiers.
        peptide_column : str, optional
            Column name containing peptide sequences.
        """
        self.handling = RazorPeptideHandling.from_str(handling)
        self.protein_column = protein_column
        self.peptide_column = peptide_column

    @property
    def name(self) -> str:
        return "RazorPeptideFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PEPTIDE

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)

        if self.handling == RazorPeptideHandling.KEEP:
            return df, self._create_result(input_count, input_count)

        if self.peptide_column not in df.columns:
            logger.warning(
                "%s: Peptide column '%s' not found, skipping filter",
                self.name,
                self.peptide_column,
            )
            return df, self._create_result(input_count, input_count)

        # Identify razor peptides (peptides mapping to multiple proteins)
        peptide_protein_counts = df.groupby(self.peptide_column)[
            self.protein_column
        ].nunique()
        razor_peptides = peptide_protein_counts[peptide_protein_counts > 1].index

        if self.handling == RazorPeptideHandling.REMOVE:
            # Remove all razor peptides
            mask = ~df[self.peptide_column].isin(razor_peptides)
            filtered_df = df[mask].copy()
        elif self.handling == RazorPeptideHandling.ASSIGN_TO_TOP:
            # Keep only assignment to protein with most peptides
            # First, count unique peptides per protein
            protein_peptide_counts = df.groupby(self.protein_column)[
                self.peptide_column
            ].nunique()

            # For each razor peptide, keep only the one assigned to top protein
            def assign_to_top(group):
                if len(group[self.protein_column].unique()) == 1:
                    return group
                # Get protein with most peptides
                proteins = group[self.protein_column].unique()
                top_protein = max(
                    proteins, key=lambda p: protein_peptide_counts.get(p, 0)
                )
                return group[group[self.protein_column] == top_protein]

            filtered_df = df.groupby(self.peptide_column, group_keys=False).apply(
                assign_to_top
            )
        else:
            filtered_df = df

        output_count = len(filtered_df)

        logger.debug(
            "%s: Handling=%s, removed %d entries",
            self.name,
            self.handling.name,
            input_count - output_count,
        )

        return filtered_df, self._create_result(
            input_count,
            output_count,
            {
                "handling": self.handling.name,
                "razor_peptides_found": len(razor_peptides),
            },
        )
