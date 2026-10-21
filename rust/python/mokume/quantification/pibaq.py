"""Rust-backed piBAQ compatibility APIs.

The wheel keeps the established DataFrame and file-oriented Python entrypoints,
but shared-peptide allocation, theoretical-peptide denominators, evidence
classification, and TPA are computed by the same native core as the CLI.
"""

import importlib
import inspect
from types import ModuleType
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from mokume.core.constants import (
    CONCENTRATION_NM,
    CONDITION,
    COPYNUMBER,
    EVIDENCE_FAMILY_ONLY,
    EVIDENCE_LEVEL,
    FAMILY_ID,
    FAMILY_SIZE,
    MOLECULARWEIGHT,
    MOLES_NMOL,
    NORM_INTENSITY,
    PIBAQ,
    PIBAQ_LOG,
    PIBAQ_NORMALIZED,
    PIBAQ_PPB,
    PROTEIN_NAME,
    SAMPLE_ID,
    TPA,
    WEIGHT_NG,
)
from mokume.core.logger import get_logger, log_execution_time
from mokume.io.fasta import extract_fasta as _extract_fasta
from mokume.model.organism import OrganismDescription
from mokume.plotting import is_plotting_available
from mokume.quantification._pibaq_allocation import (
    _detect_peptide_column,
    _empty_pibaq_frame,
    _membership_mask,
)
from mokume.quantification.families import Family

AVAGADRO = 6.02214129e23
AVERAGE_BASE_PAIR_MASS = 617.96

logger = get_logger("mokume.quantification.pibaq")
extract_fasta = _extract_fasta


def _public_signature(
    positional_names: Sequence[str], defaults: Mapping[str, object]
) -> inspect.Signature:
    """Build an inspectable signature for a static-analysis-friendly wrapper."""
    parameters = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in positional_names
    ]
    parameters.extend(
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=default,
        )
        for name, default in defaults.items()
    )
    return inspect.Signature(parameters)


def _bind(
    signature: inspect.Signature,
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Validate a compatibility call and materialize its defaults."""
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def normalize_pibaq(result: DataFrame) -> DataFrame:
    """Normalize piBAQ within each sample and condition."""
    totals = result.groupby([SAMPLE_ID, CONDITION], observed=True)[PIBAQ].transform(
        "sum"
    )
    relative = result[PIBAQ].divide(totals)
    result.loc[:, PIBAQ_NORMALIZED] = relative
    result.loc[:, PIBAQ_LOG] = np.where(relative.gt(0), np.log10(relative) + 10, 0)
    result.loc[:, PIBAQ_PPB] = relative.multiply(100_000_000)
    return result


def handle_nonstandard_aa(aa_seq: str):
    """Return removed residues and a sequence containing standard residues."""
    standard = frozenset("ARNDBCEQZGHILKMFPSTWYV")
    removed = [residue for residue in aa_seq if residue not in standard]
    return removed, "".join(residue for residue in aa_seq if residue in standard)


class ConcentrationWeightByProteomicRuler:
    """Estimate protein abundance measurements with the proteomic ruler."""

    def __init__(
        self, organism: OrganismDescription, ploidy: int, concentration_per_cell: float
    ):
        self.organism = organism
        self.ploidy = ploidy
        self.concentration_per_cell = concentration_per_cell
        self.dna_mass = (
            ploidy * organism.genome_size * AVERAGE_BASE_PAIR_MASS / AVAGADRO
        )

    def total_histone_intensities(self, protein_intensities: DataFrame) -> float:
        """Return the summed intensity of accession- or entry-matched histones."""
        identifiers = self.organism.histone_proteins + self.organism.histone_entries
        matched = protein_intensities.loc[
            protein_intensities[PROTEIN_NAME].isin(identifiers), NORM_INTENSITY
        ]
        return max(matched.sum(), 1.0)

    def apply_ruler(self, protein_intensities: DataFrame) -> DataFrame:
        """Calculate copy number, amount, weight, and concentration columns."""
        result = protein_intensities.copy()
        histone_total = self.total_histone_intensities(result)
        copy_number = result[NORM_INTENSITY] / histone_total
        copy_number *= self.dna_mass
        copy_number *= AVAGADRO
        copy_number /= result[MOLECULARWEIGHT]
        result[COPYNUMBER] = copy_number
        result[MOLES_NMOL] = result[COPYNUMBER] * (1e9 / AVAGADRO)
        result[WEIGHT_NG] = result[MOLES_NMOL] * result[MOLECULARWEIGHT]
        volume = result[WEIGHT_NG].sum() / 1e-9
        volume /= self.concentration_per_cell
        result[CONCENTRATION_NM] = volume * result[MOLES_NMOL]
        return result

    def __call__(self, protein_intensities: DataFrame) -> DataFrame:
        """Apply the ruler to one condition's protein rows."""
        return self.apply_ruler(protein_intensities)

    def apply_by_condition(self, protein_intensities: DataFrame) -> DataFrame:
        """Apply the ruler independently within each condition."""
        if protein_intensities.empty:
            return protein_intensities.copy()
        original_columns = list(protein_intensities.columns)
        groups = protein_intensities.groupby([CONDITION], sort=True)
        result = pd.concat(
            (self.apply_ruler(group) for _, group in groups),
            axis=0,
        )
        additions = [column for column in result if column not in original_columns]
        return result[original_columns + additions]


_COMPUTE_SIGNATURE = _public_signature(
    (
        "peptide_df",
        "accession_to_peptides",
        "peptide_to_accessions",
        "families",
    ),
    {
        "mw_map": None,
        "min_anchors": 1,
        "high_anchor_threshold": 3,
        "extra_group_cols": None,
    },
)

_GROUP_ID = "_mokume_group_id"
_FAMILY_ORDER = "_mokume_family_order"
_NATIVE_COLUMNS = [
    PROTEIN_NAME,
    _GROUP_ID,
    NORM_INTENSITY,
    PIBAQ,
    FAMILY_ID,
    EVIDENCE_LEVEL,
    FAMILY_SIZE,
    MOLECULARWEIGHT,
    TPA,
]


def _group_columns(
    peptide_df: DataFrame, extra_group_cols: Optional[Sequence[str]]
) -> list[str]:
    """Resolve the sample, condition, and caller-supplied grouping columns."""
    optional = (CONDITION, *(extra_group_cols or ()))
    retained = dict.fromkeys(
        column
        for column in optional
        if column in peptide_df.columns and column != SAMPLE_ID
    )
    return [SAMPLE_ID, *retained]


def _native_observations(
    peptide_df: DataFrame,
    peptide_column: str,
    group_columns: Sequence[str],
) -> tuple[list[tuple[str, str, float]], DataFrame]:
    """Encode arbitrary DataFrame groups as native sample identifiers."""
    working = peptide_df.copy()
    group_ids = working.groupby(
        list(group_columns), dropna=False, observed=True
    ).ngroup()
    working[_GROUP_ID] = group_ids.astype("int64").astype(str)
    observations = [
        (str(peptide), str(group_id), float(intensity))
        for peptide, group_id, intensity in working[
            [peptide_column, _GROUP_ID, NORM_INTENSITY]
        ].itertuples(index=False, name=None)
    ]
    group_frame = working[[_GROUP_ID, *group_columns]].drop_duplicates(_GROUP_ID)
    return observations, group_frame


def _native_result(
    rows: Sequence[tuple],
    groups: DataFrame,
    group_columns: Sequence[str],
    families: Sequence[Family],
    include_tpa: bool,
) -> DataFrame:
    """Restore DataFrame grouping columns and canonical output ordering."""
    if not rows:
        return _empty_pibaq_frame(group_columns, include_tpa)
    result = DataFrame.from_records(rows, columns=_NATIVE_COLUMNS)
    result = result.merge(groups, on=_GROUP_ID, how="left", validate="many_to_one")
    family_order = {family.family_id: order for order, family in enumerate(families)}
    result[_FAMILY_ORDER] = result[FAMILY_ID].map(family_order)
    result[_GROUP_ID] = result[_GROUP_ID].astype("int64")
    result = result.sort_values(
        [_FAMILY_ORDER, PROTEIN_NAME, _GROUP_ID], kind="stable"
    ).reset_index(drop=True)
    columns = [
        PROTEIN_NAME,
        *group_columns,
        NORM_INTENSITY,
        PIBAQ,
        FAMILY_ID,
        EVIDENCE_LEVEL,
        FAMILY_SIZE,
    ]
    if include_tpa:
        columns.extend((MOLECULARWEIGHT, TPA))
    return result[columns]


def _run_native_core(
    options: Mapping[str, object],
    families: Sequence[Family],
    observed: DataFrame,
    peptide_column: str,
    group_columns: Sequence[str],
) -> tuple[Sequence[tuple], DataFrame]:
    """Call the extension and retain the encoded group lookup."""
    observations, groups = _native_observations(observed, peptide_column, group_columns)
    native_compute = getattr(importlib.import_module("mokume._mokume"), "compute_pibaq")
    rows = native_compute(
        observations,
        dict(options["accession_to_peptides"]),
        dict(options["peptide_to_accessions"]),
        [(family.family_id, list(family.members)) for family in families],
        (
            dict(options["mw_map"]) if options["mw_map"] is not None else None,
            options["min_anchors"],
            options["high_anchor_threshold"],
        ),
    )
    return rows, groups


def _log_pibaq_summary(result: DataFrame) -> None:
    """Report family resolution counts for the compatibility API."""
    processed = result[FAMILY_ID].nunique() if not result.empty else 0
    unresolved = (
        result.loc[result[EVIDENCE_LEVEL] == EVIDENCE_FAMILY_ONLY, FAMILY_ID].nunique()
        if processed
        else 0
    )
    logger.info(
        "native piBAQ resolved %d families: %d family-only and %d member-resolving",
        processed,
        unresolved,
        processed - unresolved,
    )


def compute_pibaq(*args: object, **kwargs: object) -> DataFrame:
    """Compute paralog-aware iBAQ through the native Rust allocation core."""
    options = _bind(_COMPUTE_SIGNATURE, args, kwargs)
    peptide_df = options["peptide_df"]
    families = list(options["families"])
    group_columns = _group_columns(peptide_df, options["extra_group_cols"])
    include_tpa = options["mw_map"] is not None
    if peptide_df.empty or not families:
        return _empty_pibaq_frame(group_columns, include_tpa)

    peptide_column = _detect_peptide_column(peptide_df)
    peptide_to_accessions = options["peptide_to_accessions"]
    observed = peptide_df[
        _membership_mask(peptide_df[peptide_column], peptide_to_accessions)
    ]
    if observed.empty:
        return _empty_pibaq_frame(group_columns, include_tpa)
    rows, groups = _run_native_core(
        options, families, observed, peptide_column, group_columns
    )
    result = _native_result(rows, groups, group_columns, families, include_tpa)
    _log_pibaq_summary(result)
    return result


compute_pibaq.__signature__ = _COMPUTE_SIGNATURE


_PEPTIDES_SIGNATURE = _public_signature(
    "fasta peptides enzyme normalize min_aa max_aa tpa ruler ploidy cpc "
    "organism output verbose qc_report".split(),
    {
        "families_yaml": None,
        "min_shared": 2,
        "min_anchors": 1,
        "high_anchor_threshold": 3,
    },
)


def _native_pibaq_options(bound: Mapping[str, object]) -> dict[str, object]:
    """Translate the legacy Python call into native command options."""
    names = (
        "fasta peptides enzyme normalize min_aa max_aa tpa ruler ploidy cpc "
        "organism output min_shared min_anchors high_anchor_threshold"
    ).split()
    options = {name: bound[name] for name in names}
    options["quant_method"] = "pibaq"
    if bound["families_yaml"] is not None:
        options["families"] = bound["families_yaml"]
    return options


def _render_native_qc(module: ModuleType, bound: Mapping[str, object]) -> None:
    """Render the legacy verbose report from the native result table."""
    if not bound["verbose"]:
        return
    if not is_plotting_available():
        logger.warning(
            "QC report skipped: plotting dependencies not installed. "
            "Install: pip install mokume[analysis]"
        )
        return
    qc_report = getattr(module, "peptides2protein_qc")
    qc_report(
        protein_table=bound["output"],
        qc_report=bound["qc_report"],
        plot_column=PIBAQ_PPB if bound["normalize"] else PIBAQ,
        tpa=bound["tpa"],
        ruler=bound["ruler"],
    )


@log_execution_time(logger)
def peptides_to_protein(*args: object, **kwargs: object) -> None:
    """Run the legacy file-oriented entrypoint through native Rust piBAQ."""
    bound = _bind(_PEPTIDES_SIGNATURE, args, kwargs)
    module = importlib.import_module("mokume")
    getattr(module, "peptides2protein")(**_native_pibaq_options(bound))
    _render_native_qc(module, bound)


peptides_to_protein.__signature__ = _PEPTIDES_SIGNATURE
