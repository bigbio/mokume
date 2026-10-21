"""File-oriented piBAQ workflow around the allocation calculation core."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Optional, Tuple

import pandas as pd
from pandas import DataFrame

from mokume.core.constants import (
    CONCENTRATION_NM,
    COPYNUMBER,
    NORM_INTENSITY,
    PIBAQ,
    PIBAQ_NORMALIZED,
    PIBAQ_PPB,
    SAMPLE_ID,
    TPA,
    is_parquet,
)
from mokume.core.logger import get_logger, log_execution_time
from mokume.io.fasta import digest_fasta_full
from mokume.model.organism import OrganismDescription
from mokume.plotting import is_plotting_available
from mokume.quantification.families import (
    Family,
    discover_families,
    load_families_yaml,
    merge_overrides,
)

logger = get_logger("mokume.quantification.pibaq")


@dataclass(frozen=True)
class _PibaqDigestRequest:
    fasta: str
    enzyme: str
    min_aa: int
    max_aa: int
    tpa: bool


@dataclass(frozen=True)
class _PibaqFamilyRequest:
    families_yaml: Optional[Path]
    min_shared: int
    min_anchors: int
    high_anchor_threshold: int


@dataclass(frozen=True)
class _PibaqTableRequest:
    digest: _PibaqDigestRequest
    family: _PibaqFamilyRequest


@dataclass(frozen=True)
class _PeptideInputRequest:
    fasta: str
    peptides: str
    enzyme: str
    min_aa: int
    max_aa: int


@dataclass(frozen=True)
class _PibaqPostprocessRequest:
    normalize: bool
    tpa: bool
    ruler: bool
    ploidy: int
    cpc: float
    organism: str


@dataclass(frozen=True)
class _PibaqOutputRequest:
    output: str
    verbose: bool
    qc_report: str


@dataclass(frozen=True)
class _CommandFamilyRequest:
    families_yaml: Optional[str]
    min_shared: int
    min_anchors: int
    high_anchor_threshold: int


@dataclass(frozen=True)
class _PeptidesToProteinRequest:
    source: _PeptideInputRequest
    postprocess: _PibaqPostprocessRequest
    output: _PibaqOutputRequest
    family: _CommandFamilyRequest


@dataclass(frozen=True)
class _PlotContext:
    data: DataFrame
    plotting: ModuleType
    pdf: object
    width: float


def _pibaq_module() -> ModuleType:
    """Resolve the calculation module lazily to avoid an import cycle."""
    return importlib.import_module("mokume.quantification.pibaq")


def _resolve_families(
    accession_to_peptides,
    peptide_to_accessions,
    families_yaml: Optional[Path],
    min_shared: int,
) -> list[Family]:
    """Combine automatic connected components with optional YAML overrides."""
    automatic = discover_families(
        accession_to_peptides, peptide_to_accessions, min_shared=min_shared
    )
    if families_yaml is None:
        return automatic
    overrides = load_families_yaml(families_yaml)
    return merge_overrides(automatic, overrides)


def _compute_pibaq_table(data: DataFrame, request: _PibaqTableRequest) -> DataFrame:
    """Digest FASTA, resolve families, and dispatch to the piBAQ core."""
    digest = request.digest
    family_request = request.family
    accession_to_peptides, peptide_to_accessions, accession_to_mw = digest_fasta_full(
        digest.fasta,
        digest.enzyme,
        digest.min_aa,
        digest.max_aa,
        canonicalize_isoforms=True,
        compute_mw=digest.tpa,
    )
    families = _resolve_families(
        accession_to_peptides,
        peptide_to_accessions,
        family_request.families_yaml,
        family_request.min_shared,
    )
    mw_map: Optional[Mapping[str, float]] = accession_to_mw if digest.tpa else None
    compute_pibaq = getattr(_pibaq_module(), "compute_pibaq")
    return compute_pibaq(
        data,
        accession_to_peptides,
        peptide_to_accessions,
        families,
        mw_map=mw_map,
        min_anchors=family_request.min_anchors,
        high_anchor_threshold=family_request.high_anchor_threshold,
    )


_PEPTIDES_TO_PROTEIN_ARGUMENTS = (
    "fasta",
    "peptides",
    "enzyme",
    "normalize",
    "min_aa",
    "max_aa",
    "tpa",
    "ruler",
    "ploidy",
    "cpc",
    "organism",
    "output",
    "verbose",
    "qc_report",
)
_PEPTIDES_TO_PROTEIN_DEFAULTS = {
    "families_yaml": None,
    "min_shared": 2,
    "min_anchors": 1,
    "high_anchor_threshold": 3,
}


@log_execution_time(logger)
def peptides_to_protein(*args: object, **kwargs: object) -> None:
    """Compute file-oriented piBAQ output with the legacy public arguments.

    The first fourteen parameters retain their positional-or-keyword contract:
    ``fasta``, ``peptides``, ``enzyme``, ``normalize``, ``min_aa``, ``max_aa``,
    ``tpa``, ``ruler``, ``ploidy``, ``cpc``, ``organism``, ``output``,
    ``verbose``, and ``qc_report``. ``families_yaml``, ``min_shared``,
    ``min_anchors``, and ``high_anchor_threshold`` remain keyword-only with
    their historical defaults.
    """
    bind_arguments = getattr(_pibaq_module(), "_bind_arguments")
    bound = bind_arguments(
        "peptides_to_protein",
        args,
        kwargs,
        _PEPTIDES_TO_PROTEIN_ARGUMENTS,
        _PEPTIDES_TO_PROTEIN_DEFAULTS,
    )
    _run_peptides_to_protein(_build_peptides_request(bound))


def _build_peptides_request(
    bound: Mapping[str, object],
) -> _PeptidesToProteinRequest:
    """Group the bound legacy arguments into focused immutable requests."""
    return _PeptidesToProteinRequest(
        source=_PeptideInputRequest(
            fasta=bound["fasta"],
            peptides=bound["peptides"],
            enzyme=bound["enzyme"],
            min_aa=bound["min_aa"],
            max_aa=bound["max_aa"],
        ),
        postprocess=_PibaqPostprocessRequest(
            normalize=bound["normalize"],
            tpa=bound["tpa"],
            ruler=bound["ruler"],
            ploidy=bound["ploidy"],
            cpc=bound["cpc"],
            organism=bound["organism"],
        ),
        output=_PibaqOutputRequest(
            output=bound["output"],
            verbose=bound["verbose"],
            qc_report=bound["qc_report"],
        ),
        family=_CommandFamilyRequest(
            families_yaml=bound["families_yaml"],
            min_shared=bound["min_shared"],
            min_anchors=bound["min_anchors"],
            high_anchor_threshold=bound["high_anchor_threshold"],
        ),
    )


def _resolve_organism(name: str) -> Optional[OrganismDescription]:
    """Resolve an optional organism name with the legacy error contract."""
    if not name:
        return None
    description = OrganismDescription.get(name)
    if description is None:
        raise KeyError(f"Could not resolve organism description for {name}")
    return description


def _validate_ruler_request(request: _PeptidesToProteinRequest) -> None:
    """Validate the four inputs required by the proteomic ruler."""
    options = request.postprocess
    if options.ruler and (
        not options.tpa
        or options.ploidy <= 0
        or options.cpc <= 0
        or not options.organism
    ):
        raise ValueError(
            "Proteomic ruler requires --tpa, a positive ploidy/CPC, and an organism"
        )
    uses_cli_defaults = (
        options.ploidy == 2 and options.cpc == 200 and options.organism == "human"
    )
    uses_legacy_sentinels = (
        options.ploidy == 0 and options.cpc == 0 and not options.organism
    )
    if not options.ruler and not (uses_cli_defaults or uses_legacy_sentinels):
        raise ValueError("ploidy, CPC, and organism only apply to proteomic ruler")


def _load_peptide_data(path: str) -> DataFrame:
    """Load and retain positive finite peptide intensities."""
    data = pd.read_parquet(path) if is_parquet(path) else pd.read_csv(path)
    data[NORM_INTENSITY] = data[NORM_INTENSITY].astype(float)
    data = data.dropna(subset=[NORM_INTENSITY])
    return data[data[NORM_INTENSITY] > 0]


def _table_request(request: _PeptidesToProteinRequest) -> _PibaqTableRequest:
    """Project public command options onto the piBAQ table calculation."""
    source = request.source
    family = request.family
    return _PibaqTableRequest(
        digest=_PibaqDigestRequest(
            fasta=source.fasta,
            enzyme=source.enzyme,
            min_aa=source.min_aa,
            max_aa=source.max_aa,
            tpa=request.postprocess.tpa,
        ),
        family=_PibaqFamilyRequest(
            families_yaml=Path(family.families_yaml) if family.families_yaml else None,
            min_shared=family.min_shared,
            min_anchors=family.min_anchors,
            high_anchor_threshold=family.high_anchor_threshold,
        ),
    )


def _postprocess_pibaq(
    data: DataFrame,
    request: _PeptidesToProteinRequest,
    organism: Optional[OrganismDescription],
) -> Tuple[DataFrame, str]:
    """Apply the legacy normalization and optional ruler post-processing."""
    options = request.postprocess
    if options.normalize:
        normalize_pibaq = getattr(_pibaq_module(), "normalize_pibaq")
        data = normalize_pibaq(data).dropna(subset=[PIBAQ_NORMALIZED])
        plot_column = PIBAQ_PPB
    else:
        data = data.dropna(subset=[PIBAQ])
        plot_column = PIBAQ
    data = data.reset_index(drop=True)
    if options.ruler:
        ruler_class = getattr(_pibaq_module(), "ConcentrationWeightByProteomicRuler")
        ruler = ruler_class(organism, options.ploidy, options.cpc)
        data = ruler.apply_by_condition(data)
    return data, plot_column


def _plot_metric_pair(context: _PlotContext, column: str) -> None:
    """Write density and box plots for one piBAQ-derived metric."""
    title = f"{column} Distribution"
    distributions = getattr(context.plotting, "plot_distributions")
    box_plot = getattr(context.plotting, "plot_box_plot")
    plot_options = {
        "log2": True,
        "width": context.width,
        "title": title,
    }
    density = distributions(
        context.data,
        column,
        SAMPLE_ID,
        **plot_options,
    )
    box = box_plot(
        context.data,
        column,
        SAMPLE_ID,
        violin=False,
        **plot_options,
    )
    savefig = getattr(context.pdf, "savefig")
    savefig(density, bbox_inches="tight")
    savefig(box, bbox_inches="tight")


def _plot_pibaq_report(
    data: DataFrame, plot_column: str, request: _PeptidesToProteinRequest
) -> None:
    """Render the optional QC report without importing plotting eagerly."""
    if not request.output.verbose:
        return
    if not is_plotting_available():
        logger.warning(
            "QC report skipped: plotting dependencies not installed. "
            "Install: pip install mokume-py[plotting] (pure Python) or "
            "pip install mokume[analysis] (Rust wheel)"
        )
        return
    plotting = importlib.import_module("mokume.plotting")
    pdf = getattr(plotting, "PdfPages")(request.output.qc_report)
    context = _PlotContext(
        data=data,
        plotting=plotting,
        pdf=pdf,
        width=len(set(data[SAMPLE_ID])) * 0.5 + 10,
    )
    columns = [plot_column]
    if request.postprocess.tpa:
        columns.append(TPA)
    if request.postprocess.ruler:
        columns.extend((COPYNUMBER, CONCENTRATION_NM))
    for column in columns:
        _plot_metric_pair(context, column)
    getattr(pdf, "close")()


def _run_peptides_to_protein(request: _PeptidesToProteinRequest) -> None:
    """Execute the legacy driver after binding its public arguments."""
    _validate_ruler_request(request)
    organism = (
        _resolve_organism(request.postprocess.organism)
        if request.postprocess.ruler
        else None
    )
    data = _load_peptide_data(request.source.peptides)
    result = _compute_pibaq_table(data, _table_request(request))
    result, plot_column = _postprocess_pibaq(result, request, organism)
    _plot_pibaq_report(result, plot_column, request)
    result.to_csv(request.output.output, sep="\t", index=False)
