"""
CLI command for computing protein quantification values.
"""

import logging
import re

import click
import pandas as pd

from mokume.quantification.pibaq import peptides_to_protein
from mokume.quantification import (
    get_quantification_method,
    is_directlfq_available,
)
from mokume.model.organism import OrganismDescription
from mokume.core.constants import (
    PROTEIN_NAME,
    SAMPLE_ID,
    CONDITION,
    NORM_INTENSITY,
    PEPTIDE_CANONICAL,
    is_parquet,
)

logger = logging.getLogger(__name__)


# Methods with a fixed name. The TopN family is not listed here: ``top<N>``
# already covers ``top3``, and spelling both would advertise one method twice.
# QuantMethodParam accepts any ``top<N>``; the help text names top3 as the
# common choice.
QUANTIFICATION_METHODS = ["pibaq", "maxlfq", "sum", "directlfq"]

# A ``top<N>`` method name carries its own N (top1, top3, top5, top10, ...).
TOPN_METHOD_RE = re.compile(r"top(\d+)")


def _nonzero_threads(_ctx, _param, value):
    if value == 0:
        raise click.BadParameter("must be a non-zero integer")
    return value


def _supplied(ctx: click.Context, *names: str) -> bool:
    return any(
        ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
        for name in names
    )


def _validate_lfq_options(ctx: click.Context, method: str) -> None:
    if method != "maxlfq" and _supplied(ctx, "threads"):
        raise click.UsageError("--threads only applies to --method maxlfq")
    if method != "directlfq" and _supplied(ctx, "min_nonan"):
        raise click.UsageError("--min_nonan only applies to --method directlfq")
    if method != "maxlfq" and ctx.params.get("stabilize", False):
        raise click.UsageError("--stabilize only applies to --method maxlfq")
    if method != "maxlfq" and _supplied(ctx, "maxlfq_min_ratio_count"):
        raise click.UsageError(
            "--maxlfq-min-ratio-count only applies to --method maxlfq"
        )


def _validate_method_options(
    ctx: click.Context,
    method: str,
    *,
    verbose: bool,
    tpa: bool,
    ruler: bool,
) -> None:
    pibaq_options = (
        "fasta",
        "enzyme",
        "min_aa",
        "max_aa",
        "tpa",
        "ruler",
        "organism",
        "ploidy",
        "cpc",
        "verbose",
        "qc_report",
        "families_yaml",
        "min_shared",
        "min_anchors",
        "high_anchor_threshold",
    )
    if method != "pibaq" and _supplied(ctx, *pibaq_options):
        raise click.UsageError(
            "piBAQ digestion/TPA/ruler/QC options require --method pibaq"
        )
    _validate_lfq_options(ctx, method)
    if method == "pibaq" and _supplied(ctx, "qc_report") and not verbose:
        raise click.UsageError("--qc_report requires --verbose")
    if ruler and not tpa:
        raise click.UsageError("--ruler requires --tpa")
    if not ruler and _supplied(ctx, "ploidy", "organism", "cpc"):
        raise click.UsageError("--ploidy/--organism/--cpc require --ruler")


def get_available_methods():
    """Get list of available quantification methods based on installed packages."""
    methods = ["pibaq", "top3", "maxlfq", "sum"]
    if is_directlfq_available():
        methods.append("directlfq")
    return methods


class QuantMethodParam(click.ParamType):
    """A ``click.Choice`` that additionally accepts any ``top<N>`` method name.

    Fixed names are matched case-insensitively against ``methods``. On top of
    those, ``top<N>`` for any integer N >= 1 is accepted: the method name is the
    only place N is spelled (``top5`` means Top 5). The converted value is
    always lower-cased.
    """

    name = "quant_method"

    def __init__(self, methods):
        self.methods = [m.lower() for m in methods]

    def get_metavar(self, *_args, **_kwargs):
        return "[" + "|".join([*self.methods, "top<N>"]) + "]"

    def convert(self, value, param, ctx):
        value_lower = str(value).lower()
        if value_lower in self.methods:
            return value_lower
        # ``topn`` keeps the placeholder letter and means the canonical Top3, so
        # it normalizes here and downstream only ever sees ``top<digits>``.
        if value_lower == "topn":
            return "top3"
        match = TOPN_METHOD_RE.fullmatch(value_lower)
        if match and int(match.group(1)) >= 1:
            return value_lower
        self.fail(
            f"{value!r} is not a valid quantification method. Choose from "
            f"{', '.join(self.methods)}, or top<N> for any N >= 1 (e.g. top5).",
            param,
            ctx,
        )


@click.command("peptides2protein", short_help="Compute protein quantification values")
@click.option(
    "-f",
    "--fasta",
    help="Protein database used to compute piBAQ values (required for pibaq)",
    type=click.Path(exists=True),
)
@click.option(
    "-p",
    "--peptides",
    help="Peptide identifications with intensities following the peptide intensity output",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--method",
    help=(
        "Quantification method to use: pibaq, top<N> (top1/top3/top5/...), "
        "maxlfq, sum, directlfq (directlfq requires: pip install mokume-py[directlfq])"
    ),
    type=QuantMethodParam(QUANTIFICATION_METHODS),
    default="pibaq",
)
@click.option(
    "-e",
    "--enzyme",
    help="Enzyme used during the analysis of the dataset (default: Trypsin)",
    default="Trypsin",
)
@click.option(
    "-n",
    "--normalize",
    help="Normalize quantification values",
    is_flag=True,
)
@click.option(
    "--min_aa", help="Minimum number of amino acids to consider a peptide", default=7
)
@click.option(
    "--max_aa", help="Maximum number of amino acids to consider a peptide", default=30
)
@click.option(
    "-t", "--tpa", help="Whether to calculate TPA (piBAQ method only)", is_flag=True
)
@click.option(
    "-r",
    "--ruler",
    help="Whether to use ProteomicRuler (piBAQ method only)",
    is_flag=True,
)
@click.option("-i", "--ploidy", help="Ploidy number (default: 2)", default=2)
@click.option(
    "-m",
    "--organism",
    help="Organism source of the data (default: human)",
    type=click.Choice(
        sorted(map(str.lower, OrganismDescription.registered_organisms())),
        case_sensitive=False,
    ),
    default="human",
)
@click.option(
    "-c",
    "--cpc",
    help="Cellular protein concentration(g/L) (default: 200)",
    default=200,
)
@click.option(
    "-o",
    "--output",
    help="Output file with the proteins and quantification values",
    required=True,
)
@click.option(
    "--verbose",
    help="Print additional information about the distributions of the intensities",
    is_flag=True,
)
@click.option(
    "--qc_report",
    help="PDF file to store multiple QC images (piBAQ method only)",
    default="QCprofile.pdf",
)
@click.option(
    "--threads",
    help="Number of parallel threads for MaxLFQ (-1 for all cores, default: -1)",
    default=-1,
    type=int,
    callback=_nonzero_threads,
)
@click.option(
    "--stabilize",
    is_flag=True,
    default=False,
    help="Enable large-ratio stabilization (MaxLFQ only; default: off)",
)
@click.option(
    "--maxlfq-min-ratio-count",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Minimum shared peptide species for each MaxLFQ sample-pair ratio.",
)
@click.option(
    "--min_nonan",
    help="Minimum non-NaN ion intensities per protein for DirectLFQ (default: 1)",
    default=1,
    type=int,
)
@click.option(
    "--families",
    "families_yaml",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Optional YAML file declaring explicit protein family overrides "
        "for piBAQ (paralog-aware iBAQ; schema: families: [{name, members}])."
    ),
)
@click.option(
    "--min-shared",
    "min_shared",
    type=int,
    default=2,
    show_default=True,
    help=(
        "Minimum number of distinct peptides two proteins must share to be "
        "automatically grouped into the same piBAQ family."
    ),
)
@click.option(
    "--min-anchors",
    "min_anchors",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Unique-anchor threshold; if no family member reaches it, shared "
        "signal is split equally (piBAQ only)."
    ),
)
@click.option(
    "--high-anchor-threshold",
    "high_anchor_threshold",
    type=int,
    default=3,
    show_default=True,
    help=(
        "Minimum anchor count (weakest member) for a family to be labelled "
        "EvidenceLevel='high' (piBAQ only)."
    ),
)
@click.pass_context
def peptides2protein(
    ctx: click.Context,
    fasta: str,
    peptides: str,
    method: str,
    enzyme: str,
    normalize: bool,
    min_aa: int,
    max_aa: int,
    tpa: bool,
    ruler: bool,
    organism: str,
    ploidy: int,
    cpc: float,
    output: str,
    verbose: bool,
    qc_report: str,
    threads: int,
    maxlfq_min_ratio_count: int,
    stabilize: bool,
    min_nonan: int,
    families_yaml: str,
    min_shared: int,
    min_anchors: int,
    high_anchor_threshold: int,
) -> None:
    """
    Compute protein quantification values from peptide intensity data.

    This command processes peptide identifications and computes protein
    quantification values using various methods:

    \b
    - pibaq: Paralog-aware iBAQ with shared-peptide allocation (default, requires FASTA)
    - top<N>: Average of the N most intense peptides (top1, top3, top5, top10, ...)
    - maxlfq: Pairwise peptide ratios and least-squares protein quantification
    - sum: Sum of all peptide intensities
    - directlfq: DirectLFQ intensity traces (requires: pip install mokume-py[directlfq])

    For piBAQ, a FASTA file is required. Other methods can work
    without a FASTA file.

    \b
    Examples:
        # Using piBAQ (requires FASTA)
        mokume peptides2protein --method pibaq -f proteome.fasta -p peptides.csv -o proteins.tsv

        # Using MaxLFQ with 4 threads
        mokume peptides2protein --method maxlfq --threads 4 -p peptides.csv -o proteins.tsv

        # Using DirectLFQ (requires optional install)
        mokume peptides2protein --method directlfq -p peptides.csv -o proteins.tsv

        # Using Top5 (N comes from the method name)
        mokume peptides2protein --method top5 -p peptides.csv -o proteins.tsv
    """
    method_lower = method.lower()
    _validate_method_options(
        ctx,
        method_lower,
        verbose=verbose,
        tpa=tpa,
        ruler=ruler,
    )

    # Check DirectLFQ availability
    if method_lower == "directlfq" and not is_directlfq_available():
        raise click.UsageError(
            "DirectLFQ is not installed. Install with: pip install mokume-py[directlfq]"
        )

    if method_lower == "pibaq":
        if not fasta:
            raise click.UsageError("The --fasta option is required for piBAQ")

        peptides_to_protein(
            fasta=fasta,
            peptides=peptides,
            enzyme=enzyme,
            normalize=normalize,
            min_aa=min_aa,
            max_aa=max_aa,
            tpa=tpa,
            ruler=ruler,
            ploidy=ploidy,
            cpc=cpc,
            organism=organism,
            output=output,
            verbose=verbose,
            qc_report=qc_report,
            families_yaml=families_yaml,
            min_shared=min_shared,
            min_anchors=min_anchors,
            high_anchor_threshold=high_anchor_threshold,
        )
    else:
        # Use the generic quantification methods
        click.echo(f"Using {method} quantification method")

        # Load peptide data
        if is_parquet(peptides):
            peptide_df = pd.read_parquet(peptides)
        else:
            peptide_df = pd.read_csv(peptides)

        click.echo(f"Loaded {len(peptide_df)} peptide measurements")

        # Get the quantification method with appropriate parameters
        if TOPN_METHOD_RE.fullmatch(method_lower):
            # ``top<N>`` carries N in the name; the engine reads it back out.
            quant_method = get_quantification_method(method_lower)
        elif method_lower == "maxlfq":
            quant_method = get_quantification_method(
                method,
                threads=threads,
                min_ratio_count=maxlfq_min_ratio_count,
                stabilize=stabilize,
            )
        elif method_lower == "directlfq":
            quant_method = get_quantification_method(method, min_nonan=min_nonan)
        else:
            quant_method = get_quantification_method(method)

        # Determine column names (try to auto-detect)
        protein_col = (
            PROTEIN_NAME if PROTEIN_NAME in peptide_df.columns else "ProteinName"
        )
        sample_col = SAMPLE_ID if SAMPLE_ID in peptide_df.columns else "SampleID"
        intensity_col = (
            NORM_INTENSITY if NORM_INTENSITY in peptide_df.columns else "NormIntensity"
        )
        peptide_col = (
            PEPTIDE_CANONICAL
            if PEPTIDE_CANONICAL in peptide_df.columns
            else "PeptideSequence"
        )

        # Check for required columns
        for col, name in [
            (protein_col, "protein"),
            (sample_col, "sample"),
            (intensity_col, "intensity"),
        ]:
            if col not in peptide_df.columns:
                raise click.UsageError(
                    f"Could not find {name} column '{col}' in peptide file"
                )

        # Run quantification
        click.echo(f"Quantifying {peptide_df[protein_col].nunique()} proteins...")
        result_df = quant_method.quantify(
            peptide_df,
            protein_column=protein_col,
            peptide_column=peptide_col,
            intensity_column=intensity_col,
            sample_column=sample_col,
        )

        # Add condition if available
        if CONDITION in peptide_df.columns:
            condition_map = (
                peptide_df[[sample_col, CONDITION]]
                .drop_duplicates()
                .set_index(sample_col)[CONDITION]
            )
            result_df[CONDITION] = result_df[sample_col].map(condition_map)

        # Normalize if requested
        if normalize:
            # Find the intensity column (last column that contains 'Intensity')
            intensity_cols = [c for c in result_df.columns if "Intensity" in c]
            if intensity_cols:
                intensity_col_out = intensity_cols[-1]
                result_df[f"{intensity_col_out}Norm"] = result_df.groupby(sample_col)[
                    intensity_col_out
                ].transform(lambda x: x / x.sum())

        # Save output
        if output.endswith(".parquet"):
            result_df.to_parquet(output, index=False)
        else:
            result_df.to_csv(output, sep="\t", index=False)
        click.echo(f"Results saved to {output}")
