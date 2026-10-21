"""
Pipeline configuration dataclasses.

This module provides nested configuration dataclasses for the
quantification pipeline, replacing the flat PipelineConfig.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from mokume.analysis.ensemble_config import validate_de_ensemble

_MEMORY_LIMIT_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*[KMGT]?B?\s*$", re.IGNORECASE)
_MEMORY_UNIT_BYTES: dict[str, int] = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
}


def _parse_memory_string_to_gib(value: str) -> float:
    """Parse a DuckDB-style memory string (``"80GB"``, ``"16384MB"``) to GiB."""
    match = re.match(r"^\s*([\d.]+)\s*([KMGT]?B?)\s*$", str(value), re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse memory string: {value!r}")
    qty = float(match.group(1))
    unit_bytes = _MEMORY_UNIT_BYTES[match.group(2).upper()]
    return qty * unit_bytes / (1024**3)


@dataclass
class InputConfig:
    """Input file paths."""

    parquet: Optional[str] = None
    sdrf: Optional[str] = None
    fasta_file: Optional[str] = None
    qpx_dir: Optional[str] = None
    msstats: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.parquet is None) == (self.msstats is None):
            raise ValueError("Provide exactly one input: parquet or msstats")
        if self.msstats is not None and self.sdrf is None:
            raise ValueError("MSstats input requires an SDRF file")


@dataclass
class RuntimeConfig:
    """DuckDB resource limits propagated through to the parquet engine.

    These map onto DuckDB's ``SET memory_limit=...`` and ``SET threads=...``
    pragmas applied during :class:`mokume.io.feature.Feature` initialisation.
    Leaving both ``None`` keeps DuckDB's defaults (memory limit ~80% of total
    RAM, threads = number of cores).

    .. warning::
        ``duckdb_memory`` **only caps the DuckDB engine's internal buffer
        pool**. It does *not* limit the total Python process RSS, because
        PyArrow, polars, and pandas each have their own independent
        allocators (mimalloc / jemalloc / system malloc). Peak process RSS
        can easily reach 2-3x this cap on wide pivots. For hard
        process-level caps in production, use cgroup ``MemoryMax`` (systemd),
        SLURM ``--mem``, or container ``resources.limits.memory``.

    Attributes
    ----------
    duckdb_memory : Optional[str]
        Memory limit string accepted by DuckDB, e.g. ``"80GB"`` or
        ``"16384MB"``. Sets the ``memory_limit`` pragma on every DuckDB
        connection opened by :class:`mokume.io.feature.Feature`.
    duckdb_threads : Optional[int]
        Number of threads DuckDB may use. Must be >= 1.
    """

    duckdb_memory: Optional[str] = None
    duckdb_threads: Optional[int] = None

    def __post_init__(self) -> None:
        if self.duckdb_memory is not None and not _MEMORY_LIMIT_RE.match(
            self.duckdb_memory
        ):
            raise ValueError(
                "duckdb_memory must be a DuckDB memory string like '80GB' "
                f"or '16384MB'; got {self.duckdb_memory!r}"
            )
        if self.duckdb_threads is not None and self.duckdb_threads < 1:
            raise ValueError(f"duckdb_threads must be >= 1; got {self.duckdb_threads}")

    def effective_workers(
        self,
        default: Optional[int] = None,
        per_worker_gb: float = 10.0,
        os_reserve_gb: float = 5.0,
        system_total_gb: Optional[float] = None,
    ) -> Optional[int]:
        """Derive a soft worker-count budget for fork-based parallel sections
        (e.g. DirectLFQ's protein-estimation pool, MaxLFQ's ``joblib.Parallel``)
        from the user-facing ``duckdb_memory`` and ``duckdb_threads`` hints.

        Each forked worker COW-copies the parent's wide pivot table. The parent
        process itself grows to at least ``duckdb_memory`` (DuckDB buffer
        pool) plus the in-memory Arrow / polars / pandas tables on top, so
        the memory left for workers is ``system_total - duckdb_memory -
        os_reserve``. On the PXD030304 run a previous heuristic that budgeted
        off ``duckdb_memory`` returned 7 workers; the parent held ~70 GB
        while 7 workers each grew to ~10 GB anon-rss, totalling ~140 GB and
        OOM-killing the 125 GB host.

        Parameters
        ----------
        default : Optional[int]
            Returned when neither ``duckdb_memory`` nor ``duckdb_threads`` is
            set. Pass ``None`` to let the downstream library use its own
            default (``cpu_count()`` for DirectLFQ).
        per_worker_gb : float
            Conservative estimate of each fork worker's COW-amplified
            resident set growth. Tuned for DirectLFQ on cell-lines DIA
            inputs; callers with different workloads can override.
        os_reserve_gb : float
            GiB reserved for the OS, page cache, and other processes on the
            host before any worker is scheduled.
        system_total_gb : Optional[float]
            Total system RAM in GiB. When ``None`` (the default), it is
            probed with :mod:`psutil`. Pass an explicit value for
            reproducible tests or when the heuristic should ignore the
            actual host.

        Returns
        -------
        Optional[int]
            Worker-count budget (>= 1 when any hint is set), or ``default``
            when both hints are absent.
        """
        mem_budget: Optional[int] = None
        if self.duckdb_memory is not None:
            parent_gib = _parse_memory_string_to_gib(self.duckdb_memory)
            total_gib = system_total_gb
            if total_gib is None:
                try:
                    import psutil

                    total_gib = psutil.virtual_memory().total / (1024**3)
                except ImportError:
                    total_gib = None
            if total_gib is not None:
                # Parent fills up to duckdb_memory; workers share what is
                # left after a small OS reserve.
                usable = max(per_worker_gb, total_gib - parent_gib - os_reserve_gb)
                mem_budget = max(1, int(usable // per_worker_gb))
            else:
                # No system probe available: fall back to a conservative
                # duckdb_memory-based budget so we never accidentally let
                # cpu_count workers loose under a memory cap.
                usable = max(per_worker_gb, parent_gib - os_reserve_gb)
                mem_budget = max(1, int(usable // per_worker_gb))

        if mem_budget is None and self.duckdb_threads is None:
            return default
        if mem_budget is None:
            return self.duckdb_threads
        if self.duckdb_threads is None:
            return mem_budget
        return min(mem_budget, self.duckdb_threads)


@dataclass
class FilterConfig:
    """Peptide/protein filtering parameters."""

    min_aa: int = 7
    min_unique_peptides: int = 2
    remove_contaminants: bool = True


@dataclass
class NormalizationConfig:
    """Normalization method parameters.

    Attributes
    ----------
    run_method : str
        Run/technical replicate normalization method.
    sample_method : str
        Sample-to-sample normalization method. Supported values:
        ``none``, ``globalmedian``, ``conditionmedian``, ``hierarchical``,
        ``quantile``, ``mediancenter``, ``meancenter``, ``loess``,
        ``rlr`` (NormalyzerDE Robust Linear Regression).
    proteins_file : str, optional
        File path containing protein IDs to subset normalization to.
    """

    run_method: str = "median"
    sample_method: str = "globalMedian"
    proteins_file: Optional[str] = None


@dataclass
class QuantificationConfig:
    """Quantification method parameters.

    Attributes
    ----------
    pibaq_enzyme : str
        Protease used to digest the FASTA for the piBAQ theoretical-peptide
        denominator. Defaults to ``"Trypsin"``.
    pibaq_max_aa : int
        Maximum peptide length retained from the FASTA digest for piBAQ.
        Defaults to ``30``. The minimum length is taken from
        :attr:`FilterConfig.min_aa`.
    pibaq_min_shared : int
        Minimum number of distinct peptides two proteins must share to be
        placed in the same automatically discovered piBAQ family. Defaults
        to ``2`` (matches :func:`mokume.quantification.pibaq.peptides_to_protein`).
    pibaq_families_yaml : Optional[str]
        Path to a YAML file declaring explicit piBAQ family overrides
        (see :func:`mokume.quantification.families.load_families_yaml`).
        When ``None`` (default) family discovery is purely data-driven.
    pibaq_min_anchors : int
        Unique-anchor threshold. If no family member reaches it, shared signal
        is split equally and evidence is ``family_only``. Defaults to ``1``.
    pibaq_high_anchor_threshold : int
        Minimum anchor count (of the weakest member) for a family to be
        labelled ``EvidenceLevel == "high"``. Defaults to ``3``.
    ratio_fraction_merge : str
        How the ratio flow merges fractions of the same sample: ``"mean"``
        (default) or ``"max"``. Case-insensitive; anything else raises.
    """

    method: str = "maxlfq"
    ion_alignment: Optional[str] = None
    coverage_threshold: Optional[float] = None
    sample_correlation_threshold: Optional[float] = None
    ratio_fraction_merge: str = "mean"
    maxlfq_min_ratio_count: int = 2
    stabilize: bool = False
    # DirectLFQ-specific
    directlfq_num_cores: Optional[int] = None
    directlfq_min_nonan: int = 1
    directlfq_num_samples_quadratic: int = 50
    # piBAQ-specific (paralog-aware iBAQ family discovery / digestion)
    pibaq_enzyme: str = "Trypsin"
    pibaq_max_aa: int = 30
    pibaq_min_shared: int = 2
    pibaq_families_yaml: Optional[str] = None
    pibaq_min_anchors: int = 1
    pibaq_high_anchor_threshold: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.stabilize, bool):
            raise ValueError("stabilize must be a boolean")
        if self.stabilize and self.method.lower() != "maxlfq":
            raise ValueError("stabilize only applies to MaxLFQ")
        if (
            isinstance(self.maxlfq_min_ratio_count, bool)
            or not isinstance(self.maxlfq_min_ratio_count, int)
            or self.maxlfq_min_ratio_count < 1
        ):
            raise ValueError("maxlfq_min_ratio_count must be a positive integer")
        if self.sample_correlation_threshold is not None and not (
            -1.0 <= self.sample_correlation_threshold <= 1.0
        ):
            raise ValueError("sample_correlation_threshold must be between -1 and 1")
        # The two implementations disagree on case: the Python quantifier tests
        # ``fraction_merge_method == "max"`` exactly and falls back to mean for
        # everything else, while the Rust kernel parses the forwarded flag with
        # ``ignore_case``. A value like ``"Max"`` would therefore mean mean on
        # Python and max in Rust. Both read this field, so fold the
        # case here rather than in either reader. Folding also turns a typo
        # like ``"median"`` into an error instead of a silent mean.
        merge = self.ratio_fraction_merge.lower()
        if merge not in {"mean", "max"}:
            raise ValueError(
                "ratio_fraction_merge must be 'mean' or 'max'; got "
                f"{self.ratio_fraction_merge!r}"
            )
        self.ratio_fraction_merge = merge


@dataclass
class IRSConfig:
    """IRS (Internal Reference Scaling) normalization parameters."""

    enabled: bool = False
    reference_samples: Optional[list] = None
    sdrf_column: Optional[str] = None
    sdrf_values: Optional[list] = None
    reference_regex: str = "pool|powder|ref|reference|bridge"
    stat: str = "median"
    remove_reference: bool = False


@dataclass
class BatchCorrectionConfig:
    """Batch correction parameters."""

    enabled: bool = False
    method: str = "sample_prefix"
    column: Optional[str] = None
    covariates: Optional[list] = None
    parametric: bool = True
    mean_only: bool = False
    ref_batch: Optional[int] = None


@dataclass
class ImputationConfig:
    """Missing value imputation parameters."""

    enabled: bool = False
    method: str = "none"
    # MinProb / MinDet parameter
    quantile: float = 0.01
    # MinProb / QRILC draws
    seed: int = 42
    tune_sigma: float = 1.0
    # KNN parameters
    n_neighbors: int = 5


@dataclass
class DEConfig:
    """Differential expression analysis parameters."""

    enabled: bool = False
    contrasts: Optional[list] = None
    method: str = "auto"
    log2fc_threshold: float = 0.5
    fdr_threshold: float = 0.05
    fdr_method: str = "bh"
    output: Optional[str] = None
    # Ensemble-specific (used when method == "ensemble")
    ensemble_methods: Optional[list] = None
    ensemble_min_k: int = 2


def validate_de_config(config: DEConfig) -> tuple[str, ...] | None:
    """Validate ensemble-specific DE options and return resolved members."""
    return validate_de_ensemble(
        enabled=config.enabled,
        method=config.method,
        ensemble_methods=config.ensemble_methods,
        ensemble_min_k=config.ensemble_min_k,
    )


@dataclass
class OutputConfig:
    """Output and export parameters."""

    export_peptides: Optional[str] = None
    export_ions: Optional[str] = None
    plot_dir: Optional[str] = None
    plot_volcano: bool = False
    plot_heatmap: bool = False
    plot_pca: bool = False
    highlight_genes: Optional[list] = None
    interactive_report: bool = False
    report_output: Optional[str] = None
    export_anndata: bool = False
    anndata_view: Optional[str] = None


@dataclass
class PipelineConfig:
    """
    Configuration for the quantification pipeline.

    Organizes settings into logical groups:
    - input: File paths (parquet, sdrf, fasta)
    - filtering: Peptide/protein filters
    - normalization: Run and sample normalization methods
    - quantification: Quant method and method-specific params
    - irs: Internal Reference Scaling for multi-plex TMT
    - batch: Batch correction (ComBat)
    - imputation: Missing value imputation (MinProb, MinDet, KNN)
    - de: Differential expression analysis
    - output: Export paths, plotting, reports
    """

    input: InputConfig
    filtering: FilterConfig = field(default_factory=FilterConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    quantification: QuantificationConfig = field(default_factory=QuantificationConfig)
    irs: IRSConfig = field(default_factory=IRSConfig)
    batch: BatchCorrectionConfig = field(default_factory=BatchCorrectionConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    de: DEConfig = field(default_factory=DEConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
