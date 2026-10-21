"""
Unified pipeline: features → proteins in one step.

This module provides the main `features_to_proteins` function and
`QuantificationPipeline` class that handle the full proteomics
quantification workflow from feature-level parquet files to
protein intensities.

The pipeline automatically handles:
- Loading and filtering parquet data
- Normalization (run-level and sample-level)
- Protein quantification using various methods
- Optional intermediate exports (peptides, ions)

When DirectLFQ is selected as the quantification method, the pipeline uses the
DirectLFQ package for normalization and a Mokume streaming adapter for protein
estimation to avoid retaining ion-level intermediate results for every protein.
"""

import gc

import pandas as pd
from pathlib import Path
from typing import Optional

from mokume.core.constants import load_sdrf
from mokume.core.dataset import QpxDataset
from mokume.core.logger import get_logger
from mokume.model.normalization import parse_normalization_methods
from mokume.pipeline.config import (
    PipelineConfig,
    InputConfig,
    FilterConfig,
    NormalizationConfig,
    QuantificationConfig,
    IRSConfig,
    BatchCorrectionConfig,
    ImputationConfig,
    DEConfig,
    OutputConfig,
    RuntimeConfig,
    validate_de_config,
)
from mokume.pipeline.directlfq_streaming import estimate_protein_intensities_streamed
from mokume.pipeline.stages import (
    LoadingStage,
    NormalizationStage,
    QuantificationStage,
    ImputationStage,
    PostprocessingStage,
)

logger = get_logger("mokume.pipeline")


class QuantificationPipeline:
    """
    Unified pipeline: features → proteins.

    Thin orchestrator that delegates to stage classes:
    - LoadingStage: Data loading and filtering
    - NormalizationStage: Run-level and sample-level normalization
    - QuantificationStage: Protein quantification methods
    - PostprocessingStage: Batch correction, DE, plotting, reports

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.

    Examples
    --------
    >>> from mokume.pipeline import QuantificationPipeline, PipelineConfig
    >>> from mokume.pipeline.config import InputConfig, QuantificationConfig
    >>>
    >>> config = PipelineConfig(
    ...     input=InputConfig(parquet="data.parquet"),
    ...     quantification=QuantificationConfig(method="maxlfq"),
    ... )
    >>> pipeline = QuantificationPipeline(config)
    >>> proteins = pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._validate_config()
        self.loading = LoadingStage(config)
        self.normalization = NormalizationStage(config)
        self.quantification = QuantificationStage(config)
        self.imputation = ImputationStage(config)
        self.postprocessing = PostprocessingStage(config)
        # Intermediate tables/flags captured during a run() for run_dataset() to
        # attach to the QpxDataset. Populated as a side effect by the
        # standard-flow / ratio-flow helpers; never changes run()'s contract.
        # Held in a single dict to keep the pipeline's attribute surface small.
        self._run_capture: dict = self._empty_capture()

    @staticmethod
    def _empty_capture() -> dict:
        """Return a fresh, empty per-run capture state."""
        return {
            "peptides": None,
            "psms": None,
            "ratio_refs": None,
            "ratio_sample_to_plex": None,
            "irs_applied": False,
        }

    def _validate_config(self):
        """Validate configuration and check for required parameters."""
        validate_de_config(self.config.de)
        parse_normalization_methods(
            self.config.normalization.run_method,
            self.config.normalization.sample_method,
        )
        quant_method = self.config.quantification.method.lower()
        self._validate_normalization_config(quant_method)
        self._validate_reference_config(quant_method)
        self._validate_imputation_config()
        self._validate_batch_config()
        self._validate_de_output_config()
        self._validate_input_config(quant_method)

    def _validate_normalization_config(self, quant_method: str) -> None:
        run_method = self.config.normalization.run_method.lower()
        sample_method = self.config.normalization.sample_method.lower()
        if quant_method in {"directlfq", "ratio", "peptide_count"} and (
            run_method != "none" or sample_method != "none"
        ):
            reason = (
                "does not use intensity normalization"
                if quant_method == "peptide_count"
                else "manages normalization internally"
            )
            raise ValueError(
                f"{quant_method} {reason}; use run/sample normalization 'none'"
            )
        if self.config.normalization.proteins_file and sample_method != "hierarchical":
            raise ValueError(
                "normalization_proteins_file requires hierarchical sample "
                "normalization in mokume-py"
            )
        if self.config.output.export_peptides and quant_method in {
            "directlfq",
            "ratio",
        }:
            raise ValueError(
                f"export_peptides is not supported by the {quant_method} pipeline"
            )
        if self.config.output.export_ions and quant_method != "directlfq":
            raise ValueError("export_ions is supported only by the directlfq pipeline")

    def _validate_reference_config(self, quant_method: str) -> None:
        selectors = self._reference_selector_count()
        self._validate_reference_selectors(selectors)
        if quant_method == "peptide_count" and self.config.irs.enabled:
            raise ValueError("peptide_count quantification cannot apply IRS")
        if quant_method == "ratio":
            self._validate_ratio_reference_config()
        else:
            self._validate_irs_reference_config(selectors)
        if (
            self.config.quantification.coverage_threshold is not None
            and not self.config.input.sdrf
        ):
            raise ValueError("Coverage filtering requires an SDRF file")
        if (
            self.config.quantification.sample_correlation_threshold is not None
            and not self.config.input.sdrf
        ):
            raise ValueError("Sample correlation filtering requires an SDRF file")

    def _reference_selector_count(self) -> int:
        default_reference_regex = "pool|powder|ref|reference|bridge"
        return sum(
            (
                bool(self.config.irs.reference_samples),
                bool(self.config.irs.sdrf_column or self.config.irs.sdrf_values),
                self.config.irs.reference_regex != default_reference_regex,
            )
        )

    def _validate_reference_selectors(self, selectors: int) -> None:
        if selectors > 1:
            raise ValueError(
                "Choose one reference selector: samples, SDRF column+values, or regex"
            )
        if bool(self.config.irs.sdrf_column) != bool(self.config.irs.sdrf_values):
            raise ValueError("IRS SDRF column and values must be provided together")

    def _validate_ratio_reference_config(self) -> None:
        if self.config.irs.enabled:
            raise ValueError("Ratio quantification cannot also apply IRS")
        if (
            self.config.irs.sdrf_column
            or self.config.irs.sdrf_values
            or self.config.irs.stat.lower() != "median"
            or self.config.irs.remove_reference
        ):
            raise ValueError(
                "Ratio accepts reference_samples/reference_regex, not IRS-only options"
            )

    def _validate_irs_reference_config(self, selectors: int) -> None:
        if not self.config.irs.enabled and (
            selectors
            or self.config.irs.stat.lower() != "median"
            or self.config.irs.remove_reference
        ):
            raise ValueError("IRS options require IRS normalization to be enabled")
        if self.config.irs.enabled and not self.config.input.sdrf:
            raise ValueError("IRS normalization requires an SDRF file")

    def _validate_imputation_config(self) -> None:
        options_changed = any(
            (
                self.config.imputation.method.lower() != "none",
                self.config.imputation.quantile != 0.01,
                self.config.imputation.seed != 42,
                self.config.imputation.tune_sigma != 1.0,
                self.config.imputation.n_neighbors != 5,
            )
        )
        if not self.config.imputation.enabled and options_changed:
            raise ValueError("Imputation options require imputation to be enabled")
        if (
            self.config.imputation.enabled
            and self.config.imputation.method.lower() == "none"
        ):
            raise ValueError("Imputation requires an explicit method")

    def _validate_batch_config(self) -> None:
        options_changed = any(
            (
                self.config.batch.method.lower() != "sample_prefix",
                self.config.batch.column,
                self.config.batch.covariates,
                not self.config.batch.parametric,
                self.config.batch.mean_only,
                self.config.batch.ref_batch is not None,
            )
        )
        if not self.config.batch.enabled and options_changed:
            raise ValueError("Batch options require batch correction to be enabled")
        if self.config.batch.enabled:
            if self.config.batch.method.lower() not in {"sample_prefix", "column"}:
                raise ValueError("Batch method must be sample_prefix or column")
            if (
                self.config.batch.method.lower() == "column"
                and not self.config.batch.column
            ):
                raise ValueError("Batch method 'column' requires a batch column")
            if (
                self.config.batch.method.lower() != "column"
                and self.config.batch.column
            ):
                raise ValueError("A batch column requires batch method 'column'")

    def _validate_de_output_config(self) -> None:
        de_output_requested = bool(
            self.config.de.output
            or self.config.output.plot_volcano
            or self.config.output.interactive_report
        )
        if self.config.de.enabled and not de_output_requested:
            raise ValueError(
                "Differential expression requires a DE output, volcano plot, or interactive report"
            )
        options_changed = any(
            (
                self.config.de.contrasts,
                self.config.de.method.lower() != "auto",
                self.config.de.log2fc_threshold != 0.5,
                self.config.de.fdr_threshold != 0.05,
                self.config.de.fdr_method.lower() != "bh",
                self.config.de.output,
                self.config.de.ensemble_methods,
                self.config.de.ensemble_min_k != 2,
            )
        )
        if not self.config.de.enabled and options_changed:
            raise ValueError("Differential-expression options require DE to be enabled")
        if self.config.output.plot_volcano and not self.config.de.enabled:
            raise ValueError("Volcano plots require differential expression")
        if self.config.output.interactive_report and not self.config.de.enabled:
            raise ValueError("Interactive reports require differential expression")
        if (
            self.config.output.report_output
            and not self.config.output.interactive_report
        ):
            raise ValueError("report_output requires interactive_report")

    def _validate_input_config(self, quant_method: str) -> None:
        source_path = self.config.input.msstats or self.config.input.parquet
        if not Path(source_path).exists():
            source_name = "MSstats" if self.config.input.msstats else "Parquet"
            raise FileNotFoundError(f"{source_name} file not found: {source_path}")

        if self.config.input.msstats and quant_method == "ratio":
            raise ValueError(
                "Ratio quantification requires PSM-level QPX input; "
                "MSstats feature tables do not contain the required PSM evidence"
            )

        if quant_method == "pibaq" and not self.config.input.fasta_file:
            raise ValueError("piBAQ quantification requires --fasta-file")

        if quant_method == "ratio" and not self.config.input.sdrf:
            raise ValueError("Ratio quantification requires an SDRF file (--sdrf)")

        if (
            self.config.input.fasta_file
            and not Path(self.config.input.fasta_file).exists()
        ):
            raise FileNotFoundError(
                f"FASTA file not found: {self.config.input.fasta_file}"
            )

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline.

        Returns
        -------
        pd.DataFrame
            Protein intensities matrix (proteins x samples).
        """
        quant_method = self.config.quantification.method.lower()
        logger.info("Starting pipeline with quant_method=%s", quant_method)

        # Reset per-run capture state so run_dataset() reflects only this run.
        self._run_capture = self._empty_capture()

        if quant_method == "directlfq":
            protein_df = self._run_directlfq_pipeline()
        elif quant_method == "ratio":
            protein_df = self._run_ratio_pipeline()
        else:
            protein_df = self._run_mokume_pipeline()

        # Apply IRS normalization if configured (skip for ratio — ratios already
        # handle cross-plex normalization via per-plex reference division)
        if self.config.irs.enabled and quant_method != "ratio":
            protein_df = self.normalization.apply_irs(protein_df)
            self._run_capture["irs_applied"] = True

        protein_df = self.normalization.apply_sample_correlation_filter(protein_df)

        # Apply coverage filter if configured (generic, works with any method)
        if self.config.quantification.coverage_threshold is not None:
            protein_df = self.normalization.apply_coverage_filter(protein_df)

        # Apply imputation if configured (after coverage filter, before batch correction)
        if self.config.imputation.enabled:
            protein_df = self.imputation.impute(protein_df)

        # Apply batch correction if configured
        if self.config.batch.enabled:
            protein_df = self.postprocessing.apply_batch_correction(protein_df)

        # Run differential expression if configured
        de_results = None
        if self.config.de.enabled:
            de_results = self.postprocessing.run_differential_expression(protein_df)

        # Generate plots if configured
        if self.config.output.plot_dir and any(
            [
                self.config.output.plot_volcano,
                self.config.output.plot_heatmap,
                self.config.output.plot_pca,
            ]
        ):
            self.postprocessing.generate_plots(protein_df, de_results)

        # Generate interactive report if configured
        if self.config.output.interactive_report and de_results:
            self.postprocessing.generate_interactive_report(protein_df, de_results)

        return protein_df

    def run_dataset(self) -> QpxDataset:
        """Execute the pipeline and return results as a :class:`QpxDataset`.

        This is a thin wrapper around :meth:`run`: it runs the exact same
        quantification workflow (so protein values are identical to the legacy
        ``run() -> DataFrame`` path), then packages the wide protein matrix and
        any intermediate tables captured during the run into a hierarchical
        :class:`~mokume.core.dataset.QpxDataset` with provenance metadata.

        The dataset levels populated depend on the flow:

        * ``proteins`` -- always set to the wide (protein x sample) matrix
          returned by :meth:`run`.
        * ``peptides`` -- the assembled, normalized peptide table, when the
          standard (mokume-native) flow produced one.
        * ``psms`` -- the loaded PSM table, for the ratio flow.
        * ``sample_info`` -- SDRF metadata when an SDRF file was configured.

        Provenance (``dataset.uns["provenance"]["steps"]``) records at least a
        ``loading`` step and a ``quantification`` step; an
        ``irs_normalization`` step is appended when IRS ran.

        Returns
        -------
        QpxDataset
            Dataset with the protein matrix, intermediate levels, sample
            metadata, and recorded provenance.
        """
        quant_method = self.config.quantification.method

        # Run the legacy pipeline unchanged; this also populates the
        # self._run_capture dict (peptides / psms / ratio refs / irs flag).
        protein_df = self.run()
        capture = self._run_capture

        dataset = QpxDataset()
        dataset.proteins = protein_df
        if capture["peptides"] is not None:
            dataset.peptides = capture["peptides"]
        if capture["psms"] is not None:
            dataset.psms = capture["psms"]

        # Sample metadata from the SDRF, when one is configured. Loading is
        # best-effort: a malformed/absent SDRF must not sink an otherwise
        # successful quantification run, so file/parse errors are logged and
        # sample_info is simply left unset.
        if self.config.input.sdrf:
            try:
                dataset.sample_info = load_sdrf(self.config.input.sdrf)
            except (OSError, ValueError) as exc:  # pragma: no cover - best-effort
                logger.debug("Could not load SDRF sample_info: %s", exc)

        # --- Provenance -------------------------------------------------
        rows_in = None
        if capture["peptides"] is not None:
            rows_in = len(capture["peptides"])
        elif capture["psms"] is not None:
            rows_in = len(capture["psms"])

        dataset.record_step(
            "loading",
            parquet=(
                str(self.config.input.parquet) if self.config.input.parquet else None
            ),
            msstats=(
                str(self.config.input.msstats) if self.config.input.msstats else None
            ),
            sdrf=str(self.config.input.sdrf) if self.config.input.sdrf else None,
            rows_out=rows_in,
        )
        dataset.record_step(
            "quantification",
            method=quant_method,
            rows_out=len(protein_df) if isinstance(protein_df, pd.DataFrame) else None,
        )
        if capture["irs_applied"]:
            dataset.record_step("irs_normalization", stat=self.config.irs.stat)

        # Ratio flow exposes its detected reference samples for downstream
        # consumers (mirrors main's cecilia integration contract).
        if quant_method.lower() == "ratio" and capture["ratio_refs"] is not None:
            sample_to_plex = capture["ratio_sample_to_plex"] or {}
            dataset.uns["ratio_config"] = {
                "reference_samples": list(capture["ratio_refs"]),
                "n_plexes": len(set(sample_to_plex.values())) if sample_to_plex else 0,
            }

        return dataset

    def _run_directlfq_pipeline(self) -> pd.DataFrame:
        """Run pipeline using DirectLFQ package."""
        try:
            import directlfq.normalization as lfq_norm
            import directlfq.config as lfq_config
        except ImportError as exc:
            raise ImportError(
                "DirectLFQ quantification requires the directlfq package.\n"
                "Install with: pip install directlfq\n"
                "Or: pip install mokume-py[directlfq]"
            ) from exc

        logger.info("Loading and filtering data for DirectLFQ...")
        filtered_table = self.loading.load_for_directlfq()
        logger.info("Filtered data: %d features", filtered_table.num_rows)

        logger.info("Converting to DirectLFQ format...")
        directlfq_input = self.loading.convert_to_directlfq_format(filtered_table)
        logger.info("DirectLFQ input shape: %s", directlfq_input.shape)
        del filtered_table
        gc.collect()

        # Configure DirectLFQ
        lfq_config.set_global_protein_and_ion_id(protein_id="protein", quant_id="ion")
        lfq_config.set_compile_normalized_ion_table(False)

        # Run DirectLFQ normalization
        logger.info("Running DirectLFQ sample normalization...")
        normed_df = lfq_norm.NormalizationManagerSamplesOnSelectedProteins(
            directlfq_input,
            num_samples_quadratic=self.config.quantification.directlfq_num_samples_quadratic,
        ).complete_dataframe
        del directlfq_input
        gc.collect()

        logger.info("Running DirectLFQ protein estimation...")
        protein_df = estimate_protein_intensities_streamed(
            normed_df,
            min_nonan=self.config.quantification.directlfq_min_nonan,
            num_samples_quadratic=self.config.quantification.directlfq_num_samples_quadratic,
            num_cores=self.config.quantification.directlfq_num_cores,
            export_ions=self.config.output.export_ions,
        )

        logger.info("DirectLFQ complete: %d proteins", len(protein_df))
        return protein_df

    def _run_mokume_pipeline(self) -> pd.DataFrame:
        """Run pipeline using mokume's native implementations."""
        logger.info("Loading and filtering data...")
        peptide_df = self.loading.load_for_mokume()
        logger.info("Processed peptides: %d rows", len(peptide_df))

        # Capture the assembled peptide table so run_dataset() can attach it to
        # the QpxDataset peptides level without reloading the parquet.
        self._run_capture["peptides"] = peptide_df

        # Export peptides if requested
        if self.config.output.export_peptides:
            logger.info("Exporting peptides to %s", self.config.output.export_peptides)
            peptide_df.to_csv(self.config.output.export_peptides, index=False)

        # Quantify proteins
        logger.info(
            "Quantifying proteins with method: %s", self.config.quantification.method
        )
        protein_df = self.quantification.quantify(peptide_df)
        logger.info("Quantification complete: %d proteins", len(protein_df))

        return protein_df

    def _run_ratio_pipeline(self) -> pd.DataFrame:
        """Run ratio-based quantification (PS protocol)."""
        from mokume.quantification.ratio import RatioQuantification

        logger.info("Running ratio-based quantification (PS protocol)...")

        psm_df, ref_samples, sample_to_plex = self.loading.load_for_ratio()

        # Capture the PSM table and detected references so run_dataset() can
        # attach them to the QpxDataset without reloading the parquet.
        self._run_capture["psms"] = psm_df
        self._run_capture["ratio_refs"] = ref_samples
        self._run_capture["ratio_sample_to_plex"] = sample_to_plex

        # Run ratio quantification
        ratio_quant = RatioQuantification(
            reference_samples=ref_samples,
            sample_to_plex=sample_to_plex,
            fraction_merge_method=self.config.quantification.ratio_fraction_merge,
        )
        protein_df = ratio_quant.quantify(psm_df)

        # Remove reference samples from output columns (log2(ref/ref) = 0)
        protein_col = protein_df.columns[0]
        cols_to_keep = [protein_col] + [
            c for c in protein_df.columns if c == protein_col or c not in ref_samples
        ]
        # Deduplicate while preserving order
        seen = set()
        unique_cols = []
        for c in cols_to_keep:
            if c not in seen:
                seen.add(c)
                unique_cols.append(c)
        protein_df = protein_df[unique_cols]

        logger.info("Ratio pipeline complete: %d proteins", len(protein_df))
        return protein_df


def features_to_proteins(
    parquet: Optional[str],
    output: str,
    sdrf: Optional[str] = None,
    quant_method: str = "maxlfq",
    min_aa: int = 7,
    min_unique_peptides: int = 2,
    remove_contaminants: bool = True,
    run_normalization: Optional[str] = None,
    sample_normalization: Optional[str] = None,
    normalization_proteins_file: Optional[str] = None,
    fasta_file: Optional[str] = None,
    ion_alignment: Optional[str] = None,
    # piBAQ-specific (paralog-aware iBAQ)
    pibaq_enzyme: str = "Trypsin",
    pibaq_max_aa: int = 30,
    pibaq_min_shared: int = 2,
    pibaq_families_yaml: Optional[str] = None,
    pibaq_min_anchors: int = 1,
    pibaq_high_anchor_threshold: int = 3,
    directlfq_num_cores: Optional[int] = None,
    directlfq_min_nonan: int = 1,
    export_peptides: Optional[str] = None,
    export_ions: Optional[str] = None,
    # Batch correction parameters
    batch_correction: bool = False,
    batch_method: str = "sample_prefix",
    batch_column: Optional[str] = None,
    batch_covariates: Optional[list] = None,
    batch_parametric: bool = True,
    batch_mean_only: bool = False,
    batch_ref: Optional[int] = None,
    # IRS normalization parameters
    irs: bool = False,
    irs_reference_samples: Optional[list] = None,
    irs_sdrf_column: Optional[str] = None,
    irs_sdrf_values: Optional[list] = None,
    irs_reference_regex: str = "pool|powder|ref|reference|bridge",
    irs_stat: str = "median",
    irs_remove_reference: bool = False,
    # Differential expression parameters
    differential_expression: bool = False,
    de_contrasts: Optional[list] = None,
    de_method: str = "auto",
    de_log2fc_threshold: float = 0.5,
    de_fdr_threshold: float = 0.05,
    de_fdr_method: str = "bh",
    de_output: Optional[str] = None,
    de_ensemble_methods: Optional[list] = None,
    de_ensemble_min_k: int = 2,
    # Plotting parameters
    plot_output_dir: Optional[str] = None,
    plot_volcano: bool = False,
    plot_heatmap: bool = False,
    plot_pca: bool = False,
    highlight_genes: Optional[list] = None,
    # Coverage filter
    coverage_threshold: Optional[float] = None,
    # Sample correlation QC
    sample_correlation_threshold: Optional[float] = None,
    # Ratio quantification
    ratio_fraction_merge: str = "mean",
    # Imputation parameters
    impute: bool = False,
    impute_method: str = "none",
    impute_quantile: float = 0.01,
    impute_seed: int = 42,
    impute_tune_sigma: float = 1.0,
    impute_n_neighbors: int = 5,
    # Interactive report parameters
    interactive_report: bool = False,
    report_output: Optional[str] = None,
    # DuckDB resource limits (cap DuckDB engine only, NOT total process RSS)
    duckdb_memory: Optional[str] = None,
    duckdb_threads: Optional[int] = None,
    *,
    msstats: Optional[str] = None,
    maxlfq_min_ratio_count: int = 2,
    stabilize: bool = False,
) -> pd.DataFrame:
    """
    Quantify proteins from QPX feature parquet or legacy SDRF+MSstats data.

    This is the main entry point for the unified pipeline that handles
    the full workflow from features to proteins in one step.

    Parameters
    ----------
    parquet : str, optional
        Path to the input parquet file (quantms.io/qpx format). Mutually exclusive
        with ``msstats``.
    output : str
        Path for the output protein intensities file.
    sdrf : str, optional
        Path to SDRF file for sample metadata.
    msstats : str, optional
        Path to a quantms ``*_msstats_in.csv`` file. Requires ``sdrf``.
    quant_method : str
        Quantification method. Options:
        - 'directlfq': Uses DirectLFQ package (normalization + quantification)
        - 'pibaq': Paralog-aware iBAQ with shared-peptide allocation
        - 'maxlfq': MaxLFQ algorithm
        - 'top3': Top 3 peptides per protein
        - 'top5': Top 5 peptides per protein
        - 'sum': Sum of all peptides
        - 'median': Median of peptides
    stabilize : bool
        Enable Cox Eq. 5 large-ratio stabilization (MaxLFQ only). Default False.
    maxlfq_min_ratio_count : int
        Minimum shared peptide species per MaxLFQ sample pair. Default 2.
    min_aa : int
        Minimum amino acid length for peptides. Default: 7.
    min_unique_peptides : int
        Minimum unique peptides per protein/sample cell. Default: 2.
    remove_contaminants : bool
        Whether to remove contaminants and decoys. Default: True.
    run_normalization : str
        Run/technical replicate normalization method. Options:
        none, median, mean, max, global, max_min, IQR. DirectLFQ and Ratio
        manage normalization internally; peptide_count is unaffected by
        intensity normalization. These methods therefore require ``none``.
    sample_normalization : str
        Sample-to-sample normalization method. Options:
        - 'none': No normalization
        - 'globalMedian': Sample median / global median
        - 'conditionMedian': Condition-specific median
        - 'hierarchical': DirectLFQ-style hierarchical clustering
        DirectLFQ and Ratio manage normalization internally; peptide_count is
        unaffected by intensity normalization. These methods require ``none``.
    normalization_proteins_file : str, optional
        File with protein IDs to use for hierarchical normalization (one per
        line). Requires ``sample_normalization='hierarchical'``.
    fasta_file : str, optional
        FASTA file path. Required for piBAQ quantification.
    ion_alignment : str, optional
        Removed compatibility argument. Passing a value raises ``ValueError``.
    pibaq_enzyme : str
        Protease used to digest the FASTA for the piBAQ denominator.
        Default: 'Trypsin'.
    pibaq_max_aa : int
        Maximum peptide length retained from the FASTA digest for piBAQ.
        Default: 30 (min length comes from ``min_aa``).
    pibaq_min_shared : int
        Minimum distinct peptides two proteins must share to co-cluster
        into one piBAQ family. Default: 2.
    pibaq_families_yaml : str, optional
        Path to a YAML file with explicit piBAQ family overrides. When
        ``None`` (default) family discovery is purely data-driven.
    pibaq_min_anchors : int
        Unique-anchor threshold. If no family member reaches it, shared signal
        is split equally and evidence is ``family_only``. Default: 1.
    pibaq_high_anchor_threshold : int
        Minimum anchor count (weakest member) for a family to be labelled
        ``EvidenceLevel == "high"``. Default: 3.
    directlfq_num_cores : int, optional
        Number of cores for DirectLFQ parallel processing.
    directlfq_min_nonan : int
        Minimum non-missing values required by DirectLFQ for protein estimation.
    export_peptides : str, optional
        Path to export normalized peptides (for debugging/analysis).
    export_ions : str, optional
        Path to export normalized ions (DirectLFQ only).
    batch_correction : bool
        Whether to apply batch correction after quantification. Default: False.
    batch_method : str
        Batch detection method: sample_prefix, run, column. Default: sample_prefix.
    batch_column : str, optional
        Column name for explicit batch assignment (when batch_method='column').
    batch_covariates : list, optional
        SDRF columns to use as covariates (biological signal to preserve).
    batch_parametric : bool
        Use parametric estimation for ComBat. Default: True.
    batch_mean_only : bool
        Only adjust batch means, not individual effects. Default: False.
    batch_ref : int, optional
        Reference batch ID.
    sample_correlation_threshold : float, optional
        Minimum mean Pearson correlation to same-condition peers on the
        pairwise-complete log2 protein matrix. Requires ``sdrf``.

    Returns
    -------
    pd.DataFrame
        Protein intensities matrix.
    """
    quant_method_lower = quant_method.lower()
    manages_normalization = quant_method_lower in {
        "directlfq",
        "ratio",
        "peptide_count",
    }
    if run_normalization is None:
        run_normalization = "none" if manages_normalization else "median"
    if sample_normalization is None:
        sample_normalization = "none" if manages_normalization else "globalMedian"
    if ion_alignment is not None:
        raise ValueError("ion_alignment is no longer supported")

    config = PipelineConfig(
        input=InputConfig(
            parquet=parquet,
            sdrf=sdrf,
            fasta_file=fasta_file,
            msstats=msstats,
        ),
        filtering=FilterConfig(
            min_aa=min_aa,
            min_unique_peptides=min_unique_peptides,
            remove_contaminants=remove_contaminants,
        ),
        normalization=NormalizationConfig(
            run_method=run_normalization,
            sample_method=sample_normalization,
            proteins_file=normalization_proteins_file,
        ),
        quantification=QuantificationConfig(
            maxlfq_min_ratio_count=maxlfq_min_ratio_count,
            stabilize=stabilize,
            method=quant_method,
            ion_alignment=ion_alignment,
            coverage_threshold=coverage_threshold,
            sample_correlation_threshold=sample_correlation_threshold,
            ratio_fraction_merge=ratio_fraction_merge,
            directlfq_num_cores=directlfq_num_cores,
            directlfq_min_nonan=directlfq_min_nonan,
            pibaq_enzyme=pibaq_enzyme,
            pibaq_max_aa=pibaq_max_aa,
            pibaq_min_shared=pibaq_min_shared,
            pibaq_families_yaml=pibaq_families_yaml,
            pibaq_min_anchors=pibaq_min_anchors,
            pibaq_high_anchor_threshold=pibaq_high_anchor_threshold,
        ),
        irs=IRSConfig(
            enabled=irs,
            reference_samples=irs_reference_samples,
            sdrf_column=irs_sdrf_column,
            sdrf_values=irs_sdrf_values,
            reference_regex=irs_reference_regex,
            stat=irs_stat,
            remove_reference=irs_remove_reference,
        ),
        batch=BatchCorrectionConfig(
            enabled=batch_correction,
            method=batch_method,
            column=batch_column,
            covariates=batch_covariates,
            parametric=batch_parametric,
            mean_only=batch_mean_only,
            ref_batch=batch_ref,
        ),
        imputation=ImputationConfig(
            enabled=impute,
            method=impute_method,
            quantile=impute_quantile,
            seed=impute_seed,
            tune_sigma=impute_tune_sigma,
            n_neighbors=impute_n_neighbors,
        ),
        de=DEConfig(
            enabled=differential_expression,
            contrasts=de_contrasts,
            method=de_method,
            log2fc_threshold=de_log2fc_threshold,
            fdr_threshold=de_fdr_threshold,
            fdr_method=de_fdr_method,
            output=de_output,
            ensemble_methods=de_ensemble_methods,
            ensemble_min_k=de_ensemble_min_k,
        ),
        output=OutputConfig(
            export_peptides=export_peptides,
            export_ions=export_ions,
            plot_dir=plot_output_dir,
            plot_volcano=plot_volcano,
            plot_heatmap=plot_heatmap,
            plot_pca=plot_pca,
            highlight_genes=highlight_genes,
            interactive_report=interactive_report,
            report_output=report_output,
        ),
        runtime=RuntimeConfig(
            duckdb_memory=duckdb_memory,
            duckdb_threads=duckdb_threads,
        ),
    )

    pipeline = QuantificationPipeline(config)
    protein_df = pipeline.run()

    # Save output
    if output.lower().endswith(".parquet"):
        protein_df.to_parquet(output, index=False)
    else:
        protein_df.to_csv(output, index=False)
    logger.info("Protein intensities saved to %s", output)

    return protein_df
