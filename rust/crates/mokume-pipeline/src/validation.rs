use crate::accessions::protein_group_name;
use crate::normalization_factors::unsupported;
use crate::{
    FeatureToPeptidesConfig, FeatureToProteinsConfig, HashSet, ImputationConfig, InputConfig,
    MokumeError, NamedScoreFilterConfig, Path, PreprocessingFilterConfig, QuantMethod, Regex,
    Result, SampleNormalizationMethod, SdrfTable, de, info, invalid_input,
    parse_run_normalization_method, parse_sample_normalization_method, warn,
};

/// Standalone `features2peptides`: load features, filter, aggregate to the
/// canonical-peptide x sample level, apply run/sample normalization, and export
/// the peptide intensity matrix.
///
/// This reuses the exact ingest, run/sample-factor normalization, and peptide
/// export machinery that backs `features2proteins --export-peptides`, so the
/// deterministic output is identical to that verified path. The matrix is the
/// Python long-format table (`ProteinName, PeptideCanonical, SampleID,
/// BioReplicate, Condition, NormIntensity`), produced by:
///   * per-peptidoform (peptidoform|charge) max collapse across fractions/runs
///     (Python `get_peptidoform_normalize_intensities`),
///   * summed per canonical-peptide cell (Python `sum_peptidoform_intensities`),
///   * filtered by the per-(protein, sample) `min_unique` canonical-peptide gate
///     (Python's per-sample `groupby(PROTEIN_NAME).filter(... >= min_unique)`),
///   * `--remove_ids`: rows containing a complete parsed accession from the
///     supplied ID list are dropped during ingest,
///   * `--remove_low_frequency_peptides`: `(ProteinName, PeptideCanonical)`
///     pairs seen in fewer than 20% of samples are dropped at export
///     (Python `get_low_frequency_peptides`),
///   * optionally log2-transformed (Python `--log2`).
///
/// Implemented options:
///   * `--keep-shared-peptides`: skip the per-feature `unique == 1` filter and
///     the per-protein `min_unique` gate (Python peptide.py:268-281), keeping
///     shared/non-unique peptide rows,
///   * `--save_parquet`: additionally write the same matrix to a parquet sibling
///     (extension swapped to `.parquet`), matching Python `WriteParquetTask`,
///   * `--aggregation_level run`: append `Run` / `TechReplicate` columns and key
///     the matrix at run level (Python `sum_peptidoform_intensities` run branch).
///
/// Requested filter settings that are not implemented are rejected with
/// `MokumeError::NotImplemented` before any work, never silently skipped.
/// Up-front validation for `features2peptides`: existence of inputs, supported
/// normalization methods, and the channel-IRS-adjacent constraints. Kept
/// separate so the orchestrator stays a linear sequence of stages.
/// Reject the preprocessing-filter options not yet ported, when set to an active
/// (non-default) value. The per-row filters (intensity floor, peptide length,
/// charge, modification, missed cleavages) and the per-`(protein, sample)`
/// unique-peptide gate are wired (`passes_peptide_filter_pipeline` +
/// `run_features_to_peptides`); the group-level filters (CV, quantile, run QC)
/// are applied via the `collect_intensity_group_filters` and
/// `collect_run_qc_exclusions` pre-passes (see `validate_filter_pipeline_subset`),
/// reproducing Python's per-sample chain rather than failing fast. Explicit
/// peptide/protein FDR thresholds use the dedicated QPX q-value fields and fail
/// before output when the requested field is unpopulated.
/// Compile the `exclude_sequence_patterns` once into regexes, mirroring Python's
/// per-pattern `re.compile` (peptide.py:390). Patterns are real (un-escaped,
/// case-sensitive) regexes; an invalid one is surfaced as a configuration error
/// rather than silently dropped.
pub(crate) fn compile_exclude_sequence_patterns(patterns: &[String]) -> Result<Vec<Regex>> {
    patterns
        .iter()
        .map(|pattern| {
            Regex::new(pattern).map_err(|source| MokumeError::InvalidInput {
                message: format!("invalid exclude-sequence-pattern regex '{pattern}': {source}"),
            })
        })
        .collect()
}

fn validate_filter_pipeline_subset(config: &PreprocessingFilterConfig) -> Result<()> {
    if config.strict_mode || !config.log_filtered_counts {
        return unsupported("features2peptides filter global logging options");
    }
    let intensity = &config.intensity;
    // `cv_threshold` is applied via the `collect_intensity_group_filters` pre-pass
    // (per-`(sample, protein, canonical)` CV over the `min_unique`-gated raw
    // intensities) and the per-row drop in `ingest`; nothing to reject here.
    //
    // `min_replicate_agreement > 1` is a *degenerate* filter under Python's
    // per-sample pipeline application: `ReplicateAgreementFilter` counts
    // `nunique(SampleID)` within `dataset_df`, which holds exactly one sample
    // (`peptide.py:266` slices `df[df["sample_accession"] == sample]`), so the
    // count is always 1. Any threshold `>= 2` therefore drops every row of every
    // sample, emptying the whole output (verified end-to-end against the `mokume`
    // CLI: "ReplicateAgreementFilter removed N items (100.0%)", header-only CSV).
    // We reproduce that exactly (no row survives ingest) and warn rather than fail.
    if intensity.min_replicate_agreement > 1 {
        warn!(
            "features2peptides filter min-replicate-agreement={} empties the output: \
             Python applies the pipeline per single-sample frame, where \
             nunique(SampleID) is always 1, so the >=2 threshold removes every row \
             (matches Python's per-sample pipeline application)",
            intensity.min_replicate_agreement
        );
    }
    // `quantile_lower` / `quantile_upper` are applied via the same
    // `collect_intensity_group_filters` pre-pass (per-sample `[lower, upper]` over
    // the `min_unique`-gated, post-CV intensities) and the per-row drop in
    // `ingest`; nothing to reject here.
    let peptide = &config.peptide;
    let protein = &config.protein;
    validate_fdr_thresholds(peptide.fdr_threshold, protein.fdr_threshold)?;
    validate_named_score_filter(peptide.score.as_ref())?;
    if peptide.require_unique_peptides {
        return unsupported("features2peptides peptide unique filter");
    }
    // `exclude_sequence_patterns` is applied per-row during ingest
    // (`passes_peptide_filter_pipeline` -> `filters::sequence_matches_excluded`),
    // compiled once into `excluded_sequence_regexes`; nothing to reject here.
    if protein.min_coverage > 0.0 {
        return unsupported("features2peptides filter min-coverage");
    }
    if protein.min_peptides != 1 || !protein.protein_grouping.eq_ignore_ascii_case("none") {
        return unsupported("features2peptides protein grouping filters");
    }
    // `keep` (no-op), `remove` (drop all rows of a razor peptide), and
    // `assign_to_top` (keep only the top-protein rows, first-appearance tie-break)
    // are applied at materialization.
    match protein.razor_peptide_handling.as_str() {
        "keep" | "remove" | "assign_to_top" => {}
        _ => return unsupported("features2peptides filter razor-peptide-handling"),
    }
    // Custom contaminant patterns drop proteins, which shifts the normalization
    // median. They are now applied before the median in the `SQLFilterBuilder`
    // pre-pass (`matches_sql_contaminant` over raw `pg_accessions` in
    // `collect_intensity_factors` / `collect_run_qc_exclusions`) and at ingest via
    // `matches_protein_contaminant` (parsed `protein_group`), so this is accepted.
    let run_qc = &config.run_qc;
    // Run-QC thresholds, including missing rate, are implemented via the
    // technical-run pre-pass (`collect_run_qc_exclusions`).
    if !run_qc.max_missing_rate.is_finite() || !(0.0..=1.0).contains(&run_qc.max_missing_rate) {
        return Err(invalid_input(
            "features2peptides max-missing-rate must be between 0 and 1",
        ));
    }
    Ok(())
}

fn validate_named_score_filter(score: Option<&NamedScoreFilterConfig>) -> Result<()> {
    let Some(score) = score else {
        return Ok(());
    };
    if score.name.trim().is_empty() {
        return Err(invalid_input(
            "features2peptides filter score name cannot be empty",
        ));
    }
    if !score.threshold.is_finite() {
        return Err(invalid_input(
            "features2peptides filter score threshold must be finite",
        ));
    }
    Ok(())
}

fn validate_fdr_thresholds(peptide: Option<f64>, protein: Option<f64>) -> Result<()> {
    if peptide.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(invalid_input(
            "features2peptides peptide FDR threshold must be between 0 and 1",
        ));
    }
    if protein.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(invalid_input(
            "features2peptides protein FDR threshold must be between 0 and 1",
        ));
    }
    Ok(())
}

pub(crate) fn active_filter_pipeline(
    config: &FeatureToPeptidesConfig,
) -> Option<&PreprocessingFilterConfig> {
    config
        .filter_pipeline
        .as_ref()
        .filter(|pipeline| pipeline.enabled)
}

pub(crate) fn validate_features_to_peptides_config(config: &FeatureToPeptidesConfig) -> Result<()> {
    let parquet = required_parquet(&config.input)?;
    if !parquet.exists() {
        return Err(MokumeError::MissingInput {
            path: parquet.to_path_buf(),
        });
    }
    if config.input.msstats.is_some() {
        return Err(invalid_input(
            "features2peptides does not support --msstats input",
        ));
    }
    if let Some(sdrf) = &config.input.sdrf
        && !sdrf.exists()
    {
        return Err(MokumeError::MissingInput { path: sdrf.clone() });
    }
    if let Some(pipeline) = active_filter_pipeline(config) {
        validate_filter_pipeline_subset(pipeline)?;
    }

    // Run normalization must be a supported per-run factor method. The parsed
    // value is consumed later by `collect_intensity_factors`; validate it up
    // front so an unsupported method fails with a clear stage.
    parse_run_normalization_method(&config.run_normalization).map_err(|_| {
        MokumeError::NotImplemented {
            stage: "features2peptides run-normalization-method",
        }
    })?;
    let sample_method =
        parse_sample_normalization_method(&config.sample_normalization).map_err(|_| {
            MokumeError::NotImplemented {
                stage: "features2peptides sample-normalization-method",
            }
        })?;
    if !matches!(
        sample_method,
        None | Some(
            SampleNormalizationMethod::GlobalMedian | SampleNormalizationMethod::ConditionMedian
        )
    ) {
        return unsupported("features2peptides sample-normalization-method");
    }
    if sample_method == Some(SampleNormalizationMethod::ConditionMedian)
        && config.input.sdrf.is_none()
    {
        return Err(invalid_input(
            "conditionmedian sample normalization requires --sdrf option",
        ));
    }
    Ok(())
}

pub(crate) fn required_parquet(input: &InputConfig) -> Result<&Path> {
    input
        .parquet
        .as_deref()
        .ok_or_else(|| invalid_input("this command requires --parquet input"))
}

/// Quant methods whose aggregation collapses to `(protein, sample)` peptide
/// cells normalized by [`apply_dataset_norm_to_peptide_cells`]. These share the
/// dataset-level normalization with the `--export-peptides` path, so the
/// exported peptides match the protein matrix. piBAQ / DirectLFQ / Ratio use
/// aggregation-specific paths; MaxLFQ exports its normalized species traces.
fn is_cell_based_linear_quant(method: QuantMethod) -> bool {
    matches!(
        method,
        QuantMethod::Sum
            | QuantMethod::Intensity
            | QuantMethod::Median
            | QuantMethod::Abd
            | QuantMethod::PeptideCount
            | QuantMethod::TopN
    )
}

/// The dataset-level sample normalization to apply on the canonical-peptide x
/// sample cells, if any. The remaining factor-based methods (global/condition
/// median) are applied during ingest and are not returned here; DirectLFQ and
/// Ratio manage their own normalization.
///
/// MedianCenter/MeanCenter are dataset-level in Python (`is_dataset_level`):
/// they center the summed-canonical-peptide matrix in log2 space, NOT the raw
/// per-feature intensities, so they belong on the cell path rather than the
/// ingest-time factor path.
pub(crate) fn dataset_sample_normalization_method(
    config: &FeatureToProteinsConfig,
) -> Result<Option<SampleNormalizationMethod>> {
    if matches!(
        config.quantification,
        QuantMethod::DirectLfq
            | QuantMethod::Ratio
            | QuantMethod::PeptideCount
            | QuantMethod::SpectralCount
    ) {
        return Ok(None);
    }
    Ok(
        match parse_sample_normalization_method(&config.normalization.sample_method)? {
            Some(
                method @ (SampleNormalizationMethod::Quantile
                | SampleNormalizationMethod::Rlr
                | SampleNormalizationMethod::Loess
                | SampleNormalizationMethod::Hierarchical
                | SampleNormalizationMethod::MedianCenter
                | SampleNormalizationMethod::MeanCenter
                | SampleNormalizationMethod::Tmm),
            ) => Some(method),
            _ => None,
        },
    )
}

/// `--filter-protein-fdr`: reject a row whose QPX protein group (as reported, before any renaming) failed.
pub(crate) fn fails_protein_fdr(allowed: Option<&HashSet<String>>, accessions: &[String]) -> bool {
    allowed.is_some_and(|allowed| {
        protein_group_name(accessions).is_none_or(|group| !allowed.contains(&group))
    })
}

/// Argument rules are enforced where the config is built (`mokume-command`).
/// Before any input is read, only the referenced files must exist.
pub(crate) fn validate_features_to_proteins(config: &FeatureToProteinsConfig) -> Result<()> {
    let input = &config.input;
    [
        &input.parquet,
        &input.msstats,
        &input.psm,
        &input.sdrf,
        &input.run,
        &input.fasta,
        &config.normalization.normalization_proteins,
    ]
    .into_iter()
    .flatten()
    .try_for_each(|path| require_existing_input(path))
}

pub(crate) fn input_sample_table(input: &InputConfig) -> Result<Option<SdrfTable>> {
    match (&input.sdrf, &input.run) {
        (Some(path), _) => {
            let sdrf = SdrfTable::from_path(path)?;
            sdrf.warn_shared_source_names();
            Ok(Some(sdrf))
        }
        (_, Some(path)) => SdrfTable::from_run_path(path).map(Some),
        _ => Ok(None),
    }
}

fn require_existing_input(path: &Path) -> Result<()> {
    if path.exists() {
        Ok(())
    } else {
        Err(MokumeError::MissingInput {
            path: path.to_path_buf(),
        })
    }
}

/// Option combinations the Rust aggregation paths do not implement.
pub(crate) fn validate_implemented_subset(config: &FeatureToProteinsConfig) -> Result<()> {
    let dataset_normalization = dataset_sample_normalization_method(config)?;
    // Dataset-level sample normalization is carried into the exported peptides
    // for cell-based methods and MaxLFQ. Other aggregations use structures
    // whose normalized values are not connected to the peptide export.
    if (config.output.export_peptides.is_some()
        || config.output.export_unnormalized_peptides.is_some())
        && dataset_normalization.is_some()
        && !is_cell_based_linear_quant(config.quantification)
        && config.quantification != QuantMethod::MaxLfq
    {
        return unsupported("dataset-normalization-export-peptides");
    }
    if config.quantification == QuantMethod::Pibaq
        && dataset_normalization.is_some_and(|method| method != SampleNormalizationMethod::Quantile)
    {
        return Err(invalid_input(format!(
            "{} supports quantile as its only dataset-level sample normalization",
            config.quantification
        )));
    }
    if config.quantification == QuantMethod::MaxLfq
        && dataset_normalization.is_some_and(|method| {
            !matches!(
                method,
                SampleNormalizationMethod::Quantile | SampleNormalizationMethod::Hierarchical
            )
        })
    {
        return Err(invalid_input(format!(
            "{} supports quantile and hierarchical as its only dataset-level sample normalizations",
            config.quantification
        )));
    }
    let de = &config.differential_expression;
    // `--de-contrast-file` is expanded into `contrasts` before validation runs
    // (see `expand_de_contrasts_file`), so the list reflects both sources.
    if de.enabled && de.contrasts.as_ref().is_none_or(Vec::is_empty) {
        return Err(invalid_input(
            "differential expression requires explicit contrasts via --de-contrast \
             or --de-contrast-file",
        ));
    }
    Ok(())
}

/// Validate a matrix-imputation request; option applicability is checked by the
/// caller that knows which options were supplied.
pub(crate) fn validate_imputation_config(config: &ImputationConfig) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }
    let method = config.method.trim().to_ascii_lowercase();
    match method.as_str() {
        "" | "none" => Err(invalid_input("--impute-method must name a method")),
        "mindet" | "minprob"
            if !config.quantile.is_finite() || !(0.0..=1.0).contains(&config.quantile) =>
        {
            Err(invalid_input("impute-quantile must be between 0 and 1"))
        }
        "knn" | "seqknn" if config.n_neighbors == 0 => Err(invalid_input(
            "impute-n-neighbors must be greater than zero",
        )),
        "mean" | "median" | "constant" | "zero" | "most_frequent" | "knn" | "seqknn" | "impseq"
        | "gms" | "bpca" | "impseqrob" | "qrilc" | "mindet" | "minprob" => Ok(()),
        "missforest" => unsupported("missforest imputation is unported"),
        _ => unsupported("imputation"),
    }
}

/// Validate only the ensemble-specific DE contract before any input access.
/// Other validation keeps its established ordering in the full validators.
pub(crate) fn validate_de_ensemble_options(config: &FeatureToProteinsConfig) -> Result<()> {
    let de = &config.differential_expression;
    if de.enabled && de.method.trim().eq_ignore_ascii_case("ensemble") {
        de::validated_ensemble_member_names(de)?;
    }
    Ok(())
}

/// Resolve the `--de-method auto` sentinel to a concrete method, following
/// Python's quantification-based `_resolve_de_method` default: `directlfq`
/// quantification selects `deqms`, every other quantification selects
/// `limrots`. The LimROTS kernel accepts only complete finite log2 matrices, so
/// `auto` selects `deqms` instead when `matrix_is_complete` reports a missing
/// cell; Python's LimROTS accepts missing values and keeps `limrots`. With
/// `--de-covariate` or `--de-block`, which only limma and DEqMS support, `auto`
/// selects `deqms`.
/// A non-`auto` method is returned unchanged. Resolved just before the DE stage so
/// both the validation and the `de::run_differential_expression` dispatch
/// observe a concrete method.
pub(crate) fn resolve_de_method(
    config: &FeatureToProteinsConfig,
    matrix_is_complete: impl FnOnce() -> bool,
) -> String {
    let method = config.differential_expression.method.trim();
    if !method.eq_ignore_ascii_case("auto") {
        return method.to_string();
    }
    let resolved = if config.differential_expression.covariates.is_empty()
        && config.differential_expression.block.is_none()
        && config.quantification != QuantMethod::DirectLfq
        && matrix_is_complete()
    {
        "limrots"
    } else {
        "deqms"
    };
    info!(
        quant_method = %config.quantification,
        de_method = resolved,
        "auto-selected DE method"
    );
    resolved.to_string()
}

/// Fold `--de-contrast-file` (a TSV with `group1`/`group2` columns) into the
/// `contrasts` list, mirroring Python (features2proteins.py:768): each row
/// appends `"<group1> vs <group2>"` after the repeated `--de-contrast` entries.
/// The returned config owns the merged list and has `contrasts_file` cleared, so
/// validation and the DE stage observe a single resolved contrast list. Called
/// only when `contrasts_file.is_some()`.
pub(crate) fn expand_de_contrasts_file(
    config: &FeatureToProteinsConfig,
) -> Result<FeatureToProteinsConfig> {
    let mut config = config.clone();
    let Some(path) = config.differential_expression.contrasts_file.take() else {
        return Ok(config);
    };
    let mut contrasts = config
        .differential_expression
        .contrasts
        .take()
        .unwrap_or_default();

    let read_error = |source: csv::Error| {
        invalid_input(format!(
            "failed to read DE contrasts file `{}`: {source}",
            path.display()
        ))
    };
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&path)
        .map_err(read_error)?;
    let headers = reader.headers().map_err(read_error)?.clone();
    let (Some(group1), Some(group2)) = (
        headers.iter().position(|header| header == "group1"),
        headers.iter().position(|header| header == "group2"),
    ) else {
        return Err(invalid_input(format!(
            "DE contrasts file `{}` must have `group1` and `group2` columns",
            path.display()
        )));
    };

    for record in reader.records() {
        let record = record.map_err(read_error)?;
        let g1 = record.get(group1).unwrap_or("").trim();
        let g2 = record.get(group2).unwrap_or("").trim();
        if !g1.is_empty() && !g2.is_empty() {
            contrasts.push(format!("{g1} vs {g2}"));
        }
    }

    config.differential_expression.contrasts = (!contrasts.is_empty()).then_some(contrasts);
    Ok(config)
}
