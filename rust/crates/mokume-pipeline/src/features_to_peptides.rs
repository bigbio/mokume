use crate::accessions::protein_group_name;
use crate::aggregation::IntensityFactors;
use crate::exports::{IntermediateExports, PeptideExportOptions, parse_razor_handling};
use crate::intensity_filters::collect_intensity_group_filters;
use crate::irs::collect_irs_scale;
use crate::normalization_factors::{collect_intensity_factors, load_remove_ids};
use crate::qpx_stream::{stream_qpx_features, stream_selected_qpx_features};
use crate::run_qc::{RunQcSource, collect_run_qc_exclusions};
use crate::state::{FeatureToProteinState, LowFrequencyTracker};
use crate::validation::{
    active_filter_pipeline, compile_exclude_sequence_patterns, input_sample_table,
    required_parquet, validate_features_to_peptides_config,
};
use crate::{
    BTreeMap, BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig,
    FeatureToPeptidesConfig, FeatureToProteinsConfig, HashMap, HashSet, ImputationConfig,
    IrsConfig, MaxLfqConfig, MemoryPlan, NamedScoreFilterConfig, NormalizationConfig, OutputConfig,
    OutputFormat, Path, PibaqConfig, PreprocessingFilterConfig, QuantMethod, RatioConfig, Result,
    RuntimeConfig, SdrfLookupCache, TmtConfig, info, invalid_input,
};

pub fn run_features_to_peptides(config: &FeatureToPeptidesConfig) -> Result<()> {
    validate_features_to_peptides_config(config)?;
    let filter_pipeline = active_filter_pipeline(config);
    let named_score = filter_pipeline.and_then(|pipeline| pipeline.peptide.score.as_ref());

    // Route the deterministic peptide export through the protein pipeline's
    // ingest machinery by building a Sum-quantification config whose only
    // requested output is the peptide intermediate.
    let mut proteins_config = peptide_export_config(config);
    let parquet = required_parquet(&proteins_config.input)?;
    // When the opt-in filter pipeline is enabled, its protein block governs
    // contaminant removal and the per-`(protein, sample)` unique-peptide gate
    // (Python wires both through the same config), replacing the default
    // load-time settings.
    if let Some(pipeline) = filter_pipeline {
        proteins_config.filtering.remove_contaminants =
            pipeline.protein.remove_contaminants || pipeline.protein.remove_decoys;
        proteins_config.filtering.min_unique_peptides = pipeline.protein.min_unique_peptides;
    }
    let sdrf = input_sample_table(&proteins_config.input)?;
    // The median pre-pass mirrors Python's `SQLFilterBuilder`, which removes
    // contaminants on the **raw** `pg_accessions` before computing the median.
    // Route the filter pipeline's custom patterns through (default list -> the
    // existing `is_contaminant` fallback inside `matches_sql_contaminant`).
    let median_contaminant_patterns: &[String] =
        filter_pipeline.map_or(&[], |pipeline| &pipeline.protein.contaminant_patterns);
    let mut intensity_factors = if config.skip_normalization {
        IntensityFactors::default()
    } else {
        // The peptide path forwards the user's `--keep-shared-peptides` flag so
        // the median pre-pass includes non-unique rows when requested, matching
        // Python `peptide.py:168/176` (`require_unique = not keep_shared`).
        // `proteins_config` cannot carry this (it pins quantification to `Sum`).
        collect_intensity_factors(
            &proteins_config,
            sdrf.as_ref(),
            median_contaminant_patterns,
            config.keep_shared_peptides,
            &MemoryPlan::unlimited(),
            named_score,
            None,
        )?
    };
    // The IRS pre-pass reuses the median pre-pass's contaminant settings.
    if let Some(irs) = &config.irs {
        let irs_min_intensity =
            filter_pipeline.map_or(0.0, |pipeline| pipeline.intensity.min_intensity);
        let irs_scale_by_run = collect_irs_scale(
            parquet,
            irs,
            sdrf.as_ref(),
            proteins_config.filtering.remove_contaminants,
            median_contaminant_patterns,
            irs_min_intensity,
            named_score,
        )?;
        if irs_scale_by_run.is_empty() {
            return Err(invalid_input(format!(
                "features2peptides IRS channel `{}` produced no scaling factors",
                irs.channel
            )));
        }
        intensity_factors.irs_scale_by_run = irs_scale_by_run;
    }
    let intensity_factors = (!intensity_factors.is_empty()).then_some(&intensity_factors);
    let mut state = FeatureToProteinState::new(&proteins_config, sdrf.as_ref(), None, None)?;
    // `--keep-shared-peptides`: keep non-unique rows during ingest (Python skips
    // the `unique == 1` filter) and skip the per-protein `min_unique` gate at
    // export by forcing the effective threshold to 0 (Python skips the
    // `groupby(PROTEIN_NAME).filter(... >= min_unique)` step).
    state.keep_shared_peptides = config.keep_shared_peptides;
    // `--aggregation_level run`: re-create the peptide export so its keys carry
    // the run dimension. The protein-export path never requests run mode.
    let run_mode = config.aggregation_level.is_run();
    state.export_rows = IntermediateExports::new_with_run_mode(&proteins_config, run_mode);
    if let Some(path) = &config.remove_ids {
        state.remove_protein_ids = load_remove_ids(path)?;
    }
    if config.remove_low_frequency_peptides {
        state.low_frequency = Some(LowFrequencyTracker::default());
    }
    state.peptide_filters = filter_pipeline.cloned();
    if let Some(pipeline) = filter_pipeline {
        configure_fdr_and_sequence_filters(&mut state, parquet, pipeline, named_score)?;
        state.export_rows.rows.razor_handling =
            parse_razor_handling(&pipeline.protein.razor_peptide_handling);
    }
    // Run-QC filters run inside each sample frame and group by technical
    // replicate. Decide which runs to drop in a pre-pass (the median is computed
    // independently in `collect_intensity_factors`, so excluded runs still feed
    // it -- matching Python, where `get_median_map` runs via `SQLFilterBuilder`
    // ahead of the per-sample filter pipeline).
    if let Some(pipeline) = filter_pipeline {
        let source = RunQcSource {
            filtering: proteins_config.filtering,
            keep_shared_peptides: config.keep_shared_peptides,
            min_unique_peptides: proteins_config.filtering.min_unique_peptides,
            contaminant_patterns: &pipeline.protein.contaminant_patterns,
            remove_protein_ids: &state.remove_protein_ids,
        };
        state.run_qc_excluded_runs = collect_run_qc_exclusions(
            parquet,
            sdrf.as_ref(),
            &pipeline.run_qc,
            &source,
            named_score,
        )?;
    }
    // CVThresholdFilter and QuantileFilter are group-level filters: both need the
    // whole sample's `min_unique`-gated intensities, which the streaming ingest
    // cannot see one row at a time. Compute them together in a pre-pass (like
    // Run-QC), over the rows Python's pipeline sees at `intensity.py:293` (post
    // `min_unique` gate, pre feature-normalization). They share the pass so the
    // quantile bounds see the post-CV survivors (Python orders CV before Quantile).
    // Excluded Run-QC technical runs are skipped so they do not feed either
    // decision.
    if let Some(pipeline) = filter_pipeline {
        let source = RunQcSource {
            filtering: proteins_config.filtering,
            keep_shared_peptides: config.keep_shared_peptides,
            min_unique_peptides: proteins_config.filtering.min_unique_peptides,
            contaminant_patterns: &pipeline.protein.contaminant_patterns,
            remove_protein_ids: &state.remove_protein_ids,
        };
        let group_filters = collect_intensity_group_filters(
            parquet,
            sdrf.as_ref(),
            &source,
            &pipeline.intensity,
            &state.run_qc_excluded_runs,
            named_score,
        )?;
        state.cv_dropped = group_filters.cv_dropped;
        state.quantile_bounds = group_filters.quantile_bounds;
        // ReplicateAgreementFilter degeneracy: `min_replicate_agreement > 1` drops
        // every row under Python's per-sample application (nunique(SampleID) == 1),
        // so the whole output is header-only. Set the flag so ingest rejects all
        // features (the warning is emitted in `validate_filter_pipeline_subset`).
        state.replicate_agreement_wipes_all = pipeline.intensity.min_replicate_agreement > 1;
    }
    let mut sdrf_lookup = sdrf.as_ref().map(SdrfLookupCache::new);
    let with_modifications =
        filter_pipeline.is_some_and(|pipeline| !pipeline.peptide.exclude_modifications.is_empty());
    stream_selected_qpx_features(
        parquet,
        &MemoryPlan::unlimited(),
        named_score,
        with_modifications,
        |feature| {
            state.ingest(
                &feature,
                sdrf_lookup.as_mut(),
                proteins_config.filtering,
                intensity_factors,
                None,
            )
        },
    )?;

    info!(
        accepted_features = state.accepted_features,
        accepted_measurements = state.accepted_measurements,
        proteins = state.proteins.len(),
        samples = state.samples.len(),
        output = %config.output.display(),
        "features2peptides aggregation finished"
    );

    // `--keep-shared-peptides` skips the per-protein `min_unique` gate
    // (peptide.py:278-281, guarded by `if not keep_shared_peptides`). But the
    // filter pipeline's `MinPeptideFilter` (peptide.py:291-293 -> protein.py:134-146)
    // runs UNCONDITIONALLY. `allowed_cells` expresses both gates through one
    // threshold, so under keep_shared it may only drop to 0 when no filter pipeline
    // supplies a MinPeptide gate; otherwise it must honour the pipeline's
    // `min_unique_peptides` (the Run-QC / quantile pre-passes stay at 0 because they
    // run ahead of MinPeptideFilter).
    let min_unique = if config.keep_shared_peptides {
        filter_pipeline.map_or(0, |pipeline| pipeline.protein.min_unique_peptides)
    } else {
        proteins_config.filtering.min_unique_peptides
    };
    state.write_intermediate_exports(
        &proteins_config,
        &state.allowed_cells(min_unique),
        PeptideExportOptions {
            log2: config.log2,
            save_parquet: config.save_parquet,
        },
        // The standalone `features2peptides` command (Python
        // `mokume.normalization.peptide.peptide_normalization`) has NO post-loop
        // dataset-level pass: its per-sample loop calls the placeholder
        // `PeptideNormalizationMethod` fns (model/normalization.py:416-453),
        // which return the frame unchanged, then writes each sample
        // incrementally (peptide.py:407-410). The `_apply_dataset_normalization`
        // pivot+normalize+melt (stages.py:604) lives in `LoadingStage`, which is
        // consumed ONLY by `features_to_proteins.py:300` -- the protein command,
        // not this one. So the dataset-level methods are a deterministic no-op
        // here: the exported peptides carry only the run/factor normalization
        // applied during ingest, byte-identical to `--sample-normalization none`
        // (verified on PXD003539: quantile/mediancenter == sample=none).
        None,
    )
}

fn configure_fdr_and_sequence_filters(
    state: &mut FeatureToProteinState,
    parquet: &Path,
    pipeline: &PreprocessingFilterConfig,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<()> {
    state.excluded_sequence_regexes =
        compile_exclude_sequence_patterns(&pipeline.peptide.exclude_sequence_patterns)?;
    state.protein_fdr_allowed = collect_fdr_filter_state(
        parquet,
        pipeline.peptide.fdr_threshold,
        pipeline.protein.fdr_threshold,
        named_score,
    )?;
    Ok(())
}

/// Validate that explicitly requested QPX q-value fields carry usable values
/// and collect the protein groups passing the group-minimum protein FDR rule.
/// The pre-pass happens before output creation, so an unavailable q-value cannot
/// turn a requested filter into a successful no-op or a misleading empty file.
///
/// When a protein threshold is requested, the same pass records which
/// (non-decoy) protein groups each run observes, so the removal can be reported
/// per run and for the whole experiment (see [`log_protein_fdr_report`]).
pub(crate) fn collect_fdr_filter_state(
    parquet: &Path,
    peptide_threshold: Option<f64>,
    protein_threshold: Option<f64>,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<Option<HashSet<String>>> {
    if peptide_threshold.is_none() && protein_threshold.is_none() {
        return Ok(None);
    }

    let mut peptide_values = 0_usize;
    let mut protein_values = 0_usize;
    let mut protein_min_qvalue: HashMap<String, f64> = HashMap::new();
    let mut group_ids: HashMap<String, u32> = HashMap::new();
    let mut run_groups: BTreeMap<String, HashSet<u32>> = BTreeMap::new();
    stream_qpx_features(parquet, &MemoryPlan::unlimited(), named_score, |feature| {
        if peptide_threshold.is_some()
            && feature
                .peptide_qvalue
                .is_some_and(|qvalue| qvalue.is_finite())
        {
            peptide_values += 1;
        }
        if protein_threshold.is_none() {
            return Ok(());
        }
        let Some(protein) = protein_group_name(&feature.protein_accessions) else {
            return Ok(());
        };
        if !feature.is_decoy.unwrap_or(false) {
            let id = match group_ids.get(&protein) {
                Some(&id) => id,
                None => {
                    let id = u32::try_from(group_ids.len())
                        .map_err(|_| invalid_input("protein FDR report id overflow"))?;
                    group_ids.insert(protein.clone(), id);
                    id
                }
            };
            if let Some(groups) = run_groups.get_mut(&feature.run_file_name) {
                groups.insert(id);
            } else {
                run_groups.insert(feature.run_file_name.clone(), HashSet::from([id]));
            }
        }
        if let Some(qvalue) = feature.pg_global_qvalue.filter(|qvalue| qvalue.is_finite()) {
            protein_values += 1;
            protein_min_qvalue
                .entry(protein)
                .and_modify(|known| *known = known.min(qvalue))
                .or_insert(qvalue);
        }
        Ok(())
    })?;

    if peptide_threshold.is_some() && peptide_values == 0 {
        return Err(invalid_input(
            "--filter-peptide-fdr requires a populated QPX `peptide_qvalue` column",
        ));
    }
    let Some(threshold) = protein_threshold else {
        return Ok(None);
    };
    if protein_values == 0 {
        return Err(invalid_input(
            "--filter-protein-fdr requires a populated QPX `pg_global_qvalue` column",
        ));
    }
    let allowed: HashSet<String> = protein_min_qvalue
        .iter()
        .filter(|&(_, &qvalue)| qvalue <= threshold)
        .map(|(protein, _)| protein.clone())
        .collect();
    let report = ProteinFdrReport::new(&group_ids, &protein_min_qvalue, &run_groups, threshold);
    log_protein_fdr_report(&report);
    Ok(Some(allowed))
}

/// Protein groups before/after `--filter-protein-fdr`, per run and for the
/// whole experiment. Counts are distinct non-decoy protein groups observed in
/// the QPX input before any other filter; a group without a finite
/// `pg_global_qvalue` counts as removed (it can never satisfy the cutoff).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProteinFdrReport {
    threshold: f64,
    pub(crate) groups_before: usize,
    pub(crate) groups_after: usize,
    pub(crate) groups_without_qvalue: usize,
    pub(crate) runs: Vec<ProteinFdrRunCount>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProteinFdrRunCount {
    pub(crate) run: String,
    pub(crate) before: usize,
    pub(crate) after: usize,
}

impl ProteinFdrReport {
    pub(crate) fn new(
        group_ids: &HashMap<String, u32>,
        protein_min_qvalue: &HashMap<String, f64>,
        run_groups: &BTreeMap<String, HashSet<u32>>,
        threshold: f64,
    ) -> Self {
        let mut passes = vec![false; group_ids.len()];
        let mut groups_without_qvalue = 0_usize;
        for (protein, &id) in group_ids {
            match protein_min_qvalue.get(protein) {
                Some(&qvalue) => passes[id as usize] = qvalue <= threshold,
                None => groups_without_qvalue += 1,
            }
        }
        let runs = run_groups
            .iter()
            .map(|(run, groups)| ProteinFdrRunCount {
                run: run.clone(),
                before: groups.len(),
                after: groups.iter().filter(|&&id| passes[id as usize]).count(),
            })
            .collect();
        Self {
            threshold,
            groups_before: group_ids.len(),
            groups_after: passes.iter().filter(|&&pass| pass).count(),
            groups_without_qvalue,
            runs,
        }
    }
}

fn log_protein_fdr_report(report: &ProteinFdrReport) {
    for run in &report.runs {
        info!(
            run = %run.run,
            protein_groups_before = run.before,
            protein_groups_after = run.after,
            protein_groups_removed = run.before - run.after,
            threshold = report.threshold,
            "protein FDR filter (run)"
        );
    }
    info!(
        runs = report.runs.len(),
        protein_groups_before = report.groups_before,
        protein_groups_after = report.groups_after,
        protein_groups_removed = report.groups_before - report.groups_after,
        protein_groups_without_qvalue = report.groups_without_qvalue,
        threshold = report.threshold,
        "protein FDR filter (experiment): kept groups whose best pg_global_qvalue <= threshold"
    );
}

/// Build the internal `FeatureToProteinsConfig` that drives the peptide export.
/// Sum quantification is chosen because the peptide intermediate is emitted
/// before any protein rollup, so the quantification method never affects the
/// peptide matrix; only `output.export_peptides`, the filter thresholds, and the
/// run/sample normalization are consulted on this path.
fn peptide_export_config(config: &FeatureToPeptidesConfig) -> FeatureToProteinsConfig {
    FeatureToProteinsConfig {
        input: config.input.clone(),
        output: OutputConfig {
            protein_matrix: config.output.clone(),
            export_peptides: Some(config.output.clone()),
            export_unnormalized_peptides: None,
            export_unimputed: None,
            export_ions: None,
            export_precision: None,
            format: OutputFormat::PythonCompatible,
        },
        filtering: config.filtering,
        normalization: NormalizationConfig {
            run_method: if config.skip_normalization {
                "none".to_owned()
            } else {
                config.run_normalization.clone()
            },
            sample_method: if config.skip_normalization {
                "none".to_owned()
            } else {
                config.sample_normalization.clone()
            },
            normalization_proteins: None,
        },
        quantification: QuantMethod::Sum,
        topn_peptides: 3,
        maxlfq: MaxLfqConfig::default(),
        pibaq: PibaqConfig::default(),
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig::default(),
        tmt: TmtConfig::default(),
        coverage_threshold: None,
        sample_correlation_threshold: None,
        protein_fdr_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: None,
        },
    }
}
