use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{create_dir_all, read_to_string, File};
use std::path::{Path, PathBuf};

use csv::WriterBuilder;
use mokume_core::{
    BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig, FeatureToPeptidesConfig,
    FeatureToProteinsConfig, FilterConfig, ImputationConfig, InputConfig, IntensityFilterConfig,
    IrsChannelConfig, IrsConfig, IrsScope, IrsStat, MaxLfqConfig, MokumeError,
    NamedScoreFilterConfig, NormalizationConfig, OutputConfig, OutputFormat, PeptideId,
    PibaqConfig, PreprocessingFilterConfig, ProteinId, QuantMethod, RatioConfig, Result,
    RunQcFilterConfig, RuntimeConfig, SampleId, StringIdRegistry,
};
use mokume_imputation::imputed_values;
use mokume_io::{
    write_peptide_parquet, MsstatsReader, PeptideParquetRow, QpxFeatureRecord, QpxParquetReader,
    SdrfRawTable, SdrfRecord, SdrfTable, DEFAULT_QPX_BATCH_SIZE,
};
use mokume_normalization::{
    condition_median_sample_factors, coverage_filtered_proteins, global_median_sample_factors,
    irs_scaling_factors, mean_positive, median, median_finite, parse_run_normalization_method,
    parse_sample_normalization_method, quantile_linear, run_normalization_transforms,
    tmm_norm_factors, RunCellKey, RunNormalizationMethod, RunNormalizationTransform,
    SampleNormalizationMethod,
};
use mokume_quant::{
    direct_lfq_aligned, direct_lfq_aligned_with_ions, max_lfq_with_samples, DirectLfqIon,
    DirectLfqNormalizedIon, PeptideMeasurement,
};
use rayon::prelude::*;
use regex::{Regex, RegexBuilder};
use tracing::{info, warn};

mod de;
pub mod filters;
mod matrix;
mod memory;
mod spectral_count;
mod threading;

use memory::MemoryPlan;

pub use de::{differential_expression_matrix, MatrixDifferentialExpressionResults};
pub use matrix::{impute_matrix, normalize_matrix};

/// `min_nonan` used when `--quant-method maxlfq` delegates to the DirectLFQ-aligned
/// solver. Matches Python's maxlfq delegation, which uses 2 (the streaming path
/// hardcodes `min_nonan=2`; the class path passes `min_peptides`, default 2).
const MAXLFQ_DIRECTLFQ_MIN_NONAN: usize = 2;
const DEFAULT_REFERENCE_REGEX: &str = "pool|powder|ref|reference|bridge";
const MIN_SAMPLE_CORRELATION_OVERLAP: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellKey {
    protein: ProteinId,
    sample: SampleId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PeptideCellKey {
    protein: ProteinId,
    sample: SampleId,
    peptide: PeptideId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RunQcKey {
    sample: String,
    technical_replicate: i64,
}

/// Key for the DirectLFQ ion table: one entry per (protein, canonical
/// sequence, sample), holding the summed linear intensity. DirectLFQ's "ion"
/// is the bare sequence, so this keys on the canonical peptide id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DirectLfqCellKey {
    protein: ProteinId,
    canonical: PeptideId,
    sample: SampleId,
}

#[derive(Debug)]
struct FeatureToProteinState {
    proteins: StringIdRegistry<ProteinId>,
    peptides: StringIdRegistry<PeptideId>,
    canonical_peptides: StringIdRegistry<PeptideId>,
    conditions: StringIdRegistry<u32>,
    loading_contexts: HashMap<(u32, u32), u32>,
    contextual_peptides: HashMap<(PeptideId, u32), PeptideId>,
    contextual_canonical_peptides: HashMap<(PeptideId, u32), PeptideId>,
    samples: StringIdRegistry<SampleId>,
    peptide_to_canonical: HashMap<PeptideId, PeptideId>,
    peptide_to_contextual_canonical: HashMap<PeptideId, PeptideId>,
    unique_peptides: HashMap<CellKey, HashSet<PeptideId>>,
    export_rows: IntermediateExports,
    aggregation: FeatureAggregation,
    accepted_features: usize,
    accepted_measurements: usize,
    /// Exact protein accessions to drop (`--remove_ids`). A feature is rejected
    /// when any parsed accession matches an entry. Empty on
    /// the protein pipeline, which has no `--remove_ids` option.
    remove_protein_ids: HashSet<String>,
    /// Per-`(ProteinName, canonical sequence)` distinct-sample tracking for the
    /// `features2peptides` low-frequency filter. `None` disables tracking (the
    /// protein pipeline and `features2peptides` without
    /// `--remove_low_frequency_peptides`).
    low_frequency: Option<LowFrequencyTracker>,
    /// Keep shared/non-unique peptide rows (Python `--keep-shared-peptides`).
    /// `false` on the protein pipeline, where the per-aggregation unique policy
    /// (`keeps_shared_peptides`) governs instead; set by `features2peptides`.
    keep_shared_peptides: bool,
    /// Opt-in `features2peptides` per-row preprocessing filters (Python
    /// `--filter-config` / `--filter-*`). `None` keeps default load-time
    /// filtering only; `Some` applies the per-row filters during ingest, before
    /// the per-`(protein, sample)` unique-peptide gate (matching Python's
    /// pipeline chain). Group-level filters are validated as unsupported upstream.
    peptide_filters: Option<PreprocessingFilterConfig>,
    /// Protein groups that pass an explicitly requested protein-group q-value
    /// cutoff. `None` means no protein FDR filter was requested; `Some` is
    /// populated by a source pre-pass so every row of a passing protein group is
    /// retained, matching the Python group-minimum FDR contract.
    protein_fdr_allowed: Option<HashSet<String>>,
    /// `(sample, technical replicate)` runs dropped by the Run-QC group filters
    /// (Python `run_qc.py`: total intensity, feature/protein counts and missing
    /// rate). Computed in a pre-pass over the same initial-filtered features;
    /// ingest skips only the rejected technical run, not its whole sample. Empty
    /// when no Run-QC threshold is active. The normalization median is computed
    /// before Run-QC, matching Python's `SQLFilterBuilder` ordering.
    run_qc_excluded_runs: HashSet<RunQcKey>,
    /// Pre-compiled `exclude_sequence_patterns` regexes (Python
    /// `SequencePatternFilter`, peptide.py:415-418). Empty unless a
    /// `features2peptides` filter pipeline supplies non-empty patterns; compiled
    /// once so the per-row check never recompiles. This filter only drops rows in
    /// the ingest pipeline (not in the `SQLFilterBuilder` median pre-pass), so it
    /// does not shift the normalization median -- no two-pass handling is needed.
    excluded_sequence_regexes: Vec<Regex>,
    /// Per-sample `(lower, upper)` intensity bounds for the QuantileFilter (Python
    /// `intensity.py:293-297`): a feature is dropped when its raw intensity falls
    /// outside `[lower, upper]` (double-closed). Empty unless `quantile_lower > 0`
    /// or `quantile_upper < 1`. The bounds are computed in a pre-pass over the rows
    /// that survive the per-protein `min_unique` gate (Python applies that gate
    /// before the filter pipeline, peptide.py:278-281), so singleton-protein
    /// intensities do not move the bounds. Like Run-QC, the quantile drop happens
    /// after the median pre-pass, so it does not shift the normalization median.
    quantile_bounds: HashMap<String, (f64, f64)>,
    /// `(sample, ProteinName, PeptideCanonical)` triples dropped by the
    /// CVThresholdFilter (Python `intensity.py:111-154`): within one sample, a
    /// canonical whose raw intensities have a coefficient of variation (ddof = 1)
    /// strictly greater than `cv_threshold` is removed (a `NaN`/`None` CV --
    /// single measurement or non-positive mean -- passes). Computed in the same
    /// pre-pass as the quantile bounds so the bounds see the post-CV survivors
    /// (Python orders CV before Quantile, `intensity.py` factory). Empty unless
    /// `cv_threshold` is set. Ingest skips these triples before the quantile drop.
    cv_dropped: HashSet<(String, String, String)>,
    /// When set, the ReplicateAgreementFilter degenerates to a whole-output wipe
    /// (`min_replicate_agreement > 1`): Python's per-sample `nunique(SampleID)` is
    /// always 1, so the `>= 2` threshold drops every row. Ingest rejects all
    /// features so the output is header-only, matching Python.
    replicate_agreement_wipes_all: bool,
}

impl FeatureToProteinState {
    fn new(
        config: &FeatureToProteinsConfig,
        sdrf: Option<&SdrfTable>,
        raw_sdrf: Option<&SdrfRawTable>,
        pibaq_digest: Option<PibaqDigest>,
    ) -> Result<Self> {
        Ok(Self {
            proteins: StringIdRegistry::new(),
            peptides: StringIdRegistry::new(),
            canonical_peptides: StringIdRegistry::new(),
            conditions: StringIdRegistry::new(),
            loading_contexts: HashMap::new(),
            contextual_peptides: HashMap::new(),
            contextual_canonical_peptides: HashMap::new(),
            samples: StringIdRegistry::new(),
            peptide_to_canonical: HashMap::new(),
            peptide_to_contextual_canonical: HashMap::new(),
            unique_peptides: HashMap::new(),
            export_rows: IntermediateExports::new(config),
            aggregation: FeatureAggregation::from_config(config, sdrf, raw_sdrf, pibaq_digest)?,
            accepted_features: 0,
            accepted_measurements: 0,
            remove_protein_ids: HashSet::new(),
            low_frequency: None,
            keep_shared_peptides: false,
            peptide_filters: None,
            protein_fdr_allowed: None,
            run_qc_excluded_runs: HashSet::new(),
            excluded_sequence_regexes: Vec::new(),
            quantile_bounds: HashMap::new(),
            cv_dropped: HashSet::new(),
            replicate_agreement_wipes_all: false,
        })
    }

    fn ingest(
        &mut self,
        feature: &QpxFeatureRecord,
        sdrf: Option<&SdrfTable>,
        filtering: FilterConfig,
        intensity_factors: Option<&IntensityFactors>,
    ) -> Result<()> {
        // ReplicateAgreementFilter degeneracy (Python `intensity.py:194-242` under
        // the per-sample pipeline): `min_replicate_agreement > 1` drops every row
        // because the per-sample `nunique(SampleID)` is always 1. No feature is
        // ingested, so the output is header-only, matching Python.
        if self.replicate_agreement_wipes_all {
            return Ok(());
        }
        let keep_shared = self.keep_shared_peptides || self.aggregation.keeps_shared_peptides();
        if !passes_feature_filter(feature, filtering, keep_shared) {
            return Ok(());
        }

        let Some(protein_group) = self.aggregation.protein_name(feature) else {
            return Ok(());
        };
        // Ingest-time contaminant removal mirrors Python's `ContaminantFilter`:
        // match the parsed `protein_group` (uppercased, literal substring) against
        // the filter pipeline's patterns. With no pipeline (or the default list)
        // this falls back to `is_contaminant`, preserving the default parity.
        let contaminant_patterns: &[String] = self
            .peptide_filters
            .as_ref()
            .map_or(&[], |config| &config.protein.contaminant_patterns);
        let protein_group = if filtering.remove_contaminants {
            self.aggregation
                .resolve_contaminant_twins(&protein_group)
                .unwrap_or(protein_group)
        } else {
            protein_group
        };
        if self.rejects_protein_group(
            &protein_group,
            filtering.remove_contaminants,
            contaminant_patterns,
        ) {
            return Ok(());
        }
        let protein_group = if filtering.remove_contaminants {
            strip_contaminant_members(&protein_group, contaminant_patterns).unwrap_or(protein_group)
        } else {
            protein_group
        };
        if has_removed_accession(&feature.protein_accessions, &self.remove_protein_ids) {
            return Ok(());
        }
        // Opt-in per-row preprocessing filters run after the load-time filters
        // and before the per-`(protein, sample)` unique-peptide gate, matching
        // Python's pipeline chain (intensity/peptide filters before MinPeptide).
        if !self.passes_peptide_filter_pipeline(feature) {
            return Ok(());
        }

        let sdrf_record = sdrf_record(feature, sdrf)?;
        let sample_name = sample_name(feature, sdrf_record);
        // Python applies Run-QC within a per-sample frame but groups its rows by
        // `TechReplicate`. The pre-pass therefore rejects individual technical
        // runs; other runs from the same sample must remain available.
        let run_qc_key = run_qc_key(feature, sdrf_record, sample_name.clone());
        if self.run_qc_excluded_runs.contains(&run_qc_key) {
            return Ok(());
        }
        // CVThresholdFilter (Python `intensity.py:127-141`): drop every row of a
        // `(sample, ProteinName, PeptideCanonical)` whose within-sample CV exceeds
        // `cv_threshold`. The dropped set is precomputed over the raw,
        // `min_unique`-gated intensities; Python runs CV *before* the quantile and
        // before the per-row peptide filters (length/charge/modification/missed
        // cleavage), so the CV input includes rows those later filters would drop
        // -- the pre-pass mirrors that by buffering the same load-gated rows. Empty
        // unless `cv_threshold` is set.
        if !self.cv_dropped.is_empty()
            && self.cv_dropped.contains(&(
                sample_name.clone(),
                protein_group.clone(),
                feature.sequence.clone(),
            ))
        {
            return Ok(());
        }
        // QuantileFilter (Python `intensity.py:296-297`): drop rows whose raw
        // intensity is outside the per-sample `[lower, upper]` bounds (double-closed
        // `>= && <=`). The bounds are precomputed over the `min_unique`-gated set;
        // computing in f64 matches pandas, which upcasts the f32 column to f64 for
        // `quantile` and the comparison. The export-time `min_unique` gate then
        // re-counts canonicals on the survivors, reproducing the post-quantile
        // MinPeptideFilter.
        if let Some(&(lower, upper)) = self.quantile_bounds.get(&sample_name) {
            if !(feature.intensity >= lower && feature.intensity <= upper) {
                return Ok(());
            }
        }
        let sample = register_id(&mut self.samples, &sample_name, "sample")?;
        let peptide_key = self.aggregation.peptide_key(feature);
        let base_peptide = register_id(&mut self.peptides, &peptide_key, "peptide")?;
        let canonical_peptide = register_id(
            &mut self.canonical_peptides,
            &feature.sequence,
            "canonical peptide",
        )?;
        let biological_replicate = sdrf_record
            .and_then(|record| record.biological_replicate)
            .unwrap_or(1);
        let condition = sample_condition(&sample_name, sdrf_record);
        let (peptide, contextual_canonical_peptide) = if self.aggregation.preserves_loading_rows() {
            let condition_id = self
                .conditions
                .get_or_insert(&condition)
                .ok_or_else(|| invalid_input("condition id registry overflow"))?;
            let context = register_loading_context(
                &mut self.loading_contexts,
                (biological_replicate, condition_id),
            )?;
            let peptide = register_contextual_peptide(
                &mut self.contextual_peptides,
                (base_peptide, context),
                "peptide",
            )?;
            let contextual_canonical = if self.aggregation.preserves_contextual_canonical_rows() {
                register_contextual_peptide(
                    &mut self.contextual_canonical_peptides,
                    (canonical_peptide, context),
                    "contextual canonical peptide",
                )?
            } else {
                canonical_peptide
            };
            (peptide, contextual_canonical)
        } else {
            (base_peptide, canonical_peptide)
        };
        self.peptide_to_canonical
            .entry(peptide)
            .or_insert(canonical_peptide);
        if self.aggregation.preserves_contextual_canonical_rows() {
            self.peptide_to_contextual_canonical
                .entry(peptide)
                .or_insert(contextual_canonical_peptide);
        }
        let protein = register_id(&mut self.proteins, &protein_group, "protein")?;
        let cell = CellKey { protein, sample };

        self.unique_peptides
            .entry(cell)
            .or_default()
            .insert(canonical_peptide);
        let intensity = intensity_factors.map_or(feature.intensity, |factors| {
            factors.normalize(feature.intensity, &sample_name, &feature.run_file_name)
        });
        self.export_rows.push(IntermediateMeasurement {
            protein,
            sample,
            peptide,
            intensity,
            protein_name: &protein_group,
            peptide_canonical: &feature.sequence,
            sample_name: &sample_name,
            run_file_name: &feature.run_file_name,
            sdrf_record,
        });
        self.aggregation.push(AggregationMeasurement {
            protein,
            sample,
            peptide,
            canonical: canonical_peptide,
            intensity,
            ion_name: &feature.sequence,
            sample_name: &sample_name,
            charge: feature.charge,
        })?;
        if let Some(tracker) = &mut self.low_frequency {
            tracker.observe(&protein_group, &feature.sequence, sample);
        }
        self.accepted_measurements += 1;
        self.accepted_features += 1;
        Ok(())
    }

    fn rejects_protein_group(
        &self,
        protein_group: &str,
        remove_contaminants: bool,
        contaminant_patterns: &[String],
    ) -> bool {
        (remove_contaminants && matches_protein_contaminant(protein_group, contaminant_patterns))
            || self
                .protein_fdr_allowed
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(protein_group))
    }

    /// Apply the opt-in `features2peptides` per-row preprocessing filters
    /// (Python pipeline's per-feature filters: intensity floor, peptide length,
    /// charge state, excluded modifications, missed cleavages). Returns `true`
    /// when no pipeline is configured or the feature survives every filter. The
    /// group-level filters (CV, run QC) are rejected upstream in
    /// `validate_features_to_peptides_config`, so only per-row filters reach here.
    fn passes_peptide_filter_pipeline(&self, feature: &QpxFeatureRecord) -> bool {
        let Some(config) = &self.peptide_filters else {
            return true;
        };
        // MinIntensityFilter: `remove_zero_intensity` forces a 1e-10 floor even
        // when `min_intensity` is 0. Compared against the raw intensity, matching
        // Python's filter on `NORM_INTENSITY` before feature normalization.
        let min_intensity = filters::effective_min_intensity(
            config.intensity.min_intensity,
            config.intensity.remove_zero_intensity,
        );
        if feature.intensity < min_intensity {
            return false;
        }
        // PeptideLengthFilter on the canonical (modification-stripped) sequence.
        let length = filters::canonical_aa_length(&feature.sequence);
        if length < config.peptide.min_peptide_length || length > config.peptide.max_peptide_length
        {
            return false;
        }
        // ChargeStateFilter.
        if let Some(allowed) = &config.peptide.allowed_charge_states {
            if !filters::charge_is_allowed(feature.charge, allowed) {
                return false;
            }
        }
        // ModificationFilter, matched against the modified sequence.
        if filters::peptide_has_excluded_modification(
            &feature.peptidoform,
            &config.peptide.exclude_modifications,
        ) {
            return false;
        }
        // MissedCleavageFilter (trypsin), on the canonical sequence.
        if let Some(max) = config.peptide.max_missed_cleavages {
            if filters::trypsin_missed_cleavages(&feature.sequence) > max {
                return false;
            }
        }
        if let Some(threshold) = config.peptide.fdr_threshold {
            if !feature
                .peptide_qvalue
                .is_some_and(|qvalue| qvalue.is_finite() && qvalue <= threshold)
            {
                return false;
            }
        }
        // SequencePatternFilter (peptide.py:415-418): drop canonical sequences
        // matching any user regex. Patterns are pre-compiled in
        // `excluded_sequence_regexes`; empty unless the pipeline sets them.
        if filters::sequence_matches_excluded(&feature.sequence, &self.excluded_sequence_regexes) {
            return false;
        }
        true
    }

    fn write_intermediate_exports(
        &mut self,
        config: &FeatureToProteinsConfig,
        min_unique_peptides: usize,
        options: PeptideExportOptions,
        dataset_normalization: Option<SampleNormalizationMethod>,
    ) -> Result<()> {
        let allowed_cells = self.allowed_cells(min_unique_peptides);
        let low_frequency_peptides = self
            .low_frequency
            .as_ref()
            .map(LowFrequencyTracker::low_frequency_peptides)
            .unwrap_or_default();
        // When a dataset-level sample normalization is requested, the exported
        // peptides must carry it: Python applies it in `load_for_mokume` before
        // writing the peptide CSV. Reuse the exact aggregation helper on cells
        // rebuilt from the export rows so the export matches the protein matrix.
        let dataset_normalized = dataset_normalization
            .map(|method| self.export_dataset_normalized_values(method, &allowed_cells));
        self.export_rows.write(
            config,
            &allowed_cells,
            &low_frequency_peptides,
            options,
            dataset_normalized.as_ref(),
        )?;
        self.aggregation.write_intermediate_exports(
            config,
            &self.proteins,
            &self.canonical_peptides,
            &self.samples,
            &allowed_cells,
        )
    }

    /// Dataset-level-normalized canonical peptide values for `--export-peptides`,
    /// keyed by `(protein, sample, canonical sequence)`. Rebuilds the same
    /// max-merged `(protein, sample) -> {peptide: intensity}` cells the
    /// aggregation holds (both come from `push_peptide_max` over the same
    /// ingest), then applies the identical [`apply_dataset_norm_to_peptide_cells`]
    /// helper, so the exported peptides match the protein-matrix normalization.
    fn export_dataset_normalized_values(
        &self,
        method: SampleNormalizationMethod,
        allowed_cells: &HashSet<CellKey>,
    ) -> HashMap<(ProteinId, SampleId, String), f64> {
        let mut result = HashMap::new();
        let Some(export) = self.export_rows.peptides.as_ref() else {
            return result;
        };
        let mut cells: HashMap<CellKey, HashMap<PeptideId, f64>> = HashMap::new();
        for (key, peptides) in &export.rows {
            let cell = CellKey {
                protein: key.protein,
                sample: key.sample,
            };
            let target = cells.entry(cell).or_default();
            for (peptide, intensity) in peptides {
                let current = target.entry(*peptide).or_insert(0.0);
                if *intensity > *current {
                    *current = *intensity;
                }
            }
        }
        apply_dataset_norm_to_peptide_cells(
            &mut cells,
            method,
            allowed_cells,
            &self.peptide_to_canonical,
            &self.samples,
        );
        for (cell, canonical_values) in cells {
            for (canonical, value) in canonical_values {
                if let Some(sequence) = self.canonical_peptides.resolve(canonical) {
                    result.insert((cell.protein, cell.sample, sequence.to_owned()), value);
                }
            }
        }
        result
    }

    fn into_matrix(
        mut self,
        min_unique_peptides: usize,
        dataset_normalization: Option<SampleNormalizationMethod>,
    ) -> ProteinMatrix {
        let allowed_cells = self.allowed_cells(min_unique_peptides);
        // Per-protein unique-canonical-peptide counts for the DEqMS DE path,
        // captured BEFORE the `min_unique_peptides` cell filter so they mirror
        // Python's `groupby("anchor_protein")["sequence"].nunique()` over the
        // whole (unfiltered) parquet (stages.py:1810).
        let peptide_counts = self.protein_peptide_counts();
        if let Some(method) = dataset_normalization {
            if self.aggregation.applies_dataset_normalization() {
                self.aggregation.apply_dataset_normalization(
                    method,
                    &allowed_cells,
                    &self.peptide_to_canonical,
                    &self.samples,
                );
            }
        }
        let empty_mapping = HashMap::new();
        // TMM scales contextual peptidoforms in place, so `finalize` must still
        // collapse them; the other dataset normalizers write canonical cells.
        let collapse_mapping = if dataset_normalization.is_some()
            && dataset_normalization != Some(SampleNormalizationMethod::Tmm)
            && self.aggregation.applies_dataset_normalization()
        {
            &empty_mapping
        } else {
            &self.peptide_to_contextual_canonical
        };
        let values = self.aggregation.finalize(
            &allowed_cells,
            &mut self.proteins,
            collapse_mapping,
            &self.canonical_peptides,
            &self.samples,
        );
        let allowed_proteins = values.protein_ids();
        ProteinMatrix {
            proteins: self.proteins,
            samples: self.samples,
            allowed_proteins,
            excluded_samples: HashSet::new(),
            peptide_counts,
            values,
        }
    }

    /// Union the canonical-peptide sets across all of a protein's (protein,
    /// sample) cells into one per-protein unique-peptide count. This is the
    /// Rust equivalent of Python's `groupby("anchor_protein")["sequence"]
    /// .nunique()`: the canonical id is registered from `feature.sequence`
    /// (Python's `sequence` column), and the count is taken over the whole
    /// ingested set, with no `min_unique_peptides` filter applied.
    fn protein_peptide_counts(&self) -> HashMap<ProteinId, usize> {
        let mut by_protein = HashMap::<ProteinId, HashSet<PeptideId>>::new();
        for (cell, peptides) in &self.unique_peptides {
            by_protein
                .entry(cell.protein)
                .or_default()
                .extend(peptides.iter().copied());
        }
        by_protein
            .into_iter()
            .map(|(protein, peptides)| (protein, peptides.len()))
            .collect()
    }

    fn allowed_cells(&self, min_unique_peptides: usize) -> HashSet<CellKey> {
        self.unique_peptides
            .iter()
            .filter_map(|(cell, peptides)| (peptides.len() >= min_unique_peptides).then_some(*cell))
            .collect()
    }
}

/// Tracks, per `(ProteinName, canonical sequence)`, the distinct samples in
/// which a peptide is observed, plus the total distinct sample count. This is
/// the Rust equivalent of Python's `Feature.get_low_frequency_peptides`
/// (feature.py:447): a peptide is "low frequency" when it is seen in fewer than
/// `percentage * len(samples)` samples (`percentage` defaults to 0.2).
///
/// Two faithful approximations of the Python query on the deterministic
/// no-SDRF export path:
///   * the `(protein, sequence)` keys are the joined `ProteinName` and the
///     canonical sequence, exactly the columns the removal later matches
///     against (`peptide.py:373-378`). Python instead groups its count by the
///     raw `anchor_protein` / first `pg_accessions` entry, so for multi-
///     accession protein groups Python's removal silently no-ops (the single
///     accession never equals the joined name) while this keeps the per-group
///     count; the two agree for single-accession proteins.
///   * the sample counts come from the fully filtered ingest stream
///     (length/unique/intensity/decoy), whereas Python's total counts every
///     sample with any positive-intensity feature. These agree whenever every
///     sample contributes at least one accepted feature.
#[derive(Debug, Default)]
struct LowFrequencyTracker {
    peptide_samples: HashMap<(String, String), HashSet<SampleId>>,
    all_samples: HashSet<SampleId>,
}

impl LowFrequencyTracker {
    const LOW_FREQUENCY_PERCENTAGE: f64 = 0.2;

    fn observe(&mut self, protein_name: &str, sequence: &str, sample: SampleId) {
        self.all_samples.insert(sample);
        self.peptide_samples
            .entry((protein_name.to_owned(), sequence.to_owned()))
            .or_default()
            .insert(sample);
    }

    /// The `(ProteinName, canonical sequence)` pairs to remove. Mirrors Python:
    /// the filter is skipped entirely when there is at most one sample
    /// (`peptide.py:372` gates on `len(sample_names) > 1`), and otherwise a
    /// peptide is removed when its distinct-sample count is strictly below
    /// `percentage * total` (`feature.py:507-510` drops rows with
    /// `count >= percentage * len(samples)`, keeping the complement).
    fn low_frequency_peptides(&self) -> HashSet<(String, String)> {
        if self.all_samples.len() <= 1 {
            return HashSet::new();
        }
        let threshold = Self::LOW_FREQUENCY_PERCENTAGE * self.all_samples.len() as f64;
        self.peptide_samples
            .iter()
            .filter(|(_, samples)| (samples.len() as f64) < threshold)
            .map(|(key, _)| key.clone())
            .collect()
    }
}

#[derive(Debug, Default)]
struct IntermediateExports {
    peptides: Option<PeptideExport>,
}

/// `RazorPeptideFilter` mode (Python `preprocessing/filters/protein.py:334-378`).
/// A razor peptide is a canonical mapping to more than one protein within a
/// sample (the pipeline runs per-sample). `Keep` is the no-op default; `Remove`
/// drops every row of a razor peptide; `AssignToTop` keeps only the rows of the
/// protein with the most unique peptides in the sample, breaking ties by
/// first-appearance order (the protein seen earliest in parquet row order wins),
/// mirroring Python `protein.py:358-378` (pandas `unique()` order + `max`'s
/// keep-first tie-break).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum RazorHandling {
    #[default]
    Keep,
    Remove,
    AssignToTop,
}

/// Map the configured `razor_peptide_handling` string to [`RazorHandling`].
/// `validate_filter_pipeline_subset` rejects everything but
/// `keep`/`remove`/`assign_to_top` before this runs, so any other value falls
/// back to the no-op `Keep`.
fn parse_razor_handling(handling: &str) -> RazorHandling {
    match handling {
        "remove" => RazorHandling::Remove,
        "assign_to_top" => RazorHandling::AssignToTop,
        _ => RazorHandling::Keep,
    }
}

#[derive(Debug, Default)]
struct PeptideExport {
    rows: HashMap<PeptideExportKey, HashMap<PeptideId, f64>>,
    /// When set, the export keys on `(Run, TechReplicate)` in addition to the
    /// sample, reproducing Python `aggregation_level == "run"`. Off (sample
    /// level) for the `features2proteins --export-peptides` path, which never
    /// requests run granularity.
    run_mode: bool,
    /// `RazorPeptideFilter` handling (Python protein.py:334-378), applied at
    /// materialization after the `min_unique` gate (Python orders MinPeptide
    /// before Razor). `Keep` for the `features2proteins --export-peptides` path.
    razor_handling: RazorHandling,
    /// First-appearance order of `ProteinName` per `(sample, canonical)`, used
    /// only by `RazorHandling::AssignToTop` to break ties the same way Python's
    /// `max(group[ProteinName].unique(), key=...)` does (pandas `unique()`
    /// preserves first-occurrence order, and `max` keeps the first maximum). The
    /// `rows` map is a `HashMap`, so it cannot recover this order; `push` is
    /// called in parquet row order, so we record the order here as rows arrive.
    /// Populated only when `razor_handling == AssignToTop`, leaving the other
    /// modes' push path untouched.
    razor_order: HashMap<(SampleId, String), Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PeptideExportKey {
    protein: ProteinId,
    sample: SampleId,
    peptide_canonical: String,
    protein_name: String,
    sample_name: String,
    bio_replicate: String,
    condition: String,
    /// Run identifier (`feature.run_file_name`) and technical replicate, only
    /// populated in run mode. In sample mode both are `None`/`0`, so the key is
    /// identical to the pre-run-mode key and the protein-export path is
    /// unaffected.
    run: Option<String>,
    tech_replicate: i64,
}

#[derive(Debug, Clone, Copy)]
struct IntermediateMeasurement<'a> {
    protein: ProteinId,
    sample: SampleId,
    peptide: PeptideId,
    intensity: f64,
    protein_name: &'a str,
    peptide_canonical: &'a str,
    sample_name: &'a str,
    run_file_name: &'a str,
    sdrf_record: Option<&'a SdrfRecord>,
}

/// Options that select how the peptide intermediate is serialised. The
/// `features2proteins --export-peptides` path uses the all-default value (CSV
/// only, sample level); `features2peptides` overrides them from its config.
#[derive(Debug, Clone, Copy, Default)]
struct PeptideExportOptions {
    log2: bool,
    /// Additional parquet sibling to write (Python `--save_parquet`).
    save_parquet: bool,
}

impl IntermediateExports {
    fn new(config: &FeatureToProteinsConfig) -> Self {
        Self {
            peptides: config
                .output
                .export_peptides
                .as_ref()
                .map(|_| PeptideExport::default()),
        }
    }

    /// Variant of [`IntermediateExports::new`] that requests run-level peptide
    /// keys, used only by `features2peptides --aggregation_level run`.
    fn new_with_run_mode(config: &FeatureToProteinsConfig, run_mode: bool) -> Self {
        Self {
            peptides: config
                .output
                .export_peptides
                .as_ref()
                .map(|_| PeptideExport {
                    rows: HashMap::new(),
                    run_mode,
                    ..PeptideExport::default()
                }),
        }
    }

    fn push(&mut self, measurement: IntermediateMeasurement<'_>) {
        if let Some(peptides) = &mut self.peptides {
            peptides.push(measurement);
        }
    }

    fn write(
        &self,
        config: &FeatureToProteinsConfig,
        allowed_cells: &HashSet<CellKey>,
        low_frequency_peptides: &HashSet<(String, String)>,
        options: PeptideExportOptions,
        dataset_normalized: Option<&HashMap<(ProteinId, SampleId, String), f64>>,
    ) -> Result<()> {
        if let (Some(path), Some(peptides_export)) = (
            config.output.export_peptides.as_deref(),
            self.peptides.as_ref(),
        ) {
            let rows = peptides_export.materialize(
                allowed_cells,
                low_frequency_peptides,
                options.log2,
                dataset_normalized,
            );
            // Python's dataset-level normalization (`_apply_dataset_normalization`,
            // stages.py:961-1047) pivots the peptide table to
            // `(ProteinName, PeptideCanonical) x SampleID`, normalizes, then melts
            // back keying only on `[ProteinName, PeptideCanonical]` — dropping
            // BioReplicate and Condition. So the exported CSV carries four columns,
            // not six. The non-dataset path returns the combined table unchanged and
            // keeps all six. `dataset_normalized.is_some()` is exactly the
            // dataset-level case (features2peptides always passes `None`, so its
            // export is unaffected).
            let omit_replicate_condition = dataset_normalized.is_some();
            peptides_export.write_csv(path, &rows, omit_replicate_condition)?;
            if options.save_parquet {
                let parquet_path = parquet_sibling(path);
                write_peptide_parquet(&parquet_path, &rows, peptides_export.run_mode)?;
            }
        }
        Ok(())
    }
}

/// Derive the parquet sibling of the CSV output path, mirroring Python's
/// `WriteParquetTask`, which does `os.path.splitext(output)[0] + ".parquet"`.
fn parquet_sibling(path: &Path) -> PathBuf {
    path.with_extension("parquet")
}

impl PeptideExport {
    fn push(&mut self, measurement: IntermediateMeasurement<'_>) {
        let (run, tech_replicate) = if self.run_mode {
            (
                Some(measurement.run_file_name.to_owned()),
                measurement
                    .sdrf_record
                    .and_then(|record| record.technical_replicate)
                    .map(i64::from)
                    .unwrap_or_else(|| tech_replicate_of(measurement.run_file_name)),
            )
        } else {
            (None, 0)
        };
        let key = PeptideExportKey {
            protein: measurement.protein,
            sample: measurement.sample,
            peptide_canonical: measurement.peptide_canonical.to_owned(),
            protein_name: measurement.protein_name.to_owned(),
            sample_name: measurement.sample_name.to_owned(),
            bio_replicate: measurement
                .sdrf_record
                .and_then(|record| record.biological_replicate)
                .map_or_else(|| "1".to_owned(), |value| value.to_string()),
            // On an SDRF miss Python's `_create_unnest_view` / `enrich_with_sdrf`
            // fall the Condition column back to the `sa_fallback` (`run_file_name`
            // for new QPX, the unnested `sample_accession` for legacy), which is
            // exactly the value `sample_name()` already resolves to. Reusing
            // `sample_condition` keeps that fallback identical to Python instead of
            // emitting the literal "Empty".
            condition: sample_condition(measurement.sample_name, measurement.sdrf_record),
            run,
            tech_replicate,
        };
        // Record the first-appearance order of proteins per `(sample, canonical)`
        // for the `AssignToTop` tie-break. Only this mode needs it, so the other
        // modes pay nothing here. Push is called in parquet row order, so the vec
        // ends up in pandas `unique()` order (first occurrence wins).
        if self.razor_handling == RazorHandling::AssignToTop {
            let order = self
                .razor_order
                .entry((measurement.sample, measurement.peptide_canonical.to_owned()))
                .or_default();
            if !order.iter().any(|name| name == measurement.protein_name) {
                order.push(measurement.protein_name.to_owned());
            }
        }
        let peptide_values = self.rows.entry(key).or_default();
        peptide_values
            .entry(measurement.peptide)
            .and_modify(|current| {
                if measurement.intensity > *current {
                    *current = measurement.intensity;
                }
            })
            .or_insert(measurement.intensity);
    }

    /// Build the sorted, filtered, summed peptide rows shared by the CSV and
    /// parquet writers, so the two serialisations are value-identical. Mirrors
    /// `sum_peptidoform_intensities` (the per-cell sum) followed by the optional
    /// `--log2`, then drops cells failing the unique-peptide gate and the
    /// `(ProteinName, PeptideCanonical)` low-frequency set.
    fn materialize(
        &self,
        allowed_cells: &HashSet<CellKey>,
        low_frequency_peptides: &HashSet<(String, String)>,
        log2: bool,
        dataset_normalized: Option<&HashMap<(ProteinId, SampleId, String), f64>>,
    ) -> Vec<PeptideParquetRow> {
        let mut keys = self
            .rows
            .keys()
            .filter(|key| {
                allowed_cells.contains(&CellKey {
                    protein: key.protein,
                    sample: key.sample,
                })
            })
            // Drop low-frequency peptides by `(ProteinName, PeptideCanonical)`
            // (Python `peptide.py:373-378`); the set is empty when the
            // `--remove_low_frequency_peptides` filter is off.
            .filter(|key| {
                !low_frequency_peptides
                    .contains(&(key.protein_name.clone(), key.peptide_canonical.clone()))
            })
            .collect::<Vec<_>>();
        // RazorPeptideFilter (Python protein.py:348-357), applied after the
        // `min_unique` gate as in the Python pipeline. A razor peptide is a
        // canonical mapping to more than one protein within a sample; `Remove`
        // drops all its rows. Python re-applies no `min_unique` gate afterwards.
        if self.razor_handling == RazorHandling::Remove {
            let mut canonical_proteins: HashMap<(SampleId, &str), HashSet<&str>> = HashMap::new();
            for key in &keys {
                canonical_proteins
                    .entry((key.sample, key.peptide_canonical.as_str()))
                    .or_default()
                    .insert(key.protein_name.as_str());
            }
            keys.retain(|key| {
                canonical_proteins
                    .get(&(key.sample, key.peptide_canonical.as_str()))
                    .is_none_or(|proteins| proteins.len() <= 1)
            });
        }
        // RazorPeptideFilter `assign_to_top` (Python protein.py:358-378). For a
        // razor canonical (mapping to >1 protein in the sample), keep only the
        // rows of the protein with the most unique peptides in the sample; ties
        // are broken by first-appearance order (the protein that arrived earliest
        // in parquet row order wins), matching pandas `unique()` ordering and
        // `max`'s keep-first behaviour. Non-razor canonicals (<=1 protein) are
        // kept untouched.
        if self.razor_handling == RazorHandling::AssignToTop {
            // `(sample, canonical) -> proteins` to detect razor peptides.
            let mut canonical_proteins: HashMap<(SampleId, &str), HashSet<&str>> = HashMap::new();
            // `(sample, protein) -> canonicals` to count unique peptides per
            // protein (Python `protein_peptide_counts` = nunique canonical).
            let mut protein_canonicals: HashMap<(SampleId, &str), HashSet<&str>> = HashMap::new();
            for key in &keys {
                canonical_proteins
                    .entry((key.sample, key.peptide_canonical.as_str()))
                    .or_default()
                    .insert(key.protein_name.as_str());
                protein_canonicals
                    .entry((key.sample, key.protein_name.as_str()))
                    .or_default()
                    .insert(key.peptide_canonical.as_str());
            }
            let razor_order = &self.razor_order;
            keys.retain(|key| {
                let proteins =
                    canonical_proteins.get(&(key.sample, key.peptide_canonical.as_str()));
                // Non-razor (canonical maps to <=1 protein): keep as-is.
                if proteins.is_none_or(|proteins| proteins.len() <= 1) {
                    return true;
                }
                // Razor: pick the top protein in first-appearance order. Iterate
                // the recorded order and update `best` only on a strictly greater
                // unique-peptide count, so the earliest protein wins ties.
                let order = razor_order.get(&(key.sample, key.peptide_canonical.clone()));
                let top = order.and_then(|order| {
                    let mut best: Option<(&str, usize)> = None;
                    for name in order {
                        let count = protein_canonicals
                            .get(&(key.sample, name.as_str()))
                            .map_or(0, HashSet::len);
                        if best.is_none_or(|(_, best_count)| count > best_count) {
                            best = Some((name.as_str(), count));
                        }
                    }
                    best.map(|(name, _)| name)
                });
                // Keep only the winning protein's rows; if the order is somehow
                // missing (should not happen), drop the razor rows like `remove`.
                top.is_some_and(|top| key.protein_name == top)
            });
        }
        keys.sort_by(|left, right| {
            left.protein_name
                .cmp(&right.protein_name)
                .then_with(|| left.peptide_canonical.cmp(&right.peptide_canonical))
                .then_with(|| left.sample_name.cmp(&right.sample_name))
                .then_with(|| left.bio_replicate.cmp(&right.bio_replicate))
                .then_with(|| left.condition.cmp(&right.condition))
                .then_with(|| left.run.cmp(&right.run))
                .then_with(|| left.tech_replicate.cmp(&right.tech_replicate))
        });
        keys.into_iter()
            .filter_map(|key| {
                let peptides = self.rows.get(key)?;
                // Sum peptidoform intensities per canonical-peptide cell (Python
                // `sum_peptidoform_intensities`). When a dataset-level sample
                // normalization was applied, use the normalized canonical value
                // (Python normalizes the peptide table before export) instead of
                // the raw sum. Then optionally log2-transform (Python applies
                // `--log2` after the sum), keeping both paths consistent.
                let raw = dataset_normalized
                    .and_then(|normalized| {
                        normalized.get(&(key.protein, key.sample, key.peptide_canonical.clone()))
                    })
                    .copied()
                    .unwrap_or_else(|| sum_peptide_values(peptides));
                let value = if log2 { raw.log2() } else { raw };
                Some(PeptideParquetRow {
                    protein_name: key.protein_name.clone(),
                    peptide_canonical: key.peptide_canonical.clone(),
                    sample_id: key.sample_name.clone(),
                    bio_replicate: key.bio_replicate.parse::<i32>().unwrap_or(1),
                    condition: key.condition.clone(),
                    run: key.run.clone(),
                    tech_replicate: key.run.as_ref().map(|_| key.tech_replicate),
                    norm_intensity: value,
                })
            })
            .collect()
    }

    fn write_csv(
        &self,
        path: &Path,
        rows: &[PeptideParquetRow],
        omit_replicate_condition: bool,
    ) -> Result<()> {
        create_parent_dir(path)?;
        let file = File::create(path).map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = WriterBuilder::new().from_writer(file);
        let mut header = vec!["ProteinName", "PeptideCanonical", "SampleID"];
        if !omit_replicate_condition {
            header.push("BioReplicate");
            header.push("Condition");
        }
        if self.run_mode {
            header.push("Run");
            header.push("TechReplicate");
        }
        header.push("NormIntensity");
        writer
            .write_record(&header)
            .map_err(|source| csv_error(path, source))?;

        for row in rows {
            let bio = row.bio_replicate.to_string();
            let intensity = format_float(row.norm_intensity);
            let mut record = vec![
                row.protein_name.as_str(),
                row.peptide_canonical.as_str(),
                row.sample_id.as_str(),
            ];
            if !omit_replicate_condition {
                record.push(bio.as_str());
                record.push(row.condition.as_str());
            }
            let tech = row.tech_replicate.unwrap_or(1).to_string();
            if self.run_mode {
                record.push(row.run.as_deref().unwrap_or_default());
                record.push(tech.as_str());
            }
            record.push(intensity.as_str());
            writer
                .write_record(&record)
                .map_err(|source| csv_error(path, source))?;
        }
        writer.flush().map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }
}

#[derive(Debug)]
enum FeatureAggregation {
    Sum(HashMap<CellKey, HashMap<PeptideId, f64>>),
    Median(HashMap<CellKey, HashMap<PeptideId, f64>>),
    Abd(HashMap<CellKey, HashMap<PeptideId, f64>>),
    PeptideCount(HashMap<CellKey, HashMap<PeptideId, f64>>),
    Pibaq(Box<PibaqFeatureAggregation>),
    Ratio(RatioAggregation),
    TopN {
        topn: usize,
        cells: HashMap<CellKey, HashMap<PeptideId, f64>>,
    },
    Lfq {
        method: QuantMethod,
        /// Whether this LFQ aggregation feeds the DirectLFQ-aligned solver.
        /// True for `QuantMethod::DirectLfq`, and for `QuantMethod::MaxLfq` unless
        /// the built-in fallback is forced (mirrors Python delegating maxlfq to
        /// DirectLFQ when the package is installed). When false, the built-in
        /// MaxLFQ solver runs on `traces`.
        route_to_directlfq: bool,
        directlfq_min_nonan: usize,
        directlfq_num_samples_quadratic: usize,
        /// MaxLFQ input: max intensity per contextual peptidoform ion, collapsed
        /// to canonical peptide only after every feature row has been observed.
        /// DirectLFQ uses the same feature-to-ion contract.
        traces: MaxLfqFeatureAggregation,
        /// Populated only when DirectLFQ ion export runs before matrix
        /// materialization, avoiding a second solver pass.
        cached_directlfq_values: Option<HashMap<ProteinId, Vec<(SampleId, f64)>>>,
    },
}

#[derive(Debug, Default)]
struct MaxLfqFeatureAggregation {
    ion_cells: HashMap<CellKey, HashMap<PeptideId, f64>>,
    ion_to_canonical: HashMap<PeptideId, PeptideId>,
    canonical_traces: HashMap<PeptideCellKey, f64>,
}

#[derive(Debug)]
struct PibaqAggregation {
    accession_peptides: HashMap<String, HashSet<String>>,
    peptide_accessions: HashMap<String, HashSet<String>>,
    families: Vec<ProteinFamily>,
    peptide_names: HashMap<PeptideId, String>,
    observations: HashMap<PeptideSampleKey, f64>,
    min_anchors: usize,
    /// piBAQ "high anchor" threshold (Python `--high-anchor-threshold`). Only
    /// affects the `EvidenceLevel` annotation column via [`classify_evidence`],
    /// never the quantification values (Python `pibaq.py:732`).
    high_anchor_threshold: usize,
    /// Per-canonical monoisotopic molecular weight, populated only when the
    /// caller requests TPA (`peptides2protein --tpa`). `None` leaves the piBAQ
    /// path untouched for every other consumer.
    mw_map: Option<HashMap<String, f64>>,
    /// Digest `max_aa`, used only to classify dropped observed peptides.
    digest_max_aa: Option<usize>,
    contaminant_twins: HashMap<String, String>,
}

#[derive(Debug)]
struct PibaqFeatureAggregation {
    core: PibaqAggregation,
    ion_observations: HashMap<PeptideSampleKey, f64>,
    ion_to_canonical: HashMap<PeptideId, PeptideId>,
    canonical_names: HashMap<PeptideId, String>,
}

#[derive(Debug)]
struct RatioAggregation {
    records: Vec<RatioPsm>,
    sample_names: HashMap<SampleId, String>,
    reference_samples: HashSet<String>,
    /// SDRF-driven `source name` -> plex id map, built once via
    /// `sample_to_plex` (Python's `detect_plexes_from_sdrf`). The reference and
    /// log2-ratio steps look plexes up here so per-plex grouping is keyed by the
    /// authoritative SDRF map rather than re-parsing each sample name.
    sample_to_plex: HashMap<String, String>,
    fraction_merge: RatioFractionMerge,
    min_unique_peptides: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProteinFamily {
    family_id: String,
    members: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PeptideSampleKey {
    peptide: PeptideId,
    sample: SampleId,
}

#[derive(Debug, Clone)]
struct RatioPsm {
    protein: ProteinId,
    peptide: PeptideId,
    charge: i32,
    sample: SampleId,
    intensity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RatioFractionMerge {
    Mean,
    Max,
}

#[derive(Debug, Default)]
struct IntensityFactors {
    run: HashMap<RunCellKey, RunNormalizationTransform>,
    sample: HashMap<String, f64>,
    /// Channel-IRS scale factors keyed by technical replicate (Python
    /// `irs_scale_by_techrep`). The lookup key is the **apply-side**
    /// techreplicate ([`tech_replicate_of`], the last `_` token), matching
    /// Python's `dataset_df[TECHREPLICATE].map(...)` (`peptide.py:336`). Empty
    /// when IRS is disabled or no reference-channel rows were found.
    irs_scale_by_techrep: HashMap<i64, f64>,
}

/// Derive a peptide row's `TechReplicate` from its run name, matching Python's
/// `apply_initial_filtering` (aggregation.py:211-223): a run name containing an
/// underscore uses the integer after the last underscore; a plain integer run
/// name uses that integer; otherwise Python's per-sample run-index fallback
/// applies, which is `1` whenever each sample maps to a single run (the
/// non-empty, no-SDRF path this port verifies against — the multi-run-per-sample
/// case yields an empty matrix in Python because SDRF enrichment remaps the
/// sample key, a quirk the existing golden tests already document).
fn tech_replicate_of(run_file_name: &str) -> i64 {
    if let Some((_, last)) = run_file_name.rsplit_once('_') {
        if let Ok(value) = last.parse::<i64>() {
            return value;
        }
    }
    run_file_name.parse::<i64>().unwrap_or(1)
}

/// IRS-internal technical-replicate derivation used when *collecting* the
/// reference-channel scale factors. Mirrors Python's
/// `CASE WHEN position('_' in run) > 0 THEN CAST(split_part(run, '_', 2) AS INTEGER)
/// ELSE CAST(run AS INTEGER) END` (`feature.py:764-766`): take the **second**
/// token (the part between the first and second `_`, or the tail after the first
/// `_`), not the last one. A run with no `_` parses the whole name as an integer.
/// `None` means the run cannot supply an integer techreplicate (DuckDB's `CAST`
/// would error / the row would not contribute), so it is dropped, matching the
/// `irs_value > 0` filter that follows in spirit (the run simply yields no key).
///
/// This is deliberately distinct from [`tech_replicate_of`], which keeps the
/// **last** token (Python's apply-side `run.str.split('_').str.get(-1)`,
/// `aggregation.py:215`). The two agree for two-token `mixture_techrep` names
/// (typical quantms TMT) but diverge for names with three or more tokens.
fn irs_tech_replicate_of(run_file_name: &str) -> Option<i64> {
    if let Some((_, rest)) = run_file_name.split_once('_') {
        // `split_part(run, '_', 2)` is the token between the first and second
        // `_`; if there is no second `_`, it is the remainder after the first.
        let second = rest.split('_').next().unwrap_or(rest);
        second.parse::<i64>().ok()
    } else {
        run_file_name.parse::<i64>().ok()
    }
}

impl IntensityFactors {
    fn normalize(&self, intensity: f64, sample: &str, run: &str) -> f64 {
        let sample_factor = self.sample.get(sample).copied().unwrap_or(1.0);
        let run_intensity = self
            .run
            .get(&RunCellKey {
                sample: sample.to_owned(),
                run: run.to_owned(),
            })
            .copied()
            .map_or(intensity, |transform| transform.apply(intensity));
        // IRS scaling slots between run- and sample-level normalization, the same
        // position Python uses: `feature_normalization` (run) runs first
        // (`peptide.py:324`), then IRS multiplies `NORM_INTENSITY`
        // (`peptide.py:333-340`), then `peptide_normalized` (sample) runs later
        // (`peptide.py:370`). Missing techreplicates map to 1.0 (Python's
        // `.fillna(1.0)`).
        let irs_scale = self
            .irs_scale_by_techrep
            .get(&tech_replicate_of(run))
            .copied()
            .unwrap_or(1.0);
        run_intensity * irs_scale * sample_factor
    }

    fn is_empty(&self) -> bool {
        self.run.is_empty() && self.sample.is_empty() && self.irs_scale_by_techrep.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
struct AggregationMeasurement<'a> {
    protein: ProteinId,
    sample: SampleId,
    peptide: PeptideId,
    canonical: PeptideId,
    intensity: f64,
    ion_name: &'a str,
    sample_name: &'a str,
    charge: i32,
}

impl FeatureAggregation {
    fn from_config(
        config: &FeatureToProteinsConfig,
        sdrf: Option<&SdrfTable>,
        raw_sdrf: Option<&SdrfRawTable>,
        pibaq_digest: Option<PibaqDigest>,
    ) -> Result<Self> {
        match config.quantification {
            QuantMethod::Sum | QuantMethod::Intensity => Ok(Self::Sum(HashMap::new())),
            QuantMethod::Median => Ok(Self::Median(HashMap::new())),
            QuantMethod::Abd => Ok(Self::Abd(HashMap::new())),
            QuantMethod::PeptideCount => Ok(Self::PeptideCount(HashMap::new())),
            QuantMethod::SpectralCount => Err(invalid_input(
                "spectral_count requires the PSM-level aggregation path",
            )),
            QuantMethod::Pibaq => Ok(Self::Pibaq(Box::new(PibaqFeatureAggregation::new(
                PibaqAggregation::from_digest(
                    config,
                    pibaq_digest.ok_or_else(|| {
                        invalid_input("piBAQ requires a runtime pyOpenMS FASTA digest")
                    })?,
                )?,
            )))),
            QuantMethod::Ratio => Ok(Self::Ratio(RatioAggregation::from_config(
                config, sdrf, raw_sdrf,
            )?)),
            QuantMethod::TopN => {
                if config.topn_peptides == 0 {
                    return Err(invalid_input("topn must be greater than 0"));
                }
                Ok(Self::TopN {
                    topn: config.topn_peptides,
                    cells: HashMap::new(),
                })
            }
            QuantMethod::MaxLfq | QuantMethod::DirectLfq => {
                let route_to_directlfq = config.quantification == QuantMethod::DirectLfq
                    || (!config.maxlfq.force_builtin
                        && dataset_sample_normalization_method(config)?.is_none());
                // Python's maxlfq delegation uses min_nonan=2 (the streaming path
                // hardcodes it; the class path passes min_peptides, default 2),
                // whereas the directlfq method keeps its own config default.
                let directlfq_min_nonan = if config.quantification == QuantMethod::MaxLfq {
                    MAXLFQ_DIRECTLFQ_MIN_NONAN
                } else {
                    config.directlfq.min_nonan
                };
                Ok(Self::Lfq {
                    method: config.quantification,
                    route_to_directlfq,
                    directlfq_min_nonan,
                    directlfq_num_samples_quadratic: config.directlfq.num_samples_quadratic,
                    traces: MaxLfqFeatureAggregation::default(),
                    cached_directlfq_values: None,
                })
            }
        }
    }

    fn push(&mut self, measurement: AggregationMeasurement<'_>) -> Result<()> {
        let cell = CellKey {
            protein: measurement.protein,
            sample: measurement.sample,
        };
        match self {
            Self::Sum(cells)
            | Self::Median(cells)
            | Self::Abd(cells)
            | Self::PeptideCount(cells) => {
                push_peptide_max(cells, cell, measurement.peptide, measurement.intensity);
            }
            Self::Pibaq(pibaq) => {
                pibaq.push(measurement);
            }
            Self::Ratio(ratio) => {
                ratio.push(measurement);
            }
            Self::TopN { cells, .. } => {
                push_peptide_max(cells, cell, measurement.peptide, measurement.intensity);
            }
            Self::Lfq { traces, .. } => {
                traces.push(measurement);
            }
        }
        Ok(())
    }

    fn applies_dataset_normalization(&self) -> bool {
        match self {
            Self::Sum(_)
            | Self::Median(_)
            | Self::Abd(_)
            | Self::PeptideCount(_)
            | Self::TopN { .. }
            | Self::Pibaq(_) => true,
            // Built-in MaxLFQ runs dataset normalization on its peptide traces;
            // the DirectLFQ-aligned path (DirectLFQ, or delegated MaxLFQ) does its
            // own internal sample normalization, so mokume must not apply another.
            Self::Lfq {
                route_to_directlfq, ..
            } => !route_to_directlfq,
            Self::Ratio(_) => false,
        }
    }

    /// Apply a dataset-level sample normalization (one that needs the full
    /// protein x sample matrix, not a per-sample factor). Quantile runs
    /// on the canonical peptide cells; piBAQ and MaxLFQ support quantile on their
    /// own structures. Other dataset methods are gated out before this point.
    fn apply_dataset_normalization(
        &mut self,
        method: SampleNormalizationMethod,
        allowed_cells: &HashSet<CellKey>,
        peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
        samples: &StringIdRegistry<SampleId>,
    ) {
        match self {
            Self::Sum(cells)
            | Self::Median(cells)
            | Self::Abd(cells)
            | Self::PeptideCount(cells)
            | Self::TopN { cells, .. } => {
                apply_dataset_norm_to_peptide_cells(
                    cells,
                    method,
                    allowed_cells,
                    peptide_to_canonical,
                    samples,
                );
            }
            Self::Pibaq(pibaq) => {
                if method == SampleNormalizationMethod::Quantile {
                    pibaq.apply_quantile_normalization();
                }
            }
            Self::Lfq {
                route_to_directlfq: false,
                traces,
                ..
            } => {
                if method == SampleNormalizationMethod::Quantile {
                    traces.apply_quantile_normalization(allowed_cells);
                }
            }
            Self::Ratio(_) | Self::Lfq { .. } => {}
        }
    }

    fn write_intermediate_exports(
        &mut self,
        config: &FeatureToProteinsConfig,
        proteins: &StringIdRegistry<ProteinId>,
        canonical_peptides: &StringIdRegistry<PeptideId>,
        samples: &StringIdRegistry<SampleId>,
        allowed_cells: &HashSet<CellKey>,
    ) -> Result<()> {
        let Self::Lfq {
            method: QuantMethod::DirectLfq,
            directlfq_min_nonan,
            directlfq_num_samples_quadratic,
            traces,
            cached_directlfq_values,
            ..
        } = self
        else {
            return Ok(());
        };
        let Some(path) = config.output.export_ions.as_deref() else {
            return Ok(());
        };

        let prepared = prepare_directlfq_ions(
            traces.directlfq_sums(),
            allowed_cells,
            proteins,
            canonical_peptides,
            samples,
        );
        let result = direct_lfq_aligned_with_ions(
            &prepared.ions,
            *directlfq_min_nonan,
            *directlfq_num_samples_quadratic,
        );
        write_directlfq_ions(
            path,
            &result.normalized_ions,
            &prepared,
            proteins,
            canonical_peptides,
            samples,
        )?;
        *cached_directlfq_values =
            Some(remap_directlfq_values(result.protein_quantities, &prepared));
        Ok(())
    }

    /// piBAQ only: rename `CONTAM_` twins of digest accessions to the non-contaminant entry.
    fn resolve_contaminant_twins(&self, protein_group: &str) -> Option<String> {
        let Self::Pibaq(pibaq) = self else {
            return None;
        };
        rename_contaminant_twins(protein_group, &pibaq.core.contaminant_twins)
    }

    fn keeps_shared_peptides(&self) -> bool {
        matches!(self, Self::Pibaq(_))
    }

    fn preserves_loading_rows(&self) -> bool {
        matches!(
            self,
            Self::Sum(_)
                | Self::Median(_)
                | Self::Abd(_)
                | Self::PeptideCount(_)
                | Self::TopN { .. }
                | Self::Lfq { .. }
        )
    }

    fn preserves_contextual_canonical_rows(&self) -> bool {
        matches!(
            self,
            Self::Median(_) | Self::Abd(_) | Self::PeptideCount(_) | Self::TopN { .. }
        )
    }

    fn peptide_key(&self, feature: &QpxFeatureRecord) -> String {
        match self {
            Self::Ratio(_) => feature.sequence.clone(),
            _ => peptide_key(feature),
        }
    }

    fn protein_name(&self, feature: &QpxFeatureRecord) -> Option<String> {
        match self {
            // Ratio retains its first-accession contract. LFQ methods keep the
            // complete protein group, matching the peptide intermediate.
            Self::Ratio(_) => first_protein_name(&feature.protein_accessions),
            _ => protein_group_name(&feature.protein_accessions),
        }
    }

    fn finalize(
        self,
        allowed_cells: &HashSet<CellKey>,
        proteins: &mut StringIdRegistry<ProteinId>,
        peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
        canonical_peptides: &StringIdRegistry<PeptideId>,
        samples: &StringIdRegistry<SampleId>,
    ) -> ProteinValues {
        match self {
            Self::Sum(cells) => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        allowed_cells
                            .contains(&key)
                            .then(|| (key, sum_peptide_values(&peptides)))
                    })
                    .collect(),
            ),
            Self::Median(cells) => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        if !allowed_cells.contains(&key) {
                            return None;
                        }
                        let canonical = collapse_to_canonical(peptides, peptide_to_canonical);
                        let mut intensities = canonical.into_values().collect::<Vec<_>>();
                        median(&mut intensities).map(|value| (key, value))
                    })
                    .collect(),
            ),
            Self::Abd(cells) => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        if !allowed_cells.contains(&key) {
                            return None;
                        }
                        let canonical = collapse_to_canonical(peptides, peptide_to_canonical);
                        let mut log2_intensities = canonical
                            .into_values()
                            .filter(|value| value.is_finite() && *value > 0.0)
                            .map(f64::log2)
                            .collect::<Vec<_>>();
                        median_finite(&mut log2_intensities).map(|value| (key, value))
                    })
                    .collect(),
            ),
            Self::PeptideCount(cells) => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        if !allowed_cells.contains(&key) {
                            return None;
                        }
                        let canonical = collapse_to_canonical(peptides, peptide_to_canonical);
                        Some((key, canonical.len() as f64))
                    })
                    .collect(),
            ),
            Self::Pibaq(pibaq) => ProteinValues::Cells(pibaq.finalize(proteins)),
            Self::Ratio(ratio) => ProteinValues::Cells(ratio.finalize()),
            Self::TopN { topn, cells } => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        if !allowed_cells.contains(&key) {
                            return None;
                        }
                        let canonical = collapse_to_canonical(peptides, peptide_to_canonical);
                        let mut intensities = canonical.into_values().collect::<Vec<_>>();
                        intensities.sort_by(|left, right| right.total_cmp(left));
                        let selected = intensities.into_iter().take(topn).collect::<Vec<_>>();
                        if selected.is_empty() {
                            return None;
                        }
                        let value = selected.iter().sum::<f64>() / selected.len() as f64;
                        Some((key, value))
                    })
                    .collect(),
            ),
            Self::Lfq {
                route_to_directlfq: true,
                directlfq_min_nonan,
                directlfq_num_samples_quadratic,
                traces,
                cached_directlfq_values,
                ..
            } => {
                let values = if let Some(values) = cached_directlfq_values {
                    values
                } else {
                    let prepared = prepare_directlfq_ions(
                        traces.into_directlfq_sums(),
                        allowed_cells,
                        proteins,
                        canonical_peptides,
                        samples,
                    );
                    let values = direct_lfq_aligned(
                        &prepared.ions,
                        directlfq_min_nonan,
                        directlfq_num_samples_quadratic,
                    );
                    remap_directlfq_values(values, &prepared)
                };
                ProteinValues::Rows(values)
            }
            Self::Lfq { traces, .. } => {
                let traces = traces.into_canonical_traces();
                let (peptide_to_lexical, _) = lexical_id_remap(canonical_peptides);
                let mut samples = allowed_cells
                    .iter()
                    .map(|cell| cell.sample)
                    .collect::<Vec<_>>();
                samples.sort_by_key(|sample| sample.get());
                samples.dedup();

                let mut grouped = HashMap::<ProteinId, Vec<PeptideMeasurement>>::new();
                for (key, intensity) in traces {
                    let cell = CellKey {
                        protein: key.protein,
                        sample: key.sample,
                    };
                    if allowed_cells.contains(&cell) {
                        grouped
                            .entry(key.protein)
                            .or_default()
                            .push(PeptideMeasurement {
                                peptide: peptide_to_lexical
                                    .get(&key.peptide)
                                    .copied()
                                    .unwrap_or(key.peptide),
                                sample: key.sample,
                                intensity,
                            });
                    }
                }

                ProteinValues::Rows(
                    grouped
                        .into_iter()
                        .collect::<Vec<_>>()
                        .into_par_iter()
                        .map(|(protein, measurements)| {
                            (protein, max_lfq_with_samples(&measurements, &samples))
                        })
                        .collect(),
                )
            }
        }
    }
}

impl MaxLfqFeatureAggregation {
    fn push(&mut self, measurement: AggregationMeasurement<'_>) {
        self.ion_to_canonical
            .entry(measurement.peptide)
            .or_insert(measurement.canonical);
        push_peptide_max(
            &mut self.ion_cells,
            CellKey {
                protein: measurement.protein,
                sample: measurement.sample,
            },
            measurement.peptide,
            measurement.intensity,
        );
    }

    fn collapse_ions(&mut self) {
        let mut cells = std::mem::take(&mut self.ion_cells)
            .into_iter()
            .collect::<Vec<_>>();
        cells.sort_by_key(|(cell, _)| (cell.protein.get(), cell.sample.get()));
        for (cell, peptides) in cells {
            for (canonical, intensity) in collapse_to_canonical(peptides, &self.ion_to_canonical) {
                self.canonical_traces.insert(
                    PeptideCellKey {
                        protein: cell.protein,
                        sample: cell.sample,
                        peptide: canonical,
                    },
                    intensity,
                );
            }
        }
    }

    fn apply_quantile_normalization(&mut self, allowed_cells: &HashSet<CellKey>) {
        self.collapse_ions();
        apply_quantile_to_lfq_traces(&mut self.canonical_traces, allowed_cells);
    }

    fn into_canonical_traces(mut self) -> HashMap<PeptideCellKey, f64> {
        self.collapse_ions();
        self.canonical_traces
    }

    fn into_directlfq_sums(self) -> HashMap<DirectLfqCellKey, f64> {
        self.into_canonical_traces()
            .into_iter()
            .map(|(key, intensity)| {
                (
                    DirectLfqCellKey {
                        protein: key.protein,
                        canonical: key.peptide,
                        sample: key.sample,
                    },
                    intensity,
                )
            })
            .collect()
    }

    fn directlfq_sums(&self) -> HashMap<DirectLfqCellKey, f64> {
        let mut sums = self
            .canonical_traces
            .iter()
            .map(|(key, intensity)| {
                (
                    DirectLfqCellKey {
                        protein: key.protein,
                        canonical: key.peptide,
                        sample: key.sample,
                    },
                    *intensity,
                )
            })
            .collect::<HashMap<_, _>>();
        let mut cells = self.ion_cells.iter().collect::<Vec<_>>();
        cells.sort_by_key(|(cell, _)| (cell.protein.get(), cell.sample.get()));
        for (cell, peptides) in cells {
            for (canonical, intensity) in
                collapse_to_canonical(peptides.clone(), &self.ion_to_canonical)
            {
                sums.insert(
                    DirectLfqCellKey {
                        protein: cell.protein,
                        canonical,
                        sample: cell.sample,
                    },
                    intensity,
                );
            }
        }
        sums
    }
}

fn lexical_id_remap<I>(registry: &StringIdRegistry<I>) -> (HashMap<I, I>, HashMap<I, I>)
where
    I: Copy + Eq + std::hash::Hash + From<u32> + Into<u32>,
{
    let numeric_ids = registry.iter().map(|(id, _)| id).collect::<Vec<_>>();
    let mut by_name = registry.iter().collect::<Vec<_>>();
    by_name.sort_by(|left, right| left.1.cmp(right.1));

    let mut to_lexical = HashMap::with_capacity(by_name.len());
    let mut from_lexical = HashMap::with_capacity(by_name.len());
    for ((original, _), lexical) in by_name.into_iter().zip(numeric_ids) {
        to_lexical.insert(original, lexical);
        from_lexical.insert(lexical, original);
    }
    (to_lexical, from_lexical)
}

impl PibaqFeatureAggregation {
    fn new(core: PibaqAggregation) -> Self {
        Self {
            core,
            ion_observations: HashMap::new(),
            ion_to_canonical: HashMap::new(),
            canonical_names: HashMap::new(),
        }
    }

    fn push(&mut self, measurement: AggregationMeasurement<'_>) {
        self.ion_to_canonical
            .entry(measurement.peptide)
            .or_insert(measurement.canonical);
        self.canonical_names
            .entry(measurement.canonical)
            .or_insert_with(|| measurement.ion_name.to_owned());
        let key = PeptideSampleKey {
            peptide: measurement.peptide,
            sample: measurement.sample,
        };
        self.ion_observations
            .entry(key)
            .and_modify(|current| {
                if measurement.intensity > *current {
                    *current = measurement.intensity;
                }
            })
            .or_insert(measurement.intensity);
    }

    fn collapse_ions(&mut self) {
        let mut canonical_ions = HashMap::<PeptideSampleKey, Vec<(PeptideId, f64)>>::new();
        for (key, intensity) in std::mem::take(&mut self.ion_observations) {
            let Some(canonical) = self.ion_to_canonical.get(&key.peptide).copied() else {
                continue;
            };
            canonical_ions
                .entry(PeptideSampleKey {
                    peptide: canonical,
                    sample: key.sample,
                })
                .or_default()
                .push((key.peptide, intensity));
        }
        for (key, mut ions) in canonical_ions {
            let Some(name) = self.canonical_names.get(&key.peptide) else {
                continue;
            };
            ions.sort_by_key(|(ion, _)| *ion);
            let intensity = ions.into_iter().map(|(_, value)| value).sum();
            self.core.push(key.sample, key.peptide, name, intensity);
        }
    }

    fn apply_quantile_normalization(&mut self) {
        self.collapse_ions();
        self.core.apply_quantile_normalization();
    }

    fn finalize(mut self, proteins: &mut StringIdRegistry<ProteinId>) -> HashMap<CellKey, f64> {
        self.collapse_ions();
        self.core.finalize(proteins)
    }
}

impl PibaqAggregation {
    fn from_digest(config: &FeatureToProteinsConfig, digest: PibaqDigest) -> Result<Self> {
        info!(
            pyopenms_version = digest.provenance.pyopenms_version,
            enzyme = digest.provenance.enzyme,
            catalog_hash = digest.provenance.catalog_hash,
            min_aa = digest.provenance.min_aa,
            max_aa = digest.provenance.max_aa,
            missed_cleavages = digest.provenance.missed_cleavages,
            "using runtime pyOpenMS FASTA digest"
        );
        let mut accession_peptides = digest.accession_peptides;
        accession_peptides.retain(|_, peptides| !peptides.is_empty());
        let mut contaminant_twins = HashMap::new();
        if config.filtering.remove_contaminants {
            contaminant_twins = find_contaminant_twins(&accession_peptides);
            let removed = drop_contaminant_accessions(&mut accession_peptides);
            info!(
                removed,
                twins = contaminant_twins.len(),
                "piBAQ digest: contaminant accessions excluded from families"
            );
        }
        if accession_peptides.is_empty() {
            return Err(invalid_input(
                "FASTA did not produce theoretical peptides for piBAQ",
            ));
        }
        let peptide_accessions = invert_peptide_index(&accession_peptides);
        let auto_families = discover_families(
            &accession_peptides,
            &peptide_accessions,
            config.pibaq.min_shared,
        )?;
        let families = if let Some(path) = config.pibaq.families_yaml.as_deref() {
            merge_family_overrides(auto_families, load_family_overrides(path)?)
        } else {
            auto_families
        };
        Ok(Self {
            accession_peptides,
            peptide_accessions,
            families,
            peptide_names: HashMap::new(),
            observations: HashMap::new(),
            min_anchors: config.pibaq.min_anchors,
            high_anchor_threshold: config.pibaq.high_anchor_threshold,
            mw_map: None,
            digest_max_aa: Some(digest.provenance.max_aa),
            contaminant_twins,
        })
    }

    fn push(&mut self, sample: SampleId, peptide: PeptideId, peptide_name: &str, intensity: f64) {
        self.peptide_names
            .entry(peptide)
            .or_insert_with(|| peptide_name.to_owned());
        let key = PeptideSampleKey { peptide, sample };
        self.observations
            .entry(key)
            .and_modify(|current| {
                if intensity > *current {
                    *current = intensity;
                }
            })
            .or_insert(intensity);
    }

    fn apply_quantile_normalization(&mut self) {
        let assignments = quantile_normalized_assignments(
            self.observations
                .iter()
                .map(|(key, intensity)| (key.peptide, key.sample, *intensity)),
        );
        for (key, intensity) in &mut self.observations {
            if let Some(normalized) = assignments.get(&(key.peptide, key.sample)).copied() {
                *intensity = normalized;
            }
        }
    }

    fn finalize(self, proteins: &mut StringIdRegistry<ProteinId>) -> HashMap<CellKey, f64> {
        self.finalize_detailed(proteins)
            .into_iter()
            .map(|(key, detail)| (key, detail.cell.pibaq))
            .collect()
    }

    /// Run the full piBAQ allocation and return, per (protein, sample), both the
    /// piBAQ value and the inputs `peptides2protein` needs to reproduce the
    /// Python long-format table (`NormIntensity`, `FamilyId`, `EvidenceLevel`,
    /// `FamilySize`). The allocation math is identical to [`Self::finalize`];
    /// only the captured metadata differs, so there is a single source of truth
    /// for the algorithm.
    fn finalize_detailed(
        self,
        proteins: &mut StringIdRegistry<ProteinId>,
    ) -> HashMap<CellKey, PibaqDetailedCell> {
        let observations = self
            .observations
            .into_iter()
            .filter_map(|(key, intensity)| {
                self.peptide_names
                    .get(&key.peptide)
                    .map(|peptide| (peptide.clone(), key.sample, intensity))
            })
            .collect::<Vec<_>>();
        if observations.is_empty() {
            return HashMap::new();
        }

        let observed_peptides = observations
            .iter()
            .map(|(peptide, _, _)| peptide.clone())
            .collect::<HashSet<_>>();
        let anchor_counts = count_unique_anchors(&observed_peptides, &self.peptide_accessions);
        let peptide_owner = assign_peptides_to_owning_family(
            &self.families,
            &self.peptide_accessions,
            &anchor_counts,
        );
        let misses = summarize_digest_misses(
            &observations,
            &peptide_owner,
            &self.peptide_accessions,
            self.digest_max_aa,
        );
        info!(
            observed_peptides = misses.observed_peptides,
            dropped_peptides = misses.dropped_peptides,
            dropped_intensity_fraction = misses.dropped_intensity_fraction(),
            missed_cleavage = misses.missed_cleavage,
            too_long = misses.too_long,
            met_removed_n_term = misses.met_removed_n_term,
            other = misses.other,
            "piBAQ: observed peptides outside the theoretical digest are dropped"
        );
        let family_to_peptides = invert_peptide_ownership(&peptide_owner);
        let mut observations_by_family = HashMap::<String, Vec<(String, SampleId, f64)>>::new();
        for observation in observations {
            if let Some(owner) = peptide_owner.get(&observation.0) {
                observations_by_family
                    .entry(owner.clone())
                    .or_default()
                    .push(observation);
            }
        }

        let mut detailed = HashMap::new();
        for family in self.families {
            let Some(observations) = observations_by_family.remove(&family.family_id) else {
                continue;
            };
            let owned_peptides = family_to_peptides
                .get(&family.family_id)
                .cloned()
                .unwrap_or_default();
            if owned_peptides.is_empty() {
                continue;
            }
            let min_anchor = family
                .members
                .iter()
                .map(|member| anchor_counts.get(member).copied().unwrap_or(0))
                .min()
                .unwrap_or(0);
            let max_anchor = family
                .members
                .iter()
                .map(|member| anchor_counts.get(member).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let evidence = classify_evidence(
                min_anchor,
                max_anchor,
                self.min_anchors,
                self.high_anchor_threshold,
            );
            let family_size = family.members.len();
            let output = finalize_family_allocation(
                &family,
                &observations,
                &owned_peptides,
                &self.accession_peptides,
                &self.peptide_accessions,
                evidence == PibaqEvidence::FamilyOnly,
                proteins,
            );
            for (key, cell) in output {
                let molecular_weight = self.mw_map.as_ref().map(|mw_map| {
                    let member_mw = proteins
                        .resolve(key.protein)
                        .and_then(|member| mw_map.get(member).copied())
                        .unwrap_or(0.0);
                    if member_mw == 0.0 {
                        1.0
                    } else {
                        member_mw
                    }
                });
                let tpa = molecular_weight.map(|mw| cell.norm_intensity / mw);
                detailed.insert(
                    key,
                    PibaqDetailedCell {
                        cell,
                        family_id: family.family_id.clone(),
                        evidence_level: evidence,
                        family_size,
                        molecular_weight,
                        tpa,
                    },
                );
            }
        }
        detailed
    }
}

/// Observed peptides (and intensity) that piBAQ drops because the digest lacks them.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct DigestMissSummary {
    observed_peptides: usize,
    dropped_peptides: usize,
    observed_intensity: f64,
    dropped_intensity: f64,
    missed_cleavage: usize,
    too_long: usize,
    met_removed_n_term: usize,
    other: usize,
}

impl DigestMissSummary {
    fn dropped_intensity_fraction(&self) -> f64 {
        if self.observed_intensity > 0.0 {
            self.dropped_intensity / self.observed_intensity
        } else {
            0.0
        }
    }
}

fn summarize_digest_misses(
    observations: &[(String, SampleId, f64)],
    peptide_owner: &HashMap<String, String>,
    peptide_accessions: &HashMap<String, HashSet<String>>,
    max_aa: Option<usize>,
) -> DigestMissSummary {
    let mut summary = DigestMissSummary::default();
    let mut seen = HashSet::<&str>::new();
    for (peptide, _, intensity) in observations {
        let in_digest = peptide_owner.contains_key(peptide);
        if intensity.is_finite() {
            summary.observed_intensity += intensity;
            if !in_digest {
                summary.dropped_intensity += intensity;
            }
        }
        if !seen.insert(peptide.as_str()) {
            continue;
        }
        summary.observed_peptides += 1;
        if in_digest {
            continue;
        }
        summary.dropped_peptides += 1;
        if peptide_accessions.contains_key(&format!("M{peptide}")) {
            summary.met_removed_n_term += 1;
        } else if max_aa.is_some_and(|max_aa| peptide.chars().count() > max_aa) {
            summary.too_long += 1;
        } else if filters::trypsin_missed_cleavages(peptide) > 0 {
            summary.missed_cleavage += 1;
        } else {
            summary.other += 1;
        }
    }
    summary
}

/// Evidence buckets mirroring the Python `_classify_evidence` helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PibaqEvidence {
    FamilyOnly,
    Medium,
    High,
}

impl PibaqEvidence {
    /// Lowercase label matching the Python `EVIDENCE_*` constants.
    const fn label(self) -> &'static str {
        match self {
            Self::FamilyOnly => "family_only",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Bucket a family's anchor counts, mirroring the Python `_classify_evidence`.
fn classify_evidence(
    min_anchor: usize,
    max_anchor: usize,
    min_required: usize,
    high_threshold: usize,
) -> PibaqEvidence {
    if max_anchor < min_required {
        PibaqEvidence::FamilyOnly
    } else if min_anchor >= high_threshold {
        PibaqEvidence::High
    } else {
        PibaqEvidence::Medium
    }
}

/// A fully described piBAQ output cell for `peptides2protein`'s long-format table.
#[derive(Debug, Clone)]
struct PibaqDetailedCell {
    cell: PibaqCell,
    family_id: String,
    evidence_level: PibaqEvidence,
    family_size: usize,
    /// Per-protein molecular weight (member MW with 0 -> 1). Populated only
    /// when the caller requested TPA; mirrors the Python `_attach_tpa`
    /// `MolecularWeight` column.
    molecular_weight: Option<f64>,
    /// TPA = `NormIntensity / MolecularWeight`, populated alongside
    /// `molecular_weight`.
    tpa: Option<f64>,
}

impl RatioAggregation {
    fn from_config(
        config: &FeatureToProteinsConfig,
        sdrf: Option<&SdrfTable>,
        raw_sdrf: Option<&SdrfRawTable>,
    ) -> Result<Self> {
        let Some(sdrf) = sdrf else {
            return Err(invalid_input("Ratio quantification requires --sdrf option"));
        };
        let raw_sdrf = raw_sdrf.ok_or_else(|| {
            invalid_input("Ratio quantification requires readable raw SDRF metadata")
        })?;
        let reference_samples = resolve_ratio_reference_samples(sdrf, raw_sdrf, config)?;
        if reference_samples.is_empty() {
            return Err(invalid_input(
                "Ratio quantification requires reference samples; repeat --irs-reference-sample or mark references in SDRF",
            ));
        }
        Ok(Self {
            records: Vec::new(),
            sample_names: HashMap::new(),
            reference_samples: reference_samples.into_iter().collect(),
            sample_to_plex: derive_sample_plexes(
                sdrf,
                Some(raw_sdrf),
                config.irs.plex_column.as_deref(),
            )?
            .0,
            fraction_merge: parse_ratio_fraction_merge(&config.ratio.fraction_merge)?,
            min_unique_peptides: config.filtering.min_unique_peptides,
        })
    }

    fn push(&mut self, measurement: AggregationMeasurement<'_>) {
        self.sample_names
            .entry(measurement.sample)
            .or_insert_with(|| measurement.sample_name.to_owned());
        self.records.push(RatioPsm {
            protein: measurement.protein,
            peptide: measurement.peptide,
            charge: measurement.charge,
            sample: measurement.sample,
            intensity: measurement.intensity,
        });
    }

    fn finalize(self) -> HashMap<CellKey, f64> {
        let valid_proteins = ratio_valid_proteins(&self.records, self.min_unique_peptides);
        let averaged = average_ratio_fractions(&self.records, &valid_proteins, self.fraction_merge);
        let reference_intensity = ratio_reference_intensities(
            &averaged,
            &self.sample_names,
            &self.reference_samples,
            &self.sample_to_plex,
        );
        let peptide_ratios = ratio_peptide_log2_ratios(
            &averaged,
            &self.sample_names,
            &self.reference_samples,
            &self.sample_to_plex,
            &reference_intensity,
        );
        ratio_protein_medians(peptide_ratios)
    }
}

fn push_peptide_max(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    cell: CellKey,
    peptide: PeptideId,
    intensity: f64,
) {
    let peptides = cells.entry(cell).or_default();
    let current = peptides.entry(peptide).or_insert(0.0);
    if intensity > *current {
        *current = intensity;
    }
}

fn sum_peptide_values(peptides: &HashMap<PeptideId, f64>) -> f64 {
    let mut values = peptides.iter().collect::<Vec<_>>();
    values.sort_by_key(|(peptide, _)| **peptide);
    values.into_iter().map(|(_, value)| *value).sum()
}

/// Collapse per-(peptidoform, charge) values into one value per canonical
/// peptide by summing, mirroring the Python loader which keeps the max
/// intensity per ion across runs and then sums ions into the canonical peptide
/// (`get_peptidoform_normalize_intensities` then `sum_peptidoform_intensities`).
/// Median/Abd/TopN/PeptideCount operate on these canonical values.
fn collapse_to_canonical(
    peptides: HashMap<PeptideId, f64>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) -> HashMap<PeptideId, f64> {
    let mut canonical = HashMap::with_capacity(peptides.len());
    let mut peptides = peptides.into_iter().collect::<Vec<_>>();
    peptides.sort_by_key(|(peptide, _)| *peptide);
    for (peptide, value) in peptides {
        let key = peptide_to_canonical
            .get(&peptide)
            .copied()
            .unwrap_or(peptide);
        *canonical.entry(key).or_insert(0.0) += value;
    }
    canonical
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct QuantilePeptideKey {
    protein: ProteinId,
    peptide: PeptideId,
}

/// Apply a dataset-level sample normalization to the canonical-peptide x sample
/// cells in place. Shared by the protein-matrix aggregation (`Self::Sum` /
/// `Median` / `Abd` / `PeptideCount` / `TopN`) and the `--export-peptides`
/// path, so the exported peptides carry the exact same normalization Python
/// applies in `load_for_mokume` before writing the peptide CSV.
fn apply_dataset_norm_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    method: SampleNormalizationMethod,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    samples: &StringIdRegistry<SampleId>,
) {
    match method {
        SampleNormalizationMethod::Quantile => {
            apply_quantile_to_peptide_cells(cells, allowed_cells, peptide_to_canonical);
        }
        SampleNormalizationMethod::Rlr => {
            apply_rlr_to_peptide_cells(cells, allowed_cells, peptide_to_canonical);
        }
        SampleNormalizationMethod::Loess => {
            apply_loess_to_peptide_cells(cells, allowed_cells, peptide_to_canonical);
        }
        SampleNormalizationMethod::Hierarchical => {
            apply_hierarchical_to_peptide_cells(
                cells,
                allowed_cells,
                peptide_to_canonical,
                samples,
            );
        }
        SampleNormalizationMethod::MedianCenter => {
            apply_center_to_peptide_cells(
                cells,
                allowed_cells,
                peptide_to_canonical,
                CenterStat::Median,
            );
        }
        SampleNormalizationMethod::MeanCenter => {
            apply_center_to_peptide_cells(
                cells,
                allowed_cells,
                peptide_to_canonical,
                CenterStat::Mean,
            );
        }
        SampleNormalizationMethod::Tmm => {
            apply_tmm_to_peptide_cells(cells, allowed_cells, peptide_to_canonical, samples);
        }
        _ => {}
    }
}

fn apply_quantile_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) {
    let mut canonical_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        for (peptide, intensity) in peptides {
            let canonical = peptide_to_canonical
                .get(peptide)
                .copied()
                .unwrap_or(*peptide);
            *canonical_cells
                .entry(*cell)
                .or_default()
                .entry(canonical)
                .or_insert(0.0) += *intensity;
        }
    }

    let assignments =
        quantile_normalized_assignments(canonical_cells.iter().flat_map(|(cell, peptides)| {
            peptides.iter().map(|(peptide, intensity)| {
                (
                    QuantilePeptideKey {
                        protein: cell.protein,
                        peptide: *peptide,
                    },
                    cell.sample,
                    *intensity,
                )
            })
        }));

    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(canonical_peptides) = canonical_cells.remove(cell) else {
            continue;
        };
        peptides.clear();
        for (peptide, _) in canonical_peptides {
            let key = QuantilePeptideKey {
                protein: cell.protein,
                peptide,
            };
            if let Some(normalized) = assignments.get(&(key, cell.sample)).copied() {
                peptides.insert(peptide, normalized);
            }
        }
    }
}

/// Whether the centering statistic is the per-sample median or mean of the
/// log2 peptide values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CenterStat {
    Median,
    Mean,
}

/// Sample-level median/mean centering. Mirrors the Python pipeline's
/// `NormalizationStage.apply_median_center` / `apply_mean_center`, which are
/// `is_dataset_level` methods: `_apply_dataset_normalizer` pivots the
/// peptide-level long table to the `(protein, canonical) x sample` wide matrix
/// (`aggfunc="sum"`), `np.log2`s after replacing 0 with NaN, subtracts each
/// column's median/mean (`median_center(axis=0)` / `x - x.mean(axis=0)`, both
/// NaN-skipping), then `2 ** result`. So the center is computed over the SAME
/// summed-canonical-peptide matrix the other dataset methods use, NOT over the
/// raw per-feature intensities (which back GlobalMedian/ConditionMedian via the
/// Python `get_median_map` SQL `MEDIAN(intensity)`). Because the center is a
/// single per-sample log2 offset applied uniformly to every peptide, it commutes
/// with the downstream sum-to-protein rollup; applying it here on the canonical
/// cells reproduces Python cell-for-cell.
///
/// Algorithm:
///  1. Collapse each allowed cell's peptidoforms to canonical peptides by SUM via
///     `peptide_to_canonical`; keep only strictly-positive finite sums and store
///     their `log2`.
///  2. Per sample, compute the median (or mean) of all log2 values in that
///     sample's column.
///  3. Write back `2 ** (log2(value) - center[sample])`. Samples with no valid
///     log2 value (empty column) are left unshifted.
fn apply_center_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    stat: CenterStat,
) {
    // (1) Collapse to canonical and log2 the strictly-positive sums.
    let mut log2_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let mut summed = HashMap::<PeptideId, f64>::new();
        for (peptide, intensity) in peptides {
            let canonical = peptide_to_canonical
                .get(peptide)
                .copied()
                .unwrap_or(*peptide);
            *summed.entry(canonical).or_insert(0.0) += *intensity;
        }
        let log_row = summed
            .into_iter()
            .filter(|(_, value)| value.is_finite() && *value > 0.0)
            .map(|(canonical, value)| (canonical, value.log2()))
            .collect::<HashMap<_, _>>();
        if !log_row.is_empty() {
            log2_cells.insert(*cell, log_row);
        }
    }
    if log2_cells.is_empty() {
        return;
    }

    // (2) Per-sample center over the column's log2 values.
    let mut sample_log_values = HashMap::<SampleId, Vec<f64>>::new();
    for (cell, peptides) in &log2_cells {
        let column = sample_log_values.entry(cell.sample).or_default();
        column.extend(peptides.values().copied());
    }
    let centers = sample_log_values
        .into_iter()
        .filter_map(|(sample, mut values)| {
            let center = match stat {
                CenterStat::Median => median_finite(&mut values),
                CenterStat::Mean => mokume_core::stats::mean_finite(&values),
            }?;
            center.is_finite().then_some((sample, center))
        })
        .collect::<HashMap<SampleId, f64>>();

    // (3) Write back 2 ** (log2(value) - center[sample]).
    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(log_row) = log2_cells.get(cell) else {
            continue;
        };
        let center = centers.get(&cell.sample).copied().unwrap_or(0.0);
        peptides.clear();
        for (peptide, value) in log_row {
            peptides.insert(*peptide, (value - center).exp2());
        }
    }
}

/// Robust Linear Regression (RLR) sample normalization, mirroring NormalyzerDE
/// (Chawade et al., 2014) as implemented by Python `RlrNormalizer`.
///
/// `stages.apply_rlr` constructs `RlrNormalizer(log_transform=False)` and calls
/// it through `_apply_dataset_normalizer(..., log_space=False)`. With
/// `log_transform=False` the normalizer log2s the input itself (dropping
/// non-positive values), fits per sample, divides out the fit in log2 space,
/// and returns `2 ** result` — so the pipeline's outer `log_space=False` means
/// the wide matrix handed in is *linear* and the output is *linear*. We
/// therefore reproduce the whole transform in log2 space internally and
/// exponentiate at the end, exactly like the Python class.
///
/// Per sample, an M-estimator (Huber) robust regression is fitted between the
/// sample's log2 intensities `y` and the per-row log2 median `reference`
/// (pseudo-reference, `median(axis=1)`). The normalized value is
/// `(log2(x) - intercept) / slope`. This matches `statsmodels`
/// `sm.RLM(y, sm.add_constant(reference), M=sm.robust.norms.HuberT()).fit()`
/// with the default IRLS configuration (OLS init, MAD scale with center 0 and
/// `c = 0.6744897501960817`, Huber tuning constant `t = 1.345`, `conv="dev"`,
/// `tol = 1e-8`, `maxiter = 50`, `update_scale=True`).
///
/// Determinism: the only ordering-sensitive steps are medians; these collect
/// finite values into a `Vec` that is sorted with `f64::total_cmp` before
/// taking the median, so the result is independent of `HashMap` iteration
/// order. The IRLS recurrence is a deterministic fixed-point iteration over the
/// collected `(reference, y)` pairs, which are sorted by row id before fitting.
fn apply_rlr_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) {
    // (1) Collapse each allowed cell's peptidoforms to canonical peptides by
    // SUM, then log2 the strictly-positive sums (Python drops non-positive
    // values before `np.log2`). Rows of the wide matrix are (protein,
    // canonical); columns are samples.
    let mut log2_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let mut summed = HashMap::<PeptideId, f64>::new();
        for (peptide, intensity) in peptides {
            let canonical = peptide_to_canonical
                .get(peptide)
                .copied()
                .unwrap_or(*peptide);
            *summed.entry(canonical).or_insert(0.0) += *intensity;
        }
        let log_row = summed
            .into_iter()
            .filter(|(_, value)| value.is_finite() && *value > 0.0)
            .map(|(canonical, value)| (canonical, value.log2()))
            .collect::<HashMap<_, _>>();
        if !log_row.is_empty() {
            log2_cells.insert(*cell, log_row);
        }
    }

    // (2) Per-row pseudo-reference: log2 median across samples (skipna). A row
    // is identified by (protein, canonical); we gather every sample's log2
    // value for that row.
    let mut row_values = HashMap::<QuantilePeptideKey, Vec<f64>>::new();
    for (cell, peptides) in &log2_cells {
        for (peptide, value) in peptides {
            row_values
                .entry(QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                })
                .or_default()
                .push(*value);
        }
    }
    let reference = row_values
        .into_iter()
        .filter_map(|(key, mut values)| finite_median(&mut values).map(|median| (key, median)))
        .collect::<HashMap<_, _>>();

    // (3) Fit (intercept, slope) per sample via Huber IRLS against the
    // reference, then transform each value as (y - intercept) / slope. The
    // identity fit (0, 1) leaves the column untouched, matching the Python
    // guards (< 5 valid points, non-finite / near-zero slope, fit failure).
    let mut samples = log2_cells
        .keys()
        .map(|cell| cell.sample)
        .collect::<Vec<_>>();
    samples.sort();
    samples.dedup();

    let mut fits = HashMap::<SampleId, (f64, f64)>::new();
    for sample in samples {
        // Collect (reference, y) pairs where both are finite, in a stable
        // row-id order so the regression is deterministic.
        let mut pairs = Vec::<(QuantilePeptideKey, f64, f64)>::new();
        for (cell, peptides) in &log2_cells {
            if cell.sample != sample {
                continue;
            }
            for (peptide, value) in peptides {
                let key = QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                };
                if let Some(reference_value) = reference.get(&key).copied() {
                    if reference_value.is_finite() && value.is_finite() {
                        pairs.push((key, reference_value, *value));
                    }
                }
            }
        }
        pairs.sort_by_key(|pair| pair.0);
        let reference_column = pairs.iter().map(|(_, x, _)| *x).collect::<Vec<_>>();
        let response_column = pairs.iter().map(|(_, _, y)| *y).collect::<Vec<_>>();
        fits.insert(sample, fit_rlr_sample(&reference_column, &response_column));
    }

    // (4) Write back: clear each allowed cell and reinsert canonical ->
    // 2 ** ((log2(value) - intercept) / slope) (exponentiate because we log2'd
    // and the rollup consumes linear values).
    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(log_row) = log2_cells.get(cell) else {
            continue;
        };
        let (intercept, slope) = fits.get(&cell.sample).copied().unwrap_or((0.0, 1.0));
        peptides.clear();
        for (peptide, value) in log_row {
            let normalized = (value - intercept) / slope;
            peptides.insert(*peptide, normalized.exp2());
        }
    }
}

/// TMM (Trimmed Mean of M-values) sample normalization, mirroring
/// `TMMNormalizer(m_trim=0.3, a_trim=0.05, ref_sample=None, log_transform=False)`
/// in `python/mokume/normalization/tmm.py`.
///
/// `stages.apply_tmm` (stages.py:1087-1093) runs the normalizer through
/// `_apply_dataset_normalizer(..., log_space=False)` (stages.py:1011-1042),
/// which pivots the peptide-level long table to the `(protein, canonical) x
/// sample` wide matrix with `aggfunc="sum"` and passes it in as *raw linear*
/// intensities (TMM does its own log2 internally, `log_transform=False`). So
/// TMM operates on the SAME summed-canonical-peptide matrix the other dataset
/// methods use, and returns linear intensities.
///
/// TMM produces a single per-sample scalar `norm_factor`, then divides every
/// value in that sample's column by it (tmm.py:334-343 uses division). Because
/// the factor is uniform across every peptide of a sample it commutes with the
/// downstream canonical collapse `finalize` performs, so we divide the original
/// peptidoform cells in place (NOT the canonical-summed values) and let
/// `finalize` remain the sole canonical-collapse site. The canonical-summed wide
/// matrix built below is used only to fit the factors, matching Python's pivot;
/// scaling each peptidoform by the same factor then collapsing reproduces
/// Python's melt-back cell-for-cell without a double collapse.
///
/// Determinism: the factor math (library sizes, reference selection via
/// `np.percentile` linear interpolation, the double-trimmed weighted mean) is
/// delegated to `mokume_normalization::tmm_norm_factors`, which does all work in
/// `f64` and sorts with `f64::total_cmp`. To match pandas' pivoted column order
/// (columns sorted by sample label), we order the wide-matrix columns by the
/// sample's string name; this fixes the reference `idxmin` tie-break to the
/// first sample in label order, exactly like pandas.
fn apply_tmm_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    samples: &StringIdRegistry<SampleId>,
) {
    // (1) Collapse each allowed cell's peptidoforms to canonical peptides by
    // SUM. Rows of the wide matrix are (protein, canonical); columns are
    // samples. Keep every finite sum (including <= 0) so the matrix has the
    // exact shape Python's pivot produces; `tmm_norm_factors` treats zero /
    // non-finite as missing internally.
    let mut summed_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let mut summed = HashMap::<PeptideId, f64>::new();
        for (peptide, intensity) in peptides {
            let canonical = peptide_to_canonical
                .get(peptide)
                .copied()
                .unwrap_or(*peptide);
            *summed.entry(canonical).or_insert(0.0) += *intensity;
        }
        if !summed.is_empty() {
            summed_cells.insert(*cell, summed);
        }
    }
    if summed_cells.is_empty() {
        return;
    }

    // (2) Build the deterministic row order: every (protein, canonical) key that
    // appears in any cell, sorted by id so all columns share the same row order.
    let mut row_keys: Vec<QuantilePeptideKey> = summed_cells
        .iter()
        .flat_map(|(cell, peptides)| {
            peptides.keys().map(move |peptide| QuantilePeptideKey {
                protein: cell.protein,
                peptide: *peptide,
            })
        })
        .collect();
    row_keys.sort();
    row_keys.dedup();
    let row_index: HashMap<QuantilePeptideKey, usize> = row_keys
        .iter()
        .enumerate()
        .map(|(index, key)| (*key, index))
        .collect();

    // (3) Column order: samples sorted by their string label, matching pandas'
    // pivoted column ordering (drives the reference tie-break).
    let mut sample_ids: Vec<SampleId> = summed_cells
        .keys()
        .map(|cell| cell.sample)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    sample_ids.sort_by(|a, b| {
        let name_a = samples.resolve(*a).unwrap_or("");
        let name_b = samples.resolve(*b).unwrap_or("");
        name_a.cmp(name_b).then_with(|| {
            let raw_a: u32 = (*a).into();
            let raw_b: u32 = (*b).into();
            raw_a.cmp(&raw_b)
        })
    });
    let column_names: Vec<String> = sample_ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            samples
                .resolve(*id)
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("__sample_{index}"))
        })
        .collect();
    let sample_column: HashMap<SampleId, usize> = sample_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();

    // (4) Populate the column-major wide matrix. Missing (protein, canonical) x
    // sample entries stay 0.0, which `tmm_norm_factors` treats as missing by
    // filtering them out — the same outcome as Python, whose pivot fills gaps
    // with NaN and `replace(0, NaN)` maps zeros to NaN before `skipna` drops them.
    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; row_keys.len()]; sample_ids.len()];
    for (cell, peptides) in &summed_cells {
        let Some(&column) = sample_column.get(&cell.sample) else {
            continue;
        };
        for (peptide, value) in peptides {
            let key = QuantilePeptideKey {
                protein: cell.protein,
                peptide: *peptide,
            };
            if let Some(&row) = row_index.get(&key) {
                matrix[column][row] = *value;
            }
        }
    }

    // (5) Compute the per-sample rescaled factors (tmm.py:296-299) and divide
    // every peptidoform in that sample by its factor (tmm.py:334-343).
    //
    // TMM's factor is a single per-sample scalar, so it commutes with the
    // downstream sum/median canonical collapse that `finalize` performs. We
    // therefore scale the ORIGINAL peptidoform cells in place rather than
    // replacing them with the canonical-summed values: `finalize` remains the
    // sole place that collapses peptidoforms to canonical, which avoids a
    // double collapse (writing canonical ids back into the cell would let
    // `finalize`'s own `collapse_to_canonical` re-map them and merge distinct
    // canonicals, corrupting median/`Median` outputs). The wide matrix built
    // above (canonical-summed, matching Python's pivot) is only used to compute
    // the factors, exactly reproducing Python's `TMMNormalizer` fit on the
    // pivoted matrix; dividing every peptidoform by the same per-sample factor
    // yields the same canonical-collapsed result Python's melt-back produces.
    let factors_by_name = tmm_norm_factors(&column_names, &matrix);
    let factors_by_sample: HashMap<SampleId, f64> = sample_ids
        .iter()
        .zip(column_names.iter())
        .filter_map(|(id, name)| factors_by_name.get(name).map(|factor| (*id, *factor)))
        .collect();

    for (cell, peptides) in cells.iter_mut() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let factor = factors_by_sample.get(&cell.sample).copied().unwrap_or(1.0);
        if !tmm_factor_applies(factor) {
            continue;
        }
        for value in peptides.values_mut() {
            *value /= factor;
        }
    }
}

/// A per-sample TMM factor is applied only when it is a genuine rescale: not the
/// identity (1.0) and a positive, finite number. `finalize` divides every
/// peptidoform by the factor, so the identity and any non-finite factor are
/// skipped to leave the sample untouched (tmm.py:334-343).
fn tmm_factor_applies(factor: f64) -> bool {
    factor != 1.0 && factor > 0.0 && factor.is_finite()
}

/// Median over the finite values of `values` (non-finite dropped, sign
/// preserved). Unlike `mokume_core::stats::median` this keeps negative values,
/// which log2 references and residuals can take. Sorts in place with
/// `f64::total_cmp` for determinism.
fn finite_median(values: &mut Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[midpoint - 1] + values[midpoint]) / 2.0)
    } else {
        Some(values[midpoint])
    }
}

/// Fit a single RLR column: robust regression of `y` on `[1, reference]` via
/// Huber IRLS, returning `(intercept, slope)`.
///
/// Reproduces `sm.RLM(y, sm.add_constant(reference), M=HuberT()).fit()` with
/// statsmodels defaults. Returns the identity fit `(0.0, 1.0)` when fewer than
/// five valid points are available, when the design is singular, or when the
/// fitted slope is non-finite or `|slope| < 1e-6` — mirroring the Python
/// `_fit_sample` guards. With identity, `(y - 0) / 1 == y`, so the column is
/// left unchanged.
fn fit_rlr_sample(reference: &[f64], y: &[f64]) -> (f64, f64) {
    const HUBER_T: f64 = 1.345;
    // 0.75 quantile of the standard normal; statsmodels `scale.mad` default c.
    const MAD_C: f64 = 0.674_489_750_196_081_7;
    const TOL: f64 = 1e-8;
    const MAX_ITER: usize = 50;

    let n = reference.len();
    if n < 5 || y.len() != n {
        return (0.0, 1.0);
    }

    // Ordinary least squares initial estimate of (intercept, slope) via the
    // 2x2 normal equations X^T X b = X^T y, X = [1, reference].
    let Some((mut intercept, mut slope)) =
        solve_weighted_normal_equations(reference, y, &vec![1.0; n])
    else {
        return (0.0, 1.0);
    };

    let residual = |intercept: f64, slope: f64| -> Vec<f64> {
        reference
            .iter()
            .zip(y)
            .map(|(x, y)| y - (intercept + slope * x))
            .collect::<Vec<_>>()
    };

    // MAD scale with center 0: median(|resid|) / c.
    let mad_scale = |resid: &[f64]| -> f64 {
        let mut abs = resid.iter().map(|value| value.abs()).collect::<Vec<_>>();
        finite_median(&mut abs).unwrap_or(0.0) / MAD_C
    };

    // Huber weight: 1 if |z| <= t, else t / |z|, with z = resid / scale.
    let huber_weights = |resid: &[f64], scale: f64| -> Vec<f64> {
        resid
            .iter()
            .map(|value| {
                let abs_z = (value / scale).abs();
                if abs_z <= HUBER_T {
                    1.0
                } else {
                    HUBER_T / abs_z
                }
            })
            .collect::<Vec<_>>()
    };

    // Un-normalized Huber objective evaluated at z = resid / wls_scale, used by
    // the `conv="dev"` stopping rule. `wls_scale` is the weighted-residual
    // SSR / (n - 2), the scale `_MinimalWLS` reports for the current fit.
    let wls_scale = |resid: &[f64], weights: &[f64]| -> f64 {
        let ssr = resid
            .iter()
            .zip(weights)
            .map(|(value, weight)| weight * value * value)
            .sum::<f64>();
        if n > 2 {
            ssr / (n - 2) as f64
        } else {
            0.0
        }
    };
    let huber_objective = |resid: &[f64], scale: f64| -> f64 {
        resid
            .iter()
            .map(|value| {
                let abs_z = (value / scale).abs();
                if abs_z <= HUBER_T {
                    0.5 * abs_z * abs_z
                } else {
                    HUBER_T * abs_z - 0.5 * HUBER_T * HUBER_T
                }
            })
            .sum::<f64>()
    };

    let mut resid = residual(intercept, slope);
    let mut scale = mad_scale(&resid);
    // Iteration 1 deviance: OLS fit (unit weights) at its own WLS scale.
    let initial_wls_scale = wls_scale(&resid, &vec![1.0; n]);
    let mut previous_deviance = if initial_wls_scale > 0.0 {
        huber_objective(&resid, initial_wls_scale)
    } else {
        0.0
    };

    let mut iteration = 1;
    loop {
        if scale == 0.0 {
            break;
        }
        let weights = huber_weights(&resid, scale);
        let Some((next_intercept, next_slope)) =
            solve_weighted_normal_equations(reference, y, &weights)
        else {
            return (0.0, 1.0);
        };
        intercept = next_intercept;
        slope = next_slope;
        resid = residual(intercept, slope);
        scale = mad_scale(&resid); // update_scale=True
        let current_wls_scale = wls_scale(&resid, &weights);
        let deviance = if current_wls_scale > 0.0 {
            huber_objective(&resid, current_wls_scale)
        } else {
            0.0
        };
        iteration += 1;
        let converged = (deviance - previous_deviance).abs() <= TOL || iteration >= MAX_ITER;
        previous_deviance = deviance;
        if converged {
            break;
        }
    }

    if !slope.is_finite() || slope.abs() < 1e-6 {
        return (0.0, 1.0);
    }
    (intercept, slope)
}

/// Solve the weighted least squares of `y` on `[1, x]`, i.e. the 2x2 system
/// `(Xᵀ W X) [intercept, slope]ᵀ = Xᵀ W y` with `X = [1, x]`. Returns `None`
/// when the normal matrix is singular (zero determinant), letting the caller
/// fall back to the identity fit. The closed-form 2x2 solve reproduces the
/// `numpy.linalg.pinv`-based WLS that statsmodels uses for a full-rank,
/// well-conditioned design.
fn solve_weighted_normal_equations(x: &[f64], y: &[f64], weights: &[f64]) -> Option<(f64, f64)> {
    let mut s_w = 0.0;
    let mut s_wx = 0.0;
    let mut s_wxx = 0.0;
    let mut s_wy = 0.0;
    let mut s_wxy = 0.0;
    for ((xi, yi), wi) in x.iter().zip(y).zip(weights) {
        s_w += wi;
        s_wx += wi * xi;
        s_wxx += wi * xi * xi;
        s_wy += wi * yi;
        s_wxy += wi * xi * yi;
    }
    // Normal matrix [[s_w, s_wx], [s_wx, s_wxx]]; right-hand side [s_wy, s_wxy].
    let determinant = s_w * s_wxx - s_wx * s_wx;
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    let intercept = (s_wxx * s_wy - s_wx * s_wxy) / determinant;
    let slope = (s_w * s_wxy - s_wx * s_wy) / determinant;
    (intercept.is_finite() && slope.is_finite()).then_some((intercept, slope))
}

/// LOESS (LOcally Estimated Scatterplot Smoothing) sample normalization.
/// Mirrors the Python `LOESSNormalizer.fit_transform` (frac=0.75,
/// reference="median") applied by `NormalizationStage.apply_loess` in log2
/// space (`_apply_dataset_normalizer(..., log_space=True)`).
///
/// Algorithm (Cleveland WS, J Am Stat Assoc 1979; Yang et al, NAR 2002):
/// build the (protein, canonical) x sample matrix in log2 space. Let `ref[row]`
/// be the across-sample row reference (pandas `median(axis=1)`, NaN-skipping; for
/// an even count this is the mean of the two central order statistics). For each
/// sample column with at least 10 finite (sample, ref) pairs, form the MA points
/// `M = sample - ref` (log-ratio) and `A = (sample + ref) / 2` (average), fit
/// `M ~ A` by LOWESS (statsmodels `lowess`, frac=0.75, it=3, delta=0.0,
/// return_sorted=False), and replace each fitted value with `sample - fitted(A)`.
/// Columns with fewer than 10 valid pairs are left UNCHANGED (the Python
/// `if valid.sum() < 10: continue` guard). The pipeline log2s before
/// quantification and the rollup consumes linear values, so results are
/// exponentiated back here.
fn apply_loess_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) {
    // Collapse each allowed cell to canonical peptides by SUM, then log2 the
    // strictly positive finite values (matching `np.log2(wide.replace(0, nan))`).
    let mut log2_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let mut summed = HashMap::<PeptideId, f64>::new();
        for (peptide, intensity) in peptides {
            let canonical = peptide_to_canonical
                .get(peptide)
                .copied()
                .unwrap_or(*peptide);
            *summed.entry(canonical).or_insert(0.0) += *intensity;
        }
        let log_row = summed
            .into_iter()
            .filter(|(_, value)| value.is_finite() && *value > 0.0)
            .map(|(canonical, value)| (canonical, value.log2()))
            .collect::<HashMap<_, _>>();
        if !log_row.is_empty() {
            log2_cells.insert(*cell, log_row);
        }
    }

    // Per-row (protein, canonical) reference = NaN-skipping median across the
    // samples that observe the row. Collect every log2 value per row first.
    let mut row_values = HashMap::<QuantilePeptideKey, Vec<f64>>::new();
    for (cell, peptides) in &log2_cells {
        for (peptide, value) in peptides {
            row_values
                .entry(QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                })
                .or_default()
                .push(*value);
        }
    }
    let mut reference = HashMap::<QuantilePeptideKey, f64>::new();
    for (key, mut values) in row_values {
        if let Some(median) = median_finite(&mut values) {
            reference.insert(key, median);
        }
    }

    // Fit and apply per sample column. A sample is corrected only if it has at
    // least 10 finite (sample, reference) pairs; otherwise it passes through.
    let mut by_sample = HashMap::<SampleId, Vec<(QuantilePeptideKey, f64)>>::new();
    for (cell, peptides) in &log2_cells {
        for (peptide, value) in peptides {
            by_sample.entry(cell.sample).or_default().push((
                QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                },
                *value,
            ));
        }
    }

    // Deterministic sample ordering so the fit is reproducible run to run.
    let mut samples = by_sample.keys().copied().collect::<Vec<_>>();
    samples.sort();

    let mut corrected = HashMap::<(QuantilePeptideKey, SampleId), f64>::new();
    for sample in samples {
        let rows = &by_sample[&sample];
        // Pair each row's log2 value with its reference; drop rows lacking a
        // finite reference (matches the `sample.notna() & ref.notna()` mask).
        let mut paired = rows
            .iter()
            .filter_map(|(key, value)| {
                reference
                    .get(key)
                    .filter(|r| r.is_finite() && value.is_finite())
                    .map(|r| (*key, *value, *r))
            })
            .collect::<Vec<_>>();
        if paired.len() < 10 {
            // < 10 valid points: leave the column unchanged.
            for (key, value, _) in &paired {
                corrected.insert((*key, sample), *value);
            }
            continue;
        }
        // Deterministic tie-break: sort by (A ascending, key) before fitting so
        // the window selection and unsort match a stable reference order.
        paired.sort_by(|left, right| {
            let a_left = (left.1 + left.2) / 2.0;
            let a_right = (right.1 + right.2) / 2.0;
            a_left
                .total_cmp(&a_right)
                .then_with(|| left.0.cmp(&right.0))
        });
        let a_values = paired
            .iter()
            .map(|(_, value, r)| (value + r) / 2.0)
            .collect::<Vec<_>>();
        let m_values = paired
            .iter()
            .map(|(_, value, r)| value - r)
            .collect::<Vec<_>>();
        let fitted =
            mokume_core::stats::lowess_fit(&a_values, &m_values, LOESS_FRAC, LOESS_ITERATIONS);
        for ((key, value, _), bias) in paired.iter().zip(fitted) {
            corrected.insert((*key, sample), value - bias);
        }
    }

    // Write canonical results back into the cells, exponentiating out of log2.
    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(log_row) = log2_cells.get(cell) else {
            continue;
        };
        peptides.clear();
        for peptide in log_row.keys() {
            let key = QuantilePeptideKey {
                protein: cell.protein,
                peptide: *peptide,
            };
            if let Some(value) = corrected.get(&(key, cell.sample)).copied() {
                peptides.insert(*peptide, value.exp2());
            }
        }
    }
}

/// LOWESS smoothing fraction used by `LOESSNormalizer` (frac=0.75).
const LOESS_FRAC: f64 = 0.75;
/// Robustifying reweight iterations matching the statsmodels `lowess` default
/// (`it=3`).
const LOESS_ITERATIONS: usize = 3;

/// DirectLFQ-style hierarchical sample normalization, mirroring
/// `mokume.normalization.hierarchical.HierarchicalSampleNormalizer` exactly as
/// the pipeline invokes it in `NormalizationStage.apply_hierarchical`.
///
/// `apply_hierarchical` builds the (protein, canonical) x sample wide matrix
/// (`aggfunc="sum"`), replaces 0 with NaN, takes `np.log2`, fits/transforms with
/// `HierarchicalSampleNormalizer(num_samples_quadratic=cfg.directlfq_num_samples_quadratic,
/// selected_proteins=None)` (default `num_samples_quadratic = 50`,
/// `distance_metric = MEDIAN`, `min_overlap = 10`), then `2 ** result`. So the
/// outer wrap is `log_space = true`: we log2 the linear matrix, run the whole
/// alignment in log2 space, and exponentiate on write-back. `selected_proteins`
/// is always `None` in the pipeline call, so the protein filter is a no-op here.
///
/// Algorithm (per the Python class):
///  1. Collapse each allowed cell's peptidoforms to canonical peptides by SUM via
///     `peptide_to_canonical`; keep only strictly-positive finite sums and store
///     their `log2`. Rows of the wide matrix are `(protein, canonical)`, columns
///     are samples. Missing/non-positive cells become NaN (i.e. absent here).
///  2. Order samples ascending by their resolved NAME string -> column index
///     `0..n`. The Python pivot's column order is load-bearing because the
///     distance-matrix indexing, `leaves_list`, and the cumulative linear-shift
///     propagation (which sample anchors each cluster at shift 0) all depend on
///     it; `pivot_table(columns=SAMPLE_ID)` produces lexicographically sorted
///     sample columns, so the columns must be sorted by sample name here, NOT by
///     `SampleId` (which is parquet-stream insertion order and does not match the
///     pivot order on real data). `SampleId` breaks any name ties deterministically.
///  3. Edge cases:
///     - n == 0: nothing to do.
///     - n == 1: shift 0.0.
///     - n == 2: `shift = median(c0 over overlap) - median(c1 over overlap)` if
///       the pairwise overlap count >= min_overlap, else 0.0; factors {c0:0, c1:shift}.
///  4. n >= 3: pairwise distance matrix `D[i,j] = |median(log2 col i over i&j
///     overlap) - median(...col j...)|` when overlap >= min_overlap, else +inf
///     (MEDIAN metric). If all D are inf -> all shifts 0.0. Otherwise replace inf
///     with `max_finite * 10`, run scipy `linkage(method="average")` (UPGMA via the
///     nn-chain algorithm) + `leaves_list` to get `leaf_order`.
///  5. Compute shifts in `leaf_order`:
///     - n <= num_samples_quadratic: quadratic optimization. The Python
///       `least_squares(method="lm")` minimizes
///       `sum_{i<j, overlap>=min} w_ij * ((s_i - s_j) - md_ij)^2` with
///       `w_ij = sqrt(overlap_ij)`, `md_ij = median(col_i over i&j) -
///       median(col_j over i&j)`, `s_0` fixed at 0. Because the residuals are
///       LINEAR in the shifts, this is a weighted linear least-squares whose unique
///       minimizer (when the constraint graph is connected) is the normal-equations
///       solution `(A^T W A) s = A^T W b`; LM converges to it (verified < 1e-15).
///       If the normal matrix is singular (disconnected graph) -> fall back to the
///       linear shifts, matching Python's "did not converge -> linear" path.
///     - n > num_samples_quadratic: linear optimization. Walk `leaf_order`; each
///       step `shift = median(prev over prev&curr) - median(curr over prev&curr)`
///       if overlap >= min_overlap else 0.0; accumulate cumulatively.
///  6. Write-back: for each allowed cell, `value <- 2 ** (log2(value) +
///     shift[sample])`. Cells that were NaN/non-positive stay dropped.
///
/// Determinism: samples are sorted by `(name, SampleId)` to reproduce the Python
/// pivot's lexicographic column order; per-row sample lists are sorted by
/// `(value, row_id)` only for medians (order-independent); the nn-chain reads a
/// dense condensed distance array indexed by sorted sample position; the
/// post-merge `Z` is stably argsorted by distance then relabeled with union-find,
/// exactly reproducing scipy. Verified bit-order-identical to scipy `leaves_list`
/// on 92,573 exhaustive tie-heavy matrices (n=3..6) and 5,000 random matrices
/// (n up to 40, ties injected): 0 mismatches.
fn apply_hierarchical_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    sample_registry: &StringIdRegistry<SampleId>,
) {
    const MIN_OVERLAP: usize = 10;
    const NUM_SAMPLES_QUADRATIC: usize = 50;
    // (No trimming in hierarchical normalization.)

    // (1) Collapse to canonical and log2 the strictly-positive sums.
    let mut log2_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let mut summed = HashMap::<PeptideId, f64>::new();
        for (peptide, intensity) in peptides {
            let canonical = peptide_to_canonical
                .get(peptide)
                .copied()
                .unwrap_or(*peptide);
            *summed.entry(canonical).or_insert(0.0) += *intensity;
        }
        let log_row = summed
            .into_iter()
            .filter(|(_, value)| value.is_finite() && *value > 0.0)
            .map(|(canonical, value)| (canonical, value.log2()))
            .collect::<HashMap<_, _>>();
        if !log_row.is_empty() {
            log2_cells.insert(*cell, log_row);
        }
    }
    if log2_cells.is_empty() {
        return;
    }

    // (2) Sample columns ordered ascending by sample NAME (then SampleId as a
    // tie-break), matching the Python `pivot_table(columns=SAMPLE_ID)` column
    // order. `columns[idx]` maps each (protein, canonical) row to its log2 value
    // in sample `samples[idx]`. The ordering is load-bearing: it drives the
    // distance-matrix indices, `leaves_list`, and the cumulative shift chain.
    let mut samples = log2_cells
        .keys()
        .map(|cell| cell.sample)
        .collect::<Vec<_>>();
    samples.sort_unstable();
    samples.dedup();
    // Re-sort by resolved name (lexicographic), falling back to SampleId so the
    // order stays total and deterministic even if a name fails to resolve.
    samples.sort_by(|left, right| {
        sample_registry
            .resolve(*left)
            .cmp(&sample_registry.resolve(*right))
            .then_with(|| left.cmp(right))
    });
    let n = samples.len();

    let mut columns = vec![HashMap::<QuantilePeptideKey, f64>::new(); n];
    let sample_index = samples
        .iter()
        .enumerate()
        .map(|(idx, sample)| (*sample, idx))
        .collect::<HashMap<SampleId, usize>>();
    for (cell, peptides) in &log2_cells {
        let Some(&idx) = sample_index.get(&cell.sample) else {
            continue;
        };
        for (peptide, value) in peptides {
            columns[idx].insert(
                QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                },
                *value,
            );
        }
    }

    // (3)+(4)+(5) Compute per-column log2 shifts.
    let shifts = hierarchical_compute_shifts(&columns, MIN_OVERLAP, NUM_SAMPLES_QUADRATIC);

    // (6) Write back: 2 ** (log2(value) + shift[sample]).
    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(log_row) = log2_cells.get(cell) else {
            continue;
        };
        let Some(&idx) = sample_index.get(&cell.sample) else {
            continue;
        };
        let shift = shifts.get(idx).copied().unwrap_or(0.0);
        peptides.clear();
        for (peptide, value) in log_row {
            peptides.insert(*peptide, (value + shift).exp2());
        }
    }
}

/// Per-column log2 shift factors, returned in the same index order as `columns`.
/// Mirrors `HierarchicalSampleNormalizer.fit` (`_compute_distance_matrix` +
/// scipy clustering + `_compute_shifts`).
fn hierarchical_compute_shifts(
    columns: &[HashMap<QuantilePeptideKey, f64>],
    min_overlap: usize,
    num_samples_quadratic: usize,
) -> Vec<f64> {
    let n = columns.len();
    match n {
        0 => return Vec::new(),
        1 => return vec![0.0],
        2 => {
            // shift second column to match the first over their pairwise overlap.
            let shift =
                hierarchical_pair_median_diff(&columns[0], &columns[1], min_overlap).unwrap_or(0.0);
            return vec![0.0, shift];
        }
        _ => {}
    }

    // (4) Pairwise distance matrix (MEDIAN metric): D[i][j] = |md_ij|, +inf when
    // the overlap is below `min_overlap`. `md_ij` is the signed median difference.
    let mut distance = vec![vec![0.0_f64; n]; n];
    let mut all_inf = true;
    let mut max_finite = f64::NEG_INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let value = match hierarchical_pair_median_diff(&columns[i], &columns[j], min_overlap) {
                Some(diff) => {
                    let absolute = diff.abs();
                    if absolute.is_finite() {
                        all_inf = false;
                        max_finite = max_finite.max(absolute);
                    }
                    absolute
                }
                None => f64::INFINITY,
            };
            distance[i][j] = value;
            distance[j][i] = value;
        }
    }
    if all_inf {
        // No overlapping values between any pair -> zero shifts.
        return vec![0.0; n];
    }

    // Replace +inf with max_finite * 10 for clustering (matches the Python guard).
    let inf_replacement = max_finite * 10.0;
    for i in 0..n {
        // `split_at_mut` borrows row `i` and each later row `j` simultaneously so
        // both symmetric entries can be written without cloning.
        let (head, tail) = distance.split_at_mut(i + 1);
        let row_i = &mut head[i];
        for (offset, row_j) in tail.iter_mut().enumerate() {
            let j = i + 1 + offset;
            if row_i[j].is_infinite() {
                row_i[j] = inf_replacement;
                row_j[i] = inf_replacement;
            }
        }
    }

    // scipy linkage(method="average") + leaves_list -> clustering order.
    let leaf_order = hierarchical_leaf_order(&distance, n);

    if n <= num_samples_quadratic {
        hierarchical_shifts_quadratic(columns, &leaf_order, min_overlap, n)
    } else {
        hierarchical_shifts_linear(columns, &leaf_order, min_overlap, n)
    }
}

/// Signed median difference between two columns over their shared rows, when the
/// overlap count is at least `min_overlap`; otherwise `None`.
/// `median(col_a over overlap) - median(col_b over overlap)` in log2 space.
fn hierarchical_pair_median_diff(
    column_a: &HashMap<QuantilePeptideKey, f64>,
    column_b: &HashMap<QuantilePeptideKey, f64>,
    min_overlap: usize,
) -> Option<f64> {
    let (small, large) = if column_a.len() <= column_b.len() {
        (column_a, column_b)
    } else {
        (column_b, column_a)
    };
    let mut a_values = Vec::<f64>::new();
    let mut b_values = Vec::<f64>::new();
    for (key, &value_small) in small {
        let Some(&value_large) = large.get(key) else {
            continue;
        };
        if value_small.is_finite() && value_large.is_finite() {
            if column_a.len() <= column_b.len() {
                a_values.push(value_small);
                b_values.push(value_large);
            } else {
                a_values.push(value_large);
                b_values.push(value_small);
            }
        }
    }
    if a_values.len() < min_overlap {
        return None;
    }
    let median_a = finite_median(&mut a_values)?;
    let median_b = finite_median(&mut b_values)?;
    Some(median_a - median_b)
}

/// Cumulative linear shifts along the clustering order (`_compute_shifts_linear`).
fn hierarchical_shifts_linear(
    columns: &[HashMap<QuantilePeptideKey, f64>],
    leaf_order: &[usize],
    min_overlap: usize,
    n: usize,
) -> Vec<f64> {
    let mut shifts = vec![0.0_f64; n];
    let mut cumulative = 0.0;
    for window in leaf_order.windows(2) {
        let previous = window[0];
        let current = window[1];
        let step =
            hierarchical_pair_median_diff(&columns[previous], &columns[current], min_overlap)
                .unwrap_or(0.0);
        cumulative += step;
        shifts[current] = cumulative;
    }
    shifts
}

/// Weighted least-squares shifts (`_compute_shifts_quadratic`). The first leaf is
/// pinned at 0.0; the remaining shifts minimize
/// `sum_{i<j, overlap>=min} overlap_ij * ((s_i - s_j) - md_ij)^2`. Because the
/// residuals are linear in the shifts, the unique minimizer (connected graph) is
/// the normal-equations solution. Falls back to the linear shifts when the normal
/// matrix is singular (disconnected graph), matching Python's "did not converge".
fn hierarchical_shifts_quadratic(
    columns: &[HashMap<QuantilePeptideKey, f64>],
    leaf_order: &[usize],
    min_overlap: usize,
    n: usize,
) -> Vec<f64> {
    // Reorder columns into clustering order: `ordered[k]` is column `leaf_order[k]`.
    // Variable `k` in `1..n` is the free shift of leaf `k`; leaf 0 is fixed at 0.
    let free = n - 1;
    let mut normal = vec![vec![0.0_f64; free]; free];
    let mut rhs = vec![0.0_f64; free];

    // Each kept pair (p, q) with p < q (positions in `leaf_order`) contributes a
    // residual sqrt(w) * ((s_p - s_q) - md). md is measured in leaf order: column
    // leaf_order[p] minus column leaf_order[q].
    let mut any_constraint = false;
    for p in 0..n {
        for q in (p + 1)..n {
            let column_p = &columns[leaf_order[p]];
            let column_q = &columns[leaf_order[q]];
            // overlap count and median diff over shared rows.
            let overlap = hierarchical_pair_overlap(column_p, column_q);
            if overlap < min_overlap {
                continue;
            }
            let Some(md) = hierarchical_pair_median_diff(column_p, column_q, min_overlap) else {
                continue;
            };
            any_constraint = true;
            let weight = overlap as f64; // sqrt(w)^2 cancels into the normal matrix.
                                         // Coefficients of (s_p - s_q): +1 on variable (p-1) if p>0, -1 on (q-1).
                                         // Variable index for leaf position `k` (k>=1) is `k-1`.
            let mut coefficients = Vec::<(usize, f64)>::with_capacity(2);
            if p >= 1 {
                coefficients.push((p - 1, 1.0));
            }
            if q >= 1 {
                coefficients.push((q - 1, -1.0));
            }
            for &(row_index, row_coeff) in &coefficients {
                rhs[row_index] += weight * row_coeff * md;
                for &(col_index, col_coeff) in &coefficients {
                    normal[row_index][col_index] += weight * row_coeff * col_coeff;
                }
            }
        }
    }

    if !any_constraint {
        return hierarchical_shifts_linear(columns, leaf_order, min_overlap, n);
    }

    // Solve the symmetric system; on singularity fall back to linear shifts.
    let Some(solution) = solve_symmetric_system(normal, rhs) else {
        return hierarchical_shifts_linear(columns, leaf_order, min_overlap, n);
    };

    // Map leaf-order shifts back to original column indices.
    let mut shifts = vec![0.0_f64; n];
    for position in 1..n {
        shifts[leaf_order[position]] = solution[position - 1];
    }
    shifts
}

/// Count of shared finite rows between two columns (overlap size).
fn hierarchical_pair_overlap(
    column_a: &HashMap<QuantilePeptideKey, f64>,
    column_b: &HashMap<QuantilePeptideKey, f64>,
) -> usize {
    let (small, large) = if column_a.len() <= column_b.len() {
        (column_a, column_b)
    } else {
        (column_b, column_a)
    };
    small
        .iter()
        .filter(|(key, value)| {
            value.is_finite() && large.get(key).is_some_and(|other| other.is_finite())
        })
        .count()
}

/// Reproduce scipy `leaves_list(linkage(squareform(distance), method="average"))`.
/// Runs the nn-chain UPGMA agglomeration on a condensed distance buffer, stably
/// sorts the merges by distance, relabels with union-find, then pre-order
/// traverses the dendrogram. Verified identical to scipy across 97k+ cases.
fn hierarchical_leaf_order(distance: &[Vec<f64>], n: usize) -> Vec<usize> {
    if n == 1 {
        return vec![0];
    }
    // Condensed buffer: index(i, j) for i < j.
    let condensed_index = |i: usize, j: usize| -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        n * i - (i * (i + 1)) / 2 + (j - i - 1)
    };
    let mut condensed = vec![0.0_f64; n * (n - 1) / 2];
    for i in 0..n {
        for j in (i + 1)..n {
            condensed[condensed_index(i, j)] = distance[i][j];
        }
    }

    // nn-chain: each row of `merges` is (cluster_a, cluster_b, distance).
    let mut size = vec![1usize; n];
    let mut chain = vec![0usize; n];
    let mut chain_length = 0usize;
    let mut merges = Vec::<(usize, usize, f64)>::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        if chain_length == 0 {
            chain_length = 1;
            for (index, &cluster_size) in size.iter().enumerate() {
                if cluster_size > 0 {
                    chain[0] = index;
                    break;
                }
            }
        }
        let mut x;
        let mut y = 0usize;
        let mut current_min;
        loop {
            x = chain[chain_length - 1];
            if chain_length > 1 {
                y = chain[chain_length - 2];
                current_min = condensed[condensed_index(x, y)];
            } else {
                current_min = f64::INFINITY;
            }
            for (index, &cluster_size) in size.iter().enumerate() {
                if cluster_size == 0 || index == x {
                    continue;
                }
                let candidate = condensed[condensed_index(x, index)];
                if candidate < current_min {
                    current_min = candidate;
                    y = index;
                }
            }
            if chain_length > 1 && y == chain[chain_length - 2] {
                break;
            }
            chain[chain_length] = y;
            chain_length += 1;
        }
        chain_length -= 2;
        let (low, high) = if x > y { (y, x) } else { (x, y) };
        let size_low = size[low];
        let size_high = size[high];
        merges.push((low, high, current_min));
        // Average linkage Lance-Williams update onto `high`; deactivate `low`.
        size[low] = 0;
        size[high] = size_low + size_high;
        let denominator = (size_low + size_high) as f64;
        for index in 0..n {
            if size[index] == 0 || index == high {
                continue;
            }
            let distance_low = condensed[condensed_index(low, index)];
            let distance_high = condensed[condensed_index(high, index)];
            condensed[condensed_index(high, index)] =
                (size_low as f64 * distance_low + size_high as f64 * distance_high) / denominator;
        }
    }

    // Stable argsort of merges by distance (scipy uses a stable sort here).
    let mut order = (0..merges.len()).collect::<Vec<usize>>();
    order.sort_by(|&left, &right| {
        merges[left]
            .2
            .total_cmp(&merges[right].2)
            .then_with(|| left.cmp(&right))
    });

    // Union-find relabel: children stored as (left_child, right_child) per merged
    // node id in `[n, 2n-2]`.
    let mut parent = (0..(2 * n - 1)).collect::<Vec<usize>>();
    let mut component_size = vec![1usize; 2 * n - 1];
    let mut children = vec![(0usize, 0usize); n - 1];
    let mut next_id = n;
    let find = |parent: &mut Vec<usize>, mut node: usize| -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    };
    for (slot, &merge_index) in order.iter().enumerate() {
        let (raw_a, raw_b, _) = merges[merge_index];
        let root_a = find(&mut parent, raw_a);
        let root_b = find(&mut parent, raw_b);
        let (low, high) = if root_a < root_b {
            (root_a, root_b)
        } else {
            (root_b, root_a)
        };
        children[slot] = (low, high);
        component_size[next_id] = component_size[low] + component_size[high];
        parent[low] = next_id;
        parent[high] = next_id;
        next_id += 1;
    }

    // Pre-order traversal from the root (id 2n-2), left child first.
    let mut leaves = Vec::<usize>::with_capacity(n);
    let mut stack = vec![2 * n - 2];
    while let Some(node) = stack.pop() {
        if node < n {
            leaves.push(node);
        } else {
            let (left, right) = children[node - n];
            // push right first so the left child is visited first on pop.
            stack.push(right);
            stack.push(left);
        }
    }
    leaves
}

/// Solve the symmetric positive-(semi)definite linear system `matrix * x = rhs`
/// via Gaussian elimination with partial pivoting. Returns `None` if the matrix
/// is singular (used as the "did not converge" fall-back signal). Deterministic:
/// pivot ties resolve to the lowest row index.
fn solve_symmetric_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let size = rhs.len();
    if size == 0 {
        return Some(Vec::new());
    }
    for column in 0..size {
        // Partial pivot: largest |value| in this column at or below the diagonal.
        let mut pivot_row = column;
        let mut pivot_magnitude = matrix[column][column].abs();
        for (offset, candidate) in matrix[column + 1..].iter().enumerate() {
            let magnitude = candidate[column].abs();
            if magnitude > pivot_magnitude {
                pivot_magnitude = magnitude;
                pivot_row = column + 1 + offset;
            }
        }
        if pivot_magnitude <= 1e-12 {
            return None;
        }
        if pivot_row != column {
            matrix.swap(column, pivot_row);
            rhs.swap(column, pivot_row);
        }
        // Eliminate below the pivot. `split_at_mut` lets us borrow the pivot row
        // and a target row simultaneously without cloning.
        let (upper, lower) = matrix.split_at_mut(column + 1);
        let pivot = &upper[column];
        let pivot_value = pivot[column];
        for (offset, target) in lower.iter_mut().enumerate() {
            let row = column + 1 + offset;
            let factor = target[column] / pivot_value;
            if factor == 0.0 {
                continue;
            }
            for index in column..size {
                target[index] -= factor * pivot[index];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    // Back substitution.
    let mut solution = vec![0.0_f64; size];
    for row in (0..size).rev() {
        let mut accumulator = rhs[row];
        for column in (row + 1)..size {
            accumulator -= matrix[row][column] * solution[column];
        }
        let diagonal = matrix[row][row];
        if diagonal.abs() <= 1e-12 {
            return None;
        }
        solution[row] = accumulator / diagonal;
    }
    Some(solution)
}

fn apply_quantile_to_lfq_traces(
    traces: &mut HashMap<PeptideCellKey, f64>,
    allowed_cells: &HashSet<CellKey>,
) {
    let assignments =
        quantile_normalized_assignments(traces.iter().filter_map(|(key, intensity)| {
            allowed_cells
                .contains(&CellKey {
                    protein: key.protein,
                    sample: key.sample,
                })
                .then_some((
                    QuantilePeptideKey {
                        protein: key.protein,
                        peptide: key.peptide,
                    },
                    key.sample,
                    *intensity,
                ))
        }));
    for (key, intensity) in traces {
        if let Some(normalized) = assignments
            .get(&(
                QuantilePeptideKey {
                    protein: key.protein,
                    peptide: key.peptide,
                },
                key.sample,
            ))
            .copied()
        {
            *intensity = normalized;
        }
    }
}

/// Quantile-normalize each sample (column) by mapping every value to a mean
/// reference distribution at the value's rank fraction. Mirrors the standard
/// quantile normalization used by limma `normalizeQuantiles` / preprocessCore:
/// columns of differing observed lengths (missing values) are handled by
/// interpolating the rank fraction in [0, 1] into the mean reference. Tied
/// values share their average-rank fraction. Returns the normalized intensity
/// per (row, sample); callers overwrite the original intensity with it.
fn quantile_normalized_assignments<K>(
    measurements: impl IntoIterator<Item = (K, SampleId, f64)>,
) -> HashMap<(K, SampleId), f64>
where
    K: Copy + Eq + Ord + std::hash::Hash,
{
    let mut by_sample = HashMap::<SampleId, Vec<(K, f64)>>::new();
    for (row, sample, intensity) in measurements {
        if intensity.is_finite() {
            by_sample.entry(sample).or_default().push((row, intensity));
        }
    }

    let mut grid_size = 0usize;
    for values in by_sample.values_mut() {
        values.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        grid_size = grid_size.max(values.len());
    }
    if grid_size == 0 {
        return HashMap::new();
    }

    // Mean reference distribution on a uniform [0, 1] grid of `grid_size`
    // points: reference[j] is the cross-sample mean of each column's quantile
    // function evaluated at fraction j / (grid_size - 1).
    let mut reference = vec![0.0; grid_size];
    let column_count = by_sample.len();
    for values in by_sample.values() {
        let sorted = values.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        for (j, slot) in reference.iter_mut().enumerate() {
            *slot += interpolate_sorted(&sorted, grid_fraction(j, grid_size));
        }
    }
    for slot in &mut reference {
        *slot /= column_count as f64;
    }

    let mut assignments = HashMap::new();
    for (sample, values) in by_sample {
        let n = values.len();
        let mut index = 0;
        while index < n {
            let mut end = index + 1;
            while end < n && values[end].1.total_cmp(&values[index].1).is_eq() {
                end += 1;
            }
            // 1-based average rank of the tie group, mapped to a [0, 1] fraction.
            let average_rank = (index + 1 + end) as f64 / 2.0;
            let fraction = if n == 1 {
                0.0
            } else {
                (average_rank - 1.0) / (n - 1) as f64
            };
            let normalized = interpolate_sorted(&reference, fraction);
            for (row, _) in &values[index..end] {
                assignments.insert((*row, sample), normalized);
            }
            index = end;
        }
    }
    assignments
}

/// Fraction in [0, 1] of the `j`-th point on a uniform grid of `size` points.
fn grid_fraction(j: usize, size: usize) -> f64 {
    if size <= 1 {
        0.0
    } else {
        j as f64 / (size - 1) as f64
    }
}

/// Linear interpolation of an ascending-sorted slice at `fraction` in [0, 1],
/// where `fraction` maps to fractional index `fraction * (len - 1)`.
fn interpolate_sorted(sorted: &[f64], fraction: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        len => {
            let position = fraction.clamp(0.0, 1.0) * (len - 1) as f64;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            if lower == upper {
                sorted[lower]
            } else {
                let weight = position - lower as f64;
                sorted[lower] + (sorted[upper] - sorted[lower]) * weight
            }
        }
    }
}

#[derive(Debug)]
enum ProteinValues {
    Cells(HashMap<CellKey, f64>),
    Rows(HashMap<ProteinId, Vec<(SampleId, f64)>>),
}

impl ProteinValues {
    fn protein_ids(&self) -> HashSet<ProteinId> {
        match self {
            Self::Cells(values) => values.keys().map(|key| key.protein).collect(),
            Self::Rows(rows) => rows.keys().copied().collect(),
        }
    }

    fn sample_ids(&self) -> HashSet<SampleId> {
        match self {
            Self::Cells(values) => values.keys().map(|key| key.sample).collect(),
            Self::Rows(rows) => rows
                .values()
                .flat_map(|values| values.iter().map(|(sample, _)| *sample))
                .collect(),
        }
    }
}

#[derive(Debug)]
struct ProteinMatrix {
    proteins: StringIdRegistry<ProteinId>,
    samples: StringIdRegistry<SampleId>,
    allowed_proteins: HashSet<ProteinId>,
    excluded_samples: HashSet<SampleId>,
    /// Per-protein unique-canonical-peptide count, captured at ingest before the
    /// `min_unique_peptides` cell filter. Consumed only by the DEqMS DE path
    /// (the count-aware `spectraCounteBayes` moderation); every other stage
    /// ignores it. Mirrors Python's `_load_de_peptide_counts` Series.
    peptide_counts: HashMap<ProteinId, usize>,
    values: ProteinValues,
}

/// `pd.factorize` first-occurrence, 0-indexed labels for the batch prefixes.
/// Delegates to `mokume-stats` (`batch::factorize`) for a single source of truth.
fn factorize_batch_labels(values: &[String]) -> Vec<usize> {
    mokume_stats::batch::factorize(values)
}

/// Resolve a configured batch-detection method name case-insensitively, with
/// `-`/space normalised to `_`. Unknown names are rejected so a requested
/// strategy is never silently replaced with `sample_prefix`.
fn resolve_batch_method(name: &str) -> Result<mokume_stats::batch::BatchDetectionMethod> {
    mokume_stats::batch::BatchDetectionMethod::parse_name(name)
        .ok_or_else(|| invalid_input(format!("unknown batch method `{name}`")))
}

/// Run `mokume-stats`'s `detect_batches` for the protein-matrix flow. Methods
/// that require unavailable run metadata are rejected by the caller.
#[cfg(test)]
fn detect_batches_for_method(
    method: mokume_stats::batch::BatchDetectionMethod,
    sample_names: &[&str],
    column_values: Option<&[String]>,
) -> Result<Vec<usize>> {
    let sample_ids = sample_names
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    mokume_stats::batch::detect_batches(&sample_ids, method, None, column_values).map_err(|error| {
        invalid_input(match error {
            mokume_stats::batch::DetectBatchesError::RunInfoRequired => {
                "batch-method 'run' requires run_info".to_owned()
            }
            mokume_stats::batch::DetectBatchesError::ColumnValuesRequired => {
                "batch column detection requires explicit batch values".to_owned()
            }
            mokume_stats::batch::DetectBatchesError::ColumnLengthMismatch { values, samples } => {
                format!("batch_column_values length ({values}) must match sample count ({samples})")
            }
        })
    })
}

/// Batch-layout validation: require at least two distinct batches, each with at
/// least two samples. The caller turns `None` into an explicit input error.
fn validate_batch_sizes(batch: Vec<usize>) -> Option<Vec<usize>> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &label in &batch {
        *counts.entry(label).or_insert(0) += 1;
    }
    if counts.len() < 2 || counts.values().any(|&count| count < 2) {
        return None;
    }
    Some(batch)
}

fn resolve_reference_batch(
    requested: &str,
    original_labels: &[String],
    encoded_labels: &[usize],
) -> Result<usize> {
    if let Some(index) = original_labels.iter().position(|label| label == requested) {
        return Ok(encoded_labels[index]);
    }
    if let Ok(encoded) = requested.parse::<usize>() {
        if encoded_labels.contains(&encoded) {
            tracing::warn!(
                encoded_batch = encoded,
                "numeric --batch-ref is deprecated; use the original batch label"
            );
            return Ok(encoded);
        }
    }
    let mut labels = original_labels.to_vec();
    labels.sort();
    labels.dedup();
    Err(invalid_input(format!(
        "batch reference `{requested}` is not present; detected labels: {}",
        labels.join(", ")
    )))
}

fn validate_combat_design(
    batch: &[usize],
    covariates: &[Vec<f64>],
    reference: Option<usize>,
) -> Result<()> {
    if covariates.len() != batch.len() {
        return Err(invalid_input(
            "batch covariate row count must match the matrix sample count",
        ));
    }
    let columns = covariates.first().map_or(0, Vec::len);
    if columns == 0 || covariates.iter().any(|row| row.len() != columns) {
        return Err(invalid_input(
            "batch covariate matrix must be non-empty and rectangular",
        ));
    }
    let mut labels = batch.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let label_index = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (*label, index))
        .collect::<HashMap<_, _>>();
    let mut design = vec![vec![0.0; batch.len()]; labels.len() + columns];
    for (sample, label) in batch.iter().enumerate() {
        design[label_index[label]][sample] = 1.0;
    }
    if let Some(reference) = reference {
        let row = label_index[&reference];
        design[row].fill(1.0);
    }
    for column in 0..columns {
        for (sample, values) in covariates.iter().enumerate() {
            design[labels.len() + column][sample] = values[column];
        }
    }
    if row_rank(design) < labels.len() + columns {
        return Err(invalid_input(
            "batch covariates are confounded with batches or each other; ComBat design is singular",
        ));
    }
    Ok(())
}

fn row_rank(mut matrix: Vec<Vec<f64>>) -> usize {
    let rows = matrix.len();
    let columns = matrix.first().map_or(0, Vec::len);
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows).max_by(|&left, &right| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right][column].abs())
        }) else {
            break;
        };
        if matrix[pivot][column].abs() <= 1e-12 {
            continue;
        }
        matrix.swap(rank, pivot);
        let pivot_value = matrix[rank][column];
        for value in &mut matrix[rank][column..] {
            *value /= pivot_value;
        }
        let pivot_row = matrix[rank][column..].to_vec();
        for (row_index, row) in matrix.iter_mut().enumerate() {
            if row_index == rank {
                continue;
            }
            let factor = row[column];
            for (value, pivot) in row[column..].iter_mut().zip(&pivot_row) {
                *value -= factor * pivot;
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

/// SDRF sample-name columns Python searches in priority order
/// (`_find_sdrf_sample_column`), lowercased to match `load_sdrf`.
const SDRF_SAMPLE_COLUMNS: [&str; 4] = ["source name", "sample name", "source_name", "sample_name"];

/// Match a requested covariate column against the lowercased SDRF headers,
/// mirroring `_match_sdrf_column`: an exact lowercased match wins, otherwise the
/// first header that is a substring of the request or contains it.
fn match_sdrf_column(headers: &[String], requested: &str) -> Option<usize> {
    let lower = requested.to_ascii_lowercase();
    if let Some(index) = headers.iter().position(|header| *header == lower) {
        return Some(index);
    }
    headers
        .iter()
        .position(|header| header.contains(&lower) || lower.contains(header.as_str()))
}

/// Per-sample values of one covariate column in matrix-sample order, mirroring
/// `_sample_covariate_values`: an exact `source name` hit first, else the first
/// SDRF sample that is a substring of (or contains) the matrix sample name.
/// Duplicate source names keep the last row (pandas `dict(zip(..))`). An
/// unmatched matrix sample is rejected so a requested covariate never silently
/// turns into an artificial `unknown` category.
fn covariate_values_for_samples(
    raw: &SdrfRawTable,
    sample_col: usize,
    value_col: usize,
    sample_names: &[&str],
) -> Result<Vec<String>> {
    let mut sample_to_value: HashMap<&str, &str> = HashMap::new();
    let mut sdrf_samples: Vec<&str> = Vec::with_capacity(raw.row_count());
    for row in 0..raw.row_count() {
        let key = raw.cell(row, sample_col);
        sample_to_value.insert(key, raw.cell(row, value_col));
        sdrf_samples.push(key);
    }
    sample_names
        .iter()
        .map(|&name| {
            if let Some(value) = sample_to_value.get(name) {
                return Ok((*value).to_owned());
            }
            for sdrf_sample in &sdrf_samples {
                if sdrf_sample.contains(name) || name.contains(*sdrf_sample) {
                    if let Some(value) = sample_to_value.get(*sdrf_sample) {
                        return Ok((*value).to_owned());
                    }
                }
            }
            Err(invalid_input(format!(
                "batch covariate SDRF has no sample matching matrix column `{name}`"
            )))
        })
        .collect()
}

/// Build the sample-major covariate matrix consumed by ComBat. Numeric SDRF
/// columns remain numeric. Categorical columns use k-1 indicator columns so a
/// nominal category is never misrepresented as an ordered scalar.
fn extract_sdrf_covariates(
    raw: &SdrfRawTable,
    sample_names: &[&str],
    covariate_columns: &[String],
) -> Result<Option<Vec<Vec<f64>>>> {
    if covariate_columns.is_empty() {
        return Ok(None);
    }
    let sample_col = SDRF_SAMPLE_COLUMNS
        .iter()
        .find_map(|name| raw.column_index(name))
        .ok_or_else(|| invalid_input("batch covariates require an SDRF sample-name column"))?;

    // Covariate-major encoded columns, then transposed to sample-major below.
    let mut covar_data: Vec<Vec<f64>> = Vec::new();
    for column in covariate_columns {
        let value_col = match_sdrf_column(raw.headers(), column).ok_or_else(|| {
            invalid_input(format!(
                "batch covariate column `{column}` was not found in SDRF"
            ))
        })?;
        let values = covariate_values_for_samples(raw, sample_col, value_col, sample_names)?;
        if values.iter().collect::<HashSet<_>>().len() <= 1 {
            return Err(invalid_input(format!(
                "batch covariate column `{column}` is constant and cannot affect ComBat"
            )));
        }
        covar_data.extend(encode_covariate(column, &values)?);
    }
    Ok(Some(
        (0..sample_names.len())
            .map(|sample| covar_data.iter().map(|encoded| encoded[sample]).collect())
            .collect(),
    ))
}

fn encode_covariate(column: &str, values: &[String]) -> Result<Vec<Vec<f64>>> {
    if values.iter().any(|value| value.trim().is_empty()) {
        return Err(invalid_input(format!(
            "batch covariate column `{column}` contains a missing value"
        )));
    }
    let parsed = values
        .iter()
        .map(|value| value.parse::<f64>())
        .collect::<Vec<_>>();
    let numeric_count = parsed.iter().filter(|value| value.is_ok()).count();
    if numeric_count == values.len() {
        let numeric = parsed
            .into_iter()
            .filter_map(std::result::Result::ok)
            .collect::<Vec<_>>();
        if numeric.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input(format!(
                "batch covariate column `{column}` contains a non-finite value"
            )));
        }
        return Ok(vec![numeric]);
    }
    if numeric_count > 0 {
        return Err(invalid_input(format!(
            "batch covariate column `{column}` mixes numeric and categorical values"
        )));
    }

    let encoded = factorize_batch_labels(values);
    let category_count = encoded.iter().copied().max().unwrap_or(0) + 1;
    Ok((1..category_count)
        .map(|category| {
            encoded
                .iter()
                .map(|value| f64::from(*value == category))
                .collect()
        })
        .collect())
}

/// Explicit batch-column values per matrix sample, mirroring
/// `_get_batch_column_values`: map via `source name -> column` (last row wins on
/// duplicate source names) and default missing samples to `"unknown"`. Returns
/// `None` (Python's not-found branch) when the SDRF lacks the lowercased column
/// or a `source name` column; the caller turns that into the same hard error
/// Python's `_detect_explicit_batches(None)` raises.
fn batch_column_values_for_samples(
    raw: &SdrfRawTable,
    sample_names: &[&str],
    column: &str,
) -> Result<Vec<String>> {
    let value_col = raw
        .column_index(&column.to_ascii_lowercase())
        .ok_or_else(|| invalid_input(format!("Batch column '{column}' not found in SDRF")))?;
    let sample_col = raw
        .column_index("source name")
        .ok_or_else(|| invalid_input("batch column detection requires `source name` in SDRF"))?;
    let mut sample_to_batch: HashMap<&str, &str> = HashMap::new();
    for row in 0..raw.row_count() {
        sample_to_batch.insert(raw.cell(row, sample_col), raw.cell(row, value_col));
    }
    sample_names
        .iter()
        .map(|&name| {
            sample_to_batch
                .get(name)
                .map(|value| (*value).to_owned())
                .ok_or_else(|| {
                    invalid_input(format!(
                        "batch SDRF column `{column}` has no value for matrix sample `{name}`"
                    ))
                })
        })
        .collect()
}

fn pearson_correlation(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < MIN_SAMPLE_CORRELATION_OVERLAP {
        return None;
    }
    let count = pairs.len() as f64;
    let (sum_left, sum_right) = pairs
        .iter()
        .fold((0.0, 0.0), |(left, right), (x, y)| (left + x, right + y));
    let mean_left = sum_left / count;
    let mean_right = sum_right / count;
    let (covariance, variance_left, variance_right) = pairs.iter().fold(
        (0.0, 0.0, 0.0),
        |(covariance, variance_left, variance_right), (x, y)| {
            let centered_left = x - mean_left;
            let centered_right = y - mean_right;
            (
                covariance + centered_left * centered_right,
                variance_left + centered_left * centered_left,
                variance_right + centered_right * centered_right,
            )
        },
    );
    let denominator = (variance_left * variance_right).sqrt();
    (denominator > 0.0)
        .then(|| (covariance / denominator).clamp(-1.0, 1.0))
        .filter(|correlation| correlation.is_finite())
}

impl ProteinMatrix {
    fn apply_irs(
        &mut self,
        sdrf: Option<&SdrfTable>,
        raw_sdrf: Option<&SdrfRawTable>,
        config: &IrsConfig,
    ) -> Result<()> {
        let Some(sdrf) = sdrf else {
            return Err(invalid_input("IRS normalization requires --sdrf option"));
        };
        let raw_sdrf = raw_sdrf.ok_or_else(|| {
            invalid_input("IRS normalization requires readable raw SDRF metadata")
        })?;
        let reference_samples = resolve_irs_reference_samples(sdrf, raw_sdrf, config)?;
        if reference_samples.is_empty() {
            return Err(invalid_input(
                "IRS normalization found no reference samples in the SDRF",
            ));
        }

        let (sample_to_plex, plex_source) =
            derive_sample_plexes(sdrf, Some(raw_sdrf), config.plex_column.as_deref())?;
        let mut plexes = sample_to_plex.values().cloned().collect::<Vec<_>>();
        plexes.sort();
        plexes.dedup();
        if plexes.is_empty() {
            return Err(invalid_input(
                "IRS normalization found no plex assignments in the SDRF",
            ));
        }
        if plexes.len() == 1 {
            let partitions = data_file_partitions(sdrf);
            if partitions > 1 {
                return Err(invalid_input(format!(
                    "IRS found one plex ({plex_source:?}) but the SDRF data files split the channels into {partitions} groups"
                )));
            }
            warn!("IRS found a single plex: scaling factors are 1, only reference removal applies");
        }

        let sample_by_name = self
            .samples
            .iter()
            .map(|(sample, name)| (name.to_owned(), sample))
            .collect::<HashMap<_, _>>();
        let mut refs_by_plex = HashMap::<String, Vec<SampleId>>::new();
        for reference in &reference_samples {
            let Some(sample) = sample_by_name.get(reference).copied() else {
                continue;
            };
            let Some(plex) = sample_to_plex.get(reference) else {
                continue;
            };
            refs_by_plex.entry(plex.clone()).or_default().push(sample);
        }
        for plex in &plexes {
            if !refs_by_plex.contains_key(plex) {
                return Err(invalid_input(format!(
                    "No reference samples found in plex `{plex}`"
                )));
            }
        }

        let factors = irs_scaling_factors(
            self.allowed_proteins.iter().copied(),
            &plexes,
            &refs_by_plex,
            &config.stat,
            |protein, sample| self.value(protein, sample),
        );
        if factors.is_empty() {
            return Err(invalid_input(
                "IRS normalization could not compute any finite scaling factors",
            ));
        }
        self.scale_by_irs(&sample_to_plex, &factors);

        if config.remove_reference {
            for reference in reference_samples {
                if let Some(sample) = sample_by_name.get(&reference).copied() {
                    self.excluded_samples.insert(sample);
                }
            }
        }
        Ok(())
    }

    fn scale_by_irs(
        &mut self,
        sample_to_plex: &HashMap<String, String>,
        factors: &HashMap<ProteinId, HashMap<String, f64>>,
    ) {
        let sample_to_plex = self
            .samples
            .iter()
            .filter_map(|(sample, name)| {
                sample_to_plex.get(name).map(|plex| (sample, plex.clone()))
            })
            .collect::<HashMap<_, _>>();
        match &mut self.values {
            ProteinValues::Cells(values) => {
                for (key, value) in values {
                    if let Some(factor) = sample_to_plex.get(&key.sample).and_then(|plex| {
                        factors
                            .get(&key.protein)
                            .and_then(|by_plex| by_plex.get(plex))
                    }) {
                        *value *= *factor;
                    }
                }
            }
            ProteinValues::Rows(rows) => {
                for (protein, values) in rows {
                    for (sample, value) in values {
                        if let Some(factor) = sample_to_plex.get(sample).and_then(|plex| {
                            factors.get(protein).and_then(|by_plex| by_plex.get(plex))
                        }) {
                            *value *= *factor;
                        }
                    }
                }
            }
        }
    }

    fn sample_columns(&self, drop_empty_samples: bool) -> Vec<(SampleId, &str)> {
        let value_samples = drop_empty_samples.then(|| self.values.sample_ids());
        let mut samples = self
            .samples
            .iter()
            .filter(|(sample, _)| {
                !self.excluded_samples.contains(sample)
                    && value_samples
                        .as_ref()
                        .is_none_or(|value_samples| value_samples.contains(sample))
            })
            .collect::<Vec<_>>();
        samples.sort_by(|left, right| left.1.cmp(right.1));
        samples
    }

    fn pairwise_sample_correlation(
        &self,
        left: SampleId,
        right: SampleId,
        values_are_log2: bool,
    ) -> (Option<f64>, usize) {
        let pairs = self
            .allowed_proteins
            .iter()
            .filter_map(|protein| {
                let left = self.value(*protein, left)?;
                let right = self.value(*protein, right)?;
                if values_are_log2 {
                    return (left.is_finite() && right.is_finite()).then_some((left, right));
                }
                (left.is_finite() && left > 0.0 && right.is_finite() && right > 0.0)
                    .then_some((left.log2(), right.log2()))
            })
            .collect::<Vec<_>>();
        (pearson_correlation(&pairs), pairs.len())
    }

    fn biological_samples_by_condition(
        &self,
        sdrf: &SdrfTable,
        drop_empty_samples: bool,
    ) -> Result<HashMap<String, Vec<(SampleId, String)>>> {
        let condition_by_sample = condition_by_sample(sdrf);
        let samples = self
            .sample_columns(drop_empty_samples)
            .into_iter()
            .map(|(sample, name)| {
                condition_by_sample
                    .get(name)
                    .map(|condition| (sample, name.to_owned(), condition.clone()))
                    .ok_or_else(|| {
                        invalid_input(format!(
                            "sample correlation filtering found no condition metadata for `{name}`"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut grouped = HashMap::<String, Vec<(SampleId, String)>>::new();
        for (sample, name, condition) in samples {
            if !is_reference_condition(&condition) {
                grouped.entry(condition).or_default().push((sample, name));
            }
        }
        if grouped.is_empty() {
            return Err(invalid_input(
                "sample correlation filtering found no biological samples with condition metadata",
            ));
        }
        Ok(grouped)
    }

    fn mean_sample_correlation(
        &self,
        sample: SampleId,
        name: &str,
        condition: &str,
        condition_samples: &[(SampleId, String)],
        values_are_log2: bool,
    ) -> Result<f64> {
        let mut correlations = Vec::with_capacity(condition_samples.len() - 1);
        for (peer, peer_name) in condition_samples {
            if sample == *peer {
                continue;
            }
            let (correlation, overlap) =
                self.pairwise_sample_correlation(sample, *peer, values_are_log2);
            let correlation = correlation.ok_or_else(|| {
                invalid_input(format!(
                    "sample correlation between `{name}` and `{peer_name}` in condition `{condition}` is undefined: {overlap} pairwise-complete usable proteins (minimum {MIN_SAMPLE_CORRELATION_OVERLAP})"
                ))
            })?;
            correlations.push(correlation);
        }
        Ok(correlations.iter().sum::<f64>() / correlations.len() as f64)
    }

    fn apply_sample_correlation_filter(
        &mut self,
        sdrf: Option<&SdrfTable>,
        threshold: f64,
        drop_empty_samples: bool,
        values_are_log2: bool,
    ) -> Result<()> {
        let sdrf = sdrf
            .ok_or_else(|| invalid_input("sample correlation filtering requires --sdrf option"))?;
        let samples_by_condition =
            self.biological_samples_by_condition(sdrf, drop_empty_samples)?;

        let mut excluded = Vec::new();
        for (condition, condition_samples) in &samples_by_condition {
            if condition_samples.len() < 2 {
                return Err(invalid_input(format!(
                    "sample correlation filtering requires at least two samples in condition `{condition}`"
                )));
            }
            for (sample, name) in condition_samples {
                let mean_correlation = self.mean_sample_correlation(
                    *sample,
                    name,
                    condition,
                    condition_samples,
                    values_are_log2,
                )?;
                info!(
                    sample = name,
                    condition,
                    mean_correlation,
                    peers = condition_samples.len() - 1,
                    threshold,
                    "sample correlation QC"
                );
                if mean_correlation < threshold {
                    excluded.push((*sample, name.clone(), mean_correlation));
                }
            }
        }
        for (sample, _, _) in &excluded {
            self.excluded_samples.insert(*sample);
        }
        info!(
            evaluated_samples = samples_by_condition.values().map(Vec::len).sum::<usize>(),
            excluded_samples = excluded.len(),
            threshold,
            "sample correlation filtering complete"
        );
        Ok(())
    }

    fn value(&self, protein: ProteinId, sample: SampleId) -> Option<f64> {
        match &self.values {
            ProteinValues::Cells(values) => values.get(&CellKey { protein, sample }).copied(),
            ProteinValues::Rows(rows) => rows.get(&protein).and_then(|values| {
                values
                    .iter()
                    .find_map(|(candidate, value)| (*candidate == sample).then_some(*value))
            }),
        }
    }

    /// Build the log2-transformed protein x sample matrix over `samples`
    /// (column order preserved) for differential expression. Each cell is
    /// `log2(value)` when the linear intensity is finite and strictly positive;
    /// zero, missing, and non-positive values become `NaN`, while infinities
    /// are rejected. Rows are emitted for the kept proteins in matrix output
    /// order (sorted by accession) so protein identity lines up with the
    /// returned names.
    fn log2_rows(&self, samples: &[SampleId]) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
        let mut proteins = self
            .proteins
            .iter()
            .filter(|(protein, _)| self.allowed_proteins.contains(protein))
            .collect::<Vec<_>>();
        proteins.sort_by(|left, right| left.1.cmp(right.1));

        let mut names = Vec::with_capacity(proteins.len());
        let mut rows = Vec::with_capacity(proteins.len());
        for (protein, accession) in proteins {
            names.push(accession.to_owned());
            let mut row = Vec::with_capacity(samples.len());
            for sample in samples {
                let value = match self.value(protein, *sample) {
                    Some(value) if value.is_infinite() => {
                        return Err(invalid_input(format!(
                            "differential-expression matrix contains an infinite intensity for protein `{accession}`"
                        )));
                    }
                    Some(value) if value.is_finite() && value > 0.0 => value.log2(),
                    _ => f64::NAN,
                };
                row.push(value);
            }
            rows.push(row);
        }
        Ok((names, rows))
    }

    /// Per-protein unique-canonical-peptide counts keyed by the protein NAME
    /// (the same accession string `log2_rows` emits). This resolves the
    /// `ProteinId`-keyed [`Self::peptide_counts`] map back to names so the DEqMS
    /// DE path can align counts to its `proteins` row order, exactly as Python's
    /// `_build_count_vector` reindexes the count Series onto the matrix index.
    fn peptide_counts_by_name(&self) -> HashMap<String, usize> {
        self.peptide_counts
            .iter()
            .filter_map(|(protein, count)| {
                self.proteins
                    .resolve(*protein)
                    .map(|name| (name.to_owned(), *count))
            })
            .collect()
    }

    fn set_value(&mut self, protein: ProteinId, sample: SampleId, value: f64) {
        match &mut self.values {
            ProteinValues::Cells(values) => {
                values.insert(CellKey { protein, sample }, value);
            }
            ProteinValues::Rows(rows) => {
                let values = rows.entry(protein).or_default();
                if let Some((_, current)) = values
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == sample)
                {
                    *current = value;
                } else {
                    values.push((sample, value));
                    values.sort_by_key(|(sample, _)| sample.get());
                }
            }
        }
    }

    /// ComBat batch correction over the protein x sample matrix, mirroring
    /// Python's `apply_batch_correction` (stages.py:1701): detect batches, run
    /// ComBat on the proteins with no missing cells, and write the corrected
    /// values back in place. Proteins with any missing cell are kept uncorrected
    /// (`_complete_batch_matrix`, stages.py:1666); fewer than two samples, fewer
    /// than two batches, any batch under two samples, or no complete protein each
    /// are rejected instead of producing a successful no-op. A cell counts as
    /// present only when finite, so directlfq zeros stay (matching `isna`'s
    /// 0-is-present) while NaN/missing rows drop out.
    ///
    /// Batch detection follows `_detect_batch_indices`: `sample_prefix` from the
    /// sample-name prefix, `column` from an explicit SDRF column (`source name ->
    /// column`, missing samples `"unknown"`). `run` has no run-level mapping in
    /// the protein-matrix flow, so it raises like Python's `run_info required`.
    /// Repeated `--batch-covariate` values are extracted from the SDRF (`extract_sdrf_covariates`)
    /// and fed to the covariate ComBat design to preserve their biological signal.
    fn apply_batch_correction(
        &mut self,
        config: &BatchCorrectionConfig,
        sdrf_path: Option<&Path>,
        drop_empty_samples: bool,
    ) -> Result<()> {
        let samples = self.sample_columns(drop_empty_samples);
        if samples.len() < 2 {
            return Err(invalid_input(
                "batch correction requires at least two matrix samples",
            ));
        }
        let sample_ids = samples
            .iter()
            .map(|(sample, _)| *sample)
            .collect::<Vec<_>>();
        let sample_names = samples.iter().map(|(_, name)| *name).collect::<Vec<_>>();

        use mokume_stats::batch::BatchDetectionMethod;

        let method = resolve_batch_method(&config.method)?;

        // `column` detection and covariate extraction re-read the raw SDRF columns
        // (Python re-reads via `load_sdrf`); `validate_postprocessing_subset`
        // already guarantees `--sdrf` is present when either is requested.
        let raw = if matches!(method, BatchDetectionMethod::ExplicitColumn)
            || config.covariates.is_some()
        {
            match sdrf_path {
                Some(path) => Some(SdrfRawTable::from_path(path)?),
                None => None,
            }
        } else {
            None
        };

        let original_batch_labels = match method {
            BatchDetectionMethod::ExplicitColumn => {
                let column = config.column.as_deref().unwrap_or_default();
                let raw = raw.as_ref().ok_or_else(|| {
                    invalid_input("batch column detection requires --sdrf option")
                })?;
                // Python's `_detect_explicit_batches(None)` raises when the column
                // is absent; mirror that hard failure rather than silently skipping.
                batch_column_values_for_samples(raw, &sample_names, column)?
            }
            BatchDetectionMethod::RunName
            | BatchDetectionMethod::Fraction
            | BatchDetectionMethod::TechReplicate => {
                return Err(invalid_input(
                    format!(
                        "batch-method '{}' requires run-level information not available in the protein matrix",
                        method.as_value()
                    ),
                ));
            }
            BatchDetectionMethod::SamplePrefix => sample_names
                .iter()
                .map(|sample| mokume_stats::batch::sample_prefix(sample).to_owned())
                .collect(),
        };
        let batch = factorize_batch_labels(&original_batch_labels);
        let batch = validate_batch_sizes(batch).ok_or_else(|| {
            invalid_input(
                "batch correction requires at least two batches with at least two samples each",
            )
        })?;

        // Split proteins into complete rows (no missing cell -> corrected) and the
        // rest (kept uncorrected). ComBat's empirical-Bayes priors are pooled over
        // the complete rows but are row-order independent, so any stable order of
        // the complete set reproduces Python's result; sort by accession to match
        // the matrix output order.
        let mut proteins = self
            .proteins
            .iter()
            .filter(|(protein, _)| self.allowed_proteins.contains(protein))
            .collect::<Vec<_>>();
        proteins.sort_by(|left, right| left.1.cmp(right.1));

        let mut complete = Vec::new();
        for (protein, _) in proteins {
            let row = sample_ids
                .iter()
                .map(|&sample| self.value(protein, sample))
                .collect::<Vec<_>>();
            if row.iter().all(|cell| cell.is_some_and(f64::is_finite)) {
                let values = row
                    .into_iter()
                    .map(|cell| cell.unwrap_or(0.0))
                    .collect::<Vec<_>>();
                complete.push((protein, values));
            }
        }
        if complete.is_empty() {
            return Err(invalid_input(
                "batch correction requires at least one protein observed in every sample",
            ));
        }

        // Biological covariates to preserve (independent of the batch method),
        // matching `_batch_covariates` at the combat call site.
        let covariates = match (&config.covariates, raw.as_ref()) {
            (Some(columns), Some(raw)) => extract_sdrf_covariates(raw, &sample_names, columns)?,
            _ => None,
        };

        let ref_batch = config
            .ref_batch
            .as_deref()
            .map(|requested| resolve_reference_batch(requested, &original_batch_labels, &batch))
            .transpose()?;
        if let Some(covariates) = covariates.as_deref() {
            validate_combat_design(&batch, covariates, ref_batch)?;
        }

        let data = complete
            .iter()
            .map(|(_, values)| values.clone())
            .collect::<Vec<_>>();
        let params = mokume_stats::batch::ComBatParams {
            par_prior: config.parametric,
            mean_only: config.mean_only,
            ref_batch,
        };
        let corrected = mokume_stats::batch::combat(&data, &batch, covariates.as_deref(), params);

        for ((protein, _), corrected_row) in complete.iter().zip(corrected.iter()) {
            for (&sample, &value) in sample_ids.iter().zip(corrected_row.iter()) {
                self.set_value(*protein, sample, value);
            }
        }
        Ok(())
    }

    fn apply_coverage_filter(
        &mut self,
        sdrf: Option<&SdrfTable>,
        threshold: f64,
        drop_empty_samples: bool,
    ) -> Result<()> {
        let Some(sdrf) = sdrf else {
            return Err(invalid_input("coverage filtering requires --sdrf option"));
        };
        let condition_by_sample = condition_by_sample(sdrf);
        let mut samples_by_condition = HashMap::<String, Vec<SampleId>>::new();
        for (sample, sample_name) in self.sample_columns(drop_empty_samples) {
            let Some(condition) = condition_by_sample.get(sample_name) else {
                continue;
            };
            if is_reference_condition(condition) {
                continue;
            }
            samples_by_condition
                .entry(condition.clone())
                .or_default()
                .push(sample);
        }
        if samples_by_condition.is_empty() {
            return Err(invalid_input(
                "coverage filtering found no non-reference samples with condition metadata",
            ));
        }

        self.allowed_proteins = coverage_filtered_proteins(
            self.allowed_proteins.iter().copied(),
            &samples_by_condition,
            threshold,
            |protein, sample| self.value(protein, sample),
        );
        Ok(())
    }

    fn apply_imputation(
        &mut self,
        config: &ImputationConfig,
        drop_empty_samples: bool,
    ) -> Result<()> {
        let (proteins, samples) = self.imputation_axes(drop_empty_samples);
        if proteins.iter().any(|protein| {
            samples
                .iter()
                .any(|sample| self.value(*protein, *sample).is_some_and(f64::is_infinite))
        }) {
            return Err(invalid_input(
                "imputation matrix contains an infinite intensity",
            ));
        }
        // Match the Python pipeline, which imputes in log2 space and converts
        // back to linear: positive finite values are log2-transformed, missing
        // and non-positive values stay missing, and infinities are rejected.
        let fills = imputed_values(config, &proteins, &samples, |protein, sample| {
            self.value(protein, sample)
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(f64::log2)
        })?;
        for (protein, sample, value) in fills {
            let intensity = matrix::checked_imputed_intensity(value)?;
            self.set_value(protein, sample, intensity);
        }
        Ok(())
    }

    fn imputation_axes(&self, drop_empty_samples: bool) -> (Vec<ProteinId>, Vec<SampleId>) {
        let mut proteins = self.allowed_proteins.iter().copied().collect::<Vec<_>>();
        proteins.sort_by_key(|protein| protein.get());
        let samples = self
            .sample_columns(drop_empty_samples)
            .into_iter()
            .map(|(sample, _)| sample)
            .collect::<Vec<_>>();
        (proteins, samples)
    }

    fn write_csv(
        &self,
        path: &Path,
        output_format: OutputFormat,
        drop_empty_samples: bool,
        missing_fill: Option<f64>,
    ) -> Result<()> {
        create_parent_dir(path)?;
        let file = File::create(path).map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = WriterBuilder::new().from_writer(file);
        let samples = self.sample_columns(drop_empty_samples);
        let sample_index = samples
            .iter()
            .enumerate()
            .map(|(index, (sample, _))| (*sample, index))
            .collect::<HashMap<_, _>>();
        // Per-column fill for cells with no observed (positive, finite) value.
        // Mirrors Python's `pivot_table(fill_value=0)` over the observed samples:
        // the additive methods (sum / intensity / peptide_count / spectral_count /
        // directlfq) write `0` for a missing cell, the average/ratio methods
        // (median / topn / abd / maxlfq / ratio) leave it empty (NaN). A sample
        // with no observations at all stays empty even for the additive methods,
        // because Python's pivot never densifies it (it is re-added as a NaN
        // column), not 0.
        let observed = self.observed_samples();
        let column_missing = samples
            .iter()
            .map(|(sample, _)| match missing_fill {
                Some(value) if observed.contains(sample) => format_float(value),
                _ => String::new(),
            })
            .collect::<Vec<_>>();
        let mut header = Vec::with_capacity(samples.len() + 1);
        header.push(output_format.protein_column().to_owned());
        header.extend(samples.iter().map(|(_, sample)| (*sample).to_owned()));
        writer
            .write_record(header)
            .map_err(|source| csv_error(path, source))?;

        let mut proteins = self
            .proteins
            .iter()
            .filter(|(protein, _)| self.allowed_proteins.contains(protein))
            .collect::<Vec<_>>();
        proteins.sort_by(|left, right| left.1.cmp(right.1));
        for (protein, accession) in proteins {
            if !self.allowed_proteins.contains(&protein) {
                continue;
            }
            let mut row_values = column_missing.clone();
            self.fill_row_values(protein, &sample_index, &mut row_values);
            let mut row = Vec::with_capacity(samples.len() + 1);
            row.push(accession.to_owned());
            row.extend(row_values);
            writer
                .write_record(row)
                .map_err(|source| csv_error(path, source))?;
        }
        writer.flush().map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }

    /// Samples carrying at least one observed (finite, strictly positive) value.
    /// These are the columns Python's `pivot_table(fill_value=0)` densifies; a
    /// sample absent from this set has no observations and stays empty even for the
    /// additive (0-fill) methods.
    fn observed_samples(&self) -> HashSet<SampleId> {
        let mut observed = HashSet::new();
        match &self.values {
            ProteinValues::Cells(values) => {
                for (key, value) in values {
                    if value.is_finite() && *value > 0.0 {
                        observed.insert(key.sample);
                    }
                }
            }
            ProteinValues::Rows(rows) => {
                for values in rows.values() {
                    for (sample, value) in values {
                        if value.is_finite() && *value > 0.0 {
                            observed.insert(*sample);
                        }
                    }
                }
            }
        }
        observed
    }

    fn fill_row_values(
        &self,
        protein: ProteinId,
        sample_index: &HashMap<SampleId, usize>,
        row_values: &mut [String],
    ) {
        // Only a finite, strictly positive intensity counts as observed; a missing
        // cell (absent, or the `0` directlfq/maxlfq emit for an unquantified sample)
        // keeps the caller's per-column `column_missing` default. This is what lets
        // the same dense directlfq solver feed `directlfq` (missing -> `0`) and
        // `maxlfq` (missing -> empty) without re-running the quantification.
        match &self.values {
            ProteinValues::Cells(values) => {
                for (sample, index) in sample_index {
                    let value = values.get(&CellKey {
                        protein,
                        sample: *sample,
                    });
                    if let Some(value) = value {
                        if value.is_finite() && *value > 0.0 {
                            row_values[*index] = format_float(*value);
                        }
                    }
                }
            }
            ProteinValues::Rows(rows) => {
                if let Some(values) = rows.get(&protein) {
                    for (sample, value) in values {
                        if let Some(index) = sample_index.get(sample) {
                            if value.is_finite() && *value > 0.0 {
                                row_values[*index] = format_float(*value);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// One observed peptide measurement feeding [`run_lfq_from_peptides`]. Unlike the
/// piBAQ path the protein comes straight from the input table, not a FASTA digest.
#[derive(Debug, Clone)]
pub struct LfqPeptideObservation {
    /// Protein accession the peptide belongs to.
    pub protein: String,
    /// Canonical peptide sequence, used as the DirectLFQ ion id.
    pub peptide: String,
    /// Sample identifier the measurement belongs to.
    pub sample: String,
    /// Intensity for this (protein, peptide, sample) measurement.
    pub intensity: f64,
}

/// One row of the DirectLFQ / MaxLFQ long-format result.
#[derive(Debug, Clone)]
pub struct LfqProteinIntensity {
    /// Protein accession.
    pub protein: String,
    /// Sample identifier.
    pub sample: String,
    /// Linear protein intensity for this sample, always `> 0` (the zero/missing
    /// entries DirectLFQ would emit are dropped to mirror Python's
    /// `DirectLFQQuantification._parse_wide_output`, which keeps only
    /// `Intensity > 0`).
    pub intensity: f64,
}

/// Roll a peptide-level table up to per-protein intensities inside an explicitly
/// sized Rayon worker pool with the DirectLFQ estimator (canonical peptides as
/// ions) -- the engine behind
/// `quantify peptides2protein --quant-method directlfq` and `--quant-method maxlfq`. mokume's
/// `MaxLFQQuantification` delegates to DirectLFQ when the package is available
/// (`min_nonan = 2`, its `min_peptides`); the `directlfq` method uses its own
/// `min_nonan`. `num_samples_quadratic` is DirectLFQ's global-stage knob (the
/// directlfq default is 50). `None` retains the configured global pool. Only
/// intensities `> 0` are returned, matching Python's `_parse_wide_output`.
pub fn run_lfq_from_peptides_with_threads(
    observations: &[LfqPeptideObservation],
    min_nonan: usize,
    num_samples_quadratic: usize,
    threads: Option<usize>,
) -> Result<Vec<LfqProteinIntensity>> {
    threading::install(threads, || {
        Ok(run_lfq_from_peptides(
            observations,
            min_nonan,
            num_samples_quadratic,
        ))
    })
}

/// Run the same DirectLFQ roll-up in the current Rayon worker pool. Call
/// [`run_lfq_from_peptides_with_threads`] when the pool size must be explicit.
pub fn run_lfq_from_peptides(
    observations: &[LfqPeptideObservation],
    min_nonan: usize,
    num_samples_quadratic: usize,
) -> Vec<LfqProteinIntensity> {
    let mut proteins = StringIdRegistry::<ProteinId>::new();
    let mut peptides = StringIdRegistry::<PeptideId>::new();
    let mut samples = StringIdRegistry::<SampleId>::new();
    // Keep the DirectLFQ matrix layout independent of input row order. Peptide
    // rows receive their separate lexical sequence rank below.
    let protein_names = observations
        .iter()
        .map(|observation| observation.protein.as_str())
        .collect::<BTreeSet<_>>();
    for protein in protein_names {
        if proteins.get_or_insert(protein).is_none() {
            return Vec::new();
        }
    }
    let sample_names = observations
        .iter()
        .map(|observation| observation.sample.as_str())
        .collect::<BTreeSet<_>>();
    for sample in sample_names {
        if samples.get_or_insert(sample).is_none() {
            return Vec::new();
        }
    }
    let mut ions = Vec::with_capacity(observations.len());
    for observation in observations {
        let (Some(protein), Some(ion), Some(sample)) = (
            proteins.get(&observation.protein),
            peptides.get_or_insert(&observation.peptide),
            samples.get(&observation.sample),
        ) else {
            continue;
        };
        ions.push(DirectLfqIon {
            protein,
            ion,
            // Filled in below once all peptide strings are known.
            ion_seq_rank: 0,
            sample,
            intensity: observation.intensity,
        });
    }
    // DirectLFQ orders ion rows by `(protein, sequence)` lexically and
    // `get_normfacts` breaks merge ties by row position, so rank each distinct
    // peptide id by its sequence string and stamp it onto every ion.
    let mut distinct_peptides = ions.iter().map(|ion| ion.ion).collect::<Vec<_>>();
    distinct_peptides.sort_unstable_by_key(|peptide| peptide.get());
    distinct_peptides.dedup();
    distinct_peptides.sort_by(|a, b| {
        peptides
            .resolve(*a)
            .cmp(&peptides.resolve(*b))
            .then_with(|| a.get().cmp(&b.get()))
    });
    let peptide_rank = distinct_peptides
        .into_iter()
        .enumerate()
        .map(|(rank, peptide)| (peptide, rank as u32))
        .collect::<HashMap<_, _>>();
    for ion in &mut ions {
        ion.ion_seq_rank = peptide_rank.get(&ion.ion).copied().unwrap_or(u32::MAX);
    }

    let mut result = Vec::new();
    for (protein, per_sample) in direct_lfq_aligned(&ions, min_nonan, num_samples_quadratic) {
        let Some(protein_name) = proteins.resolve(protein) else {
            continue;
        };
        for (sample, intensity) in per_sample {
            if intensity > 0.0 {
                if let Some(sample_name) = samples.resolve(sample) {
                    result.push(LfqProteinIntensity {
                        protein: protein_name.to_owned(),
                        sample: sample_name.to_owned(),
                        intensity,
                    });
                }
            }
        }
    }
    result
}

/// FASTA-digest parameters for [`run_pibaq_from_peptides`], mirroring the piBAQ
/// knobs the `peptides2protein` CLI exposes. The high-anchor threshold is fixed
/// at 3 by the existing Rust piBAQ core, so it is not configurable here.
#[derive(Debug, Clone)]
pub struct PibaqFromPeptidesParams {
    /// Protein database used to compute theoretical peptide counts.
    pub fasta: PathBuf,
    /// Minimum peptide length kept during the FASTA digest.
    pub min_aa: usize,
    /// Maximum peptide length kept during the FASTA digest.
    pub max_aa: usize,
    /// Minimum distinct shared peptides for two proteins to join a family.
    pub min_shared: usize,
    /// Anchor threshold; if no member reaches it, shared signal is split equally.
    pub min_anchors: usize,
    /// piBAQ "high anchor" threshold (Python `--high-anchor-threshold`). Only
    /// affects the `EvidenceLevel` annotation, not the piBAQ values.
    pub high_anchor_threshold: usize,
    /// Optional YAML overrides for protein-family grouping.
    pub families_yaml: Option<PathBuf>,
    /// Digestion enzyme name registered in the runtime pyOpenMS catalog.
    pub enzyme: String,
    /// Compute the TPA `MolecularWeight` + `TPA` columns (Python `--tpa`). When
    /// `true`, every returned row carries `Some` molecular-weight and TPA
    /// values; when `false` those fields are `None` and the piBAQ path is
    /// unchanged.
    pub tpa: bool,
}

/// Runtime pyOpenMS metadata attached to one theoretical FASTA digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PibaqDigestProvenance {
    pub pyopenms_version: String,
    pub enzyme: String,
    pub catalog_hash: String,
    pub min_aa: usize,
    pub max_aa: usize,
    pub missed_cleavages: usize,
}

/// The complete canonical protein -> theoretical peptide mapping produced by
/// the pyOpenMS installation that accompanies the wheel.
#[derive(Debug, Clone)]
pub struct PibaqDigest {
    pub accession_peptides: HashMap<String, HashSet<String>>,
    pub provenance: PibaqDigestProvenance,
}

/// One observed peptide measurement feeding [`run_pibaq_from_peptides`].
#[derive(Debug, Clone)]
pub struct PeptideObservation {
    /// Canonical peptide sequence (matched against the FASTA digest).
    pub peptide: String,
    /// Sample identifier the measurement belongs to.
    pub sample: String,
    /// Normalized intensity for this (peptide, sample) measurement.
    pub intensity: f64,
}

/// One row of the piBAQ long-format result, matching the columns the Python
/// `peptides_to_protein` writes (minus the optional TPA/ruler extras).
#[derive(Debug, Clone)]
pub struct PibaqProteinRow {
    /// Protein (family member) accession.
    pub protein: String,
    /// Sample identifier.
    pub sample: String,
    /// Allocated, shared-aware numerator intensity.
    pub norm_intensity: f64,
    /// piBAQ value (numerator divided by the owned theoretical peptide count).
    pub pibaq: f64,
    /// Representative family accession.
    pub family_id: String,
    /// Evidence label: `family_only`, `medium`, or `high`.
    pub evidence_level: &'static str,
    /// Number of members in the protein family.
    pub family_size: usize,
    /// TPA molecular weight (`Some` only when [`PibaqFromPeptidesParams::tpa`] is
    /// set); the Python `MolecularWeight` column.
    pub molecular_weight: Option<f64>,
    /// TPA value `NormIntensity / MolecularWeight` (`Some` only when
    /// [`PibaqFromPeptidesParams::tpa`] is set); the Python `TPA` column.
    pub tpa: Option<f64>,
}

/// Compute piBAQ from a peptide table using a complete theoretical-peptide map
/// produced by the wheel's runtime pyOpenMS catalog.
pub fn run_pibaq_from_peptides(
    observations: &[PeptideObservation],
    params: &PibaqFromPeptidesParams,
    digest: PibaqDigest,
) -> Result<Vec<PibaqProteinRow>> {
    let config = pibaq_only_config(params);
    let mut aggregation = PibaqAggregation::from_digest(&config, digest)?;
    if params.tpa {
        aggregation.mw_map = Some(load_fasta_mw(&params.fasta)?);
    }

    finalize_pibaq_observations(observations, aggregation, true)
}

/// Compute piBAQ from caller-provided theoretical mappings and protein families.
///
/// This is the in-memory counterpart of [`run_pibaq_from_peptides`]. It lets the
/// Python compatibility API retain its DataFrame/group-column contract while
/// sharing the same Rust allocation, denominator, evidence, and TPA core used by
/// the file-oriented command.
pub fn run_pibaq_from_mapping(
    observations: &[PeptideObservation],
    accession_peptides: HashMap<String, HashSet<String>>,
    peptide_accessions: HashMap<String, HashSet<String>>,
    families: Vec<(String, Vec<String>)>,
    min_anchors: usize,
    high_anchor_threshold: usize,
    mw_map: Option<HashMap<String, f64>>,
) -> Result<Vec<PibaqProteinRow>> {
    let aggregation = PibaqAggregation {
        accession_peptides,
        peptide_accessions,
        families: families
            .into_iter()
            .map(|(family_id, members)| ProteinFamily { family_id, members })
            .collect(),
        peptide_names: HashMap::new(),
        observations: HashMap::new(),
        min_anchors,
        high_anchor_threshold,
        mw_map,
        digest_max_aa: None,
        contaminant_twins: HashMap::new(),
    };
    finalize_pibaq_observations(observations, aggregation, false)
}

fn finalize_pibaq_observations(
    observations: &[PeptideObservation],
    mut aggregation: PibaqAggregation,
    positive_only: bool,
) -> Result<Vec<PibaqProteinRow>> {
    let mut samples = StringIdRegistry::<SampleId>::new();
    let mut peptides = StringIdRegistry::<PeptideId>::new();
    for observation in observations {
        if !observation.intensity.is_finite() || (positive_only && observation.intensity <= 0.0) {
            continue;
        }
        let sample = register_id(&mut samples, &observation.sample, "sample")?;
        let peptide = register_id(&mut peptides, &observation.peptide, "peptide")?;
        aggregation.push(sample, peptide, &observation.peptide, observation.intensity);
    }

    let mut proteins = StringIdRegistry::<ProteinId>::new();
    let detailed = aggregation.finalize_detailed(&mut proteins);

    let mut rows = Vec::with_capacity(detailed.len());
    for (key, detail) in detailed {
        let (Some(protein), Some(sample)) =
            (proteins.resolve(key.protein), samples.resolve(key.sample))
        else {
            continue;
        };
        rows.push(PibaqProteinRow {
            protein: protein.to_owned(),
            sample: sample.to_owned(),
            norm_intensity: detail.cell.norm_intensity,
            pibaq: detail.cell.pibaq,
            family_id: detail.family_id,
            evidence_level: detail.evidence_level.label(),
            family_size: detail.family_size,
            molecular_weight: detail.molecular_weight,
            tpa: detail.tpa,
        });
    }
    Ok(rows)
}

/// Build a `FeatureToProteinsConfig` that exercises only the piBAQ digest path.
/// The non-piBAQ fields use defaults; they are never consumed by
/// `PibaqAggregation`.
fn pibaq_only_config(params: &PibaqFromPeptidesParams) -> FeatureToProteinsConfig {
    FeatureToProteinsConfig {
        input: InputConfig {
            parquet: None,
            msstats: None,
            psm: None,
            sdrf: None,
            fasta: Some(params.fasta.clone()),
        },
        output: OutputConfig {
            protein_matrix: PathBuf::new(),
            export_peptides: None,
            export_ions: None,
            format: OutputFormat::default(),
        },
        filtering: FilterConfig {
            min_aa: params.min_aa,
            min_unique_peptides: 0,
            remove_contaminants: false,
        },
        normalization: NormalizationConfig::default(),
        quantification: QuantMethod::Pibaq,
        topn_peptides: 3,
        maxlfq: MaxLfqConfig::default(),
        pibaq: PibaqConfig {
            enzyme: params.enzyme.clone(),
            max_aa: params.max_aa,
            min_shared: params.min_shared,
            families_yaml: params.families_yaml.clone(),
            min_anchors: params.min_anchors,
            high_anchor_threshold: params.high_anchor_threshold,
            missed_cleavages: 0,
        },
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig::default(),
        coverage_threshold: None,
        sample_correlation_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: None,
        },
    }
}

pub fn run_features_to_proteins(config: &FeatureToProteinsConfig) -> Result<()> {
    if config.quantification == QuantMethod::Pibaq {
        validate_features_to_proteins(config)?;
        return Err(invalid_input(
            "piBAQ requires the Python wheel's runtime pyOpenMS FASTA digest",
        ));
    }
    let threads = configured_threads(config);
    threading::install(threads, || run_features_to_proteins_inner(config, None))
}

/// Run `features2proteins` with a complete runtime pyOpenMS digest for piBAQ.
pub fn run_features_to_proteins_with_pibaq_digest(
    config: &FeatureToProteinsConfig,
    digest: PibaqDigest,
) -> Result<()> {
    let threads = configured_threads(config);
    threading::install(threads, move || {
        run_features_to_proteins_inner(config, Some(digest))
    })
}

fn configured_threads(config: &FeatureToProteinsConfig) -> Option<usize> {
    config.runtime.threads.or_else(|| {
        (config.quantification == QuantMethod::DirectLfq)
            .then_some(config.directlfq.cores)
            .flatten()
    })
}

fn initialize_memory_plan(runtime: &RuntimeConfig) -> Result<MemoryPlan> {
    let memory = MemoryPlan::from_runtime(runtime)?;
    memory.check("startup")?;
    if let Some(limit_bytes) = memory.limit_bytes() {
        info!(
            limit_bytes,
            qpx_batch_size = memory.qpx_batch_size(),
            qpx_window = memory.qpx_window(),
            qpx_read_ahead = memory.channel_capacity(),
            "using soft runtime memory budget"
        );
    }
    Ok(memory)
}

fn run_features_to_proteins_inner(
    config: &FeatureToProteinsConfig,
    pibaq_digest: Option<PibaqDigest>,
) -> Result<()> {
    // Fold `--de-contrast-file` into the contrast list up front (mirroring
    // Python's CLI) so validation and the DE stage see one resolved list; the
    // owned, expanded config then shadows the borrowed one for the rest of the run.
    let expanded;
    let config = if config.differential_expression.contrasts_file.is_some() {
        expanded = expand_de_contrasts_file(config)?;
        &expanded
    } else {
        config
    };

    validate_de_ensemble_options(config)?;
    validate_features_to_proteins(config)?;
    validate_implemented_subset(config)?;
    let memory = initialize_memory_plan(&config.runtime)?;
    let sdrf = config
        .input
        .sdrf
        .as_ref()
        .map(SdrfTable::from_path)
        .transpose()?;
    let raw_sdrf = config
        .input
        .sdrf
        .as_ref()
        .map(SdrfRawTable::from_path)
        .transpose()?;
    if config.quantification == QuantMethod::SpectralCount {
        let matrix = spectral_count_matrix(config, sdrf.as_ref(), &memory)?;
        return finish_protein_matrix(config, sdrf.as_ref(), raw_sdrf.as_ref(), &memory, matrix);
    }
    // The protein pipeline has no custom contaminant patterns; an empty slice
    // makes the median pre-pass fall back to the default `is_contaminant` path.
    // The median pre-pass keeps shared peptides only for piBAQ (Python
    // `stages.py:325`: `keep_shared_peptides = method == "pibaq"`).
    let intensity_factors = collect_intensity_factors(
        config,
        sdrf.as_ref(),
        &[],
        config.quantification == QuantMethod::Pibaq,
        &memory,
        None,
    )?;
    memory.check("normalization pre-pass")?;
    let intensity_factors = (!intensity_factors.is_empty()).then_some(&intensity_factors);
    let dataset_normalization = dataset_sample_normalization_method(config)?;
    let mut state =
        FeatureToProteinState::new(config, sdrf.as_ref(), raw_sdrf.as_ref(), pibaq_digest)?;
    stream_input_features(&config.input, sdrf.as_ref(), &memory, |feature| {
        state.ingest(&feature, sdrf.as_ref(), config.filtering, intensity_factors)
    })?;
    memory.check("feature aggregation")?;

    info!(
        accepted_features = state.accepted_features,
        accepted_measurements = state.accepted_measurements,
        proteins = state.proteins.len(),
        samples = state.samples.len(),
        quant_method = %config.quantification,
        output = %config.output.protein_matrix.display(),
        "features2proteins aggregation finished"
    );

    let min_unique_peptides = if config.quantification == QuantMethod::Pibaq {
        0
    } else {
        config.filtering.min_unique_peptides
    };
    // The protein pipeline never log2-transforms the peptide intermediates;
    // only the standalone `features2peptides` honors `--log2`.
    state.write_intermediate_exports(
        config,
        min_unique_peptides,
        PeptideExportOptions::default(),
        dataset_normalization,
    )?;
    let matrix = state.into_matrix(min_unique_peptides, dataset_normalization);
    memory.check("protein matrix materialization")?;
    finish_protein_matrix(config, sdrf.as_ref(), raw_sdrf.as_ref(), &memory, matrix)
}

fn finish_protein_matrix(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    raw_sdrf: Option<&SdrfRawTable>,
    memory: &MemoryPlan,
    mut matrix: ProteinMatrix,
) -> Result<()> {
    let drop_empty_samples = config.quantification == QuantMethod::Ratio;
    apply_protein_postprocessing(config, sdrf, raw_sdrf, drop_empty_samples, &mut matrix)?;
    memory.check("matrix post-processing")?;
    matrix.write_csv(
        &config.output.protein_matrix,
        config.output.format,
        drop_empty_samples,
        missing_protein_value(config.quantification),
    )?;
    if config.differential_expression.enabled {
        let mut differential_expression = config.differential_expression.clone();
        differential_expression.method = resolve_de_method(config);
        de::run_differential_expression(
            &matrix,
            sdrf,
            &differential_expression,
            drop_empty_samples,
        )?;
    }
    Ok(())
}

fn apply_protein_postprocessing(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    raw_sdrf: Option<&SdrfRawTable>,
    drop_empty_samples: bool,
    matrix: &mut ProteinMatrix,
) -> Result<()> {
    if config.irs.enabled {
        matrix.apply_irs(sdrf, raw_sdrf, &config.irs)?;
    }
    if let Some(threshold) = config.sample_correlation_threshold {
        let values_are_log2 =
            matches!(config.quantification, QuantMethod::Abd | QuantMethod::Ratio);
        matrix.apply_sample_correlation_filter(
            sdrf,
            threshold,
            drop_empty_samples,
            values_are_log2,
        )?;
    }
    if let Some(threshold) = config.coverage_threshold {
        matrix.apply_coverage_filter(sdrf, threshold, drop_empty_samples)?;
    }
    if config.imputation.enabled {
        matrix.apply_imputation(&config.imputation, drop_empty_samples)?;
    }
    if config.batch.enabled {
        matrix.apply_batch_correction(
            &config.batch,
            config.input.sdrf.as_deref(),
            drop_empty_samples,
        )?;
    }
    Ok(())
}

fn missing_protein_value(method: QuantMethod) -> Option<f64> {
    // Missing protein x sample cells follow Python's per-method convention: the
    // additive methods write `0`, the average/ratio methods leave the cell empty.
    match method {
        QuantMethod::Sum
        | QuantMethod::Intensity
        | QuantMethod::PeptideCount
        | QuantMethod::SpectralCount
        | QuantMethod::DirectLfq => Some(0.0),
        QuantMethod::Median
        | QuantMethod::TopN
        | QuantMethod::Abd
        | QuantMethod::MaxLfq
        | QuantMethod::Ratio
        | QuantMethod::Pibaq => None,
    }
}

fn spectral_count_matrix(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
) -> Result<ProteinMatrix> {
    let sdrf = sdrf.ok_or_else(|| invalid_input("spectral_count requires --sdrf option"))?;
    let counted = spectral_count::count(config, sdrf, memory)?;
    let (proteins, samples) = spectral_registries(&counted.cells)?;

    let mut values = HashMap::new();
    let mut peptide_sets = HashMap::<ProteinId, HashSet<String>>::new();
    for cell in counted.cells {
        let protein_id = proteins
            .get(&cell.protein_group)
            .ok_or_else(|| invalid_input("spectral-count protein registry is inconsistent"))?;
        let sample_id = samples
            .get(&cell.sample)
            .ok_or_else(|| invalid_input("spectral-count sample registry is inconsistent"))?;
        values.insert(
            CellKey {
                protein: protein_id,
                sample: sample_id,
            },
            cell.psms as f64,
        );
        peptide_sets
            .entry(protein_id)
            .or_default()
            .extend(cell.sequences);
    }
    let allowed_proteins = values.keys().map(|cell| cell.protein).collect();
    let peptide_counts = peptide_sets
        .into_iter()
        .map(|(protein, sequences)| (protein, sequences.len()))
        .collect();
    info!(
        target_psms = counted.target_psms,
        unique_psms = counted.unique_psms,
        proteins = proteins.len(),
        samples = samples.len(),
        "PSM spectral counting finished"
    );
    Ok(ProteinMatrix {
        proteins,
        samples,
        allowed_proteins,
        excluded_samples: HashSet::new(),
        peptide_counts,
        values: ProteinValues::Cells(values),
    })
}

fn spectral_registries(
    cells: &[spectral_count::SpectralCountCell],
) -> Result<(StringIdRegistry<ProteinId>, StringIdRegistry<SampleId>)> {
    let protein_names = cells
        .iter()
        .map(|cell| cell.protein_group.as_str())
        .collect::<BTreeSet<_>>();
    let sample_names = cells
        .iter()
        .map(|cell| cell.sample.as_str())
        .collect::<BTreeSet<_>>();
    let mut proteins = StringIdRegistry::new();
    let mut samples = StringIdRegistry::new();
    for protein in protein_names {
        register_id(&mut proteins, protein, "protein")?;
    }
    for sample in sample_names {
        register_id(&mut samples, sample, "sample")?;
    }
    Ok((proteins, samples))
}

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
fn compile_exclude_sequence_patterns(patterns: &[String]) -> Result<Vec<Regex>> {
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

fn active_filter_pipeline(config: &FeatureToPeptidesConfig) -> Option<&PreprocessingFilterConfig> {
    config
        .filter_pipeline
        .as_ref()
        .filter(|pipeline| pipeline.enabled)
}

fn validate_features_to_peptides_config(config: &FeatureToPeptidesConfig) -> Result<()> {
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
    if let Some(sdrf) = &config.input.sdrf {
        if !sdrf.exists() {
            return Err(MokumeError::MissingInput { path: sdrf.clone() });
        }
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
    let sdrf = proteins_config
        .input
        .sdrf
        .as_ref()
        .map(SdrfTable::from_path)
        .transpose()?;
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
        )?
    };
    // The IRS pre-pass reuses the median pre-pass's contaminant settings.
    if let Some(irs) = &config.irs {
        let irs_min_intensity =
            filter_pipeline.map_or(0.0, |pipeline| pipeline.intensity.min_intensity);
        let irs_scale_by_techrep = collect_irs_scale(
            parquet,
            irs,
            sdrf.as_ref(),
            proteins_config.filtering.remove_contaminants,
            median_contaminant_patterns,
            irs_min_intensity,
            named_score,
        )?;
        if irs_scale_by_techrep.is_empty() {
            return Err(invalid_input(format!(
                "features2peptides IRS channel `{}` produced no scaling factors",
                irs.channel
            )));
        }
        intensity_factors.irs_scale_by_techrep = irs_scale_by_techrep;
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
        if let Some(peptides) = &mut state.export_rows.peptides {
            peptides.razor_handling =
                parse_razor_handling(&pipeline.protein.razor_peptide_handling);
        }
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
    let reader = QpxParquetReader::open(parquet, DEFAULT_QPX_BATCH_SIZE)?;
    stream_qpx_features_maybe_score(reader, named_score, |feature| {
        state.ingest(
            &feature,
            sdrf.as_ref(),
            proteins_config.filtering,
            intensity_factors,
        )
    })?;

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
        min_unique,
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
fn collect_fdr_filter_state(
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
    let reader = QpxParquetReader::open(parquet, DEFAULT_QPX_BATCH_SIZE)?;
    stream_qpx_features_maybe_score(reader, named_score, |feature| {
        if peptide_threshold.is_some()
            && feature
                .peptide_qvalue
                .is_some_and(|qvalue| qvalue.is_finite())
        {
            peptide_values += 1;
        }
        if protein_threshold.is_some() {
            if let (Some(qvalue), Some(protein)) = (
                feature.pg_global_qvalue.filter(|qvalue| qvalue.is_finite()),
                protein_group_name(&feature.protein_accessions),
            ) {
                protein_values += 1;
                protein_min_qvalue
                    .entry(protein)
                    .and_modify(|known| *known = known.min(qvalue))
                    .or_insert(qvalue);
            }
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
    Ok(Some(
        protein_min_qvalue
            .into_iter()
            .filter_map(|(protein, qvalue)| (qvalue <= threshold).then_some(protein))
            .collect(),
    ))
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
            export_ions: None,
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
        coverage_threshold: None,
        sample_correlation_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: None,
        },
    }
}

/// Streams QPX features through `consume`, overlapping the parallelizable
/// decode + flatten work with the serial consumer.
///
/// A background thread reads raw Arrow batches serially (a single Parquet
/// reader cannot be shared) while the calling thread flattens each window with
/// the invocation-scoped Rayon pool. Batches and rows keep reader order, so the
/// floating-point accumulation order is identical to a fully serial pass and
/// the protein matrix stays cell-for-cell identical. The consumer mutates
/// shared state and therefore stays serial on the calling thread.
fn stream_qpx_features<F>(mut reader: QpxParquetReader, mut consume: F) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    stream_qpx_features_with_optional_score(
        &mut reader,
        &MemoryPlan::unlimited(),
        None,
        &mut consume,
    )
}

fn stream_qpx_features_with_score<F>(
    mut reader: QpxParquetReader,
    score: &NamedScoreFilterConfig,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    let mut seen = 0_usize;
    let mut direction = None;
    stream_qpx_features_with_optional_score(
        &mut reader,
        &MemoryPlan::unlimited(),
        Some(score.name.as_str()),
        &mut |feature| {
            let Some(value) = feature.selected_score else {
                return Ok(());
            };
            seen += 1;
            if let Some(expected) = direction {
                if expected != value.higher_better {
                    return Err(invalid_input(format!(
                        "QPX score `{}` has inconsistent higher_better values",
                        score.name
                    )));
                }
            } else {
                direction = Some(value.higher_better);
            }
            let passes = if value.higher_better {
                value.value >= score.threshold
            } else {
                value.value <= score.threshold
            };
            if passes {
                consume(feature)?;
            }
            Ok(())
        },
    )?;
    if seen == 0 {
        return Err(invalid_input(format!(
            "--filter-score requires QPX `additional_scores` entry `{}`",
            score.name
        )));
    }
    Ok(())
}

fn stream_qpx_features_maybe_score<F>(
    reader: QpxParquetReader,
    score: Option<&NamedScoreFilterConfig>,
    consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    match score {
        Some(score) => stream_qpx_features_with_score(reader, score, consume),
        None => stream_qpx_features(reader, consume),
    }
}

fn stream_qpx_features_with_plan<F>(
    reader: &mut QpxParquetReader,
    memory: &MemoryPlan,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    stream_qpx_features_with_optional_score(reader, memory, None, consume)
}

fn stream_qpx_features_with_optional_score<F>(
    reader: &mut QpxParquetReader,
    memory: &MemoryPlan,
    score_name: Option<&str>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    if memory.channel_capacity() == 0 {
        stream_qpx_features_synchronously(reader, memory, score_name, consume)
    } else {
        stream_qpx_features_buffered(reader, memory, score_name, consume)
    }
}

fn stream_qpx_features_synchronously<F>(
    reader: &mut QpxParquetReader,
    memory: &MemoryPlan,
    score_name: Option<&str>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    while let Some(batch) = reader.next_raw_batch() {
        let batch = batch?;
        let features = match score_name {
            Some(name) => mokume_io::flatten_qpx_batch_with_score(&batch, name)?,
            None => mokume_io::flatten_qpx_batch(&batch)?,
        };
        for feature in features {
            consume(feature)?;
        }
        memory.check("QPX streaming")?;
    }
    Ok(())
}

fn stream_qpx_features_buffered<F>(
    reader: &mut QpxParquetReader,
    memory: &MemoryPlan,
    score_name: Option<&str>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    let window_size = memory.qpx_window();
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<mokume_io::RecordBatch>>>(
        memory.channel_capacity(),
    );

    std::thread::scope(|scope| -> Result<()> {
        scope.spawn(move || loop {
            let mut window: Vec<mokume_io::RecordBatch> = Vec::with_capacity(window_size);
            for _ in 0..window_size {
                match reader.next_raw_batch() {
                    Some(Ok(batch)) => window.push(batch),
                    Some(Err(error)) => {
                        let _ = tx.send(Err(error));
                        return;
                    }
                    None => break,
                }
            }
            if window.is_empty() {
                return;
            }
            if tx.send(Ok(window)).is_err() {
                return;
            }
        });

        let result = consume_qpx_windows(&rx, memory, score_name, consume);
        // A consumer or RSS-budget error must drop the receiver before the
        // scoped reader thread is joined. This wakes a sender blocked by the
        // bounded channel instead of deadlocking on scope exit.
        drop(rx);
        result
    })
}

fn consume_qpx_windows<F>(
    receiver: &std::sync::mpsc::Receiver<Result<Vec<mokume_io::RecordBatch>>>,
    memory: &MemoryPlan,
    score_name: Option<&str>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    for message in receiver {
        let flattened: Result<Vec<Vec<QpxFeatureRecord>>> = message?
            .into_par_iter()
            .map(|batch| match score_name {
                Some(name) => mokume_io::flatten_qpx_batch_with_score(&batch, name),
                None => mokume_io::flatten_qpx_batch(&batch),
            })
            .collect();
        for features in flattened? {
            for feature in features {
                consume(feature)?;
            }
            memory.check("QPX streaming")?;
        }
    }
    Ok(())
}

fn stream_input_features<F>(
    input: &InputConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    match (&input.parquet, &input.msstats) {
        (Some(parquet), None) => {
            let mut reader = QpxParquetReader::open(parquet, memory.qpx_batch_size())?;
            stream_qpx_features_with_plan(&mut reader, memory, &mut consume)
        }
        (None, Some(msstats)) => {
            let sdrf = sdrf.ok_or_else(|| invalid_input("MSstats input requires --sdrf option"))?;
            for (index, feature) in MsstatsReader::open(msstats, sdrf)?.enumerate() {
                consume(feature?)?;
                if index % 4096 == 0 {
                    memory.check("MSstats streaming")?;
                }
            }
            Ok(())
        }
        _ => Err(invalid_input(
            "provide exactly one feature input: --parquet or --msstats",
        )),
    }
}

fn required_parquet(input: &InputConfig) -> Result<&Path> {
    input
        .parquet
        .as_deref()
        .ok_or_else(|| invalid_input("this command requires --parquet input"))
}

/// Quant methods whose aggregation collapses to `(protein, sample)` peptide
/// cells normalized by [`apply_dataset_norm_to_peptide_cells`]. These share the
/// dataset-level normalization with the `--export-peptides` path, so the
/// exported peptides match the protein matrix. piBAQ / LFQ / Ratio normalize
/// through aggregation-specific paths not connected to the peptide export.
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
fn dataset_sample_normalization_method(
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

fn validate_features_to_proteins(config: &FeatureToProteinsConfig) -> Result<()> {
    validate_feature_input(config)?;
    if let Some(sdrf) = &config.input.sdrf {
        if !sdrf.exists() {
            return Err(MokumeError::MissingInput { path: sdrf.clone() });
        }
    }
    if config.directlfq.cores.is_some() && config.quantification != QuantMethod::DirectLfq {
        return Err(invalid_input(
            "DirectLfqConfig.cores only applies to --quant-method directlfq; use RuntimeConfig.threads for other methods",
        ));
    }
    if config.directlfq.min_nonan != 1 && config.quantification != QuantMethod::DirectLfq {
        return Err(invalid_input(
            "--directlfq-min-nonan only applies to --quant-method directlfq",
        ));
    }
    if config.directlfq.num_samples_quadratic != 50
        && !matches!(
            config.quantification,
            QuantMethod::DirectLfq | QuantMethod::MaxLfq
        )
    {
        return Err(invalid_input(
            "--directlfq-num-samples-quadratic only applies to DirectLFQ/MaxLFQ",
        ));
    }
    if let Some(fasta) = &config.input.fasta {
        if !fasta.exists() {
            return Err(MokumeError::MissingInput {
                path: fasta.clone(),
            });
        }
    }
    if let Some(path) = &config.normalization.normalization_proteins {
        if !path.exists() {
            return Err(MokumeError::MissingInput { path: path.clone() });
        }
        if !matches!(
            config
                .normalization
                .sample_method
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "globalmedian" | "conditionmedian"
        ) {
            return Err(invalid_input(
                "--normalization-proteins requires globalmedian or conditionmedian sample normalization",
            ));
        }
    }
    if config.quantification == QuantMethod::Pibaq && config.input.fasta.is_none() {
        return Err(invalid_input(
            "piBAQ quantification requires --fasta option",
        ));
    }
    if config.quantification != QuantMethod::Pibaq
        && (config.input.fasta.is_some()
            || !config.pibaq.enzyme.eq_ignore_ascii_case("Trypsin")
            || config.pibaq.max_aa != 30
            || config.pibaq.min_shared != 2
            || config.pibaq.families_yaml.is_some()
            || config.pibaq.min_anchors != 1
            || config.pibaq.high_anchor_threshold != 3
            || config.pibaq.missed_cleavages != 0)
    {
        return Err(invalid_input(
            "piBAQ FASTA/digestion options require --quant-method pibaq",
        ));
    }
    if config.quantification == QuantMethod::Ratio && config.input.sdrf.is_none() {
        return Err(invalid_input("Ratio quantification requires --sdrf option"));
    }
    if !config.ratio.fraction_merge.eq_ignore_ascii_case("mean")
        && config.quantification != QuantMethod::Ratio
    {
        return Err(invalid_input(
            "--ratio-fraction-merge only applies to --quant-method ratio",
        ));
    }
    if matches!(
        config.quantification,
        QuantMethod::DirectLfq
            | QuantMethod::Ratio
            | QuantMethod::PeptideCount
            | QuantMethod::SpectralCount
    ) && (!config.normalization.run_method.eq_ignore_ascii_case("none")
        || !config
            .normalization
            .sample_method
            .eq_ignore_ascii_case("none")
        || config.normalization.normalization_proteins.is_some())
    {
        let reason = if matches!(
            config.quantification,
            QuantMethod::PeptideCount | QuantMethod::SpectralCount
        ) {
            "does not use intensity normalization"
        } else {
            "manages normalization internally"
        };
        return Err(invalid_input(format!(
            "{} {reason}; use --run-normalization none and --sample-normalization \
             none, and do not pass --normalization-proteins",
            config.quantification
        )));
    }
    if config.batch.enabled
        && config.batch.method.eq_ignore_ascii_case("column")
        && config.batch.column.is_none()
    {
        return Err(invalid_input(
            "Batch correction with method 'column' requires --batch-column option",
        ));
    }
    if config.batch.enabled
        && !matches!(
            config.batch.method.trim().to_ascii_lowercase().as_str(),
            "sample_prefix" | "column"
        )
    {
        return Err(invalid_input(
            "batch-method must be `sample_prefix` or `column` for protein matrices",
        ));
    }
    if !config.batch.enabled
        && (!config.batch.method.eq_ignore_ascii_case("sample_prefix")
            || config.batch.column.is_some()
            || config.batch.covariates.is_some()
            || !config.batch.parametric
            || config.batch.mean_only
            || config.batch.ref_batch.is_some())
    {
        return Err(invalid_input("batch options require --batch-correction"));
    }
    if config.batch.enabled
        && !config.batch.method.eq_ignore_ascii_case("column")
        && config.batch.column.is_some()
    {
        return Err(invalid_input(
            "--batch-column requires --batch-method column",
        ));
    }
    if config.batch.enabled
        && config.input.sdrf.is_none()
        && (config.batch.column.is_some() || config.batch.covariates.is_some())
    {
        return Err(invalid_input(
            "Batch correction with --batch-column or --batch-covariate requires --sdrf option",
        ));
    }
    Ok(())
}

fn validate_feature_input(config: &FeatureToProteinsConfig) -> Result<()> {
    match (
        &config.input.parquet,
        &config.input.msstats,
        &config.input.psm,
    ) {
        (Some(parquet), None, None) => validate_feature_qpx_input(parquet, config.quantification),
        (None, Some(msstats), None) => validate_msstats_input(msstats, config),
        (Some(parquet), None, Some(psm)) => validate_spectral_count_inputs(parquet, psm, config),
        (None, None, Some(_)) => Err(invalid_input(
            "spectral_count requires matching QPX inputs via --psm and --parquet",
        )),
        _ => Err(invalid_input(
            "provide --parquet, --msstats, or matching --psm and --parquet inputs",
        )),
    }
}

fn validate_feature_qpx_input(parquet: &Path, method: QuantMethod) -> Result<()> {
    require_existing_input(parquet)?;
    if method == QuantMethod::SpectralCount {
        return Err(invalid_input(
            "spectral_count requires matching QPX inputs via --psm and --parquet",
        ));
    }
    Ok(())
}

fn validate_msstats_input(path: &Path, config: &FeatureToProteinsConfig) -> Result<()> {
    require_existing_input(path)?;
    if config.input.sdrf.is_none() {
        return Err(invalid_input("MSstats input requires --sdrf option"));
    }
    match config.quantification {
        QuantMethod::Ratio => Err(invalid_input(
            "Ratio quantification requires PSM-level QPX input; MSstats feature tables do not contain PSM evidence",
        )),
        QuantMethod::SpectralCount => Err(invalid_input(
            "spectral_count requires matching QPX inputs via --psm and --parquet",
        )),
        _ => Ok(()),
    }
}

fn validate_spectral_count_inputs(
    parquet: &Path,
    psm: &Path,
    config: &FeatureToProteinsConfig,
) -> Result<()> {
    require_existing_input(parquet)?;
    require_existing_input(psm)?;
    if config.quantification != QuantMethod::SpectralCount {
        return Err(invalid_input(
            "--psm with --parquet only applies to spectral_count quantification",
        ));
    }
    if config.input.sdrf.is_none() {
        return Err(invalid_input("spectral_count requires --sdrf option"));
    }
    Ok(())
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

fn validate_implemented_subset(config: &FeatureToProteinsConfig) -> Result<()> {
    if config.runtime.threads == Some(0) || config.directlfq.cores == Some(0) {
        return Err(invalid_input("thread counts must be greater than zero"));
    }
    if config.runtime.threads.is_some() && config.directlfq.cores.is_some() {
        return Err(invalid_input(
            "choose either runtime threads or directlfq cores, not both",
        ));
    }
    if config.output.export_peptides.is_some()
        && matches!(
            config.quantification,
            QuantMethod::DirectLfq | QuantMethod::Ratio | QuantMethod::SpectralCount
        )
    {
        return Err(invalid_input(format!(
            "export-peptides is not supported by {} quantification",
            config.quantification
        )));
    }
    // Dataset-level sample normalization is carried into the exported peptides
    // only for the cell-based linear methods (their aggregation and export share
    // one normalization helper); other aggregations would emit peptides whose
    // normalization does not match the protein matrix.
    if config.output.export_peptides.is_some()
        && dataset_sample_normalization_method(config)?.is_some()
        && !is_cell_based_linear_quant(config.quantification)
    {
        return unsupported("dataset-normalization-export-peptides");
    }
    if config.output.export_ions.is_some() && config.quantification != QuantMethod::DirectLfq {
        return unsupported("export-ions");
    }
    if let Some(alignment) = &config.maxlfq.ion_alignment {
        if !alignment.eq_ignore_ascii_case("none") {
            return unsupported("ion-alignment");
        }
    }
    match config.quantification {
        QuantMethod::Sum
        | QuantMethod::Median
        | QuantMethod::TopN
        | QuantMethod::MaxLfq
        | QuantMethod::DirectLfq
        | QuantMethod::Abd
        | QuantMethod::Intensity
        | QuantMethod::PeptideCount
        | QuantMethod::SpectralCount
        | QuantMethod::Pibaq
        | QuantMethod::Ratio => {}
    }

    if !matches!(
        config.quantification,
        QuantMethod::DirectLfq
            | QuantMethod::Ratio
            | QuantMethod::PeptideCount
            | QuantMethod::SpectralCount
    ) {
        if !supports_run_normalization(&config.normalization.run_method) {
            return unsupported("run-normalization-method");
        }
        if !supports_sample_normalization(&config.normalization.sample_method) {
            return unsupported("sample-normalization-method");
        }
        if matches!(
            config.quantification,
            QuantMethod::Pibaq | QuantMethod::MaxLfq
        ) && dataset_sample_normalization_method(config)?
            .is_some_and(|method| method != SampleNormalizationMethod::Quantile)
        {
            return Err(invalid_input(format!(
                "{} supports quantile as its only dataset-level sample normalization",
                config.quantification
            )));
        }
    }

    validate_postprocessing_subset(config)
}

fn validate_postprocessing_subset(config: &FeatureToProteinsConfig) -> Result<()> {
    // ComBat (parametric / non-parametric / mean_only / ref_batch) is wired into
    // the pipeline (`apply_batch_correction`) for `sample_prefix` and explicit
    // `column` detection plus SDRF covariate extraction. `run` detection has no
    // run-level mapping in the protein-matrix flow and errors at runtime, the
    // same way Python's `_detect_batch_indices` raises `run_info required`.
    if matches!(
        config.quantification,
        QuantMethod::PeptideCount | QuantMethod::SpectralCount
    ) && config.irs.enabled
    {
        return Err(invalid_input(format!(
            "{} quantification cannot apply IRS",
            config.quantification
        )));
    }
    if config
        .irs
        .reference_samples
        .as_ref()
        .is_some_and(Vec::is_empty)
    {
        return Err(invalid_input("IRS reference sample list must not be empty"));
    }
    if config.irs.sdrf_column.is_some() != config.irs.sdrf_values.is_some() {
        return Err(invalid_input(
            "--irs-sdrf-column and --irs-sdrf-value must be provided together",
        ));
    }
    let custom_regex = config.irs.reference_regex != DEFAULT_REFERENCE_REGEX;
    let selector_count = usize::from(config.irs.reference_samples.is_some())
        + usize::from(config.irs.sdrf_column.is_some())
        + usize::from(custom_regex);
    if selector_count > 1 {
        return Err(invalid_input(
            "choose one reference selector: samples, SDRF column+values, or regex",
        ));
    }
    if config.quantification == QuantMethod::Ratio {
        if config.irs.enabled {
            return Err(invalid_input("Ratio quantification cannot also apply IRS"));
        }
        if config.irs.sdrf_column.is_some()
            || !config.irs.stat.eq_ignore_ascii_case("median")
            || config.irs.remove_reference
        {
            return Err(invalid_input(
                "Ratio accepts IRS reference samples/regex only; IRS normalization options require --irs",
            ));
        }
        if config.input.sdrf.is_none() {
            return Err(invalid_input(
                "Ratio reference detection requires --sdrf option",
            ));
        }
    } else {
        let uses_irs_parameters = selector_count > 0
            || !config.irs.stat.eq_ignore_ascii_case("median")
            || config.irs.remove_reference;
        if uses_irs_parameters && !config.irs.enabled {
            return Err(invalid_input("IRS options require --irs"));
        }
        if config.irs.enabled {
            if config.input.sdrf.is_none() {
                return Err(invalid_input("IRS options require --sdrf option"));
            }
            if !matches!(
                config.irs.stat.trim().to_ascii_lowercase().as_str(),
                "median" | "mean"
            ) {
                return unsupported("irs-stat");
            }
        }
    }
    if let Some(threshold) = config.coverage_threshold {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(invalid_input("coverage-threshold must be between 0 and 1"));
        }
        if config.input.sdrf.is_none() {
            return Err(invalid_input("coverage-threshold requires --sdrf option"));
        }
    }
    if let Some(threshold) = config.sample_correlation_threshold {
        if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
            return Err(invalid_input(
                "min-sample-correlation must be between -1 and 1",
            ));
        }
        if config.input.sdrf.is_none() {
            return Err(invalid_input(
                "min-sample-correlation requires --sdrf option",
            ));
        }
    }
    if !matches!(
        config
            .ratio
            .fraction_merge
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "mean" | "max"
    ) {
        return unsupported("ratio-fraction-merge");
    }
    validate_imputation_config(&config.imputation)?;
    validate_de_subset(config)?;
    Ok(())
}

pub(crate) fn validate_imputation_config(config: &ImputationConfig) -> Result<()> {
    let method = config.method.trim().to_ascii_lowercase();
    if !config.enabled {
        if method != "none"
            || (config.quantile - 0.01).abs() > f64::EPSILON
            || (config.shift - 1.6).abs() > f64::EPSILON
            || (config.scale - 0.3).abs() > f64::EPSILON
            || config.n_neighbors != 5
        {
            return Err(invalid_input("imputation options require --impute-method"));
        }
        return Ok(());
    }
    if !matches!(method.as_str(), "mindet" | "minprob")
        && (config.quantile - 0.01).abs() > f64::EPSILON
    {
        return Err(invalid_input(
            "--impute-quantile only applies to mindet/minprob",
        ));
    }
    if method != "minprob"
        && ((config.shift - 1.6).abs() > f64::EPSILON || (config.scale - 0.3).abs() > f64::EPSILON)
    {
        return Err(invalid_input(
            "--impute-shift/--impute-scale only apply to minprob",
        ));
    }
    if !matches!(method.as_str(), "knn" | "seqknn") && config.n_neighbors != 5 {
        return Err(invalid_input(
            "--impute-n-neighbors only applies to knn/seqknn",
        ));
    }
    match method.as_str() {
        "" | "none" => return Err(invalid_input("--impute-method must name a method")),
        "mindet" | "minprob" => {
            if !config.quantile.is_finite() || !(0.0..=1.0).contains(&config.quantile) {
                return Err(invalid_input("impute-quantile must be between 0 and 1"));
            }
            if method == "minprob"
                && (!config.shift.is_finite() || !config.scale.is_finite() || config.scale < 0.0)
            {
                return Err(invalid_input(
                    "minprob imputation requires finite shift and non-negative scale",
                ));
            }
        }
        "knn" | "seqknn" if config.n_neighbors == 0 => {
            return Err(invalid_input(
                "impute-n-neighbors must be greater than zero",
            ));
        }
        "mean" | "median" | "constant" | "zero" | "most_frequent" | "knn" | "seqknn" | "impseq"
        | "gms" | "bpca" | "impseqrob" | "qrilc" => {}
        "missforest" => return unsupported("missforest imputation is unported"),
        _ => return unsupported("imputation"),
    }
    Ok(())
}

/// Validate the differential-expression subset the Rust port implements.
///
/// The limma, deqms, rots, limrots, proda, and ensemble paths with
/// BH, IHW, BKY, or Storey correction are wired (mirroring
/// `mokume.analysis.differential_expression.DifferentialExpression(method=...)`).
/// `auto` is accepted and resolved to a concrete method just before the DE stage
/// (Python's `_resolve_de_method`: directlfq -> deqms, otherwise -> limrots).
/// Any other requested method, and every unsupported option, returns an error
/// rather than silently running a different test. `rots`, `limrots`, and `proda`
/// are faithful ports that match Python in
/// algorithm/distribution, not cell-for-cell: `rots`/`limrots` are RNG-based,
/// and `proda`'s per-protein MLE is optimizer-dependent (best-effort full port,
/// deterministic kernels cell-exact, end-to-end within the optimizer tolerance).
fn validate_de_subset(config: &FeatureToProteinsConfig) -> Result<()> {
    let de = &config.differential_expression;
    if !de.enabled {
        if de.contrasts.is_some()
            || de.contrasts_file.is_some()
            || de.ensemble_methods.is_some()
            || de.effect_size_gate.is_some()
            || de.output.is_some()
            || !de.method.eq_ignore_ascii_case("auto")
            || de.ensemble_min_k != 2
            || (de.log2fc_threshold - 0.5).abs() > f64::EPSILON
            || (de.fdr_threshold - 0.05).abs() > f64::EPSILON
            || !de.fdr_method.eq_ignore_ascii_case("bh")
        {
            return Err(invalid_input(
                "differential-expression options require --de",
            ));
        }
        return Ok(());
    }

    if config.input.sdrf.is_none() {
        return Err(invalid_input(
            "differential expression requires an SDRF file (--sdrf)",
        ));
    }

    if de.output.is_none() {
        return Err(invalid_input(
            "differential expression requires --de-output so results are not discarded",
        ));
    }

    de::validate_config(de, true)?;
    let effective_method = if de.method.eq_ignore_ascii_case("auto") {
        resolve_de_method(config)
    } else {
        de.method.trim().to_ascii_lowercase()
    };
    if matches!(effective_method.as_str(), "rots" | "limrots")
        && !de.fdr_method.eq_ignore_ascii_case("bh")
    {
        return Err(invalid_input(format!(
            "--de-fdr-method {} does not apply to {effective_method}, which retains its permutation FDR",
            de.fdr_method
        )));
    }
    if !de.method.eq_ignore_ascii_case("ensemble") && de.ensemble_min_k != 2 {
        return Err(invalid_input(
            "--de-ensemble-min-k only applies to --de-method ensemble",
        ));
    }

    // `--de-contrast-file` is expanded into `contrasts` before validation runs
    // (see `expand_de_contrasts_file`), so by here `contrasts_file` is already
    // folded in and the contrasts list below reflects both sources.
    match &de.contrasts {
        Some(contrasts) if !contrasts.is_empty() => {}
        _ => {
            return Err(invalid_input(
                "differential expression requires explicit contrasts via --de-contrast \
                 or --de-contrast-file",
            ));
        }
    }

    Ok(())
}

/// Validate only the ensemble-specific DE contract before any input access.
/// Other validation keeps its established ordering in the full validators.
fn validate_de_ensemble_options(config: &FeatureToProteinsConfig) -> Result<()> {
    let de = &config.differential_expression;
    if de.enabled && de.method.trim().eq_ignore_ascii_case("ensemble") {
        de::validated_ensemble_member_names(de)?;
    }
    Ok(())
}

/// Resolve the `--de-method auto` sentinel to a concrete method, mirroring
/// Python's `_resolve_de_method` (stages.py:1784): `directlfq` quantification
/// selects `deqms`, every other quantification selects `limrots`. A non-`auto`
/// method is returned unchanged. Resolved just before the DE stage so both the
/// validation and the `de::run_differential_expression` dispatch observe a
/// concrete method.
fn resolve_de_method(config: &FeatureToProteinsConfig) -> String {
    let method = config.differential_expression.method.trim();
    if !method.eq_ignore_ascii_case("auto") {
        return method.to_string();
    }
    if config.quantification == QuantMethod::DirectLfq {
        "deqms".to_string()
    } else {
        "limrots".to_string()
    }
}

/// Fold `--de-contrast-file` (a TSV with `group1`/`group2` columns) into the
/// `contrasts` list, mirroring Python (features2proteins.py:768): each row
/// appends `"<group1> vs <group2>"` after the repeated `--de-contrast` entries.
/// The returned config owns the merged list and has `contrasts_file` cleared, so
/// validation and the DE stage observe a single resolved contrast list. Called
/// only when `contrasts_file.is_some()`.
fn expand_de_contrasts_file(config: &FeatureToProteinsConfig) -> Result<FeatureToProteinsConfig> {
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

fn supports_run_normalization(method: &str) -> bool {
    parse_run_normalization_method(method).is_ok()
}

fn supports_sample_normalization(method: &str) -> bool {
    parse_sample_normalization_method(method).is_ok()
}

#[derive(Debug, Default)]
struct RunQcProteinStats {
    canonicals: HashSet<String>,
    runs: HashMap<i64, RunQcRunProteinStats>,
}

#[derive(Debug, Default)]
struct RunQcRunProteinStats {
    canonicals: HashSet<String>,
    total_intensity: f64,
}

#[derive(Debug, Default)]
struct RunQcRunStats {
    total_intensity: f64,
    feature_count: usize,
    protein_count: usize,
}

struct RunQcSource<'a> {
    filtering: FilterConfig,
    keep_shared_peptides: bool,
    min_unique_peptides: usize,
    contaminant_patterns: &'a [String],
    remove_protein_ids: &'a HashSet<String>,
}

/// Compute the Run-QC exclusions for `features2peptides`. Python applies the
/// filter pipeline to one sample frame at a time, then groups Run-QC decisions by
/// `TechReplicate`; the exclusion key is therefore `(sample, technical replicate)`.
/// The pre-pass mirrors the Python order:
///
/// 1. apply the per-`(sample, protein)` distinct-canonical `min_unique` gate;
/// 2. reject technical runs below total-intensity / distinct-feature / protein
///    thresholds;
/// 3. build each remaining sample's complete distinct `(protein, canonical)`
///    feature universe and reject runs whose absent fraction exceeds
///    `max_missing_rate`.
///
/// The normalization median is computed separately before Run-QC, so rejected
/// runs still contribute to it, matching Python.
fn collect_run_qc_exclusions(
    parquet: &Path,
    sdrf: Option<&SdrfTable>,
    run_qc: &RunQcFilterConfig,
    source: &RunQcSource<'_>,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<HashSet<RunQcKey>> {
    if !run_qc_is_active(run_qc) {
        return Ok(HashSet::new());
    }

    let proteins = collect_run_qc_proteins(parquet, sdrf, source, named_score)?;
    let min_unique = if source.keep_shared_peptides {
        0
    } else {
        source.min_unique_peptides
    };
    let runs = summarize_run_qc_runs(&proteins, min_unique);
    let mut excluded = primary_run_qc_exclusions(&runs, run_qc);
    extend_missing_rate_exclusions(&proteins, min_unique, run_qc, &mut excluded);
    Ok(excluded)
}

fn run_qc_is_active(run_qc: &RunQcFilterConfig) -> bool {
    run_qc.min_total_intensity > 0.0
        || run_qc.min_identified_features > 0
        || run_qc.min_identified_proteins > 0
        || run_qc.max_missing_rate < 1.0
}

/// Buffer per-run feature sets under each `(sample, protein)`: the protein-level
/// `min_unique` decision needs the whole sample, while Run-QC metrics need the
/// technical-run subsets.
fn collect_run_qc_proteins(
    parquet: &Path,
    sdrf: Option<&SdrfTable>,
    source: &RunQcSource<'_>,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<HashMap<(String, String), RunQcProteinStats>> {
    let mut proteins: HashMap<(String, String), RunQcProteinStats> = HashMap::new();
    let reader = QpxParquetReader::open(parquet, DEFAULT_QPX_BATCH_SIZE)?;
    stream_qpx_features_maybe_score(reader, named_score, |feature| {
        if !passes_feature_filter(&feature, source.filtering, source.keep_shared_peptides) {
            return Ok(());
        }
        // Run-QC shares the median pre-pass contaminant semantics (Python computes
        // both via `SQLFilterBuilder` ahead of the per-sample chain): match the
        // **raw** `pg_accessions` as a case-sensitive substring; structured
        // `is_decoy` flags are handled independently by the load filter.
        if source.filtering.remove_contaminants
            && matches_sql_contaminant(&feature.protein_accessions, source.contaminant_patterns)
        {
            return Ok(());
        }
        let Some(protein_group) = protein_group_name(&feature.protein_accessions) else {
            return Ok(());
        };
        if has_removed_accession(&feature.protein_accessions, source.remove_protein_ids) {
            return Ok(());
        }
        let sdrf_record = sdrf_record(&feature, sdrf)?;
        let sample = sample_name(&feature, sdrf_record);
        let technical_replicate =
            run_qc_key(&feature, sdrf_record, sample.clone()).technical_replicate;
        let protein = proteins.entry((sample, protein_group)).or_default();
        protein.canonicals.insert(feature.sequence.clone());
        let run = protein.runs.entry(technical_replicate).or_default();
        run.canonicals.insert(feature.sequence.clone());
        run.total_intensity += feature.intensity;
        Ok(())
    })?;
    Ok(proteins)
}

fn summarize_run_qc_runs(
    proteins: &HashMap<(String, String), RunQcProteinStats>,
    min_unique: usize,
) -> HashMap<RunQcKey, RunQcRunStats> {
    let mut runs: HashMap<RunQcKey, RunQcRunStats> = HashMap::new();
    for ((sample, _protein), protein) in proteins {
        if protein.canonicals.len() < min_unique {
            continue;
        }
        for (technical_replicate, protein_run) in &protein.runs {
            let run = runs
                .entry(RunQcKey {
                    sample: sample.clone(),
                    technical_replicate: *technical_replicate,
                })
                .or_default();
            run.total_intensity += protein_run.total_intensity;
            run.feature_count += protein_run.canonicals.len();
            run.protein_count += 1;
        }
    }
    runs
}

fn primary_run_qc_exclusions(
    runs: &HashMap<RunQcKey, RunQcRunStats>,
    run_qc: &RunQcFilterConfig,
) -> HashSet<RunQcKey> {
    let mut excluded = HashSet::new();
    for (key, run) in runs {
        let drop = (run_qc.min_total_intensity > 0.0
            && run.total_intensity < run_qc.min_total_intensity)
            || (run_qc.min_identified_features > 0
                && run.feature_count < run_qc.min_identified_features)
            || (run_qc.min_identified_proteins > 0
                && run.protein_count < run_qc.min_identified_proteins);
        if drop {
            excluded.insert(key.clone());
        }
    }
    excluded
}

fn extend_missing_rate_exclusions(
    proteins: &HashMap<(String, String), RunQcProteinStats>,
    min_unique: usize,
    run_qc: &RunQcFilterConfig,
    excluded: &mut HashSet<RunQcKey>,
) {
    if run_qc.max_missing_rate >= 1.0 {
        return;
    }

    // MissingRateFilter runs after the preceding Run-QC filters. Its universe is
    // the distinct `(ProteinName, PeptideCanonical)` set still present in the
    // sample, while the numerator is each surviving run's distinct detected set.
    let mut sample_universe_size: HashMap<String, usize> = HashMap::new();
    let mut detected_by_run: HashMap<RunQcKey, usize> = HashMap::new();
    for ((sample, _protein), protein) in proteins {
        if protein.canonicals.len() < min_unique {
            continue;
        }
        let mut surviving_canonicals = HashSet::new();
        for (technical_replicate, protein_run) in &protein.runs {
            let key = RunQcKey {
                sample: sample.clone(),
                technical_replicate: *technical_replicate,
            };
            if excluded.contains(&key) {
                continue;
            }
            surviving_canonicals.extend(protein_run.canonicals.iter());
            *detected_by_run.entry(key).or_insert(0) += protein_run.canonicals.len();
        }
        *sample_universe_size.entry(sample.clone()).or_insert(0) += surviving_canonicals.len();
    }

    for (key, detected) in detected_by_run {
        let universe = sample_universe_size.get(&key.sample).copied().unwrap_or(0);
        if universe == 0 {
            continue;
        }
        let missing_rate = 1.0 - (detected as f64 / universe as f64);
        if missing_rate > run_qc.max_missing_rate {
            excluded.insert(key);
        }
    }
}

/// The group-level intensity filters that need a per-sample buffering pre-pass
/// (CVThresholdFilter and QuantileFilter), computed together so their Python
/// ordering (CV before Quantile, `intensity.py` factory) is honoured: the
/// quantile bounds are taken over the *post-CV* survivors.
#[derive(Debug, Default)]
struct IntensityGroupFilters {
    /// `(sample, ProteinName, PeptideCanonical)` triples dropped by the CV filter.
    cv_dropped: HashSet<(String, String, String)>,
    /// Per-sample `[lower, upper]` QuantileFilter bounds over the post-CV set.
    quantile_bounds: HashMap<String, (f64, f64)>,
}

/// Pre-pass for the buffering intensity filters (CVThresholdFilter at
/// Python `intensity.py:111-154`, QuantileFilter at `intensity.py:278-318`).
///
/// Both run on `dataset_df`, which holds a single sample (`peptide.py:266`), so
/// the natural unit is the per-`(sample, ProteinName, PeptideCanonical)` group.
/// We buffer that group's raw intensities once, then:
///   1. drop the group when its within-sample CV (ddof = 1) exceeds
///      `cv_threshold` (a `None` CV -- a single measurement or non-positive mean
///      -- always passes, mirroring Python keeping `NaN` CVs);
///   2. pool the surviving groups' intensities per sample and take the linear
///      quantiles for the `[lower, upper]` bounds.
///
/// Step 2 deliberately runs over the step-1 survivors: Python wires CV ahead of
/// Quantile, so the quantile bounds must not include the CV-dropped rows
/// (verified end-to-end against the `mokume` CLI). Both steps see the same
/// load-gated, `min_unique`-gated rows Python's pipeline sees at
/// `intensity.py:293` -- crucially *before* the per-row peptide filters
/// (length/charge/modification/missed cleavage), which Python applies only after
/// the intensity block, so a charge/length-doomed row still feeds the CV. The
/// bounds use f64, matching pandas upcasting the f32 column for `quantile`.
fn collect_intensity_group_filters(
    parquet: &Path,
    sdrf: Option<&SdrfTable>,
    source: &RunQcSource<'_>,
    intensity: &IntensityFilterConfig,
    run_qc_excluded: &HashSet<RunQcKey>,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<IntensityGroupFilters> {
    let cv_active = intensity.cv_threshold.is_some();
    let quantile_active = intensity.quantile_lower > 0.0 || intensity.quantile_upper < 1.0;
    if !cv_active && !quantile_active {
        return Ok(IntensityGroupFilters::default());
    }
    // MinIntensityFilter runs in the pipeline ahead of both CV and Quantile, so
    // apply the same floor before collecting (a no-op beyond the `> 0` load filter
    // when `min_intensity == 0`).
    let min_floor =
        filters::effective_min_intensity(intensity.min_intensity, intensity.remove_zero_intensity);
    // Phase 1: buffer per `(sample, protein, canonical)` the raw intensities, and
    // per `(sample, protein)` the distinct canonicals (for the `min_unique` gate),
    // applying the same load-time filters as ingest.
    //
    // Do NOT drop contaminant rows here. Python applies the pipeline's
    // `ContaminantFilter` *after* CV/QuantileFilter (factory.py wires the
    // protein-level filters behind the intensity-level ones), and the CLI's
    // pre-pipeline `remove_decoy_contaminants` flag defaults to off
    // (features2peptides.py:61). So at `intensity.py:293` the frame still holds
    // contaminant rows, and the per-protein `min_unique` gate (peptide.py:279),
    // the CV groups, and the quantile bounds are all computed over them. The
    // ingest per-row drop still removes contaminants from the exported output,
    // matching the later `ContaminantFilter`; only the group decisions must
    // include them, so this pre-pass deliberately omits the contaminant check.
    let mut group_intensities: HashMap<(String, String, String), Vec<f64>> = HashMap::new();
    let mut cell_canonicals: HashMap<(String, String), HashSet<String>> = HashMap::new();
    let reader = QpxParquetReader::open(parquet, DEFAULT_QPX_BATCH_SIZE)?;
    stream_qpx_features_maybe_score(reader, named_score, |feature| {
        if !passes_feature_filter(&feature, source.filtering, source.keep_shared_peptides) {
            return Ok(());
        }
        let Some(protein_group) = protein_group_name(&feature.protein_accessions) else {
            return Ok(());
        };
        if feature.intensity < min_floor {
            return Ok(());
        }
        let sdrf_record = sdrf_record(&feature, sdrf)?;
        let sample = sample_name(&feature, sdrf_record);
        let run_key = run_qc_key(&feature, sdrf_record, sample.clone());
        if run_qc_excluded.contains(&run_key) {
            return Ok(());
        }
        cell_canonicals
            .entry((sample.clone(), protein_group.clone()))
            .or_default()
            .insert(feature.sequence.clone());
        group_intensities
            .entry((sample, protein_group, feature.sequence.clone()))
            .or_default()
            .push(feature.intensity);
        Ok(())
    })?;

    let min_unique = if source.keep_shared_peptides {
        0
    } else {
        source.min_unique_peptides
    };

    // Phase 2: per group, apply the `min_unique` gate (Python applies it before the
    // filter pipeline, peptide.py:278-281), then the CV decision. Surviving groups'
    // intensities are pooled per sample for the quantile bounds.
    let mut cv_dropped = HashSet::new();
    let mut per_sample: HashMap<String, Vec<f64>> = HashMap::new();
    for ((sample, protein, canonical), intensities) in group_intensities {
        // The `min_unique` gate keys on `(sample, protein)`: a protein with too
        // few distinct canonicals never reaches the filter pipeline, so its rows
        // feed neither the CV nor the quantile bounds.
        let canonical_count = cell_canonicals
            .get(&(sample.clone(), protein.clone()))
            .map_or(0, HashSet::len);
        if canonical_count < min_unique {
            continue;
        }
        if let Some(threshold) = intensity.cv_threshold {
            // `None` CV (single value or non-positive mean) keeps the group, exactly
            // like Python keeping `NaN` CVs (intensity.py:136). Python computes the
            // CV on the *float32* `NormIntensity` column (`pandas.Series.std/mean`),
            // so its `cv` carries float32 precision before the `<= threshold`
            // comparison upcasts it to f64. We compute the CV in f64 (the
            // oracle-locked primitive) then round it to f32, reproducing that
            // precision. The residual f32 rounding can only flip the keep/drop
            // decision when a group's true CV lands within ~1e-7 (relative) of the
            // threshold; a 30k-group fuzz against pandas showed zero such flips at a
            // typical threshold, so this is a measure-zero boundary, not a bias.
            if let Some(cv) = filters::coefficient_of_variation(&intensities) {
                if f64::from(cv as f32) > threshold {
                    cv_dropped.insert((sample, protein, canonical));
                    continue;
                }
            }
        }
        if quantile_active {
            per_sample.entry(sample).or_default().extend(intensities);
        }
    }

    let mut quantile_bounds = HashMap::new();
    if quantile_active {
        for (sample, mut values) in per_sample {
            values.retain(|value| value.is_finite());
            if values.is_empty() {
                continue;
            }
            let lower = quantile_linear(&mut values, intensity.quantile_lower);
            let upper = quantile_linear(&mut values, intensity.quantile_upper);
            if let (Some(lower), Some(upper)) = (lower, upper) {
                quantile_bounds.insert(sample, (lower, upper));
            }
        }
    }
    Ok(IntensityGroupFilters {
        cv_dropped,
        quantile_bounds,
    })
}

/// First `_`-token of `value`, i.e. DuckDB `split_part(value, '_', 1)`. A name
/// with no `_` is its own first token. This is Python's `mixture` key
/// (`feature.py:274-276` / `420-422`): the first token of the **sample
/// accession**, which with no SDRF is the `run_file_name`.
fn irs_mixture_first_token(value: &str) -> &str {
    value.split('_').next().unwrap_or(value)
}

/// Channel-IRS pre-pass (Python `get_irs_scaling_factors`, `feature.py:711-817`)
/// for every `irs_scope`. Streams the QPX features, keeps only the reference
/// channel's rows (`label == irs.channel`), applies the same explicit filters
/// Python's SQL uses (`intensity > 0`; the contaminant `NOT LIKE` only when
/// `remove_contaminants`; the `min_intensity` floor only when positive), groups
/// the surviving intensities by run into a per-run `irs_value` via the chosen
/// statistic, then derives the scale per technical replicate under `irs.scope`.
///
/// Each run also carries its `mixture` (Python `split_part(sample_accession,
/// '_', 1)`): with no SDRF the sample accession is the `run_file_name`, so the
/// mixture is its first `_`-token; with an SDRF the mixture is the first token
/// of the joined `source name`. `by_mixture` / `two_stage` use this; `global`
/// ignores it.
///
/// Returns the `techreplicate -> scale` map. The techreplicate key is Python's
/// IRS-internal `split_part(run, '_', 2)` ([`irs_tech_replicate_of`]); the
/// **apply** side keys by the last `_` token ([`tech_replicate_of`]) — both are
/// reproduced exactly because they diverge for runs with three or more tokens.
/// An empty map means no valid scale could be computed; the command layer turns
/// that into an explicit input error.
fn collect_irs_scale(
    parquet: &Path,
    irs: &IrsChannelConfig,
    sdrf: Option<&SdrfTable>,
    remove_contaminants: bool,
    contaminant_patterns: &[String],
    min_intensity: f64,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<HashMap<i64, f64>> {
    // Phase 1: per run, buffer the reference-channel intensities and remember the
    // IRS-internal techreplicate and mixture. A run with no integer techreplicate
    // cannot key into the scale dict, so it is dropped (Python's `CAST` would
    // error out).
    struct RunBuffer {
        techrep: i64,
        mixture: String,
        intensities: Vec<f64>,
    }
    let mut runs: HashMap<String, RunBuffer> = HashMap::new();
    let reader = QpxParquetReader::open(parquet, DEFAULT_QPX_BATCH_SIZE)?;
    stream_qpx_features_maybe_score(reader, named_score, |feature| {
        // Channel match is on the *raw* feature label (Python filters on
        // `channel = ?` before any reformat). A row with no label never matches.
        if feature.label.as_deref() != Some(irs.channel.as_str()) {
            return Ok(());
        }
        if !(feature.intensity.is_finite() && feature.intensity > 0.0) {
            return Ok(());
        }
        if min_intensity > 0.0 && feature.intensity < min_intensity {
            return Ok(());
        }
        // Contaminant `NOT LIKE` on the raw `pg_accessions`, mirroring the median
        // pre-pass / Python's `SQLFilterBuilder` (case-sensitive substring; the
        // default pattern list routes through the `is_contaminant` fallback).
        if remove_contaminants
            && matches_sql_contaminant(&feature.protein_accessions, contaminant_patterns)
        {
            return Ok(());
        }
        let Some(techrep) = irs_tech_replicate_of(&feature.run_file_name) else {
            return Ok(());
        };
        let sdrf_record = sdrf_record(&feature, sdrf)?;
        let sample_accession = sdrf_record.map_or(feature.run_file_name.as_str(), |record| {
            record.sample_accession.as_str()
        });
        runs.entry(feature.run_file_name.clone())
            .or_insert_with(|| {
                // Mixture is the first `_`-token of the *sample accession*, which is
                // the joined SDRF `source name` when an SDRF is present and the
                // `run_file_name` otherwise (Python `parquet_db` view). The SDRF
                // join keys on the run name (with the raw feature label), mirroring
                // `enrich_with_sdrf`; when an SDRF is present, an unmatched run or
                // label is rejected instead of falling back to the run name.
                RunBuffer {
                    techrep,
                    mixture: irs_mixture_first_token(sample_accession).to_owned(),
                    intensities: Vec::new(),
                }
            })
            .intensities
            .push(feature.intensity);
        Ok(())
    })?;

    // Phase 2: collapse each run's buffered intensities into one `irs_value`, then
    // derive the scale under `irs.scope`. Collect into a `BTreeMap` keyed by run
    // name so the runs are visited in a stable (sorted) order; this makes the
    // (rare) techreplicate collision resolve deterministically (last run name
    // wins), matching Python's `dict(zip(...))`.
    let per_run: Vec<(i64, String, Vec<f64>)> = runs
        .into_iter()
        .map(|(run_name, buffer)| {
            (
                run_name,
                (buffer.techrep, buffer.mixture, buffer.intensities),
            )
        })
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect();
    Ok(match irs.scope {
        IrsScope::Global => {
            let runs = per_run
                .into_iter()
                .map(|(techrep, _mixture, intensities)| (techrep, intensities))
                .collect();
            irs_global_scale_from_runs(runs, irs.stat)
        }
        IrsScope::ByMixture => irs_by_mixture_scale_from_runs(per_run, irs.stat),
        IrsScope::TwoStage => irs_two_stage_scale_from_runs(per_run, irs.stat),
    })
}

/// Pure global-scope IRS math (Python `get_irs_scaling_factors`,
/// `feature.py:776-815`, global branch). Given each run's `(techreplicate,
/// reference-channel intensities)` in a stable order, collapse to one
/// `irs_value` per run via `stat`, drop non-positive `irs_value`s, take the
/// global center via the same `stat`, and return `scale[techrep] = center /
/// irs_value`. Runs sharing a techreplicate resolve last-wins, matching Python's
/// `dict(zip(...))`. Returns an empty map when nothing positive survives.
fn irs_global_scale_from_runs(per_run: Vec<(i64, Vec<f64>)>, stat: IrsStat) -> HashMap<i64, f64> {
    let aggregate = |values: &mut Vec<f64>| match stat {
        IrsStat::Median => median(values),
        IrsStat::Mean => mean_positive(values),
    };
    let mut irs_values: Vec<(i64, f64)> = Vec::new();
    for (techrep, mut intensities) in per_run {
        // `irs_df = irs_df[irs_df["irs_value"] > 0]` drops non-positive centers.
        if let Some(irs_value) = aggregate(&mut intensities) {
            if irs_value > 0.0 {
                irs_values.push((techrep, irs_value));
            }
        }
    }
    if irs_values.is_empty() {
        return HashMap::new();
    }
    let mut centers: Vec<f64> = irs_values.iter().map(|&(_, value)| value).collect();
    let Some(global_center) = aggregate(&mut centers) else {
        return HashMap::new();
    };
    let mut scale_by_techrep: HashMap<i64, f64> = HashMap::new();
    for (techrep, irs_value) in irs_values {
        scale_by_techrep.insert(techrep, global_center / irs_value);
    }
    scale_by_techrep
}

/// Collapse each run's buffered reference-channel intensities into a single
/// `irs_value` and drop the runs whose center is non-positive, mirroring
/// Python's per-run aggregate followed by `irs_df = irs_df[irs_df["irs_value"]
/// > 0]`. Preserves input (stable, run-name-sorted) order so any later
/// `dict(zip(...))` collision resolves last-wins exactly as in pandas. Returns
/// `(techrep, mixture, irs_value)` per surviving run.
fn irs_values_per_run(
    per_run: Vec<(i64, String, Vec<f64>)>,
    stat: IrsStat,
) -> Vec<(i64, String, f64)> {
    let aggregate = |values: &mut Vec<f64>| match stat {
        IrsStat::Median => median(values),
        IrsStat::Mean => mean_positive(values),
    };
    let mut out = Vec::with_capacity(per_run.len());
    for (techrep, mixture, mut intensities) in per_run {
        if let Some(irs_value) = aggregate(&mut intensities) {
            if irs_value > 0.0 {
                out.push((techrep, mixture, irs_value));
            }
        }
    }
    out
}

/// Per-mixture center over the surviving runs' `irs_value`s, mirroring Python's
/// `irs_df.groupby("mixture")["irs_value"].transform(stat)`. The returned map is
/// keyed by mixture; every input mixture is present because each contributes at
/// least one run. `median` / `mean_positive` keep only positive finite values,
/// which is a no-op here (all `irs_value`s are already positive).
fn irs_mixture_centers(irs_values: &[(i64, String, f64)], stat: IrsStat) -> HashMap<String, f64> {
    let mut by_mixture: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (_, mixture, irs_value) in irs_values {
        by_mixture
            .entry(mixture.clone())
            .or_default()
            .push(*irs_value);
    }
    let mut centers: HashMap<String, f64> = HashMap::new();
    for (mixture, mut values) in by_mixture {
        let center = match stat {
            IrsStat::Median => median(&mut values),
            IrsStat::Mean => mean_positive(&values),
        };
        if let Some(center) = center {
            centers.insert(mixture, center);
        }
    }
    centers
}

/// Pure `by_mixture`-scope IRS math (Python `get_irs_scaling_factors`,
/// `feature.py:779-784`). For each surviving run, `scale = mixture_center /
/// irs_value`, where `mixture_center` is the `stat` over that mixture's
/// `irs_value`s. Runs sharing a techreplicate resolve last-wins over the stable
/// input order, matching Python's `dict(zip(techrep_guess, scale))`. Returns an
/// empty map when no run survives the positive-`irs_value` filter.
fn irs_by_mixture_scale_from_runs(
    per_run: Vec<(i64, String, Vec<f64>)>,
    stat: IrsStat,
) -> HashMap<i64, f64> {
    let irs_values = irs_values_per_run(per_run, stat);
    if irs_values.is_empty() {
        return HashMap::new();
    }
    let centers = irs_mixture_centers(&irs_values, stat);
    let mut scale_by_techrep: HashMap<i64, f64> = HashMap::new();
    for (techrep, mixture, irs_value) in irs_values {
        if let Some(&mixture_center) = centers.get(&mixture) {
            scale_by_techrep.insert(techrep, mixture_center / irs_value);
        }
    }
    scale_by_techrep
}

/// Pure `two_stage`-scope IRS math (Python `get_irs_scaling_factors`,
/// `feature.py:785-804`). Stage 1 is `by_mixture` (`mixture_center /
/// irs_value`); stage 2 re-anchors each mixture to a global center over the
/// **distinct** mixture centers (`global_center / mixture_center`). The applied
/// scale is the product, which algebraically equals `global_center / irs_value`
/// but is computed exactly as the two factors so float rounding matches Python.
/// `global_center` is the `stat` over one center per distinct mixture, mirroring
/// `irs_df[["mixture","mixture_center"]].drop_duplicates()`. Runs sharing a
/// techreplicate resolve last-wins over the stable input order. Returns an empty
/// map when no run survives or the global center cannot be formed.
fn irs_two_stage_scale_from_runs(
    per_run: Vec<(i64, String, Vec<f64>)>,
    stat: IrsStat,
) -> HashMap<i64, f64> {
    let irs_values = irs_values_per_run(per_run, stat);
    if irs_values.is_empty() {
        return HashMap::new();
    }
    let centers = irs_mixture_centers(&irs_values, stat);
    // Global center over one center per distinct mixture (`drop_duplicates` on
    // the `(mixture, mixture_center)` pair). `BTreeMap::values` visits mixtures
    // in sorted order, but the `stat` is order-independent.
    let mut distinct_centers: Vec<f64> = centers.values().copied().collect();
    let global_center = match stat {
        IrsStat::Median => median(&mut distinct_centers),
        IrsStat::Mean => mean_positive(&distinct_centers),
    };
    let Some(global_center) = global_center else {
        return HashMap::new();
    };
    let mut scale_by_techrep: HashMap<i64, f64> = HashMap::new();
    for (techrep, mixture, irs_value) in irs_values {
        if let Some(&mixture_center) = centers.get(&mixture) {
            let stage1 = mixture_center / irs_value;
            let stage2 = global_center / mixture_center;
            scale_by_techrep.insert(techrep, stage1 * stage2);
        }
    }
    scale_by_techrep
}

/// Resolve the IRS reference channel from the SDRF when `--irs_channel` is not
/// given but `--irs_autodetect_regex` is (Python `peptide.py:219-233`). Rows
/// whose `source name` matches the regex (case-insensitive, `str.contains`) vote
/// with their `comment[label]`; the winning channel is the most frequent label,
/// ties broken by the lexicographically smallest (pandas `mode().iloc[0]`, whose
/// result is sorted).
///
/// Returns `Ok(None)` — IRS is skipped, matching Python's warning path — when the
/// SDRF lacks a `source name` or `comment[label]` column, when no row matches, or
/// when every matching row has an empty label. The regex is compiled
/// case-insensitively (`(?i)`), mirroring pandas' `case=False`; an invalid regex
/// surfaces as `InvalidInput`.
pub fn resolve_irs_autodetect_channel(
    sdrf_path: &Path,
    autodetect_regex: &str,
) -> Result<Option<String>> {
    let table = SdrfRawTable::from_path(sdrf_path)?;
    // `load_sdrf` lowercases headers, so the canonical column names match.
    let (Some(source_col), Some(label_col)) = (
        table.column_index("source name"),
        table.column_index("comment[label]"),
    ) else {
        return Ok(None);
    };
    let regex = Regex::new(&format!("(?i){autodetect_regex}")).map_err(|source| {
        MokumeError::InvalidInput {
            message: format!("invalid --irs_autodetect_regex `{autodetect_regex}`: {source}"),
        }
    })?;
    // Tally label frequencies among matching rows. A `BTreeMap` keeps the keys
    // sorted so the tie-break naturally selects the smallest label, reproducing
    // pandas `mode().iloc[0]` (mode returns sorted values; `.iloc[0]` is the
    // first). Empty labels are ignored (an empty `comment[label]` is not a
    // channel).
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in 0..table.row_count() {
        if regex.is_match(table.cell(row, source_col)) {
            let label = table.cell(row, label_col);
            if !label.is_empty() {
                *counts.entry(label.to_owned()).or_insert(0) += 1;
            }
        }
    }
    // Highest count wins; the sorted iteration order makes the first max the
    // smallest label, so a plain "strictly greater" comparison yields the
    // tie-break Python applies.
    let mut best: Option<(String, usize)> = None;
    for (label, count) in counts {
        if best
            .as_ref()
            .is_none_or(|(_, best_count)| count > *best_count)
        {
            best = Some((label, count));
        }
    }
    Ok(best.map(|(label, _)| label))
}

fn collect_intensity_factors(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    contaminant_patterns: &[String],
    keep_shared_peptides: bool,
    memory: &MemoryPlan,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<IntensityFactors> {
    if matches!(
        config.quantification,
        QuantMethod::DirectLfq | QuantMethod::Ratio
    ) {
        return Ok(IntensityFactors::default());
    }

    let run_method = parse_run_normalization_method(&config.normalization.run_method)?;
    let sample_method = parse_sample_normalization_method(&config.normalization.sample_method)?;
    let sample_uses_factors = sample_method_uses_factors(sample_method);
    if run_method.is_none() && !sample_uses_factors {
        return Ok(IntensityFactors::default());
    }
    if sample_method == Some(SampleNormalizationMethod::ConditionMedian) && sdrf.is_none() {
        return Err(invalid_input(
            "conditionmedian sample normalization requires --sdrf option",
        ));
    }

    let normalization_proteins = config
        .normalization
        .normalization_proteins
        .as_deref()
        .filter(|_| sample_uses_factors)
        .map(load_normalization_proteins)
        .transpose()?;
    let mut collector = NormalizationFactorCollector::new(
        run_method,
        sample_method,
        sdrf,
        normalization_proteins,
        contaminant_patterns,
    );
    // `keep_shared_peptides` mirrors Python's `require_unique = not keep_shared`
    // in `SQLFilterBuilder` (`peptide.py:168/176`, `stages.py:329`): when shared
    // peptides are kept, the median pre-pass must include non-unique rows so the
    // per-sample median (and the resulting sample/run factors) matches Python.
    // The caller decides the value: the protein path keeps the legacy
    // `quantification == Pibaq` rule (`stages.py:325`), while the peptide path
    // forwards the `--keep-shared-peptides` flag from `FeatureToPeptidesConfig`,
    // which `peptide_export_config` cannot represent (it pins quantification to
    // `Sum`, so the old in-function `== Pibaq` derivation always read `false`).
    stream_normalization_features(config, sdrf, memory, named_score, &mut |feature| {
        collector.push(feature, config.filtering, keep_shared_peptides)
    })?;
    if collector.normalization_proteins.is_some()
        && sample_uses_factors
        && !collector.has_sample_factor_values()
    {
        return Err(invalid_input(
            "no features matched --normalization-proteins for sample normalization",
        ));
    }
    Ok(collector.into_factors())
}

fn stream_normalization_features<F>(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
    named_score: Option<&NamedScoreFilterConfig>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    match (&config.input.parquet, named_score) {
        (Some(parquet), Some(score)) => {
            let reader = QpxParquetReader::open(parquet, memory.qpx_batch_size())?;
            stream_qpx_features_with_score(reader, score, consume)
        }
        _ => stream_input_features(&config.input, sdrf, memory, consume),
    }
}

fn sample_method_uses_factors(method: Option<SampleNormalizationMethod>) -> bool {
    matches!(
        method,
        Some(SampleNormalizationMethod::GlobalMedian | SampleNormalizationMethod::ConditionMedian)
    )
}

#[derive(Debug)]
struct NormalizationFactorCollector<'a> {
    run_method: Option<RunNormalizationMethod>,
    sample_method: Option<SampleNormalizationMethod>,
    sdrf: Option<&'a SdrfTable>,
    normalization_proteins: Option<HashSet<String>>,
    /// Custom contaminant patterns for the median pre-pass (Python
    /// `SQLFilterBuilder.contaminant_patterns`). Matched against the **raw**
    /// `pg_accessions` via [`matches_sql_contaminant`], which falls back to the
    /// default [`is_contaminant`] when this equals the default list.
    contaminant_patterns: &'a [String],
    sample_values: HashMap<String, Vec<f64>>,
    condition_sample_values: HashMap<String, HashMap<String, Vec<f64>>>,
    run_values: HashMap<RunCellKey, Vec<f64>>,
}

impl<'a> NormalizationFactorCollector<'a> {
    fn new(
        run_method: Option<RunNormalizationMethod>,
        sample_method: Option<SampleNormalizationMethod>,
        sdrf: Option<&'a SdrfTable>,
        normalization_proteins: Option<HashSet<String>>,
        contaminant_patterns: &'a [String],
    ) -> Self {
        Self {
            run_method,
            sample_method,
            sdrf,
            normalization_proteins,
            contaminant_patterns,
            sample_values: HashMap::new(),
            condition_sample_values: HashMap::new(),
            run_values: HashMap::new(),
        }
    }

    fn push(
        &mut self,
        feature: QpxFeatureRecord,
        filtering: FilterConfig,
        keep_shared_peptides: bool,
    ) -> Result<()> {
        if !passes_feature_filter(&feature, filtering, keep_shared_peptides) {
            return Ok(());
        }
        // Contaminant removal in the median pre-pass mirrors Python's
        // `SQLFilterBuilder`: match the **raw** `pg_accessions` (not the parsed
        // group), as a case-sensitive literal substring. Structured `is_decoy`
        // flags are handled independently by `passes_feature_filter`.
        if filtering.remove_contaminants
            && matches_sql_contaminant(&feature.protein_accessions, self.contaminant_patterns)
        {
            return Ok(());
        }
        let Some(protein_group) = protein_group_name(&feature.protein_accessions) else {
            return Ok(());
        };

        let sdrf_record = sdrf_record(&feature, self.sdrf)?;
        let sample = sample_name(&feature, sdrf_record);
        if self
            .normalization_proteins
            .as_ref()
            .is_none_or(|proteins| proteins.contains(&protein_group))
        {
            match self.sample_method {
                Some(SampleNormalizationMethod::GlobalMedian) => {
                    self.sample_values
                        .entry(sample.clone())
                        .or_default()
                        .push(feature.intensity);
                }
                // MedianCenter/MeanCenter are dataset-level (centered on the
                // summed-canonical-peptide matrix after ingest), so they collect
                // no ingest-time factor here, matching the other cell-path
                // methods below.
                Some(
                    SampleNormalizationMethod::Quantile
                    | SampleNormalizationMethod::Rlr
                    | SampleNormalizationMethod::Loess
                    | SampleNormalizationMethod::Hierarchical
                    | SampleNormalizationMethod::MedianCenter
                    | SampleNormalizationMethod::MeanCenter
                    | SampleNormalizationMethod::Tmm,
                ) => {}
                Some(SampleNormalizationMethod::ConditionMedian) => {
                    let condition = sample_condition(&sample, sdrf_record);
                    self.condition_sample_values
                        .entry(condition)
                        .or_default()
                        .entry(sample.clone())
                        .or_default()
                        .push(feature.intensity);
                }
                None => {}
            }
        }

        if self.run_method.is_some() {
            self.run_values
                .entry(RunCellKey {
                    sample,
                    run: feature.run_file_name,
                })
                .or_default()
                .push(feature.intensity);
        }
        Ok(())
    }

    fn into_factors(self) -> IntensityFactors {
        let mut factors = IntensityFactors::default();
        if let Some(method) = self.run_method {
            factors.run = run_normalization_transforms(method, self.run_values);
        }
        match self.sample_method {
            Some(SampleNormalizationMethod::GlobalMedian) => {
                factors.sample = global_median_sample_factors(self.sample_values);
            }
            Some(SampleNormalizationMethod::ConditionMedian) => {
                factors.sample = condition_median_sample_factors(self.condition_sample_values);
            }
            // MedianCenter/MeanCenter are applied on the cell path post-ingest,
            // so they produce no per-sample factor here.
            Some(
                SampleNormalizationMethod::Quantile
                | SampleNormalizationMethod::Rlr
                | SampleNormalizationMethod::Loess
                | SampleNormalizationMethod::Hierarchical
                | SampleNormalizationMethod::MedianCenter
                | SampleNormalizationMethod::MeanCenter
                | SampleNormalizationMethod::Tmm,
            ) => {}
            None => {}
        }
        factors
    }

    fn has_sample_factor_values(&self) -> bool {
        !self.sample_values.is_empty() || !self.condition_sample_values.is_empty()
    }
}

/// Read exact accessions from `--remove_ids`, one per non-empty line.
fn load_remove_ids(path: &Path) -> Result<HashSet<String>> {
    let contents = read_to_string(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(contents
        .split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(parse_protein_accession)
        .filter(|accession| !accession.is_empty())
        .collect())
}

fn has_removed_accession(accessions: &[String], removed: &HashSet<String>) -> bool {
    accessions
        .iter()
        .map(|accession| parse_protein_accession(accession))
        .any(|accession| removed.contains(&accession))
}

fn load_normalization_proteins(path: &Path) -> Result<HashSet<String>> {
    let contents = read_to_string(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let proteins = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(parse_protein_accession)
        .filter(|protein| !protein.is_empty())
        .collect::<HashSet<_>>();
    if proteins.is_empty() {
        return Err(invalid_input(format!(
            "normalization proteins file `{}` does not contain protein IDs",
            path.display()
        )));
    }
    Ok(proteins)
}

fn sample_condition(sample: &str, sdrf_record: Option<&SdrfRecord>) -> String {
    sdrf_record
        .and_then(|record| record.condition.clone())
        .unwrap_or_else(|| sample.to_owned())
}

fn unsupported(stage: &'static str) -> Result<()> {
    Err(MokumeError::NotImplemented { stage })
}

fn passes_feature_filter(
    feature: &QpxFeatureRecord,
    filtering: FilterConfig,
    keep_shared_peptides: bool,
) -> bool {
    feature.sequence.len() >= filtering.min_aa
        && !feature.is_decoy.unwrap_or(false)
        && (keep_shared_peptides || feature.unique.unwrap_or(true))
        && feature.intensity.is_finite()
        && feature.intensity > 0.0
        && !feature.protein_accessions.is_empty()
}

/// Convert one aggregated `(protein, canonical, sample)` cell into a
/// [`DirectLfqIon`]; DirectLFQ's "ion" is the bare canonical sequence.
/// `ion_seq_rank` is the canonical's global alphabetical-by-sequence rank, used
/// to reproduce DirectLFQ's `sort([protein, sequence])` row order.
fn directlfq_ion((key, intensity): (DirectLfqCellKey, f64), ion_seq_rank: u32) -> DirectLfqIon {
    DirectLfqIon {
        protein: key.protein,
        ion: key.canonical,
        ion_seq_rank,
        sample: key.sample,
        intensity,
    }
}

struct PreparedDirectLfq {
    ions: Vec<DirectLfqIon>,
    lexical_to_protein: HashMap<ProteinId, ProteinId>,
    lexical_to_sample: HashMap<SampleId, SampleId>,
    solver_samples: Vec<SampleId>,
}

fn remap_directlfq_ion(
    entry: (DirectLfqCellKey, f64),
    ion_seq_rank: u32,
    protein_to_lexical: &HashMap<ProteinId, ProteinId>,
    sample_to_lexical: &HashMap<SampleId, SampleId>,
) -> DirectLfqIon {
    let mut ion = directlfq_ion(entry, ion_seq_rank);
    ion.protein = protein_to_lexical
        .get(&ion.protein)
        .copied()
        .unwrap_or(ion.protein);
    ion.sample = sample_to_lexical
        .get(&ion.sample)
        .copied()
        .unwrap_or(ion.sample);
    ion
}

/// Build DirectLFQ input with name-sorted ids. Matrix row and column positions
/// participate in DirectLFQ tie-breaking, so encounter-order registry ids must
/// not leak into the solver.
fn prepare_directlfq_ions(
    directlfq_sums: HashMap<DirectLfqCellKey, f64>,
    allowed_cells: &HashSet<CellKey>,
    proteins: &StringIdRegistry<ProteinId>,
    canonical_peptides: &StringIdRegistry<PeptideId>,
    samples: &StringIdRegistry<SampleId>,
) -> PreparedDirectLfq {
    let (protein_to_lexical, lexical_to_protein) = lexical_id_remap(proteins);
    let (sample_to_lexical, lexical_to_sample) = lexical_id_remap(samples);
    let seq_rank = canonical_sequence_ranks(&directlfq_sums, canonical_peptides);
    let mut ions = directlfq_sums
        .into_iter()
        .filter(|(key, _)| {
            allowed_cells.contains(&CellKey {
                protein: key.protein,
                sample: key.sample,
            })
        })
        .map(|entry| {
            let rank = seq_rank
                .get(&entry.0.canonical)
                .copied()
                .unwrap_or(u32::MAX);
            remap_directlfq_ion(entry, rank, &protein_to_lexical, &sample_to_lexical)
        })
        .collect::<Vec<_>>();
    ions.sort_by_key(|ion| {
        (
            ion.protein.get(),
            ion.ion_seq_rank,
            ion.ion.get(),
            ion.sample.get(),
        )
    });
    let mut solver_samples = ions.iter().map(|ion| ion.sample).collect::<Vec<_>>();
    solver_samples.sort_by_key(|sample| sample.get());
    solver_samples.dedup();
    PreparedDirectLfq {
        ions,
        lexical_to_protein,
        lexical_to_sample,
        solver_samples,
    }
}

fn remap_directlfq_values(
    values: Vec<(ProteinId, Vec<(SampleId, f64)>)>,
    prepared: &PreparedDirectLfq,
) -> HashMap<ProteinId, Vec<(SampleId, f64)>> {
    values
        .into_iter()
        .map(|(protein, per_sample)| {
            let protein = prepared
                .lexical_to_protein
                .get(&protein)
                .copied()
                .unwrap_or(protein);
            let per_sample = per_sample
                .into_iter()
                .map(|(sample, intensity)| {
                    let sample = prepared
                        .lexical_to_sample
                        .get(&sample)
                        .copied()
                        .unwrap_or(sample);
                    (sample, intensity)
                })
                .collect();
            (protein, per_sample)
        })
        .collect()
}

fn directlfq_sample_columns<'a>(
    prepared: &PreparedDirectLfq,
    samples: &'a StringIdRegistry<SampleId>,
) -> Result<Vec<(SampleId, &'a str)>> {
    prepared
        .solver_samples
        .iter()
        .map(|lexical| {
            let original = prepared
                .lexical_to_sample
                .get(lexical)
                .copied()
                .unwrap_or(*lexical);
            let name = samples
                .resolve(original)
                .ok_or_else(|| invalid_input("DirectLFQ ion export sample id is not registered"))?;
            Ok((*lexical, name))
        })
        .collect()
}

fn directlfq_ion_row(
    normalized: &DirectLfqNormalizedIon,
    prepared: &PreparedDirectLfq,
    proteins: &StringIdRegistry<ProteinId>,
    canonical_peptides: &StringIdRegistry<PeptideId>,
    sample_columns: &[(SampleId, &str)],
) -> Result<Vec<String>> {
    let original_protein = prepared
        .lexical_to_protein
        .get(&normalized.protein)
        .copied()
        .unwrap_or(normalized.protein);
    let protein = proteins
        .resolve(original_protein)
        .ok_or_else(|| invalid_input("DirectLFQ ion export protein id is not registered"))?;
    let ion = canonical_peptides
        .resolve(normalized.ion)
        .ok_or_else(|| invalid_input("DirectLFQ ion export peptide id is not registered"))?;
    let quantities = normalized
        .quantities
        .iter()
        .copied()
        .collect::<HashMap<_, _>>();
    let mut row = vec![protein.to_owned(), ion.to_owned()];
    row.extend(
        sample_columns
            .iter()
            .map(|(sample, _)| format_float(quantities.get(sample).copied().unwrap_or(0.0))),
    );
    Ok(row)
}

fn write_directlfq_ions(
    path: &Path,
    normalized_ions: &[DirectLfqNormalizedIon],
    prepared: &PreparedDirectLfq,
    proteins: &StringIdRegistry<ProteinId>,
    canonical_peptides: &StringIdRegistry<PeptideId>,
    samples: &StringIdRegistry<SampleId>,
) -> Result<()> {
    create_parent_dir(path)?;
    let file = File::create(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = WriterBuilder::new().from_writer(file);

    let sample_columns = directlfq_sample_columns(prepared, samples)?;
    let mut header = vec!["protein".to_owned(), "ion".to_owned()];
    header.extend(sample_columns.iter().map(|(_, name)| (*name).to_owned()));
    writer
        .write_record(header)
        .map_err(|source| csv_error(path, source))?;

    let mut rows = normalized_ions.iter().collect::<Vec<_>>();
    rows.sort_by_key(|normalized| {
        (
            normalized.protein.get(),
            normalized.ion_seq_rank,
            normalized.ion.get(),
        )
    });
    for normalized in rows {
        let row = directlfq_ion_row(
            normalized,
            prepared,
            proteins,
            canonical_peptides,
            &sample_columns,
        )?;
        writer
            .write_record(row)
            .map_err(|source| csv_error(path, source))?;
    }

    writer.flush().map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Assign every distinct canonical id appearing in `directlfq_sums` a rank by
/// sorting their bare sequence strings lexically, mirroring DirectLFQ's global
/// `sort([protein, sequence])` (`stages.py:887`). Ties on the (unique) sequence
/// strings cannot occur; canonical ids missing from the registry sort last by
/// their id so the order stays deterministic.
fn canonical_sequence_ranks(
    directlfq_sums: &HashMap<DirectLfqCellKey, f64>,
    canonical_peptides: &StringIdRegistry<PeptideId>,
) -> HashMap<PeptideId, u32> {
    let mut canonicals = directlfq_sums
        .keys()
        .map(|key| key.canonical)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    canonicals.sort_by(|a, b| {
        let seq_a = canonical_peptides.resolve(*a);
        let seq_b = canonical_peptides.resolve(*b);
        seq_a.cmp(&seq_b).then_with(|| a.get().cmp(&b.get()))
    });
    canonicals
        .into_iter()
        .enumerate()
        .map(|(rank, canonical)| (canonical, rank as u32))
        .collect()
}

fn protein_group_name(accessions: &[String]) -> Option<String> {
    let proteins = accessions
        .iter()
        .map(|accession| parse_protein_accession(accession))
        .filter(|accession| !accession.is_empty())
        .collect::<Vec<_>>();

    (!proteins.is_empty()).then(|| proteins.join(";"))
}

fn first_protein_name(accessions: &[String]) -> Option<String> {
    accessions
        .first()
        .map(|accession| parse_protein_accession(accession))
        .filter(|accession| !accession.is_empty() && !accession.contains(';'))
}

fn parse_protein_accession(accession: &str) -> String {
    let accession = accession.trim();
    let parts = accession.split('|').collect::<Vec<_>>();
    if parts.len() == 1 {
        return accession.to_owned();
    }
    if matches!(
        parts[0].to_ascii_lowercase().as_str(),
        "sp" | "tr" | "sw" | "nxp"
    ) && parts.len() >= 2
    {
        parts[1].to_owned()
    } else {
        parts[0].to_owned()
    }
}

fn sdrf_record<'a>(
    feature: &QpxFeatureRecord,
    sdrf: Option<&'a SdrfTable>,
) -> Result<Option<&'a SdrfRecord>> {
    match sdrf {
        Some(table) => table
            .lookup(&feature.run_file_name, feature.label.as_deref())
            .map(Some),
        None => Ok(None),
    }
}

fn sample_name(feature: &QpxFeatureRecord, sdrf_record: Option<&SdrfRecord>) -> String {
    if let Some(record) = sdrf_record {
        return record.sample_accession.clone();
    }

    feature
        .sample_accession
        .clone()
        .unwrap_or_else(|| feature.run_file_name.clone())
}

fn run_qc_key(
    feature: &QpxFeatureRecord,
    sdrf_record: Option<&SdrfRecord>,
    sample: String,
) -> RunQcKey {
    let technical_replicate = sdrf_record
        .and_then(|record| record.technical_replicate)
        .map(i64::from)
        .unwrap_or_else(|| tech_replicate_of(&feature.run_file_name));
    RunQcKey {
        sample,
        technical_replicate,
    }
}

fn peptide_key(feature: &QpxFeatureRecord) -> String {
    format!("{}|z{}", feature.peptidoform, feature.charge)
}

fn is_contaminant(accession: &str) -> bool {
    // Mirrors Python's default contaminant deletion semantics exactly.
    let upper = accession.to_ascii_uppercase();
    upper.contains("CONTAMINANT")
        || upper.contains("CONTAM_")
        || upper.contains("ENTRAP")
        || upper.contains("DECOY")
}

fn is_decoy_accession(accession: &str) -> bool {
    accession.to_ascii_uppercase().contains("DECOY")
}

/// A group is removed when any member is a decoy or all members are contaminants.
fn is_contaminant_group<'a, I, F, D>(accessions: I, is_contam: F, is_decoy: D) -> bool
where
    I: IntoIterator<Item = &'a str>,
    F: Fn(&str) -> bool,
    D: Fn(&str) -> bool,
{
    let mut any = false;
    let mut all_contaminant = true;
    for accession in accessions {
        let accession = accession.trim();
        if accession.is_empty() {
            continue;
        }
        any = true;
        if is_decoy(accession) {
            return true;
        }
        all_contaminant &= is_contam(accession);
    }
    any && all_contaminant
}

fn is_decoy_pattern(pattern: &str) -> bool {
    pattern.eq_ignore_ascii_case("DECOY")
}

/// The default contaminant patterns.
/// When the configured patterns are empty or equal this list, the two custom
/// matchers below are bypassed in favour of [`is_contaminant`].
fn is_default_contaminant_patterns(patterns: &[String]) -> bool {
    patterns.is_empty()
        || (patterns.len() == 4
            && patterns[0] == "CONTAMINANT"
            && patterns[1] == "CONTAM_"
            && patterns[2] == "ENTRAP"
            && patterns[3] == "DECOY")
}

/// Pre-pass group rule on raw `pg_accessions` (case-sensitive, as Python's SQL filter).
fn matches_sql_contaminant(accessions: &[String], patterns: &[String]) -> bool {
    let members = accessions.iter().flat_map(|value| value.split(';'));
    if is_default_contaminant_patterns(patterns) {
        return is_contaminant_group(members, is_contaminant, is_decoy_accession);
    }
    is_contaminant_group(
        members,
        |accession| {
            patterns
                .iter()
                .any(|pattern| accession.contains(pattern.as_str()))
        },
        |accession| {
            patterns
                .iter()
                .any(|pattern| is_decoy_pattern(pattern) && accession.contains(pattern.as_str()))
        },
    )
}

/// Ingest group rule on the parsed `;`-joined group (case-insensitive).
fn matches_protein_contaminant(protein_group: &str, patterns: &[String]) -> bool {
    let members = protein_group.split(';');
    if is_default_contaminant_patterns(patterns) {
        return is_contaminant_group(members, is_contaminant, is_decoy_accession);
    }
    let upper_patterns = patterns
        .iter()
        .map(|pattern| pattern.to_ascii_uppercase())
        .collect::<Vec<_>>();
    let matches = |accession: &str, decoy_only: bool| {
        let upper = accession.to_ascii_uppercase();
        upper_patterns
            .iter()
            .any(|pattern| (!decoy_only || is_decoy_pattern(pattern)) && upper.contains(pattern))
    };
    is_contaminant_group(
        members,
        |accession| matches(accession, false),
        |accession| matches(accession, true),
    )
}

/// Drop contaminant members from a mixed group; `None` if nothing changes.
fn strip_contaminant_members(protein_group: &str, patterns: &[String]) -> Option<String> {
    let members = protein_group.split(';').collect::<Vec<_>>();
    if members.len() < 2 {
        return None;
    }
    let kept = members
        .iter()
        .copied()
        .filter(|member| !matches_protein_contaminant(member, patterns))
        .collect::<Vec<_>>();
    (!kept.is_empty() && kept.len() < members.len()).then(|| kept.join(";"))
}

/// `CONTAM_<ACC>` -> `<ACC>` when `<ACC>` is a non-contaminant entry covering all its peptides.
fn find_contaminant_twins(
    accession_peptides: &HashMap<String, HashSet<String>>,
) -> HashMap<String, String> {
    accession_peptides
        .iter()
        .filter(|(accession, _)| is_contaminant(accession))
        .filter_map(|(accession, peptides)| {
            let upper = accession.to_ascii_uppercase();
            let start = upper.find("CONTAM_")? + "CONTAM_".len();
            let target = accession.get(start..)?;
            let target_peptides = accession_peptides.get(target)?;
            (!is_contaminant(target) && peptides.is_subset(target_peptides))
                .then(|| (accession.clone(), target.to_owned()))
        })
        .collect()
}

fn rename_contaminant_twins(
    protein_group: &str,
    twins: &HashMap<String, String>,
) -> Option<String> {
    if twins.is_empty() {
        return None;
    }
    let mut changed = false;
    let mut members = Vec::<&str>::new();
    for member in protein_group.split(';') {
        let resolved = twins.get(member).map_or(member, |target| {
            changed = true;
            target.as_str()
        });
        if !members.contains(&resolved) {
            members.push(resolved);
        }
    }
    changed.then(|| members.join(";"))
}

/// Remove contaminant accessions from a piBAQ digest; returns how many were dropped.
fn drop_contaminant_accessions(accession_peptides: &mut HashMap<String, HashSet<String>>) -> usize {
    let before = accession_peptides.len();
    accession_peptides.retain(|accession, _| !is_contaminant(accession));
    before - accession_peptides.len()
}

fn register_id<I>(registry: &mut StringIdRegistry<I>, value: &str, namespace: &str) -> Result<I>
where
    I: Copy + From<u32> + Into<u32>,
{
    registry
        .get_or_insert(value)
        .ok_or_else(|| invalid_input(format!("{namespace} id registry overflow")))
}

fn register_loading_context(
    registry: &mut HashMap<(u32, u32), u32>,
    key: (u32, u32),
) -> Result<u32> {
    if let Some(id) = registry.get(&key) {
        return Ok(*id);
    }
    let id = u32::try_from(registry.len())
        .map_err(|_| invalid_input("loading context id registry overflow"))?;
    registry.insert(key, id);
    Ok(id)
}

fn register_contextual_peptide(
    registry: &mut HashMap<(PeptideId, u32), PeptideId>,
    key: (PeptideId, u32),
    namespace: &str,
) -> Result<PeptideId> {
    if let Some(id) = registry.get(&key) {
        return Ok(*id);
    }
    let id = PeptideId::new(
        u32::try_from(registry.len())
            .map_err(|_| invalid_input(format!("{namespace} id registry overflow")))?,
    );
    registry.insert(key, id);
    Ok(id)
}

/// Per-canonical monoisotopic molecular weights for the piBAQ TPA column.
///
/// Mirrors `mokume.io.fasta.digest_fasta_full(..., compute_mw=True)`: every
/// FASTA entry's MW is computed from its non-standard-stripped sequence, then
/// keyed by the canonical (isoform-collapsed) accession with first-seen
/// priority (the canonical entry's own MW wins because FASTA order is
/// preserved). Only consumed by `peptides2protein --tpa`.
fn load_fasta_mw(fasta: &Path) -> Result<HashMap<String, f64>> {
    let contents = read_to_string(fasta).map_err(|source| MokumeError::Io {
        path: fasta.to_path_buf(),
        source,
    })?;
    let mut mw_map = HashMap::<String, f64>::new();
    for (identifier, sequence) in parse_fasta_entries(&contents) {
        let accession = canonicalize_isoform(&parse_protein_accession(&identifier));
        let weight = protein_mono_weight(&strip_nonstandard_amino_acids(&sequence));
        mw_map.entry(accession).or_insert(weight);
    }
    Ok(mw_map)
}

fn parse_fasta_entries(contents: &str) -> Vec<(String, String)> {
    let mut entries = Vec::new();
    let mut current_identifier: Option<String> = None;
    let mut current_sequence = String::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('>') {
            if let Some(identifier) = current_identifier.replace(fasta_identifier(header)) {
                entries.push((identifier, std::mem::take(&mut current_sequence)));
            }
        } else if current_identifier.is_some() {
            current_sequence.push_str(line);
        }
    }
    if let Some(identifier) = current_identifier {
        entries.push((identifier, current_sequence));
    }
    entries
}

fn fasta_identifier(header: &str) -> String {
    header
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned()
}

fn strip_nonstandard_amino_acids(sequence: &str) -> String {
    sequence
        .chars()
        .filter_map(|aa| {
            let aa = aa.to_ascii_uppercase();
            (!matches!(aa, 'X' | 'B' | 'Z' | 'J' | 'U' | 'O')).then_some(aa)
        })
        .collect()
}

fn canonicalize_isoform(accession: &str) -> String {
    let Some((base, suffix)) = accession.rsplit_once('-') else {
        return accession.to_owned();
    };
    if !base.is_empty() && suffix.chars().all(|character| character.is_ascii_digit()) {
        base.to_owned()
    } else {
        accession.to_owned()
    }
}

/// Monoisotopic mass of water (`H2O`), the OpenMS `EmpiricalFormula("H2O")`
/// mono weight added once per peptide/protein by `AASequence::getMonoWeight`.
const WATER_MONO_MASS: f64 = 18.0105650638;

/// Per-residue monoisotopic masses (the internal residue masses pyOpenMS
/// returns from `Residue::getMonoWeight(Internal)`), used to reproduce
/// `AASequence::getMonoWeight` for the piBAQ TPA `MolecularWeight` column.
/// The full-protein mono weight is `WATER_MONO_MASS + sum(residue masses)`.
const fn residue_mono_mass(residue: char) -> Option<f64> {
    let mass = match residue {
        'A' => 71.03711415949999,
        'C' => 103.00918488949999,
        'D' => 115.02694415949999,
        'E' => 129.0425942233,
        'F' => 147.06841428710004,
        'G' => 57.0214640957,
        'H' => 137.0589122233,
        'I' => 113.0840643509,
        'K' => 128.09496338280002,
        'L' => 113.0840643509,
        'M' => 131.04048501709997,
        'N' => 114.04292819140001,
        'P' => 97.05276422329999,
        'Q' => 128.05857825520002,
        'R' => 156.1011113828,
        'S' => 87.0320291595,
        'T' => 101.04767922330001,
        'V' => 99.0684142871,
        'W' => 186.079313319,
        'Y' => 163.06332928710003,
        _ => return None,
    };
    Some(mass)
}

/// Monoisotopic molecular weight of a protein sequence, matching the pyOpenMS
/// `AASequence.fromString(seq).getMonoWeight()` value the Python piBAQ TPA path
/// reads via `digest_fasta_full(..., compute_mw=True)`. The accumulation order
/// (water first, then residues left to right) mirrors how OpenMS sums the
/// empirical formula so the result is bit-for-bit comparable within 1e-9 for
/// the peptide/protein lengths this path handles. Residues outside the
/// 20 standard amino acids are skipped, consistent with the upstream
/// `_strip_nonstandard_aa` applied before the digest.
fn protein_mono_weight(sequence: &str) -> f64 {
    let mut total = WATER_MONO_MASS;
    for residue in sequence.chars() {
        if let Some(mass) = residue_mono_mass(residue) {
            total += mass;
        }
    }
    total
}

fn invert_peptide_index(
    accession_peptides: &HashMap<String, HashSet<String>>,
) -> HashMap<String, HashSet<String>> {
    let mut peptide_accessions = HashMap::<String, HashSet<String>>::new();
    for (accession, peptides) in accession_peptides {
        for peptide in peptides {
            peptide_accessions
                .entry(peptide.clone())
                .or_default()
                .insert(accession.clone());
        }
    }
    peptide_accessions
}

fn discover_families(
    accession_peptides: &HashMap<String, HashSet<String>>,
    peptide_accessions: &HashMap<String, HashSet<String>>,
    min_shared: usize,
) -> Result<Vec<ProteinFamily>> {
    if min_shared == 0 {
        return Err(invalid_input("pibaq-min-shared must be greater than 0"));
    }

    let mut pair_counts = HashMap::<(String, String), usize>::new();
    for accessions in peptide_accessions.values() {
        if accessions.len() < 2 {
            continue;
        }
        let mut accessions = accessions.iter().cloned().collect::<Vec<_>>();
        accessions.sort();
        for left_index in 0..accessions.len() {
            for right_index in (left_index + 1)..accessions.len() {
                *pair_counts
                    .entry((
                        accessions[left_index].clone(),
                        accessions[right_index].clone(),
                    ))
                    .or_insert(0) += 1;
            }
        }
    }

    let mut adjacency = accession_peptides
        .keys()
        .map(|accession| (accession.clone(), HashSet::<String>::new()))
        .collect::<HashMap<_, _>>();
    for ((left, right), count) in pair_counts {
        if count >= min_shared {
            adjacency
                .entry(left.clone())
                .or_default()
                .insert(right.clone());
            adjacency.entry(right).or_default().insert(left);
        }
    }

    let mut seen = HashSet::<String>::new();
    let mut families = Vec::new();
    for start in adjacency.keys() {
        if seen.contains(start) {
            continue;
        }
        let mut stack = vec![start.clone()];
        let mut members = Vec::new();
        while let Some(accession) = stack.pop() {
            if !seen.insert(accession.clone()) {
                continue;
            }
            members.push(accession.clone());
            if let Some(neighbors) = adjacency.get(&accession) {
                stack.extend(
                    neighbors
                        .iter()
                        .filter(|neighbor| !seen.contains(*neighbor))
                        .cloned(),
                );
            }
        }
        members.sort();
        if members.is_empty() {
            continue;
        }
        let family_id = choose_family_id(&members, accession_peptides);
        families.push(ProteinFamily { family_id, members });
    }
    Ok(families)
}

fn choose_family_id(
    members: &[String],
    accession_peptides: &HashMap<String, HashSet<String>>,
) -> String {
    members
        .iter()
        .max_by(|left, right| {
            let left_key = (
                accession_peptides.get(*left).map_or(0, HashSet::len),
                left.as_str(),
            );
            let right_key = (
                accession_peptides.get(*right).map_or(0, HashSet::len),
                right.as_str(),
            );
            left_key.cmp(&right_key)
        })
        .cloned()
        .unwrap_or_default()
}

fn load_family_overrides(path: &Path) -> Result<Vec<ProteinFamily>> {
    let contents = read_to_string(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut families = Vec::new();
    let mut current_name: Option<String> = None;
    for line in contents.lines() {
        let trimmed = line.trim();
        let name_value = trimmed
            .strip_prefix("- name:")
            .or_else(|| trimmed.strip_prefix("name:"));
        if let Some(name) = name_value {
            current_name = Some(unquote_yaml_scalar(name.trim()));
            continue;
        }
        if let Some(members) = trimmed.strip_prefix("members:") {
            let Some(name) = current_name.take() else {
                return Err(invalid_input(format!(
                    "family override `{}` has members without a name",
                    path.display()
                )));
            };
            let members = parse_yaml_members(members)?;
            if members.is_empty() {
                return Err(invalid_input(format!(
                    "family override `{name}` has no members"
                )));
            }
            families.push(ProteinFamily {
                family_id: name,
                members,
            });
        }
    }
    if let Some(name) = current_name {
        return Err(invalid_input(format!(
            "family override `{name}` is missing members"
        )));
    }
    Ok(families)
}

fn parse_yaml_members(raw: &str) -> Result<Vec<String>> {
    let raw = raw.trim();
    let Some(raw) = raw
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
    else {
        return Err(invalid_input(
            "family override members must use inline list syntax, e.g. members: [P1, P2]",
        ));
    };
    let mut members = raw
        .split(',')
        .map(|member| unquote_yaml_scalar(member.trim()))
        .filter(|member| !member.is_empty())
        .collect::<Vec<_>>();
    members.sort();
    members.dedup();
    Ok(members)
}

fn unquote_yaml_scalar(value: &str) -> String {
    value.trim_matches('"').trim_matches('\'').trim().to_owned()
}

fn merge_family_overrides(
    auto_families: Vec<ProteinFamily>,
    overrides: Vec<ProteinFamily>,
) -> Vec<ProteinFamily> {
    if overrides.is_empty() {
        return auto_families;
    }
    let pinned = overrides
        .iter()
        .flat_map(|family| family.members.iter().cloned())
        .collect::<HashSet<_>>();
    let mut merged = overrides;
    for family in auto_families {
        let original_len = family.members.len();
        let remaining = family
            .members
            .into_iter()
            .filter(|member| !pinned.contains(member))
            .collect::<Vec<_>>();
        if remaining.is_empty() {
            continue;
        }
        let family_id = if remaining.len() == original_len {
            family.family_id
        } else {
            remaining.first().cloned().unwrap_or_default()
        };
        merged.push(ProteinFamily {
            family_id,
            members: remaining,
        });
    }
    merged
}

fn count_unique_anchors(
    observed_peptides: &HashSet<String>,
    peptide_accessions: &HashMap<String, HashSet<String>>,
) -> HashMap<String, usize> {
    let mut counts = HashMap::<String, usize>::new();
    for peptide in observed_peptides {
        let Some(accessions) = peptide_accessions.get(peptide) else {
            continue;
        };
        if accessions.len() == 1 {
            if let Some(accession) = accessions.iter().next() {
                *counts.entry(accession.clone()).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn assign_peptides_to_owning_family(
    families: &[ProteinFamily],
    peptide_accessions: &HashMap<String, HashSet<String>>,
    anchor_counts: &HashMap<String, usize>,
) -> HashMap<String, String> {
    let member_to_family = families
        .iter()
        .flat_map(|family| {
            family
                .members
                .iter()
                .map(|member| (member.clone(), family.family_id.clone()))
        })
        .collect::<HashMap<_, _>>();
    let mut owners = HashMap::new();
    for (peptide, accessions) in peptide_accessions {
        let mut best: Option<(usize, String)> = None;
        for accession in accessions {
            let Some(family_id) = member_to_family.get(accession) else {
                continue;
            };
            let score = (
                *anchor_counts.get(accession).unwrap_or(&0),
                family_id.clone(),
            );
            if best.as_ref().is_none_or(|current| {
                score.0 > current.0 || (score.0 == current.0 && score.1 < current.1)
            }) {
                best = Some(score);
            }
        }
        if let Some((_, family_id)) = best {
            owners.insert(peptide.clone(), family_id);
        }
    }
    owners
}

fn invert_peptide_ownership(
    peptide_owner: &HashMap<String, String>,
) -> HashMap<String, HashSet<String>> {
    let mut family_peptides = HashMap::<String, HashSet<String>>::new();
    for (peptide, owner) in peptide_owner {
        family_peptides
            .entry(owner.clone())
            .or_default()
            .insert(peptide.clone());
    }
    family_peptides
}

/// One piBAQ output cell: the allocated numerator (`norm_intensity`, the
/// per-(protein, sample) shared-aware intensity that the piBAQ allocator sums)
/// and the piBAQ value (numerator divided by the owned theoretical peptide
/// count). Keeping both lets `peptides2protein` reproduce the Python
/// `peptides_to_protein` long-format `NormIntensity` + `PiBAQ` columns while
/// `features2proteins` projects out only the piBAQ value.
#[derive(Debug, Clone, Copy)]
struct PibaqCell {
    norm_intensity: f64,
    pibaq: f64,
}

fn finalize_family_allocation(
    family: &ProteinFamily,
    observations: &[(String, SampleId, f64)],
    owned_peptides: &HashSet<String>,
    accession_peptides: &HashMap<String, HashSet<String>>,
    peptide_accessions: &HashMap<String, HashSet<String>>,
    force_equal_shared: bool,
    proteins: &mut StringIdRegistry<ProteinId>,
) -> HashMap<CellKey, PibaqCell> {
    let mut output = HashMap::new();
    let member_set = family.members.iter().cloned().collect::<HashSet<_>>();
    let mut peptide_members = HashMap::<String, Vec<String>>::new();
    for (peptide, _, _) in observations {
        let members = peptide_accessions
            .get(peptide)
            .map(|accessions| {
                let mut members = accessions
                    .intersection(&member_set)
                    .cloned()
                    .collect::<Vec<_>>();
                members.sort();
                members
            })
            .unwrap_or_default();
        if !members.is_empty() {
            peptide_members.entry(peptide.clone()).or_insert(members);
        }
    }

    let mut anchor_intensity = HashMap::<(String, SampleId), f64>::new();
    let mut combined = HashMap::<(String, SampleId), f64>::new();
    for (peptide, sample, intensity) in observations {
        let Some(members) = peptide_members.get(peptide) else {
            continue;
        };
        if members.len() == 1 {
            let key = (members[0].clone(), *sample);
            *anchor_intensity.entry(key.clone()).or_insert(0.0) += *intensity;
            *combined.entry(key).or_insert(0.0) += *intensity;
        }
    }

    for (peptide, sample, intensity) in observations {
        let Some(members) = peptide_members.get(peptide) else {
            continue;
        };
        if members.len() <= 1 {
            continue;
        }
        let weights = members
            .iter()
            .map(|member| {
                anchor_intensity
                    .get(&(member.clone(), *sample))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        let weight_sum = weights.iter().sum::<f64>();
        for (member, weight) in members.iter().zip(weights) {
            let allocated = if weight_sum > 0.0 && !force_equal_shared {
                *intensity * weight / weight_sum
            } else {
                *intensity / members.len() as f64
            };
            *combined.entry((member.clone(), *sample)).or_insert(0.0) += allocated;
        }
    }

    let denominators = family
        .members
        .iter()
        .map(|member| {
            let denominator = accession_peptides
                .get(member)
                .map(|peptides| peptides.intersection(owned_peptides).count())
                .unwrap_or(0);
            (member.clone(), denominator)
        })
        .collect::<HashMap<_, _>>();
    for ((member, sample), intensity) in combined {
        let denominator = denominators.get(&member).copied().unwrap_or(0);
        if denominator > 0 {
            insert_pibaq_value(
                &member,
                sample,
                intensity,
                intensity / denominator as f64,
                proteins,
                &mut output,
            );
        }
    }
    output
}

fn insert_pibaq_value(
    member: &str,
    sample: SampleId,
    norm_intensity: f64,
    value: f64,
    proteins: &mut StringIdRegistry<ProteinId>,
    output: &mut HashMap<CellKey, PibaqCell>,
) {
    if !value.is_finite() || value <= 0.0 {
        return;
    }
    if let Some(protein) = proteins.get_or_insert(member) {
        output.insert(
            CellKey { protein, sample },
            PibaqCell {
                norm_intensity,
                pibaq: value,
            },
        );
    }
}

fn parse_ratio_fraction_merge(value: &str) -> Result<RatioFractionMerge> {
    match value.trim().to_ascii_lowercase().as_str() {
        "mean" => Ok(RatioFractionMerge::Mean),
        "max" => Ok(RatioFractionMerge::Max),
        _ => Err(MokumeError::NotImplemented {
            stage: "ratio-fraction-merge",
        }),
    }
}

fn detect_reference_samples(raw: &SdrfRawTable, reference_regex: &str) -> Result<Vec<String>> {
    let sample_col = raw
        .column_index("source name")
        .ok_or_else(|| invalid_input("SDRF reference detection requires a `source name` column"))?;
    let scan_cols = raw
        .headers()
        .iter()
        .enumerate()
        .filter_map(|(index, header)| {
            (header.starts_with("factor value[") || header.starts_with("characteristics["))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let pattern = RegexBuilder::new(reference_regex)
        .case_insensitive(true)
        .build()
        .map_err(|source| {
            invalid_input(format!(
                "invalid IRS reference regex `{reference_regex}`: {source}"
            ))
        })?;
    let mut samples = Vec::new();
    for row in 0..raw.row_count() {
        if scan_cols
            .iter()
            .any(|&column| pattern.is_match(raw.cell(row, column)))
        {
            samples.push(raw.cell(row, sample_col));
        }
    }
    Ok(sorted_unique(samples.into_iter()))
}

/// Detect reference samples for ratio quantification, mirroring Python's
/// `LoadingStage.load_for_ratio` priority:
///   1. repeated `--irs-reference-sample`,
///   2. an explicitly changed reference regex,
///   3. `characteristics[pooled sample]` autodetection,
///   4. the default regex across every factor/characteristic column.
fn resolve_ratio_reference_samples(
    sdrf: &SdrfTable,
    raw: &SdrfRawTable,
    config: &FeatureToProteinsConfig,
) -> Result<Vec<String>> {
    if let Some(samples) = &config.irs.reference_samples {
        return Ok(sorted_unique(samples.iter().map(String::as_str)));
    }
    if config.irs.reference_regex != DEFAULT_REFERENCE_REGEX {
        return detect_reference_samples(raw, &config.irs.reference_regex);
    }
    let pooled = detect_pooled_reference_samples(sdrf);
    if !pooled.is_empty() {
        return Ok(pooled);
    }
    detect_reference_samples(raw, DEFAULT_REFERENCE_REGEX)
}

/// Reproduce Python's `detect_pooled_from_sdrf`: a sample is a reference channel
/// when its `characteristics[pooled sample]` value is `pooled` or begins with
/// `SN=` (a pooled-member list). Returns an empty vector when the column is
/// absent or no sample qualifies, so callers fall through to other detectors.
fn detect_pooled_reference_samples(sdrf: &SdrfTable) -> Vec<String> {
    let mut samples = sdrf
        .records()
        .iter()
        .filter(|record| {
            record.pooled_sample.as_deref().is_some_and(|value| {
                let value = value.trim().to_ascii_lowercase();
                value == "pooled" || value.starts_with("sn=")
            })
        })
        .map(|record| record.sample_accession.clone())
        .collect::<Vec<_>>();
    samples.sort();
    samples.dedup();
    samples
}

fn resolve_irs_reference_samples(
    sdrf: &SdrfTable,
    raw: &SdrfRawTable,
    config: &IrsConfig,
) -> Result<Vec<String>> {
    if let Some(samples) = &config.reference_samples {
        return Ok(sorted_unique(samples.iter().map(String::as_str)));
    }

    if let (Some(column), Some(values)) = (&config.sdrf_column, &config.sdrf_values) {
        return detect_reference_samples_by_sdrf_column(raw, column, values);
    }
    if config.reference_regex != DEFAULT_REFERENCE_REGEX {
        return detect_reference_samples(raw, &config.reference_regex);
    }
    let pooled = detect_pooled_reference_samples(sdrf);
    if !pooled.is_empty() {
        return Ok(pooled);
    }
    detect_reference_samples(raw, DEFAULT_REFERENCE_REGEX)
}

fn detect_reference_samples_by_sdrf_column(
    raw: &SdrfRawTable,
    column: &str,
    values: &[String],
) -> Result<Vec<String>> {
    let accepted = values
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<HashSet<_>>();
    if accepted.is_empty() {
        return Ok(Vec::new());
    }
    let requested = column.trim().to_ascii_lowercase();
    let value_col = raw.column_index(&requested).ok_or_else(|| {
        invalid_input(format!(
            "IRS SDRF column `{column}` was not found; available columns: {}",
            raw.headers().join(", ")
        ))
    })?;
    let sample_col = raw
        .column_index("source name")
        .ok_or_else(|| invalid_input("SDRF reference detection requires a `source name` column"))?;
    let samples = (0..raw.row_count())
        .filter(|&row| accepted.contains(&raw.cell(row, value_col).trim().to_ascii_lowercase()))
        .map(|row| raw.cell(row, sample_col));
    Ok(sorted_unique(samples))
}

fn sorted_unique<'a>(values: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut values = values
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn ratio_valid_proteins(records: &[RatioPsm], min_unique_peptides: usize) -> HashSet<ProteinId> {
    let mut peptides = HashMap::<ProteinId, HashSet<PeptideId>>::new();
    for record in records {
        peptides
            .entry(record.protein)
            .or_default()
            .insert(record.peptide);
    }
    peptides
        .into_iter()
        .filter_map(|(protein, peptides)| {
            (peptides.len() >= min_unique_peptides).then_some(protein)
        })
        .collect()
}

fn average_ratio_fractions(
    records: &[RatioPsm],
    valid_proteins: &HashSet<ProteinId>,
    fraction_merge: RatioFractionMerge,
) -> Vec<(ProteinId, PeptideId, i32, SampleId, f64)> {
    let mut grouped = HashMap::<(ProteinId, PeptideId, i32, SampleId), Vec<f64>>::new();
    for record in records {
        if valid_proteins.contains(&record.protein) {
            grouped
                .entry((record.protein, record.peptide, record.charge, record.sample))
                .or_default()
                .push(record.intensity);
        }
    }
    grouped
        .into_iter()
        .filter_map(|((protein, peptide, charge, sample), values)| {
            merge_ratio_values(values, fraction_merge)
                .map(|intensity| (protein, peptide, charge, sample, intensity))
        })
        .collect()
}

fn merge_ratio_values(mut values: Vec<f64>, fraction_merge: RatioFractionMerge) -> Option<f64> {
    values.retain(|value| value.is_finite() && *value > 0.0);
    if values.is_empty() {
        return None;
    }
    match fraction_merge {
        RatioFractionMerge::Mean => Some(values.iter().sum::<f64>() / values.len() as f64),
        RatioFractionMerge::Max => values.into_iter().max_by(f64::total_cmp),
    }
}

fn ratio_reference_intensities(
    averaged: &[(ProteinId, PeptideId, i32, SampleId, f64)],
    sample_names: &HashMap<SampleId, String>,
    reference_samples: &HashSet<String>,
    sample_to_plex: &HashMap<String, String>,
) -> HashMap<(ProteinId, PeptideId, i32, String), f64> {
    let mut grouped = HashMap::<(ProteinId, PeptideId, i32, String), Vec<f64>>::new();
    for (protein, peptide, charge, sample, intensity) in averaged {
        let Some(sample_name) = sample_names.get(sample) else {
            continue;
        };
        if reference_samples.contains(sample_name) {
            grouped
                .entry((
                    *protein,
                    *peptide,
                    *charge,
                    plex_of(sample_to_plex, sample_name),
                ))
                .or_default()
                .push(*intensity);
        }
    }
    grouped
        .into_iter()
        .filter_map(|(key, values)| mean_finite(values).map(|value| (key, value)))
        .collect()
}

fn ratio_peptide_log2_ratios(
    averaged: &[(ProteinId, PeptideId, i32, SampleId, f64)],
    sample_names: &HashMap<SampleId, String>,
    reference_samples: &HashSet<String>,
    sample_to_plex: &HashMap<String, String>,
    reference_intensity: &HashMap<(ProteinId, PeptideId, i32, String), f64>,
) -> HashMap<(ProteinId, PeptideId, SampleId), Vec<f64>> {
    let mut ratios = HashMap::<(ProteinId, PeptideId, SampleId), Vec<f64>>::new();
    for (protein, peptide, charge, sample, intensity) in averaged {
        let Some(sample_name) = sample_names.get(sample) else {
            continue;
        };
        if reference_samples.contains(sample_name) {
            continue;
        }
        let plex = plex_of(sample_to_plex, sample_name);
        let Some(reference) = reference_intensity.get(&(*protein, *peptide, *charge, plex)) else {
            continue;
        };
        if *intensity > 0.0 && *reference > 0.0 {
            let ratio = (*intensity / *reference).log2();
            if ratio.is_finite() {
                ratios
                    .entry((*protein, *peptide, *sample))
                    .or_default()
                    .push(ratio);
            }
        }
    }
    ratios
}

fn ratio_protein_medians(
    peptide_ratios: HashMap<(ProteinId, PeptideId, SampleId), Vec<f64>>,
) -> HashMap<CellKey, f64> {
    let mut protein_ratios = HashMap::<CellKey, Vec<f64>>::new();
    for ((protein, _, sample), mut values) in peptide_ratios {
        if let Some(value) = median_finite(&mut values) {
            protein_ratios
                .entry(CellKey { protein, sample })
                .or_default()
                .push(value);
        }
    }
    protein_ratios
        .into_iter()
        .filter_map(|(cell, mut values)| median_finite(&mut values).map(|value| (cell, value)))
        .collect()
}

fn mean_finite(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite() && *value > 0.0);
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn sample_plex(sample_name: &str) -> String {
    let Some((prefix, suffix)) = sample_name.rsplit_once('_') else {
        return "plex1".to_owned();
    };
    if suffix.chars().all(|character| character.is_ascii_digit()) && !prefix.is_empty() {
        prefix.to_owned()
    } else {
        "plex1".to_owned()
    }
}

/// Resolve a sample's plex from the SDRF-driven `sample_to_plex` map, the
/// authoritative source mirroring Python's `detect_plexes_from_sdrf`. Samples
/// absent from the map (e.g. a QPX run with no SDRF row, where `sample_name`
/// falls back to the run file) reuse `sample_plex`, which is exactly the value
/// `detect_plexes_from_sdrf` would assign that name.
fn plex_of(sample_to_plex: &HashMap<String, String>, sample_name: &str) -> String {
    sample_to_plex
        .get(sample_name)
        .cloned()
        .unwrap_or_else(|| sample_plex(sample_name))
}

fn sample_to_plex(sdrf: &SdrfTable) -> HashMap<String, String> {
    sdrf.records()
        .iter()
        .map(|record| {
            (
                record.sample_accession.clone(),
                sample_plex(&record.sample_accession),
            )
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlexSource {
    Column,
    DataFiles,
    SourceName,
}

/// Sample -> plex: SDRF column, else shared data files, else legacy source-name rule.
fn derive_sample_plexes(
    sdrf: &SdrfTable,
    raw_sdrf: Option<&SdrfRawTable>,
    plex_column: Option<&str>,
) -> Result<(HashMap<String, String>, PlexSource)> {
    let (plexes, source) = if let Some(column) = plex_column {
        let raw = raw_sdrf.ok_or_else(|| {
            invalid_input("--irs-plex-column requires readable raw SDRF metadata")
        })?;
        (plexes_from_column(raw, column)?, PlexSource::Column)
    } else if let Some(plexes) = plexes_from_data_files(sdrf) {
        (plexes, PlexSource::DataFiles)
    } else {
        (sample_to_plex(sdrf), PlexSource::SourceName)
    };
    let mut channels = BTreeMap::<&str, usize>::new();
    for plex in plexes.values() {
        *channels.entry(plex.as_str()).or_default() += 1;
    }
    info!(
        plexes = channels.len(),
        source = ?source,
        channels_per_plex = ?channels,
        "TMT plex assignment"
    );
    Ok((plexes, source))
}

fn plexes_from_column(raw: &SdrfRawTable, column: &str) -> Result<HashMap<String, String>> {
    let requested = column.trim().to_ascii_lowercase();
    let value_col = raw.column_index(&requested).ok_or_else(|| {
        invalid_input(format!(
            "IRS plex column `{column}` was not found; available columns: {}",
            raw.headers().join(", ")
        ))
    })?;
    let sample_col = raw
        .column_index("source name")
        .ok_or_else(|| invalid_input("--irs-plex-column requires a `source name` column"))?;
    let mut plexes = HashMap::<String, String>::new();
    for row in 0..raw.row_count() {
        let sample = raw.cell(row, sample_col).trim();
        let plex = raw.cell(row, value_col).trim();
        if sample.is_empty() || plex.is_empty() {
            continue;
        }
        if let Some(previous) = plexes.insert(sample.to_owned(), plex.to_owned()) {
            if previous != plex {
                return Err(invalid_input(format!(
                    "sample `{sample}` has more than one value in plex column `{column}`"
                )));
            }
        }
    }
    if plexes.is_empty() {
        return Err(invalid_input(format!(
            "IRS plex column `{column}` is empty"
        )));
    }
    Ok(plexes)
}

/// Samples with the same data-file set share a plex; `None` if not multiplexed.
fn plexes_from_data_files(sdrf: &SdrfTable) -> Option<HashMap<String, String>> {
    let mut files_by_sample = BTreeMap::<&str, BTreeSet<String>>::new();
    let mut samples_by_file = HashMap::<String, HashSet<&str>>::new();
    for record in sdrf.records() {
        let file = mokume_io::normalize_file_key(&record.data_file);
        if file.is_empty() {
            continue;
        }
        files_by_sample
            .entry(record.sample_accession.as_str())
            .or_default()
            .insert(file.clone());
        samples_by_file
            .entry(file)
            .or_default()
            .insert(record.sample_accession.as_str());
    }
    if !samples_by_file.values().any(|samples| samples.len() > 1) {
        return None;
    }
    let mut members = BTreeMap::<&BTreeSet<String>, Vec<&str>>::new();
    for (sample, files) in &files_by_sample {
        members.entry(files).or_default().push(sample);
    }
    // Keep the legacy source-name prefix as the plex id when it is unambiguous.
    let legacy = members
        .values()
        .map(|samples| {
            let prefixes = samples
                .iter()
                .map(|sample| sample_plex(sample))
                .collect::<BTreeSet<_>>();
            (prefixes.len() == 1).then(|| prefixes.into_iter().next().unwrap_or_default())
        })
        .collect::<Option<Vec<_>>>()
        .filter(|names| names.iter().collect::<HashSet<_>>().len() == names.len());
    let mut plexes = HashMap::new();
    for (index, samples) in members.values().enumerate() {
        let plex = legacy.as_ref().map_or_else(
            || format!("plex{}", index + 1),
            |names| names[index].clone(),
        );
        for sample in samples {
            plexes.insert((*sample).to_owned(), plex.clone());
        }
    }
    Some(plexes)
}

/// Number of distinct data-file sets across samples (an upper bound on plexes).
fn data_file_partitions(sdrf: &SdrfTable) -> usize {
    plexes_from_data_files(sdrf).map_or(1, |plexes| plexes.values().collect::<HashSet<_>>().len())
}

fn condition_by_sample(sdrf: &SdrfTable) -> HashMap<String, String> {
    sdrf.records()
        .iter()
        .filter_map(|record| {
            record
                .condition
                .as_ref()
                .map(|condition| (record.sample_accession.clone(), condition.clone()))
        })
        .collect()
}

fn is_reference_condition(condition: &str) -> bool {
    let condition = condition.to_ascii_lowercase();
    condition.contains("powder") || condition.contains("pool")
}

fn format_float(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        String::new()
    }
}

fn create_parent_dir(path: &Path) -> Result<()> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    create_dir_all(parent).map_err(|source| MokumeError::Io {
        path: parent.to_path_buf(),
        source,
    })
}

fn csv_error(path: &Path, source: csv::Error) -> MokumeError {
    invalid_input(format!("failed to write `{}`: {source}", path.display()))
}

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        fs,
        path::PathBuf,
    };

    use mokume_core::{
        BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig,
        FeatureToProteinsConfig, FilterConfig, ImputationConfig, InputConfig, IrsConfig,
        MaxLfqConfig, MokumeError, NormalizationConfig, OutputConfig, OutputFormat, PeptideId,
        PibaqConfig, ProteinId, QuantMethod, RatioConfig, RuntimeConfig, SampleId,
        StringIdRegistry,
    };
    use mokume_io::{QpxFeatureRecord, SdrfRawTable, SdrfTable};
    use mokume_normalization::SampleNormalizationMethod;

    use super::{
        batch_column_values_for_samples, detect_reference_samples, expand_de_contrasts_file,
        extract_sdrf_covariates, factorize_batch_labels, irs_by_mixture_scale_from_runs,
        irs_global_scale_from_runs, irs_mixture_first_token, irs_tech_replicate_of,
        irs_two_stage_scale_from_runs, load_normalization_proteins, match_sdrf_column,
        resolve_de_method, resolve_irs_autodetect_channel, resolve_irs_reference_samples,
        resolve_reference_batch, run_features_to_proteins, sum_peptide_values, tech_replicate_of,
        validate_batch_sizes, validate_combat_design, validate_features_to_proteins,
        validate_implemented_subset, CellKey, IrsStat, NormalizationFactorCollector, ProteinMatrix,
        ProteinValues,
    };

    #[test]
    fn peptide_sums_follow_stable_id_order() {
        let expected = HashMap::from([
            (PeptideId::new(0), 1.0),
            (PeptideId::new(1), 1.0),
            (PeptideId::new(2), 1.0e16),
        ]);
        let reordered = HashMap::from([
            (PeptideId::new(2), 1.0e16),
            (PeptideId::new(0), 1.0),
            (PeptideId::new(1), 1.0),
        ]);

        assert_eq!(
            sum_peptide_values(&expected),
            sum_peptide_values(&reordered)
        );
        assert_eq!(sum_peptide_values(&expected), (1.0 + 1.0) + 1.0e16);
    }

    #[test]
    fn pibaq_shared_allocation_is_exact_and_conservative() {
        let family = super::ProteinFamily {
            family_id: "A|B".to_owned(),
            members: vec!["A".to_owned(), "B".to_owned()],
        };
        let sample_anchored = SampleId::new(1);
        let sample_unanchored = SampleId::new(2);
        let observations = vec![
            ("unique_a".to_owned(), sample_anchored, 100.0),
            ("shared".to_owned(), sample_anchored, 300.0),
            ("shared".to_owned(), sample_unanchored, 200.0),
        ];
        let owned_peptides = HashSet::from(["unique_a".to_owned(), "shared".to_owned()]);
        let accession_peptides = HashMap::from([
            (
                "A".to_owned(),
                HashSet::from(["unique_a".to_owned(), "shared".to_owned()]),
            ),
            ("B".to_owned(), HashSet::from(["shared".to_owned()])),
        ]);
        let peptide_accessions = HashMap::from([
            ("unique_a".to_owned(), HashSet::from(["A".to_owned()])),
            (
                "shared".to_owned(),
                HashSet::from(["A".to_owned(), "B".to_owned()]),
            ),
        ]);
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let output = super::finalize_family_allocation(
            &family,
            &observations,
            &owned_peptides,
            &accession_peptides,
            &peptide_accessions,
            false,
            &mut proteins,
        );

        let Some(a) = proteins.get("A") else {
            panic!("A output missing");
        };
        let Some(b) = proteins.get("B") else {
            panic!("B output missing");
        };
        let cell = |protein, sample| output.get(&super::CellKey { protein, sample });

        assert!(cell(b, sample_anchored).is_none());
        assert_eq!(
            cell(a, sample_anchored).map(|value| value.norm_intensity),
            Some(400.0)
        );
        assert_eq!(
            cell(a, sample_unanchored).map(|value| value.norm_intensity),
            Some(100.0)
        );
        assert_eq!(
            cell(b, sample_unanchored).map(|value| value.norm_intensity),
            Some(100.0)
        );
        assert_eq!(
            super::classify_evidence(0, 1, 1, 3),
            super::PibaqEvidence::Medium
        );

        let mut equal_proteins = StringIdRegistry::<ProteinId>::new();
        let equal_output = super::finalize_family_allocation(
            &family,
            &observations,
            &owned_peptides,
            &accession_peptides,
            &peptide_accessions,
            true,
            &mut equal_proteins,
        );
        let Some(a) = equal_proteins.get("A") else {
            panic!("A equal output missing");
        };
        let Some(b) = equal_proteins.get("B") else {
            panic!("B equal output missing");
        };
        let norm_intensity = |protein, sample| {
            equal_output
                .get(&super::CellKey { protein, sample })
                .map(|value| value.norm_intensity)
        };
        assert_eq!(norm_intensity(a, sample_anchored), Some(250.0));
        assert_eq!(norm_intensity(b, sample_anchored), Some(150.0));
    }

    #[test]
    fn rejects_pibaq_without_fasta_before_loading() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("pibaq_without_fasta")?;
        let mut config = base_config(parquet);
        config.quantification = QuantMethod::Pibaq;

        let error = run_features_to_proteins(&config).err();

        assert_eq!(
            error.map(|error| error.to_string()).as_deref(),
            Some("invalid input: piBAQ quantification requires --fasta option")
        );
        Ok(())
    }

    #[test]
    fn rejects_ratio_without_sdrf_before_loading() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("ratio_without_sdrf")?;
        let mut config = base_config(parquet);
        config.quantification = QuantMethod::Ratio;

        let error = validate_features_to_proteins(&config).err();

        assert_eq!(
            error.map(|error| error.to_string()).as_deref(),
            Some("invalid input: Ratio quantification requires --sdrf option")
        );
        Ok(())
    }

    #[test]
    fn rejects_batch_column_without_column_name_before_loading(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("batch_column_without_name")?;
        let sdrf = existing_dummy_path("batch_column_without_name_sdrf")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(sdrf);
        config.batch.enabled = true;
        config.batch.method = "column".to_string();

        let error = validate_features_to_proteins(&config).err();

        assert_eq!(
            error.map(|error| error.to_string()).as_deref(),
            Some(
                "invalid input: Batch correction with method 'column' requires --batch-column option"
            )
        );
        Ok(())
    }

    #[test]
    fn accepts_normalization_proteins_file_for_supported_subset(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("normalization_proteins")?;
        let proteins = existing_dummy_path("normalization_proteins_list")?;
        fs::write(&proteins, "P1\nP2\n")?;
        let mut config = base_config(parquet);
        config.normalization.normalization_proteins = Some(proteins);

        validate_features_to_proteins(&config)?;
        validate_implemented_subset(&config)?;

        Ok(())
    }

    #[test]
    fn rejects_empty_normalization_proteins_file_before_loading(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let proteins = existing_dummy_path("empty_normalization_proteins_list")?;
        fs::write(&proteins, "\n")?;

        let error = load_normalization_proteins(&proteins).err();

        assert_eq!(
            error.map(|error| error.to_string()).as_deref(),
            Some(
                format!(
                    "invalid input: normalization proteins file `{}` does not contain protein IDs",
                    proteins.display()
                )
                .as_str()
            )
        );
        Ok(())
    }

    #[test]
    fn rejects_unmatched_normalization_proteins_after_loading(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut collector = NormalizationFactorCollector::new(
            None,
            Some(SampleNormalizationMethod::GlobalMedian),
            None,
            Some(HashSet::from(["P999".to_owned()])),
            &[],
        );

        collector.push(
            QpxFeatureRecord {
                sequence: "PEPTIDEAK".to_owned(),
                peptidoform: "PEPTIDEAK".to_owned(),
                charge: 2,
                run_file_name: "run1.raw".to_owned(),
                sample_accession: None,
                protein_accessions: vec!["P1".to_owned()],
                anchor_protein: None,
                unique: Some(true),
                is_decoy: Some(false),
                peptide_qvalue: None,
                pg_global_qvalue: None,
                selected_score: None,
                label: None,
                intensity: 100.0,
            },
            FilterConfig::default(),
            false,
        )?;

        assert!(
            !collector.has_sample_factor_values(),
            "unmatched normalization proteins must not feed sample factor values"
        );
        Ok(())
    }

    #[test]
    fn accepts_median_center_sample_normalization_subset() -> Result<(), Box<dyn std::error::Error>>
    {
        let parquet = existing_dummy_path("median_center_sample_normalization")?;
        let mut config = base_config(parquet);
        config.quantification = QuantMethod::Median;
        config.normalization.sample_method = "mediancenter".to_string();

        validate_implemented_subset(&config)?;

        Ok(())
    }

    // The IRS collect side uses Python's `split_part(run, '_', 2)` (second token),
    // while the apply side uses `run.str.split('_').str.get(-1)` (last token). They
    // agree for two-token names and diverge for three-or-more-token names; both
    // must be reproduced exactly (see `irs_tech_replicate_of` / `tech_replicate_of`).
    #[test]
    fn irs_and_apply_techreplicate_derivations_agree_for_two_token_runs() {
        for run in ["M1_3", "M2_2", "mixture_7"] {
            assert_eq!(
                irs_tech_replicate_of(run),
                Some(tech_replicate_of(run)),
                "two-token run {run} should match on both derivations",
            );
        }
    }

    #[test]
    fn irs_and_apply_techreplicate_derivations_diverge_for_three_token_runs() {
        // "plex_2_5": collect side -> split_part(_,2) = 2; apply side -> last = 5.
        assert_eq!(irs_tech_replicate_of("plex_2_5"), Some(2));
        assert_eq!(tech_replicate_of("plex_2_5"), 5);
        // A run with no underscore parses the whole name as the techreplicate.
        assert_eq!(irs_tech_replicate_of("4"), Some(4));
        assert_eq!(tech_replicate_of("4"), 4);
        // A non-integer second token yields no IRS key (Python's CAST would error).
        assert_eq!(irs_tech_replicate_of("M1_xx"), None);
    }

    // Locks the synthetic TMT oracle (also verified against Python's
    // `get_irs_scaling_factors`): reference-channel `irs_value` per run = 200 / 100
    // / 400 -> global_center = median(200,100,400) = 200 ->
    // scale = {techrep1: 1.0, techrep2: 2.0, techrep3: 0.5}.
    #[test]
    fn irs_global_scale_matches_synthetic_tmt_oracle_median() {
        let per_run = vec![
            (1_i64, vec![150.0_f64, 250.0]), // median 200
            (2_i64, vec![80.0, 120.0]),      // median 100
            (3_i64, vec![350.0, 450.0]),     // median 400
        ];
        let scale = irs_global_scale_from_runs(per_run, IrsStat::Median);
        assert_eq!(scale.get(&1).copied(), Some(1.0));
        assert_eq!(scale.get(&2).copied(), Some(2.0));
        assert_eq!(scale.get(&3).copied(), Some(0.5));
        assert_eq!(scale.len(), 3);
    }

    #[test]
    fn irs_global_scale_matches_synthetic_tmt_oracle_mean() {
        // Same per-run intensities; mean center = mean(200,100,400) = 233.333...
        let per_run = vec![
            (1_i64, vec![150.0_f64, 250.0]),
            (2_i64, vec![80.0, 120.0]),
            (3_i64, vec![350.0, 450.0]),
        ];
        let scale = irs_global_scale_from_runs(per_run, IrsStat::Mean);
        let center = (200.0 + 100.0 + 400.0) / 3.0;
        assert!((scale[&1] - center / 200.0).abs() < 1e-12);
        assert!((scale[&2] - center / 100.0).abs() < 1e-12);
        assert!((scale[&3] - center / 400.0).abs() < 1e-12);
    }

    #[test]
    fn irs_global_scale_drops_nonpositive_and_handles_empty() {
        // A run whose only intensities aggregate to a non-positive center is
        // dropped (Python's `irs_value > 0` filter); empty input yields no scale.
        assert!(irs_global_scale_from_runs(Vec::new(), IrsStat::Median).is_empty());
        let per_run = vec![(1_i64, vec![100.0_f64]), (2_i64, vec![0.0, 0.0])];
        let scale = irs_global_scale_from_runs(per_run, IrsStat::Median);
        // techrep 2 aggregates to 0 (median drops zeros -> None), so only techrep 1
        // survives; with one surviving run the center equals its own value -> 1.0.
        assert_eq!(scale.get(&1).copied(), Some(1.0));
        assert!(!scale.contains_key(&2));
    }

    // Multi-mixture synthetic TMT fixture shared by the by_mixture / two_stage
    // scope tests. Two mixtures, two techreplicates each (passed in the stable
    // run-name-sorted order the collect side produces):
    //   mixA_1 -> [100,200,300]  (median 200, mean 200)   techrep 1
    //   mixA_2 -> [ 50,150,250]  (median 150, mean 150)   techrep 2
    //   mixB_3 -> [400,500,600]  (median 500, mean 500)   techrep 3
    //   mixB_4 -> [ 80,120,400]  (median 120, mean 200)   techrep 4
    // Values are locked against Python `get_irs_scaling_factors` run on the
    // matching pyarrow fixture (see scratchpad oracle).
    fn multi_mixture_runs() -> Vec<(i64, String, Vec<f64>)> {
        vec![
            (1, "mixA".to_owned(), vec![100.0, 200.0, 300.0]),
            (2, "mixA".to_owned(), vec![50.0, 150.0, 250.0]),
            (3, "mixB".to_owned(), vec![400.0, 500.0, 600.0]),
            (4, "mixB".to_owned(), vec![80.0, 120.0, 400.0]),
        ]
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn irs_by_mixture_scale_matches_python_oracle_median() {
        // mixA center = median(200,150) = 175; mixB center = median(500,120) = 310.
        let scale = irs_by_mixture_scale_from_runs(multi_mixture_runs(), IrsStat::Median);
        assert_eq!(scale.len(), 4);
        assert_close(scale[&1], 175.0 / 200.0); // 0.875
        assert_close(scale[&2], 175.0 / 150.0); // 1.16666...
        assert_close(scale[&3], 310.0 / 500.0); // 0.62
        assert_close(scale[&4], 310.0 / 120.0); // 2.58333...
    }

    #[test]
    fn irs_by_mixture_scale_matches_python_oracle_mean() {
        // mixA center = mean(200,150) = 175; mixB center = mean(500,200) = 350.
        let scale = irs_by_mixture_scale_from_runs(multi_mixture_runs(), IrsStat::Mean);
        assert_eq!(scale.len(), 4);
        assert_close(scale[&1], 175.0 / 200.0); // 0.875
        assert_close(scale[&2], 175.0 / 150.0); // 1.16666...
        assert_close(scale[&3], 350.0 / 500.0); // 0.7
        assert_close(scale[&4], 350.0 / 200.0); // 1.75
    }

    #[test]
    fn irs_two_stage_scale_matches_python_oracle_median() {
        // Stage-1 centers: mixA 175, mixB 310. Global center over the distinct
        // mixture centers = median(175,310) = 242.5. Applied scale is the product
        // (stage1 * stage2), which equals global_center / irs_value.
        let scale = irs_two_stage_scale_from_runs(multi_mixture_runs(), IrsStat::Median);
        let global = 242.5;
        assert_eq!(scale.len(), 4);
        assert_close(scale[&1], (175.0 / 200.0) * (global / 175.0)); // 1.2125
        assert_close(scale[&2], (175.0 / 150.0) * (global / 175.0)); // 1.61666...
        assert_close(scale[&3], (310.0 / 500.0) * (global / 310.0)); // 0.485
        assert_close(scale[&4], (310.0 / 120.0) * (global / 310.0)); // 2.02083...
    }

    #[test]
    fn irs_two_stage_scale_matches_python_oracle_mean() {
        // Stage-1 centers: mixA 175, mixB 350. Global center = mean(175,350) = 262.5.
        let scale = irs_two_stage_scale_from_runs(multi_mixture_runs(), IrsStat::Mean);
        let global = 262.5;
        assert_eq!(scale.len(), 4);
        assert_close(scale[&1], (175.0 / 200.0) * (global / 175.0)); // 1.3125
        assert_close(scale[&2], (175.0 / 150.0) * (global / 175.0)); // 1.75
        assert_close(scale[&3], (350.0 / 500.0) * (global / 350.0)); // 0.525
        assert_close(scale[&4], (350.0 / 200.0) * (global / 350.0)); // 1.3125
    }

    #[test]
    fn irs_scope_functions_handle_empty_and_nonpositive() {
        assert!(irs_by_mixture_scale_from_runs(Vec::new(), IrsStat::Median).is_empty());
        assert!(irs_two_stage_scale_from_runs(Vec::new(), IrsStat::Median).is_empty());
        // A run aggregating to a non-positive center is dropped before the
        // mixture center is formed (Python's `irs_value > 0`).
        let runs = vec![
            (1_i64, "m".to_owned(), vec![100.0]),
            (2_i64, "m".to_owned(), vec![0.0, 0.0]),
        ];
        let by_mixture = irs_by_mixture_scale_from_runs(runs.clone(), IrsStat::Median);
        assert_eq!(by_mixture.get(&1).copied(), Some(1.0));
        assert!(!by_mixture.contains_key(&2));
        let two_stage = irs_two_stage_scale_from_runs(runs, IrsStat::Median);
        assert_eq!(two_stage.get(&1).copied(), Some(1.0));
        assert!(!two_stage.contains_key(&2));
    }

    #[test]
    fn irs_mixture_first_token_takes_first_underscore_segment() {
        assert_eq!(irs_mixture_first_token("mixA_1"), "mixA");
        assert_eq!(irs_mixture_first_token("plex_2_5"), "plex");
        assert_eq!(irs_mixture_first_token("noUnderscore"), "noUnderscore");
        assert_eq!(irs_mixture_first_token(""), "");
    }

    fn write_temp_sdrf(name: &str, contents: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = tempfile::Builder::new()
            .prefix(&format!(
                "mokume_irs_autodetect_{name}_{}_{}_",
                std::process::id(),
                unique_suffix()
            ))
            .tempdir()?
            .keep()
            .join("autodetect.sdrf.tsv");
        fs::write(&path, contents)?;
        Ok(path)
    }

    #[test]
    fn resolve_irs_autodetect_channel_picks_mode_label() -> Result<(), Box<dyn std::error::Error>> {
        // Three pooled rows: TMT131 appears twice, TMT130 once -> mode TMT131.
        // A non-pooled row must not vote.
        let sdrf = write_temp_sdrf(
            "pooled",
            "source name\tcomment[label]\tcharacteristics[pooled sample]\n\
             sample_pool_1\tTMT131\tpooled\n\
             sample_pool_2\tTMT131\tpooled\n\
             sample_pool_3\tTMT130\tpooled\n\
             sample_normal\tTMT126\tnot pooled\n",
        )?;
        let channel = resolve_irs_autodetect_channel(&sdrf, "pool")?;
        assert_eq!(channel.as_deref(), Some("TMT131"));
        fs::remove_file(&sdrf)?;
        Ok(())
    }

    #[test]
    fn resolve_irs_autodetect_channel_breaks_ties_lexicographically(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TMT127 and TMT131 each match once; pandas `mode().iloc[0]` returns the
        // smallest of the tied labels -> TMT127.
        let sdrf = write_temp_sdrf(
            "tie",
            "source name\tcomment[label]\n\
             pool_a\tTMT131\n\
             pool_b\tTMT127\n",
        )?;
        let channel = resolve_irs_autodetect_channel(&sdrf, "pool")?;
        assert_eq!(channel.as_deref(), Some("TMT127"));
        fs::remove_file(&sdrf)?;
        Ok(())
    }

    #[test]
    fn resolve_irs_autodetect_channel_returns_none_on_no_match(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // No source name matches the regex -> None (Python skips IRS).
        let sdrf = write_temp_sdrf(
            "nomatch",
            "source name\tcomment[label]\n\
             ordinary_1\tTMT126\n",
        )?;
        assert!(resolve_irs_autodetect_channel(&sdrf, "pool")?.is_none());
        fs::remove_file(&sdrf)?;
        // Missing comment[label] column -> None.
        let sdrf2 = write_temp_sdrf("nolabel", "source name\npool_1\n")?;
        assert!(resolve_irs_autodetect_channel(&sdrf2, "pool")?.is_none());
        fs::remove_file(&sdrf2)?;
        Ok(())
    }

    #[test]
    fn reference_detection_uses_real_regex_and_explicit_priority(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let text = concat!(
            "source name\tcomment[data file]\tfactor value[role]\tcharacteristics[pooled sample]\n",
            "S1\ta.raw\tordinary\tnot pooled\n",
            "S2\tb.raw\tBridge-42\tpooled\n",
            "S3\tc.raw\tbridge-X\tnot pooled\n",
        );
        let raw = SdrfRawTable::from_reader(text.as_bytes())?;
        let typed = SdrfTable::from_reader(text.as_bytes())?;
        assert_eq!(
            detect_reference_samples(&raw, r"^bridge-\d+$")?,
            vec!["S2".to_owned()]
        );
        assert!(detect_reference_samples(&raw, "[").is_err());

        let explicit = IrsConfig {
            reference_samples: Some(vec!["S1".to_owned()]),
            ..IrsConfig::default()
        };
        assert_eq!(
            resolve_irs_reference_samples(&typed, &raw, &explicit)?,
            vec!["S1".to_owned()]
        );

        let custom_regex = IrsConfig {
            reference_regex: "^ordinary$".to_owned(),
            ..IrsConfig::default()
        };
        assert_eq!(
            resolve_irs_reference_samples(&typed, &raw, &custom_regex)?,
            vec!["S1".to_owned()]
        );
        Ok(())
    }

    #[test]
    fn combat_design_rejects_batch_confounded_covariate() {
        let batch = [0, 0, 1, 1];
        let confounded = vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]];
        assert!(validate_combat_design(&batch, &confounded, None).is_err());
        let orthogonal = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];
        assert!(validate_combat_design(&batch, &orthogonal, None).is_ok());
    }

    #[test]
    fn accepts_directlfq_option_subset() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("directlfq_options")?;
        let mut config = base_config(parquet);
        config.quantification = QuantMethod::DirectLfq;
        config.directlfq.min_nonan = 2;
        config.directlfq.num_samples_quadratic = 10;

        validate_implemented_subset(&config)?;

        Ok(())
    }

    #[test]
    fn rejects_missing_normalization_proteins_file_before_loading(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("missing_normalization_proteins")?;
        let mut config = base_config(parquet);
        let missing_path = tempfile::Builder::new()
            .prefix(&format!(
                "mokume_missing_normalization_proteins_{}_{}_",
                std::process::id(),
                unique_suffix()
            ))
            .tempdir()?
            .keep()
            .join("missing.txt");
        config.normalization.normalization_proteins = Some(missing_path.clone());

        let error = validate_features_to_proteins(&config).err();

        assert_eq!(
            error.map(|error| error.to_string()).as_deref(),
            Some(format!("input file does not exist: {}", missing_path.display()).as_str())
        );
        Ok(())
    }

    #[test]
    fn accepts_quantile_sample_normalization_subset() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("quantile_sample_normalization")?;
        let mut config = base_config(parquet);
        config.normalization.sample_method = "quantile".to_string();

        validate_implemented_subset(&config)?;

        Ok(())
    }

    #[test]
    fn accepts_limma_de_subset() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("limma_de")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "limma".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(PathBuf::from("de.csv"));

        validate_implemented_subset(&config)?;
        Ok(())
    }

    #[test]
    fn accepts_rots_de_subset() -> Result<(), Box<dyn std::error::Error>> {
        // rots is a faithful (RNG-based) port and is now a supported method, so
        // a well-formed rots DE config must validate.
        let parquet = existing_dummy_path("rots_de")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "rots".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(PathBuf::from("de.csv"));

        validate_implemented_subset(&config)?;
        Ok(())
    }

    #[test]
    fn rejects_limma_de_without_sdrf() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("limma_de_no_sdrf")?;
        let mut config = base_config(parquet);
        config.differential_expression.enabled = true;
        config.differential_expression.method = "limma".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);

        let error = validate_implemented_subset(&config).err();
        assert!(matches!(error, Some(MokumeError::InvalidInput { .. })));
        Ok(())
    }

    #[test]
    fn rejects_limma_de_without_contrasts() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("limma_de_no_contrasts")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "limma".to_string();

        let error = validate_implemented_subset(&config).err();
        assert!(matches!(error, Some(MokumeError::InvalidInput { .. })));
        Ok(())
    }

    #[test]
    fn accepts_limrots_de_subset() -> Result<(), Box<dyn std::error::Error>> {
        // limrots is a faithful (RNG-based) port and is now a supported method, so
        // a well-formed limrots DE config must validate.
        let parquet = existing_dummy_path("limrots_de")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "limrots".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(PathBuf::from("de.csv"));

        validate_implemented_subset(&config)?;
        Ok(())
    }

    #[test]
    fn resolves_auto_de_method() -> Result<(), Box<dyn std::error::Error>> {
        // `auto` mirrors Python's `_resolve_de_method` (stages.py:1784): directlfq
        // quantification selects `deqms`, every other quantification selects
        // `limrots`. It validates (both targets are ported) and is resolved to the
        // concrete method just before the DE stage runs.
        let parquet = existing_dummy_path("de_method_auto")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "auto".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(PathBuf::from("de.csv"));

        config.quantification = QuantMethod::DirectLfq;
        validate_implemented_subset(&config)?;
        assert_eq!(resolve_de_method(&config), "deqms");

        config.quantification = QuantMethod::MaxLfq;
        validate_implemented_subset(&config)?;
        assert_eq!(resolve_de_method(&config), "limrots");
        Ok(())
    }

    #[test]
    fn expands_de_contrasts_file_appending_to_inline() -> Result<(), Box<dyn std::error::Error>> {
        // Mirror Python (features2proteins.py:768): a TSV with group1/group2
        // columns appends "<g1> vs <g2>" after the repeated --de-contrast entries,
        // empty rows are skipped, and contrasts_file is cleared so downstream sees
        // one resolved list.
        let parquet = existing_dummy_path("de_contrasts_file")?;
        let contrasts_file = existing_dummy_path("de_contrasts_file_tsv")?;
        fs::write(&contrasts_file, "group1\tgroup2\nTumor\tNormal\nA\tB\n\t\n")?;

        let mut config = base_config(parquet);
        config.differential_expression.contrasts = Some(vec!["X vs Y".to_string()]);
        config.differential_expression.contrasts_file = Some(contrasts_file);

        let resolved = expand_de_contrasts_file(&config)?;
        assert_eq!(
            resolved.differential_expression.contrasts,
            Some(vec![
                "X vs Y".to_string(),
                "Tumor vs Normal".to_string(),
                "A vs B".to_string(),
            ])
        );
        assert!(resolved.differential_expression.contrasts_file.is_none());
        Ok(())
    }

    #[test]
    fn rejects_de_contrasts_file_missing_group_columns() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("de_contrasts_bad")?;
        let contrasts_file = existing_dummy_path("de_contrasts_bad_tsv")?;
        fs::write(&contrasts_file, "groupA\tgroupB\nTumor\tNormal\n")?;

        let mut config = base_config(parquet);
        config.differential_expression.contrasts_file = Some(contrasts_file);

        assert!(matches!(
            expand_de_contrasts_file(&config),
            Err(MokumeError::InvalidInput { .. })
        ));
        Ok(())
    }

    #[test]
    fn run_lfq_from_peptides_estimates_and_filters_zeros() {
        let make = |protein: &str, peptide: &str, sample: &str, intensity: f64| {
            super::LfqPeptideObservation {
                protein: protein.to_owned(),
                peptide: peptide.to_owned(),
                sample: sample.to_owned(),
                intensity,
            }
        };
        // P1 and P2 each carry two peptides across two samples.
        let observations = vec![
            make("P1", "PEP1", "S1", 100.0),
            make("P1", "PEP2", "S1", 200.0),
            make("P1", "PEP1", "S2", 110.0),
            make("P1", "PEP2", "S2", 220.0),
            make("P2", "PEP3", "S1", 50.0),
            make("P2", "PEP4", "S1", 70.0),
            make("P2", "PEP3", "S2", 55.0),
            make("P2", "PEP4", "S2", 77.0),
        ];

        let result = super::run_lfq_from_peptides(&observations, 1, 50);
        let proteins: HashSet<&str> = result.iter().map(|row| row.protein.as_str()).collect();
        assert_eq!(proteins, ["P1", "P2"].into_iter().collect());
        assert!(result.iter().all(|row| row.intensity > 0.0));
        assert_eq!(result.len(), 4); // 2 proteins x 2 samples, zeros dropped

        // A min_nonan above the available ion count drops every protein.
        assert!(super::run_lfq_from_peptides(&observations, 100, 50).is_empty());
    }

    #[test]
    fn run_lfq_from_peptides_is_invariant_to_observation_order() {
        let make = |peptide: &str, sample: &str, intensity: f64| super::LfqPeptideObservation {
            protein: "A5Z2X5".to_owned(),
            peptide: peptide.to_owned(),
            sample: sample.to_owned(),
            intensity,
        };
        // Minimal real PXD002099 support pattern that exposed DirectLFQ's
        // encounter-order sensitivity: reversing these rows used to shift all
        // three reported cells by about 2% despite identical observations.
        let observations = vec![
            make("LRTDETLR", "Sample 1", 1_661_357.0),
            make("LRTDETLR", "Sample 2", 2_011_746.0),
            make("LRTDETLR", "Sample 3", 771_094.875),
            make("LTGNPELSSLDEVLAK", "Sample 1", 3_310_899.0),
            make("LTGNPELSSLDEVLAK", "Sample 2", 2_914_053.0),
            make("LTGNPELSSLDEVLAK", "Sample 3", 4_332_763.0),
            make("LTGNPELSSLDEVLAK", "Sample 4", 5_018_000.0),
            make("LTGNPELSSLDEVLAK", "Sample 5", 4_421_422.0),
        ];
        let forward = super::run_lfq_from_peptides(&observations, 2, 50)
            .into_iter()
            .map(|row| ((row.protein, row.sample), row.intensity))
            .collect::<HashMap<_, _>>();
        let mut reversed_observations = observations;
        reversed_observations.reverse();
        let reversed = super::run_lfq_from_peptides(&reversed_observations, 2, 50)
            .into_iter()
            .map(|row| ((row.protein, row.sample), row.intensity))
            .collect::<HashMap<_, _>>();

        assert_eq!(forward.len(), 3);
        assert_eq!(
            forward.keys().collect::<HashSet<_>>(),
            reversed.keys().collect()
        );
        for (cell, expected) in forward {
            let actual = reversed.get(&cell).copied().unwrap_or(f64::NAN);
            let tolerance = expected.abs().max(1.0) * 1e-12;
            assert!(
                (actual - expected).abs() <= tolerance,
                "order changed {cell:?}: forward={expected}, reversed={actual}"
            );
        }
    }

    #[test]
    fn sample_correlation_filter_excludes_only_the_original_matrix_outlier(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut samples = StringIdRegistry::<SampleId>::new();
        let sample_ids = ["A1", "A2", "A3", "B1", "B2"]
            .into_iter()
            .map(|name| samples.get_or_insert(name).ok_or("sample id"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let protein_ids = ["P1", "P2", "P3", "P4"]
            .into_iter()
            .map(|name| proteins.get_or_insert(name).ok_or("protein id"))
            .collect::<Result<Vec<_>, _>>()?;
        let columns = [
            [1.0, 2.0, 4.0, 8.0],
            [2.0, 4.0, 8.0, 16.0],
            [8.0, 4.0, 2.0, 1.0],
            [3.0, 6.0, 12.0, 24.0],
            [6.0, 12.0, 24.0, 48.0],
        ];
        let mut cells = HashMap::new();
        for (sample, values) in sample_ids.iter().zip(columns) {
            for (protein, value) in protein_ids.iter().zip(values) {
                cells.insert(
                    CellKey {
                        protein: *protein,
                        sample: *sample,
                    },
                    value,
                );
            }
        }
        let mut matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins: protein_ids.iter().copied().collect(),
            excluded_samples: HashSet::new(),
            peptide_counts: HashMap::new(),
            values: ProteinValues::Cells(cells),
        };
        let sdrf = SdrfTable::from_reader(
            b"source name\tcomment[data file]\tfactor value[group]\n\
A1\tA1.raw\tA\nA2\tA2.raw\tA\nA3\tA3.raw\tA\n\
B1\tB1.raw\tB\nB2\tB2.raw\tB\n"
                .as_slice(),
        )?;

        matrix.apply_sample_correlation_filter(Some(&sdrf), -0.5, false, false)?;

        let kept = matrix
            .sample_columns(false)
            .into_iter()
            .map(|(_, name)| name)
            .collect::<Vec<_>>();
        assert_eq!(kept, vec!["A1", "A2", "B1", "B2"]);
        assert_eq!(matrix.value(protein_ids[0], sample_ids[0]), Some(1.0));
        Ok(())
    }

    #[test]
    fn sample_correlation_uses_negative_values_for_log2_methods(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut samples = StringIdRegistry::<SampleId>::new();
        let left = samples.get_or_insert("A1").ok_or("left sample")?;
        let right = samples.get_or_insert("A2").ok_or("right sample")?;
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let protein_ids = ["P1", "P2", "P3"]
            .into_iter()
            .map(|name| proteins.get_or_insert(name).ok_or("protein id"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut cells = HashMap::new();
        for (protein, left_value, right_value) in protein_ids
            .iter()
            .zip([-2.0, 0.0, 2.0])
            .zip([-1.0, 1.0, 3.0])
            .map(|((protein, left_value), right_value)| (protein, left_value, right_value))
        {
            cells.insert(
                CellKey {
                    protein: *protein,
                    sample: left,
                },
                left_value,
            );
            cells.insert(
                CellKey {
                    protein: *protein,
                    sample: right,
                },
                right_value,
            );
        }
        let matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins: protein_ids.iter().copied().collect(),
            excluded_samples: HashSet::new(),
            peptide_counts: HashMap::new(),
            values: ProteinValues::Cells(cells),
        };

        let (log2_correlation, log2_overlap) =
            matrix.pairwise_sample_correlation(left, right, true);
        let (linear_correlation, linear_overlap) =
            matrix.pairwise_sample_correlation(left, right, false);
        assert_eq!(log2_overlap, 3);
        assert!(log2_correlation.is_some_and(|value| (value - 1.0).abs() < 1e-12));
        assert_eq!((linear_correlation, linear_overlap), (None, 1));
        Ok(())
    }

    #[test]
    fn factorize_batch_labels_uses_first_occurrence_order() {
        let values = ["B", "A", "B", "A", "C"].map(String::from);
        assert_eq!(factorize_batch_labels(&values), vec![0, 1, 0, 1, 2]);
    }

    #[test]
    fn remove_ids_matches_complete_accessions_not_substrings() {
        let accessions = ["P1", "sp|P10|protein ten"].map(str::to_owned);
        assert!(super::has_removed_accession(
            &accessions,
            &HashSet::from(["P1".to_owned()])
        ));
        assert!(!super::has_removed_accession(
            &["P10".to_owned()],
            &HashSet::from(["P1".to_owned()])
        ));
    }

    #[test]
    fn resolve_batch_method_rejects_unknown_values() {
        use mokume_stats::batch::BatchDetectionMethod;
        // Known names parse (case/separator-insensitive).
        assert_eq!(
            super::resolve_batch_method("Sample-Prefix").ok(),
            Some(BatchDetectionMethod::SamplePrefix)
        );
        assert_eq!(
            super::resolve_batch_method("COLUMN").ok(),
            Some(BatchDetectionMethod::ExplicitColumn)
        );
        assert!(super::resolve_batch_method("nonsense").is_err());
    }

    #[test]
    fn detect_batches_for_method_in_pipeline_flow() {
        use mokume_stats::batch::BatchDetectionMethod;
        let names = ["PXD001-S1", "PXD001-S2", "PXD002-S1", "PXD002-S2"];

        // sample_prefix factorizes the sample prefixes.
        let labels =
            super::detect_batches_for_method(BatchDetectionMethod::SamplePrefix, &names, None);
        assert_eq!(labels.ok(), Some(vec![0, 0, 1, 1]));

        // Without run_info, the pipeline never invokes `run` here (the caller
        // returns an error), but the helper still surfaces the `run_info`
        // requirement for `run` -- mirroring Python's `ValueError`.
        let run = super::detect_batches_for_method(BatchDetectionMethod::RunName, &names, None);
        assert!(run.is_err());

        // The `column` path validates the explicit value length.
        let mismatched = ["A".to_owned(), "B".to_owned()];
        let bad = super::detect_batches_for_method(
            BatchDetectionMethod::ExplicitColumn,
            &names,
            Some(&mismatched),
        );
        assert!(bad.is_err());
    }

    #[test]
    fn validate_batch_sizes_requires_two_batches_each_two_samples() {
        assert_eq!(
            validate_batch_sizes(vec![0, 0, 1, 1]),
            Some(vec![0, 0, 1, 1])
        );
        assert_eq!(validate_batch_sizes(vec![0, 0, 0]), None); // a single batch
        assert_eq!(validate_batch_sizes(vec![0, 0, 1]), None); // batch 1 has one sample
    }

    #[test]
    fn resolves_original_batch_label_before_numeric_compatibility() {
        let labels = ["0", "0", "batch-b", "batch-b"].map(str::to_owned);
        let encoded = [0, 0, 1, 1];
        assert_eq!(
            resolve_reference_batch("batch-b", &labels, &encoded).ok(),
            Some(1)
        );
        assert_eq!(
            resolve_reference_batch("0", &labels, &encoded).ok(),
            Some(0)
        );
        assert!(resolve_reference_batch("missing", &labels, &encoded).is_err());
    }

    fn covariate_fixture() -> Result<SdrfRawTable, Box<dyn std::error::Error>> {
        // Mirror the Python oracle fixture (`gen_oracle.py`).
        let input = concat!(
            "source name\tcharacteristics[sex]\tcharacteristics[organism part]\tbatch\tconst_col\n",
            "S1\tmale\tliver\tb1\tx\n",
            "S2\tfemale\tliver\tb1\tx\n",
            "S3\tmale\tbrain\tb2\tx\n",
            "S4\tfemale\tbrain\tb2\tx\n",
        );
        Ok(SdrfRawTable::from_reader(input.as_bytes())?)
    }

    #[test]
    fn match_sdrf_column_prefers_exact_then_substring() {
        let headers: Vec<String> = ["source name", "characteristics[sex]", "batch"]
            .map(String::from)
            .to_vec();
        // Exact lowercased match.
        assert_eq!(match_sdrf_column(&headers, "Characteristics[Sex]"), Some(1));
        // Substring: "sex" is contained in "characteristics[sex]".
        assert_eq!(match_sdrf_column(&headers, "sex"), Some(1));
        assert_eq!(match_sdrf_column(&headers, "missing"), None);
    }

    #[test]
    fn extract_sdrf_covariates_matches_python_oracle() -> Result<(), Box<dyn std::error::Error>> {
        let raw = covariate_fixture()?;

        // Case A: full names, matrix order == SDRF order. Oracle: [[0,0],[1,0],[0,1],[1,1]].
        let names = ["S1", "S2", "S3", "S4"];
        let covs = extract_sdrf_covariates(
            &raw,
            &names,
            &[
                "characteristics[sex]".into(),
                "characteristics[organism part]".into(),
            ],
        )?;
        assert_eq!(
            covs,
            Some(vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ])
        );

        // Case B: substring column match. Oracle: [[0],[1],[0],[1]].
        let covs_b = extract_sdrf_covariates(&raw, &names, &["sex".into()])?;
        assert_eq!(
            covs_b,
            Some(vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]])
        );

        // Case C: matrix sample name is a superstring of the SDRF source name.
        let names_c = ["S1_frac1", "S2_frac1", "S3_frac1", "S4_frac1"];
        let covs_c = extract_sdrf_covariates(&raw, &names_c, &["characteristics[sex]".into()])?;
        assert_eq!(
            covs_c,
            Some(vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]])
        );

        // Explicitly requested covariates reject unmatched samples and constant
        // columns instead of silently changing or dropping the design.
        let names_d = ["S1", "S2", "ZZZ", "S4"];
        assert!(extract_sdrf_covariates(
            &raw,
            &names_d,
            &["characteristics[organism part]".into()]
        )
        .is_err());
        assert!(extract_sdrf_covariates(&raw, &names, &["const_col".into()]).is_err());
        assert_eq!(extract_sdrf_covariates(&raw, &names, &[])?, None);
        Ok(())
    }

    #[test]
    fn covariates_keep_numeric_values_and_one_hot_nominal_categories(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let input = concat!(
            "source name\tage\tsite\n",
            "S1\t20\tnorth\n",
            "S2\t35\tsouth\n",
            "S3\t50\twest\n",
            "S4\t65\twest\n",
        );
        let raw = SdrfRawTable::from_reader(input.as_bytes())?;
        let covariates = extract_sdrf_covariates(
            &raw,
            &["S1", "S2", "S3", "S4"],
            &["age".into(), "site".into()],
        )?;
        assert_eq!(
            covariates,
            Some(vec![
                vec![20.0, 0.0, 0.0],
                vec![35.0, 1.0, 0.0],
                vec![50.0, 0.0, 1.0],
                vec![65.0, 0.0, 1.0],
            ])
        );
        Ok(())
    }

    #[test]
    fn covariates_reject_missing_nonfinite_and_mixed_values(
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (header, rows, expected) in [
            (
                "missing",
                ["20", "", "50", "65"],
                "contains a missing value",
            ),
            (
                "nonfinite",
                ["20", "NaN", "50", "65"],
                "contains a non-finite value",
            ),
            (
                "mixed",
                ["20", "unknown", "50", "65"],
                "mixes numeric and categorical values",
            ),
        ] {
            let input = format!(
                "source name\t{header}\nS1\t{}\nS2\t{}\nS3\t{}\nS4\t{}\n",
                rows[0], rows[1], rows[2], rows[3]
            );
            let raw = SdrfRawTable::from_reader(input.as_bytes())?;
            let Err(error) =
                extract_sdrf_covariates(&raw, &["S1", "S2", "S3", "S4"], &[header.into()])
            else {
                panic!("invalid covariate was accepted");
            };
            assert!(error.to_string().contains(expected), "{error}");
        }
        Ok(())
    }

    #[test]
    fn batch_column_values_require_complete_source_name_mapping(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let raw = covariate_fixture()?;
        // Oracle: ['b1','b1','b2','b2'].
        assert_eq!(
            batch_column_values_for_samples(&raw, &["S1", "S2", "S3", "S4"], "batch")?,
            vec![
                "b1".to_owned(),
                "b1".to_owned(),
                "b2".to_owned(),
                "b2".to_owned()
            ]
        );
        assert!(batch_column_values_for_samples(&raw, &["S1", "ZZZ"], "BATCH").is_err());
        assert!(batch_column_values_for_samples(&raw, &["S1"], "nonexistent").is_err());
        Ok(())
    }

    #[test]
    fn apply_batch_correction_corrects_complete_rows_and_keeps_incomplete(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 4 samples -> 2 batches of 2 via sample_prefix ("A-*" vs "B-*"). P1/P2 are
        // complete (ComBat-corrected); P3 has a missing cell (kept uncorrected).
        let mut samples = StringIdRegistry::<SampleId>::new();
        let sample_ids = ["A-1", "A-2", "B-1", "B-2"]
            .iter()
            .map(|name| samples.get_or_insert(name).ok_or("sample id"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let p1 = proteins.get_or_insert("P1").ok_or("p1")?;
        let p2 = proteins.get_or_insert("P2").ok_or("p2")?;
        let p3 = proteins.get_or_insert("P3").ok_or("p3")?;

        let p1_row = [10.0, 12.0, 30.0, 33.0];
        let p2_row = [5.0, 6.0, 20.0, 19.0];
        let mut cells: std::collections::HashMap<CellKey, f64> = std::collections::HashMap::new();
        for (sample, &value) in sample_ids.iter().zip(p1_row.iter()) {
            cells.insert(
                CellKey {
                    protein: p1,
                    sample: *sample,
                },
                value,
            );
        }
        for (sample, &value) in sample_ids.iter().zip(p2_row.iter()) {
            cells.insert(
                CellKey {
                    protein: p2,
                    sample: *sample,
                },
                value,
            );
        }
        // P3: only the first three samples present (the fourth stays missing).
        for (sample, &value) in sample_ids.iter().zip([7.0, 8.0, 9.0].iter()) {
            cells.insert(
                CellKey {
                    protein: p3,
                    sample: *sample,
                },
                value,
            );
        }

        let allowed_proteins = [p1, p2, p3].into_iter().collect();
        let mut matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins,
            excluded_samples: HashSet::new(),
            peptide_counts: std::collections::HashMap::new(),
            values: ProteinValues::Cells(cells),
        };

        matrix.apply_batch_correction(
            &BatchCorrectionConfig {
                enabled: true,
                ..BatchCorrectionConfig::default()
            },
            None,
            false,
        )?;

        // Complete rows match a direct ComBat call over [P1, P2] with batch [0,0,1,1].
        let expected = mokume_stats::batch::combat(
            &[p1_row.to_vec(), p2_row.to_vec()],
            &[0, 0, 1, 1],
            None,
            mokume_stats::batch::ComBatParams::default(),
        );
        for (row, protein) in [p1, p2].into_iter().enumerate() {
            for (col, sample) in sample_ids.iter().enumerate() {
                let got = matrix.value(protein, *sample).ok_or("corrected value")?;
                assert!((got - expected[row][col]).abs() < 1e-9);
            }
        }
        // Incomplete row P3 is untouched, and its missing cell stays missing.
        for (col, sample) in sample_ids.iter().take(3).enumerate() {
            let got = matrix.value(p3, *sample).ok_or("p3 value")?;
            assert!((got - [7.0, 8.0, 9.0][col]).abs() < 1e-12);
        }
        assert!(matrix.value(p3, sample_ids[3]).is_none());
        Ok(())
    }

    #[test]
    fn apply_batch_correction_column_method_reads_sdrf_and_errors_on_missing(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Sample names that are each their own `sample_prefix` (no '-', no
        // `[a-z]+\d+_`), so prefix detection would see four singleton batches and
        // no-op. The explicit `column` method reads the SDRF `batch_id` column,
        // grouping them into two batches of two so correction actually runs --
        // proving the column path is wired, not the prefix path.
        type BuiltMatrix = (ProteinMatrix, Vec<SampleId>, ProteinId, ProteinId);
        let p1_row = [10.0, 12.0, 30.0, 33.0];
        let p2_row = [5.0, 6.0, 20.0, 19.0];
        let build = || -> Result<BuiltMatrix, Box<dyn std::error::Error>> {
            let mut samples = StringIdRegistry::<SampleId>::new();
            let sample_ids = ["X1", "X2", "X3", "X4"]
                .iter()
                .map(|name| samples.get_or_insert(name).ok_or("sample id"))
                .collect::<Result<Vec<_>, _>>()?;
            let mut proteins = StringIdRegistry::<ProteinId>::new();
            let p1 = proteins.get_or_insert("P1").ok_or("p1")?;
            let p2 = proteins.get_or_insert("P2").ok_or("p2")?;
            let mut cells: std::collections::HashMap<CellKey, f64> =
                std::collections::HashMap::new();
            for (sample, &value) in sample_ids.iter().zip(p1_row.iter()) {
                cells.insert(
                    CellKey {
                        protein: p1,
                        sample: *sample,
                    },
                    value,
                );
            }
            for (sample, &value) in sample_ids.iter().zip(p2_row.iter()) {
                cells.insert(
                    CellKey {
                        protein: p2,
                        sample: *sample,
                    },
                    value,
                );
            }
            let allowed_proteins = [p1, p2].into_iter().collect();
            Ok((
                ProteinMatrix {
                    proteins,
                    samples,
                    allowed_proteins,
                    excluded_samples: HashSet::new(),
                    peptide_counts: std::collections::HashMap::new(),
                    values: ProteinValues::Cells(cells),
                },
                sample_ids,
                p1,
                p2,
            ))
        };

        let dir = tempfile::Builder::new()
            .prefix("mokume-batch-column-wiring-")
            .tempdir()?
            .keep();
        let sdrf_path = dir.join("col.sdrf.tsv");
        fs::write(
            &sdrf_path,
            "source name\tbatch_id\nX1\tg1\nX2\tg1\nX3\tg2\nX4\tg2\n",
        )?;

        let column_config = |method: &str, column: Option<&str>| BatchCorrectionConfig {
            enabled: true,
            method: method.to_owned(),
            column: column.map(ToOwned::to_owned),
            ..BatchCorrectionConfig::default()
        };

        let (mut matrix, sample_ids, p1, p2) = build()?;
        matrix.apply_batch_correction(
            &column_config("column", Some("batch_id")),
            Some(&sdrf_path),
            false,
        )?;
        // Corrected rows match a direct ComBat over the column-derived batch [0,0,1,1].
        let expected = mokume_stats::batch::combat(
            &[p1_row.to_vec(), p2_row.to_vec()],
            &[0, 0, 1, 1],
            None,
            mokume_stats::batch::ComBatParams::default(),
        );
        for (row, protein) in [p1, p2].into_iter().enumerate() {
            for (col, sample) in sample_ids.iter().enumerate() {
                let got = matrix.value(protein, *sample).ok_or("corrected value")?;
                assert!((got - expected[row][col]).abs() < 1e-9);
            }
        }

        // An absent column errors like Python's `_detect_explicit_batches(None)`.
        let (mut missing, ..) = build()?;
        assert!(missing
            .apply_batch_correction(
                &column_config("column", Some("nope")),
                Some(&sdrf_path),
                false
            )
            .is_err());

        // `run` has no run-level mapping in the protein-matrix flow -> error.
        let (mut run_matrix, ..) = build()?;
        assert!(run_matrix
            .apply_batch_correction(&column_config("run", None), Some(&sdrf_path), false)
            .is_err());

        fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn apply_batch_correction_rejects_removed_and_unknown_methods(
    ) -> Result<(), Box<dyn std::error::Error>> {
        type BuiltMatrix = (ProteinMatrix, Vec<SampleId>, ProteinId, ProteinId);
        let p1_row = [10.0, 12.0, 30.0, 33.0];
        let p2_row = [5.0, 6.0, 20.0, 19.0];
        let build = || -> Result<BuiltMatrix, Box<dyn std::error::Error>> {
            let mut samples = StringIdRegistry::<SampleId>::new();
            let sample_ids = ["g1-A", "g1-B", "g2-A", "g2-B"]
                .iter()
                .map(|name| samples.get_or_insert(name).ok_or("sample id"))
                .collect::<Result<Vec<_>, _>>()?;
            let mut proteins = StringIdRegistry::<ProteinId>::new();
            let p1 = proteins.get_or_insert("P1").ok_or("p1")?;
            let p2 = proteins.get_or_insert("P2").ok_or("p2")?;
            let mut cells: std::collections::HashMap<CellKey, f64> =
                std::collections::HashMap::new();
            for (sample, &value) in sample_ids.iter().zip(p1_row.iter()) {
                cells.insert(
                    CellKey {
                        protein: p1,
                        sample: *sample,
                    },
                    value,
                );
            }
            for (sample, &value) in sample_ids.iter().zip(p2_row.iter()) {
                cells.insert(
                    CellKey {
                        protein: p2,
                        sample: *sample,
                    },
                    value,
                );
            }
            let allowed_proteins = [p1, p2].into_iter().collect();
            Ok((
                ProteinMatrix {
                    proteins,
                    samples,
                    allowed_proteins,
                    excluded_samples: HashSet::new(),
                    peptide_counts: std::collections::HashMap::new(),
                    values: ProteinValues::Cells(cells),
                },
                sample_ids,
                p1,
                p2,
            ))
        };

        let method_config = |method: &str| BatchCorrectionConfig {
            enabled: true,
            method: method.to_owned(),
            ..BatchCorrectionConfig::default()
        };

        for method in ["fraction", "techreplicate", "totally-unknown"] {
            let (mut matrix, ..) = build()?;
            assert!(matrix
                .apply_batch_correction(&method_config(method), None, false)
                .is_err());
        }
        Ok(())
    }

    #[test]
    fn write_csv_missing_fill_matches_python_per_method_convention(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Columns sort by sample name: S1=col 1, S2=col 2, S3=col 3. P1 is observed
        // in S1 and S2; P2 only in S1 (missing in the observed sample S2); S3 carries
        // no observation at all.
        let mut samples = StringIdRegistry::<SampleId>::new();
        let s1 = samples.get_or_insert("S1").ok_or("s1")?;
        let s2 = samples.get_or_insert("S2").ok_or("s2")?;
        let _s3 = samples.get_or_insert("S3").ok_or("s3")?;
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let p1 = proteins.get_or_insert("P1").ok_or("p1")?;
        let p2 = proteins.get_or_insert("P2").ok_or("p2")?;
        let mut cells: std::collections::HashMap<CellKey, f64> = std::collections::HashMap::new();
        cells.insert(
            CellKey {
                protein: p1,
                sample: s1,
            },
            10.0,
        );
        cells.insert(
            CellKey {
                protein: p1,
                sample: s2,
            },
            20.0,
        );
        cells.insert(
            CellKey {
                protein: p2,
                sample: s1,
            },
            30.0,
        );
        let matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins: [p1, p2].into_iter().collect(),
            excluded_samples: HashSet::new(),
            peptide_counts: std::collections::HashMap::new(),
            values: ProteinValues::Cells(cells),
        };

        let cell = |csv: &str, protein: &str, col: usize| -> String {
            csv.lines()
                .find(|line| line.starts_with(&format!("{protein},")))
                .and_then(|line| line.split(',').nth(col))
                .unwrap_or("ABSENT")
                .to_owned()
        };

        // Additive method (`Some(0.0)`): a missing cell in the observed sample S2 is
        // `0`, but the all-empty sample S3 stays blank (Python never densifies it).
        let zero_path = tempfile::Builder::new()
            .prefix("mokume-missing-zero-")
            .tempdir()?
            .keep()
            .join("missing_zero.csv");
        matrix.write_csv(&zero_path, OutputFormat::PythonCompatible, false, Some(0.0))?;
        let zero = fs::read_to_string(&zero_path)?;
        assert_eq!(cell(&zero, "P1", 1), "10");
        assert_eq!(cell(&zero, "P2", 2), "0");
        assert_eq!(cell(&zero, "P2", 3), "");

        // Average/ratio method (`None`): every missing cell is blank.
        let nan_path = tempfile::Builder::new()
            .prefix("mokume-missing-nan-")
            .tempdir()?
            .keep()
            .join("missing_nan.csv");
        matrix.write_csv(&nan_path, OutputFormat::PythonCompatible, false, None)?;
        let nan = fs::read_to_string(&nan_path)?;
        assert_eq!(cell(&nan, "P1", 1), "10");
        assert_eq!(cell(&nan, "P2", 2), "");
        assert_eq!(cell(&nan, "P2", 3), "");
        Ok(())
    }

    #[test]
    fn accepts_ensemble_de_subset() -> Result<(), Box<dyn std::error::Error>> {
        // ensemble runs member methods and fuses them with the deterministic
        // top-k consensus combiner; a well-formed ensemble DE config must validate
        // (including an explicit member list via repeated --de-ensemble-method).
        let parquet = existing_dummy_path("ensemble_de")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "ensemble".to_string();
        config.differential_expression.ensemble_methods =
            Some(vec!["limma".to_string(), "deqms".to_string()]);
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(PathBuf::from("de.csv"));

        validate_implemented_subset(&config)?;
        Ok(())
    }

    #[test]
    fn rejects_invalid_ensemble_members_before_execution() -> Result<(), Box<dyn std::error::Error>>
    {
        let invalid_members = [
            (Vec::<String>::new(), "empty"),
            (vec!["".to_string()], "empty"),
            (vec!["unknown".to_string()], "unknown"),
            (vec!["auto".to_string()], "unknown"),
            (vec!["ensemble".to_string()], "nested"),
            (
                vec!["limma".to_string(), " LIMMA ".to_string()],
                "duplicate",
            ),
        ];

        for (members, expected) in invalid_members {
            let parquet = existing_dummy_path("invalid_ensemble_member")?;
            let mut config = base_config(parquet);
            config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
            config.differential_expression.enabled = true;
            config.differential_expression.method = "ensemble".to_string();
            config.differential_expression.ensemble_methods = Some(members);
            config.differential_expression.ensemble_min_k = 1;
            config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
            config.differential_expression.output = Some(PathBuf::from("de.csv"));

            let Some(error) = validate_implemented_subset(&config).err() else {
                return Err("invalid ensemble member configuration was accepted".into());
            };
            let error = error.to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} in error, got {error:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn validates_ensemble_min_k_against_configured_members(
    ) -> Result<(), Box<dyn std::error::Error>> {
        for min_k in [0, 3] {
            let parquet = existing_dummy_path("invalid_ensemble_min_k")?;
            let mut config = base_config(parquet);
            config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
            config.differential_expression.enabled = true;
            config.differential_expression.method = "ensemble".to_string();
            config.differential_expression.ensemble_methods =
                Some(vec!["limma".to_string(), "deqms".to_string()]);
            config.differential_expression.ensemble_min_k = min_k;
            config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
            config.differential_expression.output = Some(PathBuf::from("de.csv"));

            let Some(error) = validate_implemented_subset(&config).err() else {
                return Err("invalid ensemble min-k was accepted".into());
            };
            let error = error.to_string();
            assert!(error.contains("min-k"), "unexpected error: {error}");
        }
        Ok(())
    }

    #[test]
    fn invalid_ensemble_precedes_missing_input_without_output(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tempdir = tempfile::tempdir()?;
        let output = tempdir.path().join("de.csv");
        let mut config = base_config(tempdir.path().join("missing.parquet"));
        config.input.sdrf = Some(tempdir.path().join("missing.sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "ensemble".to_string();
        config.differential_expression.ensemble_methods = Some(vec!["unknown".to_string()]);
        config.differential_expression.ensemble_min_k = 1;
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(output.clone());

        let Some(error) = run_features_to_proteins(&config).err() else {
            return Err("invalid ensemble configuration was accepted".into());
        };
        let error = error.to_string();

        assert!(error.contains("unknown"), "unexpected error: {error}");
        assert!(!output.exists(), "invalid ensemble created DE output");
        Ok(())
    }

    #[test]
    fn rejects_ensemble_methods_for_single_method() -> Result<(), Box<dyn std::error::Error>> {
        // --de-ensemble-method is meaningless for a single-method run and must be
        // rejected rather than silently ignored.
        let parquet = existing_dummy_path("ensemble_methods_on_limma")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "limma".to_string();
        config.differential_expression.ensemble_methods = Some(vec!["deqms".to_string()]);
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);

        let error = validate_implemented_subset(&config).err();
        assert!(
            matches!(error, Some(MokumeError::InvalidInput { .. })),
            "expected InvalidInput, got {error:?}"
        );
        Ok(())
    }

    #[test]
    fn accepts_ihw_fdr_method() -> Result<(), Box<dyn std::error::Error>> {
        // IHW is now a supported FDR method (ported in mokume-stats'
        // `ihw_correction` and applied per method in `de::run_member`), so the
        // validation must let it through.
        let parquet = existing_dummy_path("de_ihw")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "limma".to_string();
        config.differential_expression.fdr_method = "ihw".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(PathBuf::from("de.csv"));

        validate_implemented_subset(&config)?;
        Ok(())
    }

    #[test]
    fn accepts_adaptive_fdr_methods() -> Result<(), Box<dyn std::error::Error>> {
        for method in ["bky", "storey"] {
            let parquet = existing_dummy_path(&format!("de_{method}"))?;
            let mut config = base_config(parquet);
            config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
            config.differential_expression.enabled = true;
            config.differential_expression.method = "limma".to_string();
            config.differential_expression.fdr_method = method.to_string();
            config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
            config.differential_expression.output = Some(PathBuf::from("de.csv"));
            validate_implemented_subset(&config)?;
        }
        Ok(())
    }

    #[test]
    fn rejects_unknown_fdr_method() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("de_unknown_fdr")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "limma".to_string();
        config.differential_expression.fdr_method = "unknown".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(PathBuf::from("de.csv"));

        assert!(matches!(
            validate_implemented_subset(&config),
            Err(MokumeError::NotImplemented {
                stage: "differential-expression-fdr-method"
            })
        ));
        Ok(())
    }

    #[test]
    fn accepts_most_frequent_imputation() -> Result<(), Box<dyn std::error::Error>> {
        let parquet = existing_dummy_path("impute_most_frequent")?;
        let mut config = base_config(parquet);
        config.imputation.enabled = true;
        config.imputation.method = "most_frequent".to_string();

        validate_implemented_subset(&config)?;
        Ok(())
    }

    #[test]
    fn disabled_de_options_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        // Non-default DE options must not be accepted when no DE result will run.
        let parquet = existing_dummy_path("de_disabled")?;
        let mut config = base_config(parquet);
        config.differential_expression.method = "deqms".to_string();
        config.differential_expression.fdr_method = "ihw".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);

        assert!(matches!(
            validate_implemented_subset(&config),
            Err(MokumeError::InvalidInput { .. })
        ));
        Ok(())
    }

    fn existing_dummy_path(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = tempfile::Builder::new()
            .prefix(&format!(
                "mokume_pipeline_{name}_{}_{}_",
                std::process::id(),
                unique_suffix()
            ))
            .tempdir()?
            .keep()
            .join("dummy");
        fs::write(&path, [])?;
        Ok(path)
    }

    fn unique_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    }

    fn base_config(parquet: PathBuf) -> FeatureToProteinsConfig {
        FeatureToProteinsConfig {
            input: InputConfig {
                parquet: Some(parquet),
                msstats: None,
                psm: None,
                sdrf: None,
                fasta: None,
            },
            output: OutputConfig {
                protein_matrix: PathBuf::from("protein.csv"),
                export_peptides: None,
                export_ions: None,
                format: OutputFormat::PythonCompatible,
            },
            filtering: FilterConfig::default(),
            normalization: NormalizationConfig::default(),
            quantification: QuantMethod::MaxLfq,
            topn_peptides: 3,
            maxlfq: MaxLfqConfig::default(),
            pibaq: PibaqConfig::default(),
            directlfq: DirectLfqConfig::default(),
            batch: BatchCorrectionConfig::default(),
            irs: IrsConfig::default(),
            coverage_threshold: None,
            sample_correlation_threshold: None,
            ratio: RatioConfig::default(),
            imputation: ImputationConfig::default(),
            differential_expression: DifferentialExpressionConfig::default(),
            runtime: RuntimeConfig {
                memory: None,
                threads: None,
            },
        }
    }

    // Both LFQ methods apply `min_unique_peptides` per `(protein, sample)`.
    fn lfq_aggregation(
        method: super::QuantMethod,
        directlfq_sums: std::collections::HashMap<super::DirectLfqCellKey, f64>,
    ) -> super::FeatureAggregation {
        let canonical_traces = directlfq_sums
            .into_iter()
            .map(|(key, intensity)| {
                (
                    super::PeptideCellKey {
                        protein: key.protein,
                        sample: key.sample,
                        peptide: key.canonical,
                    },
                    intensity,
                )
            })
            .collect();
        super::FeatureAggregation::Lfq {
            method,
            route_to_directlfq: true,
            directlfq_min_nonan: 1,
            directlfq_num_samples_quadratic: 50,
            traces: super::MaxLfqFeatureAggregation {
                canonical_traces,
                ..super::MaxLfqFeatureAggregation::default()
            },
            cached_directlfq_values: None,
        }
    }

    struct LfqMinUniqueFixture {
        scattered: ProteinId,
        dense: ProteinId,
        sums: HashMap<super::DirectLfqCellKey, f64>,
        allowed_cells: HashSet<super::CellKey>,
        canonical_peptides: StringIdRegistry<PeptideId>,
    }

    fn lfq_min_unique_fixture() -> LfqMinUniqueFixture {
        let scattered = ProteinId::new(1);
        let dense = ProteinId::new(2);
        let pep_a = PeptideId::new(10);
        let pep_b = PeptideId::new(11);
        let s1 = SampleId::new(1);
        let s2 = SampleId::new(2);
        let s3 = SampleId::new(3);
        let s4 = SampleId::new(4);

        // `scattered`: two distinct peptides, each in two samples, but never
        // both in the same sample (pep_a in s1/s2, pep_b in s3/s4).
        // `dense`: two distinct peptides in every sample.
        let mut sums = std::collections::HashMap::new();
        let cell = |protein, canonical, sample| super::DirectLfqCellKey {
            protein,
            canonical,
            sample,
        };
        sums.insert(cell(scattered, pep_a, s1), 1000.0);
        sums.insert(cell(scattered, pep_a, s2), 1100.0);
        sums.insert(cell(scattered, pep_b, s3), 2000.0);
        sums.insert(cell(scattered, pep_b, s4), 2200.0);
        for &sample in &[s1, s2, s3, s4] {
            sums.insert(cell(dense, pep_a, sample), 1000.0);
            sums.insert(cell(dense, pep_b, sample), 1100.0);
        }

        // `allowed_cells` is the per-(protein, sample) gate: only `dense` ever
        // carries two distinct peptides in a single sample.
        let allowed_cells = [s1, s2, s3, s4]
            .into_iter()
            .map(|sample| super::CellKey {
                protein: dense,
                sample,
            })
            .collect::<HashSet<_>>();
        // Register canonical sequences so `pep_a` (id 10) and `pep_b` (id 11)
        // resolve; `seq_NN` strings keep alphabetical order aligned with the id
        // order, so the DirectLFQ row order is unchanged by the sequence-rank
        // sort under test elsewhere.
        let mut canonical_peptides = super::StringIdRegistry::<PeptideId>::new();
        for index in 0..=11 {
            let _ = canonical_peptides.get_or_insert(&format!("seq_{index:02}"));
        }
        LfqMinUniqueFixture {
            scattered,
            dense,
            sums,
            allowed_cells,
            canonical_peptides,
        }
    }

    fn lfq_fixture_protein_ids(
        method: super::QuantMethod,
        fixture: &LfqMinUniqueFixture,
    ) -> HashSet<ProteinId> {
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        lfq_aggregation(method, fixture.sums.clone())
            .finalize(
                &fixture.allowed_cells,
                &mut proteins,
                &HashMap::new(),
                &fixture.canonical_peptides,
                &StringIdRegistry::<SampleId>::new(),
            )
            .protein_ids()
    }

    #[test]
    fn lfq_min_unique_gate_is_per_cell_for_both_methods() {
        let fixture = lfq_min_unique_fixture();
        let max_proteins = lfq_fixture_protein_ids(super::QuantMethod::MaxLfq, &fixture);
        let direct_proteins = lfq_fixture_protein_ids(super::QuantMethod::DirectLfq, &fixture);

        assert!(
            max_proteins.contains(&fixture.dense),
            "MaxLFQ keeps a protein with a two-peptide cell"
        );
        assert!(
            !max_proteins.contains(&fixture.scattered),
            "MaxLFQ drops a protein whose cells never reach two distinct peptides"
        );
        assert!(direct_proteins.contains(&fixture.dense));
        assert!(
            !direct_proteins.contains(&fixture.scattered),
            "DirectLFQ applies the same per-cell unique-peptide gate"
        );
    }

    // Lock the monoisotopic molecular weight against the pyOpenMS
    // `AASequence.fromString(seq).getMonoWeight()` values that the Python piBAQ
    // TPA path reads via `digest_fasta_full(..., compute_mw=True)`. Captured
    // with `conda run -n Bigbio python -c "from pyopenms import AASequence;
    // print(AASequence.fromString(SEQ).getMonoWeight())"`.
    #[test]
    fn protein_mono_weight_matches_pyopenms() {
        let cases = [
            ("PEPTIDEAK", 998.492047233),
            ("ALYAAEK", 764.4068587863999),
            ("PEPTIDEAKAPEPTIDECKASHAEDPEPK", 3143.460508429001),
            ("THIDPEAKATHIDPECK", 1903.9098218451002),
            ("ACDEFGHIKLMNPQRSTVWY", 2394.1249175321004),
        ];
        for (sequence, expected) in cases {
            let got = super::protein_mono_weight(sequence);
            assert!(
                (got - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{sequence}: got {got}, expected {expected}"
            );
        }
    }
}

#[cfg(test)]
mod contaminant_and_plex_tests {
    use std::collections::{HashMap, HashSet};

    use mokume_core::SampleId;
    use mokume_io::{SdrfRawTable, SdrfTable};

    use super::{
        derive_sample_plexes, drop_contaminant_accessions, find_contaminant_twins,
        matches_protein_contaminant, matches_sql_contaminant, rename_contaminant_twins,
        strip_contaminant_members, summarize_digest_misses, PlexSource,
    };

    fn owned(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn contaminant_twin_group_is_kept_with_human_accession() {
        assert!(!matches_protein_contaminant("CONTAM_P05787;P05787", &[]));
        assert_eq!(
            strip_contaminant_members("CONTAM_P05787;P05787", &[]).as_deref(),
            Some("P05787")
        );
        assert!(!matches_sql_contaminant(
            &owned(&["sp|CONTAM_P04179|CONTAM_SODM_HUMAN", "sp|P04179|SODM_HUMAN"]),
            &[]
        ));
    }

    #[test]
    fn all_contaminant_group_is_dropped() {
        assert!(matches_protein_contaminant("CONTAM_P00761", &[]));
        assert!(matches_protein_contaminant(
            "CONTAM_P02769;CONTAM_P02768",
            &[]
        ));
        assert_eq!(
            strip_contaminant_members("CONTAM_P02769;CONTAM_P02768", &[]),
            None
        );
        assert!(matches_sql_contaminant(
            &owned(&["sp|CONTAM_P00761|TRYP_PIG"]),
            &[]
        ));
    }

    #[test]
    fn mixed_group_keeps_non_contaminant_members() {
        assert!(!matches_protein_contaminant(
            "P02768;CONTAM_P02769;Q9XXXX",
            &[]
        ));
        assert_eq!(
            strip_contaminant_members("P02768;CONTAM_P02769;Q9XXXX", &[]).as_deref(),
            Some("P02768;Q9XXXX")
        );
        assert_eq!(strip_contaminant_members("P02768;Q9XXXX", &[]), None);
    }

    #[test]
    fn decoy_member_still_drops_group() {
        assert!(matches_protein_contaminant("P12345;DECOY_P99999", &[]));
        assert!(matches_sql_contaminant(
            &owned(&["P12345", "DECOY_P99999"]),
            &[]
        ));
        let custom = owned(&["CONTAM", "DECOY"]);
        assert!(matches_protein_contaminant("P1;DECOY_P2", &custom));
        assert!(!matches_protein_contaminant("CONTAM_P1;P1", &custom));
        assert!(matches_protein_contaminant("contam_P1", &custom));
        assert!(!matches_sql_contaminant(
            &owned(&["CONTAM_P1", "P1"]),
            &custom
        ));
    }

    #[test]
    fn digest_drops_contaminant_accessions() {
        let mut digest = HashMap::<String, HashSet<String>>::new();
        digest.insert(
            "P05787".to_owned(),
            HashSet::from(["LESGMQNMSIHTK".to_owned()]),
        );
        digest.insert(
            "CONTAM_P05787".to_owned(),
            HashSet::from(["LESGMQNMSIHTK".to_owned()]),
        );
        assert_eq!(drop_contaminant_accessions(&mut digest), 1);
        assert!(digest.contains_key("P05787"));
        assert!(!digest.contains_key("CONTAM_P05787"));
    }

    #[test]
    fn contaminant_only_twin_group_is_renamed_to_human() {
        let peptides = |values: &[&str]| values.iter().map(|v| (*v).to_owned()).collect();
        let mut digest = HashMap::<String, HashSet<String>>::new();
        digest.insert("P02533".to_owned(), peptides(&["AAAK", "BBBK"]));
        digest.insert("CONTAM_P02533".to_owned(), peptides(&["AAAK", "BBBK"]));
        digest.insert("CONTAM_P00761".to_owned(), peptides(&["TRYPK"]));
        digest.insert("P11111".to_owned(), peptides(&["CCCK"]));
        digest.insert("CONTAM_P11111".to_owned(), peptides(&["CCCK", "BOVINEK"]));
        let twins = find_contaminant_twins(&digest);
        assert_eq!(twins.len(), 1);
        assert_eq!(twins["CONTAM_P02533"], "P02533");
        assert_eq!(
            rename_contaminant_twins("CONTAM_P02533", &twins).as_deref(),
            Some("P02533")
        );
        assert_eq!(
            rename_contaminant_twins("CONTAM_P02533;P02533", &twins).as_deref(),
            Some("P02533")
        );
        assert_eq!(rename_contaminant_twins("CONTAM_P00761", &twins), None);
    }

    #[test]
    fn digest_misses_are_counted_by_reason() {
        let mut peptide_accessions = HashMap::<String, HashSet<String>>::new();
        for peptide in ["PEPTIDEK", "MAEPTIDEK", "SECONDPEPK"] {
            peptide_accessions.insert(peptide.to_owned(), HashSet::from(["P1".to_owned()]));
        }
        let owner = peptide_accessions
            .keys()
            .map(|peptide| (peptide.clone(), "F1".to_owned()))
            .collect::<HashMap<_, _>>();
        let sample = SampleId::from(0);
        let observations = vec![
            ("PEPTIDEK".to_owned(), sample, 100.0),
            ("AEPTIDEK".to_owned(), sample, 10.0),
            ("PEPTIDEKSECONDPEPK".to_owned(), sample, 20.0),
            ("A".repeat(31) + "K", sample, 30.0),
            ("SEMITYPTICPEPTIDE".to_owned(), sample, 40.0),
            ("SEMITYPTICPEPTIDE".to_owned(), SampleId::from(1), 0.0),
        ];
        let summary = summarize_digest_misses(&observations, &owner, &peptide_accessions, Some(30));
        assert_eq!(summary.observed_peptides, 5);
        assert_eq!(summary.dropped_peptides, 4);
        assert_eq!(summary.met_removed_n_term, 1);
        assert_eq!(summary.missed_cleavage, 1);
        assert_eq!(summary.too_long, 1);
        assert_eq!(summary.other, 1);
        assert!((summary.dropped_intensity_fraction() - 100.0 / 200.0).abs() < 1e-12);
    }

    const TMT_HEADER: &str = "source name\tcomment[data file]\tcomment[label]\tcomment[plex]\n";

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    fn tmt_sdrf(rows: &[(&str, &str, &str, &str)]) -> TestResult<(SdrfTable, SdrfRawTable)> {
        let mut text = TMT_HEADER.to_owned();
        for (sample, file, label, plex) in rows {
            text.push_str(&format!("{sample}\t{file}\t{label}\t{plex}\n"));
        }
        Ok((
            SdrfTable::from_reader(text.as_bytes())?,
            SdrfRawTable::from_reader(text.as_bytes())?,
        ))
    }

    fn plex_sizes(plexes: &HashMap<String, String>) -> Vec<usize> {
        let mut sizes = HashMap::<&str, usize>::new();
        for plex in plexes.values() {
            *sizes.entry(plex.as_str()).or_default() += 1;
        }
        let mut sizes = sizes.into_values().collect::<Vec<_>>();
        sizes.sort_unstable();
        sizes
    }

    #[test]
    fn set_style_source_names_split_by_data_files() -> TestResult {
        let mut rows = Vec::new();
        for (set, files) in [
            ("Set1", ["s1_f1.raw", "s1_f2.raw"]),
            ("Set2", ["s2_f1.raw", "s2_f2.raw"]),
        ] {
            for label in ["TMT126", "TMT127C", "TMT131"] {
                for file in files {
                    rows.push((format!("PXD011896-{set}-{label}"), file, label));
                }
            }
        }
        let rows = rows
            .iter()
            .map(|(sample, file, label)| (sample.as_str(), *file, *label, ""))
            .collect::<Vec<_>>();
        let (sdrf, raw) = tmt_sdrf(&rows)?;
        let (plexes, source) = derive_sample_plexes(&sdrf, Some(&raw), None)?;
        assert_eq!(source, PlexSource::DataFiles);
        assert_eq!(plex_sizes(&plexes), vec![3, 3]);
        assert_eq!(
            plexes["PXD011896-Set1-TMT126"],
            plexes["PXD011896-Set1-TMT131"]
        );
        assert_ne!(
            plexes["PXD011896-Set1-TMT126"],
            plexes["PXD011896-Set2-TMT126"]
        );
        Ok(())
    }

    #[test]
    fn sample_style_source_names_split_by_data_files() -> TestResult {
        let (sdrf, raw) = tmt_sdrf(&[
            ("MSV000085836-Sample-1", "a.raw", "TMT126", ""),
            ("MSV000085836-Sample-2", "a.raw", "TMT127N", ""),
            ("MSV000085836-Sample-3", "b.raw", "TMT126", ""),
            ("MSV000085836-Sample-4", "b.raw", "TMT127N", ""),
        ])?;
        let (plexes, source) = derive_sample_plexes(&sdrf, Some(&raw), None)?;
        assert_eq!(source, PlexSource::DataFiles);
        assert_eq!(plex_sizes(&plexes), vec![2, 2]);
        assert_ne!(
            plexes["MSV000085836-Sample-1"],
            plexes["MSV000085836-Sample-3"]
        );
        Ok(())
    }

    #[test]
    fn legacy_plex_prefix_is_kept_as_plex_id() -> TestResult {
        let (sdrf, raw) = tmt_sdrf(&[
            ("p1_1", "a.raw", "TMT126", ""),
            ("p1_2", "a.raw", "TMT127N", ""),
            ("p2_1", "b.raw", "TMT126", ""),
            ("p2_2", "b.raw", "TMT127N", ""),
        ])?;
        let (plexes, _) = derive_sample_plexes(&sdrf, Some(&raw), None)?;
        assert_eq!(plexes["p1_2"], "p1");
        assert_eq!(plexes["p2_1"], "p2");
        Ok(())
    }

    #[test]
    fn explicit_plex_column_overrides_data_files() -> TestResult {
        let (sdrf, raw) = tmt_sdrf(&[
            ("S1", "a.raw", "TMT126", "A"),
            ("S2", "a.raw", "TMT127N", "A"),
            ("S3", "b.raw", "TMT126", "B"),
            ("S4", "b.raw", "TMT127N", "B"),
        ])?;
        let (plexes, source) = derive_sample_plexes(&sdrf, Some(&raw), Some("comment[plex]"))?;
        assert_eq!(source, PlexSource::Column);
        assert_eq!(plexes["S2"], "A");
        assert_eq!(plexes["S3"], "B");
        assert!(derive_sample_plexes(&sdrf, Some(&raw), Some("comment[missing]")).is_err());
        Ok(())
    }

    #[test]
    fn label_free_falls_back_to_source_name_rule() -> TestResult {
        let (sdrf, raw) = tmt_sdrf(&[
            ("p1_1", "a.raw", "label free sample", ""),
            ("p1_2", "b.raw", "label free sample", ""),
        ])?;
        let (plexes, source) = derive_sample_plexes(&sdrf, Some(&raw), None)?;
        assert_eq!(source, PlexSource::SourceName);
        assert_eq!(plexes["p1_2"], "p1");
        Ok(())
    }
}
