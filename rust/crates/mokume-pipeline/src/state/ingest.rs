use crate::accessions::{
    matches_protein_contaminant, register_id, register_loading_context, run_qc_key, sample_name,
    sdrf_record, strip_contaminant_members,
};
use crate::aggregation::{FeatureAggregation, IntensityFactors};
use crate::exports::{IntermediateExports, IntermediateMeasurement};
use crate::from_peptides::PibaqDigest;
use crate::normalization_factors::{
    has_removed_accession, passes_feature_filter, sample_condition,
};
use crate::state::{ContextualIds, FeatureToProteinState, IngestRecord, PeptideIndex};
use crate::validation::fails_protein_fdr;
use crate::{
    FeatureToProteinsConfig, FilterConfig, HashMap, HashSet, QpxFeatureRecord, Result,
    SdrfLookupCache, SdrfRawTable, SdrfRecord, SdrfTable, StringIdRegistry, filters, invalid_input,
};
impl FeatureToProteinState {
    pub(crate) fn new(
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
            contextual: ContextualIds::default(),
            samples: StringIdRegistry::new(),
            index: PeptideIndex::default(),
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

    /// Ingest one feature. With `deferred`, the accepted feature's ids are
    /// appended there for [`ingest_features`]'s pipeline threads instead of
    /// being applied to the peptide index and the aggregation here.
    pub(crate) fn ingest(
        &mut self,
        feature: &QpxFeatureRecord,
        sdrf: Option<&mut SdrfLookupCache<'_>>,
        filtering: FilterConfig,
        intensity_factors: Option<&IntensityFactors>,
        deferred: Option<&mut Vec<IngestRecord>>,
    ) -> Result<()> {
        // ReplicateAgreementFilter degeneracy (Python `intensity.py:194-242` under
        // the per-sample pipeline): `min_replicate_agreement > 1` drops every row
        // because the per-sample `nunique(SampleID)` is always 1. No feature is
        // ingested, so the output is header-only, matching Python.
        if self.replicate_agreement_wipes_all {
            return Ok(());
        }
        let Some((protein_group, cv_group)) = self.ingest_protein_group(feature, filtering) else {
            return Ok(());
        };
        let sdrf_record = sdrf_record(feature, sdrf)?;
        let sample_name = sample_name(feature, sdrf_record);
        if !self.passes_ingest_group_filters(feature, sdrf_record, &sample_name, cv_group) {
            return Ok(());
        }
        let record = self.register_ingest_record(
            feature,
            sdrf_record,
            &sample_name,
            &protein_group,
            intensity_factors,
        )?;
        let measurement = IntermediateMeasurement {
            protein: record.protein,
            sample: record.sample,
            peptide: record.peptide,
            canonical: record.canonical,
            intensity: record.intensity,
            protein_name: &protein_group,
            peptide_canonical: &feature.sequence,
            sample_name: &sample_name,
            run_file_name: &feature.run_file_name,
            sdrf_record,
        };
        self.accept_ingest_record(record, feature, measurement, intensity_factors, deferred)
    }

    fn ingest_protein_group(
        &self,
        feature: &QpxFeatureRecord,
        filtering: FilterConfig,
    ) -> Option<(String, Option<String>)> {
        let keep_shared = self.keep_shared_peptides || self.aggregation.keeps_shared_peptides();
        if !passes_feature_filter(feature, filtering, keep_shared) {
            return None;
        }

        let protein_group = self.aggregation.protein_name(feature)?;
        // The CV pre-pass keys groups by the name before the twin and
        // contaminant-member rewrites below.
        let cv_group = (!self.cv_dropped.is_empty()).then(|| protein_group.clone());
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
            feature,
            &protein_group,
            filtering.remove_contaminants,
            contaminant_patterns,
        ) {
            return None;
        }
        let protein_group = if filtering.remove_contaminants {
            strip_contaminant_members(&protein_group, contaminant_patterns).unwrap_or(protein_group)
        } else {
            protein_group
        };
        if has_removed_accession(&feature.protein_accessions, &self.remove_protein_ids) {
            return None;
        }
        // Opt-in per-row preprocessing filters run after the load-time filters
        // and before the per-`(protein, sample)` unique-peptide gate, matching
        // Python's pipeline chain (intensity/peptide filters before MinPeptide).
        if !self.passes_peptide_filter_pipeline(feature) {
            return None;
        }

        Some((protein_group, cv_group))
    }

    fn passes_ingest_group_filters(
        &self,
        feature: &QpxFeatureRecord,
        sdrf_record: Option<&SdrfRecord>,
        sample_name: &str,
        cv_group: Option<String>,
    ) -> bool {
        // Python applies Run-QC within a per-sample frame but groups its rows by
        // `TechReplicate`. The pre-pass therefore rejects individual technical
        // runs; other runs from the same sample must remain available.
        if !self.run_qc_excluded_runs.is_empty()
            && self.run_qc_excluded_runs.contains(&run_qc_key(
                feature,
                sdrf_record,
                sample_name.to_owned(),
            ))
        {
            return false;
        }
        // CVThresholdFilter (Python `intensity.py:127-141`): drop every row of a
        // `(sample, ProteinName, PeptideCanonical)` whose within-sample CV exceeds
        // `cv_threshold`. The dropped set is precomputed over the raw,
        // `min_unique`-gated intensities; Python runs CV *before* the quantile and
        // before the per-row peptide filters (length/charge/modification/missed
        // cleavage), so the CV input includes rows those later filters would drop
        // -- the pre-pass mirrors that by buffering the same load-gated rows. Empty
        // unless `cv_threshold` is set.
        if cv_group.is_some_and(|group| {
            self.cv_dropped
                .contains(&(sample_name.to_owned(), group, feature.sequence.clone()))
        }) {
            return false;
        }
        // QuantileFilter (Python `intensity.py:296-297`): drop rows whose raw
        // intensity is outside the per-sample `[lower, upper]` bounds (double-closed
        // `>= && <=`). The bounds are precomputed over the `min_unique`-gated set;
        // computing in f64 matches pandas, which upcasts the f32 column to f64 for
        // `quantile` and the comparison. The export-time `min_unique` gate then
        // re-counts canonicals on the survivors, reproducing the post-quantile
        // MinPeptideFilter.
        if let Some(&(lower, upper)) = self.quantile_bounds.get(sample_name)
            && !(feature.intensity >= lower && feature.intensity <= upper)
        {
            return false;
        }
        true
    }

    fn register_ingest_record(
        &mut self,
        feature: &QpxFeatureRecord,
        sdrf_record: Option<&SdrfRecord>,
        sample_name: &str,
        protein_group: &str,
        intensity_factors: Option<&IntensityFactors>,
    ) -> Result<IngestRecord> {
        let sample = register_id(&mut self.samples, sample_name, "sample")?;
        let peptide_key = self.aggregation.peptide_key(feature);
        let base_peptide = register_id(&mut self.peptides, &peptide_key, "peptide")?;
        let canonical_peptide = register_id(
            &mut self.canonical_peptides,
            &feature.sequence,
            "canonical peptide",
        )?;
        let context = if self.aggregation.preserves_loading_rows() {
            let biological_replicate = sdrf_record
                .and_then(|record| record.biological_replicate)
                .unwrap_or(1);
            let condition = sample_condition(sample_name, sdrf_record);
            let condition_id = self
                .conditions
                .get_or_insert(&condition)
                .ok_or_else(|| invalid_input("condition id registry overflow"))?;
            Some(register_loading_context(
                &mut self.loading_contexts,
                (biological_replicate, condition_id),
            )?)
        } else {
            None
        };
        let protein = register_id(&mut self.proteins, protein_group, "protein")?;
        let intensity = intensity_factors.map_or(feature.intensity, |factors| {
            factors.normalize(feature.intensity, sample_name, &feature.run_file_name)
        });
        Ok(IngestRecord {
            protein,
            sample,
            peptide: base_peptide,
            canonical: canonical_peptide,
            contextual_canonical: canonical_peptide,
            species: base_peptide,
            context,
            intensity,
            charge: feature.charge,
            transferred: feature.transferred,
        })
    }

    fn accept_ingest_record(
        &mut self,
        mut record: IngestRecord,
        feature: &QpxFeatureRecord,
        measurement: IntermediateMeasurement<'_>,
        intensity_factors: Option<&IntensityFactors>,
        deferred: Option<&mut Vec<IngestRecord>>,
    ) -> Result<()> {
        let contextual = self.aggregation.preserves_contextual_canonical_rows();
        self.export_measurement(
            &mut record,
            measurement,
            feature.intensity,
            intensity_factors,
            deferred.is_some(),
        )?;
        if let Some(records) = deferred {
            records.push(record);
        } else {
            self.index.add(&record, contextual);
            self.aggregation
                .push(record.measurement(&feature.sequence, measurement.sample_name))?;
        }
        if let Some(tracker) = &mut self.low_frequency {
            tracker.observe(measurement.protein_name, &feature.sequence, record.sample);
        }
        self.accepted_measurements += 1;
        self.accepted_features += 1;
        Ok(())
    }

    fn export_measurement(
        &mut self,
        record: &mut IngestRecord,
        measurement: IntermediateMeasurement<'_>,
        raw_intensity: f64,
        intensity_factors: Option<&IntensityFactors>,
        deferred: bool,
    ) -> Result<()> {
        // Exports need contextual ion ids before the deferred aggregation worker.
        if !deferred || self.export_rows.has_peptides() {
            self.contextual.resolve(
                record,
                self.aggregation.preserves_contextual_canonical_rows(),
            )?;
        }
        self.export_rows.push(
            IntermediateMeasurement {
                peptide: record.peptide,
                ..measurement
            },
            raw_intensity,
            intensity_factors,
        )
    }

    fn rejects_protein_group(
        &self,
        feature: &QpxFeatureRecord,
        protein_group: &str,
        remove_contaminants: bool,
        contaminant_patterns: &[String],
    ) -> bool {
        (remove_contaminants && matches_protein_contaminant(protein_group, contaminant_patterns))
            || self.fails_protein_fdr(feature)
    }

    fn fails_protein_fdr(&self, feature: &QpxFeatureRecord) -> bool {
        fails_protein_fdr(
            self.protein_fdr_allowed.as_ref(),
            &feature.protein_accessions,
        )
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
        if let Some(allowed) = &config.peptide.allowed_charge_states
            && !filters::charge_is_allowed(feature.charge, allowed)
        {
            return false;
        }
        // ModificationFilter, matched against the modified sequence.
        if filters::peptide_has_excluded_modification(
            &feature.peptidoform,
            feature.modifications.as_deref(),
            &config.peptide.exclude_modifications,
        ) {
            return false;
        }
        // MissedCleavageFilter (trypsin), on the canonical sequence.
        if let Some(max) = config.peptide.max_missed_cleavages
            && filters::trypsin_missed_cleavages(&feature.sequence) > max
        {
            return false;
        }
        if let Some(threshold) = config.peptide.fdr_threshold
            && !feature
                .peptide_qvalue
                .is_some_and(|qvalue| qvalue.is_finite() && qvalue <= threshold)
        {
            return false;
        }
        // SequencePatternFilter (peptide.py:415-418): drop canonical sequences
        // matching any user regex. Patterns are pre-compiled in
        // `excluded_sequence_regexes`; empty unless the pipeline sets them.
        !filters::sequence_matches_excluded(&feature.sequence, &self.excluded_sequence_regexes)
    }
}
