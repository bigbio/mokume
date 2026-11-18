mod finish;
mod ingest;
use crate::accessions::register_contextual_peptide;
use crate::aggregation::{AggregationMeasurement, FeatureAggregation};
use crate::exports::IntermediateExports;
use crate::{
    HashMap, HashSet, PeptideId, PreprocessingFilterConfig, ProteinId, Receiver, Regex, Result,
    SampleId, StringIdRegistry,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CellKey {
    pub(crate) protein: ProteinId,
    pub(crate) sample: SampleId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PeptideCellKey {
    pub(crate) protein: ProteinId,
    pub(crate) sample: SampleId,
    pub(crate) peptide: PeptideId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RunQcKey {
    pub(crate) sample: String,
    pub(crate) technical_replicate: i64,
}

/// Key for the DirectLFQ ion table: one entry per (protein, canonical
/// sequence, sample), holding the summed linear intensity. DirectLFQ's "ion"
/// is the bare sequence, so this keys on the canonical peptide id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DirectLfqCellKey {
    pub(crate) protein: ProteinId,
    pub(crate) canonical: PeptideId,
    pub(crate) sample: SampleId,
}

#[derive(Debug)]
pub(crate) struct FeatureToProteinState {
    pub(crate) proteins: StringIdRegistry<ProteinId>,
    peptides: StringIdRegistry<PeptideId>,
    canonical_peptides: StringIdRegistry<PeptideId>,
    conditions: StringIdRegistry<u32>,
    loading_contexts: HashMap<(u32, u32), u32>,
    pub(crate) contextual: ContextualIds,
    pub(crate) samples: StringIdRegistry<SampleId>,
    pub(crate) index: PeptideIndex,
    pub(crate) export_rows: IntermediateExports,
    pub(crate) aggregation: FeatureAggregation,
    pub(crate) accepted_features: usize,
    pub(crate) accepted_measurements: usize,
    /// Exact protein accessions to drop (`--remove_ids`). A feature is rejected
    /// when any parsed accession matches an entry. Empty on
    /// the protein pipeline, which has no `--remove_ids` option.
    pub(crate) remove_protein_ids: HashSet<String>,
    /// Per-`(ProteinName, canonical sequence)` distinct-sample tracking for the
    /// `features2peptides` low-frequency filter. `None` disables tracking (the
    /// protein pipeline and `features2peptides` without
    /// `--remove_low_frequency_peptides`).
    pub(crate) low_frequency: Option<LowFrequencyTracker>,
    /// Keep shared/non-unique peptide rows (Python `--keep-shared-peptides`).
    /// `false` on the protein pipeline, where the per-aggregation unique policy
    /// (`keeps_shared_peptides`) governs instead; set by `features2peptides`.
    pub(crate) keep_shared_peptides: bool,
    /// Opt-in `features2peptides` per-row preprocessing filters (Python
    /// `--filter-config` / `--filter-*`). `None` keeps default load-time
    /// filtering only; `Some` applies the per-row filters during ingest, before
    /// the per-`(protein, sample)` unique-peptide gate (matching Python's
    /// pipeline chain). Group-level filters are validated as unsupported upstream.
    pub(crate) peptide_filters: Option<PreprocessingFilterConfig>,
    /// Protein groups that pass an explicitly requested protein-group q-value
    /// cutoff. `None` means no protein FDR filter was requested; `Some` is
    /// populated by a source pre-pass so every row of a passing protein group is
    /// retained, matching the Python group-minimum FDR contract.
    pub(crate) protein_fdr_allowed: Option<HashSet<String>>,
    /// `(sample, technical replicate)` runs dropped by the Run-QC group filters
    /// (Python `run_qc.py`: total intensity, feature/protein counts and missing
    /// rate). Computed in a pre-pass over the same initial-filtered features;
    /// ingest skips only the rejected technical run, not its whole sample. Empty
    /// when no Run-QC threshold is active. The normalization median is computed
    /// before Run-QC, matching Python's `SQLFilterBuilder` ordering.
    pub(crate) run_qc_excluded_runs: HashSet<RunQcKey>,
    /// Pre-compiled `exclude_sequence_patterns` regexes (Python
    /// `SequencePatternFilter`, peptide.py:415-418). Empty unless a
    /// `features2peptides` filter pipeline supplies non-empty patterns; compiled
    /// once so the per-row check never recompiles. This filter only drops rows in
    /// the ingest pipeline (not in the `SQLFilterBuilder` median pre-pass), so it
    /// does not shift the normalization median -- no two-pass handling is needed.
    pub(crate) excluded_sequence_regexes: Vec<Regex>,
    /// Per-sample `(lower, upper)` intensity bounds for the QuantileFilter (Python
    /// `intensity.py:293-297`): a feature is dropped when its raw intensity falls
    /// outside `[lower, upper]` (double-closed). Empty unless `quantile_lower > 0`
    /// or `quantile_upper < 1`. The bounds are computed in a pre-pass over the rows
    /// that survive the per-protein `min_unique` gate (Python applies that gate
    /// before the filter pipeline, peptide.py:278-281), so singleton-protein
    /// intensities do not move the bounds. Like Run-QC, the quantile drop happens
    /// after the median pre-pass, so it does not shift the normalization median.
    pub(crate) quantile_bounds: HashMap<String, (f64, f64)>,
    /// `(sample, ProteinName, PeptideCanonical)` triples dropped by the
    /// CVThresholdFilter (Python `intensity.py:111-154`): within one sample, a
    /// canonical whose raw intensities have a coefficient of variation (ddof = 1)
    /// strictly greater than `cv_threshold` is removed (a `NaN`/`None` CV --
    /// single measurement or non-positive mean -- passes). Computed in the same
    /// pre-pass as the quantile bounds so the bounds see the post-CV survivors
    /// (Python orders CV before Quantile, `intensity.py` factory). Empty unless
    /// `cv_threshold` is set. Ingest skips these triples before the quantile drop.
    pub(crate) cv_dropped: HashSet<(String, String, String)>,
    /// When set, the ReplicateAgreementFilter degenerates to a whole-output wipe
    /// (`min_replicate_agreement > 1`): Python's per-sample `nunique(SampleID)` is
    /// always 1, so the `>= 2` threshold drops every row. Ingest rejects all
    /// features so the output is header-only, matching Python.
    pub(crate) replicate_agreement_wipes_all: bool,
}

/// Per-peptide canonical lookups and the unique canonical peptides of each
/// (protein, sample) cell, filled by ingest.
#[derive(Debug, Default)]
pub(crate) struct PeptideIndex {
    peptide_to_canonical: HashMap<PeptideId, PeptideId>,
    peptide_to_contextual_canonical: HashMap<PeptideId, PeptideId>,
    unique_peptides: HashMap<CellKey, HashSet<PeptideId>>,
}

impl PeptideIndex {
    fn add(&mut self, record: &IngestRecord, contextual: bool) {
        self.peptide_to_canonical
            .entry(record.peptide)
            .or_insert(record.canonical);
        if contextual {
            self.peptide_to_contextual_canonical
                .entry(record.peptide)
                .or_insert(record.contextual_canonical);
        }
        self.unique_peptides
            .entry(CellKey {
                protein: record.protein,
                sample: record.sample,
            })
            .or_default()
            .insert(record.canonical);
    }

    pub(crate) fn add_batches(&mut self, batches: Receiver<Vec<IngestRecord>>, contextual: bool) {
        for batch in batches {
            for record in &batch {
                self.add(record, contextual);
            }
        }
    }
}

/// Peptide and canonical peptide ids within a loading context, numbered in
/// first-seen order.
#[derive(Debug, Default)]
pub(crate) struct ContextualIds {
    peptides: HashMap<(PeptideId, u32), PeptideId>,
    canonical_peptides: HashMap<(PeptideId, u32), PeptideId>,
}

impl ContextualIds {
    /// Replace the record's peptide ids by their ids within its pending
    /// loading context, if any.
    pub(crate) fn resolve(&mut self, record: &mut IngestRecord, canonical: bool) -> Result<()> {
        let Some(context) = record.context.take() else {
            return Ok(());
        };
        record.peptide =
            register_contextual_peptide(&mut self.peptides, (record.peptide, context), "peptide")?;
        if canonical {
            record.contextual_canonical = register_contextual_peptide(
                &mut self.canonical_peptides,
                (record.canonical, context),
                "contextual canonical peptide",
            )?;
        }
        Ok(())
    }
}

/// The ids of one accepted feature: all the aggregation and the peptide index
/// need once ingest has resolved its names and filters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IngestRecord {
    protein: ProteinId,
    sample: SampleId,
    peptide: PeptideId,
    canonical: PeptideId,
    contextual_canonical: PeptideId,
    species: PeptideId,
    /// Loading context whose peptide ids [`ContextualIds::resolve`] has yet
    /// to assign.
    context: Option<u32>,
    intensity: f64,
    charge: i32,
    /// Quantified by match-between-runs without an identification in its run.
    transferred: bool,
}

impl IngestRecord {
    pub(crate) fn measurement<'a>(
        &self,
        ion_name: &'a str,
        sample_name: &'a str,
    ) -> AggregationMeasurement<'a> {
        AggregationMeasurement {
            protein: self.protein,
            sample: self.sample,
            peptide: self.peptide,
            canonical: self.canonical,
            species: self.species,
            intensity: self.intensity,
            ion_name,
            sample_name,
            charge: self.charge,
            transferred: self.transferred,
        }
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
pub(crate) struct LowFrequencyTracker {
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
