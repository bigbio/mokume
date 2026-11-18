mod feature;
pub(crate) mod maxlfq;
use crate::solvers::DirectLfqCells;
use crate::state::{CellKey, PeptideCellKey};
use crate::{
    HashMap, HashSet, PathBuf, PeptideId, PibaqFamilyRows, PibaqSharedMode, ProteinId, QuantMethod,
    RunCellKey, RunNormalizationTransform, SampleId,
};

#[derive(Debug)]
pub(crate) enum FeatureAggregation {
    Sum(HashMap<CellKey, HashMap<PeptideId, f64>>),
    Median(HashMap<CellKey, HashMap<PeptideId, f64>>),
    Abd(HashMap<CellKey, HashMap<PeptideId, f64>>),
    PeptideCount(HashMap<CellKey, HashMap<PeptideId, f64>>),
    Pibaq(Box<PibaqFeatureAggregation>),
    Ratio(RatioAggregation),
    TopN {
        topn: usize,
        cells: HashMap<CellKey, HashMap<PeptideId, f64>>,
        /// Weights keyed like `cells` when transferred features are
        /// down-weighted: selection then takes identified peptides first.
        weights: Option<FeatureWeights>,
    },
    Lfq {
        method: QuantMethod,
        /// DirectLFQ alone performs hierarchical sample/ion alignment.
        route_to_directlfq: bool,
        maxlfq_min_ratio_count: usize,
        stabilize: bool,
        directlfq_min_nonan: usize,
        directlfq_num_samples_quadratic: usize,
        directlfq_normalize_samples: bool,
        /// Contextual ion maxima. MaxLFQ preserves modification and charge;
        /// DirectLFQ collapses them into canonical peptide traces.
        traces: MaxLfqFeatureAggregation,
        /// Also estimate each cell's quantification precision for
        /// `--de-precision-weights`.
        cell_precision: bool,
        /// Populated only when DirectLFQ ion export runs before matrix
        /// materialization, avoiding a second solver pass.
        cached_directlfq: Option<DirectLfqCells>,
    },
}

#[derive(Debug, Default)]
pub(crate) struct MaxLfqFeatureAggregation {
    pub(crate) ion_cells: HashMap<CellKey, HashMap<PeptideId, f64>>,
    /// Present when transferred features are down-weighted.
    pub(crate) weights: Option<Box<TraceWeights>>,
    pub(crate) ion_to_canonical: HashMap<PeptideId, PeptideId>,
    pub(crate) ion_to_species: HashMap<PeptideId, PeptideId>,
    pub(crate) peptide_traces: HashMap<PeptideCellKey, f64>,
}

/// Weights keyed like `ion_cells`, and those of `peptide_traces` once the
/// ions are collapsed into traces.
#[derive(Debug)]
pub(crate) struct TraceWeights {
    ions: FeatureWeights,
    traces: HashMap<PeptideCellKey, f64>,
}

impl TraceWeights {
    /// Move the ion weights onto traces through `mapping`, as
    /// [`MaxLfqFeatureAggregation::collapse_ions`] moves the ion cells.
    fn collapse_ions(&mut self, mapping: &HashMap<PeptideId, PeptideId>) {
        for (cell, peptides) in std::mem::take(&mut self.ions.cells) {
            for (peptide, weight) in collapse_weights(peptides, mapping) {
                let key = PeptideCellKey {
                    protein: cell.protein,
                    sample: cell.sample,
                    peptide,
                };
                self.traces.insert(key, weight);
            }
        }
    }
}

/// Feature weights of a quantification that down-weights transferred
/// features (`--transferred-weight`); identified features weigh 1. A peptide
/// observed more than once in a cell keeps its largest weight.
#[derive(Debug)]
pub(crate) struct FeatureWeights {
    transferred: f64,
    cells: HashMap<CellKey, HashMap<PeptideId, f64>>,
}

impl FeatureWeights {
    fn new(transferred: f64) -> Self {
        Self {
            transferred,
            cells: HashMap::new(),
        }
    }

    fn push(&mut self, cell: CellKey, peptide: PeptideId, transferred: bool) {
        let weight = if transferred { self.transferred } else { 1.0 };
        let current = self
            .cells
            .entry(cell)
            .or_default()
            .entry(peptide)
            .or_insert(weight);
        *current = current.max(weight);
    }

    /// Rekey every cell like peptide cells collapsed through `mapping`.
    fn collapse(&mut self, mapping: &HashMap<PeptideId, PeptideId>) {
        for peptides in self.cells.values_mut() {
            *peptides = collapse_weights(std::mem::take(peptides), mapping);
        }
    }

    /// Remove one cell's weights, collapsed through `mapping`.
    fn take_cell(
        &mut self,
        cell: &CellKey,
        mapping: &HashMap<PeptideId, PeptideId>,
    ) -> HashMap<PeptideId, f64> {
        collapse_weights(self.cells.remove(cell).unwrap_or_default(), mapping)
    }
}

/// [`collapse_to_canonical`] for weights: merged peptides keep the largest.
fn collapse_weights(
    peptides: HashMap<PeptideId, f64>,
    mapping: &HashMap<PeptideId, PeptideId>,
) -> HashMap<PeptideId, f64> {
    let mut collapsed = HashMap::with_capacity(peptides.len());
    for (peptide, weight) in peptides {
        let key = mapping.get(&peptide).copied().unwrap_or(peptide);
        let current = collapsed.entry(key).or_insert(weight);
        *current = current.max(weight);
    }
    collapsed
}

#[derive(Debug)]
pub(crate) struct PibaqAggregation {
    pub(crate) accession_peptides: HashMap<String, HashSet<String>>,
    pub(crate) peptide_accessions: HashMap<String, HashSet<String>>,
    pub(crate) families: Vec<ProteinFamily>,
    pub(crate) peptide_names: HashMap<PeptideId, String>,
    pub(crate) observations: HashMap<PeptideSampleKey, f64>,
    pub(crate) min_anchors: usize,
    /// piBAQ "high anchor" threshold (Python `--high-anchor-threshold`). Only
    /// affects the `EvidenceLevel` annotation column via [`classify_evidence`],
    /// never the quantification values (Python `pibaq.py:732`).
    pub(crate) high_anchor_threshold: usize,
    /// Per-canonical monoisotopic molecular weight, populated only when the
    /// caller requests TPA (`peptides2protein --tpa`). `None` leaves the piBAQ
    /// path untouched for every other consumer.
    pub(crate) mw_map: Option<HashMap<String, f64>>,
    /// Digest `max_aa`, used only to classify dropped observed peptides.
    pub(crate) digest_max_aa: Option<usize>,
    pub(crate) contaminant_twins: HashMap<String, String>,
    pub(crate) shared_mode: PibaqSharedMode,
    pub(crate) family_rows: PibaqFamilyRows,
    pub(crate) evidence_output: Option<PathBuf>,
}

#[derive(Debug)]
pub(crate) struct PibaqFeatureAggregation {
    pub(crate) core: PibaqAggregation,
    pub(crate) ion_observations: HashMap<PeptideSampleKey, f64>,
    pub(crate) ion_to_canonical: HashMap<PeptideId, PeptideId>,
    pub(crate) canonical_names: HashMap<PeptideId, String>,
}

#[derive(Debug)]
pub(crate) struct RatioAggregation {
    pub(crate) records: Vec<RatioPsm>,
    pub(crate) sample_names: HashMap<SampleId, String>,
    pub(crate) reference_samples: HashSet<String>,
    /// SDRF-driven `source name` -> plex id map, built once via
    /// `sample_to_plex` (Python's `detect_plexes_from_sdrf`). The reference and
    /// log2-ratio steps look plexes up here so per-plex grouping is keyed by the
    /// authoritative SDRF map rather than re-parsing each sample name.
    pub(crate) sample_to_plex: HashMap<String, String>,
    pub(crate) fraction_merge: RatioFractionMerge,
    pub(crate) min_unique_peptides: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProteinFamily {
    pub(crate) family_id: String,
    pub(crate) members: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PeptideSampleKey {
    pub(crate) peptide: PeptideId,
    pub(crate) sample: SampleId,
}

#[derive(Debug, Clone)]
pub(crate) struct RatioPsm {
    pub(crate) protein: ProteinId,
    pub(crate) peptide: PeptideId,
    pub(crate) charge: i32,
    pub(crate) sample: SampleId,
    pub(crate) intensity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RatioFractionMerge {
    Mean,
    Max,
}

#[derive(Debug, Default)]
pub(crate) struct IntensityFactors {
    pub(crate) run: HashMap<RunCellKey, RunNormalizationTransform>,
    pub(crate) sample: HashMap<String, f64>,
    /// Channel-IRS scale factors keyed by the QPX run identity. A run is one
    /// TMT plex, so this remains unambiguous when technical-replicate numbers
    /// repeat across mixtures. Empty when IRS is disabled or no reference-
    /// channel rows were found.
    pub(crate) irs_scale_by_run: HashMap<String, f64>,
}

/// Derive a peptide row's `TechReplicate` from its run name, matching Python's
/// `apply_initial_filtering` (aggregation.py:211-223): a run name containing an
/// underscore uses the integer after the last underscore; a plain integer run
/// name uses that integer; otherwise Python's per-sample run-index fallback
/// applies, which is `1` whenever each sample maps to a single run (the
/// non-empty, no-SDRF path this port verifies against — the multi-run-per-sample
/// case yields an empty matrix in Python because SDRF enrichment remaps the
/// sample key, a quirk the existing golden tests already document).
pub(crate) fn tech_replicate_of(run_file_name: &str) -> i64 {
    if let Some((_, last)) = run_file_name.rsplit_once('_')
        && let Ok(value) = last.parse::<i64>()
    {
        return value;
    }
    run_file_name.parse::<i64>().unwrap_or(1)
}

impl IntensityFactors {
    pub(crate) fn normalize(&self, intensity: f64, sample: &str, run: &str) -> f64 {
        let sample_factor = self.sample.get(sample).copied().unwrap_or(1.0);
        self.normalize_run(intensity, sample, run) * sample_factor
    }

    pub(crate) fn normalize_run(&self, intensity: f64, sample: &str, run: &str) -> f64 {
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
        // (`peptide.py:370`). Missing runs map to 1.0.
        let irs_scale = self.irs_scale_by_run.get(run).copied().unwrap_or(1.0);
        run_intensity * irs_scale
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.run.is_empty() && self.sample.is_empty() && self.irs_scale_by_run.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AggregationMeasurement<'a> {
    pub(crate) protein: ProteinId,
    pub(crate) sample: SampleId,
    pub(crate) peptide: PeptideId,
    pub(crate) canonical: PeptideId,
    pub(crate) species: PeptideId,
    pub(crate) intensity: f64,
    pub(crate) ion_name: &'a str,
    pub(crate) sample_name: &'a str,
    pub(crate) charge: i32,
    /// Quantified by match-between-runs without an identification in its run.
    pub(crate) transferred: bool,
}
