use crate::accessions::register_id;
use crate::aggregation::{PibaqAggregation, ProteinFamily};
use crate::pibaq::fasta::load_fasta_mw;
use crate::{
    BTreeSet, BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig, DirectLfqIon,
    FeatureToProteinsConfig, FilterConfig, HashMap, HashSet, ImputationConfig, InputConfig,
    IrsConfig, MaxLfqConfig, NormalizationConfig, OutputConfig, OutputFormat, PathBuf, PeptideId,
    PibaqConfig, PibaqFamilyRows, PibaqSharedMode, ProteinId, QuantMethod, RatioConfig, Result,
    RuntimeConfig, SampleId, StringIdRegistry, TmtConfig, direct_lfq_aligned, threading,
};

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
    /// Relative reliability of the measurement in MaxLFQ pair medians
    /// (1 when unweighted); DirectLFQ ignores it.
    pub weight: f64,
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
/// ions), used by `quantify peptides2protein --quant-method directlfq`.
/// `num_samples_quadratic` is DirectLFQ's global-stage knob (the
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
    for (protein, per_sample) in direct_lfq_aligned(ions, min_nonan, num_samples_quadratic, true) {
        let Some(protein_name) = proteins.resolve(protein) else {
            continue;
        };
        for (sample, intensity) in per_sample {
            if intensity > 0.0
                && let Some(sample_name) = samples.resolve(sample)
            {
                result.push(LfqProteinIntensity {
                    protein: protein_name.to_owned(),
                    sample: sample_name.to_owned(),
                    intensity,
                });
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
        shared_mode: PibaqSharedMode::Proportional,
        family_rows: PibaqFamilyRows::None,
        evidence_output: None,
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
    let (detailed, _) = aggregation.finalize_detailed(&mut proteins);

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
            fasta: Some(params.fasta.clone()),
            ..InputConfig::default()
        },
        output: OutputConfig {
            protein_matrix: PathBuf::new(),
            export_peptides: None,
            export_unnormalized_peptides: None,
            export_unimputed: None,
            export_ions: None,
            export_precision: None,
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
        pibaq: peptides_pibaq_config(params),
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

/// `peptides2protein` keeps the proportional shared-peptide split.
fn peptides_pibaq_config(params: &PibaqFromPeptidesParams) -> PibaqConfig {
    PibaqConfig {
        enzyme: params.enzyme.clone(),
        max_aa: params.max_aa,
        min_shared: params.min_shared,
        families_yaml: params.families_yaml.clone(),
        min_anchors: params.min_anchors,
        high_anchor_threshold: params.high_anchor_threshold,
        missed_cleavages: 0,
        shared_mode: PibaqSharedMode::Proportional,
        family_rows: None,
        evidence_output: None,
    }
}
