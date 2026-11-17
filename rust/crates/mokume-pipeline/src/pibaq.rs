pub(crate) mod allocation;
pub(crate) mod evidence;
mod families;
pub(crate) mod fasta;
use crate::accessions::exclude_digest_contaminants;
use crate::aggregation::{
    AggregationMeasurement, PeptideSampleKey, PibaqAggregation, PibaqFeatureAggregation,
    ProteinFamily,
};
use crate::from_peptides::PibaqDigest;
use crate::peptide_normalization::lfq_traces::quantile_normalized_assignments;
use crate::pibaq::allocation::{
    FamilyCounts, PibaqCell, PibaqObservation, finalize_family_allocation, finalize_family_resolved,
};
use crate::pibaq::evidence::{
    PibaqEvidenceRow, PibaqRowType, family_evidence_rows, write_pibaq_evidence,
};
use crate::pibaq::families::{
    assign_peptides_to_owning_family, count_unique_anchors, discover_families,
    invert_peptide_index, invert_peptide_ownership, load_family_overrides, merge_family_overrides,
};
use crate::state::CellKey;
use crate::{
    FeatureToProteinsConfig, HashMap, HashSet, PeptideId, PibaqSharedMode, ProteinId, Result,
    SampleId, StringIdRegistry, filters, info, invalid_input,
};

impl PibaqFeatureAggregation {
    pub(crate) fn new(core: PibaqAggregation) -> Self {
        Self {
            core,
            ion_observations: HashMap::new(),
            ion_to_canonical: HashMap::new(),
            canonical_names: HashMap::new(),
        }
    }

    pub(crate) fn push(&mut self, measurement: AggregationMeasurement<'_>) {
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

    pub(crate) fn apply_quantile_normalization(&mut self) {
        self.collapse_ions();
        self.core.apply_quantile_normalization();
    }

    pub(crate) fn finalize(
        mut self,
        proteins: &mut StringIdRegistry<ProteinId>,
    ) -> Result<HashMap<CellKey, f64>> {
        self.collapse_ions();
        self.core.finalize(proteins)
    }
}

impl PibaqAggregation {
    pub(crate) fn from_digest(
        config: &FeatureToProteinsConfig,
        digest: PibaqDigest,
    ) -> Result<Self> {
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
        let contaminant_twins = if config.filtering.remove_contaminants {
            exclude_digest_contaminants(&mut accession_peptides)
        } else {
            HashMap::new()
        };
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
            shared_mode: config.pibaq.shared_mode,
            family_rows: config.pibaq.resolved_family_rows(),
            evidence_output: config.pibaq.evidence_output.clone(),
        })
    }

    pub(crate) fn push(
        &mut self,
        sample: SampleId,
        peptide: PeptideId,
        peptide_name: &str,
        intensity: f64,
    ) {
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

    fn finalize(self, proteins: &mut StringIdRegistry<ProteinId>) -> Result<HashMap<CellKey, f64>> {
        let evidence_output = self.evidence_output.clone();
        let shared_mode = self.shared_mode;
        let (detailed, evidence) = self.finalize_detailed(proteins);
        info!(
            shared_mode = shared_mode.label(),
            family_rows = evidence
                .iter()
                .filter(|row| row.row_type == PibaqRowType::Family)
                .count(),
            members_not_quantified = evidence
                .iter()
                .filter(|row| row.row_type == PibaqRowType::MemberNotQuantified)
                .count(),
            "piBAQ shared-peptide handling"
        );
        if let Some(path) = evidence_output {
            write_pibaq_evidence(&path, &evidence, shared_mode)?;
        }
        Ok(detailed
            .into_iter()
            .map(|(key, detail)| (key, detail.cell.pibaq))
            .collect())
    }

    /// Run the full piBAQ allocation and return, per (protein, sample), both the
    /// piBAQ value and the inputs `peptides2protein` needs to reproduce the
    /// Python long-format table (`NormIntensity`, `FamilyId`, `EvidenceLevel`,
    /// `FamilySize`). The allocation math is identical to [`Self::finalize`];
    /// only the captured metadata differs, so there is a single source of truth
    /// for the algorithm.
    pub(crate) fn finalize_detailed(
        mut self,
        proteins: &mut StringIdRegistry<ProteinId>,
    ) -> (HashMap<CellKey, PibaqDetailedCell>, Vec<PibaqEvidenceRow>) {
        let observations = std::mem::take(&mut self.observations)
            .into_iter()
            .filter_map(|(key, intensity)| {
                self.peptide_names
                    .get(&key.peptide)
                    .map(|peptide| (peptide.clone(), key.sample, intensity))
            })
            .collect::<Vec<_>>();
        if observations.is_empty() {
            return (HashMap::new(), Vec::new());
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
        let mut observations_by_family = HashMap::<String, Vec<PibaqObservation>>::new();
        for observation in observations {
            if let Some(owner) = peptide_owner.get(&observation.0) {
                observations_by_family
                    .entry(owner.clone())
                    .or_default()
                    .push(observation);
            }
        }

        let mut detailed = HashMap::new();
        let mut evidence_rows = Vec::new();
        for family in &self.families {
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
            let evidence = self.family_evidence(family, &anchor_counts);
            let family_size = family.members.len();
            let (output, rows) =
                self.allocate_family(family, &observations, &owned_peptides, evidence, proteins);
            evidence_rows.extend(rows);
            for (key, cell) in output {
                let molecular_weight = self.mw_map.as_ref().map(|mw_map| {
                    let member_mw = proteins
                        .resolve(key.protein)
                        .and_then(|member| mw_map.get(member).copied())
                        .unwrap_or(0.0);
                    if member_mw == 0.0 { 1.0 } else { member_mw }
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
        (detailed, evidence_rows)
    }

    /// Bucket a family by the unique-peptide anchors of its weakest and strongest member.
    fn family_evidence(
        &self,
        family: &ProteinFamily,
        anchor_counts: &HashMap<String, usize>,
    ) -> PibaqEvidence {
        let anchors = family
            .members
            .iter()
            .map(|member| anchor_counts.get(member).copied().unwrap_or(0));
        classify_evidence(
            anchors.clone().min().unwrap_or(0),
            anchors.max().unwrap_or(0),
            self.min_anchors,
            self.high_anchor_threshold,
        )
    }

    /// Allocate one family under the configured shared-peptide mode and describe its
    /// rows for the `--pibaq-evidence` side-car.
    fn allocate_family(
        &self,
        family: &ProteinFamily,
        observations: &[PibaqObservation],
        owned_peptides: &HashSet<String>,
        evidence: PibaqEvidence,
        proteins: &mut StringIdRegistry<ProteinId>,
    ) -> (HashMap<CellKey, PibaqCell>, Vec<PibaqEvidenceRow>) {
        let counts = FamilyCounts::new(
            family,
            observations,
            owned_peptides,
            &self.accession_peptides,
            &self.peptide_accessions,
        );
        let output = if self.shared_mode == PibaqSharedMode::Proportional {
            finalize_family_allocation(
                family,
                observations,
                owned_peptides,
                &self.accession_peptides,
                &self.peptide_accessions,
                evidence == PibaqEvidence::FamilyOnly,
                proteins,
            )
        } else {
            finalize_family_resolved(
                family,
                observations,
                &counts,
                evidence,
                self.shared_mode,
                self.family_rows,
                proteins,
            )
        };
        let rows = family_evidence_rows(
            family,
            &counts,
            evidence,
            &output,
            self.shared_mode,
            proteins,
        );
        (output, rows)
    }
}

/// Observed peptides (and intensity) that piBAQ drops because the digest lacks them.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct DigestMissSummary {
    pub(crate) observed_peptides: usize,
    pub(crate) dropped_peptides: usize,
    observed_intensity: f64,
    dropped_intensity: f64,
    pub(crate) missed_cleavage: usize,
    pub(crate) too_long: usize,
    pub(crate) met_removed_n_term: usize,
    pub(crate) other: usize,
}

impl DigestMissSummary {
    pub(crate) fn dropped_intensity_fraction(&self) -> f64 {
        if self.observed_intensity > 0.0 {
            self.dropped_intensity / self.observed_intensity
        } else {
            0.0
        }
    }
}

pub(crate) fn summarize_digest_misses(
    observations: &[PibaqObservation],
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
pub(crate) enum PibaqEvidence {
    FamilyOnly,
    Medium,
    High,
}

impl PibaqEvidence {
    /// Lowercase label matching the Python `EVIDENCE_*` constants.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::FamilyOnly => "family_only",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Bucket a family's anchor counts, mirroring the Python `_classify_evidence`.
pub(crate) fn classify_evidence(
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
pub(crate) struct PibaqDetailedCell {
    pub(crate) cell: PibaqCell,
    pub(crate) family_id: String,
    pub(crate) evidence_level: PibaqEvidence,
    pub(crate) family_size: usize,
    /// Per-protein molecular weight (member MW with 0 -> 1). Populated only
    /// when the caller requested TPA; mirrors the Python `_attach_tpa`
    /// `MolecularWeight` column.
    pub(crate) molecular_weight: Option<f64>,
    /// TPA = `NormIntensity / MolecularWeight`, populated alongside
    /// `molecular_weight`.
    pub(crate) tpa: Option<f64>,
}
