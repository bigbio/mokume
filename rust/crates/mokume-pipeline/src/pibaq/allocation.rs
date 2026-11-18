use crate::aggregation::ProteinFamily;
use crate::pibaq::PibaqEvidence;
use crate::state::CellKey;
use crate::{
    BTreeMap, HashMap, HashSet, PibaqFamilyRows, PibaqSharedMode, ProteinId, SampleId,
    StringIdRegistry, median_finite,
};

/// One piBAQ output cell: the allocated numerator (`norm_intensity`, the
/// per-(protein, sample) shared-aware intensity that the piBAQ allocator sums)
/// and the piBAQ value (numerator divided by the owned theoretical peptide
/// count). Keeping both lets `peptides2protein` reproduce the Python
/// `peptides_to_protein` long-format `NormIntensity` + `PiBAQ` columns while
/// `features2proteins` projects out only the piBAQ value.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PibaqCell {
    pub(crate) norm_intensity: f64,
    pub(crate) pibaq: f64,
}

pub(crate) fn finalize_family_allocation(
    family: &ProteinFamily,
    observations: &[PibaqObservation],
    owned_peptides: &HashSet<String>,
    accession_peptides: &HashMap<String, HashSet<String>>,
    peptide_accessions: &HashMap<String, HashSet<String>>,
    force_equal_shared: bool,
    proteins: &mut StringIdRegistry<ProteinId>,
) -> HashMap<CellKey, PibaqCell> {
    let mut output = HashMap::new();
    let peptide_members = observed_family_members(family, observations, peptide_accessions);
    let (anchor_intensity, mut combined) =
        unique_family_intensities(observations, &peptide_members);
    add_shared_family_intensities(
        observations,
        &peptide_members,
        &anchor_intensity,
        &mut combined,
        force_equal_shared,
    );

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

fn observed_family_members(
    family: &ProteinFamily,
    observations: &[PibaqObservation],
    peptide_accessions: &HashMap<String, HashSet<String>>,
) -> HashMap<String, Vec<String>> {
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
    peptide_members
}

type FamilyIntensities = HashMap<(String, SampleId), f64>;

fn unique_family_intensities(
    observations: &[PibaqObservation],
    peptide_members: &HashMap<String, Vec<String>>,
) -> (FamilyIntensities, FamilyIntensities) {
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
    (anchor_intensity, combined)
}

fn add_shared_family_intensities(
    observations: &[PibaqObservation],
    peptide_members: &HashMap<String, Vec<String>>,
    anchor_intensity: &FamilyIntensities,
    combined: &mut FamilyIntensities,
    force_equal_shared: bool,
) {
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

/// Median IRS factor per plex (geometric) over proteins that have a reference there.
pub(crate) fn plex_median_factors(
    factors: &HashMap<ProteinId, HashMap<String, f64>>,
) -> HashMap<String, f64> {
    let mut by_plex = HashMap::<String, Vec<f64>>::new();
    for by_protein in factors.values() {
        for (plex, factor) in by_protein {
            if *factor > 0.0 && factor.is_finite() {
                by_plex.entry(plex.clone()).or_default().push(factor.ln());
            }
        }
    }
    by_plex
        .into_iter()
        .filter_map(|(plex, mut logs)| median_finite(&mut logs).map(|median| (plex, median.exp())))
        .collect()
}

/// Per-member theoretical and observed peptide counts for one piBAQ family.
#[derive(Debug, Default)]
pub(crate) struct FamilyCounts {
    /// Observed peptide -> sorted indices of the family members containing it.
    peptide_members: HashMap<String, Vec<usize>>,
    pub(crate) theoretical_total: Vec<usize>,
    pub(crate) theoretical_unique: Vec<usize>,
    pub(crate) observed_unique: Vec<usize>,
    pub(crate) observed_shared: Vec<usize>,
    /// Theoretical peptides the family owns (the family-row denominator).
    pub(crate) owned_count: usize,
}

impl FamilyCounts {
    pub(crate) fn new(
        family: &ProteinFamily,
        observations: &[PibaqObservation],
        owned_peptides: &HashSet<String>,
        accession_peptides: &HashMap<String, HashSet<String>>,
        peptide_accessions: &HashMap<String, HashSet<String>>,
    ) -> Self {
        let index = family
            .members
            .iter()
            .enumerate()
            .map(|(position, member)| (member.as_str(), position))
            .collect::<HashMap<_, _>>();
        let members_of = |peptide: &str| {
            let mut members = peptide_accessions
                .get(peptide)
                .map(|accessions| {
                    accessions
                        .iter()
                        .filter_map(|accession| index.get(accession.as_str()).copied())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            members.sort_unstable();
            members
        };
        let size = family.members.len();
        let mut counts = Self {
            peptide_members: HashMap::new(),
            theoretical_total: vec![0; size],
            theoretical_unique: vec![0; size],
            observed_unique: vec![0; size],
            observed_shared: vec![0; size],
            owned_count: owned_peptides.len(),
        };
        counts.count_theoretical(family, owned_peptides, accession_peptides, &members_of);
        counts.count_observed(observations, &members_of);
        counts
    }

    fn count_theoretical(
        &mut self,
        family: &ProteinFamily,
        owned_peptides: &HashSet<String>,
        accession_peptides: &HashMap<String, HashSet<String>>,
        members_of: &impl Fn(&str) -> Vec<usize>,
    ) {
        for (position, member) in family.members.iter().enumerate() {
            let Some(peptides) = accession_peptides.get(member) else {
                continue;
            };
            for peptide in peptides.intersection(owned_peptides) {
                self.theoretical_total[position] += 1;
                if members_of(peptide).len() == 1 {
                    self.theoretical_unique[position] += 1;
                }
            }
        }
    }

    fn count_observed(
        &mut self,
        observations: &[PibaqObservation],
        members_of: &impl Fn(&str) -> Vec<usize>,
    ) {
        for (peptide, _, _) in observations {
            if self.peptide_members.contains_key(peptide) {
                continue;
            }
            let members = members_of(peptide);
            if members.is_empty() {
                continue;
            }
            if let [only] = members.as_slice() {
                self.observed_unique[*only] += 1;
            } else {
                for member in &members {
                    self.observed_shared[*member] += 1;
                }
            }
            self.peptide_members.insert(peptide.clone(), members);
        }
    }

    pub(crate) fn observed_shared_peptides(&self) -> usize {
        self.peptide_members
            .values()
            .filter(|members| members.len() > 1)
            .count()
    }
}

/// One observed peptide sequence with its sample and intensity.
pub(crate) type PibaqObservation = (String, SampleId, f64);
/// Per-member summed intensity by sample, indexed like `ProteinFamily::members`.
type MemberSums = Vec<BTreeMap<SampleId, f64>>;
/// One shared-peptide observation: the members sharing it, its sample and intensity.
type SharedObservation<'a> = (&'a [usize], SampleId, f64);

/// Member values from unique evidence (optionally plus stable-ratio shared signal) and
/// family rows; never invents member values from shared peptides alone.
pub(crate) fn finalize_family_resolved(
    family: &ProteinFamily,
    observations: &[PibaqObservation],
    counts: &FamilyCounts,
    evidence: PibaqEvidence,
    mode: PibaqSharedMode,
    family_rows: PibaqFamilyRows,
    proteins: &mut StringIdRegistry<ProteinId>,
) -> HashMap<CellKey, PibaqCell> {
    let mut output = HashMap::new();
    let size = family.members.len();
    let mut ordered = observations.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    if size == 1 || evidence != PibaqEvidence::FamilyOnly {
        let (unique, shared) = split_member_evidence(&ordered, counts, size);
        let (numerators, denominators) = if mode == PibaqSharedMode::StableRatio {
            (
                stable_ratio_numerators(&unique, &shared, counts),
                &counts.theoretical_total,
            )
        } else {
            (unique, &counts.theoretical_unique)
        };
        insert_member_values(family, numerators, denominators, proteins, &mut output);
    }
    if size > 1 && wants_family_row(family_rows, evidence) && counts.owned_count > 0 {
        insert_family_row(family, &ordered, counts, proteins, &mut output);
    }
    output
}

/// Sum unique-peptide intensity per member and collect the shared-peptide observations.
fn split_member_evidence<'a>(
    ordered: &[&PibaqObservation],
    counts: &'a FamilyCounts,
    size: usize,
) -> (MemberSums, Vec<SharedObservation<'a>>) {
    let mut unique = vec![BTreeMap::<SampleId, f64>::new(); size];
    let mut shared = Vec::<SharedObservation<'a>>::new();
    for (peptide, sample, intensity) in ordered {
        let Some(members) = counts.peptide_members.get(peptide) else {
            continue;
        };
        if let [only] = members.as_slice() {
            *unique[*only].entry(*sample).or_insert(0.0) += *intensity;
        } else {
            shared.push((members.as_slice(), *sample, *intensity));
        }
    }
    (unique, shared)
}

/// Unique sums plus each shared peptide split by stable member ratios, credited only
/// to members with unique evidence in that sample.
fn stable_ratio_numerators(
    unique: &MemberSums,
    shared: &[SharedObservation<'_>],
    counts: &FamilyCounts,
) -> MemberSums {
    let mut numerators = unique.clone();
    let mut ratio_cache = HashMap::<&[usize], Vec<f64>>::new();
    for (members, sample, intensity) in shared {
        let ratios = ratio_cache
            .entry(members)
            .or_insert_with(|| stable_member_ratios(members, unique, &counts.theoretical_unique));
        for (member, ratio) in members.iter().zip(ratios.iter()) {
            if *ratio > 0.0 && unique[*member].get(sample).is_some_and(|u| *u > 0.0) {
                *numerators[*member].entry(*sample).or_insert(0.0) += ratio * intensity;
            }
        }
    }
    numerators
}

fn insert_member_values(
    family: &ProteinFamily,
    numerators: MemberSums,
    denominators: &[usize],
    proteins: &mut StringIdRegistry<ProteinId>,
    output: &mut HashMap<CellKey, PibaqCell>,
) {
    for (position, cells) in numerators.into_iter().enumerate() {
        let denominator = denominators[position];
        if denominator == 0 {
            continue;
        }
        for (sample, numerator) in cells {
            insert_pibaq_value(
                &family.members[position],
                sample,
                numerator,
                numerator / denominator as f64,
                proteins,
                output,
            );
        }
    }
}

fn wants_family_row(family_rows: PibaqFamilyRows, evidence: PibaqEvidence) -> bool {
    match family_rows {
        PibaqFamilyRows::None => false,
        PibaqFamilyRows::FamilyOnly => evidence == PibaqEvidence::FamilyOnly,
        PibaqFamilyRows::All => true,
    }
}

/// One family-level row: all owned peptide intensity over the owned peptide count.
fn insert_family_row(
    family: &ProteinFamily,
    ordered: &[&PibaqObservation],
    counts: &FamilyCounts,
    proteins: &mut StringIdRegistry<ProteinId>,
    output: &mut HashMap<CellKey, PibaqCell>,
) {
    let mut totals = BTreeMap::<SampleId, f64>::new();
    for (peptide, sample, intensity) in ordered {
        if counts.peptide_members.contains_key(peptide) {
            *totals.entry(*sample).or_insert(0.0) += *intensity;
        }
    }
    let row_id = family_row_id(family);
    for (sample, numerator) in totals {
        insert_pibaq_value(
            &row_id,
            sample,
            numerator,
            numerator / counts.owned_count as f64,
            proteins,
            output,
        );
    }
}

pub(crate) fn family_row_id(family: &ProteinFamily) -> String {
    family.members.join(";")
}

/// Cross-sample member shares for one shared-peptide member set: the median share over
/// samples where every member has unique evidence, else each member's median level.
fn stable_member_ratios(
    members: &[usize],
    unique: &[BTreeMap<SampleId, f64>],
    theoretical_unique: &[usize],
) -> Vec<f64> {
    let mut shares = co_observed_member_shares(members, unique, theoretical_unique);
    let mut ratios = if shares.first().is_some_and(|values| !values.is_empty()) {
        shares
            .iter_mut()
            .map(|values| sorted_median(values))
            .collect::<Vec<_>>()
    } else {
        members
            .iter()
            .map(|member| {
                let mut levels = unique[*member]
                    .keys()
                    .filter_map(|sample| unique_level(unique, theoretical_unique, *member, sample))
                    .collect::<Vec<_>>();
                sorted_median(&mut levels)
            })
            .collect::<Vec<_>>()
    };
    let total = ratios.iter().sum::<f64>();
    for ratio in &mut ratios {
        *ratio = if total > 0.0 { *ratio / total } else { 0.0 };
    }
    ratios
}

/// Each member's share of the summed levels in every sample where all members have
/// unique evidence.
fn co_observed_member_shares(
    members: &[usize],
    unique: &[BTreeMap<SampleId, f64>],
    theoretical_unique: &[usize],
) -> Vec<Vec<f64>> {
    let mut shares = vec![Vec::<f64>::new(); members.len()];
    let Some(first) = members.first() else {
        return shares;
    };
    for sample in unique[*first].keys() {
        let levels = members
            .iter()
            .map(|member| unique_level(unique, theoretical_unique, *member, sample))
            .collect::<Option<Vec<_>>>();
        let Some(levels) = levels else {
            continue;
        };
        let total = levels.iter().sum::<f64>();
        for (slot, value) in shares.iter_mut().zip(levels) {
            slot.push(value / total);
        }
    }
    shares
}

/// A member's unique intensity per theoretical unique peptide, if it has any.
fn unique_level(
    unique: &[BTreeMap<SampleId, f64>],
    theoretical_unique: &[usize],
    member: usize,
    sample: &SampleId,
) -> Option<f64> {
    let denominator = theoretical_unique[member];
    unique[member]
        .get(sample)
        .copied()
        .filter(|value| *value > 0.0 && denominator > 0)
        .map(|value| value / denominator as f64)
}

fn sorted_median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.is_empty() {
        0.0
    } else if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}
