use crate::aggregation::tech_replicate_of;
use crate::state::RunQcKey;
use crate::{
    HashMap, HashSet, PeptideId, QpxFeatureRecord, Result, SdrfLookupCache, SdrfRecord,
    StringIdRegistry, info, invalid_input,
};

pub(crate) fn protein_group_name(accessions: &[String]) -> Option<String> {
    let proteins = accessions
        .iter()
        .map(|accession| parse_protein_accession(accession))
        .filter(|accession| !accession.is_empty())
        .collect::<Vec<_>>();

    (!proteins.is_empty()).then(|| proteins.join(";"))
}

pub(crate) fn first_protein_name(accessions: &[String]) -> Option<String> {
    accessions
        .first()
        .map(|accession| parse_protein_accession(accession))
        .filter(|accession| !accession.is_empty() && !accession.contains(';'))
        .map(str::to_owned)
}

pub(crate) fn parse_protein_accession(accession: &str) -> &str {
    let accession = accession.trim();
    let Some((database, rest)) = accession.split_once('|') else {
        return accession;
    };
    if ["sp", "tr", "sw", "nxp", "ups"]
        .iter()
        .any(|prefix| database.eq_ignore_ascii_case(prefix))
    {
        rest.split_once('|')
            .map_or(rest, |(accession, _)| accession)
    } else {
        database
    }
}

pub(crate) fn sdrf_record<'a>(
    feature: &QpxFeatureRecord,
    sdrf: Option<&mut SdrfLookupCache<'a>>,
) -> Result<Option<&'a SdrfRecord>> {
    sdrf.map(|lookup| lookup.lookup(&feature.run_file_name, feature.label.as_deref()))
        .transpose()
}

pub(crate) fn sample_name(feature: &QpxFeatureRecord, sdrf_record: Option<&SdrfRecord>) -> String {
    if let Some(record) = sdrf_record {
        return record.sample_accession.clone();
    }

    feature
        .sample_accession
        .clone()
        .unwrap_or_else(|| feature.run_file_name.clone())
}

pub(crate) fn run_qc_key(
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

pub(crate) fn peptide_key(feature: &QpxFeatureRecord) -> String {
    format!("{}|z{}", feature.peptidoform, feature.charge)
}

pub(crate) fn is_contaminant(accession: &str) -> bool {
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
pub(crate) fn matches_sql_contaminant(accessions: &[String], patterns: &[String]) -> bool {
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
pub(crate) fn matches_protein_contaminant(protein_group: &str, patterns: &[String]) -> bool {
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
pub(crate) fn strip_contaminant_members(
    protein_group: &str,
    patterns: &[String],
) -> Option<String> {
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
pub(crate) fn find_contaminant_twins(
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

pub(crate) fn rename_contaminant_twins(
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
pub(crate) fn drop_contaminant_accessions(
    accession_peptides: &mut HashMap<String, HashSet<String>>,
) -> usize {
    let before = accession_peptides.len();
    accession_peptides.retain(|accession, _| !is_contaminant(accession));
    before - accession_peptides.len()
}

/// Drop contaminant accessions from a piBAQ digest; returns their non-contaminant twins.
pub(crate) fn exclude_digest_contaminants(
    accession_peptides: &mut HashMap<String, HashSet<String>>,
) -> HashMap<String, String> {
    let twins = find_contaminant_twins(accession_peptides);
    let removed = drop_contaminant_accessions(accession_peptides);
    info!(
        removed,
        twins = twins.len(),
        "piBAQ digest: contaminant accessions excluded from families"
    );
    twins
}

pub(crate) fn register_id<I>(
    registry: &mut StringIdRegistry<I>,
    value: &str,
    namespace: &str,
) -> Result<I>
where
    I: Copy + From<u32> + Into<u32>,
{
    registry
        .get_or_insert(value)
        .ok_or_else(|| invalid_input(format!("{namespace} id registry overflow")))
}

pub(crate) fn register_loading_context(
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

pub(crate) fn register_contextual_peptide(
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
