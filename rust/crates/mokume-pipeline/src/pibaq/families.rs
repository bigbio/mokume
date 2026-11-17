use crate::aggregation::ProteinFamily;
use crate::{HashMap, HashSet, MokumeError, Path, Result, invalid_input, read_to_string};

pub(crate) fn invert_peptide_index(
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

pub(crate) fn discover_families(
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

pub(crate) fn load_family_overrides(path: &Path) -> Result<Vec<ProteinFamily>> {
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

pub(crate) fn merge_family_overrides(
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

pub(crate) fn count_unique_anchors(
    observed_peptides: &HashSet<String>,
    peptide_accessions: &HashMap<String, HashSet<String>>,
) -> HashMap<String, usize> {
    let mut counts = HashMap::<String, usize>::new();
    for peptide in observed_peptides {
        let Some(accessions) = peptide_accessions.get(peptide) else {
            continue;
        };
        if accessions.len() == 1
            && let Some(accession) = accessions.iter().next()
        {
            *counts.entry(accession.clone()).or_insert(0) += 1;
        }
    }
    counts
}

pub(crate) fn assign_peptides_to_owning_family(
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

pub(crate) fn invert_peptide_ownership(
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
