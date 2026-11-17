use crate::{
    BTreeMap, BTreeSet, HashMap, HashSet, IrsConfig, Result, SdrfRawTable, SdrfTable, info,
    invalid_input, warn,
};

pub(crate) fn sample_plex(sample_name: &str) -> String {
    if let Some(mixture) = sample_name.split('_').find(|part| {
        part.to_ascii_lowercase()
            .strip_prefix("mixture")
            .is_some_and(|suffix| {
                !suffix.is_empty() && suffix.chars().all(|character| character.is_ascii_digit())
            })
    }) {
        return mixture.to_ascii_lowercase();
    }
    let Some((prefix, suffix)) = sample_name.rsplit_once('_') else {
        return "plex1".to_owned();
    };
    let channel = suffix.strip_suffix(['N', 'C', 'n', 'c']).unwrap_or(suffix);
    if !channel.is_empty()
        && channel.chars().all(|character| character.is_ascii_digit())
        && !prefix.is_empty()
    {
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
pub(crate) fn plex_of(sample_to_plex: &HashMap<String, String>, sample_name: &str) -> String {
    sample_to_plex
        .get(sample_name)
        .cloned()
        .unwrap_or_else(|| sample_plex(sample_name))
}

pub(crate) fn sample_to_plex(sdrf: &SdrfTable) -> HashMap<String, String> {
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
pub(crate) enum PlexSource {
    Column,
    DataFiles,
    SourceName,
}

/// Sample -> plex map and the sorted distinct plexes used by IRS.
pub(crate) fn resolve_irs_plexes(
    sdrf: &SdrfTable,
    raw_sdrf: &SdrfRawTable,
    config: &IrsConfig,
) -> Result<(HashMap<String, String>, Vec<String>)> {
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
    Ok((sample_to_plex, plexes))
}

/// Sample -> plex: SDRF column, else shared data files, else legacy source-name rule.
pub(crate) fn derive_sample_plexes(
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
        if let Some(previous) = plexes.insert(sample.to_owned(), plex.to_owned())
            && previous != plex
        {
            return Err(invalid_input(format!(
                "sample `{sample}` has more than one value in plex column `{column}`"
            )));
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

pub(crate) fn condition_by_sample(sdrf: &SdrfTable) -> HashMap<String, String> {
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

pub(crate) fn is_reference_condition(condition: &str) -> bool {
    let condition = condition.to_ascii_lowercase();
    condition.contains("powder") || condition.contains("pool")
}
