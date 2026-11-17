use crate::accessions::{
    matches_sql_contaminant, protein_group_name, run_qc_key, sample_name, sdrf_record,
};
use crate::normalization_factors::{has_removed_accession, passes_feature_filter};
use crate::qpx_stream::stream_qpx_features;
use crate::state::RunQcKey;
use crate::{
    FilterConfig, HashMap, HashSet, MemoryPlan, NamedScoreFilterConfig, Path, Result,
    RunQcFilterConfig, SdrfLookupCache, SdrfTable,
};

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

pub(crate) struct RunQcSource<'a> {
    pub(crate) filtering: FilterConfig,
    pub(crate) keep_shared_peptides: bool,
    pub(crate) min_unique_peptides: usize,
    pub(crate) contaminant_patterns: &'a [String],
    pub(crate) remove_protein_ids: &'a HashSet<String>,
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
pub(crate) fn collect_run_qc_exclusions(
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
    let mut sdrf_lookup = sdrf.map(SdrfLookupCache::new);
    stream_qpx_features(parquet, &MemoryPlan::unlimited(), named_score, |feature| {
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
        let sdrf_record = sdrf_record(&feature, sdrf_lookup.as_mut())?;
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
