use crate::accessions::{matches_sql_contaminant, sdrf_record};
use crate::qpx_stream::stream_qpx_features;
use crate::{
    BTreeMap, HashMap, IrsChannelConfig, IrsScope, IrsStat, MemoryPlan, MokumeError,
    NamedScoreFilterConfig, Path, QpxFeatureRecord, Regex, Result, SdrfLookupCache, SdrfRawTable,
    SdrfTable, mean_positive, median,
};

/// First `_`-token of `value`, i.e. DuckDB `split_part(value, '_', 1)`. A name
/// with no `_` is its own first token. This is Python's `mixture` key
/// (`feature.py:274-276` / `420-422`): the first token of the **sample
/// accession**, which with no SDRF is the `run_file_name`.
pub(crate) fn irs_mixture_first_token(value: &str) -> &str {
    value.split('_').next().unwrap_or(value)
}

/// Channel-IRS pre-pass (Python `get_irs_scaling_factors`, `feature.py:711-817`)
/// for every `irs_scope`. Streams the QPX features, keeps only the reference
/// channel's rows (`label == irs.channel`), applies the same explicit filters
/// Python's SQL uses (`intensity > 0`; the contaminant `NOT LIKE` only when
/// `remove_contaminants`; the `min_intensity` floor only when positive), groups
/// the surviving intensities by run into a per-run `irs_value` via the chosen
/// statistic, then derives one scale per run under `irs.scope`.
///
/// Each run also carries its `mixture` (Python `split_part(sample_accession,
/// '_', 1)`): with no SDRF the sample accession is the `run_file_name`, so the
/// mixture is its first `_`-token; with an SDRF the mixture is the first token
/// of the joined `source name`. `by_mixture` / `two_stage` use this; `global`
/// ignores it.
///
/// Returns the `run_file_name -> scale` map. Run identity is the correct plex
/// key: unlike a technical-replicate number, it is stable for arbitrary file
/// names and cannot collide when the same replicate number occurs in multiple
/// mixtures. An empty map means no valid scale could be computed; the command
/// layer turns that into an explicit input error.
pub(crate) fn collect_irs_scale(
    parquet: &Path,
    irs: &IrsChannelConfig,
    sdrf: Option<&SdrfTable>,
    remove_contaminants: bool,
    contaminant_patterns: &[String],
    min_intensity: f64,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<HashMap<String, f64>> {
    let runs = buffer_irs_runs(
        parquet,
        irs,
        sdrf,
        remove_contaminants,
        contaminant_patterns,
        min_intensity,
        named_score,
    )?;

    // Collapse each run's buffered intensities into one `irs_value`, then
    // derive the scale under `irs.scope`. Collect into a `BTreeMap` keyed by run
    // name so the runs are visited in a stable (sorted) order.
    let per_run: Vec<(String, String, Vec<f64>)> = runs
        .into_iter()
        .map(|(run_name, buffer)| (run_name, (buffer.mixture, buffer.intensities)))
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|(run_name, (mixture, intensities))| (run_name, mixture, intensities))
        .collect();
    Ok(match irs.scope {
        IrsScope::Global => {
            let runs = per_run
                .into_iter()
                .map(|(run, _mixture, intensities)| (run, intensities))
                .collect();
            irs_global_scale_from_runs(runs, irs.stat)
        }
        IrsScope::ByMixture => irs_by_mixture_scale_from_runs(per_run, irs.stat),
        IrsScope::TwoStage => irs_two_stage_scale_from_runs(per_run, irs.stat),
    })
}

struct RunBuffer {
    mixture: String,
    intensities: Vec<f64>,
}

// Per run, buffer the reference-channel intensities and remember the mixture
// used by the non-global scopes.
fn buffer_irs_runs(
    parquet: &Path,
    irs: &IrsChannelConfig,
    sdrf: Option<&SdrfTable>,
    remove_contaminants: bool,
    contaminant_patterns: &[String],
    min_intensity: f64,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<HashMap<String, RunBuffer>> {
    let mut runs: HashMap<String, RunBuffer> = HashMap::new();
    let mut sdrf_lookup = sdrf.map(SdrfLookupCache::new);
    stream_qpx_features(parquet, &MemoryPlan::unlimited(), named_score, |feature| {
        if !is_irs_reference_feature(
            &feature,
            &irs.channel,
            remove_contaminants,
            contaminant_patterns,
            min_intensity,
        ) {
            return Ok(());
        }
        let sdrf_record = sdrf_record(&feature, sdrf_lookup.as_mut())?;
        let sample_accession = sdrf_record.map_or(feature.run_file_name.as_str(), |record| {
            record.sample_accession.as_str()
        });
        runs.entry(feature.run_file_name.clone())
            .or_insert_with(|| {
                // Mixture is the first `_`-token of the *sample accession*, which is
                // the joined SDRF `source name` when an SDRF is present and the
                // `run_file_name` otherwise (Python `parquet_db` view). The SDRF
                // join keys on the run name (with the raw feature label), mirroring
                // `enrich_with_sdrf`; when an SDRF is present, an unmatched run or
                // label is rejected instead of falling back to the run name.
                RunBuffer {
                    mixture: irs_mixture_first_token(sample_accession).to_owned(),
                    intensities: Vec::new(),
                }
            })
            .intensities
            .push(feature.intensity);
        Ok(())
    })?;

    Ok(runs)
}

fn is_irs_reference_feature(
    feature: &QpxFeatureRecord,
    channel: &str,
    remove_contaminants: bool,
    contaminant_patterns: &[String],
    min_intensity: f64,
) -> bool {
    // Channel match is on the *raw* feature label (Python filters on
    // `channel = ?` before any reformat). A row with no label never matches.
    if feature.label.as_deref() != Some(channel) {
        return false;
    }
    if !(feature.intensity.is_finite() && feature.intensity > 0.0) {
        return false;
    }
    if min_intensity > 0.0 && feature.intensity < min_intensity {
        return false;
    }
    // Contaminant `NOT LIKE` on the raw `pg_accessions`, mirroring the median
    // pre-pass / Python's `SQLFilterBuilder` (case-sensitive substring; the
    // default pattern list routes through the `is_contaminant` fallback).
    if remove_contaminants
        && matches_sql_contaminant(&feature.protein_accessions, contaminant_patterns)
    {
        return false;
    }
    true
}

/// Pure global-scope IRS math (Python `get_irs_scaling_factors`,
/// `feature.py:776-815`, global branch). Given each run's `(run identity,
/// reference-channel intensities)` in a stable order, collapse to one
/// `irs_value` per run via `stat`, drop non-positive `irs_value`s, take the
/// global center via the same `stat`, and return `scale[run] = center /
/// irs_value`. Returns an empty map when nothing positive survives.
pub(crate) fn irs_global_scale_from_runs(
    per_run: Vec<(String, Vec<f64>)>,
    stat: IrsStat,
) -> HashMap<String, f64> {
    let aggregate = |values: &mut Vec<f64>| match stat {
        IrsStat::Median => median(values),
        IrsStat::Mean => mean_positive(values),
    };
    let mut irs_values: Vec<(String, f64)> = Vec::new();
    for (run, mut intensities) in per_run {
        // `irs_df = irs_df[irs_df["irs_value"] > 0]` drops non-positive centers.
        if let Some(irs_value) = aggregate(&mut intensities)
            && irs_value > 0.0
        {
            irs_values.push((run, irs_value));
        }
    }
    if irs_values.is_empty() {
        return HashMap::new();
    }
    let mut centers: Vec<f64> = irs_values.iter().map(|(_, value)| *value).collect();
    let Some(global_center) = aggregate(&mut centers) else {
        return HashMap::new();
    };
    let mut scale_by_run: HashMap<String, f64> = HashMap::new();
    for (run, irs_value) in irs_values {
        scale_by_run.insert(run, global_center / irs_value);
    }
    scale_by_run
}

/// Collapse each run's buffered reference-channel intensities into a single
/// `irs_value` and drop the runs whose center is non-positive, mirroring the
/// Python filter `irs_value > 0`. Preserves input (stable, run-name-sorted)
/// order. Returns `(run identity, mixture, irs_value)` per surviving run.
fn irs_values_per_run(
    per_run: Vec<(String, String, Vec<f64>)>,
    stat: IrsStat,
) -> Vec<(String, String, f64)> {
    let aggregate = |values: &mut Vec<f64>| match stat {
        IrsStat::Median => median(values),
        IrsStat::Mean => mean_positive(values),
    };
    let mut out = Vec::with_capacity(per_run.len());
    for (run, mixture, mut intensities) in per_run {
        if let Some(irs_value) = aggregate(&mut intensities)
            && irs_value > 0.0
        {
            out.push((run, mixture, irs_value));
        }
    }
    out
}

/// Per-mixture center over the surviving runs' `irs_value`s, mirroring Python's
/// `irs_df.groupby("mixture")["irs_value"].transform(stat)`. The returned map is
/// keyed by mixture; every input mixture is present because each contributes at
/// least one run. `median` / `mean_positive` keep only positive finite values,
/// which is a no-op here (all `irs_value`s are already positive).
fn irs_mixture_centers(
    irs_values: &[(String, String, f64)],
    stat: IrsStat,
) -> HashMap<String, f64> {
    let mut by_mixture: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (_, mixture, irs_value) in irs_values {
        by_mixture
            .entry(mixture.clone())
            .or_default()
            .push(*irs_value);
    }
    let mut centers: HashMap<String, f64> = HashMap::new();
    for (mixture, mut values) in by_mixture {
        let center = match stat {
            IrsStat::Median => median(&mut values),
            IrsStat::Mean => mean_positive(&values),
        };
        if let Some(center) = center {
            centers.insert(mixture, center);
        }
    }
    centers
}

/// Pure `by_mixture`-scope IRS math (Python `get_irs_scaling_factors`,
/// `feature.py:779-784`). For each surviving run, `scale = mixture_center /
/// irs_value`, where `mixture_center` is the `stat` over that mixture's
/// `irs_value`s. Returns an empty map when no run survives the positive-
/// `irs_value` filter.
pub(crate) fn irs_by_mixture_scale_from_runs(
    per_run: Vec<(String, String, Vec<f64>)>,
    stat: IrsStat,
) -> HashMap<String, f64> {
    let irs_values = irs_values_per_run(per_run, stat);
    if irs_values.is_empty() {
        return HashMap::new();
    }
    let centers = irs_mixture_centers(&irs_values, stat);
    let mut scale_by_run: HashMap<String, f64> = HashMap::new();
    for (run, mixture, irs_value) in irs_values {
        if let Some(&mixture_center) = centers.get(&mixture) {
            scale_by_run.insert(run, mixture_center / irs_value);
        }
    }
    scale_by_run
}

/// Pure `two_stage`-scope IRS math (Python `get_irs_scaling_factors`,
/// `feature.py:785-804`). Stage 1 is `by_mixture` (`mixture_center /
/// irs_value`); stage 2 re-anchors each mixture to a global center over the
/// **distinct** mixture centers (`global_center / mixture_center`). The applied
/// scale is the product, which algebraically equals `global_center / irs_value`
/// but is computed exactly as the two factors so float rounding matches Python.
/// `global_center` is the `stat` over one center per distinct mixture, mirroring
/// `irs_df[["mixture","mixture_center"]].drop_duplicates()`. Returns an empty
/// map when no run survives or the global center cannot be formed.
pub(crate) fn irs_two_stage_scale_from_runs(
    per_run: Vec<(String, String, Vec<f64>)>,
    stat: IrsStat,
) -> HashMap<String, f64> {
    let irs_values = irs_values_per_run(per_run, stat);
    if irs_values.is_empty() {
        return HashMap::new();
    }
    let centers = irs_mixture_centers(&irs_values, stat);
    // Global center over one center per distinct mixture (`drop_duplicates` on
    // the `(mixture, mixture_center)` pair). `BTreeMap::values` visits mixtures
    // in sorted order, but the `stat` is order-independent.
    let mut distinct_centers: Vec<f64> = centers.values().copied().collect();
    let global_center = match stat {
        IrsStat::Median => median(&mut distinct_centers),
        IrsStat::Mean => mean_positive(&distinct_centers),
    };
    let Some(global_center) = global_center else {
        return HashMap::new();
    };
    let mut scale_by_run: HashMap<String, f64> = HashMap::new();
    for (run, mixture, irs_value) in irs_values {
        if let Some(&mixture_center) = centers.get(&mixture) {
            let stage1 = mixture_center / irs_value;
            let stage2 = global_center / mixture_center;
            scale_by_run.insert(run, stage1 * stage2);
        }
    }
    scale_by_run
}

/// Resolve the IRS reference channel from the SDRF when `--irs_channel` is not
/// given but `--irs_autodetect_regex` is (Python `peptide.py:219-233`). Rows
/// whose `source name` matches the regex (case-insensitive, `str.contains`) vote
/// with their `comment[label]`; the winning channel is the most frequent label,
/// ties broken by the lexicographically smallest (pandas `mode().iloc[0]`, whose
/// result is sorted).
///
/// Returns `Ok(None)` — IRS is skipped, matching Python's warning path — when the
/// SDRF lacks a `source name` or `comment[label]` column, when no row matches, or
/// when every matching row has an empty label. The regex is compiled
/// case-insensitively (`(?i)`), mirroring pandas' `case=False`; an invalid regex
/// surfaces as `InvalidInput`.
pub fn resolve_irs_autodetect_channel(
    sdrf_path: &Path,
    autodetect_regex: &str,
) -> Result<Option<String>> {
    let table = SdrfRawTable::from_path(sdrf_path)?;
    // `load_sdrf` lowercases headers, so the canonical column names match.
    let (Some(source_col), Some(label_col)) = (
        table.column_index("source name"),
        table.column_index("comment[label]"),
    ) else {
        return Ok(None);
    };
    let regex = Regex::new(&format!("(?i){autodetect_regex}")).map_err(|source| {
        MokumeError::InvalidInput {
            message: format!("invalid --irs_autodetect_regex `{autodetect_regex}`: {source}"),
        }
    })?;
    // Tally label frequencies among matching rows. A `BTreeMap` keeps the keys
    // sorted so the tie-break naturally selects the smallest label, reproducing
    // pandas `mode().iloc[0]` (mode returns sorted values; `.iloc[0]` is the
    // first). Empty labels are ignored (an empty `comment[label]` is not a
    // channel).
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in 0..table.row_count() {
        if regex.is_match(table.cell(row, source_col)) {
            let label = table.cell(row, label_col);
            if !label.is_empty() {
                *counts.entry(label.to_owned()).or_insert(0) += 1;
            }
        }
    }
    // Highest count wins; the sorted iteration order makes the first max the
    // smallest label, so a plain "strictly greater" comparison yields the
    // tie-break Python applies.
    let mut best: Option<(String, usize)> = None;
    for (label, count) in counts {
        if best
            .as_ref()
            .is_none_or(|(_, best_count)| count > *best_count)
        {
            best = Some((label, count));
        }
    }
    Ok(best.map(|(label, _)| label))
}
