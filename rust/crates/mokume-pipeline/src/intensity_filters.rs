use crate::accessions::{protein_group_name, run_qc_key, sample_name, sdrf_record};
use crate::normalization_factors::passes_feature_filter;
use crate::qpx_stream::stream_qpx_features;
use crate::run_qc::RunQcSource;
use crate::state::RunQcKey;
use crate::{
    HashMap, HashSet, IntensityFilterConfig, MemoryPlan, NamedScoreFilterConfig, Path, Result,
    SdrfLookupCache, SdrfTable, filters, quantile_linear,
};

/// The group-level intensity filters that need a per-sample buffering pre-pass
/// (CVThresholdFilter and QuantileFilter), computed together so their Python
/// ordering (CV before Quantile, `intensity.py` factory) is honoured: the
/// quantile bounds are taken over the *post-CV* survivors.
#[derive(Debug, Default)]
pub(crate) struct IntensityGroupFilters {
    /// `(sample, ProteinName, PeptideCanonical)` triples dropped by the CV filter.
    pub(crate) cv_dropped: HashSet<(String, String, String)>,
    /// Per-sample `[lower, upper]` QuantileFilter bounds over the post-CV set.
    pub(crate) quantile_bounds: HashMap<String, (f64, f64)>,
}

/// Pre-pass for the buffering intensity filters (CVThresholdFilter at
/// Python `intensity.py:111-154`, QuantileFilter at `intensity.py:278-318`).
///
/// Both run on `dataset_df`, which holds a single sample (`peptide.py:266`), so
/// the natural unit is the per-`(sample, ProteinName, PeptideCanonical)` group.
/// We buffer that group's raw intensities once, then:
///   1. drop the group when its within-sample CV (ddof = 1) exceeds
///      `cv_threshold` (a `None` CV -- a single measurement or non-positive mean
///      -- always passes, mirroring Python keeping `NaN` CVs);
///   2. pool the surviving groups' intensities per sample and take the linear
///      quantiles for the `[lower, upper]` bounds.
///
/// Step 2 deliberately runs over the step-1 survivors: Python wires CV ahead of
/// Quantile, so the quantile bounds must not include the CV-dropped rows
/// (verified end-to-end against the `mokume` CLI). Both steps see the same
/// load-gated, `min_unique`-gated rows Python's pipeline sees at
/// `intensity.py:293` -- crucially *before* the per-row peptide filters
/// (length/charge/modification/missed cleavage), which Python applies only after
/// the intensity block, so a charge/length-doomed row still feeds the CV. The
/// bounds use f64, matching pandas upcasting the f32 column for `quantile`.
pub(crate) fn collect_intensity_group_filters(
    parquet: &Path,
    sdrf: Option<&SdrfTable>,
    source: &RunQcSource<'_>,
    intensity: &IntensityFilterConfig,
    run_qc_excluded: &HashSet<RunQcKey>,
    named_score: Option<&NamedScoreFilterConfig>,
) -> Result<IntensityGroupFilters> {
    let cv_active = intensity.cv_threshold.is_some();
    let quantile_active = intensity.quantile_lower > 0.0 || intensity.quantile_upper < 1.0;
    if !cv_active && !quantile_active {
        return Ok(IntensityGroupFilters::default());
    }
    // MinIntensityFilter runs in the pipeline ahead of both CV and Quantile, so
    // apply the same floor before collecting (a no-op beyond the `> 0` load filter
    // when `min_intensity == 0`).
    let min_floor =
        filters::effective_min_intensity(intensity.min_intensity, intensity.remove_zero_intensity);
    // Phase 1: buffer per `(sample, protein, canonical)` the raw intensities, and
    // per `(sample, protein)` the distinct canonicals (for the `min_unique` gate),
    // applying the same load-time filters as ingest.
    //
    // Do NOT drop contaminant rows here. Python applies the pipeline's
    // `ContaminantFilter` *after* CV/QuantileFilter (factory.py wires the
    // protein-level filters behind the intensity-level ones), and the CLI's
    // pre-pipeline `remove_decoy_contaminants` flag defaults to off
    // (features2peptides.py:61). So at `intensity.py:293` the frame still holds
    // contaminant rows, and the per-protein `min_unique` gate (peptide.py:279),
    // the CV groups, and the quantile bounds are all computed over them. The
    // ingest per-row drop still removes contaminants from the exported output,
    // matching the later `ContaminantFilter`; only the group decisions must
    // include them, so this pre-pass deliberately omits the contaminant check.
    let mut group_intensities: HashMap<(String, String, String), Vec<f64>> = HashMap::new();
    let mut cell_canonicals: HashMap<(String, String), HashSet<String>> = HashMap::new();
    let mut sdrf_lookup = sdrf.map(SdrfLookupCache::new);
    stream_qpx_features(parquet, &MemoryPlan::unlimited(), named_score, |feature| {
        if !passes_feature_filter(&feature, source.filtering, source.keep_shared_peptides) {
            return Ok(());
        }
        let Some(protein_group) = protein_group_name(&feature.protein_accessions) else {
            return Ok(());
        };
        if feature.intensity < min_floor {
            return Ok(());
        }
        let sdrf_record = sdrf_record(&feature, sdrf_lookup.as_mut())?;
        let sample = sample_name(&feature, sdrf_record);
        let run_key = run_qc_key(&feature, sdrf_record, sample.clone());
        if run_qc_excluded.contains(&run_key) {
            return Ok(());
        }
        cell_canonicals
            .entry((sample.clone(), protein_group.clone()))
            .or_default()
            .insert(feature.sequence.clone());
        group_intensities
            .entry((sample, protein_group, feature.sequence.clone()))
            .or_default()
            .push(feature.intensity);
        Ok(())
    })?;

    let min_unique = if source.keep_shared_peptides {
        0
    } else {
        source.min_unique_peptides
    };

    // Phase 2: per group, apply the `min_unique` gate (Python applies it before the
    // filter pipeline, peptide.py:278-281), then the CV decision. Surviving groups'
    // intensities are pooled per sample for the quantile bounds.
    let mut cv_dropped = HashSet::new();
    let mut per_sample: HashMap<String, Vec<f64>> = HashMap::new();
    for ((sample, protein, canonical), intensities) in group_intensities {
        // The `min_unique` gate keys on `(sample, protein)`: a protein with too
        // few distinct canonicals never reaches the filter pipeline, so its rows
        // feed neither the CV nor the quantile bounds.
        let canonical_count = cell_canonicals
            .get(&(sample.clone(), protein.clone()))
            .map_or(0, HashSet::len);
        if canonical_count < min_unique {
            continue;
        }
        if let Some(threshold) = intensity.cv_threshold {
            // `None` CV (single value or non-positive mean) keeps the group, exactly
            // like Python keeping `NaN` CVs (intensity.py:136). Python computes the
            // CV on the *float32* `NormIntensity` column (`pandas.Series.std/mean`),
            // so its `cv` carries float32 precision before the `<= threshold`
            // comparison upcasts it to f64. We compute the CV in f64 (the
            // oracle-locked primitive) then round it to f32, reproducing that
            // precision. The residual f32 rounding can only flip the keep/drop
            // decision when a group's true CV lands within ~1e-7 (relative) of the
            // threshold; a 30k-group fuzz against pandas showed zero such flips at a
            // typical threshold, so this is a measure-zero boundary, not a bias.
            if let Some(cv) = filters::coefficient_of_variation(&intensities)
                && f64::from(cv as f32) > threshold
            {
                cv_dropped.insert((sample, protein, canonical));
                continue;
            }
        }
        if quantile_active {
            per_sample.entry(sample).or_default().extend(intensities);
        }
    }

    let mut quantile_bounds = HashMap::new();
    if quantile_active {
        for (sample, mut values) in per_sample {
            values.retain(|value| value.is_finite());
            if values.is_empty() {
                continue;
            }
            let lower = quantile_linear(&mut values, intensity.quantile_lower);
            let upper = quantile_linear(&mut values, intensity.quantile_upper);
            if let (Some(lower), Some(upper)) = (lower, upper) {
                quantile_bounds.insert(sample, (lower, upper));
            }
        }
    }
    Ok(IntensityGroupFilters {
        cv_dropped,
        quantile_bounds,
    })
}
