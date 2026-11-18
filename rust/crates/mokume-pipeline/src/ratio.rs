use crate::aggregation::{AggregationMeasurement, RatioAggregation, RatioFractionMerge, RatioPsm};
use crate::plex::{derive_sample_plexes, plex_of};
use crate::state::CellKey;
use crate::{
    DEFAULT_REFERENCE_REGEX, FeatureToProteinsConfig, HashMap, HashSet, IrsConfig, MokumeError,
    PeptideId, ProteinId, RegexBuilder, Result, SampleId, SdrfRawTable, SdrfTable, invalid_input,
    median_finite,
};

impl RatioAggregation {
    pub(crate) fn from_config(
        config: &FeatureToProteinsConfig,
        sdrf: Option<&SdrfTable>,
        raw_sdrf: Option<&SdrfRawTable>,
    ) -> Result<Self> {
        let Some(sdrf) = sdrf else {
            return Err(invalid_input("Ratio quantification requires --sdrf option"));
        };
        let raw_sdrf = raw_sdrf.ok_or_else(|| {
            invalid_input("Ratio quantification requires readable raw SDRF metadata")
        })?;
        let reference_samples = resolve_ratio_reference_samples(sdrf, raw_sdrf, config)?;
        if reference_samples.is_empty() {
            return Err(invalid_input(
                "Ratio quantification requires reference samples; repeat --irs-reference-sample or mark references in SDRF",
            ));
        }
        Ok(Self {
            records: Vec::new(),
            sample_names: HashMap::new(),
            reference_samples: reference_samples.into_iter().collect(),
            sample_to_plex: derive_sample_plexes(
                sdrf,
                Some(raw_sdrf),
                config.irs.plex_column.as_deref(),
            )?
            .0,
            fraction_merge: parse_ratio_fraction_merge(&config.ratio.fraction_merge)?,
            min_unique_peptides: config.filtering.min_unique_peptides,
        })
    }

    pub(crate) fn push(&mut self, measurement: AggregationMeasurement<'_>) {
        self.sample_names
            .entry(measurement.sample)
            .or_insert_with(|| measurement.sample_name.to_owned());
        self.records.push(RatioPsm {
            protein: measurement.protein,
            peptide: measurement.peptide,
            charge: measurement.charge,
            sample: measurement.sample,
            intensity: measurement.intensity,
        });
    }

    pub(crate) fn finalize(self) -> HashMap<CellKey, f64> {
        let valid_proteins = ratio_valid_proteins(&self.records, self.min_unique_peptides);
        let averaged = average_ratio_fractions(&self.records, &valid_proteins, self.fraction_merge);
        let reference_intensity = ratio_reference_intensities(
            &averaged,
            &self.sample_names,
            &self.reference_samples,
            &self.sample_to_plex,
        );
        let peptide_ratios = ratio_peptide_log2_ratios(&averaged, &self, &reference_intensity);
        ratio_protein_medians(peptide_ratios)
    }
}

fn parse_ratio_fraction_merge(value: &str) -> Result<RatioFractionMerge> {
    match value.trim().to_ascii_lowercase().as_str() {
        "mean" => Ok(RatioFractionMerge::Mean),
        "max" => Ok(RatioFractionMerge::Max),
        _ => Err(MokumeError::NotImplemented {
            stage: "ratio-fraction-merge",
        }),
    }
}

pub(crate) fn detect_reference_samples(
    raw: &SdrfRawTable,
    reference_regex: &str,
) -> Result<Vec<String>> {
    let sample_col = raw
        .column_index("source name")
        .ok_or_else(|| invalid_input("SDRF reference detection requires a `source name` column"))?;
    let scan_cols = raw
        .headers()
        .iter()
        .enumerate()
        .filter_map(|(index, header)| {
            (header.starts_with("factor value[") || header.starts_with("characteristics["))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let pattern = RegexBuilder::new(reference_regex)
        .case_insensitive(true)
        .build()
        .map_err(|source| {
            invalid_input(format!(
                "invalid IRS reference regex `{reference_regex}`: {source}"
            ))
        })?;
    let mut samples = Vec::new();
    for row in 0..raw.row_count() {
        if scan_cols
            .iter()
            .any(|&column| pattern.is_match(raw.cell(row, column)))
        {
            samples.push(raw.cell(row, sample_col));
        }
    }
    Ok(sorted_unique(samples.into_iter()))
}

/// Detect reference samples for ratio quantification, mirroring Python's
/// `LoadingStage.load_for_ratio` priority:
///   1. repeated `--irs-reference-sample`,
///   2. an explicitly changed reference regex,
///   3. `characteristics[pooled sample]` autodetection,
///   4. the default regex across every factor/characteristic column.
fn resolve_ratio_reference_samples(
    sdrf: &SdrfTable,
    raw: &SdrfRawTable,
    config: &FeatureToProteinsConfig,
) -> Result<Vec<String>> {
    if let Some(samples) = &config.irs.reference_samples {
        return Ok(sorted_unique(samples.iter().map(String::as_str)));
    }
    if config.irs.reference_regex != DEFAULT_REFERENCE_REGEX {
        return detect_reference_samples(raw, &config.irs.reference_regex);
    }
    let pooled = detect_pooled_reference_samples(sdrf);
    if !pooled.is_empty() {
        return Ok(pooled);
    }
    detect_reference_samples(raw, DEFAULT_REFERENCE_REGEX)
}

/// Reproduce Python's `detect_pooled_from_sdrf`: a sample is a reference channel
/// when its `characteristics[pooled sample]` value is `pooled` or begins with
/// `SN=` (a pooled-member list). Returns an empty vector when the column is
/// absent or no sample qualifies, so callers fall through to other detectors.
fn detect_pooled_reference_samples(sdrf: &SdrfTable) -> Vec<String> {
    let mut samples = sdrf
        .records()
        .iter()
        .filter(|record| {
            record.pooled_sample.as_deref().is_some_and(|value| {
                let value = value.trim().to_ascii_lowercase();
                value == "pooled" || value.starts_with("sn=")
            })
        })
        .map(|record| record.sample_accession.clone())
        .collect::<Vec<_>>();
    samples.sort();
    samples.dedup();
    samples
}

pub(crate) fn resolve_irs_reference_samples(
    sdrf: &SdrfTable,
    raw: &SdrfRawTable,
    config: &IrsConfig,
) -> Result<Vec<String>> {
    if let Some(samples) = &config.reference_samples {
        return Ok(sorted_unique(samples.iter().map(String::as_str)));
    }

    if let (Some(column), Some(values)) = (&config.sdrf_column, &config.sdrf_values) {
        return detect_reference_samples_by_sdrf_column(raw, column, values);
    }
    if config.reference_regex != DEFAULT_REFERENCE_REGEX {
        return detect_reference_samples(raw, &config.reference_regex);
    }
    let pooled = detect_pooled_reference_samples(sdrf);
    if !pooled.is_empty() {
        return Ok(pooled);
    }
    detect_reference_samples(raw, DEFAULT_REFERENCE_REGEX)
}

fn detect_reference_samples_by_sdrf_column(
    raw: &SdrfRawTable,
    column: &str,
    values: &[String],
) -> Result<Vec<String>> {
    let accepted = values
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<HashSet<_>>();
    if accepted.is_empty() {
        return Ok(Vec::new());
    }
    let requested = column.trim().to_ascii_lowercase();
    let value_col = raw.column_index(&requested).ok_or_else(|| {
        invalid_input(format!(
            "IRS SDRF column `{column}` was not found; available columns: {}",
            raw.headers().join(", ")
        ))
    })?;
    let sample_col = raw
        .column_index("source name")
        .ok_or_else(|| invalid_input("SDRF reference detection requires a `source name` column"))?;
    let samples = (0..raw.row_count())
        .filter(|&row| accepted.contains(&raw.cell(row, value_col).trim().to_ascii_lowercase()))
        .map(|row| raw.cell(row, sample_col));
    Ok(sorted_unique(samples))
}

fn sorted_unique<'a>(values: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut values = values
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn ratio_valid_proteins(records: &[RatioPsm], min_unique_peptides: usize) -> HashSet<ProteinId> {
    let mut peptides = HashMap::<ProteinId, HashSet<PeptideId>>::new();
    for record in records {
        peptides
            .entry(record.protein)
            .or_default()
            .insert(record.peptide);
    }
    peptides
        .into_iter()
        .filter_map(|(protein, peptides)| {
            (peptides.len() >= min_unique_peptides).then_some(protein)
        })
        .collect()
}

fn average_ratio_fractions(
    records: &[RatioPsm],
    valid_proteins: &HashSet<ProteinId>,
    fraction_merge: RatioFractionMerge,
) -> Vec<(ProteinId, PeptideId, i32, SampleId, f64)> {
    let mut grouped = HashMap::<(ProteinId, PeptideId, i32, SampleId), Vec<f64>>::new();
    for record in records {
        if valid_proteins.contains(&record.protein) {
            grouped
                .entry((record.protein, record.peptide, record.charge, record.sample))
                .or_default()
                .push(record.intensity);
        }
    }
    grouped
        .into_iter()
        .filter_map(|((protein, peptide, charge, sample), values)| {
            merge_ratio_values(values, fraction_merge)
                .map(|intensity| (protein, peptide, charge, sample, intensity))
        })
        .collect()
}

fn merge_ratio_values(mut values: Vec<f64>, fraction_merge: RatioFractionMerge) -> Option<f64> {
    values.retain(|value| value.is_finite() && *value > 0.0);
    if values.is_empty() {
        return None;
    }
    match fraction_merge {
        RatioFractionMerge::Mean => Some(values.iter().sum::<f64>() / values.len() as f64),
        RatioFractionMerge::Max => values.into_iter().max_by(f64::total_cmp),
    }
}

fn ratio_reference_intensities(
    averaged: &[(ProteinId, PeptideId, i32, SampleId, f64)],
    sample_names: &HashMap<SampleId, String>,
    reference_samples: &HashSet<String>,
    sample_to_plex: &HashMap<String, String>,
) -> HashMap<(ProteinId, PeptideId, i32, String), f64> {
    let mut grouped = HashMap::<(ProteinId, PeptideId, i32, String), Vec<f64>>::new();
    for (protein, peptide, charge, sample, intensity) in averaged {
        let Some(sample_name) = sample_names.get(sample) else {
            continue;
        };
        if reference_samples.contains(sample_name) {
            grouped
                .entry((
                    *protein,
                    *peptide,
                    *charge,
                    plex_of(sample_to_plex, sample_name),
                ))
                .or_default()
                .push(*intensity);
        }
    }
    grouped
        .into_iter()
        .filter_map(|(key, values)| mean_finite(values).map(|value| (key, value)))
        .collect()
}

type RatioMeasurement = (ProteinId, PeptideId, i32, SampleId, f64);

fn ratio_peptide_log2_ratios(
    averaged: &[RatioMeasurement],
    aggregation: &RatioAggregation,
    reference_intensity: &HashMap<(ProteinId, PeptideId, i32, String), f64>,
) -> HashMap<(ProteinId, PeptideId, SampleId), Vec<f64>> {
    let mut ratios = HashMap::<(ProteinId, PeptideId, SampleId), Vec<f64>>::new();
    for (protein, peptide, charge, sample, intensity) in averaged {
        let Some(sample_name) = aggregation.sample_names.get(sample) else {
            continue;
        };
        if aggregation.reference_samples.contains(sample_name) {
            continue;
        }
        let plex = plex_of(&aggregation.sample_to_plex, sample_name);
        let Some(reference) = reference_intensity.get(&(*protein, *peptide, *charge, plex)) else {
            continue;
        };
        if *intensity > 0.0 && *reference > 0.0 {
            let ratio = (*intensity / *reference).log2();
            if ratio.is_finite() {
                ratios
                    .entry((*protein, *peptide, *sample))
                    .or_default()
                    .push(ratio);
            }
        }
    }
    ratios
}

fn ratio_protein_medians(
    peptide_ratios: HashMap<(ProteinId, PeptideId, SampleId), Vec<f64>>,
) -> HashMap<CellKey, f64> {
    let mut protein_ratios = HashMap::<CellKey, Vec<f64>>::new();
    for ((protein, _, sample), mut values) in peptide_ratios {
        if let Some(value) = median_finite(&mut values) {
            protein_ratios
                .entry(CellKey { protein, sample })
                .or_default()
                .push(value);
        }
    }
    protein_ratios
        .into_iter()
        .filter_map(|(cell, mut values)| median_finite(&mut values).map(|value| (cell, value)))
        .collect()
}

fn mean_finite(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite() && *value > 0.0);
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}
