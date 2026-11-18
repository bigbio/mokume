use crate::protein_matrix::ProteinMatrix;
use crate::protein_matrix::covariates::{batch_column_values_for_samples, extract_sdrf_covariates};
use crate::{
    BatchCorrectionConfig, HashMap, Path, ProteinId, Result, SampleId, SdrfRawTable, invalid_input,
};

/// `pd.factorize` first-occurrence, 0-indexed labels for the batch prefixes.
/// Delegates to `mokume-stats` (`batch::factorize`) for a single source of truth.
pub(crate) fn factorize_batch_labels(values: &[String]) -> Vec<usize> {
    mokume_stats::batch::factorize(values)
}

/// Resolve a configured batch-detection method name case-insensitively, with
/// `-`/space normalised to `_`. Unknown names are rejected so a requested
/// strategy is never silently replaced with `sample_prefix`.
pub(crate) fn resolve_batch_method(
    name: &str,
) -> Result<mokume_stats::batch::BatchDetectionMethod> {
    mokume_stats::batch::BatchDetectionMethod::parse_name(name)
        .ok_or_else(|| invalid_input(format!("unknown batch method `{name}`")))
}

/// `column` detection and covariate extraction re-read the raw SDRF columns
/// (Python re-reads via `load_sdrf`); `validate_postprocessing_subset` already
/// guarantees `--sdrf` is present when either is requested.
fn read_batch_sdrf(
    method: mokume_stats::batch::BatchDetectionMethod,
    config: &BatchCorrectionConfig,
    sdrf_path: Option<&Path>,
) -> Result<Option<SdrfRawTable>> {
    use mokume_stats::batch::BatchDetectionMethod;

    if !matches!(method, BatchDetectionMethod::ExplicitColumn) && config.covariates.is_none() {
        return Ok(None);
    }
    sdrf_path.map(SdrfRawTable::from_path).transpose()
}

/// Original batch label of every matrix sample under the selected method.
fn batch_labels(
    method: mokume_stats::batch::BatchDetectionMethod,
    config: &BatchCorrectionConfig,
    raw: Option<&SdrfRawTable>,
    sample_names: &[&str],
) -> Result<Vec<String>> {
    use mokume_stats::batch::BatchDetectionMethod;

    match method {
        BatchDetectionMethod::ExplicitColumn => {
            let column = config.column.as_deref().unwrap_or_default();
            let raw =
                raw.ok_or_else(|| invalid_input("batch column detection requires --sdrf option"))?;
            // Python's `_detect_explicit_batches(None)` raises when the column
            // is absent; mirror that hard failure rather than silently skipping.
            batch_column_values_for_samples(raw, sample_names, column)
        }
        BatchDetectionMethod::RunName
        | BatchDetectionMethod::Fraction
        | BatchDetectionMethod::TechReplicate => Err(invalid_input(format!(
            "batch-method '{}' requires run-level information not available in the protein matrix",
            method.as_value()
        ))),
        BatchDetectionMethod::SamplePrefix => Ok(sample_names
            .iter()
            .map(|sample| mokume_stats::batch::sample_prefix(sample).to_owned())
            .collect()),
    }
}

/// Run `mokume-stats`'s `detect_batches` for the protein-matrix flow. Methods
/// that require unavailable run metadata are rejected by the caller.
#[cfg(test)]
pub(crate) fn detect_batches_for_method(
    method: mokume_stats::batch::BatchDetectionMethod,
    sample_names: &[&str],
    column_values: Option<&[String]>,
) -> Result<Vec<usize>> {
    let sample_ids = sample_names
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    mokume_stats::batch::detect_batches(&sample_ids, method, None, column_values).map_err(|error| {
        invalid_input(match error {
            mokume_stats::batch::DetectBatchesError::RunInfoRequired => {
                "batch-method 'run' requires run_info".to_owned()
            }
            mokume_stats::batch::DetectBatchesError::ColumnValuesRequired => {
                "batch column detection requires explicit batch values".to_owned()
            }
            mokume_stats::batch::DetectBatchesError::ColumnLengthMismatch { values, samples } => {
                format!("batch_column_values length ({values}) must match sample count ({samples})")
            }
        })
    })
}

/// Batch-layout validation: require at least two distinct batches, each with at
/// least two samples. The caller turns `None` into an explicit input error.
pub(crate) fn validate_batch_sizes(batch: Vec<usize>) -> Option<Vec<usize>> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &label in &batch {
        *counts.entry(label).or_insert(0) += 1;
    }
    if counts.len() < 2 || counts.values().any(|&count| count < 2) {
        return None;
    }
    Some(batch)
}

/// Run ComBat on the complete rows with the configured priors.
fn combat_rows(
    complete: &[(ProteinId, Vec<f64>)],
    batch: &[usize],
    covariates: Option<&[Vec<f64>]>,
    ref_batch: Option<usize>,
    config: &BatchCorrectionConfig,
) -> Vec<Vec<f64>> {
    let data = complete
        .iter()
        .map(|(_, values)| values.clone())
        .collect::<Vec<_>>();
    let params = mokume_stats::batch::ComBatParams {
        par_prior: config.parametric,
        mean_only: config.mean_only,
        ref_batch,
    };
    mokume_stats::batch::combat(&data, batch, covariates, params)
}

pub(crate) fn resolve_reference_batch(
    requested: &str,
    original_labels: &[String],
    encoded_labels: &[usize],
) -> Result<usize> {
    if let Some(index) = original_labels.iter().position(|label| label == requested) {
        return Ok(encoded_labels[index]);
    }
    if let Ok(encoded) = requested.parse::<usize>()
        && encoded_labels.contains(&encoded)
    {
        tracing::warn!(
            encoded_batch = encoded,
            "numeric --batch-ref is deprecated; use the original batch label"
        );
        return Ok(encoded);
    }
    let mut labels = original_labels.to_vec();
    labels.sort();
    labels.dedup();
    Err(invalid_input(format!(
        "batch reference `{requested}` is not present; detected labels: {}",
        labels.join(", ")
    )))
}

pub(crate) fn validate_combat_design(
    batch: &[usize],
    covariates: &[Vec<f64>],
    reference: Option<usize>,
) -> Result<()> {
    if covariates.len() != batch.len() {
        return Err(invalid_input(
            "batch covariate row count must match the matrix sample count",
        ));
    }
    let columns = covariates.first().map_or(0, Vec::len);
    if columns == 0 || covariates.iter().any(|row| row.len() != columns) {
        return Err(invalid_input(
            "batch covariate matrix must be non-empty and rectangular",
        ));
    }
    let mut labels = batch.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let label_index = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (*label, index))
        .collect::<HashMap<_, _>>();
    let mut design = vec![vec![0.0; batch.len()]; labels.len() + columns];
    for (sample, label) in batch.iter().enumerate() {
        design[label_index[label]][sample] = 1.0;
    }
    if let Some(reference) = reference {
        let row = label_index[&reference];
        design[row].fill(1.0);
    }
    for column in 0..columns {
        for (sample, values) in covariates.iter().enumerate() {
            design[labels.len() + column][sample] = values[column];
        }
    }
    if row_rank(design) < labels.len() + columns {
        return Err(invalid_input(
            "batch covariates are confounded with batches or each other; ComBat design is singular",
        ));
    }
    Ok(())
}

fn row_rank(mut matrix: Vec<Vec<f64>>) -> usize {
    let rows = matrix.len();
    let columns = matrix.first().map_or(0, Vec::len);
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows).max_by(|&left, &right| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right][column].abs())
        }) else {
            break;
        };
        if matrix[pivot][column].abs() <= 1e-12 {
            continue;
        }
        matrix.swap(rank, pivot);
        let pivot_value = matrix[rank][column];
        for value in &mut matrix[rank][column..] {
            *value /= pivot_value;
        }
        let pivot_row = matrix[rank][column..].to_vec();
        for (row_index, row) in matrix.iter_mut().enumerate() {
            if row_index == rank {
                continue;
            }
            let factor = row[column];
            for (value, pivot) in row[column..].iter_mut().zip(&pivot_row) {
                *value -= factor * pivot;
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}
impl ProteinMatrix {
    /// ComBat batch correction over the protein x sample matrix, mirroring
    /// Python's `apply_batch_correction` (stages.py:1701): detect batches, run
    /// ComBat on the proteins with no missing cells, and write the corrected
    /// values back in place. Proteins with any missing cell are kept uncorrected
    /// (`_complete_batch_matrix`, stages.py:1666); fewer than two samples, fewer
    /// than two batches, any batch under two samples, or no complete protein each
    /// are rejected instead of producing a successful no-op. A cell counts as
    /// present only when finite, so directlfq zeros stay (matching `isna`'s
    /// 0-is-present) while NaN/missing rows drop out.
    ///
    /// Batch detection follows `_detect_batch_indices`: `sample_prefix` from the
    /// sample-name prefix, `column` from an explicit SDRF column (`source name ->
    /// column`, missing samples `"unknown"`). `run` has no run-level mapping in
    /// the protein-matrix flow, so it raises like Python's `run_info required`.
    /// Repeated `--batch-covariate` values are extracted from the SDRF (`extract_sdrf_covariates`)
    /// and fed to the covariate ComBat design to preserve their biological signal.
    pub(crate) fn apply_batch_correction(
        &mut self,
        config: &BatchCorrectionConfig,
        sdrf_path: Option<&Path>,
        drop_empty_samples: bool,
        values_are_log2: bool,
    ) -> Result<()> {
        let samples = self.sample_columns(drop_empty_samples);
        if samples.len() < 2 {
            return Err(invalid_input(
                "batch correction requires at least two matrix samples",
            ));
        }
        let sample_ids = samples
            .iter()
            .map(|(sample, _)| *sample)
            .collect::<Vec<_>>();
        let sample_names = samples.iter().map(|(_, name)| *name).collect::<Vec<_>>();

        let method = resolve_batch_method(&config.method)?;
        let raw = read_batch_sdrf(method, config, sdrf_path)?;
        let original_batch_labels = batch_labels(method, config, raw.as_ref(), &sample_names)?;
        let batch = factorize_batch_labels(&original_batch_labels);
        let batch = validate_batch_sizes(batch).ok_or_else(|| {
            invalid_input(
                "batch correction requires at least two batches with at least two samples each",
            )
        })?;

        let complete = self.complete_batch_rows(&sample_ids, values_are_log2);
        if complete.is_empty() {
            return Err(invalid_input(
                "batch correction requires at least one protein observed in every sample",
            ));
        }

        // Biological covariates to preserve (independent of the batch method),
        // matching `_batch_covariates` at the combat call site.
        let covariates = match (&config.covariates, raw.as_ref()) {
            (Some(columns), Some(raw)) => extract_sdrf_covariates(raw, &sample_names, columns)?,
            _ => None,
        };

        let ref_batch = config
            .ref_batch
            .as_deref()
            .map(|requested| resolve_reference_batch(requested, &original_batch_labels, &batch))
            .transpose()?;
        if let Some(covariates) = covariates.as_deref() {
            validate_combat_design(&batch, covariates, ref_batch)?;
        }

        let corrected = combat_rows(&complete, &batch, covariates.as_deref(), ref_batch, config);
        self.write_batch_corrected(&complete, &corrected, &sample_ids, values_are_log2);
        Ok(())
    }

    /// Proteins observed in every sample, as rows on ComBat's (log2) scale.
    ///
    /// Incomplete proteins are left out and stay uncorrected. ComBat's
    /// empirical-Bayes priors are pooled over the complete rows but are
    /// row-order independent; rows are sorted by accession to match the matrix
    /// output order. ComBat models additive batch effects, so linear
    /// intensities are corrected on the log2 scale, where a zero or negative
    /// intensity is not finite and counts as missing (DirectLFQ writes missing
    /// proteins as zero).
    fn complete_batch_rows(
        &self,
        sample_ids: &[SampleId],
        values_are_log2: bool,
    ) -> Vec<(ProteinId, Vec<f64>)> {
        let mut proteins = self
            .proteins
            .iter()
            .filter(|(protein, _)| self.allowed_proteins.contains(protein))
            .collect::<Vec<_>>();
        proteins.sort_by(|left, right| left.1.cmp(right.1));
        let modeled = |value: f64| if values_are_log2 { value } else { value.log2() };
        proteins
            .into_iter()
            .filter_map(|(protein, _)| {
                let row = sample_ids
                    .iter()
                    .map(|&sample| {
                        self.value(protein, sample)
                            .map(modeled)
                            .filter(|value| value.is_finite())
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some((protein, row))
            })
            .collect()
    }

    /// Write ComBat's rows back on the matrix scale.
    ///
    /// ComBat leaves a row undefined (NaN) when its priors cannot be estimated,
    /// e.g. with a single complete row; such a row keeps its measured values
    /// instead of turning into missing values.
    fn write_batch_corrected(
        &mut self,
        complete: &[(ProteinId, Vec<f64>)],
        corrected: &[Vec<f64>],
        sample_ids: &[SampleId],
        values_are_log2: bool,
    ) {
        for ((protein, _), corrected_row) in complete.iter().zip(corrected) {
            if corrected_row.iter().any(|value| !value.is_finite()) {
                continue;
            }
            for (&sample, &value) in sample_ids.iter().zip(corrected_row) {
                let value = if values_are_log2 { value } else { value.exp2() };
                self.set_value(*protein, sample, value);
            }
        }
    }
}
