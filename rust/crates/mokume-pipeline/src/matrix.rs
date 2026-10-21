use std::collections::{HashMap, HashSet};

use mokume_core::{ImputationConfig, PeptideId, ProteinId, Result, SampleId, StringIdRegistry};
use mokume_normalization::SampleNormalizationMethod;

use super::{
    apply_dataset_norm_to_peptide_cells, imputed_values, invalid_input, threading,
    validate_imputation_config, CellKey,
};

/// Normalize a row-major linear-intensity matrix without running quantification.
///
/// `values[row][column]` is one protein/sample cell and `sample_names` labels
/// the columns. Non-finite cells are missing. Supported methods are `none`,
/// `median`, `mean`, `quantile`, `rlr`, `loess`, `hierarchical`, and `tmm`.
/// Quantile accepts signed values and retains fully missing rows; with multiple
/// rows it needs at least two finite values per column, since preprocessCore's
/// rank interpolation is undefined below that. TMM accepts non-negative values
/// and needs positive library sizes unless the entire matrix is zero/missing.
pub fn normalize_matrix(
    values: &[Vec<f64>],
    sample_names: &[String],
    method: &str,
    threads: Option<usize>,
) -> Result<Vec<Vec<f64>>> {
    if threads == Some(0) {
        return Err(invalid_input("thread count must be greater than zero"));
    }
    let width = validate_rectangular(values)?;
    if width != sample_names.len() {
        return Err(invalid_input(format!(
            "matrix has {width} columns but {} sample names were supplied",
            sample_names.len()
        )));
    }
    let Some(method) = parse_matrix_normalization(method)? else {
        return Ok(values.to_vec());
    };
    threading::install(threads, || normalize_inner(values, sample_names, method))
}

fn normalize_inner(
    values: &[Vec<f64>],
    sample_names: &[String],
    method: SampleNormalizationMethod,
) -> Result<Vec<Vec<f64>>> {
    let mut samples = StringIdRegistry::<SampleId>::new();
    let mut sample_ids = Vec::with_capacity(sample_names.len());
    let mut unique_names = HashSet::with_capacity(sample_names.len());
    for name in sample_names {
        if !unique_names.insert(name) {
            return Err(invalid_input(format!("duplicate sample name `{name}`")));
        }
        let id = samples
            .get_or_insert(name)
            .ok_or_else(|| invalid_input("too many matrix columns"))?;
        sample_ids.push(id);
    }

    match method {
        SampleNormalizationMethod::Quantile => quantile_matrix(values, &sample_ids),
        SampleNormalizationMethod::Hierarchical => Ok(hierarchical_matrix(values)),
        SampleNormalizationMethod::Tmm => tmm_matrix(values, sample_names),
        _ => normalize_peptide_matrix(values, &sample_ids, &samples, method),
    }
}

fn normalize_peptide_matrix(
    values: &[Vec<f64>],
    sample_ids: &[SampleId],
    samples: &StringIdRegistry<SampleId>,
    method: SampleNormalizationMethod,
) -> Result<Vec<Vec<f64>>> {
    let mut cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    let mut allowed = HashSet::<CellKey>::new();
    for (row_index, row) in values.iter().enumerate() {
        let raw_row =
            u32::try_from(row_index).map_err(|_| invalid_input("matrix has too many rows"))?;
        let protein = ProteinId::new(raw_row);
        let peptide = PeptideId::new(raw_row);
        for (&sample, &value) in sample_ids.iter().zip(row) {
            let observed = value.is_finite() && value > 0.0;
            if observed {
                let cell = CellKey { protein, sample };
                cells.entry(cell).or_default().insert(peptide, value);
                allowed.insert(cell);
            }
        }
    }

    apply_dataset_norm_to_peptide_cells(&mut cells, method, &allowed, &HashMap::new(), samples);

    let mut normalized = vec![vec![f64::NAN; sample_ids.len()]; values.len()];
    for (cell, peptides) in cells {
        let row = usize::try_from(cell.protein.get())
            .map_err(|_| invalid_input("matrix row index is not representable"))?;
        let column = usize::try_from(cell.sample.get())
            .map_err(|_| invalid_input("matrix column index is not representable"))?;
        if let Some(value) = peptides.get(&PeptideId::new(cell.protein.get())) {
            normalized[row][column] = *value;
        }
    }
    Ok(normalized)
}

/// Retain all input rows, including entirely missing rows in the target grid.
fn quantile_matrix(values: &[Vec<f64>], samples: &[SampleId]) -> Result<Vec<Vec<f64>>> {
    let minimum = values.len().min(2);
    for column in 0..samples.len() {
        if values.iter().filter(|row| row[column].is_finite()).count() < minimum {
            return Err(invalid_input(format!(
                "quantile column {column} needs at least {minimum} finite values; preprocessCore's missing-value rank interpolation is undefined below this"
            )));
        }
    }
    let assignments = super::quantile_normalized_assignments(values.iter().enumerate().flat_map(
        |(row, values)| {
            samples
                .iter()
                .zip(values)
                .map(move |(&sample, &value)| (row, sample, value))
        },
    ));
    Ok(values
        .iter()
        .enumerate()
        .map(|(row, _)| {
            samples
                .iter()
                .map(|&sample| assignments.get(&(row, sample)).copied().unwrap_or(f64::NAN))
                .collect()
        })
        .collect())
}

/// The matrix entrypoint preserves the caller's sample order, which determines
/// DirectLFQ's anchor when pairwise distances tie.
fn hierarchical_matrix(values: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows: Vec<Vec<f64>> = values
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| {
                    if value.is_finite() && value > 0.0 {
                        value.log2()
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect();
    let shifts = super::hierarchical_sample_shifts(&rows, values.first().map_or(0, Vec::len), 50);
    rows.iter()
        .map(|row| {
            row.iter()
                .zip(&shifts)
                .map(|(value, shift)| (value + shift).exp2())
                .collect()
        })
        .collect()
}

/// Missing counts are zero only when fitting edgeR factors; the output retains
/// the original missing mask and measured zero values.
fn tmm_matrix(values: &[Vec<f64>], samples: &[String]) -> Result<Vec<Vec<f64>>> {
    if values
        .iter()
        .flatten()
        .any(|value| value.is_finite() && *value < 0.0)
    {
        return Err(invalid_input(
            "TMM requires non-negative linear intensities",
        ));
    }
    let columns: Vec<Vec<f64>> = (0..samples.len())
        .map(|column| values.iter().map(|row| row[column]).collect())
        .collect();
    if values
        .iter()
        .flatten()
        .any(|value| value.is_finite() && *value > 0.0)
        && columns
            .iter()
            .any(|column| !column.iter().any(|v| v.is_finite() && *v > 0.0))
    {
        return Err(invalid_input(
            "TMM requires a positive library size in every sample",
        ));
    }
    let factors = mokume_normalization::tmm_norm_factors(samples, &columns);
    Ok(values
        .iter()
        .map(|row| {
            row.iter()
                .zip(samples)
                .map(|(&value, sample)| {
                    if value.is_finite() {
                        value / factors[sample]
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect())
}

fn parse_matrix_normalization(method: &str) -> Result<Option<SampleNormalizationMethod>> {
    let parsed = match method.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "no" | "false" => None,
        "median" | "mediancenter" | "median_center" => {
            Some(SampleNormalizationMethod::MedianCenter)
        }
        "mean" | "meancenter" | "mean_center" => Some(SampleNormalizationMethod::MeanCenter),
        "quantile" => Some(SampleNormalizationMethod::Quantile),
        "rlr" => Some(SampleNormalizationMethod::Rlr),
        "loess" => Some(SampleNormalizationMethod::Loess),
        "hierarchical" => Some(SampleNormalizationMethod::Hierarchical),
        "tmm" => Some(SampleNormalizationMethod::Tmm),
        _ => {
            return Err(invalid_input(format!(
                "unknown matrix normalization method `{method}`"
            )))
        }
    };
    Ok(parsed)
}

/// Impute a row-major linear-intensity matrix without running quantification.
///
/// Imputation is fitted in log2 space and returned in linear space, matching the
/// full `features2proteins` pipeline. Columns with no positive finite value are
/// retained as entirely missing rather than populated with invented values;
/// infinite input or output is rejected.
pub fn impute_matrix(
    values: &[Vec<f64>],
    config: &ImputationConfig,
    threads: Option<usize>,
) -> Result<Vec<Vec<f64>>> {
    if threads == Some(0) {
        return Err(invalid_input("thread count must be greater than zero"));
    }
    validate_rectangular(values)?;
    validate_no_infinite(values, "imputation")?;
    validate_imputation_config(config)?;
    if !config.enabled
        || matches!(
            config.method.trim().to_ascii_lowercase().as_str(),
            "" | "none"
        )
    {
        return Ok(values.to_vec());
    }
    threading::install(threads, || impute_inner(values, config))
}

fn impute_inner(values: &[Vec<f64>], config: &ImputationConfig) -> Result<Vec<Vec<f64>>> {
    let (proteins, samples) = imputation_axes(values)?;
    let mut output = canonical_imputation_matrix(values);
    let fills = imputed_values(config, &proteins, &samples, |protein, sample| {
        let row = usize::try_from(protein.get()).ok()?;
        let column = usize::try_from(sample.get()).ok()?;
        output
            .get(row)?
            .get(column)
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .map(f64::log2)
    })?;
    for (protein, sample, value) in fills {
        let row = usize::try_from(protein.get())
            .map_err(|_| invalid_input("matrix row index is not representable"))?;
        let column = usize::try_from(sample.get())
            .map_err(|_| invalid_input("matrix column index is not representable"))?;
        output[row][column] = checked_imputed_intensity(value)?;
    }
    Ok(output)
}

fn imputation_axes(values: &[Vec<f64>]) -> Result<(Vec<ProteinId>, Vec<SampleId>)> {
    let width = values.first().map_or(0, Vec::len);
    let proteins = (0..values.len())
        .map(|row| {
            u32::try_from(row)
                .map(ProteinId::new)
                .map_err(|_| invalid_input("matrix has too many rows"))
        })
        .collect::<Result<Vec<_>>>()?;
    let samples = (0..width)
        .filter(|&column| {
            values
                .iter()
                .any(|row| row[column].is_finite() && row[column] > 0.0)
        })
        .map(|column| {
            u32::try_from(column)
                .map(SampleId::new)
                .map_err(|_| invalid_input("matrix has too many columns"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((proteins, samples))
}

fn canonical_imputation_matrix(values: &[Vec<f64>]) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| {
                    if value.is_finite() && *value > 0.0 {
                        *value
                    } else {
                        f64::NAN
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(crate) fn validate_no_infinite(values: &[Vec<f64>], operation: &str) -> Result<()> {
    if let Some((row, column)) = values.iter().enumerate().find_map(|(row, values)| {
        values
            .iter()
            .position(|value| value.is_infinite())
            .map(|column| (row, column))
    }) {
        return Err(invalid_input(format!(
            "{operation} matrix contains an infinite value at row {row}, column {column}"
        )));
    }
    Ok(())
}

pub(crate) fn checked_imputed_intensity(value: f64) -> Result<f64> {
    let intensity = value.exp2();
    if value.is_finite() && intensity.is_finite() {
        Ok(intensity)
    } else {
        Err(invalid_input(
            "imputation produced a non-finite linear intensity",
        ))
    }
}

pub(crate) fn validate_rectangular(values: &[Vec<f64>]) -> Result<usize> {
    let width = values.first().map_or(0, Vec::len);
    if let Some((row, actual)) = values
        .iter()
        .enumerate()
        .find_map(|(row, values)| (values.len() != width).then_some((row, values.len())))
    {
        return Err(invalid_input(format!(
            "matrix row {row} has {actual} columns; expected {width}"
        )));
    }
    Ok(width)
}

#[cfg(test)]
mod tests {
    use mokume_core::ImputationConfig;

    use super::{checked_imputed_intensity, impute_matrix, normalize_matrix};

    #[test]
    fn median_normalization_centers_each_log2_column() {
        let matrix = vec![vec![2.0, 8.0], vec![8.0, 32.0], vec![32.0, 128.0]];
        let samples = vec!["a".to_owned(), "b".to_owned()];
        let normalized = normalize_matrix(&matrix, &samples, "median", Some(2));
        let Ok(normalized) = normalized else {
            panic!("matrix normalization should succeed");
        };
        assert_eq!(
            normalized,
            vec![vec![0.25, 0.25], vec![1.0, 1.0], vec![4.0, 4.0]]
        );
    }

    #[test]
    fn imputation_preserves_an_entirely_missing_column() {
        let matrix = vec![vec![1.0, f64::NAN], vec![4.0, f64::NAN]];
        let config = ImputationConfig {
            enabled: true,
            method: "mean".to_owned(),
            ..ImputationConfig::default()
        };
        let imputed = impute_matrix(&matrix, &config, Some(2));
        let Ok(imputed) = imputed else {
            panic!("matrix imputation should succeed");
        };
        assert!(imputed.iter().all(|row| row[1].is_nan()));
    }

    #[test]
    fn imputation_rejects_infinite_input() {
        let matrix = vec![vec![1.0, f64::INFINITY], vec![4.0, 8.0]];
        let config = ImputationConfig {
            enabled: true,
            method: "mean".to_owned(),
            ..ImputationConfig::default()
        };
        let Err(error) = impute_matrix(&matrix, &config, Some(2)) else {
            panic!("Inf must be rejected");
        };
        assert!(error.to_string().contains("contains an infinite value"));
    }

    #[test]
    fn imputation_rejects_linear_scale_overflow() {
        let Err(error) = checked_imputed_intensity(1024.0) else {
            panic!("exp2 overflow must fail");
        };
        assert!(error.to_string().contains("non-finite linear intensity"));
    }
}
