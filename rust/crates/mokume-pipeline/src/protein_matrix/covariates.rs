use crate::protein_matrix::batch::factorize_batch_labels;
use crate::{HashMap, HashSet, Result, SdrfRawTable, invalid_input};

/// SDRF sample-name columns Python searches in priority order
/// (`_find_sdrf_sample_column`), lowercased to match `load_sdrf`.
pub(crate) const SDRF_SAMPLE_COLUMNS: [&str; 4] =
    ["source name", "sample name", "source_name", "sample_name"];

/// Match a requested covariate column against the lowercased SDRF headers,
/// mirroring `_match_sdrf_column`: an exact lowercased match wins, otherwise the
/// first header that is a substring of the request or contains it.
pub(crate) fn match_sdrf_column(headers: &[String], requested: &str) -> Option<usize> {
    let lower = requested.to_ascii_lowercase();
    if let Some(index) = headers.iter().position(|header| *header == lower) {
        return Some(index);
    }
    headers
        .iter()
        .position(|header| header.contains(&lower) || lower.contains(header.as_str()))
}

/// Per-sample values of one covariate column in matrix-sample order, mirroring
/// `_sample_covariate_values`: an exact `source name` hit first, else the first
/// SDRF sample that is a substring of (or contains) the matrix sample name.
/// Duplicate source names keep the last row (pandas `dict(zip(..))`). An
/// unmatched matrix sample is rejected so a requested covariate never silently
/// turns into an artificial `unknown` category.
fn covariate_values_for_samples(
    raw: &SdrfRawTable,
    sample_col: usize,
    value_col: usize,
    sample_names: &[&str],
) -> Result<Vec<String>> {
    let mut sample_to_value: HashMap<&str, &str> = HashMap::new();
    let mut sdrf_samples: Vec<&str> = Vec::with_capacity(raw.row_count());
    for row in 0..raw.row_count() {
        let key = raw.cell(row, sample_col);
        sample_to_value.insert(key, raw.cell(row, value_col));
        sdrf_samples.push(key);
    }
    sample_names
        .iter()
        .map(|&name| {
            if let Some(value) = sample_to_value.get(name) {
                return Ok((*value).to_owned());
            }
            for sdrf_sample in &sdrf_samples {
                if (sdrf_sample.contains(name) || name.contains(*sdrf_sample))
                    && let Some(value) = sample_to_value.get(*sdrf_sample)
                {
                    return Ok((*value).to_owned());
                }
            }
            Err(invalid_input(format!(
                "batch covariate SDRF has no sample matching matrix column `{name}`"
            )))
        })
        .collect()
}

/// Build the sample-major covariate matrix consumed by ComBat. Numeric SDRF
/// columns remain numeric. Categorical columns use k-1 indicator columns so a
/// nominal category is never misrepresented as an ordered scalar.
pub(crate) fn extract_sdrf_covariates(
    raw: &SdrfRawTable,
    sample_names: &[&str],
    covariate_columns: &[String],
) -> Result<Option<Vec<Vec<f64>>>> {
    if covariate_columns.is_empty() {
        return Ok(None);
    }
    let sample_col = SDRF_SAMPLE_COLUMNS
        .iter()
        .find_map(|name| raw.column_index(name))
        .ok_or_else(|| invalid_input("batch covariates require an SDRF sample-name column"))?;

    // Covariate-major encoded columns, then transposed to sample-major below.
    let mut covar_data: Vec<Vec<f64>> = Vec::new();
    for column in covariate_columns {
        let value_col = match_sdrf_column(raw.headers(), column).ok_or_else(|| {
            invalid_input(format!(
                "batch covariate column `{column}` was not found in SDRF"
            ))
        })?;
        let values = covariate_values_for_samples(raw, sample_col, value_col, sample_names)?;
        if values.iter().collect::<HashSet<_>>().len() <= 1 {
            return Err(invalid_input(format!(
                "batch covariate column `{column}` is constant and cannot affect ComBat"
            )));
        }
        covar_data.extend(encode_covariate(column, &values)?);
    }
    Ok(Some(
        (0..sample_names.len())
            .map(|sample| covar_data.iter().map(|encoded| encoded[sample]).collect())
            .collect(),
    ))
}

pub(crate) fn encode_covariate(column: &str, values: &[String]) -> Result<Vec<Vec<f64>>> {
    if values.iter().any(|value| value.trim().is_empty()) {
        return Err(invalid_input(format!(
            "covariate column `{column}` contains a missing value"
        )));
    }
    let parsed = values
        .iter()
        .map(|value| value.parse::<f64>())
        .collect::<Vec<_>>();
    if parsed.iter().all(std::result::Result::is_ok) {
        let numeric = parsed
            .into_iter()
            .filter_map(std::result::Result::ok)
            .collect::<Vec<_>>();
        if numeric.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input(format!(
                "covariate column `{column}` contains a non-finite value"
            )));
        }
        return Ok(vec![numeric]);
    }

    // Any non-numeric value makes the whole column nominal, including labels
    // that look like numbers (e.g. the cell line 5637).
    let encoded = factorize_batch_labels(values);
    let category_count = encoded.iter().copied().max().unwrap_or(0) + 1;
    Ok((1..category_count)
        .map(|category| {
            encoded
                .iter()
                .map(|value| f64::from(*value == category))
                .collect()
        })
        .collect())
}

/// Explicit batch-column values per matrix sample, mirroring
/// `_get_batch_column_values`: map via `source name -> column` (last row wins on
/// duplicate source names) and default missing samples to `"unknown"`. Returns
/// `None` (Python's not-found branch) when the SDRF lacks the lowercased column
/// or a `source name` column; the caller turns that into the same hard error
/// Python's `_detect_explicit_batches(None)` raises.
pub(crate) fn batch_column_values_for_samples(
    raw: &SdrfRawTable,
    sample_names: &[&str],
    column: &str,
) -> Result<Vec<String>> {
    let value_col = raw
        .column_index(&column.to_ascii_lowercase())
        .ok_or_else(|| invalid_input(format!("Batch column '{column}' not found in SDRF")))?;
    let sample_col = raw
        .column_index("source name")
        .ok_or_else(|| invalid_input("batch column detection requires `source name` in SDRF"))?;
    let mut sample_to_batch: HashMap<&str, &str> = HashMap::new();
    for row in 0..raw.row_count() {
        sample_to_batch.insert(raw.cell(row, sample_col), raw.cell(row, value_col));
    }
    sample_names
        .iter()
        .map(|&name| {
            sample_to_batch
                .get(name)
                .map(|value| (*value).to_owned())
                .ok_or_else(|| {
                    invalid_input(format!(
                        "batch SDRF column `{column}` has no value for matrix sample `{name}`"
                    ))
                })
        })
        .collect()
}
