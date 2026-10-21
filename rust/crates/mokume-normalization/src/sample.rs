use std::collections::HashMap;

use mokume_core::stats::median;
use mokume_core::Result;

use crate::math::{unsupported, valid_scale};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleNormalizationMethod {
    GlobalMedian,
    ConditionMedian,
    Quantile,
    MedianCenter,
    MeanCenter,
    Rlr,
    Loess,
    Hierarchical,
    Tmm,
}

pub fn parse_sample_normalization_method(
    method: &str,
) -> Result<Option<SampleNormalizationMethod>> {
    match method.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "no" | "false" => Ok(None),
        "globalmedian" | "global_median" | "median" => {
            Ok(Some(SampleNormalizationMethod::GlobalMedian))
        }
        "conditionmedian" | "condition_median" => {
            Ok(Some(SampleNormalizationMethod::ConditionMedian))
        }
        "quantile" => Ok(Some(SampleNormalizationMethod::Quantile)),
        "mediancenter" | "median_center" => Ok(Some(SampleNormalizationMethod::MedianCenter)),
        "meancenter" | "mean_center" => Ok(Some(SampleNormalizationMethod::MeanCenter)),
        "rlr" => Ok(Some(SampleNormalizationMethod::Rlr)),
        "loess" => Ok(Some(SampleNormalizationMethod::Loess)),
        "hierarchical" => Ok(Some(SampleNormalizationMethod::Hierarchical)),
        "tmm" => Ok(Some(SampleNormalizationMethod::Tmm)),
        _ => unsupported("sample-normalization-method"),
    }
}

pub fn global_median_sample_factors(
    sample_values: HashMap<String, Vec<f64>>,
) -> HashMap<String, f64> {
    let sample_medians = sample_medians(sample_values);
    let mut medians = sample_medians.values().copied().collect::<Vec<_>>();
    let Some(global_median) = median(&mut medians) else {
        return HashMap::new();
    };
    sample_medians
        .into_iter()
        .filter_map(|(sample, sample_median)| {
            valid_scale(sample_median).then_some((sample, global_median / sample_median))
        })
        .collect()
}

pub fn condition_median_sample_factors(
    condition_sample_values: HashMap<String, HashMap<String, Vec<f64>>>,
) -> HashMap<String, f64> {
    let mut factors = HashMap::new();
    for sample_values in condition_sample_values.into_values() {
        let mut sample_medians = sample_medians(sample_values)
            .into_iter()
            .collect::<Vec<_>>();
        if sample_medians.is_empty() {
            continue;
        }
        sample_medians.sort_by(|(left, _), (right, _)| left.cmp(right));
        let condition_mean = sample_medians
            .iter()
            .map(|(_, median)| *median)
            .sum::<f64>()
            / sample_medians.len() as f64;
        if !valid_scale(condition_mean) {
            continue;
        }
        factors.extend(
            sample_medians
                .into_iter()
                .filter_map(|(sample, sample_median)| {
                    valid_scale(sample_median).then_some((sample, condition_mean / sample_median))
                }),
        );
    }
    factors
}

fn sample_medians(sample_values: HashMap<String, Vec<f64>>) -> HashMap<String, f64> {
    sample_values
        .into_iter()
        .filter_map(|(sample, mut values)| median(&mut values).map(|median| (sample, median)))
        .collect()
}

/// edgeR `calcNormFactors(method="TMM")` defaults.
const TMM_M_TRIM: f64 = 0.3;
const TMM_A_TRIM: f64 = 0.05;

/// Compute edgeR TMM composition factors for a column-major non-negative matrix.
///
/// The columns have a shared feature order. Non-finite entries are treated as
/// absent counts (zero) for fitting; edgeR itself requires the caller to perform
/// this conversion. All-zero rows are excluded from reference selection. The
/// factors have geometric mean one. Dividing intensities by these factors is
/// Mokume's application convention, not edgeR's library-size scaling or CPM.
pub fn tmm_norm_factors(columns: &[String], matrix: &[Vec<f64>]) -> HashMap<String, f64> {
    if columns.is_empty() || matrix.len() != columns.len() {
        return HashMap::new();
    }
    let n_features = matrix[0].len();
    if matrix.iter().any(|column| column.len() != n_features) {
        return HashMap::new();
    }
    let active: Vec<_> = (0..n_features)
        .filter(|&row| {
            matrix
                .iter()
                .any(|column| column[row].is_finite() && column[row] > 0.0)
        })
        .collect();
    let observed: Vec<Vec<f64>> = matrix
        .iter()
        .map(|column| {
            active
                .iter()
                .map(|&row| {
                    if column[row].is_finite() {
                        column[row]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    if active.is_empty() || columns.len() == 1 {
        return columns.iter().map(|name| (name.clone(), 1.0)).collect();
    }
    let lib_sizes = sample_library_sizes(&observed);
    let ref_index = select_reference_index(&observed, &lib_sizes);
    let factors: Vec<_> = observed
        .iter()
        .enumerate()
        .map(|(index, column)| {
            compute_tmm_factor(
                column,
                &observed[ref_index],
                lib_sizes[index],
                lib_sizes[ref_index],
            )
        })
        .collect();
    rescale_factors(columns, &factors)
}

fn sample_library_sizes(matrix: &[Vec<f64>]) -> Vec<f64> {
    matrix.iter().map(|column| column.iter().sum()).collect()
}

/// edgeR uses library-size-normalized upper quartiles, including zero counts.
/// With sparse quartiles it instead maximizes the sum of square-root counts.
fn select_reference_index(matrix: &[Vec<f64>], lib_sizes: &[f64]) -> usize {
    let mut upper_quartiles: Vec<f64> = matrix
        .iter()
        .zip(lib_sizes)
        .map(|(column, size)| percentile_linear(&mut column.clone(), 75.0) / size)
        .collect();
    let center = upper_quartiles.iter().sum::<f64>() / upper_quartiles.len() as f64;
    if median(&mut upper_quartiles.clone()).unwrap_or(0.0) < 1e-20 {
        let scores: Vec<f64> = matrix
            .iter()
            .map(|column| column.iter().map(|v| v.sqrt()).sum())
            .collect();
        return scores
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
            .map_or(0, |(index, _)| index);
    }
    for value in &mut upper_quartiles {
        *value = (*value - center).abs();
    }
    upper_quartiles
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| a.total_cmp(b).then(i.cmp(j)))
        .map_or(0, |(index, _)| index)
}

fn rescale_factors(columns: &[String], raw_factors: &[f64]) -> HashMap<String, f64> {
    let geometric_mean = (raw_factors.iter().map(|factor| factor.ln()).sum::<f64>()
        / raw_factors.len() as f64)
        .exp();
    columns
        .iter()
        .zip(raw_factors)
        .map(|(name, factor)| (name.clone(), factor / geometric_mean))
        .collect()
}

fn compute_tmm_factor(y_sample: &[f64], y_ref: &[f64], n_sample: f64, n_ref: f64) -> f64 {
    let mut m_values = Vec::new();
    let mut a_values = Vec::new();
    let mut variances = Vec::new();
    for (&observed, &reference) in y_sample.iter().zip(y_ref) {
        let m = ((observed / n_sample) / (reference / n_ref)).log2();
        let a = ((observed / n_sample).log2() + (reference / n_ref).log2()) / 2.0;
        if m.is_finite() && a.is_finite() && a > -1e10 {
            m_values.push(m);
            a_values.push(a);
            variances.push(
                (n_sample - observed) / n_sample / observed
                    + (n_ref - reference) / n_ref / reference,
            );
        }
    }
    if m_values.iter().all(|m| m.abs() < 1e-6) {
        return 1.0;
    }
    let keep_m = trimmed_rank_mask(&m_values, TMM_M_TRIM);
    let keep_a = trimmed_rank_mask(&a_values, TMM_A_TRIM);
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for index in 0..m_values.len() {
        if keep_m[index] && keep_a[index] {
            let term = m_values[index] / variances[index];
            let weight = 1.0 / variances[index];
            // R's sum(..., na.rm=TRUE) drops NaN, but retains infinities.
            if !term.is_nan() {
                weighted_sum += term;
            }
            if !weight.is_nan() {
                weight_sum += weight;
            }
        }
    }
    let average = weighted_sum / weight_sum;
    if average.is_nan() {
        1.0
    } else {
        average.exp2()
    }
}

/// One-based average ranks retain or discard whole tie groups at trim limits.
fn trimmed_rank_mask(values: &[f64], trim: f64) -> Vec<bool> {
    let n = values.len();
    let lower = (n as f64 * trim).floor() + 1.0;
    let upper = n as f64 + 1.0 - lower;
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
    let mut keep = vec![false; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && values[order[start]] == values[order[end]] {
            end += 1;
        }
        let rank = (start + 1 + end) as f64 / 2.0;
        for &index in &order[start..end] {
            keep[index] = rank >= lower && rank <= upper;
        }
        start = end;
    }
    keep
}

/// 75th-percentile-style linear interpolation matching `np.percentile(vals, q)`
/// with the default `interpolation="linear"`. `values` is sorted in place with
/// `f64::total_cmp`; `q` is a percentile in `[0, 100]`.
fn percentile_linear(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    if values.len() == 1 {
        return values[0];
    }
    // NumPy maps the percentile to a fractional rank over [0, n-1].
    let rank = (q / 100.0) * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return values[lower];
    }
    let fraction = rank - lower as f64;
    values[lower] + (values[upper] - values[lower]) * fraction
}

#[cfg(test)]
mod tmm_tests {
    use super::*;

    fn condition_values(entries: &[(&str, f64)]) -> HashMap<String, HashMap<String, Vec<f64>>> {
        let samples = entries
            .iter()
            .map(|(sample, value)| ((*sample).to_owned(), vec![*value]))
            .collect();
        HashMap::from([("condition".to_owned(), samples)])
    }

    #[test]
    fn condition_factors_are_independent_of_hashmap_order() {
        let entries = [("a", 1.0), ("b", 1.0), ("c", 1.0e16)];
        let expected = condition_median_sample_factors(condition_values(&entries));

        for _ in 0..32 {
            let reordered = [entries[2], entries[0], entries[1]];
            let actual = condition_median_sample_factors(condition_values(&reordered));
            assert_eq!(actual, expected);
        }
    }

    /// Independent edgeR 4.8.2 `calcNormFactors(X, method="TMM")` on this
    /// 12-feature matrix, with all other options at their defaults.
    #[test]
    fn tmm_factors_match_edger() {
        let columns = vec![
            "Sample1".to_string(),
            "Sample2".to_string(),
            "Sample3".to_string(),
        ];
        // Row-major features; transposed into column-major below.
        let features: [[f64; 3]; 12] = [
            [100.0, 210.0, 95.0],
            [200.0, 390.0, 205.0],
            [300.0, 610.0, 295.0],
            [400.0, 820.0, 390.0],
            [500.0, 1010.0, 505.0],
            [600.0, 1180.0, 610.0],
            [700.0, 1420.0, 690.0],
            [800.0, 1590.0, 810.0],
            [900.0, 1810.0, 895.0],
            [1000.0, 2020.0, 990.0],
            [1100.0, 2180.0, 1110.0],
            [1200.0, 2410.0, 1190.0],
        ];
        let matrix: Vec<Vec<f64>> = (0..3)
            .map(|sample| features.iter().map(|row| row[sample]).collect())
            .collect();

        let factors = tmm_norm_factors(&columns, &matrix);
        // Independently verified with edgeR; tolerance applies to factors.
        let expected = [
            ("Sample1", 0.999_723_651_114_078_7),
            ("Sample2", 1.002_352_970_935_732_2),
            ("Sample3", 0.997_928_328_921_841_2),
        ];
        for (name, want) in expected {
            let got = factors[name];
            assert!(
                (got - want).abs() < 1e-9,
                "sample {name}: got {got}, want {want}"
            );
        }
    }
}
