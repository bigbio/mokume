use std::collections::HashMap;

use mokume_core::Result;
use mokume_core::stats::{mean_positive, median, quantile_linear};

use crate::math::{unsupported, valid_scale};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunCellKey {
    pub sample: String,
    pub run: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunNormalizationMethod {
    Mean,
    Median,
    Max,
    Global,
    MaxMin,
    Iqr,
}

pub fn parse_run_normalization_method(method: &str) -> Result<Option<RunNormalizationMethod>> {
    match method.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "no" | "false" => Ok(None),
        "mean" => Ok(Some(RunNormalizationMethod::Mean)),
        "median" => Ok(Some(RunNormalizationMethod::Median)),
        "max" => Ok(Some(RunNormalizationMethod::Max)),
        "global" => Ok(Some(RunNormalizationMethod::Global)),
        "max_min" | "maxmin" | "max-min" => Ok(Some(RunNormalizationMethod::MaxMin)),
        "iqr" => Ok(Some(RunNormalizationMethod::Iqr)),
        _ => unsupported("run-normalization-method"),
    }
}

pub fn run_normalization_factors(
    method: RunNormalizationMethod,
    run_values: HashMap<RunCellKey, Vec<f64>>,
) -> HashMap<RunCellKey, f64> {
    run_normalization_transforms(method, run_values)
        .into_iter()
        .filter_map(|(key, transform)| transform.as_factor().map(|factor| (key, factor)))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunNormalizationTransform {
    scale: f64,
    offset: f64,
}

impl RunNormalizationTransform {
    pub const fn new(scale: f64, offset: f64) -> Self {
        Self { scale, offset }
    }

    pub fn apply(self, value: f64) -> f64 {
        value * self.scale + self.offset
    }

    pub fn as_factor(self) -> Option<f64> {
        (self.offset == 0.0).then_some(self.scale)
    }
}

pub fn run_normalization_transforms(
    method: RunNormalizationMethod,
    run_values: HashMap<RunCellKey, Vec<f64>>,
) -> HashMap<RunCellKey, RunNormalizationTransform> {
    if method == RunNormalizationMethod::MaxMin {
        return max_min_run_transforms(run_values);
    }

    let mut run_metrics = Vec::new();
    for (key, mut values) in run_values {
        if let Some(metric) = run_metric(method, &mut values) {
            if valid_scale(metric) {
                run_metrics.push((key, metric));
            }
        }
    }
    run_metrics.sort_by(|(left, _), (right, _)| {
        left.sample
            .cmp(&right.sample)
            .then_with(|| left.run.cmp(&right.run))
    });

    let mut metrics_by_sample = HashMap::<String, Vec<f64>>::new();
    for (key, metric) in &run_metrics {
        metrics_by_sample
            .entry(key.sample.clone())
            .or_default()
            .push(*metric);
    }
    let sample_average = metrics_by_sample
        .into_iter()
        .filter_map(|(sample, metrics)| {
            (metrics.len() > 1)
                .then(|| (sample, metrics.iter().sum::<f64>() / metrics.len() as f64))
        })
        .collect::<HashMap<_, _>>();

    run_metrics
        .into_iter()
        .filter_map(|(key, metric)| {
            let average = sample_average.get(&key.sample).copied()?;
            valid_scale(average)
                .then_some((key, RunNormalizationTransform::new(average / metric, 0.0)))
        })
        .collect()
}

fn run_metric(method: RunNormalizationMethod, values: &mut Vec<f64>) -> Option<f64> {
    match method {
        RunNormalizationMethod::Mean => mean_positive(values),
        RunNormalizationMethod::Median => median(values),
        RunNormalizationMethod::Max => values
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .max_by(f64::total_cmp),
        RunNormalizationMethod::Global => {
            let sum = values
                .iter()
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
                .sum::<f64>();
            (sum > 0.0).then_some(sum)
        }
        RunNormalizationMethod::MaxMin => None,
        RunNormalizationMethod::Iqr => {
            // Drop missing-as-zero and non-finite values before the quartiles,
            // matching the Mean/Median/Max/Global metrics above; otherwise zeros
            // distort the range and NaN (sorted last by total_cmp) poisons
            // q75.
            values.retain(|value| value.is_finite() && *value > 0.0);
            let q25 = quantile_linear(values, 0.25)?;
            let q75 = quantile_linear(values, 0.75)?;
            Some(q75 - q25)
        }
    }
}

fn max_min_run_transforms(
    run_values: HashMap<RunCellKey, Vec<f64>>,
) -> HashMap<RunCellKey, RunNormalizationTransform> {
    let mut run_ranges = Vec::<(RunCellKey, (f64, f64))>::new();
    for (key, values) in run_values {
        if let Some(range) = positive_range(&values) {
            run_ranges.push((key, range));
        }
    }
    run_ranges.sort_by(|(left, _), (right, _)| {
        left.sample
            .cmp(&right.sample)
            .then_with(|| left.run.cmp(&right.run))
    });

    let mut ranges_by_sample = HashMap::<String, Vec<(f64, f64)>>::new();
    for (key, range) in &run_ranges {
        ranges_by_sample
            .entry(key.sample.clone())
            .or_default()
            .push(*range);
    }
    let target_ranges = ranges_by_sample
        .into_iter()
        .filter_map(|(sample, ranges)| {
            (ranges.len() > 1).then(|| {
                let minimum =
                    ranges.iter().map(|(minimum, _)| minimum).sum::<f64>() / ranges.len() as f64;
                let maximum =
                    ranges.iter().map(|(_, maximum)| maximum).sum::<f64>() / ranges.len() as f64;
                (sample, (minimum, maximum))
            })
        })
        .collect::<HashMap<_, _>>();

    run_ranges
        .into_iter()
        .filter_map(|(key, (source_minimum, source_maximum))| {
            let (target_minimum, target_maximum) = target_ranges.get(&key.sample).copied()?;
            affine_range_transform(
                source_minimum,
                source_maximum,
                target_minimum,
                target_maximum,
            )
            .map(|transform| (key, transform))
        })
        .collect()
}

fn positive_range(values: &[f64]) -> Option<(f64, f64)> {
    let mut values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0);
    let first = values.next()?;
    let mut minimum = first;
    let mut maximum = first;
    for value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    (maximum > minimum).then_some((minimum, maximum))
}

fn affine_range_transform(
    source_minimum: f64,
    source_maximum: f64,
    target_minimum: f64,
    target_maximum: f64,
) -> Option<RunNormalizationTransform> {
    let source_width = source_maximum - source_minimum;
    let target_width = target_maximum - target_minimum;
    if source_width <= 0.0
        || target_width <= 0.0
        || !source_width.is_finite()
        || !target_width.is_finite()
        || !source_minimum.is_finite()
        || !target_minimum.is_finite()
    {
        return None;
    }
    let scale = target_width / source_width;
    let offset = target_minimum - source_minimum * scale;
    (scale.is_finite() && offset.is_finite())
        .then_some(RunNormalizationTransform::new(scale, offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keyed_run_values(entries: &[(&str, f64)]) -> HashMap<RunCellKey, Vec<f64>> {
        entries
            .iter()
            .map(|(run, value)| {
                (
                    RunCellKey {
                        sample: "sample".to_owned(),
                        run: (*run).to_owned(),
                    },
                    vec![*value],
                )
            })
            .collect()
    }

    #[test]
    fn run_factor_targets_are_independent_of_hashmap_order() {
        let entries = [("a", 1.0), ("b", 1.0), ("c", 1.0e16)];
        let expected = run_normalization_transforms(
            RunNormalizationMethod::Median,
            keyed_run_values(&entries),
        );

        for _ in 0..32 {
            let reordered = [entries[2], entries[0], entries[1]];
            let actual = run_normalization_transforms(
                RunNormalizationMethod::Median,
                keyed_run_values(&reordered),
            );
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn iqr_metric_excludes_zeros_and_non_finite() {
        // IQR must drop missing-as-zero and non-finite values like
        // the Mean/Median/Max/Global metrics, so it reflects only the observed
        // intensities. Over [100,200,300,400]: q25=175, q75=325, IQR=150.
        let mut with_zeros = vec![0.0, 0.0, 0.0, 0.0, 100.0, 200.0, 300.0, 400.0];
        let Some(metric) = run_metric(RunNormalizationMethod::Iqr, &mut with_zeros) else {
            panic!("IQR metric should be Some for finite-positive values");
        };
        assert!(
            (metric - 150.0).abs() < 1e-9,
            "zeros not excluded: {metric}"
        );

        // NaN must not survive into the quantiles (it sorts last under total_cmp).
        let mut with_nan = vec![f64::NAN, 100.0, 200.0, 300.0, 400.0];
        let Some(metric) = run_metric(RunNormalizationMethod::Iqr, &mut with_nan) else {
            panic!("IQR metric should be Some after dropping NaN");
        };
        assert!(
            (metric - 150.0).abs() < 1e-9,
            "NaN poisoned the metric: {metric}"
        );
    }
}
