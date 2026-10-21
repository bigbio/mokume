//! Shared numeric primitives reused across quantification, normalization, and
//! imputation crates. These are intentionally dependency-free so any crate that
//! depends on `mokume-core` can reuse them without cross-crate coupling.

mod lowess;
pub use lowess::lowess_fit;

/// Linear-interpolated quantile of `values`. Sorts `values` in place. Returns
/// `None` for an empty slice or an out-of-range quantile.
pub fn quantile_linear(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() || !quantile.is_finite() || !(0.0..=1.0).contains(&quantile) {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let position = (values.len() - 1) as f64 * quantile;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return values.get(lower).copied();
    }
    let lower_value = *values.get(lower)?;
    let upper_value = *values.get(upper)?;
    Some(lower_value + (upper_value - lower_value) * (position - lower as f64))
}

/// Median over the finite, strictly positive values of `values` (others are
/// dropped). Mutates `values`.
pub fn median(values: &mut Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite() && *value > 0.0);
    median_sorted(values)
}

/// Median over the finite values of `values` (non-finite are dropped). Mutates
/// `values`.
pub fn median_finite(values: &mut Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    median_sorted(values)
}

/// Mean over the finite, strictly positive values of `values`.
pub fn mean_positive(values: &[f64]) -> Option<f64> {
    let values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Mean over the finite values of `values`.
pub fn mean_finite(values: &[f64]) -> Option<f64> {
    let values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Population standard deviation over the finite values of `values`.
pub fn finite_sd(values: &[f64]) -> Option<f64> {
    let values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt())
}

/// Median of an already-filtered slice; sorts in place. No value filtering.
pub fn median_sorted(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[midpoint - 1] + values[midpoint]) / 2.0)
    } else {
        Some(values[midpoint])
    }
}
