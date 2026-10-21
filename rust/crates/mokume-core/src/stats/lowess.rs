//! LOWESS following statsmodels 0.14.6 `_smoothers_lowess.pyx`.
//!
//! Supports evaluation at the input points with `delta=0`. Missing-value
//! filtering and sorting belong to the caller, as with the upstream Cython core.

use rayon::prelude::*;

use super::median_sorted;

/// Smooth finite `y` at sorted, finite `x`, using local linear regression and
/// `iterations` residual reweightings. The inputs must have the same length and
/// `frac` must lie in `[0, 1]`.
pub fn lowess_fit(x: &[f64], y: &[f64], frac: f64, iterations: usize) -> Vec<f64> {
    assert_eq!(x.len(), y.len(), "LOWESS x/y lengths differ");
    assert!((0.0..=1.0).contains(&frac), "LOWESS frac must be in [0, 1]");
    let n = x.len();
    if n < 2 {
        return y.to_vec();
    }
    let window = ((frac * n as f64 + 1e-10) as usize).clamp(2, n);
    let fit_points = window_starts(x, window);
    let mut fitted = vec![0.0; n];
    let mut robustness = vec![1.0; n];
    for iteration in 0..=iterations {
        let fits = fit_points
            .par_iter()
            .map_init(
                || vec![0.0; window],
                |weights, &(i, left)| {
                    let right = left + window;
                    let radius = (x[i] - x[left]).max(x[right - 1] - x[i]);
                    local_fit(
                        &x[left..right],
                        &y[left..right],
                        &robustness[left..right],
                        weights,
                        x[i],
                        radius,
                        y[i],
                    )
                },
            )
            .collect::<Vec<_>>();
        for (&(i, _), fit) in fit_points.iter().zip(fits) {
            fitted[i] = fit;
        }
        // Upstream reuses the fit for tied x values, even if regression
        // failed and the first tied point's original y was returned.
        for i in 1..n {
            if x[i] == x[i - 1] {
                fitted[i] = fitted[i - 1];
            }
        }
        if iteration < iterations {
            update_robustness(y, &fitted, &mut robustness);
        }
    }
    fitted
}

/// The first point of each run of tied x values, with its window start.
///
/// Upstream slides the window start while x[i] exceeds the window midpoint.
/// The midpoints never decrease, so a binary search finds the same start for
/// each point, and the points can then be fitted independently.
fn window_starts(x: &[f64], window: usize) -> Vec<(usize, usize)> {
    let midpoints = (0..x.len() - window)
        .map(|left| (x[left] + x[left + window]) / 2.0)
        .collect::<Vec<_>>();
    (0..x.len())
        .filter(|&i| i == 0 || x[i] != x[i - 1])
        .map(|i| (i, midpoints.partition_point(|&midpoint| x[i] > midpoint)))
        .collect()
}

fn local_fit(
    x: &[f64],
    y: &[f64],
    robustness: &[f64],
    weights: &mut [f64],
    xval: f64,
    radius: f64,
    original: f64,
) -> f64 {
    for ((weight, &xj), &robust) in weights.iter_mut().zip(x).zip(robustness) {
        let distance = (xj - xval).abs() / radius;
        let tricube = 1.0 - distance * distance * distance;
        *weight = tricube * tricube * tricube * robust;
    }
    if weights.iter().filter(|&&weight| weight > 1e-12).count() < 2 {
        return original;
    }
    // Upstream divides every weight by their sum before the moments below;
    // dividing the unnormalized moments once is the same fit up to rounding.
    let [sum, weighted_x] = lane_sums(weights, x, y, |w, xj, _| [w, w * xj]);
    let center = weighted_x / sum;
    let [spread, level, slope] = lane_sums(weights, x, y, |w, xj, yj| {
        let offset = w * (xj - center);
        [offset * (xj - center), w * yj, offset * yj]
    });
    let variance = (spread / sum).max(1e-12);
    (level + (xval - center) * slope / variance) / sum
}

const LANES: usize = 8;

/// Window sums of `terms(w, x, y)` in eight interleaved accumulators each; one
/// accumulator would serialize the window on floating-point add latency.
fn lane_sums<const N: usize>(
    weights: &[f64],
    x: &[f64],
    y: &[f64],
    terms: impl Fn(f64, f64, f64) -> [f64; N],
) -> [f64; N] {
    let mut lanes = [[0.0; LANES]; N];
    let chunks = weights
        .chunks_exact(LANES)
        .zip(x.chunks_exact(LANES))
        .zip(y.chunks_exact(LANES));
    for ((w, xs), ys) in chunks {
        for lane in 0..LANES {
            for (sums, term) in lanes.iter_mut().zip(terms(w[lane], xs[lane], ys[lane])) {
                sums[lane] += term;
            }
        }
    }
    let start = weights.len() / LANES * LANES;
    let mut tail = [0.0; N];
    for ((&w, &xj), &yj) in weights[start..].iter().zip(&x[start..]).zip(&y[start..]) {
        for (sum, term) in tail.iter_mut().zip(terms(w, xj, yj)) {
            *sum += term;
        }
    }
    std::array::from_fn(|k| {
        let [a, b, c, d, e, f, g, h] = lanes[k];
        ((a + b) + (c + d)) + ((e + f) + (g + h)) + tail[k]
    })
}

fn update_robustness(y: &[f64], fitted: &[f64], robustness: &mut [f64]) {
    let mut residuals = y
        .iter()
        .zip(fitted)
        .map(|(y, f)| (y - f).abs())
        .collect::<Vec<_>>();
    let median = median_sorted(&mut residuals).unwrap_or(0.0);
    for ((weight, &yj), &fit) in robustness.iter_mut().zip(y).zip(fitted) {
        let residual = (yj - fit).abs();
        let scaled = if median == 0.0 {
            f64::from(residual > 0.0)
        } else {
            (residual / (6.0 * median)).min(1.0)
        };
        *weight = (1.0 - scaled * scaled).powi(2);
    }
}

#[cfg(test)]
mod tests {
    use super::lowess_fit;
    use std::collections::BTreeMap;

    #[test]
    fn matches_statsmodels_0_14_6() -> Result<(), Box<dyn std::error::Error>> {
        let mut cases = BTreeMap::<_, Vec<_>>::new();
        for line in include_str!("../../tests/data/lowess_statsmodels.tsv").lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            cases
                .entry((fields[0], fields[1], fields[2]))
                .or_default()
                .push((
                    fields[3].parse::<f64>()?,
                    fields[4].parse::<f64>()?,
                    fields[5].parse::<f64>()?,
                ));
        }
        for ((name, frac, iterations), rows) in cases {
            let x = rows.iter().map(|r| r.0).collect::<Vec<_>>();
            let y = rows.iter().map(|r| r.1).collect::<Vec<_>>();
            let actual = lowess_fit(&x, &y, frac.parse()?, iterations.parse()?);
            for (i, (&value, row)) in actual.iter().zip(&rows).enumerate() {
                let tolerance = 1e-10 + 1e-9 * row.2.abs();
                assert!(
                    (value - row.2).abs() <= tolerance,
                    "{name}[{i}]: {value} != {}",
                    row.2
                );
            }
        }
        Ok(())
    }
}
