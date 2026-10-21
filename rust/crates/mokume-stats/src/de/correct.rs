//! Multiple-testing correction for differential expression.
//!
//! The adaptive procedures mirror `mokume.analysis.adaptive_fdr`: Storey
//! q-values use the same smoothed pi0 estimate and reliability guard, while BKY
//! uses the same two-stage procedure as statsmodels. Both fall back to BH when
//! the pi0 estimate is not trustworthy.

use rayon::prelude::*;

use super::ensemble::median;

const DEFAULT_LAMBDAS: [f64; 19] = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95,
];

/// Adaptive FDR method requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveFdrMethod {
    Bky,
    Storey,
}

/// FDR method actually applied after the reliability gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppliedFdrMethod {
    Bh,
    Bky,
    Storey,
}

/// Benjamini-Hochberg adjusted p-values.
///
/// Finite p-values are ranked ascending, scaled by `n / rank`, made monotone by
/// a reverse cumulative minimum, and clipped to `1.0`. `NaN` inputs stay `NaN`
/// and are excluded from the count `n`, matching how mokume drops non-finite
/// p-values before correcting.
pub(crate) fn bh_adjust(pvalues: &[f64]) -> Vec<f64> {
    let mut finite = pvalues
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, value)| (index, *value))
        .collect::<Vec<_>>();
    let n = finite.len();
    let mut adjusted = vec![f64::NAN; pvalues.len()];
    if n == 0 {
        return adjusted;
    }

    finite.sort_by(|(_, a), (_, b)| a.total_cmp(b));
    let scaled = finite
        .iter()
        .enumerate()
        .map(|(rank, (_, value))| value * n as f64 / (rank + 1) as f64)
        .collect::<Vec<_>>();

    // Reverse cumulative minimum, then clip to 1.0.
    let mut running_min = f64::INFINITY;
    let mut monotone = vec![0.0; n];
    for index in (0..n).rev() {
        running_min = running_min.min(scaled[index]);
        monotone[index] = running_min.min(1.0);
    }

    for (rank, (original_index, _)) in finite.iter().enumerate() {
        adjusted[*original_index] = monotone[rank];
    }
    adjusted
}

/// Native ROTS/LimROTS permutation FDR (`ROTS::calculateFDR`,
/// `LimROTS::calculateFalseDiscoveryRate`).
///
/// For each permutation, the statistic ranked k-th by `|value|` gets the number
/// of permuted `|value|`s above it (an equal value counts once) divided by k.
/// The FDR is the median of that ratio over permutations, capped at 1 and made
/// monotone from the least extreme statistic upwards. The median keeps
/// permutations that reproduce the observed grouping, which are frequent in
/// small designs, from setting a floor under every FDR. `permuted` holds one
/// vector per permutation; a `NaN` statistic gets `NaN`.
pub(crate) fn permutation_fdr(observed: &[f64], permuted: &[Vec<f64>]) -> Vec<f64> {
    let mut fdr = vec![f64::NAN; observed.len()];
    let mut order = (0..observed.len())
        .filter(|&index| !observed[index].is_nan())
        .collect::<Vec<_>>();
    if order.is_empty() || permuted.is_empty() {
        return fdr;
    }
    order.sort_by(|&a, &b| observed[b].abs().total_cmp(&observed[a].abs()));
    let sorted = order
        .iter()
        .map(|&index| observed[index].abs())
        .collect::<Vec<_>>();
    let ratios = permuted
        .par_iter()
        .map(|null| {
            let mut null = null
                .iter()
                .map(|value| value.abs())
                .filter(|value| !value.is_nan())
                .collect::<Vec<_>>();
            null.sort_by(|a, b| b.total_cmp(a));
            let mut above = 0;
            sorted
                .iter()
                .enumerate()
                .map(|(rank, &value)| {
                    while above < null.len() && null[above] > value {
                        above += 1;
                    }
                    let tied = usize::from(null.get(above) == Some(&value));
                    (above + tied) as f64 / (rank + 1) as f64
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut running_min = f64::INFINITY;
    for rank in (0..sorted.len()).rev() {
        let mut column = ratios.iter().map(|row| row[rank]).collect::<Vec<_>>();
        running_min = running_min.min(median(&mut column).min(1.0));
        fdr[order[rank]] = running_min;
    }
    fdr
}

/// Apply BKY or Storey correction, falling back to BH when pi0 is unreliable.
pub fn adaptive_adjust(
    pvalues: &[f64],
    method: AdaptiveFdrMethod,
    alpha: f64,
) -> (Vec<f64>, AppliedFdrMethod) {
    let finite = pvalues
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, value)| (index, *value))
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return (
            vec![f64::NAN; pvalues.len()],
            match method {
                AdaptiveFdrMethod::Bky => AppliedFdrMethod::Bky,
                AdaptiveFdrMethod::Storey => AppliedFdrMethod::Storey,
            },
        );
    }

    let finite_values = finite.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    let pi0 = estimate_pi0(&finite_values);
    if !pi0_is_reliable(&finite_values, pi0) {
        return (bh_adjust(pvalues), AppliedFdrMethod::Bh);
    }

    let finite_adjusted = match method {
        AdaptiveFdrMethod::Bky => bky_adjust(&finite_values, alpha),
        AdaptiveFdrMethod::Storey => storey_qvalues(&finite_values, pi0),
    };
    let mut adjusted = vec![f64::NAN; pvalues.len()];
    for ((index, _), value) in finite.into_iter().zip(finite_adjusted) {
        adjusted[index] = value;
    }
    let applied = match method {
        AdaptiveFdrMethod::Bky => AppliedFdrMethod::Bky,
        AdaptiveFdrMethod::Storey => AppliedFdrMethod::Storey,
    };
    (adjusted, applied)
}

/// Benjamini-Krieger-Yekutieli two-stage adjusted p-values.
fn bky_adjust(pvalues: &[f64], alpha: f64) -> Vec<f64> {
    let bh = bh_adjust(pvalues);
    let n = pvalues.len();
    if n == 0 {
        return bh;
    }
    let alpha_prime = alpha / (1.0 + alpha);
    let first_stage_rejections = bh.iter().filter(|value| **value <= alpha_prime).count();
    let factor = if first_stage_rejections == 0 || first_stage_rejections == n {
        1.0 + alpha
    } else {
        (n - first_stage_rejections) as f64 / n as f64 * (1.0 + alpha)
    };
    bh.into_iter()
        .map(|value| (value * factor).min(1.0))
        .collect()
}

fn storey_qvalues(pvalues: &[f64], pi0: f64) -> Vec<f64> {
    let n = pvalues.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|left, right| pvalues[*left].total_cmp(&pvalues[*right]));
    let mut sorted = order
        .iter()
        .enumerate()
        .map(|(rank, index)| pi0 * n as f64 * pvalues[*index] / (rank + 1) as f64)
        .collect::<Vec<_>>();
    let mut running_min = f64::INFINITY;
    for index in (0..n).rev() {
        running_min = running_min.min(sorted[index]);
        sorted[index] = running_min.min(1.0);
    }
    let mut adjusted = vec![0.0; n];
    for (rank, index) in order.into_iter().enumerate() {
        adjusted[index] = sorted[rank];
    }
    adjusted
}

fn estimate_pi0(pvalues: &[f64]) -> f64 {
    let curve = pi0_curve(pvalues);
    let smoothed = smooth_spline_df(&DEFAULT_LAMBDAS, &curve, 3.0)
        .last()
        .copied()
        .unwrap_or(1.0)
        .clamp(f64::MIN_POSITIVE, 1.0);
    let lower_bound = curve
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .clamp(f64::MIN_POSITIVE, 1.0);
    smoothed.max(lower_bound)
}

fn pi0_curve(pvalues: &[f64]) -> Vec<f64> {
    DEFAULT_LAMBDAS
        .iter()
        .map(|lambda| {
            let right_tail = pvalues.iter().filter(|value| **value >= *lambda).count();
            right_tail as f64 / (pvalues.len() as f64 * (1.0 - lambda))
        })
        .collect()
}

fn pi0_is_reliable(pvalues: &[f64], pi0: f64) -> bool {
    if pvalues.len() < 100 || pi0 >= 1.0 - 1e-3 {
        return false;
    }
    let left_density =
        pvalues.iter().filter(|value| **value < 0.05).count() as f64 / pvalues.len() as f64 / 0.05;
    let right_density =
        pvalues.iter().filter(|value| **value >= 0.5).count() as f64 / pvalues.len() as f64 / 0.5;
    left_density > right_density * 1.1
}

/// Natural cubic smoothing spline with effective degrees of freedom pinned to
/// `target_df`, matching Python's `_smooth_spline_df` implementation.
fn smooth_spline_df(x: &[f64], y: &[f64], target_df: f64) -> Vec<f64> {
    let n = x.len();
    if n <= 3 || target_df >= n as f64 {
        return y.to_vec();
    }
    let penalty = spline_penalty(x);
    let (mut low, mut high) = (1e-8_f64, 1e8_f64);
    for _ in 0..60 {
        let middle = (low * high).sqrt();
        if fit_spline(&penalty, y, middle).1 > target_df {
            low = middle;
        } else {
            high = middle;
        }
    }
    fit_spline(&penalty, y, (low * high).sqrt()).0
}

fn spline_penalty(x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let m = n - 2;
    let h = x
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let mut q = vec![vec![0.0; m]; n];
    let mut r = vec![vec![0.0; m]; m];
    for j in 1..n - 1 {
        let column = j - 1;
        q[j - 1][column] = 1.0 / h[j - 1];
        q[j][column] = -(1.0 / h[j - 1] + 1.0 / h[j]);
        q[j + 1][column] = 1.0 / h[j];
        r[column][column] = (h[j - 1] + h[j]) / 3.0;
        if column + 1 < m {
            r[column][column + 1] = h[j] / 6.0;
            r[column + 1][column] = h[j] / 6.0;
        }
    }

    let r_inverse = invert(&r);
    let mut penalty = vec![vec![0.0; n]; n];
    for row in 0..n {
        for column in 0..n {
            for left in 0..m {
                for right in 0..m {
                    penalty[row][column] +=
                        q[row][left] * r_inverse[left][right] * q[column][right];
                }
            }
        }
    }
    penalty
}

fn fit_spline(penalty: &[Vec<f64>], y: &[f64], lambda: f64) -> (Vec<f64>, f64) {
    let mut system = penalty.to_vec();
    for (index, row) in system.iter_mut().enumerate() {
        for value in row.iter_mut() {
            *value *= lambda;
        }
        row[index] += 1.0;
    }
    let inverse = invert(&system);
    let fitted = inverse
        .iter()
        .map(|row| {
            row.iter()
                .zip(y)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect::<Vec<f64>>();
    let df = (0..inverse.len())
        .map(|index| inverse[index][index])
        .sum::<f64>();
    (fitted, df)
}

fn invert(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut augmented = vec![vec![0.0; 2 * n]; n];
    for row in 0..n {
        augmented[row][..n].copy_from_slice(&matrix[row]);
        augmented[row][n + row] = 1.0;
    }
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|left, right| {
                augmented[*left][column]
                    .abs()
                    .total_cmp(&augmented[*right][column].abs())
            })
            .unwrap_or(column);
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        for value in &mut augmented[column] {
            *value /= scale;
        }
        let pivot_row = augmented[column].clone();
        for (row, row_values) in augmented.iter_mut().enumerate() {
            if row == column {
                continue;
            }
            let factor = row_values[column];
            for (value, pivot_value) in row_values.iter_mut().zip(&pivot_row) {
                *value -= factor * pivot_value;
            }
        }
    }
    augmented.into_iter().map(|row| row[n..].to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::{adaptive_adjust, bh_adjust, permutation_fdr, AdaptiveFdrMethod, AppliedFdrMethod};

    fn assert_close(actual: f64, expected: f64) {
        let tol = 1e-9 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual} expected={expected}"
        );
    }

    // Reference from statsmodels fdr_bh on the limma oracle p-values (6 tests,
    // including a tie at 3.1734e-05). Adjusted values verified against
    // `multipletests(p, method="fdr_bh")`.
    #[test]
    fn matches_statsmodels_fdr_bh() {
        let p = [
            1.712_569_861_574_927e-7,
            3.1734308960086415e-05,
            3.1734308960086415e-05,
            0.544_763_715_713_143_1,
            1.0,
            0.941_320_895_121_956_2,
        ];
        let adjusted = bh_adjust(&p);
        assert_close(adjusted[0], 1.0275419169449564e-06);
        assert_close(adjusted[1], 6.346861792017283e-05);
        assert_close(adjusted[2], 6.346861792017283e-05);
        assert_close(adjusted[3], 0.817_145_573_569_714_6);
        assert_close(adjusted[4], 1.0);
        assert_close(adjusted[5], 1.0);
    }

    #[test]
    fn passes_through_non_finite() {
        let p = [0.01, f64::NAN, 0.02];
        let adjusted = bh_adjust(&p);
        // n = 2 finite values: 0.01 -> 0.02 (0.01*2/1), 0.02 -> 0.02 (0.02*2/2).
        assert_close(adjusted[0], 0.02);
        assert!(adjusted[1].is_nan());
        assert_close(adjusted[2], 0.02);
    }

    #[test]
    fn adaptive_methods_match_python_oracle() {
        let mut p = (0..900)
            .map(|index| (index as f64 + 0.5) / 900.0)
            .collect::<Vec<_>>();
        p.extend((0..300).map(|index| 0.05 * ((index as f64 + 0.5) / 300.0).powi(3)));

        let (bky, bky_used) = adaptive_adjust(&p, AdaptiveFdrMethod::Bky, 0.05);
        let (storey, storey_used) = adaptive_adjust(&p, AdaptiveFdrMethod::Storey, 0.05);
        assert_eq!(bky_used, AppliedFdrMethod::Bky);
        assert_eq!(storey_used, AppliedFdrMethod::Storey);
        assert_eq!(bky.iter().filter(|value| **value < 0.05).count(), 168);
        assert_eq!(storey.iter().filter(|value| **value < 0.05).count(), 187);
        assert_close(bky[0], 0.008_955_882_352_941_176);
        assert_close(bky[900], 2.537_500_000_000_000_6e-7);
        assert_close(storey[0], 0.007_352_941_176_470_587);
        assert_close(storey[900], 2.083_333_333_333_333_3e-7);
    }

    #[test]
    fn adaptive_methods_fall_back_when_pi0_is_unreliable() {
        let p = (0..80)
            .map(|index| 0.05 * ((index as f64 + 0.5) / 80.0).powi(3))
            .collect::<Vec<_>>();
        let expected = bh_adjust(&p);
        for method in [AdaptiveFdrMethod::Bky, AdaptiveFdrMethod::Storey] {
            let (adjusted, used) = adaptive_adjust(&p, method, 0.05);
            assert_eq!(used, AppliedFdrMethod::Bh);
            for (actual, expected) in adjusted.into_iter().zip(&expected) {
                assert_close(actual, *expected);
            }
        }
    }

    #[test]
    fn adaptive_methods_preserve_non_finite_positions() {
        let p = [0.001, f64::NAN, 0.5];
        let (adjusted, used) = adaptive_adjust(&p, AdaptiveFdrMethod::Storey, 0.05);
        assert_eq!(used, AppliedFdrMethod::Bh);
        assert_close(adjusted[0], 0.002);
        assert!(adjusted[1].is_nan());
        assert_close(adjusted[2], 0.5);
    }

    #[test]
    fn permutation_fdr_follows_rots_calculate_fdr() {
        // Ranked |d|: 3, 2, 1. Per permutation, (above + tie) / rank:
        // [0, 1/2, 2/3], [1, 1/2, 1/3], [0, 0, 1/3]; medians 0, 1/2, 1/3;
        // the cumulative minimum from the bottom gives 0, 1/3, 1/3.
        let fdr = permutation_fdr(
            &[3.0, -1.0, 2.0],
            &[
                vec![0.5, 2.5, 1.0],
                vec![-3.0, 0.0, 0.1],
                vec![1.0, 1.0, -1.0],
            ],
        );
        assert_close(fdr[0], 0.0);
        assert_close(fdr[1], 1.0 / 3.0);
        assert_close(fdr[2], 1.0 / 3.0);
    }

    #[test]
    fn permutation_fdr_is_not_floored_by_a_reproduced_grouping() {
        // One permutation in three reproduces the observed statistics; a pooled
        // estimate could not drop below 1/3, the median ignores it.
        let observed = [4.0, 3.0, 0.2, 0.1];
        let null = vec![0.3, 0.2, 0.1, 0.0];
        let fdr = permutation_fdr(&observed, &[null.clone(), null, observed.to_vec()]);
        assert_close(fdr[0], 0.0);
        assert_close(fdr[1], 0.0);
        assert!(fdr[2] > 0.0);
    }
}
