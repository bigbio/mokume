//! Two-column cell-means QR fit, followed by limma's equal-df moderation.
//!
//! QR residual effects matter for bootstrap samples with zero within-group
//! variance: subtracting group means can manufacture exact zero residuals and
//! trigger a different variance-prior branch than R's `lm.fit`.

use super::super::limma::squeeze_var;

pub(super) struct Fit {
    pub contrast: Vec<f64>,
    pub standard_error: Vec<f64>,
}

impl Fit {
    pub fn statistics(&self, a1: f64, a2: f64) -> Vec<f64> {
        self.contrast
            .iter()
            .zip(&self.standard_error)
            .map(|(&d, &s)| d.abs() / (a1 + a2 * s))
            .collect()
    }
}

/// Fit in the original column order, including for permuted group labels.
pub(super) fn fit(data: &[Vec<f64>], group_a: &[bool]) -> Fit {
    let qr = TwoGroupQr::new(group_a);
    let coefficients_and_variance = data.iter().map(|row| qr.fit(row)).collect::<Vec<_>>();
    let variance = coefficients_and_variance
        .iter()
        .map(|r| r.1)
        .collect::<Vec<_>>();
    let df = vec![(group_a.len() - 2) as f64; data.len()];
    let (posterior, _, _) = squeeze_var(&variance, &df);
    Fit {
        contrast: coefficients_and_variance.iter().map(|r| r.0).collect(),
        standard_error: posterior.iter().map(|s| s.sqrt() * qr.unscaled).collect(),
    }
}

struct TwoGroupQr {
    reflectors: [Vec<f64>; 2],
    diagonal: [f64; 2],
    upper: f64,
    unscaled: f64,
}

impl TwoGroupQr {
    fn new(group_a: &[bool]) -> Self {
        let mut columns = [
            group_a.iter().map(|&a| f64::from(a)).collect::<Vec<_>>(),
            group_a.iter().map(|&a| f64::from(!a)).collect::<Vec<_>>(),
        ];
        let first_norm = norm(&columns[0]);
        for x in &mut columns[0] {
            *x *= 1.0 / first_norm;
        }
        columns[0][0] += 1.0;
        let [first, second] = &mut columns;
        reflect(first, second);
        let upper = second[0];
        let second_norm =
            norm(&second[1..]).copysign(if second[1] == 0.0 { 1.0 } else { second[1] });
        for x in &mut second[1..] {
            *x *= 1.0 / second_norm;
        }
        second[1] += 1.0;
        let diagonal = [-first_norm, -second_norm];
        // limma's contrasts.fit treats the two one-hot coefficients as
        // orthogonal (correlation <1e-14). Preserve sqrt-then-square rounding
        // of stdev.unscaled rather than retaining a numerical cross term.
        let inv00 = 1.0 / diagonal[0];
        let inv11 = 1.0 / diagonal[1];
        let inv01 = -upper * inv00 * inv11;
        let sd_a = (inv00 * inv00 + inv01 * inv01).sqrt();
        let sd_b = (inv11 * inv11).sqrt();
        let unscaled = (sd_a * sd_a + sd_b * sd_b).sqrt();
        Self {
            reflectors: columns,
            diagonal,
            upper,
            unscaled,
        }
    }

    fn fit(&self, row: &[f64]) -> (f64, f64) {
        let mut effects = row.to_vec();
        reflect(&self.reflectors[0], &mut effects);
        reflect(&self.reflectors[1][1..], &mut effects[1..]);
        let b = effects[1] / self.diagonal[1];
        let a = (effects[0] - self.upper * b) / self.diagonal[0];
        // lm.series computes sigma from the residual effects, then eBayes
        // squares sigma. Keep that rounding and the accurate colMeans sum.
        let sigma = (sum_squares(&effects[2..]) / (row.len() - 2) as f64).sqrt();
        (a - b, sigma * sigma)
    }
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn sum_squares(values: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    let mut correction = 0.0;
    for &x in values {
        let value = x * x;
        let next = sum + value;
        correction += if sum >= value {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

fn reflect(v: &[f64], y: &mut [f64]) {
    let projection = -v
        .iter()
        .zip(y.iter())
        .fold(0.0, |sum, (&a, &b)| sum + a * b)
        / v[0];
    for (&vj, yj) in v.iter().zip(y) {
        *yj += projection * vj;
    }
}
