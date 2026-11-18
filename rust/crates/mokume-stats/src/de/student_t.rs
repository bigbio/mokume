//! Student's t-distribution survival function for moderated p-values.
//!
//! limma forms two-sided p-values as `2 * scipy.stats.t.sf(|t|, df)`. This
//! module provides [`student_t_sf`] via the regularized incomplete beta
//! function `I_x(a, b)`, evaluated with a Lentz continued fraction (the
//! Numerical Recipes `betai`/`betacf` scheme), matching SciPy to ~1e-12.

/// Upper-tail Student's t survival function `P(T > t)` with `df` degrees of
/// freedom: `sf(t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)` for `t >= 0`, and
/// `sf(t) = 1 - sf(-t)` for `t < 0`.
pub(crate) fn student_t_sf(t: f64, df: f64) -> f64 {
    if !t.is_finite() || !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    let half = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x);
    if t >= 0.0 { half } else { 1.0 - half }
}

/// Natural logarithm of the two-sided Student's t p-value.
///
/// This evaluates `log(I_x(df/2, 1/2))` directly, so probabilities smaller
/// than the positive `f64` range retain their ordering information.
pub(crate) fn student_t_two_sided_log_pvalue(t: f64, df: f64) -> f64 {
    if !t.is_finite() || !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    let t_abs = t.abs();
    let x = df / (df + t_abs * t_abs);
    log_regularized_incomplete_beta(df / 2.0, 0.5, x)
}

/// Regularized incomplete beta function `I_x(a, b)`.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b);
    let front = (ln_beta + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Natural logarithm of the regularized incomplete beta `I_x(a, b)`.
fn log_regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return 0.0;
    }
    let log_front =
        libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b) + a * x.ln() + b * (-x).ln_1p();
    if x < (a + 1.0) / (a + b + 2.0) {
        log_front + beta_continued_fraction(a, b, x).ln() - a.ln()
    } else {
        let complement = (log_front.exp() * beta_continued_fraction(b, a, 1.0 - x) / b).min(1.0);
        (-complement).ln_1p()
    }
}

/// Lentz evaluation of the continued fraction used by the incomplete beta.
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 300;
    const EPSILON: f64 = 1e-15;
    const TINY: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITERATIONS {
        let m = m as f64;
        let m2 = 2.0 * m;

        let even = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + even * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + even / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        let odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + odd * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + odd / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::{student_t_sf, student_t_two_sided_log_pvalue};

    fn assert_close(actual: f64, expected: f64) {
        let tol = 1e-11 * expected.abs().max(1e-12);
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual} expected={expected}"
        );
    }

    // Reference values from scipy.stats.t.sf at the limma moderated df.
    #[test]
    fn matches_scipy_t_sf() {
        assert_close(
            student_t_sf(2.0, 4.560_029_503_876_107),
            0.053_713_779_501_067_73,
        );
        assert_close(
            student_t_sf(16.3657, 4.560_029_503_876_107),
            1.586730940101415e-05,
        );
    }

    #[test]
    fn symmetric_about_zero() {
        let df = 7.0;
        assert_close(student_t_sf(0.0, df), 0.5);
        let positive = student_t_sf(1.3, df);
        let negative = student_t_sf(-1.3, df);
        assert_close(positive + negative, 1.0);
    }

    #[test]
    fn log_pvalue_survives_float_underflow() {
        let first = student_t_two_sided_log_pvalue(40.837, 10_000.0);
        let second = student_t_two_sided_log_pvalue(87.944, 10_000.0);
        assert!(first.is_finite());
        assert!(second.is_finite());
        assert!((first - -775.038_224_714_875_7).abs() < 3e-11);
        assert!((second - -2_868.950_716_179_171).abs() < 3e-11);
        assert!(second < first);
        assert_eq!(first.exp(), 0.0);
    }
}
