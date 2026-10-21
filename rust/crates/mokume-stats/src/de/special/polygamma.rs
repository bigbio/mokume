//! Special functions used by the limma differential-expression port.
//!
//! `digamma`, `trigamma`, and `tetragamma` are the first three polygamma
//! functions (psi, psi', psi''). They are evaluated with the standard
//! recurrence-plus-asymptotic-series approach: the argument is shifted up to a
//! region where the Bernoulli asymptotic series converges quickly, then the
//! recurrence correction is added back. This matches `scipy.special.digamma`
//! and `scipy.special.polygamma(1|2, x)` to roughly 1e-13 for the positive
//! arguments limma evaluates (residual degrees of freedom halved, and the
//! Newton iterates inside `trigamma_inverse`).
//!
//! Only strictly positive arguments are supported, which is all limma needs.

/// Argument threshold above which the asymptotic series is accurate enough.
const ASYMPTOTIC_THRESHOLD: f64 = 10.0;

/// Digamma psi(x) = d/dx ln(Gamma(x)), for x > 0.
pub(crate) fn digamma(mut x: f64) -> f64 {
    let mut correction = 0.0;
    while x < ASYMPTOTIC_THRESHOLD {
        correction -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    correction + x.ln()
        - 0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0))))
}

/// Trigamma psi'(x) = d/dx digamma(x), for x > 0.
pub(crate) fn trigamma(mut x: f64) -> f64 {
    let mut correction = 0.0;
    while x < ASYMPTOTIC_THRESHOLD {
        correction += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    correction
        + inv
        + inv2
            * (0.5
                + inv
                    * (1.0 / 6.0
                        + inv2
                            * (-1.0 / 30.0
                                + inv2
                                    * (1.0 / 42.0
                                        + inv2
                                            * (-1.0 / 30.0
                                                + inv2
                                                    * (5.0 / 66.0
                                                        + inv2
                                                            * (-691.0 / 2730.0
                                                                + inv2
                                                                    * (7.0 / 6.0
                                                                        - inv2 * 3617.0
                                                                            / 510.0))))))))
}

/// Tetragamma psi''(x) = d/dx trigamma(x), for x > 0.
pub(crate) fn tetragamma(mut x: f64) -> f64 {
    let mut correction = 0.0;
    while x < ASYMPTOTIC_THRESHOLD {
        correction -= 2.0 / (x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    correction
        - inv2
            * (1.0
                + inv
                + inv2
                    * (0.5
                        + inv2
                            * (-1.0 / 6.0
                                + inv2
                                    * (1.0 / 6.0
                                        + inv2
                                            * (-3.0 / 10.0
                                                + inv2
                                                    * (5.0 / 6.0
                                                        + inv2
                                                            * (-691.0 / 210.0
                                                                + inv2
                                                                    * (35.0 / 2.0
                                                                        - inv2 * 3617.0
                                                                            / 30.0))))))))
}
