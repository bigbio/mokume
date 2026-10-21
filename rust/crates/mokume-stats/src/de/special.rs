//! Special functions used by the limma differential-expression port.

mod polygamma;
pub(crate) use polygamma::{digamma, tetragamma, trigamma};

/// `1 / sqrt(2 * pi)`, the standard-normal pdf normalising constant.
#[cfg(test)]
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// `sqrt(2)`, used to map the normal cdf onto `erfc`.
const SQRT2: f64 = std::f64::consts::SQRT_2;

/// Standard-normal probability density `phi(x)`. Matches `scipy.stats.norm.pdf`.
#[cfg(test)]
fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard-normal log density `ln phi(x)`. Matches `scipy.stats.norm.logpdf`.
pub(crate) fn norm_logpdf(x: f64) -> f64 {
    -0.5 * x * x - 0.918_938_533_204_672_8
}

/// Standard-normal cdf `Phi(x) = 0.5 * erfc(-x / sqrt(2))`. Matches
/// `scipy.stats.norm.cdf`.
pub(crate) fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT2)
}

/// Standard-normal survival `Phi(-x) = 0.5 * erfc(x / sqrt(2))`. Matches
/// `scipy.stats.norm.sf`.
pub(crate) fn norm_sf(x: f64) -> f64 {
    0.5 * libm::erfc(x / SQRT2)
}

/// Standard-normal log-cdf `ln Phi(x)`, matching `scipy.stats.norm.logcdf`
/// (SciPy's `log_ndtr`). Accurate deep in the left tail where `ln(cdf)` would
/// underflow to `-inf`: for `x <= -14` an asymptotic series in `1/x^2` is used,
/// for `-14 < x <= 6` the direct `ln(0.5 * erfc(-x / sqrt(2)))`, and for `x > 6`
/// the numerically stable `log1p(-sf(x))`.
pub(crate) fn norm_logcdf(x: f64) -> f64 {
    if x > 6.0 {
        // cdf is ~1; ln(1 - sf) via log1p avoids cancellation.
        return libm::log1p(-norm_sf(x));
    }
    if x > -14.0 {
        return (0.5 * libm::erfc(-x / SQRT2)).ln();
    }
    log_ndtr_asymptotic(x)
}

/// Standard-normal log-survival `ln Phi(-x)`, matching `scipy.stats.norm.logsf`.
/// By symmetry `logsf(x) = logcdf(-x)`.
pub(crate) fn norm_logsf(x: f64) -> f64 {
    norm_logcdf(-x)
}

/// Asymptotic expansion of `ln Phi(x)` for `x <= -14` (SciPy's `log_ndtr` tail
/// branch). `Phi(x) ~ phi(x)/(-x) * (1 - 1/x^2 + 3/x^4 - 15/x^6 + ...)`, so the
/// log is `ln phi(x) - ln(-x) + ln(series)`.
fn log_ndtr_asymptotic(x: f64) -> f64 {
    // SciPy sums the alternating series term_k = (-1)^k (2k-1)!! / x^(2k) until a
    // term stops shrinking the partial sum. Eight terms are ample for x <= -14.
    let inv2 = 1.0 / (x * x);
    let mut term = 1.0_f64;
    let mut series = 1.0_f64;
    let mut last = 0.0_f64;
    for k in 1..=8 {
        let factor = (2 * k - 1) as f64;
        term *= -factor * inv2;
        let next = series + term;
        if next == last {
            break;
        }
        last = series;
        series = next;
    }
    norm_logpdf(x) - (-x).ln() + series.ln()
}

/// Log density of a (location-scale) Student-t `ln f(x; df, loc, scale)`,
/// matching `scipy.stats.t.logpdf(x, df, loc, scale)`. Closed form:
/// `lgamma((df+1)/2) - lgamma(df/2) - 0.5 ln(df pi) - ln(scale)
///  - (df+1)/2 * ln(1 + z^2/df)` with `z = (x - loc) / scale`.
pub(crate) fn t_logpdf(x: f64, df: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    libm::lgamma((df + 1.0) / 2.0)
        - libm::lgamma(df / 2.0)
        - 0.5 * (df * std::f64::consts::PI).ln()
        - scale.ln()
        - (df + 1.0) / 2.0 * libm::log1p(z * z / df)
}

/// Log density of the F-distribution `ln f(x; d1, d2)`, matching
/// `scipy.stats.f.logpdf`. Built from the closed form
/// `ln f = (d1/2) ln(d1/d2) + (d1/2 - 1) ln x - ((d1+d2)/2) ln(1 + d1 x / d2)
/// - ln B(d1/2, d2/2)` for `x > 0`; returns `-inf` for `x <= 0`.
pub(crate) fn f_logpdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return f64::NEG_INFINITY;
    }
    let a = d1 / 2.0;
    let b = d2 / 2.0;
    let ln_beta = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);
    a * (d1 / d2).ln() + (a - 1.0) * x.ln() - (a + b) * libm::log1p(d1 * x / d2) - ln_beta
}

/// F-distribution cdf `P(F <= x)`, matching `scipy.stats.f.cdf`. Uses the
/// regularized incomplete beta `I_{d1 x / (d1 x + d2)}(d1/2, d2/2)`.
///
/// proDA's variance prior only needs [`f_logpdf`]; `f_cdf`/`f_sf` are the
/// remaining F-distribution deliverables, kept (with golden tests) but compiled
/// only under `cfg(test)` since no production path calls them.
#[cfg(test)]
pub(crate) fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let y = d1 * x / (d1 * x + d2);
    regularized_incomplete_beta(d1 / 2.0, d2 / 2.0, y)
}

/// F-distribution survival `P(F > x)`, matching `scipy.stats.f.sf`. Computed as
/// `I_{d2 / (d1 x + d2)}(d2/2, d1/2)` to keep accuracy in the upper tail. See
/// [`f_cdf`] for why this is `cfg(test)`.
#[cfg(test)]
pub(crate) fn f_sf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let y = d2 / (d1 * x + d2);
    regularized_incomplete_beta(d2 / 2.0, d1 / 2.0, y)
}

/// Regularized upper incomplete gamma `Q(a, x) = Gamma(a, x) / Gamma(a)`,
/// matching `scipy.special.gammaincc(a, x)`. This is the chi-squared survival
/// kernel the ensemble Fisher combiner needs.
///
/// Uses the Numerical Recipes split (`gammp`/`gammq`): the power series for the
/// lower regularized `P(a, x)` when `x < a + 1`, and the Lentz continued
/// fraction for `Q(a, x)` when `x >= a + 1`, which is the regime each method
/// converges fastest in. Matches SciPy to ~1e-12 across the chi2 inputs the
/// Fisher combiner produces (small `x` near the mode through deep upper tail).
pub(crate) fn gammaincc(a: f64, x: f64) -> f64 {
    if !a.is_finite() || !x.is_finite() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        // Q(a, 0) = 1 (the whole mass is above 0).
        return 1.0;
    }
    if x < a + 1.0 {
        // Series gives the lower P(a, x); Q = 1 - P.
        1.0 - gamma_lower_series(a, x)
    } else {
        // Continued fraction gives Q(a, x) directly.
        gamma_upper_continued_fraction(a, x)
    }
}

/// Power series for the regularized lower incomplete gamma `P(a, x)` (Numerical
/// Recipes `gser`), valid and rapidly convergent for `x < a + 1`.
fn gamma_lower_series(a: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 1000;
    const EPSILON: f64 = 1e-16;

    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut term = sum;
    for _ in 0..MAX_ITERATIONS {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * EPSILON {
            break;
        }
    }
    // P(a, x) = sum * exp(-x + a ln x - lgamma(a)).
    sum * (-x + a * x.ln() - libm::lgamma(a)).exp()
}

/// Lentz continued fraction for the regularized upper incomplete gamma
/// `Q(a, x)` (Numerical Recipes `gcf`), valid and rapidly convergent for
/// `x >= a + 1`.
fn gamma_upper_continued_fraction(a: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 1000;
    const EPSILON: f64 = 1e-16;
    const TINY: f64 = 1e-300;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITERATIONS {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
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
    // Q(a, x) = h * exp(-x + a ln x - lgamma(a)).
    h * (-x + a * x.ln() - libm::lgamma(a)).exp()
}

/// Chi-squared survival `P(X > x)` with `df` degrees of freedom, matching
/// `scipy.stats.chi2.sf(x, df)`. This is the regularized upper incomplete gamma
/// `Q(df/2, x/2)` (chi2 is `Gamma(shape=df/2, scale=2)`). Used by the ensemble
/// Fisher combiner, whose statistic `-2 sum ln p` is chi-squared on `2k` df.
pub(crate) fn chi2_sf(x: f64, df: f64) -> f64 {
    if !x.is_finite() || !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    gammaincc(df / 2.0, x / 2.0)
}

/// Regularized incomplete beta `I_x(a, b)`, the same Lentz continued-fraction
/// (Numerical Recipes `betai`/`betacf`) scheme as `student_t.rs`. Only the
/// `cfg(test)` F cdf/sf use it, so it is gated to match.
#[cfg(test)]
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

/// Lentz evaluation of the continued fraction for the incomplete beta.
#[cfg(test)]
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
    use super::{
        chi2_sf, digamma, f_cdf, f_logpdf, f_sf, norm_cdf, norm_logcdf, norm_logpdf, norm_logsf,
        norm_pdf, norm_sf, t_logpdf, tetragamma, trigamma,
    };

    fn assert_close(actual: f64, expected: f64) {
        let tol = 1e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn digamma_matches_scipy() {
        assert_close(digamma(0.5), -1.9635100260214235);
        assert_close(digamma(1.0), -0.5772156649015329);
        assert_close(digamma(2.0), 0.4227843350984671);
        assert_close(digamma(5.0), 1.5061176684318003);
    }

    #[test]
    fn trigamma_matches_scipy() {
        assert_close(trigamma(0.5), 4.93480220054468);
        assert_close(trigamma(1.0), 1.6449340668482266);
        assert_close(trigamma(2.0), 0.6449340668482266);
        assert_close(trigamma(5.0), 0.22132295573711533);
    }

    #[test]
    fn tetragamma_matches_scipy() {
        assert_close(tetragamma(0.5), -16.828796644234316);
        assert_close(tetragamma(1.0), -2.404113806319188);
        assert_close(tetragamma(2.0), -0.404_113_806_319_188_6);
    }

    // Looser comparator for tail values where 1e-12 relative is the practical
    // floor of the erfc/lgamma chain.
    fn assert_close_tol(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol * expected.abs().max(1.0),
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }

    // scipy.stats.norm.{pdf,cdf,sf,logpdf,logcdf,logsf}. Oracle values captured
    // verbatim from `conda run -n Bigbio python`.
    #[test]
    fn norm_pdf_cdf_sf_match_scipy() {
        // (x, pdf, cdf, sf)
        let cases = [
            (
                -3.0,
                0.0044318484119380075,
                0.0013498980316300933,
                0.9986501019683699,
            ),
            (
                -1.5,
                0.12951759566589174,
                0.06680720126885807,
                0.9331927987311419,
            ),
            (
                -0.3,
                0.38138781546052414,
                0.3820885778110474,
                0.6179114221889526,
            ),
            (0.0, 0.3989422804014327, 0.5, 0.5),
            (
                0.7,
                0.31225393336676127,
                0.758036347776927,
                0.24196365222307303,
            ),
            (
                2.4,
                0.0223945302948429,
                0.9918024640754038,
                0.008197535924596131,
            ),
            (
                5.0,
                1.4867195147342979e-06,
                0.9999997133484281,
                2.866515718791933e-07,
            ),
        ];
        for (x, pdf, cdf, sf) in cases {
            assert_close(norm_pdf(x), pdf);
            assert_close(norm_cdf(x), cdf);
            assert_close(norm_sf(x), sf);
        }
    }

    #[test]
    fn norm_logpdf_logcdf_logsf_match_scipy() {
        // (x, logpdf, logcdf, logsf)
        let cases = [
            (
                -3.0,
                -5.418938533204672,
                -6.60772622151035,
                -0.0013508099647481925,
            ),
            (
                -1.5,
                -2.0439385332046727,
                -2.7059444008238898,
                -0.06914345561223399,
            ),
            (
                -0.3,
                -0.9639385332046727,
                -0.9621028181688505,
                -0.4814101615884813,
            ),
            // norm.logcdf(0)=norm.logsf(0)=ln(0.5)=-ln 2 (std constant avoids
            // clippy::approx_constant flagging the bare literal).
            (
                0.0,
                -0.9189385332046727,
                -std::f64::consts::LN_2,
                -std::f64::consts::LN_2,
            ),
            (
                0.7,
                -1.1639385332046726,
                -0.2770239422771313,
                -1.4189677615315315,
            ),
            (
                2.4,
                -3.7989385332046726,
                -0.008231320482313337,
                -4.8039216668706715,
            ),
            (
                5.0,
                -13.418938533204672,
                -2.8665161296376294e-07,
                -15.064998393988727,
            ),
        ];
        for (x, logpdf, logcdf, logsf) in cases {
            assert_close(norm_logpdf(x), logpdf);
            assert_close_tol(norm_logcdf(x), logcdf, 1e-11);
            assert_close_tol(norm_logsf(x), logsf, 1e-11);
        }
    }

    // Deep-tail logcdf where `ln(cdf)` underflows to -inf: scipy's log_ndtr
    // asymptotic branch. Oracle from scipy.stats.norm.logcdf.
    #[test]
    fn norm_logcdf_deep_tail_matches_scipy() {
        let cases = [
            (-50.0, -1254.8313611394199),
            (-20.0, -203.9171553710973),
            (-14.5, -108.7227881543205),
            (-14.0, -101.56303440744996),
            (-13.0, -87.98971997102255),
            (6.5, -4.016000583939729e-11),
            (20.0, -2.7536241186061556e-89),
        ];
        for (x, logcdf) in cases {
            assert_close_tol(norm_logcdf(x), logcdf, 1e-10);
        }
    }

    // scipy.stats.t.logpdf(x, df, loc, scale). Oracle captured verbatim.
    #[test]
    fn t_logpdf_matches_scipy() {
        let cases = [
            (11.5, 3.0, 11.0, 0.8366600265340756, -1.047507344507524),
            (0.0, 3.0, 0.0, 1.0, -1.0008888496235098),
            (2.0, 3.0, 0.0, 1.0, -2.695484570397917),
        ];
        for (x, df, loc, scale, expected) in cases {
            assert_close_tol(t_logpdf(x, df, loc, scale), expected, 1e-12);
        }
    }

    // scipy.stats.chi2.sf(x, df). Oracle captured verbatim from
    // `conda run -n Bigbio python` (see scratchpad ensemble_oracle.py). Covers
    // the mode region, the deep upper tail (df=4 at x=50 -> 3.6e-10; df=20 at
    // x=100 -> 1.3e-12), and the near-zero lower argument (x=1e-6). This is the
    // Fisher combiner kernel: its statistic `-2 sum ln p` is chi-squared on 2k
    // df, so it lives mostly in this small-to-moderate x range.
    #[test]
    fn chi2_sf_matches_scipy() {
        let cases = [
            (1.0, 2.0, 0.6065306597126334),
            (5.0, 2.0, 0.0820849986238988),
            (10.0, 4.0, 0.04042768199451279),
            (3.5, 6.0, 0.7439696953972181),
            (0.5, 1.0, 0.47950012218695337),
            (20.0, 8.0, 0.010336050675925726),
            (15.3, 10.0, 0.1215011696305226),
            (2.0, 2.0, 0.36787944117144245),
            (50.0, 4.0, 3.610865404890647e-10),
            (100.0, 20.0, 1.2596084591660847e-12),
            (1e-6, 2.0, 0.999999500000125),
        ];
        for (x, df, sf) in cases {
            assert_close_tol(chi2_sf(x, df), sf, 1e-11);
        }
        // Degenerate guards: non-positive x -> survival 1, invalid df -> NaN.
        assert_close(chi2_sf(0.0, 2.0), 1.0);
        assert_close(chi2_sf(-1.0, 2.0), 1.0);
        assert!(chi2_sf(1.0, 0.0).is_nan());
    }

    // scipy.stats.f.{logpdf,sf,cdf}. Oracle captured verbatim.
    #[test]
    fn f_dist_matches_scipy() {
        // (x, d1, d2, logpdf, sf, cdf)
        let cases = [
            (
                0.5,
                3.0,
                4.0,
                -0.5709289178901953,
                0.7021977290675252,
                0.29780227093247486,
            ),
            (
                1.0,
                2.0,
                5.0,
                -1.1776528281742449,
                0.43120115037169215,
                0.5687988496283078,
            ),
            (2.5, 4.0, 10.0, -2.3671236141316143, 0.109375, 0.890625),
            (
                0.1,
                1.0,
                1.0,
                -0.0887475191567022,
                0.8050177709578635,
                0.19498222904213666,
            ),
            (
                5.0,
                8.0,
                3.0,
                -3.605765563264878,
                0.10649010422154723,
                0.8935098957784527,
            ),
        ];
        for (x, d1, d2, logpdf, sf, cdf) in cases {
            assert_close_tol(f_logpdf(x, d1, d2), logpdf, 1e-11);
            assert_close_tol(f_sf(x, d1, d2), sf, 1e-11);
            assert_close_tol(f_cdf(x, d1, d2), cdf, 1e-11);
        }
    }
}
