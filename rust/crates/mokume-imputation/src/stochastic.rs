//! Reproducible random draws and standard-normal transforms for imputers.

pub(crate) struct Random {
    state: u64,
    spare: Option<f64>,
}

impl Random {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        // Midpoints of 2^52 bins lie strictly within (0, 1).
        (((z ^ (z >> 31)) >> 12) as f64 + 0.5) / 4503599627370496.0
    }

    pub(crate) fn normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
}

const NORMAL_A: [f64; 6] = [
    -39.69683028665376,
    220.9460984245205,
    -275.9285104469687,
    138.357_751_867_269,
    -30.66479806614716,
    2.506628277459239,
];
const NORMAL_B: [f64; 6] = [
    -54.47609879822406,
    161.5858368580409,
    -155.6989798598866,
    66.80131188771972,
    -13.28068155288572,
    1.0,
];
const NORMAL_C: [f64; 6] = [
    -0.007784894002430293,
    -0.3223964580411365,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
];
const NORMAL_D: [f64; 5] = [
    0.007784695709041462,
    0.3224671290700398,
    2.445134137142996,
    3.754408661907416,
    1.0,
];

/// Acklam rational approximation followed by one Halley refinement.
pub(crate) fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let mut x = if !(0.02425..=0.97575).contains(&p) {
        let q = (-2.0 * p.min(1.0 - p).ln()).sqrt();
        let tail = polynomial(&NORMAL_C, q) / polynomial(&NORMAL_D, q);
        if p < 0.5 {
            tail
        } else {
            -tail
        }
    } else {
        let q = p - 0.5;
        polynomial(&NORMAL_A, q * q) * q / polynomial(&NORMAL_B, q * q)
    };
    let error = if x > 0.0 {
        (1.0 - p) - normal_cdf(-x)
    } else {
        normal_cdf(x) - p
    };
    let correction = error * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x -= correction / (1.0 + x * correction / 2.0);
    x
}

fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .fold(0.0, |value, coefficient| value * x + coefficient)
}
