//! Data-driven log2-fold-change thresholds.
//!
//! This ports `mokume.analysis.effect_size_gate`: `mixture` fits a deterministic
//! two-component Gaussian mixture to median-centred absolute fold changes, and
//! `null_quantile` estimates a robust high quantile of the central null.

use std::f64::consts::TAU;

const MIN_VALUES: usize = 50;
const REGULARIZATION: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeGateMethod {
    Mixture,
    NullQuantile,
}

/// Estimate a data-driven absolute log2-fold-change threshold.
///
/// Non-finite values are ignored. `fallback` is returned only when fewer than
/// 50 finite values remain or the selected estimator is degenerate.
pub fn estimate_effect_size_gate(
    log2_fold_changes: &[f64],
    method: EffectSizeGateMethod,
    fallback: f64,
) -> f64 {
    let values = log2_fold_changes
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.len() < MIN_VALUES {
        return fallback;
    }
    let estimate = match method {
        EffectSizeGateMethod::Mixture => mixture_gate(&values),
        EffectSizeGateMethod::NullQuantile => Some(null_quantile_gate(&values)),
    };
    match estimate {
        Some(value) if value.is_finite() && value > 0.0 => value,
        _ => fallback,
    }
}

fn null_quantile_gate(values: &[f64]) -> f64 {
    let center = quantile(values, 0.5);
    let deviations = values
        .iter()
        .map(|value| (value - center).abs())
        .collect::<Vec<_>>();
    let mad = quantile(&deviations, 0.5) * 1.4826;
    if mad <= 0.0 {
        return quantile(&deviations, 0.95);
    }
    let mut central = values
        .iter()
        .copied()
        .filter(|value| (value - center).abs() <= 2.0 * mad)
        .collect::<Vec<_>>();
    if central.len() < 20 {
        central = values.to_vec();
    }
    let central_deviations = central
        .iter()
        .map(|value| (value - center).abs())
        .collect::<Vec<_>>();
    quantile(&central_deviations, 0.95)
}

fn mixture_gate(values: &[f64]) -> Option<f64> {
    let center = quantile(values, 0.5);
    let absolute = values
        .iter()
        .map(|value| (value - center).abs())
        .collect::<Vec<_>>();
    let mut sorted = absolute.clone();
    sorted.sort_by(f64::total_cmp);
    if sorted.first() == sorted.last() {
        return None;
    }

    let labels = kmeans_labels(&absolute)?;
    let mut model = GaussianMixture::from_labels(&absolute, &labels)?;
    model.fit(&absolute)?;
    let (null, signal) = if model.means[0] <= model.means[1] {
        (0, 1)
    } else {
        (1, 0)
    };
    if model.means[signal] <= model.means[null] {
        return None;
    }

    let maximum = quantile(&absolute, 0.999);
    for index in 0..400 {
        let point = maximum * index as f64 / 399.0;
        if model.log_probability(point, signal) >= model.log_probability(point, null) {
            return Some(point);
        }
    }
    None
}

/// sklearn KMeans(n_clusters=2, n_init=1, random_state=0), specialized to 1D.
fn kmeans_labels(values: &[f64]) -> Option<Vec<usize>> {
    let mut random = Mt19937::new(0);
    let mut centers = initial_kmeans_centers(values, &mut random);
    let tolerance = variance(values) * 1e-4;
    let mut previous_labels = vec![usize::MAX; values.len()];
    let mut strict_convergence = false;
    for _ in 0..300 {
        let labels = assign_labels(values, centers);
        let next = kmeans_centers(values, &labels)?;
        let shift = (next[0] - centers[0]).powi(2) + (next[1] - centers[1]).powi(2);
        centers = next;
        if labels == previous_labels {
            previous_labels = labels;
            strict_convergence = true;
            break;
        }
        previous_labels = labels;
        if shift <= tolerance {
            break;
        }
    }
    if strict_convergence {
        Some(previous_labels)
    } else {
        Some(assign_labels(values, centers))
    }
}

fn initial_kmeans_centers(values: &[f64], random: &mut Mt19937) -> [f64; 2] {
    let first = (random.random_f64() * values.len() as f64).floor() as usize;
    let mut centers = [values[first], 0.0];
    let closest = values
        .iter()
        .map(|value| (value - centers[0]).powi(2))
        .collect::<Vec<_>>();
    let potential = closest.iter().sum::<f64>();
    let cumulative = cumulative_sum(&closest);
    let mut best_potential = f64::INFINITY;
    for _ in 0..2 {
        let target = random.random_f64() * potential;
        let candidate = cumulative
            .partition_point(|value| *value < target)
            .min(values.len() - 1);
        let candidate_potential = values
            .iter()
            .zip(&closest)
            .map(|(value, distance)| distance.min((value - values[candidate]).powi(2)))
            .sum::<f64>();
        if candidate_potential < best_potential {
            best_potential = candidate_potential;
            centers[1] = values[candidate];
        }
    }
    centers
}

fn kmeans_centers(values: &[f64], labels: &[usize]) -> Option<[f64; 2]> {
    let mut counts = [0usize; 2];
    let mut sums = [0.0; 2];
    for (value, label) in values.iter().zip(labels) {
        counts[*label] += 1;
        sums[*label] += value;
    }
    if counts.contains(&0) {
        return None;
    }
    Some([sums[0] / counts[0] as f64, sums[1] / counts[1] as f64])
}

fn assign_labels(values: &[f64], centers: [f64; 2]) -> Vec<usize> {
    values
        .iter()
        .map(|value| usize::from((value - centers[1]).powi(2) < (value - centers[0]).powi(2)))
        .collect()
}

fn cumulative_sum(values: &[f64]) -> Vec<f64> {
    let mut total = 0.0;
    values
        .iter()
        .map(|value| {
            total += value;
            total
        })
        .collect()
}

struct GaussianMixture {
    weights: [f64; 2],
    means: [f64; 2],
    variances: [f64; 2],
}

struct Expectation {
    counts: [f64; 2],
    weighted_sums: [f64; 2],
    responsibilities: Vec<[f64; 2]>,
    lower_bound: f64,
}

impl GaussianMixture {
    fn from_labels(values: &[f64], labels: &[usize]) -> Option<Self> {
        let mut groups = [Vec::new(), Vec::new()];
        for (value, label) in values.iter().zip(labels) {
            groups[*label].push(*value);
        }
        if groups.iter().any(Vec::is_empty) {
            return None;
        }
        let n = values.len() as f64;
        Some(Self {
            weights: [groups[0].len() as f64 / n, groups[1].len() as f64 / n],
            means: [mean(&groups[0]), mean(&groups[1])],
            variances: [
                variance(&groups[0]) + REGULARIZATION,
                variance(&groups[1]) + REGULARIZATION,
            ],
        })
    }

    fn fit(&mut self, values: &[f64]) -> Option<()> {
        let mut previous_lower_bound = f64::NEG_INFINITY;
        for _ in 0..100 {
            let expectation = self.expectation(values)?;
            self.maximize(values, &expectation)?;
            if (expectation.lower_bound - previous_lower_bound).abs() < 1e-3 {
                break;
            }
            previous_lower_bound = expectation.lower_bound;
        }
        Some(())
    }

    fn expectation(&self, values: &[f64]) -> Option<Expectation> {
        let mut counts = [0.0; 2];
        let mut weighted_sums = [0.0; 2];
        let mut responsibilities = Vec::with_capacity(values.len());
        let mut lower_bound = 0.0;
        for value in values {
            let log_probabilities = [
                self.log_probability(*value, 0),
                self.log_probability(*value, 1),
            ];
            let normalizer = log_sum_exp(log_probabilities[0], log_probabilities[1]);
            if !normalizer.is_finite() {
                return None;
            }
            let responsibility = [
                (log_probabilities[0] - normalizer).exp(),
                (log_probabilities[1] - normalizer).exp(),
            ];
            for component in 0..2 {
                counts[component] += responsibility[component];
                weighted_sums[component] += responsibility[component] * value;
            }
            responsibilities.push(responsibility);
            lower_bound += normalizer;
        }
        Some(Expectation {
            counts,
            weighted_sums,
            responsibilities,
            lower_bound: lower_bound / values.len() as f64,
        })
    }

    fn maximize(&mut self, values: &[f64], expectation: &Expectation) -> Option<()> {
        for component in 0..2 {
            if expectation.counts[component] <= 0.0 {
                return None;
            }
            self.weights[component] = expectation.counts[component] / values.len() as f64;
            self.means[component] =
                expectation.weighted_sums[component] / expectation.counts[component];
        }
        let mut variance_sums = [0.0; 2];
        for (value, responsibility) in values.iter().zip(&expectation.responsibilities) {
            for component in 0..2 {
                let delta = value - self.means[component];
                variance_sums[component] += responsibility[component] * delta * delta;
            }
        }
        for (component, variance_sum) in variance_sums.into_iter().enumerate() {
            self.variances[component] =
                variance_sum / expectation.counts[component] + REGULARIZATION;
        }
        Some(())
    }

    fn log_probability(&self, value: f64, component: usize) -> f64 {
        self.weights[component].ln()
            - 0.5
                * (TAU.ln()
                    + self.variances[component].ln()
                    + (value - self.means[component]).powi(2) / self.variances[component])
    }
}

/// NumPy's legacy `RandomState` MT19937 stream, used by sklearn for
/// `GaussianMixture(random_state=0)` initialization.
struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for index in 1..624 {
            state[index] = 1_812_433_253u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        Self { state, index: 624 }
    }

    fn random_f64(&mut self) -> f64 {
        let high = (self.next_u32() >> 5) as u64;
        let low = (self.next_u32() >> 6) as u64;
        (high * 67_108_864 + low) as f64 / 9_007_199_254_740_992.0
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn twist(&mut self) {
        for index in 0..624 {
            let combined =
                (self.state[index] & 0x8000_0000) | (self.state[(index + 1) % 624] & 0x7fff_ffff);
            let mut next = self.state[(index + 397) % 624] ^ (combined >> 1);
            if combined & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[index] = next;
        }
        self.index = 0;
    }
}

fn log_sum_exp(left: f64, right: f64) -> f64 {
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    let center = mean(values);
    values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

fn quantile(values: &[f64], probability: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{EffectSizeGateMethod, Mt19937, estimate_effect_size_gate};

    fn deterministic_mixture() -> Vec<f64> {
        let mut values = (0..90)
            .map(|index| -0.02 + 0.04 * index as f64 / 89.0)
            .collect::<Vec<_>>();
        values.extend((0..10).map(|index| 0.25 + 0.10 * index as f64 / 9.0));
        values
    }

    #[test]
    fn estimators_match_python_oracles() {
        let values = deterministic_mixture();
        let mixture = estimate_effect_size_gate(&values, EffectSizeGateMethod::Mixture, 0.5);
        let quantile = estimate_effect_size_gate(&values, EffectSizeGateMethod::NullQuantile, 0.5);
        assert!((mixture - 0.059_947_478_246_177_23).abs() < 1e-12);
        assert!((quantile - 0.020_247_191_011_235_954).abs() < 1e-12);
    }

    #[test]
    fn mixture_matches_sklearn_kmeans_initialization_oracle() {
        let mut values = (0..50)
            .map(|index| -0.05 + 0.10 * index as f64 / 49.0)
            .collect::<Vec<_>>();
        values.extend((0..20).map(|index| 0.05 + 0.05 * index as f64 / 19.0));

        let gate = estimate_effect_size_gate(&values, EffectSizeGateMethod::Mixture, 0.5);
        assert!((gate - 0.037_018_315_660_256_98).abs() < 1e-12);
    }

    #[test]
    fn degenerate_inputs_use_the_requested_fallback() {
        assert_eq!(
            estimate_effect_size_gate(&[0.0; 49], EffectSizeGateMethod::Mixture, 0.25),
            0.25
        );
        assert_eq!(
            estimate_effect_size_gate(&[0.0; 100], EffectSizeGateMethod::Mixture, 0.25),
            0.25
        );
    }

    #[test]
    fn non_finite_padding_does_not_change_the_gate() {
        let values = deterministic_mixture();
        let clean = estimate_effect_size_gate(&values, EffectSizeGateMethod::Mixture, 0.5);
        let mut padded = values;
        padded.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        let padded = estimate_effect_size_gate(&padded, EffectSizeGateMethod::Mixture, 0.5);
        assert_eq!(clean, padded);
    }

    #[test]
    fn random_state_matches_numpy_legacy_mt19937() {
        let mut random = Mt19937::new(0);
        assert_eq!(random.random_f64(), 0.548_813_503_927_324_8);
        assert_eq!(random.random_f64(), 0.715_189_366_372_419_5);
        assert_eq!(random.random_f64(), 0.602_763_376_071_643_9);
    }
}
