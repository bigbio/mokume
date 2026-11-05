//! Per-cell quantification precision from a protein's ion residuals.
//!
//! Each ion's log2 differences to the quantified protein profile are centred on
//! their median, which removes the ion's response factor. The residual sum of
//! squares of a protein, on `observations - ions - (samples - 1)` degrees of
//! freedom, is shrunk towards the dataset's median residual variance with
//! [`SHRINK_DF`] prior degrees of freedom. A cell's precision (1/SE^2) is its
//! observed ion count over that variance, so a value supported by more and more
//! consistent ions weighs more in a precision-weighted test.

use std::collections::HashMap;

use mokume_core::SampleId;

use crate::median;

/// Prior degrees of freedom that shrink a protein's residual variance towards
/// the dataset median.
const SHRINK_DF: f64 = 4.0;

/// Proteins need this many residual degrees of freedom to enter the median.
const MIN_PRIOR_DOF: f64 = 3.0;

/// Residual variance used when no protein reaches [`MIN_PRIOR_DOF`].
const FALLBACK_VARIANCE: f64 = 0.04;

/// One protein's ion residuals around its quantified profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileResiduals {
    rss: f64,
    dof: f64,
    /// Observed ions of each quantified sample.
    ions: Vec<(SampleId, usize)>,
}

/// Residuals of one protein: each entry of `ions` holds one ion's
/// `(sample, log2 intensity)` observations, and `profile` the protein's log2
/// value per quantified sample. Sums follow the given order.
pub fn profile_residuals(
    ions: &[Vec<(SampleId, f64)>],
    profile: &[(SampleId, f64)],
) -> ProfileResiduals {
    let levels = profile
        .iter()
        .filter(|(_, value)| value.is_finite())
        .copied()
        .collect::<HashMap<_, _>>();
    let mut rss = 0.0;
    let mut observations = 0;
    let mut fitted_ions = 0;
    let mut per_sample = HashMap::<SampleId, usize>::new();
    for ion in ions {
        let differences = ion
            .iter()
            .filter(|(_, value)| value.is_finite())
            .filter_map(|(sample, value)| Some((*sample, value - levels.get(sample)?)))
            .collect::<Vec<_>>();
        if differences.len() < 2 {
            continue;
        }
        let Some(offset) = median(&mut differences.iter().map(|(_, d)| *d).collect()) else {
            continue;
        };
        rss += differences
            .iter()
            .map(|(_, d)| (d - offset).powi(2))
            .sum::<f64>();
        observations += differences.len();
        fitted_ions += 1;
        for (sample, _) in &differences {
            *per_sample.entry(*sample).or_default() += 1;
        }
    }
    let free = observations as f64 - fitted_ions as f64 - (levels.len() as f64 - 1.0);
    let mut ions = per_sample.into_iter().collect::<Vec<_>>();
    ions.sort_by_key(|(sample, _)| sample.get());
    ProfileResiduals {
        rss,
        dof: free.max(0.0),
        ions,
    }
}

/// Precision of every quantified cell, keyed by `(protein, sample)`.
pub fn cell_precisions<K: Copy + Eq + std::hash::Hash>(
    residuals: &[(K, ProfileResiduals)],
) -> HashMap<(K, SampleId), f64> {
    let mut variances = residuals
        .iter()
        .filter(|(_, fit)| fit.dof >= MIN_PRIOR_DOF)
        .map(|(_, fit)| fit.rss / fit.dof)
        .collect::<Vec<_>>();
    let prior = median(&mut variances).unwrap_or(FALLBACK_VARIANCE);
    residuals
        .iter()
        .flat_map(|(protein, fit)| {
            let variance = (fit.rss + SHRINK_DF * prior) / (fit.dof + SHRINK_DF);
            fit.ions
                .iter()
                .map(move |&(sample, count)| ((*protein, sample), count as f64 / variance))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(index: u32) -> SampleId {
        SampleId::new(index)
    }

    #[test]
    fn residuals_remove_each_ion_offset() {
        // Ion 1 tracks the profile with a constant offset; ion 2 deviates by
        // +0.1 / -0.1 around its median offset in two of three samples.
        let profile = [(sample(0), 10.0), (sample(1), 11.0), (sample(2), 12.0)];
        let ions = [
            vec![(sample(0), 8.0), (sample(1), 9.0), (sample(2), 10.0)],
            vec![(sample(0), 9.1), (sample(1), 10.0), (sample(2), 10.9)],
        ];
        let fit = profile_residuals(&ions, &profile);
        assert!((fit.rss - 0.02).abs() < 1e-12);
        // 6 observations - 2 ions - (3 samples - 1).
        assert_eq!(fit.dof, 2.0);
        assert_eq!(
            fit.ions,
            vec![(sample(0), 2), (sample(1), 2), (sample(2), 2)]
        );
    }

    #[test]
    fn precision_counts_ions_over_the_shrunk_variance() {
        let fits = [
            (
                "a",
                ProfileResiduals {
                    rss: 0.3,
                    dof: 3.0,
                    ions: vec![(sample(0), 2)],
                },
            ),
            (
                "b",
                ProfileResiduals {
                    rss: 1.5,
                    dof: 5.0,
                    ions: vec![(sample(0), 4)],
                },
            ),
            (
                "c",
                ProfileResiduals {
                    rss: 0.0,
                    dof: 1.0,
                    ions: vec![(sample(1), 1)],
                },
            ),
        ];
        let precision = cell_precisions(&fits);
        // Prior = median(0.1, 0.3) = 0.2; "c" has too few dof to enter it.
        let expected_a = 2.0 / ((0.3 + 4.0 * 0.2) / 7.0);
        let expected_c = 1.0 / ((0.0 + 4.0 * 0.2) / 5.0);
        assert!((precision[&("a", sample(0))] - expected_a).abs() < 1e-12);
        assert!((precision[&("c", sample(1))] - expected_c).abs() < 1e-12);
        assert_eq!(precision.len(), 3);
    }

    #[test]
    fn missing_profile_samples_get_no_precision() {
        let ions = [vec![(sample(0), 8.0), (sample(1), 9.0)]];
        let fit = profile_residuals(&ions, &[(sample(0), 10.0), (sample(1), f64::NAN)]);
        assert!(fit.ions.is_empty());
        assert_eq!(fit.dof, 0.0);
    }

    #[test]
    fn singleton_ions_do_not_inflate_precision() {
        let profile = [(sample(0), 10.0), (sample(1), 11.0), (sample(2), 12.0)];
        let mut ions = vec![vec![(sample(0), 8.0), (sample(1), 9.0), (sample(2), 10.0)]];
        let baseline = profile_residuals(&ions, &profile);
        ions.push(vec![(sample(0), 7.0)]);
        assert_eq!(profile_residuals(&ions, &profile), baseline);
    }
}
