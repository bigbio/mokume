//! LimROTS 1.2.8 `Optimizing`, `calOverlaps`, and pooled `empPvals` semantics.

use super::super::rots::build_ssq_grid;
use super::fit::Fit;
use rayon::prelude::*;

pub(super) struct Optimum {
    pub a1: f64,
    pub a2: f64,
    pub k: usize,
    pub reproducibility: f64,
    pub z: f64,
}

pub(super) fn optimize(boots: &[Fit], null: &[Fit], k_values: &[usize]) -> Option<Optimum> {
    let mut parameters = build_ssq_grid()
        .into_iter()
        .map(|a1| (a1, 1.0))
        .collect::<Vec<_>>();
    parameters.push((1.0, 0.0));
    let scores = parameters
        .par_iter()
        .map(|&(a1, a2)| score_parameters(boots, null, k_values, a1, a2))
        .collect::<Vec<_>>();
    let mut best: Option<Optimum> = None;
    // R's matrix which(..., arr.ind=TRUE) returns column-major matches.
    for (ki, &k) in k_values.iter().enumerate() {
        for (pi, &(a1, a2)) in parameters.iter().enumerate() {
            let (z, reproducibility) = scores[pi][ki];
            if z.is_finite() && best.as_ref().is_none_or(|b| z > b.z) {
                best = Some(Optimum {
                    a1,
                    a2,
                    k,
                    reproducibility,
                    z,
                });
            }
        }
    }
    best
}

fn score_parameters(
    boots: &[Fit],
    null: &[Fit],
    k_values: &[usize],
    a1: f64,
    a2: f64,
) -> Vec<(f64, f64)> {
    let niter = boots.len() / 2;
    let d = boots
        .iter()
        .map(|fit| fit.statistics(a1, a2))
        .collect::<Vec<_>>();
    let p = null
        .iter()
        .map(|fit| fit.statistics(a1, a2))
        .collect::<Vec<_>>();
    let real = (0..niter)
        .map(|b| overlaps(&d[b], &d[b + niter], k_values))
        .collect::<Vec<_>>();
    let perm = (0..niter)
        .map(|b| overlaps(&p[b], &p[b + niter], k_values))
        .collect::<Vec<_>>();
    k_values
        .iter()
        .enumerate()
        .map(|(ki, _)| {
            let real = real.iter().map(|r| r[ki]).collect::<Vec<_>>();
            let perm = perm.iter().map(|r| r[ki]).collect::<Vec<_>>();
            let mean = real.iter().sum::<f64>() / niter as f64;
            let null_mean = perm.iter().sum::<f64>() / niter as f64;
            let sd =
                (real.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (niter - 1) as f64).sqrt();
            ((mean - null_mean) / sd, mean)
        })
        .collect()
}

/// Sort by both statistics, then count against the second vector's kth value.
/// Using two top-k index sets would handle ties differently from calOverlaps.
fn overlaps(first: &[f64], second: &[f64], k_values: &[usize]) -> Vec<f64> {
    let mut order = (0..first.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        second_order(first[b], first[a]).then(second_order(second[b], second[a]))
    });
    let mut sorted = second.to_vec();
    sorted.sort_by(|a, b| second_order(*b, *a));
    k_values
        .iter()
        .map(|&k| {
            let cutoff = sorted[k - 1];
            order[..k].iter().filter(|&&i| second[i] >= cutoff).count() as f64 / k as f64
        })
        .collect()
}

fn second_order(a: f64, b: f64) -> std::cmp::Ordering {
    // Upstream order(..., decreasing=TRUE) puts missing values last.
    match (a.is_nan(), b.is_nan()) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.total_cmp(&b),
    }
}

/// qvalue::empPvals(pool=TRUE) puts observed values before equal null values.
/// Therefore count strictly greater null statistics and apply a 1/m0 floor.
pub(super) fn empirical_pvalues(observed: &[f64], null: &[Vec<f64>]) -> Vec<f64> {
    let mut pooled = null.iter().flatten().copied().collect::<Vec<_>>();
    pooled.sort_by(f64::total_cmp);
    let m0 = pooled.len() as f64;
    observed
        .iter()
        .map(|&value| {
            let count = pooled.len() - pooled.partition_point(|&v| v <= value);
            (count.max(1) as f64) / m0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{empirical_pvalues, overlaps};

    #[test]
    fn ties_use_official_threshold_overlap() {
        assert_eq!(
            overlaps(&[3., 2., 1., 0.], &[2., 2., 3., 0.], &[2]),
            vec![1.0]
        );
    }

    #[test]
    fn pooled_pvalues_exclude_equal_null_values() {
        // qvalue::empPvals(c(2,1,2,5),matrix(c(0,1,2,2,3,4,5,6),4),pool=TRUE)
        assert_eq!(
            empirical_pvalues(
                &[2., 1., 2., 5.],
                &[vec![0., 1., 2., 2.], vec![3., 4., 5., 6.]]
            ),
            vec![0.5, 0.75, 0.5, 0.125]
        );
    }
}
