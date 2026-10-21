//! Held-out weight learning and the official ordered lambda search.
use super::{bh_adjust, invalid, lp, weighted_p, Result, SplitMix64};

#[derive(Clone, Copy)]
pub(super) struct Point {
    pub p: f64,
    pub group: usize,
}

pub(super) struct FoldConfig {
    pub alpha: f64,
    pub n_bins: usize,
    pub n_folds: usize,
    pub inner_folds: usize,
    pub inner_splits: usize,
}

pub(super) struct FoldFit {
    pub weights: Vec<f64>,
    pub labels: Vec<usize>,
    pub lambdas: Vec<f64>,
}
impl FoldFit {
    pub fn uniform(n: usize) -> Self {
        Self {
            weights: vec![1.0; n],
            labels: vec![0; n],
            lambdas: vec![0.0],
        }
    }
}

pub(super) fn fit(
    points: &[Point],
    labels: Option<&[usize]>,
    lambdas: &[f64],
    config: &FoldConfig,
    rng: &mut SplitMix64,
) -> Result<FoldFit> {
    let labels = labels.map_or_else(
        || {
            (0..points.len())
                .map(|_| rng.next_below(config.n_folds))
                .collect()
        },
        <[usize]>::to_vec,
    );
    let mut weights = vec![f64::NAN; points.len()];
    let mut chosen = Vec::with_capacity(config.n_folds);
    for fold in 0..config.n_folds {
        let training: Vec<Point> = points
            .iter()
            .zip(&labels)
            .filter_map(|(&p, &f)| (f != fold).then_some(p))
            .collect();
        let mut held_counts = vec![0.0; config.n_bins];
        for (p, &f) in points.iter().zip(&labels) {
            if f == fold {
                held_counts[p.group] += 1.0;
            }
        }
        if held_counts.iter().sum::<f64>() == 0.0 {
            return Err(invalid(format!("IHW fold {fold} contains no hypotheses")));
        }
        let lambda = choose_lambda(&training, lambdas, config, rng)?;
        let groups = split(&training, config.n_bins);
        let learned = lp::learn(&groups, &held_counts, config.alpha, lambda)?;
        for (i, (p, &f)) in points.iter().zip(&labels).enumerate() {
            if f == fold {
                weights[i] = learned[p.group];
            }
        }
        chosen.push(lambda);
    }
    Ok(FoldFit {
        weights,
        labels,
        lambdas: chosen,
    })
}

fn choose_lambda(
    points: &[Point],
    lambdas: &[f64],
    config: &FoldConfig,
    rng: &mut SplitMix64,
) -> Result<f64> {
    if lambdas.len() == 1 {
        return Ok(lambdas[0]);
    }
    let inner = FoldConfig {
        n_folds: config.inner_folds,
        ..*config
    };
    let mut best = lambdas[0];
    let mut best_rejections = None;
    for &lambda in lambdas {
        let mut rejections = 0;
        for _ in 0..config.inner_splits {
            let output = fit(points, None, &[lambda], &inner, rng)?;
            let weighted: Vec<f64> = points
                .iter()
                .zip(output.weights)
                .map(|(p, w)| weighted_p(p.p, w))
                .collect();
            rejections += bh_adjust(&weighted)
                .iter()
                .filter(|&&p| p <= config.alpha)
                .count();
        }
        if best_rejections.is_none_or(|count| rejections > count) {
            best = lambda;
            best_rejections = Some(rejections);
        }
    }
    Ok(best)
}

fn split(points: &[Point], bins: usize) -> Vec<Vec<f64>> {
    let mut groups = vec![Vec::new(); bins];
    for point in points {
        groups[point.group].push(point.p);
    }
    groups
}
