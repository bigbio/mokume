//! Post-fit, label-blind plex rescale (ComBat-style scale `delta`).
//!
//! BRIDLE's offsets are additive, so a batch whose values are compressed or
//! inflated (e.g. TMT ratio compression per plex) keeps its spread after the
//! fit. Port of the benchmark's `posthoc_delta_all.py` (arm
//! `A3gnc_phdelta_all`). Batches are dataset x plex where the fit has plexes,
//! else datasets. Per (batch `b`, gene `g`), over ALL the batch's observed
//! values (line identity is never used):
//!
//! ```text
//! s2[b,g]  within-batch variance (ddof 1), n[b,g] values, mean_b[b,g]
//! sig2[g]  pooled within-batch variance over batches with n >= 2
//! l^[b,g]  = log(s2 / sig2) + 1/(n-1) - 1/(N_g - B_g)   (chi^2 log bias),
//!            sampling variance 2/(n-1); estimable when n >= 5, s2, sig2 > 1e-8
//! l[b,g]   EB posterior: normal prior per batch across genes, mean mu_b
//!          (itself shrunk to 0 with sd 0.5), variance t2_b, 5 moment rounds
//! delta    = clip(exp(l / 2), 0.25, 4)
//! v'       = mean_b + (v - mean_b) / delta
//! ```
//!
//! Genes that are not estimable in a batch take `l = mu_b`. A batch with < 50
//! estimable genes takes `mu_b` = the median of its dataset's other plexes
//! (in batch-name order, including plexes filled before it), or 0 for a batch
//! without plexes; `lambda = 1` (no partial rescale). The reference is
//! rescaled like any other batch (pooled ComBat target).
//!
//! Benchmark (5-fold leave-lines-out, 21 cell-line datasets): median |error|
//! 0.786 vs 0.799 for BERT, better on cis-RNA, EMT and proliferation guards;
//! 0.007 less accurate than the same fit without the rescale but better on 7
//! biology guards. The rescale was chosen after a fold-1 diagnostic (post-hoc
//! selection).

use rayon::prelude::*;

/// Prior sd of a batch's mean log variance ratio (shrunk to 0).
const PRIOR_SD: f64 = 0.5;
/// Minimum values of a gene in a batch for a gene-level estimate.
const MIN_VALUES: usize = 5;
/// Minimum estimable genes for a batch-level estimate.
const MIN_GENES: usize = 50;
/// Clamp of `delta`.
const DELTA_RANGE: (f64, f64) = (0.25, 4.0);
const EB_ITERS: usize = 5;
const T2_FLOOR: f64 = 1e-4;
const VAR_FLOOR: f64 = 1e-8;

/// Batches of [`plex_rescale`]; `batch` is indexed by profile.
pub(super) struct RescaleBatches<'a> {
    pub batch: &'a [usize],
    /// `dataset` or `dataset|plex`, per batch.
    pub names: &'a [String],
    /// Dataset of each batch.
    pub dataset: &'a [usize],
    /// The batch is a plex of its dataset.
    pub plexed: &'a [bool],
}

/// Per-batch outcome of [`plex_rescale`].
#[derive(Debug, Clone)]
pub struct PlexRescale {
    /// `dataset` or `dataset|plex`.
    pub batch: String,
    pub n_profiles: usize,
    /// Genes with a gene-level estimate (0 for a fallback batch).
    pub n_genes: usize,
    /// Batch mean log variance ratio `mu_b` (prior of every gene).
    pub mu: f64,
    /// `mu_b` taken from sibling plexes (or 0): < 50 estimable genes.
    pub fallback: bool,
    /// Median `delta` over the batch's observed genes.
    pub median_delta: f64,
}

/// `delta` outside this range counts as extreme in [`DeltaSummary`].
pub const DELTA_EXTREME: (f64, f64) = (0.67, 1.5);

/// Per-dataset summary of the rescale `delta` over all its (batch, gene)
/// cells with >= 1 observed value.
#[derive(Debug, Clone, Copy)]
pub struct DeltaSummary {
    pub median: f64,
    /// Fraction of cells with `delta` outside [`DELTA_EXTREME`].
    pub extreme_frac: f64,
}

/// Per (batch, gene) moments.
#[derive(Clone, Copy)]
struct Cell {
    n: usize,
    mean: f64,
    var: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let h = v.len() / 2;
    if v.len() % 2 == 1 {
        v[h]
    } else {
        0.5 * (v[h - 1] + v[h])
    }
}

/// EB prior mean `mu` and variance `t2` of one batch's log ratios `x` with
/// sampling variances `v`.
fn eb_prior(x: &[f64], v: &[f64]) -> (f64, f64) {
    let k = x.len() as f64;
    let xm = x.iter().sum::<f64>() / k;
    let xvar = x.iter().map(|a| (a - xm).powi(2)).sum::<f64>() / k;
    let mut t2 = (xvar - v.iter().sum::<f64>() / k).max(T2_FLOOR);
    let mut mu = 0.0;
    for _ in 0..EB_ITERS {
        let (mut swx, mut sw) = (0.0, 0.0);
        for (a, b) in x.iter().zip(v) {
            let w = 1.0 / (b + t2);
            swx += w * a;
            sw += w;
        }
        mu = swx / (sw + 1.0 / (PRIOR_SD * PRIOR_SD));
        t2 = (x
            .iter()
            .zip(v)
            .map(|(a, b)| (a - mu).powi(2) - b)
            .sum::<f64>()
            / k)
            .max(T2_FLOOR);
    }
    (mu, t2)
}

/// Rescale `values` (row-major `profiles x genes`) in place; see the module
/// doc. Returns one entry per batch with >= 1 profile, in batch-name order,
/// and the `delta` summary of each dataset (indexed by `RescaleBatches::dataset`).
pub(super) fn plex_rescale(
    values: &mut [f64],
    g_n: usize,
    b: &RescaleBatches,
) -> (Vec<PlexRescale>, Vec<DeltaSummary>) {
    let n_b = b.names.len();
    let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); n_b];
    for (i, &bb) in b.batch.iter().enumerate() {
        rows_of[bb].push(i);
    }
    let vals: &[f64] = values;
    // gene-major moments, batch-minor
    let cells: Vec<Vec<Cell>> = (0..g_n)
        .into_par_iter()
        .map(|g| {
            rows_of
                .iter()
                .map(|rows| {
                    let x: Vec<f64> = rows
                        .iter()
                        .map(|&i| vals[i * g_n + g])
                        .filter(|v| v.is_finite())
                        .collect();
                    let n = x.len();
                    let mean = x.iter().sum::<f64>() / n.max(1) as f64;
                    let var = if n >= 2 {
                        x.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n - 1) as f64
                    } else {
                        f64::NAN
                    };
                    Cell { n, mean, var }
                })
                .collect()
        })
        .collect();
    // gene-level log ratios: (lh, sampling variance) where estimable
    let est: Vec<Vec<Option<(f64, f64)>>> = cells
        .par_iter()
        .map(|cg| {
            let (mut ss, mut nn, mut nb) = (0.0, 0_usize, 0_usize);
            for c in cg.iter().filter(|c| c.n >= 2) {
                ss += c.var * (c.n - 1) as f64;
                nn += c.n;
                nb += 1;
            }
            let df = nn as f64 - nb as f64;
            let sig2 = if nb > 0 { ss / df } else { f64::NAN };
            cg.iter()
                .map(|c| {
                    let ok = c.n >= MIN_VALUES && c.var > VAR_FLOOR && sig2 > VAR_FLOOR;
                    ok.then(|| {
                        let m = (c.n - 1) as f64;
                        ((c.var / sig2).ln() + 1.0 / m - 1.0 / df, 2.0 / m)
                    })
                })
                .collect()
        })
        .collect();
    // batch priors in name order; < MIN_GENES -> sibling plexes or 0
    let mut order: Vec<usize> = (0..n_b).filter(|&k| !rows_of[k].is_empty()).collect();
    order.sort_by(|&x, &y| b.names[x].cmp(&b.names[y]));
    let mut prior: Vec<Option<(f64, f64)>> = vec![None; n_b]; // (mu, t2); t2 NaN = fallback
    for &k in &order {
        let (x, v): (Vec<f64>, Vec<f64>) = (0..g_n).filter_map(|g| est[g][k]).unzip();
        if x.len() >= MIN_GENES {
            prior[k] = Some(eb_prior(&x, &v));
        }
    }
    let mut report = Vec::with_capacity(order.len());
    for &k in &order {
        let mut fallback = false;
        if prior[k].is_none() {
            fallback = true;
            let sib: Vec<f64> = if b.plexed[k] {
                (0..n_b)
                    .filter(|&j| j != k && b.plexed[j] && b.dataset[j] == b.dataset[k])
                    .filter_map(|j| prior[j].map(|p| p.0))
                    .collect()
            } else {
                Vec::new()
            };
            let mu = if sib.is_empty() { 0.0 } else { median(sib) };
            prior[k] = Some((mu, f64::NAN));
        }
        report.push(PlexRescale {
            batch: b.names[k].clone(),
            n_profiles: rows_of[k].len(),
            n_genes: if fallback {
                0
            } else {
                (0..g_n).filter(|&g| est[g][k].is_some()).count()
            },
            mu: prior[k].map_or(0.0, |p| p.0),
            fallback,
            median_delta: f64::NAN,
        });
    }
    // delta per (gene, batch) and the rescale
    let delta: Vec<Vec<f64>> = (0..g_n)
        .into_par_iter()
        .map(|g| {
            (0..n_b)
                .map(|k| {
                    let Some((mu, t2)) = prior[k] else {
                        return 1.0;
                    };
                    let l = match est[g][k] {
                        Some((x, v)) if t2.is_finite() => (x / v + mu / t2) / (1.0 / v + 1.0 / t2),
                        _ => mu,
                    };
                    (l / 2.0).exp().clamp(DELTA_RANGE.0, DELTA_RANGE.1)
                })
                .collect()
        })
        .collect();
    for (k, rows) in rows_of.iter().enumerate() {
        for &i in rows {
            for g in 0..g_n {
                let v = &mut values[i * g_n + g];
                if v.is_finite() {
                    let c = cells[g][k];
                    *v = c.mean + (*v - c.mean) / delta[g][k];
                }
            }
        }
    }
    for (r, &k) in report.iter_mut().zip(&order) {
        r.median_delta = median(
            (0..g_n)
                .filter(|&g| cells[g][k].n > 0)
                .map(|g| delta[g][k])
                .collect(),
        );
        tracing::info!(
            "BRIDLE plex rescale: {}: {} profiles, {} genes estimated, mu = {:.3}{}, median delta = {:.3}",
            r.batch,
            r.n_profiles,
            r.n_genes,
            r.mu,
            if r.fallback { " (fallback)" } else { "" },
            r.median_delta
        );
    }
    let n_ds = b.dataset.iter().max().map_or(0, |m| m + 1);
    let mut per_ds: Vec<Vec<f64>> = vec![Vec::new(); n_ds];
    for (k, rows) in rows_of.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        per_ds[b.dataset[k]].extend((0..g_n).filter(|&g| cells[g][k].n > 0).map(|g| delta[g][k]));
    }
    let summary = per_ds
        .into_iter()
        .map(|d| {
            let extreme = d
                .iter()
                .filter(|&&x| x < DELTA_EXTREME.0 || x > DELTA_EXTREME.1)
                .count();
            let frac = if d.is_empty() {
                f64::NAN
            } else {
                extreme as f64 / d.len() as f64
            };
            DeltaSummary {
                median: median(d),
                extreme_frac: frac,
            }
        })
        .collect();
    (report, summary)
}

#[cfg(test)]
mod tests {
    use super::super::NumpyRandomState;
    use super::*;

    const G: usize = 80;

    /// Batches (name, dataset, plexed, profiles, spread); value = 20 + g/10 +
    /// spread * z, a few cells missing.
    fn collection(plan: &[(&str, usize, bool, usize, f64)]) -> (Vec<f64>, Vec<usize>) {
        let mut rs = NumpyRandomState::new(3);
        let mut values = Vec::new();
        let mut batch = Vec::new();
        for (k, &(_, _, _, n, spread)) in plan.iter().enumerate() {
            for _ in 0..n {
                batch.push(k);
                for g in 0..G {
                    let z = rs.next_gauss();
                    let miss = rs.next_f64() < 0.05;
                    values.push(if miss {
                        f64::NAN
                    } else {
                        20.0 + g as f64 / 10.0 + spread * z
                    });
                }
            }
        }
        (values, batch)
    }

    fn run(
        plan: &[(&str, usize, bool, usize, f64)],
        values: &mut [f64],
        batch: &[usize],
    ) -> Vec<PlexRescale> {
        run_with_summary(plan, values, batch).0
    }

    fn run_with_summary(
        plan: &[(&str, usize, bool, usize, f64)],
        values: &mut [f64],
        batch: &[usize],
    ) -> (Vec<PlexRescale>, Vec<DeltaSummary>) {
        let names: Vec<String> = plan.iter().map(|p| p.0.to_owned()).collect();
        let dataset: Vec<usize> = plan.iter().map(|p| p.1).collect();
        let plexed: Vec<bool> = plan.iter().map(|p| p.2).collect();
        plex_rescale(
            values,
            G,
            &RescaleBatches {
                batch,
                names: &names,
                dataset: &dataset,
                plexed: &plexed,
            },
        )
    }

    fn batch_sd(values: &[f64], batch: &[usize], k: usize) -> f64 {
        // mean over genes of the within-batch SD
        let mut tot = 0.0;
        for g in 0..G {
            let x: Vec<f64> = (0..batch.len())
                .filter(|&i| batch[i] == k)
                .map(|i| values[i * G + g])
                .filter(|v| v.is_finite())
                .collect();
            let m = x.iter().sum::<f64>() / x.len() as f64;
            tot += (x.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt();
        }
        tot / G as f64
    }

    #[test]
    fn inflated_plex_is_shrunk_onto_the_pooled_spread() {
        let plan = [
            ("A", 0, false, 12, 1.0),
            ("B|p1", 1, true, 10, 2.0),
            ("B|p2", 1, true, 10, 1.0),
        ];
        let (mut v, batch) = collection(&plan);
        let before = v.clone();
        let rep = run(&plan, &mut v, &batch);
        let names: Vec<&str> = rep.iter().map(|r| r.batch.as_str()).collect();
        assert_eq!(names, ["A", "B|p1", "B|p2"]);
        assert!(rep.iter().all(|r| !r.fallback && r.n_genes >= 50));
        // B|p1 is twice as spread: pooled sd ~1.4, so delta ~1.4 for B|p1
        // and ~0.7 for the others
        assert!(rep[1].median_delta > 1.3, "{:?}", rep[1]);
        assert!(rep[0].median_delta < 0.8 && rep[2].median_delta < 0.8);
        let ratio = batch_sd(&v, &batch, 1) / batch_sd(&v, &batch, 2);
        assert!((ratio - 1.0).abs() < 0.15, "sd ratio after {ratio}");
        assert!(batch_sd(&before, &batch, 1) / batch_sd(&before, &batch, 2) > 1.8);
        // missing stays missing; batch-gene means are unchanged
        for (x, y) in before.iter().zip(&v) {
            assert_eq!(x.is_finite(), y.is_finite());
        }
        for k in 0..3 {
            for g in 0..G {
                let m = |vals: &[f64]| {
                    let x: Vec<f64> = (0..batch.len())
                        .filter(|&i| batch[i] == k)
                        .map(|i| vals[i * G + g])
                        .filter(|v| v.is_finite())
                        .collect();
                    x.iter().sum::<f64>() / x.len() as f64
                };
                assert!((m(&before) - m(&v)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn rescale_is_blind_to_profile_order_within_a_batch() {
        let plan = [("A", 0, false, 12, 1.0), ("B", 1, false, 10, 1.6)];
        let (v0, batch) = collection(&plan);
        let mut a = v0.clone();
        run(&plan, &mut a, &batch);
        // reverse B's profiles (the rescale never sees line identity)
        let b_rows: Vec<usize> = (0..batch.len()).filter(|&i| batch[i] == 1).collect();
        let mut perm: Vec<usize> = (0..batch.len()).collect();
        for (x, y) in b_rows.iter().zip(b_rows.iter().rev()) {
            perm[*x] = *y;
        }
        let mut p: Vec<f64> = perm
            .iter()
            .flat_map(|&j| v0[j * G..(j + 1) * G].to_vec())
            .collect();
        run(&plan, &mut p, &batch);
        for (i, &j) in perm.iter().enumerate() {
            for g in 0..G {
                let (x, y) = (a[j * G + g], p[i * G + g]);
                assert!((x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn small_batches_fall_back_to_sibling_plexes_or_no_rescale() {
        // C|p3 (4 profiles < 5 values) takes the median of C|p1, C|p2; D (4
        // profiles, no plexes) gets mu = 0 -> delta = 1, values unchanged
        let plan = [
            ("A", 0, false, 12, 1.0),
            ("C|p1", 1, true, 10, 1.8),
            ("C|p2", 1, true, 10, 2.2),
            ("C|p3", 1, true, 4, 2.0),
            ("D", 2, false, 4, 3.0),
        ];
        let (mut v, batch) = collection(&plan);
        let before = v.clone();
        let rep = run(&plan, &mut v, &batch);
        let by = |nm: &str| rep.iter().find(|r| r.batch == nm).cloned();
        let (p1, p2, p3, d) = (by("C|p1"), by("C|p2"), by("C|p3"), by("D"));
        let (p1, p2, p3, d) = match (p1, p2, p3, d) {
            (Some(a), Some(b), Some(c), Some(e)) => (a, b, c, e),
            _ => panic!("missing batch report"),
        };
        assert!(p3.fallback && p3.n_genes == 0);
        assert!((p3.mu - 0.5 * (p1.mu + p2.mu)).abs() < 1e-12);
        assert!((p3.median_delta - (p3.mu / 2.0).exp().clamp(0.25, 4.0)).abs() < 1e-12);
        assert!(d.fallback && d.mu == 0.0 && d.median_delta == 1.0);
        for i in (0..batch.len()).filter(|&i| batch[i] == 4) {
            for g in 0..G {
                let (x, y) = (before[i * G + g], v[i * G + g]);
                assert!((x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn delta_summary_pools_the_plexes_of_a_dataset() {
        let plan = [
            ("A", 0, false, 12, 1.0),
            ("B|p1", 1, true, 10, 3.0),
            ("B|p2", 1, true, 10, 1.0),
        ];
        let (mut v, batch) = collection(&plan);
        let (rep, sum) = run_with_summary(&plan, &mut v, &batch);
        assert_eq!(sum.len(), 2);
        // pooled sd ~1.9: A and B|p2 delta ~0.5, B|p1 ~1.6 (all extreme)
        assert!((sum[0].median - rep[0].median_delta).abs() < 1e-12);
        assert!(sum[1].median > rep[2].median_delta && sum[1].median < rep[1].median_delta);
        assert!(sum[1].extreme_frac > 0.8, "{:?}", sum[1]);
        assert!(sum[0].extreme_frac > 0.8, "{:?}", sum[0]);
        assert!((0.0..=1.0).contains(&sum[0].extreme_frac));
    }
}
