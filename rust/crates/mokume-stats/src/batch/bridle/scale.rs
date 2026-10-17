//! Per-dataset scale from anchor samples (opt-in post-fit step).
//!
//! BRIDLE's offsets are additive, so a dataset whose log-ratios are compressed
//! (TMT ratio compression) keeps a smaller spread than the reference after
//! correction. For a dataset sharing at least `min_anchors` anchor samples with
//! the reference, the slope is
//!
//! ```text
//! b_s = median_g sd(v[s, anchors, g]) / sd(v[ref, anchors, g])
//! ```
//!
//! over genes observed on >= [`MIN_GENE_ANCHORS`] shared anchors with a
//! reference SD > [`MIN_REF_SD`] and a Pearson r > [`MIN_CORR`] between the two
//! copies, and every value of the dataset becomes
//! `v' = refmean_g + (v - studymean_g) / b_s` (means over the shared anchors).
//! Port of `scale_fix()` in the Cell Line Collection `fixrun.py` (CCLE
//! b ~ 0.69, NCI-60 ~ 1.08).
//!
//! Genes without an anchor mean on one side are scaled around the dataset's
//! own mean instead of being dropped (the prototype lost these values): the
//! centre is the dataset's anchor mean when only the reference mean is missing,
//! else the dataset's mean over all its profiles. The gene keeps the location
//! BRIDLE's offset gave it (there is no anchor information to move it) and gets
//! the same slope as the rest of the dataset, so within-dataset spreads stay
//! comparable across genes. Every finite value stays finite.

use std::collections::HashMap;

use rayon::prelude::*;

use super::features::nan_median;
use super::BridleData;

/// Minimum shared anchors on which a gene must be observed in the dataset.
pub const MIN_GENE_ANCHORS: usize = 10;
/// Minimum reference SD over the shared anchors.
pub const MIN_REF_SD: f64 = 0.3;
/// Minimum Pearson r between the dataset's and the reference's copies.
pub const MIN_CORR: f64 = 0.5;

/// Per-dataset outcome of [`anchor_scale`].
#[derive(Debug, Clone)]
pub struct AnchorScale {
    pub name: String,
    /// Anchor samples shared with the reference.
    pub n_shared: usize,
    /// Genes that passed the filters (the median's support).
    pub n_genes: usize,
    /// Slope `b_s` (1 when not applied).
    pub b: f64,
    pub applied: bool,
}

/// Sample SD (`ddof = 1`) of the finite values; `NaN` with fewer than 2.
fn sd(values: &[f64]) -> f64 {
    let v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Mean of the finite values; `NaN` when there is none.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (s, c) = values
        .filter(|x| x.is_finite())
        .fold((0.0, 0_usize), |(s, c), x| (s + x, c + 1));
    if c == 0 {
        f64::NAN
    } else {
        s / c as f64
    }
}

/// Pearson r over the pairs where both values are finite.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (*x, *y))
        .unzip();
    if x.len() < 2 {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (p, q) in x.iter().zip(&y) {
        sxy += (p - mx) * (q - my);
        sxx += (p - mx).powi(2);
        syy += (q - my).powi(2);
    }
    sxy / (sxx * syy).sqrt()
}

/// Rescale each dataset's values (row-major `profiles x genes`, the rows of
/// `data`) onto the reference's spread; see the module doc. Datasets with
/// fewer than `min_anchors` shared anchors, and the reference, keep `b = 1`
/// and are left untouched. Returns one entry per dataset, sorted by name.
pub fn anchor_scale(
    data: &BridleData,
    values: &mut [f64],
    reference: &str,
    min_anchors: usize,
) -> Vec<AnchorScale> {
    let g_n = data.genes.len();
    let n = data.n_profiles();
    let ref_row: HashMap<&str, usize> = (0..n)
        .filter(|&i| data.datasets[i] == reference)
        .map(|i| (data.lines[i].as_str(), i))
        .collect();
    let mut names: Vec<&str> = data.datasets.iter().map(String::as_str).collect();
    names.sort_unstable();
    names.dedup();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let rows: Vec<usize> = (0..n).filter(|&i| data.datasets[i] == name).collect();
        // (dataset row, reference row) of each shared anchor
        let pairs: Vec<(usize, usize)> = if name == reference {
            Vec::new()
        } else {
            rows.iter()
                .filter_map(|&i| ref_row.get(data.lines[i].as_str()).map(|&j| (i, j)))
                .collect()
        };
        let mut rep = AnchorScale {
            name: name.to_owned(),
            n_shared: pairs.len(),
            n_genes: 0,
            b: 1.0,
            applied: false,
        };
        if name == reference {
            out.push(rep);
            continue;
        }
        if pairs.len() < min_anchors {
            tracing::info!(
                "BRIDLE anchor scale: {name}: {} shared anchor samples < {min_anchors}, b = 1",
                pairs.len()
            );
            out.push(rep);
            continue;
        }
        let vals: &[f64] = values;
        let copies = |g: usize| -> (Vec<f64>, Vec<f64>) {
            pairs
                .iter()
                .map(|&(i, j)| (vals[i * g_n + g], vals[j * g_n + g]))
                .unzip()
        };
        let ratios: Vec<f64> = (0..g_n)
            .into_par_iter()
            .filter_map(|g| {
                let (a, r) = copies(g);
                let n_a = a.iter().filter(|x| x.is_finite()).count();
                let sd_r = sd(&r);
                // `>` is false for NaN, so undefined SDs / correlations fail
                let keep =
                    n_a >= MIN_GENE_ANCHORS && sd_r > MIN_REF_SD && pearson(&a, &r) > MIN_CORR;
                keep.then(|| sd(&a) / sd_r)
            })
            .collect();
        let b = nan_median(ratios.iter().copied());
        rep.n_genes = ratios.iter().filter(|x| x.is_finite()).count();
        if b.is_nan() || b <= 0.0 {
            tracing::warn!(
                "BRIDLE anchor scale: {name}: no gene passed the filters \
                 ({} shared anchor samples), b = 1",
                pairs.len()
            );
            out.push(rep);
            continue;
        }
        // per gene: (target centre, source centre)
        let centres: Vec<(f64, f64)> = (0..g_n)
            .into_par_iter()
            .map(|g| {
                let (a, r) = copies(g);
                let am = mean(a.into_iter());
                let rm = mean(r.into_iter());
                if am.is_finite() && rm.is_finite() {
                    (rm, am)
                } else {
                    let own = if am.is_finite() {
                        am
                    } else {
                        mean(rows.iter().map(|&i| vals[i * g_n + g]))
                    };
                    (own, own)
                }
            })
            .collect();
        for &i in &rows {
            for (v, &(to, from)) in values[i * g_n..(i + 1) * g_n].iter_mut().zip(&centres) {
                if v.is_finite() {
                    *v = to + (*v - from) / b;
                }
            }
        }
        tracing::info!(
            "BRIDLE anchor scale: {name}: b = {b:.3} from {} genes on {} shared anchor samples",
            rep.n_genes,
            pairs.len()
        );
        rep.b = b;
        rep.applied = true;
        out.push(rep);
    }
    out
}
