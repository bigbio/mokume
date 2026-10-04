//! Per-dataset value report of a BRIDLE fit: what each dataset contributes to
//! the collection and how well it agrees with the rest, from a single fit (no
//! refits). Port of the cheap metrics of the 2026-10 B4 analysis
//! (`b4_desc.py`, sections 1, 2 and 4 of its spec).
//!
//! Units: a *profile* is a (dataset, anchor) pair, a *shared* anchor is one
//! measured by >= 2 datasets. `raw` is the input `y`, `v` the corrected output.
//!
//! Coverage (input only):
//! * `uniq_lines` (only source), `anchor_lines` (shared), `bridge_lines`
//!   (shared by exactly 2 datasets: single-source if this one is removed),
//!   `anchor_partners` (datasets sharing >= `min_shared_lines` anchors),
//!   `uniq_genes` (genes observed in no other dataset), `uniq_gene_cells`
//!   ((anchor, gene) cells observed in no other dataset).
//!
//! Fit diagnostics:
//! * `noise_var`: median over the dataset's observed genes of the fitted noise
//!   variance `sig2[s,g]`; `c_abs`: median |sample loading|; `offset_sd`: SD
//!   (ddof 1) of `A[s,g]` over observed genes; `delta_median` /
//!   `delta_extreme`: plex rescale summary.
//! * Cross-dataset disagreement on shared cells (`e = v_a - v_b`, same anchor
//!   and gene, different datasets): `agree_med` = median |e| over pairs that
//!   involve the dataset, `disagree_var` = median(e^2) / 0.4549 (robust
//!   variance), `excess_var` = the dataset's share `d_s` of that variance, by
//!   non-negative least squares over dataset pairs with >= `min_pair_cells`
//!   cells: `median(e^2) / 0.4549 = d_a + d_b`, weights sqrt(log1p(n)). There
//!   is no replicate noise floor here, so `d_s` includes the dataset's noise.
//!
//! Identity:
//! * `abund_rho`: per profile, Spearman over its observed genes of `raw` vs
//!   the gene's median level in the OTHER datasets (median over datasets of
//!   each dataset's median); fit-free.
//! * Against other datasets, on gene-centred `v` (minus the per-gene median
//!   over all profiles), pairwise-complete Pearson between profiles of
//!   different datasets (NaN below `min_overlap_genes` shared genes):
//!   `self_r` = mean r with the same anchor elsewhere, `self_rank` = 1 + number
//!   of other anchors' profiles correlating better than the best same-anchor
//!   profile, `best_line` / `best_r` the best-matching profile's anchor.
//! * Optional RNA (or any per-anchor reference): Pearson of the gene-centred
//!   profile with each reference anchor (reference gene-centred by its mean
//!   over anchors), `rna_r_self` and `rna_rank` (1 = own anchor best).
//!
//! Redundancy (closed form): per shared anchor and gene with >= 2 observed
//! sources, inverse-variance consensus `sum(w v) / sum(w)` with
//! `w = 1 / sig2[s,g]`; per source: `shift` = |consensus - consensus without
//! it|, `se_gain` = 1 - sqrt((W - w) / W) (= 1 - SE_with / SE_without),
//! `wshare` = w / W; profile values are medians over genes, dataset values
//! medians over its shared profiles. `redund_ge3` = fraction of its shared
//! profiles whose anchor has >= 3 other sources.
//!
//! `no_anchors_cannot_audit` is set for datasets without shared anchors: no
//! identity, disagreement or redundancy metric exists for them (e.g. an
//! enrichment artefact in an unanchored dataset is invisible here).
//!
//! Not implemented: leave-one-dataset-out refits (influence on the others),
//! planned as an opt-in `--influence`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rayon::prelude::*;

use super::{BridleData, BridleResult};

/// Median of chi^2(1) used by the prototype to turn median(e^2) into a variance.
const CHI2_MEDIAN: f64 = 0.4549;
const NNLS_SWEEPS: usize = 100_000;
const NNLS_TOL: f64 = 1e-15;

/// Thresholds of the value report (defaults = the B4 prototype).
#[derive(Debug, Clone)]
pub struct ValueParams {
    /// Shared anchors that make another dataset an anchor partner.
    pub min_shared_lines: usize,
    /// Shared genes for a profile-profile (or profile-reference) correlation.
    pub min_overlap_genes: usize,
    /// Shared cells for a dataset pair to enter the excess-variance fit.
    pub min_pair_cells: usize,
}

impl Default for ValueParams {
    fn default() -> Self {
        Self {
            min_shared_lines: 3,
            min_overlap_genes: 200,
            min_pair_cells: 200,
        }
    }
}

/// Per-anchor reference profiles (e.g. DepMap RNA), aligned to the genes of
/// the BRIDLE input.
#[derive(Debug, Clone)]
pub struct RnaReference {
    pub lines: Vec<String>,
    /// Row-major `lines x genes` (genes of `BridleData::genes`), `NaN` = missing.
    pub values: Vec<f64>,
}

/// Per-dataset value metrics (see the module doc).
#[derive(Debug, Clone)]
pub struct DatasetValue {
    pub name: String,
    pub n_lines: usize,
    pub n_cells: usize,
    pub n_genes: usize,
    pub genes_per_profile: f64,
    pub uniq_lines: usize,
    pub anchor_lines: usize,
    pub bridge_lines: usize,
    pub anchor_partners: usize,
    pub partners_any: usize,
    pub uniq_genes: usize,
    pub uniq_gene_cells: usize,
    pub noise_var: f64,
    pub c_abs: f64,
    pub offset_sd: f64,
    pub delta_median: f64,
    pub delta_extreme: f64,
    pub pair_cells: usize,
    pub agree_med: f64,
    pub disagree_var: f64,
    pub excess_var: f64,
    pub abund_rho: f64,
    pub abund_rho_min: f64,
    pub id_self_r: f64,
    pub id_rank: f64,
    pub id_top1: f64,
    pub id_best_is_self: f64,
    pub id_r_max_any: f64,
    pub id_r_med_any: f64,
    pub id_rna_self: f64,
    pub id_rna_rank: f64,
    pub id_rna_top1: f64,
    pub id_rna_top5: f64,
    pub marg_shift: f64,
    pub marg_se_gain: f64,
    pub wshare: f64,
    pub redund_ge3: f64,
    pub other_src_med: f64,
    pub no_anchors_cannot_audit: bool,
}

/// Per-profile metrics (see the module doc).
#[derive(Debug, Clone)]
pub struct ProfileValue {
    pub dataset: String,
    pub line: String,
    pub n_genes: usize,
    pub n_other_sources: usize,
    pub abund_rho: f64,
    pub self_r: f64,
    pub self_rank: f64,
    pub best_line: String,
    pub best_r: f64,
    pub med_r_any: f64,
    pub rna_r_self: f64,
    pub rna_rank: f64,
    pub marg_shift: f64,
    pub marg_se_gain: f64,
    pub wshare: f64,
}

#[derive(Debug, Clone)]
pub struct ValueReport {
    pub datasets: Vec<DatasetValue>,
    pub profiles: Vec<ProfileValue>,
}

/// The fitted quantities the report reads (indexed like [`BridleResult`]).
pub(super) struct FitView<'a> {
    pub corrected: &'a [f64],
    /// `datasets x genes`, dataset order = sorted names.
    pub noise_var: &'a [f64],
    pub sample_loading: &'a [f64],
    pub offsets: &'a [f64],
    /// Per dataset `(delta_median, delta_extreme)`.
    pub delta: Vec<(f64, f64)>,
}

/// Value report of a fit; `data` is the fit's input.
pub fn dataset_value(
    data: &BridleData,
    res: &BridleResult,
    rna: Option<&RnaReference>,
    params: &ValueParams,
) -> ValueReport {
    let view = FitView {
        corrected: &res.corrected,
        noise_var: &res.noise_var,
        sample_loading: &res.sample_loading,
        offsets: &res.offsets,
        delta: res
            .report
            .datasets
            .iter()
            .map(|d| (d.delta_median, d.delta_extreme))
            .collect(),
    };
    value_report(data, &view, rna, params)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
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

fn mean_finite(v: impl Iterator<Item = f64>) -> f64 {
    let (s, n) = v
        .filter(|x| x.is_finite())
        .fold((0.0, 0_usize), |(s, n), x| (s + x, n + 1));
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

fn frac(v: &[f64], pred: impl Fn(f64) -> bool) -> f64 {
    let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        f64::NAN
    } else {
        f.iter().filter(|&&x| pred(x)).count() as f64 / f.len() as f64
    }
}

/// Average ranks (ties share their mean rank), 1-based.
fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

/// Pearson correlation of complete pairs (`NaN` below `min_n` pairs).
fn pearson(x: &[f64], y: &[f64], min_n: usize) -> f64 {
    let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0_usize, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            n += 1;
            sx += a;
            sy += b;
            sxx += a * a;
            syy += b * b;
            sxy += a * b;
        }
    }
    if n < min_n.max(2) {
        return f64::NAN;
    }
    let nf = n as f64;
    let den = ((sxx - sx * sx / nf) * (syy - sy * sy / nf)).sqrt();
    if den > 0.0 {
        (sxy - sx * sy / nf) / den
    } else {
        f64::NAN
    }
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 3 {
        return f64::NAN;
    }
    pearson(&ranks(x), &ranks(y), 3)
}

/// Weighted non-negative least squares `min |W (X d - y)|^2, d >= 0` by
/// cyclic coordinate descent on the normal equations (`x` row-major, `k`
/// columns).
fn nnls(x: &[f64], y: &[f64], w: &[f64], k: usize) -> Vec<f64> {
    let m = y.len();
    let mut ata = vec![0.0; k * k];
    let mut atb = vec![0.0; k];
    for r in 0..m {
        let w2 = w[r] * w[r];
        for a in 0..k {
            let xa = x[r * k + a];
            if xa == 0.0 {
                continue;
            }
            atb[a] += w2 * xa * y[r];
            for b in 0..k {
                ata[a * k + b] += w2 * xa * x[r * k + b];
            }
        }
    }
    let mut d = vec![0.0; k];
    for _ in 0..NNLS_SWEEPS {
        let mut change: f64 = 0.0;
        for a in 0..k {
            if ata[a * k + a] <= 0.0 {
                continue;
            }
            let grad: f64 = (0..k).map(|b| ata[a * k + b] * d[b]).sum::<f64>() - atb[a];
            let new = (d[a] - grad / ata[a * k + a]).max(0.0);
            change = change.max((new - d[a]).abs());
            d[a] = new;
        }
        if change < NNLS_TOL {
            break;
        }
    }
    d
}

/// Core of [`dataset_value`] on explicit fitted quantities.
pub(super) fn value_report(
    data: &BridleData,
    fit: &FitView,
    rna: Option<&RnaReference>,
    params: &ValueParams,
) -> ValueReport {
    let n = data.n_profiles();
    let g_n = data.genes.len();
    let raw = &data.values;
    let v = fit.corrected;
    let names: Vec<String> = data
        .datasets
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let s_n = names.len();
    let sidx: HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let srow: Vec<usize> = data.datasets.iter().map(|d| sidx[d.as_str()]).collect();
    let mut line_rows: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        line_rows.entry(data.lines[i].as_str()).or_default().push(i);
    }
    // datasets per line (profiles are unique per (dataset, line))
    let nst = |i: usize| -> usize {
        line_rows[data.lines[i].as_str()]
            .iter()
            .map(|&j| srow[j])
            .collect::<BTreeSet<_>>()
            .len()
    };
    let nsrc: Vec<usize> = (0..n).map(nst).collect();
    let rows_of: Vec<Vec<usize>> = (0..s_n)
        .map(|s| (0..n).filter(|&i| srow[i] == s).collect())
        .collect();

    // ---------- coverage
    let mut gene_ds = vec![vec![false; s_n]; g_n];
    for i in 0..n {
        for g in 0..g_n {
            if raw[i * g_n + g].is_finite() {
                gene_ds[g][srow[i]] = true;
            }
        }
    }
    let gst: Vec<usize> = gene_ds
        .iter()
        .map(|d| d.iter().filter(|&&b| b).count())
        .collect();
    // (line, gene) -> number of datasets observing it, for each profile's cells
    let cell_st = |i: usize, g: usize| -> usize {
        line_rows[data.lines[i].as_str()]
            .iter()
            .filter(|&&j| raw[j * g_n + g].is_finite())
            .map(|&j| srow[j])
            .collect::<BTreeSet<_>>()
            .len()
    };

    // ---------- abundance-shape rho (raw, fit-free)
    let gmed: Vec<Vec<f64>> = (0..s_n)
        .map(|s| {
            (0..g_n)
                .map(|g| median(rows_of[s].iter().map(|&i| raw[i * g_n + g]).collect()))
                .collect()
        })
        .collect();
    let ref_wo: Vec<Vec<f64>> = (0..s_n)
        .map(|s| {
            (0..g_n)
                .map(|g| median((0..s_n).filter(|&t| t != s).map(|t| gmed[t][g]).collect()))
                .collect()
        })
        .collect();
    let abund_rho: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (a, b): (Vec<f64>, Vec<f64>) = (0..g_n)
                .filter(|&g| raw[i * g_n + g].is_finite() && ref_wo[srow[i]][g].is_finite())
                .map(|g| (raw[i * g_n + g], ref_wo[srow[i]][g]))
                .unzip();
            spearman(&a, &b)
        })
        .collect();

    // ---------- identity on gene-centred corrected values
    let gcen: Vec<f64> = (0..g_n)
        .map(|g| median((0..n).map(|i| v[i * g_n + g]).collect()))
        .collect();
    let w: Vec<f64> = (0..n * g_n).map(|c| v[c] - gcen[c % g_n]).collect();
    let row = |i: usize| &w[i * g_n..(i + 1) * g_n];
    struct Ident {
        self_r: f64,
        self_rank: f64,
        best_line: String,
        best_r: f64,
        med_r: f64,
    }
    let ident: Vec<Ident> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut best: Option<(f64, usize)> = None;
            let mut all = Vec::new();
            let mut same = Vec::new();
            let mut other = Vec::new();
            for j in 0..n {
                if srow[j] == srow[i] {
                    continue;
                }
                let r = pearson(row(i), row(j), params.min_overlap_genes);
                let is_same = data.lines[j] == data.lines[i];
                if is_same {
                    same.push(r);
                } else {
                    other.push(r);
                }
                if r.is_finite() {
                    all.push(r);
                    if best.is_none_or(|(b, _)| r > b) {
                        best = Some((r, j));
                    }
                }
            }
            let (best_r, best_line) = best.map_or((f64::NAN, String::new()), |(r, j)| {
                (r, data.lines[j].clone())
            });
            let rs = same
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f64::NAN, f64::max);
            let self_rank = if rs.is_finite() {
                (other.iter().filter(|&&r| r > rs).count() + 1) as f64
            } else {
                f64::NAN
            };
            Ident {
                self_r: mean_finite(same.into_iter()),
                self_rank,
                best_line,
                best_r,
                med_r: median(all),
            }
        })
        .collect();

    // ---------- optional reference (RNA) identity
    let rna_id: Vec<(f64, f64)> = match rna {
        Some(rf) => {
            let nl = rf.lines.len();
            let rcen: Vec<f64> = (0..g_n)
                .map(|g| mean_finite((0..nl).map(|l| rf.values[l * g_n + g])))
                .collect();
            let rv: Vec<f64> = (0..nl * g_n)
                .map(|c| rf.values[c] - rcen[c % g_n])
                .collect();
            let rpos: HashMap<&str, usize> = rf
                .lines
                .iter()
                .enumerate()
                .map(|(i, l)| (l.as_str(), i))
                .collect();
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let Some(&own) = rpos.get(data.lines[i].as_str()) else {
                        return (f64::NAN, f64::NAN);
                    };
                    let x = row(i);
                    let r: Vec<f64> = (0..nl)
                        .map(|l| pearson(x, &rv[l * g_n..(l + 1) * g_n], params.min_overlap_genes))
                        .collect();
                    let rs = r[own];
                    if !rs.is_finite() {
                        return (f64::NAN, f64::NAN);
                    }
                    (rs, (r.iter().filter(|&&a| a > rs).count() + 1) as f64)
                })
                .collect()
        }
        None => vec![(f64::NAN, f64::NAN); n],
    };

    // ---------- cross-dataset disagreement and redundancy on shared anchors
    let mut pair_e2: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
    let mut ds_abs: Vec<Vec<f64>> = vec![Vec::new(); s_n];
    let mut ds_e2: Vec<Vec<f64>> = vec![Vec::new(); s_n];
    let mut prof_shift: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut prof_gain: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut prof_wshare: Vec<Vec<f64>> = vec![Vec::new(); n];
    for rows in line_rows.values().filter(|r| r.len() >= 2) {
        for (x, &a) in rows.iter().enumerate() {
            for &b in &rows[x + 1..] {
                let (sa, sb) = (srow[a], srow[b]);
                let key = (sa.min(sb), sa.max(sb));
                for g in 0..g_n {
                    let e = v[a * g_n + g] - v[b * g_n + g];
                    if e.is_finite() {
                        pair_e2.entry(key).or_default().push(e * e);
                        for s in [sa, sb] {
                            ds_abs[s].push(e.abs());
                            ds_e2[s].push(e * e);
                        }
                    }
                }
            }
        }
        for g in 0..g_n {
            let src: Vec<(usize, f64, f64)> = rows
                .iter()
                .filter_map(|&i| {
                    let val = v[i * g_n + g];
                    let var = fit.noise_var[srow[i] * g_n + g];
                    (val.is_finite() && var.is_finite() && var > 0.0).then(|| (i, 1.0 / var, val))
                })
                .collect();
            if src.len() < 2 {
                continue;
            }
            let sw: f64 = src.iter().map(|s| s.1).sum();
            let swv: f64 = src.iter().map(|s| s.1 * s.2).sum();
            let cons = swv / sw;
            for &(i, wi, vi) in &src {
                let wo = (swv - wi * vi) / (sw - wi);
                prof_shift[i].push((cons - wo).abs());
                prof_gain[i].push(1.0 - ((sw - wi) / sw).sqrt());
                prof_wshare[i].push(wi / sw);
            }
        }
    }
    // excess variance: NNLS over dataset pairs
    let fitted: Vec<(&(usize, usize), f64, usize)> = pair_e2
        .iter()
        .filter(|(_, e)| e.len() >= params.min_pair_cells)
        .map(|(k, e)| (k, median(e.clone()) / CHI2_MEDIAN, e.len()))
        .collect();
    let in_fit: BTreeSet<usize> = fitted.iter().flat_map(|(k, _, _)| [k.0, k.1]).collect();
    let mut excess = vec![f64::NAN; s_n];
    if !fitted.is_empty() {
        let mut xm = vec![0.0; fitted.len() * s_n];
        for (r, (k, _, _)) in fitted.iter().enumerate() {
            xm[r * s_n + k.0] = 1.0;
            xm[r * s_n + k.1] = 1.0;
        }
        let y: Vec<f64> = fitted.iter().map(|f| f.1).collect();
        let wt: Vec<f64> = fitted.iter().map(|f| (f.2 as f64).ln_1p().sqrt()).collect();
        let d = nnls(&xm, &y, &wt, s_n);
        for &s in &in_fit {
            excess[s] = d[s];
        }
    }

    // ---------- per profile
    let profiles: Vec<ProfileValue> = (0..n)
        .map(|i| ProfileValue {
            dataset: data.datasets[i].clone(),
            line: data.lines[i].clone(),
            n_genes: (0..g_n).filter(|&g| raw[i * g_n + g].is_finite()).count(),
            n_other_sources: nsrc[i] - 1,
            abund_rho: abund_rho[i],
            self_r: ident[i].self_r,
            self_rank: ident[i].self_rank,
            best_line: ident[i].best_line.clone(),
            best_r: ident[i].best_r,
            med_r_any: ident[i].med_r,
            rna_r_self: rna_id[i].0,
            rna_rank: rna_id[i].1,
            marg_shift: median(prof_shift[i].clone()),
            marg_se_gain: median(prof_gain[i].clone()),
            wshare: median(prof_wshare[i].clone()),
        })
        .collect();

    // ---------- per dataset
    let datasets = (0..s_n)
        .map(|s| {
            let rows = &rows_of[s];
            let col = |f: &dyn Fn(&ProfileValue) -> f64| -> Vec<f64> {
                rows.iter().map(|&i| f(&profiles[i])).collect()
            };
            let mut partners: BTreeMap<usize, usize> = BTreeMap::new();
            for &i in rows {
                for &j in &line_rows[data.lines[i].as_str()] {
                    if srow[j] != s {
                        *partners.entry(srow[j]).or_insert(0) += 1;
                    }
                }
            }
            let obs_genes: Vec<usize> = (0..g_n).filter(|&g| gene_ds[g][s]).collect();
            let offs: Vec<f64> = obs_genes
                .iter()
                .map(|&g| fit.offsets[s * g_n + g])
                .collect();
            let offset_sd = if offs.len() >= 2 {
                let m = offs.iter().sum::<f64>() / offs.len() as f64;
                (offs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (offs.len() - 1) as f64).sqrt()
            } else {
                f64::NAN
            };
            let anchor_lines = rows.iter().filter(|&&i| nsrc[i] >= 2).count();
            // shared profiles with >= 1 consensus gene (prototype LS rows)
            let shared: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&i| !prof_shift[i].is_empty())
                .collect();
            let other_src: Vec<f64> = shared.iter().map(|&i| (nsrc[i] - 1) as f64).collect();
            let rank = col(&|p| p.self_rank);
            let rna_rank = col(&|p| p.rna_rank);
            let best_is_self: Vec<f64> = rows
                .iter()
                .filter(|&&i| nsrc[i] >= 2 && !profiles[i].best_line.is_empty())
                .map(|&i| f64::from(u8::from(profiles[i].best_line == profiles[i].line)))
                .collect();
            DatasetValue {
                name: names[s].clone(),
                n_lines: rows.len(),
                n_cells: rows.iter().map(|&i| profiles[i].n_genes).sum(),
                n_genes: obs_genes.len(),
                genes_per_profile: median(col(&|p| p.n_genes as f64)),
                uniq_lines: rows.iter().filter(|&&i| nsrc[i] == 1).count(),
                anchor_lines,
                bridge_lines: rows.iter().filter(|&&i| nsrc[i] == 2).count(),
                anchor_partners: partners
                    .values()
                    .filter(|&&c| c >= params.min_shared_lines)
                    .count(),
                partners_any: partners.len(),
                uniq_genes: obs_genes.iter().filter(|&&g| gst[g] == 1).count(),
                uniq_gene_cells: rows
                    .iter()
                    .map(|&i| {
                        (0..g_n)
                            .filter(|&g| raw[i * g_n + g].is_finite() && cell_st(i, g) == 1)
                            .count()
                    })
                    .sum(),
                noise_var: median(
                    obs_genes
                        .iter()
                        .map(|&g| fit.noise_var[s * g_n + g])
                        .collect(),
                ),
                c_abs: median(rows.iter().map(|&i| fit.sample_loading[i].abs()).collect()),
                offset_sd,
                delta_median: fit.delta.get(s).map_or(f64::NAN, |d| d.0),
                delta_extreme: fit.delta.get(s).map_or(f64::NAN, |d| d.1),
                pair_cells: ds_abs[s].len(),
                agree_med: median(ds_abs[s].clone()),
                disagree_var: median(ds_e2[s].clone()) / CHI2_MEDIAN,
                excess_var: excess[s],
                abund_rho: median(col(&|p| p.abund_rho)),
                abund_rho_min: col(&|p| p.abund_rho)
                    .into_iter()
                    .filter(|x| x.is_finite())
                    .fold(f64::NAN, f64::min),
                id_self_r: median(col(&|p| p.self_r)),
                id_rank: median(rank.clone()),
                id_top1: frac(&rank, |x| x == 1.0),
                id_best_is_self: if best_is_self.is_empty() {
                    f64::NAN
                } else {
                    best_is_self.iter().sum::<f64>() / best_is_self.len() as f64
                },
                id_r_max_any: median(col(&|p| p.best_r)),
                id_r_med_any: median(col(&|p| p.med_r_any)),
                id_rna_self: median(col(&|p| p.rna_r_self)),
                id_rna_rank: median(rna_rank.clone()),
                id_rna_top1: frac(&rna_rank, |x| x == 1.0),
                id_rna_top5: frac(&rna_rank, |x| x <= 5.0),
                marg_shift: median(shared.iter().map(|&i| profiles[i].marg_shift).collect()),
                marg_se_gain: median(shared.iter().map(|&i| profiles[i].marg_se_gain).collect()),
                wshare: median(shared.iter().map(|&i| profiles[i].wshare).collect()),
                redund_ge3: frac(&other_src, |x| x >= 3.0),
                other_src_med: median(other_src),
                no_anchors_cannot_audit: anchor_lines == 0,
            }
        })
        .collect();
    ValueReport { datasets, profiles }
}

#[cfg(test)]
#[path = "value_tests.rs"]
mod tests;
