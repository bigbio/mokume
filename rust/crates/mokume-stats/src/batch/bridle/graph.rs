//! Graph prior: plex-aware anchor offsets chained to the reference.
//!
//! Port of the benchmark's `graphp` arm (`arms_simple.py::graph("pb", 3)`) and
//! of its use as the prior of BRIDLE's offsets (`bvar_run.py gprior`). Batches
//! are datasets, split by plex where the fit has plexes (the reference dataset
//! is always one batch). Per gene, independently:
//!
//! 1. Anchor cells are (line, gene) values observed in >= 2 profiles. Batch
//!    offsets `o[b]` and line levels `t[l]` are fitted by `iters` rounds of
//!    alternating means on them (`t = mean(v - o)`, then `o = mean(v - t)`).
//! 2. Two batches are linked when they share >= `min_shared` anchor lines; a
//!    batch connected (possibly through other batches) to the reference batch
//!    gets the offset `o[b] - o[reference]`.
//! 3. A profile whose batch has no such offset falls back to centring: when
//!    its batch has >= 5 profiles and >= 3 values of the gene, the offset is
//!    `mean_b(gene) - G` with `G` the reference dataset's mean of the gene (else
//!    the mean of the batch means); otherwise it is 0.
//!
//! The prior of dataset `s` and gene `g` is the mean of these per-profile
//! offsets over the dataset's observed values of `g`. It exists where at
//! least one of them is non-zero (graph-linked or centred), and never for the
//! reference.

use std::collections::HashMap;

use rayon::prelude::*;

/// Minimum profiles in a batch, and values of the gene in the batch, for the
/// centring fallback.
const CENTRE_MIN_PROFILES: usize = 5;
const CENTRE_MIN_VALUES: usize = 3;

/// Inputs of [`graph_prior`]; arrays are indexed by profile (row).
pub(super) struct GraphInput<'a> {
    /// Row-major `profiles x genes`, `NaN` = not observed.
    pub values: &'a [f64],
    pub g_n: usize,
    pub s_n: usize,
    /// Dataset of each profile.
    pub srow: &'a [usize],
    /// Line of each profile.
    pub li: &'a [usize],
    /// Batch (dataset or dataset x plex) of each profile, `< n_batch`.
    pub batch: &'a [usize],
    pub n_batch: usize,
    pub ref_s: usize,
    pub min_shared: usize,
    pub iters: usize,
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Row-major `datasets x genes` prior offsets (`NaN` = no prior).
pub(super) fn graph_prior(inp: &GraphInput) -> Vec<f64> {
    let (g_n, s_n) = (inp.g_n, inp.s_n);
    let n = inp.srow.len();
    let ref_batch = (0..n)
        .find(|&i| inp.srow[i] == inp.ref_s)
        .map_or(usize::MAX, |i| inp.batch[i]);
    // profiles per batch with >= 1 observed value
    let mut batch_profiles = vec![0_usize; inp.n_batch];
    for i in 0..n {
        if inp.values[i * g_n..(i + 1) * g_n]
            .iter()
            .any(|v| v.is_finite())
        {
            batch_profiles[inp.batch[i]] += 1;
        }
    }
    let cols: Vec<Vec<f64>> = (0..g_n)
        .into_par_iter()
        .map(|g| gene_prior(inp, g, ref_batch, &batch_profiles))
        .collect();
    let mut out = vec![f64::NAN; s_n * g_n];
    for (g, col) in cols.iter().enumerate() {
        for s in 0..s_n {
            out[s * g_n + g] = col[s];
        }
    }
    out
}

/// Prior of every dataset for gene `g`.
fn gene_prior(inp: &GraphInput, g: usize, ref_batch: usize, batch_profiles: &[usize]) -> Vec<f64> {
    let (g_n, s_n) = (inp.g_n, inp.s_n);
    let n = inp.srow.len();
    let rows: Vec<usize> = (0..n)
        .filter(|&i| inp.values[i * g_n + g].is_finite())
        .collect();
    let val = |i: usize| inp.values[i * g_n + g];
    // anchor cells: lines observed in >= 2 profiles for this gene
    let mut by_line: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in &rows {
        by_line.entry(inp.li[i]).or_default().push(i);
    }
    let mut lines: Vec<(usize, Vec<usize>)> =
        by_line.into_iter().filter(|(_, r)| r.len() >= 2).collect();
    lines.sort_unstable_by_key(|(l, _)| *l);
    let mut node_of: HashMap<usize, usize> = HashMap::new();
    let mut cells: Vec<(usize, usize, f64)> = Vec::new(); // (node, line slot, value)
    for (slot, (_, rr)) in lines.iter().enumerate() {
        for &i in rr {
            let next = node_of.len();
            let node = *node_of.entry(inp.batch[i]).or_insert(next);
            cells.push((node, slot, val(i)));
        }
    }
    let n_node = node_of.len();
    let n_slot = lines.len();
    let mut node_n = vec![0.0; n_node];
    let mut slot_n = vec![0.0; n_slot];
    for &(b, l, _) in &cells {
        node_n[b] += 1.0;
        slot_n[l] += 1.0;
    }
    // alternating means
    let mut o = vec![0.0; n_node];
    let mut t = vec![0.0; n_slot];
    for _ in 0..inp.iters {
        t.fill(0.0);
        for &(b, l, v) in &cells {
            t[l] += v - o[b];
        }
        for (x, c) in t.iter_mut().zip(&slot_n) {
            *x /= c;
        }
        o.fill(0.0);
        for &(b, l, v) in &cells {
            o[b] += v - t[l];
        }
        for (x, c) in o.iter_mut().zip(&node_n) {
            *x /= c;
        }
    }
    // batches linked by >= min_shared shared anchor lines
    let mut shared: HashMap<(usize, usize), usize> = HashMap::new();
    for (_, rr) in &lines {
        let nodes: Vec<usize> = rr.iter().map(|&i| node_of[&inp.batch[i]]).collect();
        for (a, &x) in nodes.iter().enumerate() {
            for &y in &nodes[a + 1..] {
                if x != y {
                    *shared.entry((x.min(y), x.max(y))).or_insert(0) += 1;
                }
            }
        }
    }
    let mut parent: Vec<usize> = (0..n_node).collect();
    let mut edges: Vec<(usize, usize)> = shared
        .into_iter()
        .filter(|&(_, c)| c >= inp.min_shared)
        .map(|(e, _)| e)
        .collect();
    edges.sort_unstable();
    for (x, y) in edges {
        let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
        if rx != ry {
            parent[rx.max(ry)] = rx.min(ry);
        }
    }
    let root: Vec<usize> = (0..n_node).map(|x| find(&mut parent, x)).collect();
    let ref_node = node_of.get(&ref_batch).copied();
    let graph_off = |b: usize| -> Option<f64> {
        let node = *node_of.get(&b)?;
        let r = ref_node?;
        (root[node] == root[r]).then(|| o[node] - o[r])
    };
    let mut batch_off: HashMap<usize, Option<f64>> = HashMap::new();
    // centring fallback: batch means and the reference level G
    let mut bsum: HashMap<usize, (f64, usize)> = HashMap::new();
    let (mut rsum, mut rcnt) = (0.0, 0_usize);
    for &i in &rows {
        let e = bsum.entry(inp.batch[i]).or_insert((0.0, 0));
        e.0 += val(i);
        e.1 += 1;
        if inp.srow[i] == inp.ref_s {
            rsum += val(i);
            rcnt += 1;
        }
    }
    let level = if rcnt > 0 {
        rsum / rcnt as f64
    } else {
        let mut means: Vec<(usize, f64)> =
            bsum.iter().map(|(&b, &(s, c))| (b, s / c as f64)).collect();
        means.sort_unstable_by_key(|(b, _)| *b);
        means.iter().map(|(_, m)| m).sum::<f64>() / means.len().max(1) as f64
    };
    let mut sum = vec![0.0; s_n];
    let mut cnt = vec![0_usize; s_n];
    let mut nonzero = vec![false; s_n];
    for &i in &rows {
        let b = inp.batch[i];
        let off = match *batch_off.entry(b).or_insert_with(|| graph_off(b)) {
            Some(x) => x,
            None => {
                let (s, c) = bsum[&b];
                if batch_profiles[b] >= CENTRE_MIN_PROFILES && c >= CENTRE_MIN_VALUES {
                    s / c as f64 - level
                } else {
                    0.0
                }
            }
        };
        let s = inp.srow[i];
        sum[s] += off;
        cnt[s] += 1;
        nonzero[s] |= off != 0.0;
    }
    (0..s_n)
        .map(|s| {
            if s != inp.ref_s && nonzero[s] {
                sum[s] / cnt[s] as f64
            } else {
                f64::NAN
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Profiles (dataset, batch, line, offset) on 2 genes; value = t[line, g]
    /// + offset, noise-free, so every linked offset is recovered exactly.
    fn run(plan: &[(usize, usize, usize, f64)], s_n: usize, n_batch: usize) -> Vec<f64> {
        let g_n = 2;
        let t =
            |l: usize, g: usize| 20.0 + 0.37 * l as f64 - 0.81 * g as f64 + (l * l) as f64 * 0.01;
        let values: Vec<f64> = plan
            .iter()
            .flat_map(|&(_, _, l, off)| (0..g_n).map(move |g| t(l, g) + off))
            .collect();
        let srow: Vec<usize> = plan.iter().map(|p| p.0).collect();
        let batch: Vec<usize> = plan.iter().map(|p| p.1).collect();
        let li: Vec<usize> = plan.iter().map(|p| p.2).collect();
        graph_prior(&GraphInput {
            values: &values,
            g_n,
            s_n,
            srow: &srow,
            li: &li,
            batch: &batch,
            n_batch,
            ref_s: 0,
            min_shared: 3,
            iters: 400,
        })
    }

    #[test]
    fn plex_offsets_are_chained_to_the_reference() {
        let mut plan = Vec::new();
        // REF (dataset 0, batch 0): lines 0..6
        plan.extend((0..6).map(|l| (0, 0, l, 0.0)));
        // B (dataset 1): plex batch 6 = lines 0..3 (+1), plex batch 7 = 3..6 (+2)
        plan.extend((0..3).map(|l| (1, 6, l, 1.0)));
        plan.extend((3..6).map(|l| (1, 7, l, 2.0)));
        // C (dataset 2): lines 0..3 and 9..12, -0.5
        plan.extend((0..3).chain(9..12).map(|l| (2, 2, l, -0.5)));
        // F (dataset 5): only C's own lines 9..12 -> chained through C, +0.7
        plan.extend((9..12).map(|l| (5, 5, l, 0.7)));
        // E (dataset 4): 2 shared lines, 2 profiles -> neither linked nor centred
        plan.extend((0..2).map(|l| (4, 4, l, 4.0)));
        // D (dataset 3): 2 shared lines + 3 own lines -> centred (5 profiles)
        plan.extend((0..2).chain(6..9).map(|l| (3, 3, l, 3.0)));
        let p = run(&plan, 6, 8);
        let g_n = 2;
        for g in 0..g_n {
            assert!(p[g].is_nan(), "reference has no prior");
            assert!((p[g_n + g] - 1.5).abs() < 1e-9, "B: {}", p[g_n + g]);
            assert!((p[2 * g_n + g] + 0.5).abs() < 1e-9, "C: {}", p[2 * g_n + g]);
            assert!((p[5 * g_n + g] - 0.7).abs() < 1e-9, "F: {}", p[5 * g_n + g]);
            assert!(p[4 * g_n + g].is_nan(), "E: {}", p[4 * g_n + g]);
        }
        // D: mean over its profiles minus the reference mean
        let t =
            |l: usize, g: usize| 20.0 + 0.37 * l as f64 - 0.81 * g as f64 + (l * l) as f64 * 0.01;
        for g in 0..g_n {
            let d: f64 = [0, 1, 6, 7, 8].iter().map(|&l| t(l, g) + 3.0).sum::<f64>() / 5.0;
            let r: f64 = (0..6).map(|l| t(l, g)).sum::<f64>() / 6.0;
            assert!((p[3 * g_n + g] - (d - r)).abs() < 1e-9);
        }
    }

    #[test]
    fn missing_reference_gene_falls_back_to_batch_means() {
        // gene 1 is never observed in REF: no graph offset, centring level =
        // mean of the batch means
        let g_n = 2;
        let mut values = Vec::new();
        let mut srow = Vec::new();
        let mut li = Vec::new();
        for (s, off) in [(0_usize, 0.0), (1, 1.0)] {
            for l in 0..5 {
                srow.push(s);
                li.push(l);
                values.push(10.0 + l as f64 + off);
                values.push(if s == 0 { f64::NAN } else { 5.0 + l as f64 });
            }
        }
        let p = graph_prior(&GraphInput {
            values: &values,
            g_n,
            s_n: 2,
            srow: &srow,
            li: &li,
            batch: &srow,
            n_batch: 2,
            ref_s: 0,
            min_shared: 3,
            iters: 400,
        });
        assert!((p[g_n] - 1.0).abs() < 1e-9);
        // B is the only batch with gene 1: mean_b - mean of batch means = 0
        assert!(p[g_n + 1].is_nan());
    }
}
