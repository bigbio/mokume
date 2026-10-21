//! One-predictor Gaussian LOESS used by DEqMS::spectraCounteBayes.
//!
//! R stats loess defaults: degree=2, span=.75, surface="interpolate",
//! cell=.2. Nodes follow the one-dimensional loess kd tree; fitted values
//! interpolate local intercepts and derivatives with a cubic Hermite basis.
use nalgebra::{DMatrix, DVector};

const SPAN: f64 = 0.75;
const CELL: f64 = 0.2;

pub(super) fn predict(y: &[f64], x: &[f64], valid: &[bool]) -> Vec<f64> {
    let mut pairs = x
        .iter()
        .zip(y)
        .zip(valid)
        .filter(|(_, keep)| **keep)
        .map(|((&x, &y), _)| (x, y))
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    if pairs.len() < 3 {
        return vec![f64::NAN; x.len()];
    }
    let knots = tree_knots(&pairs);
    let fits = knots
        .iter()
        .map(|&q| local_quadratic(&pairs, q))
        .collect::<Vec<_>>();
    x.iter()
        .map(|&query| interpolate(query, &knots, &fits))
        .collect()
}

fn tree_knots(pairs: &[(f64, f64)]) -> Vec<f64> {
    let n = pairs.len();
    let (min, max) = (pairs[0].0, pairs[n - 1].0);
    let margin = 0.005 * (max - min).max(1e-10 * min.abs().max(max.abs()) + 1e-30);
    let mut knots = vec![min - margin, max + margin];
    let leaf_size = (n as f64 * SPAN * CELL).floor() as usize;
    let mut pending = vec![(0, n - 1, knots[0], knots[1])];
    while let Some((lo, hi, left, right)) = pending.pop() {
        if hi - lo < leaf_size {
            continue;
        }
        let mid = split_index(pairs, lo, hi);
        let split = pairs[mid].0;
        if split == left || split == right {
            continue;
        }
        knots.push(split);
        pending.push((lo, mid, left, split));
        if mid < hi {
            pending.push((mid + 1, hi, split, right));
        }
    }
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    knots
}

/// ehg124 searches offsets 0,+1,-1,+2,-2,... so equal predictors stay together.
fn split_index(pairs: &[(f64, f64)], lo: usize, hi: usize) -> usize {
    let mid = (lo + hi) / 2;
    let mut offset = 0_isize;
    loop {
        let candidate = mid as isize + offset;
        if candidate < lo as isize || candidate >= hi as isize {
            return mid;
        }
        let i = candidate as usize;
        if pairs[i].0 != pairs[i + 1].0 {
            return i;
        }
        offset = if offset <= 0 { 1 - offset } else { -offset };
    }
}

fn local_quadratic(pairs: &[(f64, f64)], q: f64) -> (f64, f64) {
    let nf = ((pairs.len() as f64 * SPAN).floor() as usize).min(pairs.len());
    let mut order = (0..pairs.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| (pairs[a].0 - q).abs().total_cmp(&(pairs[b].0 - q).abs()));
    let radius = (pairs[order[nf - 1]].0 - q).abs();
    if radius == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let mut design = DMatrix::zeros(nf, 3);
    let mut response = DVector::zeros(nf);
    for (row, &i) in order[..nf].iter().enumerate() {
        let delta = pairs[i].0 - q;
        let weight = (1.0 - (delta.abs() / radius).powi(3))
            .powi(3)
            .max(0.0)
            .sqrt();
        design[(row, 0)] = weight;
        design[(row, 1)] = weight * delta;
        design[(row, 2)] = weight * delta * delta;
        response[row] = weight * pairs[i].1;
    }
    // R equilibrates each column before QR/SVD and truncates at 100*eps*smax.
    let scales = (0..3)
        .map(|j| design.column(j).norm().max(f64::MIN_POSITIVE))
        .collect::<Vec<_>>();
    for (j, scale) in scales.iter().enumerate() {
        design.column_mut(j).scale_mut(1.0 / scale);
    }
    let svd = design.svd(true, true);
    let cutoff = svd.singular_values[0] * 100.0 * f64::EPSILON;
    match svd.solve(&response, cutoff) {
        Ok(beta) => (beta[0] / scales[0], beta[1] / scales[1]),
        Err(_) => (f64::NAN, f64::NAN),
    }
}

fn interpolate(query: f64, knots: &[f64], fits: &[(f64, f64)]) -> f64 {
    if !query.is_finite() || query < knots[0] || query > knots[knots.len() - 1] {
        return f64::NAN;
    }
    let upper = knots
        .partition_point(|&v| v < query)
        .clamp(1, knots.len() - 1);
    let lower = upper - 1;
    let width = knots[upper] - knots[lower];
    let h = (query - knots[lower]) / width;
    let phi0 = (1.0 - h).powi(2) * (1.0 + 2.0 * h);
    let phi1 = h * h * (3.0 - 2.0 * h);
    let psi0 = h * (1.0 - h).powi(2);
    let psi1 = h * h * (h - 1.0);
    phi0 * fits[lower].0
        + phi1 * fits[upper].0
        + (psi0 * fits[lower].1 + psi1 * fits[upper].1) * width
}
