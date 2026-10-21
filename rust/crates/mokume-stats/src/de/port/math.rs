pub(super) fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
pub(super) fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}
pub(super) fn forward(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; b.len()];
    for i in 0..b.len() {
        out[i] = (b[i] - dot(&l[i][..i], &out[..i])) / l[i][i];
    }
    out
}
pub(super) fn backward(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut out = b.to_vec();
    for i in (0..n).rev() {
        out[i] /= l[i][i];
        let value = out[i];
        for j in 0..i {
            out[j] -= value * l[i][j];
        }
    }
    out
}
pub(super) fn cholesky(a: &[Vec<f64>], alpha: f64) -> (Vec<Vec<f64>>, Option<usize>) {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut diagonal = 0.0;
        for j in 0..i {
            l[i][j] = (a[i][j] - dot(&l[i][..j], &l[j][..j])) / l[j][j];
            diagonal += l[i][j] * l[i][j];
        }
        let reduced = a[i][i] + alpha - diagonal;
        l[i][i] = reduced;
        if reduced <= 0.0 || !reduced.is_finite() {
            return (l, Some(i));
        }
        l[i][i] = reduced.sqrt();
    }
    (l, None)
}
pub(super) fn failed_bound(mut l: Vec<Vec<f64>>, failed: usize) -> f64 {
    let diagonal = l[failed][failed];
    l[failed][failed] = 1.0;
    let mut unit = vec![0.0; failed + 1];
    unit[failed] = 1.0;
    let direction = backward(&l, &unit);
    let size = norm(&direction);
    -diagonal / size / size
}
/// PORT's deterministic inverse-norm singular-value estimate (DL7SVN).
pub(super) fn singular_estimate(l: &[Vec<f64>]) -> (f64, Vec<f64>, Vec<f64>) {
    let n = l.len();
    let mut x = vec![0.0; n];
    let mut seed = 2;
    if n == 0 || (0..n).any(|i| l[i][i] == 0.0) {
        return (0.0, x.clone(), x);
    }
    let mut random = || {
        seed = (3432 * seed) % 9973;
        0.5 * (1.0 + seed as f64 / 9973.0)
    };
    let last = n - 1;
    x[last] = random() / l[last][last];
    for i in 0..last {
        x[i] = x[last] * l[last][i];
    }
    for j in (0..last).rev() {
        let b = random();
        let plus = (b - x[j]) / l[j][j];
        let minus = (-b - x[j]) / l[j][j];
        let sp = (b - x[j]).abs() + (0..j).map(|i| (x[i] + l[j][i] * plus).abs()).sum::<f64>();
        let sm = (-b - x[j]).abs() + (0..j).map(|i| (x[i] + l[j][i] * minus).abs()).sum::<f64>();
        x[j] = if sm > sp { minus } else { plus };
        for i in 0..j {
            x[i] += x[j] * l[j][i];
        }
    }
    let scale = norm(&x);
    for value in &mut x {
        *value /= scale;
    }
    let y = forward(l, &x);
    (1.0 / norm(&y), x, y)
}
/// Refined Gershgorin spectral bounds used to bracket the diagonal shift.
pub(super) fn eigen_bounds(h: &[Vec<f64>]) -> (f64, f64) {
    let n = h.len();
    let sums: Vec<f64> = (0..n)
        .map(|i| (0..n).filter(|&j| j != i).map(|j| h[i][j].abs()).sum())
        .collect();
    let mut low = 0;
    let mut high = 0;
    for i in 1..n {
        if h[i][i] - sums[i] < h[low][low] - sums[low] {
            low = i;
        }
        if h[i][i] + sums[i] > h[high][high] + sums[high] {
            high = i;
        }
    }
    let bound = |k: usize, sign: f64| {
        let mut extra: f64 = 0.0;
        for i in 0..n {
            if i != k {
                let off = h[k][i].abs();
                let t = 0.5 * (sign * (h[i][i] - h[k][k]) + sums[i] - off);
                extra = extra.max(t + (t * t + sums[k] * off).sqrt());
            }
        }
        h[k][k] + sign * extra
    };
    (bound(low, -1.0), bound(high, 1.0))
}
