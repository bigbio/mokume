//! Small dense solves for LIM's closed-form ridge blocks.
//!
//! Every solve LIM performs is a symmetric positive (semi-)definite normal
//! system of dimension <= ~40 (the feature ridge) or `rank` (the ALS factors),
//! so a hand-written Cholesky with an LU fallback is enough and avoids pulling
//! a linear-algebra dependency into `mokume-stats`.

/// Solve `a x = b` for a row-major `k x k` matrix `a`.
///
/// Tries Cholesky first (the systems are SPD after the ridge term) and falls
/// back to LU with partial pivoting. Returns `None` only for a numerically
/// singular matrix.
pub fn solve(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), k * k);
    debug_assert_eq!(b.len(), k);
    cholesky_solve(a, b, k).or_else(|| lu_solve(a, b, k))
}

fn cholesky_solve(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }
    // Forward then backward substitution.
    let mut y = vec![0.0_f64; k];
    for i in 0..k {
        let mut sum = b[i];
        for p in 0..i {
            sum -= l[i * k + p] * y[p];
        }
        y[i] = sum / l[i * k + i];
    }
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut sum = y[i];
        for p in (i + 1)..k {
            sum -= l[p * k + i] * x[p];
        }
        x[i] = sum / l[i * k + i];
    }
    Some(x)
}

fn lu_solve(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..k {
        let pivot =
            (col..k).max_by(|&r1, &r2| m[r1 * k + col].abs().total_cmp(&m[r2 * k + col].abs()))?;
        if m[pivot * k + col].abs() < 1e-300 {
            return None;
        }
        if pivot != col {
            for j in 0..k {
                m.swap(pivot * k + j, col * k + j);
            }
            x.swap(pivot, col);
        }
        for row in (col + 1)..k {
            let factor = m[row * k + col] / m[col * k + col];
            if factor != 0.0 {
                for j in col..k {
                    m[row * k + j] -= factor * m[col * k + j];
                }
                x[row] -= factor * x[col];
            }
        }
    }
    for i in (0..k).rev() {
        let mut sum = x[i];
        for j in (i + 1)..k {
            sum -= m[i * k + j] * x[j];
        }
        x[i] = sum / m[i * k + i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Weighted least-squares line `y = a + b x` with weights `w` (the
/// `np.polyfit(x, y, 1, w=sqrt(w))` call of the prototype). Returns `(b, a)`.
pub fn weighted_line(x: &[f64], y: &[f64], w: &[f64]) -> Option<(f64, f64)> {
    let sw: f64 = w.iter().sum();
    if sw <= 0.0 {
        return None;
    }
    let mx = x.iter().zip(w).map(|(xi, wi)| xi * wi).sum::<f64>() / sw;
    let my = y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sw;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for ((xi, yi), wi) in x.iter().zip(y).zip(w) {
        sxx += wi * (xi - mx) * (xi - mx);
        sxy += wi * (xi - mx) * (yi - my);
    }
    if sxx <= 0.0 {
        return None;
    }
    let b = sxy / sxx;
    Some((b, my - b * mx))
}

#[cfg(test)]
mod tests {
    use super::{solve, weighted_line};

    #[test]
    fn ridge_normal_equations_are_solved() {
        // (X'X + I) beta = X'y for X = [[1,0],[1,1],[1,2]], y = [1,3,5]
        // X'X = [[3,3],[3,5]]; +I -> [[4,3],[3,6]]; X'y = [9,13]
        let a = [4.0, 3.0, 3.0, 6.0];
        let x = solve(&a, &[9.0, 13.0], 2).unwrap_or_default();
        assert!((4.0 * x[0] + 3.0 * x[1] - 9.0).abs() < 1e-12);
        assert!((3.0 * x[0] + 6.0 * x[1] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn indefinite_system_falls_back_to_lu() {
        let a = [0.0, 1.0, 1.0, 0.0];
        let x = solve(&a, &[2.0, 3.0], 2).unwrap_or_default();
        assert_eq!(x, vec![3.0, 2.0]);
        assert!(solve(&[0.0; 4], &[1.0, 1.0], 2).is_none());
    }

    #[test]
    fn weighted_line_recovers_exact_fit() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|v| 0.5 - 2.0 * v).collect();
        let (b, a) = weighted_line(&x, &y, &[1.0, 2.0, 3.0, 4.0]).unwrap_or((0.0, 0.0));
        assert!((b + 2.0).abs() < 1e-12 && (a - 0.5).abs() < 1e-12);
    }
}
