//! Principal-component scores for the batch-correction outlier pass.
//!
//! Like sklearn's full PCA solver, center the samples x features matrix,
//! factor it directly as U S V^T, and project onto V. The right-singular-vector
//! sign convention matches `svd_flip(u_based_decision=false)`.
//!
//! A bidiagonal SVD replaces the old feature-Gram eigensolver, which stopped
//! after 100 individual Jacobi rotations even on much larger matrices. Direct
//! SVD also avoids squaring the condition number or allocating features^2.

use nalgebra::{linalg::SVD, DMatrix};

/// Bounded SVD iterations; convergence uses machine precision, not this cap.
const SVD_MAX_ITERATIONS: usize = 100_000;

/// PCA scores with min(n_components, samples, features) columns.
///
/// Repeated singular values can rotate the corresponding basis; distances and
/// the retained subspace, rather than arbitrary coordinate signs, identify PCA.
///
/// # Panics
/// Panics on a ragged/non-finite matrix or failure of the numerical solver.
/// Unconverged components must not be used to decide which samples to remove.
pub fn pca_scores(data: &[Vec<f64>], n_components: usize) -> Vec<Vec<f64>> {
    let n_samples = data.len();
    let n_features = data.first().map_or(0, Vec::len);
    let keep = n_components.min(n_samples).min(n_features);
    if keep == 0 {
        return vec![Vec::new(); n_samples];
    }
    assert!(
        data.iter()
            .all(|r| r.len() == n_features && r.iter().all(|v| v.is_finite())),
        "PCA requires a finite rectangular matrix"
    );
    let means = (0..n_features)
        .map(|j| data.iter().map(|r| r[j]).sum::<f64>() / n_samples as f64)
        .collect::<Vec<_>>();
    let centered = DMatrix::from_fn(n_samples, n_features, |i, j| data[i][j] - means[j]);
    assert!(
        centered.iter().all(|v| v.is_finite()),
        "PCA centering overflowed"
    );
    let Some(svd) = SVD::try_new(centered, false, true, f64::EPSILON, SVD_MAX_ITERATIONS) else {
        panic!("PCA SVD did not converge within {SVD_MAX_ITERATIONS} iterations");
    };
    let Some(v_t) = svd.v_t else {
        unreachable!("right singular vectors were requested");
    };
    project(data, &means, &v_t, keep)
}

fn project(data: &[Vec<f64>], means: &[f64], v_t: &DMatrix<f64>, keep: usize) -> Vec<Vec<f64>> {
    let mut scores = vec![vec![0.0; keep]; data.len()];
    for k in 0..keep {
        let mut largest = 0;
        for j in 1..means.len() {
            if v_t[(k, j)].abs() > v_t[(k, largest)].abs() {
                largest = j;
            }
        }
        let sign = if v_t[(k, largest)] < 0.0 { -1.0 } else { 1.0 };
        let center = means
            .iter()
            .enumerate()
            .map(|(j, v)| v * v_t[(k, j)])
            .sum::<f64>();
        for (i, row) in scores.iter_mut().enumerate() {
            // The public Python path calls PCA.fit().transform(), whose
            // projection subtracts the projected mean after multiplication.
            row[k] = sign
                * (data[i]
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v * v_t[(k, j)])
                    .sum::<f64>()
                    - center);
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct OfficialFixtures {
        cases: Vec<OfficialCase>,
    }

    #[derive(serde::Deserialize)]
    struct OfficialCase {
        name: String,
        data: Vec<Vec<f64>>,
        n_components: usize,
        expected: Vec<Vec<f64>>,
    }

    #[test]
    fn full_svd_matches_sklearn_distances() -> Result<(), Box<dyn std::error::Error>> {
        let fixtures: OfficialFixtures =
            serde_json::from_str(include_str!("../../tests/data/pca_sklearn_full.json"))?;
        let distance = |x: &[f64], y: &[f64]| {
            x.iter()
                .zip(y)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        for case in fixtures.cases {
            let actual = pca_scores(&case.data, case.n_components);
            for i in 0..actual.len() {
                for j in 0..i {
                    let a = distance(&actual[i], &actual[j]);
                    let e = distance(&case.expected[i], &case.expected[j]);
                    assert!(
                        (a - e).abs() <= 1e-9 + 1e-9 * e.abs(),
                        "{}[{i},{j}]: {a} != {e}",
                        case.name
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn pca_clamps_components_and_centers() {
        // Two features, three samples; rank is at most 2.
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 9.0]];
        let scores = pca_scores(&data, 5);
        // keep = min(5, 3 samples, 2 features) = 2.
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|row| row.len() == 2));
        // Centered scores sum to (near) zero per component.
        for k in 0..2 {
            let sum: f64 = scores.iter().map(|row| row[k]).sum();
            assert!(sum.abs() < 1e-9, "component {k} not centered: {sum}");
        }
    }

    #[test]
    fn pca_preserves_pairwise_distances_on_full_rank() {
        // With n_components >= rank, PCA is a rigid rotation: pairwise Euclidean
        // distances in score space equal those in centered feature space.
        let data = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];
        let scores = pca_scores(&data, 3);
        let dist = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        for i in 0..data.len() {
            for j in 0..data.len() {
                let original = dist(&data[i], &data[j]);
                let projected = dist(&scores[i], &scores[j]);
                assert!(
                    (original - projected).abs() < 1e-9,
                    "distance {i},{j} changed: {original} vs {projected}"
                );
            }
        }
    }
}
