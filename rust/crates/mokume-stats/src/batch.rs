//! Batch-effect correction.
//!
//! The empirical-Bayes ComBat implementation (covariate-aware standardization,
//! parametric and non-parametric priors) is ported into `batch/combat.rs`,
//! mirroring `mokume/postprocessing/batch_correction.py`. Batch-label detection
//! (`detection`) and the iterative PCA + HDBSCAN outlier pass (`outlier`,
//! backed by `pca` and `hdbscan`) cover the remaining detection-side ports
//! of the same Python module.
//!
//! [`bridle`] is BRIDLE (Batch Removal via Intrinsic Detectability and Latent
//! Estimation, linear variant) for multi-dataset collections: per-dataset
//! technical offsets from intrinsic protein features (detectability) plus
//! anchor-sample residuals, TMT plex effects and sample loadings, fitted
//! jointly with shared latent per-sample biology
//! (`correct-batches --method bridle`).

pub mod bridle;
mod combat;
mod detection;
mod hdbscan;
mod outlier;
mod pca;

pub use combat::{combat, combat_parametric, ComBatParams};
pub use detection::{
    detect_batches, factorize, sample_prefix, BatchDetectionMethod, DetectBatchesError,
};
pub use hdbscan::hdbscan_labels;
pub use outlier::{iterative_outlier_removal, OutlierIteration, OutlierRemoval};
pub use pca::pca_scores;

#[cfg(test)]
mod golden_tests {
    use super::{hdbscan_labels, iterative_outlier_removal, pca_scores};

    /// Fixed `samples x features` matrix matching the Python golden oracle
    /// (`gen_golden.py`): two dense clusters plus two far outliers, no RNG.
    const GOLDEN_X: [[f64; 4]; 14] = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.1, 0.0],
        [0.0, 0.1, 0.0, 0.1],
        [0.2, 0.1, 0.0, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.2, 0.1],
        [10.0, 10.0, 10.0, 10.0],
        [10.1, 10.0, 10.1, 10.0],
        [10.0, 10.1, 10.0, 10.1],
        [10.2, 10.1, 10.0, 10.0],
        [10.1, 10.2, 10.1, 10.0],
        [10.0, 10.0, 10.2, 10.1],
        [50.0, 50.0, 50.0, 50.0],
        [-30.0, -30.0, -30.0, -30.0],
    ];

    /// `PCA(n_components=5).fit(GOLDEN_X).transform(GOLDEN_X)` from the Python
    /// oracle (`gen_golden.py`), exactly the `compute_pca` call path. Reproduced
    /// including the `svd_flip(u_based_decision=False)` sign convention.
    const GOLDEN_PCA: [[f64; 4]; 14] = [
        [-11.528571, 0.011538, -0.012416, 0.018543],
        [-11.428573, 0.015381, 0.086463, 0.004108],
        [-11.428569, 0.007693, -0.111293, 0.032974],
        [-11.378573, -0.142323, 0.046767, 0.036571],
        [-11.328574, -0.071874, -0.025728, -0.094888],
        [-11.378570, 0.169239, 0.027284, -0.013925],
        [8.471429, 0.011240, -0.012082, 0.018062],
        [8.571427, 0.015083, 0.086798, 0.003626],
        [8.571431, 0.007395, -0.110958, 0.032493],
        [8.621427, -0.142621, 0.047102, 0.036089],
        [8.671426, -0.072172, -0.025393, -0.095370],
        [8.621430, 0.168941, 0.027619, -0.014406],
        [88.471429, 0.010049, -0.010742, 0.016135],
        [-71.528571, 0.012432, -0.013421, 0.019988],
    ];

    fn golden_rows() -> Vec<Vec<f64>> {
        GOLDEN_X.iter().map(|row| row.to_vec()).collect()
    }

    #[test]
    fn pca_scores_match_sklearn_golden() {
        let scores = pca_scores(&golden_rows(), 5);
        assert_eq!(scores.len(), 14);
        // keep = min(5, 14 samples, 4 features) = 4.
        assert!(scores.iter().all(|row| row.len() == 4));
        for (i, (got, want)) in scores.iter().zip(GOLDEN_PCA.iter()).enumerate() {
            for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < 1e-4,
                    "PCA sample {i} component {k}: got {g}, want {w}"
                );
            }
        }
    }

    #[test]
    fn hdbscan_labels_match_sklearn_golden() {
        // sklearn HDBSCAN over the PCA scores: clusters {0,1}, outliers 12/13.
        let scores = pca_scores(&golden_rows(), 5);
        for &(mcs, ms) in &[(5usize, 5usize), (3, 3), (4, 4)] {
            let labels = hdbscan_labels(&scores, mcs, ms, 0.01, true);
            // Far outliers are noise.
            assert_eq!(labels[12], -1, "mcs{mcs}_ms{ms} sample 12");
            assert_eq!(labels[13], -1, "mcs{mcs}_ms{ms} sample 13");
            // Two dense clusters split 0..6 vs 6..12.
            assert!(labels[..6].iter().all(|&l| l == labels[0] && l != -1));
            assert!(labels[6..12].iter().all(|&l| l == labels[6] && l != -1));
            assert_ne!(labels[0], labels[6]);
        }
    }

    #[test]
    fn iterative_outlier_removal_matches_python_golden() {
        // Python oracle (`gen_iter_oracle.py`): the iterative pass over the
        // features x samples matrix (transpose of GOLDEN_X) drops the two far
        // samples (12, 13) in iteration 1, then converges in iteration 2 with no
        // outliers. The 12 dense samples survive.
        let n_samples = GOLDEN_X.len();
        let n_features = GOLDEN_X[0].len();
        let mut data = vec![vec![0.0; n_samples]; n_features];
        for (sample, row) in GOLDEN_X.iter().enumerate() {
            for (feature, &value) in row.iter().enumerate() {
                data[feature][sample] = value;
            }
        }

        for &(mcs, ms) in &[(5usize, 5usize), (3, 3), (4, 4)] {
            let result = iterative_outlier_removal(&data, 5, mcs, ms, 10);
            // Surviving samples are exactly 0..=11 (the dense block).
            assert_eq!(
                result.kept_sample_indices,
                (0..12).collect::<Vec<_>>(),
                "mcs{mcs}_ms{ms} kept samples"
            );
            // Iteration 1 removes the two far samples (local indices 12, 13).
            assert_eq!(
                result.iterations[0].outlier_local_indices,
                vec![12, 13],
                "mcs{mcs}_ms{ms} iter1 outliers"
            );
            // Converges: the final iteration finds nothing.
            let last_empty = result
                .iterations
                .last()
                .map(|iteration| iteration.outlier_local_indices.is_empty());
            assert_eq!(last_empty, Some(true), "mcs{mcs}_ms{ms} convergence");
        }
    }
}
