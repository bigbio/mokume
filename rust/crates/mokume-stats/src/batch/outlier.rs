//! Iterative PCA + HDBSCAN outlier removal for batch correction.
//!
//! Port of `mokume.postprocessing.batch_correction.iterative_outlier_removal`
//! (`batch_correction.py:444`). The Python input `df` is a `features x samples`
//! matrix (proteins are rows, samples are columns). Each iteration:
//!   1. PCA on `df.T` (`compute_pca`, samples become the points) keeping
//!      `n_components` components;
//!   2. HDBSCAN over the PCA score rows (`find_clusters`), whose `-1` labels are
//!      the outlier samples;
//!   3. drop the outlier columns from `df`.
//!
//! The loop runs up to `n_iter` times and stops early the first iteration that
//! finds no outliers. The Python plotting side effect is intentionally omitted
//! (it is a Python-periphery concern); only the surviving-sample selection is
//! reproduced here.
//!
//! ## Cross-language tolerance
//!
//! The outlier set is the HDBSCAN noise set, which is reproduced exactly when
//! clusters are cleanly separated (see [`super::hdbscan`]). On data with samples
//! sitting on a density boundary, a boundary sample may flip between cluster and
//! noise relative to sklearn under a different floating-point tie ordering; that
//! is the documented tolerance. Genuine outliers (far from every dense cluster)
//! agree with sklearn.

use super::hdbscan::hdbscan_labels;
use super::pca::pca_scores;

/// `cluster_selection_epsilon` baked into `find_clusters`.
const CLUSTER_SELECTION_EPSILON: f64 = 0.01;
/// `allow_single_cluster` baked into `find_clusters`.
const ALLOW_SINGLE_CLUSTER: bool = true;

/// Result of one outlier-removal iteration: the outlier sample indices removed,
/// relative to the *surviving* sample order entering the iteration.
#[derive(Debug, Clone)]
pub struct OutlierIteration {
    pub outlier_local_indices: Vec<usize>,
}

/// Outcome of [`iterative_outlier_removal`]: the column indices (into the
/// original `data` sample order) that survive every iteration, in ascending
/// order, plus a per-iteration trace for inspection/tests.
#[derive(Debug, Clone)]
pub struct OutlierRemoval {
    pub kept_sample_indices: Vec<usize>,
    pub iterations: Vec<OutlierIteration>,
}

/// Run the iterative PCA + HDBSCAN outlier removal, mirroring
/// `iterative_outlier_removal`. `data` is the `features x samples` matrix
/// (`data[f][s]`); the returned `kept_sample_indices` are the original sample
/// columns that were never flagged as outliers. `n_components`,
/// `min_cluster_size`, `min_samples`, and `n_iter` mirror the Python keyword
/// arguments. A run with fewer than two samples keeps everything (nothing to
/// cluster).
pub fn iterative_outlier_removal(
    data: &[Vec<f64>],
    n_components: usize,
    min_cluster_size: usize,
    min_samples: usize,
    n_iter: usize,
) -> OutlierRemoval {
    let n_samples = data.first().map_or(0, Vec::len);
    let mut kept: Vec<usize> = (0..n_samples).collect();
    let mut iterations = Vec::new();

    if n_samples < 2 {
        return OutlierRemoval {
            kept_sample_indices: kept,
            iterations,
        };
    }

    for _ in 0..n_iter {
        // Build the samples x features point matrix (PCA runs on df.T) over the
        // currently surviving columns.
        let points: Vec<Vec<f64>> = kept
            .iter()
            .map(|&sample| data.iter().map(|row| row[sample]).collect())
            .collect();

        let scores = pca_scores(&points, n_components);
        let labels = hdbscan_labels(
            &scores,
            min_cluster_size,
            min_samples,
            CLUSTER_SELECTION_EPSILON,
            ALLOW_SINGLE_CLUSTER,
        );

        let outlier_local_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &label)| label == -1)
            .map(|(idx, _)| idx)
            .collect();

        iterations.push(OutlierIteration {
            outlier_local_indices: outlier_local_indices.clone(),
        });

        if outlier_local_indices.is_empty() {
            break;
        }

        // Drop the flagged columns; survivors keep their original indices.
        let outlier_set: std::collections::HashSet<usize> =
            outlier_local_indices.into_iter().collect();
        kept = kept
            .into_iter()
            .enumerate()
            .filter(|(local, _)| !outlier_set.contains(local))
            .map(|(_, original)| original)
            .collect();
    }

    OutlierRemoval {
        kept_sample_indices: kept,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic features x samples matrix: two tight sample clusters
    /// plus three far outlier samples, matching the structure of the Python
    /// oracle (`gen_batch_oracle.py`, separated regime).
    fn synthetic_matrix() -> (Vec<Vec<f64>>, Vec<String>) {
        let n_features = 8;
        let mut columns: Vec<(String, Vec<f64>)> = Vec::new();
        // Cluster A near 0.0, cluster B near 10.0, deterministic jitter.
        for i in 0..6 {
            let jitter = (i as f64) * 0.01;
            columns.push((format!("A{i:02}"), vec![jitter; n_features]));
        }
        for i in 0..6 {
            let jitter = (i as f64) * 0.01;
            columns.push((format!("B{i:02}"), vec![10.0 + jitter; n_features]));
        }
        // Three isolated outliers far from both clusters.
        columns.push(("OUT0".into(), vec![50.0; n_features]));
        columns.push(("OUT1".into(), vec![-40.0; n_features]));
        columns.push(("OUT2".into(), vec![90.0; n_features]));

        let names: Vec<String> = columns.iter().map(|(name, _)| name.clone()).collect();
        let n_samples = columns.len();
        let mut data = vec![vec![0.0; n_samples]; n_features];
        for (sample, (_, values)) in columns.iter().enumerate() {
            for (feature, &value) in values.iter().enumerate() {
                data[feature][sample] = value;
            }
        }
        (data, names)
    }

    #[test]
    fn removes_far_outliers_and_converges() {
        let (data, names) = synthetic_matrix();
        let result = iterative_outlier_removal(&data, 5, 5, 5, 10);

        // The three far outliers (last three columns) are dropped.
        let kept_names: Vec<&str> = result
            .kept_sample_indices
            .iter()
            .map(|&i| names[i].as_str())
            .collect();
        assert!(!kept_names.contains(&"OUT0"));
        assert!(!kept_names.contains(&"OUT1"));
        assert!(!kept_names.contains(&"OUT2"));
        // All twelve dense samples survive.
        assert_eq!(kept_names.len(), 12);
        assert!(
            kept_names
                .iter()
                .all(|name| name.starts_with('A') || name.starts_with('B'))
        );

        // Converges: first iteration removes the outliers, second finds none.
        assert!(!result.iterations.is_empty());
        assert_eq!(result.iterations[0].outlier_local_indices.len(), 3);
        let last_outliers = result
            .iterations
            .last()
            .map(|iteration| iteration.outlier_local_indices.is_empty());
        assert_eq!(last_outliers, Some(true));
    }

    #[test]
    fn single_sample_keeps_everything() {
        let data = vec![vec![1.0], vec![2.0]];
        let result = iterative_outlier_removal(&data, 5, 5, 5, 10);
        assert_eq!(result.kept_sample_indices, vec![0]);
        assert!(result.iterations.is_empty());
    }
}
