use mokume_core::stats::mean_finite;
use mokume_core::{ProteinId, SampleId};

pub(crate) fn knn_imputed_values<F>(
    proteins: &[ProteinId],
    samples: &[SampleId],
    n_neighbors: usize,
    value_at: &mut F,
) -> Vec<(ProteinId, SampleId, f64)>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let n_neighbors = n_neighbors.max(1);
    let matrix = proteins
        .iter()
        .map(|protein| {
            samples
                .iter()
                .map(|sample| value_at(*protein, *sample).filter(|value| value.is_finite()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let column_means = column_means(&matrix, samples.len());

    let mut imputed = Vec::new();
    for (receiver_index, protein) in proteins.iter().enumerate() {
        let receiver = &matrix[receiver_index];
        if receiver.iter().all(Option::is_some) {
            continue;
        }
        let ranked = ranked_donors(&matrix, receiver);
        for (sample_index, sample) in samples.iter().enumerate() {
            if receiver[sample_index].is_some() {
                continue;
            }
            let fill = neighbour_mean(&matrix, &ranked, sample_index, n_neighbors)
                .unwrap_or(column_means[sample_index]);
            if fill.is_finite() {
                imputed.push((*protein, *sample, fill));
            }
        }
    }
    imputed
}

/// Mean of each sample's observed values, 0 for a sample without any.
fn column_means(matrix: &[Vec<Option<f64>>], n_samples: usize) -> Vec<f64> {
    (0..n_samples)
        .map(|sample_index| {
            mean_finite(
                &matrix
                    .iter()
                    .filter_map(|row| row[sample_index])
                    .collect::<Vec<_>>(),
            )
            .unwrap_or(0.0)
        })
        .collect()
}

/// Rows with a distance to `receiver`, nearest first and ties by row index.
/// Distances do not depend on the missing cell, so a protein ranks them once
/// and each of its missing cells takes the nearest rows observed in that sample.
fn ranked_donors(matrix: &[Vec<Option<f64>>], receiver: &[Option<f64>]) -> Vec<usize> {
    let mut donors = matrix
        .iter()
        .enumerate()
        .filter_map(|(donor_index, donor)| {
            nan_euclidean_distance(receiver, donor).map(|distance| (distance, donor_index))
        })
        .collect::<Vec<_>>();
    donors.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    donors
        .into_iter()
        .map(|(_, donor_index)| donor_index)
        .collect()
}

/// Mean of the first `n_neighbors` ranked rows observed in the sample, if any.
fn neighbour_mean(
    matrix: &[Vec<Option<f64>>],
    ranked: &[usize],
    sample_index: usize,
    n_neighbors: usize,
) -> Option<f64> {
    let values = ranked
        .iter()
        .filter_map(|&donor_index| matrix[donor_index][sample_index])
        .take(n_neighbors)
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn nan_euclidean_distance(left: &[Option<f64>], right: &[Option<f64>]) -> Option<f64> {
    let mut observed = 0usize;
    let mut squared = 0.0;
    for (left, right) in left.iter().zip(right) {
        let (Some(left), Some(right)) = (left, right) else {
            continue;
        };
        observed += 1;
        let delta = left - right;
        squared += delta * delta;
    }
    (observed > 0).then(|| (squared * left.len() as f64 / observed as f64).sqrt())
}
