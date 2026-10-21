//! imputeLCMD MinProb: column low quantiles and a shared median protein SD.
use mokume_core::stats::quantile_linear;
use mokume_core::{MokumeError, ProteinId, Result, SampleId};

use crate::stochastic::Random;

pub(crate) fn minprob_imputed_values<F>(
    proteins: &[ProteinId],
    samples: &[SampleId],
    quantile: f64,
    tune_sigma: f64,
    seed: u64,
    value_at: &mut F,
) -> Result<Vec<(ProteinId, SampleId, f64)>>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let matrix = proteins
        .iter()
        .map(|protein| {
            samples
                .iter()
                .map(|sample| value_at(*protein, *sample).filter(|v| v.is_finite()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if proteins.is_empty() || samples.is_empty() {
        return Ok(Vec::new());
    }
    let sd = median_protein_sd(&matrix, samples.len())? * tune_sigma;
    let mut random = Random::new(seed);
    let mut imputed = Vec::new();
    for (j, sample) in samples.iter().enumerate() {
        let mut observed = matrix.iter().filter_map(|row| row[j]).collect::<Vec<_>>();
        let mean =
            quantile_linear(&mut observed, quantile).ok_or_else(|| MokumeError::InvalidInput {
                message: "MinProb needs observed values in every sample and a quantile in [0,1]"
                    .to_owned(),
            })?;
        for (i, protein) in proteins.iter().enumerate() {
            // The official entry point draws an entire column before applying its mask.
            let draw = mean + sd * random.normal();
            if matrix[i][j].is_none() {
                imputed.push((*protein, *sample, draw));
            }
        }
    }
    Ok(imputed)
}

fn median_protein_sd(matrix: &[Vec<Option<f64>>], n_samples: usize) -> Result<f64> {
    let mut standard_deviations = matrix
        .iter()
        .filter_map(|row| {
            let values = row.iter().flatten().copied().collect::<Vec<_>>();
            if values.len() * 2 <= n_samples || values.len() < 2 {
                return None;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            Some(
                (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64)
                    .sqrt(),
            )
        })
        .collect::<Vec<_>>();
    quantile_linear(&mut standard_deviations, 0.5).ok_or_else(|| MokumeError::InvalidInput {
        message: "MinProb cannot estimate SD: no proteins with more than 50% observed samples and at least two values".to_owned(),
    })
}
