//! imputeLCMD 2.1 QRILC: quantile-profile regression and truncated-normal draws.
use mokume_core::stats::quantile_linear;
use mokume_core::{MokumeError, ProteinId, Result, SampleId};

use crate::stochastic::{normal_cdf, normal_quantile, Random};

pub(crate) fn qrilc_imputed_values<F>(
    proteins: &[ProteinId],
    samples: &[SampleId],
    tune_sigma: f64,
    seed: u64,
    value_at: &mut F,
) -> Result<Vec<(ProteinId, SampleId, f64)>>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let mut random = Random::new(seed);
    let mut imputed = Vec::new();
    for sample in samples {
        let column = proteins
            .iter()
            .map(|p| value_at(*p, *sample).filter(|v| v.is_finite()))
            .collect::<Vec<_>>();
        let mut observed = column.iter().flatten().copied().collect::<Vec<_>>();
        if column.is_empty() {
            continue;
        }
        let missing_fraction = 1.0 - observed.len() as f64 / column.len() as f64;
        let (mean, sd) = distribution_parameters(&mut observed, missing_fraction)?;
        let upper = mean + sd * normal_quantile(missing_fraction + 0.001);
        // tmvtnorm 1.7's univariate Gibbs branch passes sigma directly as SD.
        let draw_sd = sd * tune_sigma;
        let upper_probability = normal_cdf((upper - mean) / draw_sd);
        for (protein, value) in proteins.iter().zip(column) {
            let draw = mean + draw_sd * normal_quantile(random.uniform() * upper_probability);
            if value.is_none() {
                imputed.push((*protein, *sample, draw));
            }
        }
    }
    Ok(imputed)
}

fn distribution_parameters(observed: &mut [f64], missing_fraction: f64) -> Result<(f64, f64)> {
    if observed.len() < 2 || missing_fraction >= 0.99 {
        return Err(MokumeError::InvalidInput {
            message:
                "QRILC needs at least two observations and less than 99% missingness per sample"
                    .to_owned(),
        });
    }
    let pairs = (0..100)
        .map(|i| {
            let x = normal_quantile(
                missing_fraction + 0.001 + (0.99 - missing_fraction) * i as f64 / 99.0,
            );
            let y = quantile_linear(observed, 0.001 + i as f64 * 0.01).unwrap_or(f64::NAN);
            (x, y)
        })
        .collect::<Vec<_>>();
    let mx = pairs.iter().map(|v| v.0).sum::<f64>() / 100.0;
    let my = pairs.iter().map(|v| v.1).sum::<f64>() / 100.0;
    let numerator = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum::<f64>();
    let denominator = pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum::<f64>();
    let sd = numerator / denominator;
    let mean = my - sd * mx;
    if !mean.is_finite() || !sd.is_finite() || sd <= 0.0 {
        return Err(MokumeError::InvalidInput {
            message: "QRILC fitted distribution has nonpositive or invalid scale".to_owned(),
        });
    }
    Ok((mean, sd))
}
