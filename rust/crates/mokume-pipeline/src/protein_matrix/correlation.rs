use crate::plex::{condition_by_sample, is_reference_condition};
use crate::protein_matrix::ProteinMatrix;
use crate::{
    HashMap, MIN_SAMPLE_CORRELATION_OVERLAP, Result, SampleId, SdrfTable, info, invalid_input,
};

fn pearson_correlation(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < MIN_SAMPLE_CORRELATION_OVERLAP {
        return None;
    }
    let count = pairs.len() as f64;
    let (sum_left, sum_right) = pairs
        .iter()
        .fold((0.0, 0.0), |(left, right), (x, y)| (left + x, right + y));
    let mean_left = sum_left / count;
    let mean_right = sum_right / count;
    let (covariance, variance_left, variance_right) = pairs.iter().fold(
        (0.0, 0.0, 0.0),
        |(covariance, variance_left, variance_right), (x, y)| {
            let centered_left = x - mean_left;
            let centered_right = y - mean_right;
            (
                covariance + centered_left * centered_right,
                variance_left + centered_left * centered_left,
                variance_right + centered_right * centered_right,
            )
        },
    );
    let denominator = (variance_left * variance_right).sqrt();
    (denominator > 0.0)
        .then(|| (covariance / denominator).clamp(-1.0, 1.0))
        .filter(|correlation| correlation.is_finite())
}
impl ProteinMatrix {
    pub(crate) fn sample_columns(&self, drop_empty_samples: bool) -> Vec<(SampleId, &str)> {
        let value_samples = drop_empty_samples.then(|| self.values.sample_ids());
        let mut samples = self
            .samples
            .iter()
            .filter(|(sample, _)| {
                !self.excluded_samples.contains(sample)
                    && value_samples
                        .as_ref()
                        .is_none_or(|value_samples| value_samples.contains(sample))
            })
            .collect::<Vec<_>>();
        samples.sort_by(|left, right| left.1.cmp(right.1));
        samples
    }

    pub(crate) fn pairwise_sample_correlation(
        &self,
        left: SampleId,
        right: SampleId,
        values_are_log2: bool,
    ) -> (Option<f64>, usize) {
        let pairs = self
            .allowed_proteins
            .iter()
            .filter_map(|protein| {
                let left = self.value(*protein, left)?;
                let right = self.value(*protein, right)?;
                if values_are_log2 {
                    return (left.is_finite() && right.is_finite()).then_some((left, right));
                }
                (left.is_finite() && left > 0.0 && right.is_finite() && right > 0.0)
                    .then_some((left.log2(), right.log2()))
            })
            .collect::<Vec<_>>();
        (pearson_correlation(&pairs), pairs.len())
    }

    fn biological_samples_by_condition(
        &self,
        sdrf: &SdrfTable,
        drop_empty_samples: bool,
    ) -> Result<HashMap<String, Vec<(SampleId, String)>>> {
        let condition_by_sample = condition_by_sample(sdrf);
        let samples = self
            .sample_columns(drop_empty_samples)
            .into_iter()
            .map(|(sample, name)| {
                condition_by_sample
                    .get(name)
                    .map(|condition| (sample, name.to_owned(), condition.clone()))
                    .ok_or_else(|| {
                        invalid_input(format!(
                            "sample correlation filtering found no condition metadata for `{name}`"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut grouped = HashMap::<String, Vec<(SampleId, String)>>::new();
        for (sample, name, condition) in samples {
            if !is_reference_condition(&condition) {
                grouped.entry(condition).or_default().push((sample, name));
            }
        }
        if grouped.is_empty() {
            return Err(invalid_input(
                "sample correlation filtering found no biological samples with condition metadata",
            ));
        }
        Ok(grouped)
    }

    fn mean_sample_correlation(
        &self,
        sample: SampleId,
        name: &str,
        condition: &str,
        condition_samples: &[(SampleId, String)],
        values_are_log2: bool,
    ) -> Result<f64> {
        let mut correlations = Vec::with_capacity(condition_samples.len() - 1);
        for (peer, peer_name) in condition_samples {
            if sample == *peer {
                continue;
            }
            let (correlation, overlap) =
                self.pairwise_sample_correlation(sample, *peer, values_are_log2);
            let correlation = correlation.ok_or_else(|| {
                invalid_input(format!(
                    "sample correlation between `{name}` and `{peer_name}` in condition `{condition}` is undefined: {overlap} pairwise-complete usable proteins (minimum {MIN_SAMPLE_CORRELATION_OVERLAP})"
                ))
            })?;
            correlations.push(correlation);
        }
        Ok(correlations.iter().sum::<f64>() / correlations.len() as f64)
    }

    pub(crate) fn apply_sample_correlation_filter(
        &mut self,
        sdrf: Option<&SdrfTable>,
        threshold: f64,
        drop_empty_samples: bool,
        values_are_log2: bool,
    ) -> Result<()> {
        let sdrf = sdrf
            .ok_or_else(|| invalid_input("sample correlation filtering requires --sdrf option"))?;
        let samples_by_condition =
            self.biological_samples_by_condition(sdrf, drop_empty_samples)?;

        let mut excluded = Vec::new();
        for (condition, condition_samples) in &samples_by_condition {
            if condition_samples.len() < 2 {
                return Err(invalid_input(format!(
                    "sample correlation filtering requires at least two samples in condition `{condition}`"
                )));
            }
            for (sample, name) in condition_samples {
                let mean_correlation = self.mean_sample_correlation(
                    *sample,
                    name,
                    condition,
                    condition_samples,
                    values_are_log2,
                )?;
                info!(
                    sample = name,
                    condition,
                    mean_correlation,
                    peers = condition_samples.len() - 1,
                    threshold,
                    "sample correlation QC"
                );
                if mean_correlation < threshold {
                    excluded.push((*sample, name.clone(), mean_correlation));
                }
            }
        }
        for (sample, _, _) in &excluded {
            self.excluded_samples.insert(*sample);
        }
        info!(
            evaluated_samples = samples_by_condition.values().map(Vec::len).sum::<usize>(),
            excluded_samples = excluded.len(),
            threshold,
            "sample correlation filtering complete"
        );
        Ok(())
    }
}
