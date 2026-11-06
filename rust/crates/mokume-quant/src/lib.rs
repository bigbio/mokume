use mokume_core::{PeptideId, QuantMethod, SampleId};

mod directlfq_aligned;
mod maxlfq;
mod precision;

pub use directlfq_aligned::{
    direct_lfq_aligned, direct_lfq_aligned_with, sample_shifts as direct_lfq_sample_shifts,
    sparse_sample_shifts as direct_lfq_sparse_sample_shifts, DirectLfqIon, DirectLfqNormalizedIon,
    DirectLfqResult, DirectLfqRetain,
};
pub use maxlfq::{
    max_lfq, max_lfq_with_samples, solve_max_lfq, solve_max_lfq_with_stabilization, MaxLfqResult,
    DEFAULT_MAXLFQ_MIN_RATIO_COUNT,
};
pub use precision::{cell_precisions, profile_residuals, ProfileResiduals};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeptideMeasurement {
    pub peptide: PeptideId,
    pub sample: SampleId,
    pub intensity: f64,
}

pub fn supported_methods() -> &'static [QuantMethod] {
    &[
        QuantMethod::DirectLfq,
        QuantMethod::Pibaq,
        QuantMethod::MaxLfq,
        QuantMethod::TopN,
        QuantMethod::Sum,
        QuantMethod::Median,
        QuantMethod::Ratio,
        QuantMethod::Abd,
        QuantMethod::Intensity,
        QuantMethod::PeptideCount,
        QuantMethod::SpectralCount,
    ]
}

pub(crate) fn sample_order(measurements: &[PeptideMeasurement]) -> Vec<SampleId> {
    let mut samples = measurements
        .iter()
        .filter(|measurement| measurement.intensity.is_finite() && measurement.intensity > 0.0)
        .map(|measurement| measurement.sample)
        .collect::<Vec<_>>();
    samples.sort_by_key(|sample| sample.get());
    samples.dedup();
    samples
}

pub(crate) fn median(values: &mut Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    // Selection yields the same order statistics as a full sort.
    let even = values.len().is_multiple_of(2);
    let midpoint = values.len() / 2;
    let (below, &mut upper, _) = values.select_nth_unstable_by(midpoint, f64::total_cmp);
    if even {
        below
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .map(|lower| (lower + upper) / 2.0)
    } else {
        Some(upper)
    }
}

#[cfg(test)]
mod tests {
    use super::{max_lfq, PeptideMeasurement};
    use mokume_core::{PeptideId, SampleId};

    #[test]
    fn max_lfq_preserves_two_sample_ratio() -> mokume_core::Result<()> {
        let measurements = vec![
            PeptideMeasurement {
                peptide: PeptideId::new(0),
                sample: SampleId::new(0),
                intensity: 100.0,
            },
            PeptideMeasurement {
                peptide: PeptideId::new(0),
                sample: SampleId::new(1),
                intensity: 200.0,
            },
            PeptideMeasurement {
                peptide: PeptideId::new(1),
                sample: SampleId::new(0),
                intensity: 400.0,
            },
            PeptideMeasurement {
                peptide: PeptideId::new(1),
                sample: SampleId::new(1),
                intensity: 800.0,
            },
        ];

        let quantities = max_lfq(&measurements)?;
        let sample_0 = value_for(&quantities, SampleId::new(0));
        let sample_1 = value_for(&quantities, SampleId::new(1));

        assert!((sample_0 - 500.0).abs() < 1e-6);
        assert!((sample_1 - 1000.0).abs() < 1e-6);
        assert!((sample_1 / sample_0 - 2.0).abs() < 1e-6);
        Ok(())
    }

    fn value_for(quantities: &[(SampleId, f64)], sample: SampleId) -> f64 {
        quantities
            .iter()
            .find_map(|(candidate, value)| (*candidate == sample).then_some(*value))
            .unwrap_or(0.0)
    }
}
