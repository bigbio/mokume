//! Shared helpers used by multiple imputation methods to read observed values
//! through the caller-provided `value_at` accessor.

use std::collections::HashMap;

use mokume_core::{ProteinId, SampleId};

pub(crate) fn finite_sample_values<F>(
    proteins: &[ProteinId],
    sample: SampleId,
    value_at: &mut F,
) -> Vec<f64>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    proteins
        .iter()
        .filter_map(|protein| value_at(*protein, sample))
        .filter(|value| value.is_finite())
        .collect()
}

pub(crate) fn fill_missing_by_sample<F>(
    proteins: &[ProteinId],
    samples: &[SampleId],
    fills: &HashMap<SampleId, f64>,
    value_at: &mut F,
) -> Vec<(ProteinId, SampleId, f64)>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let mut imputed = Vec::new();
    for protein in proteins {
        for sample in samples {
            if !has_finite_value(*protein, *sample, value_at)
                && let Some(fill) = fills.get(sample).copied()
            {
                imputed.push((*protein, *sample, fill));
            }
        }
    }
    imputed
}

pub(crate) fn has_finite_value<F>(protein: ProteinId, sample: SampleId, value_at: &mut F) -> bool
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    value_at(protein, sample).is_some_and(f64::is_finite)
}
