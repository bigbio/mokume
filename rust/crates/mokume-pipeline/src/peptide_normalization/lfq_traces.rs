use crate::peptide_normalization::QuantilePeptideKey;
use crate::peptide_normalization::hierarchical::hierarchical_shifts_by_sample;
use crate::state::{CellKey, PeptideCellKey};
use crate::{
    HashMap, HashSet, IntoParallelRefIterator, IntoParallelRefMutIterator, Mutex, ParallelIterator,
    SampleId, StringIdRegistry,
};

pub(crate) fn apply_quantile_to_lfq_traces(
    traces: &mut HashMap<PeptideCellKey, f64>,
    allowed_cells: &HashSet<CellKey>,
) {
    let assignments =
        quantile_normalized_assignments(traces.iter().filter_map(|(key, intensity)| {
            allowed_cells
                .contains(&CellKey {
                    protein: key.protein,
                    sample: key.sample,
                })
                .then_some((
                    QuantilePeptideKey {
                        protein: key.protein,
                        peptide: key.peptide,
                    },
                    key.sample,
                    *intensity,
                ))
        }));
    for (key, intensity) in traces {
        if let Some(normalized) = assignments
            .get(&(
                QuantilePeptideKey {
                    protein: key.protein,
                    peptide: key.peptide,
                },
                key.sample,
            ))
            .copied()
        {
            *intensity = normalized;
        }
    }
}

/// DirectLFQ sample normalization on MaxLFQ's modification/charge species.
/// Every trace of a sample is scaled by that sample's shift. Python currently
/// uses an independent clustering/least-squares normalizer on the same species.
pub(crate) fn apply_hierarchical_to_lfq_traces(
    traces: &mut HashMap<PeptideCellKey, f64>,
    allowed_cells: &HashSet<CellKey>,
    sample_registry: &StringIdRegistry<SampleId>,
) {
    let allowed = |key: &PeptideCellKey| {
        allowed_cells.contains(&CellKey {
            protein: key.protein,
            sample: key.sample,
        })
    };
    let columns = log2_trace_columns(traces, |key, intensity| {
        allowed(key) && intensity.is_finite() && intensity > 0.0
    });
    if columns.is_empty() {
        return;
    }
    let shifts = hierarchical_shifts_by_sample(columns, sample_registry);
    traces.par_iter_mut().for_each(|(key, intensity)| {
        if let Some(shift) = shifts.get(&key.sample).filter(|_| allowed(key)) {
            *intensity *= shift.exp2();
        }
    });
}

/// The log2 values of the `kept` traces, one column per sample, in no
/// particular order. Every column is sized before it is filled: growing
/// per-thread columns and appending them held several copies of all values at
/// once.
fn log2_trace_columns(
    traces: &HashMap<PeptideCellKey, f64>,
    kept: impl Fn(&PeptideCellKey, f64) -> bool + Sync,
) -> HashMap<SampleId, Vec<(QuantilePeptideKey, f64)>> {
    let counts = traces
        .par_iter()
        .filter(|(key, intensity)| kept(key, **intensity))
        .fold(HashMap::<SampleId, usize>::new, |mut counts, (key, _)| {
            *counts.entry(key.sample).or_default() += 1;
            counts
        })
        .reduce(HashMap::new, |mut merged, counts| {
            for (sample, count) in counts {
                *merged.entry(sample).or_default() += count;
            }
            merged
        });
    let columns = counts
        .into_iter()
        .map(|(sample, count)| (sample, Mutex::new(Vec::with_capacity(count))))
        .collect::<HashMap<_, _>>();
    traces
        .par_iter()
        .filter(|(key, intensity)| kept(key, **intensity))
        .for_each(|(key, intensity)| {
            if let Some(Ok(mut column)) = columns.get(&key.sample).map(Mutex::lock) {
                column.push((
                    QuantilePeptideKey {
                        protein: key.protein,
                        peptide: key.peptide,
                    },
                    intensity.log2(),
                ));
            }
        });
    columns
        .into_iter()
        .filter_map(|(sample, column)| Some((sample, column.into_inner().ok()?)))
        .collect()
}

/// Quantile-normalize each sample (column) by mapping every value to a mean
/// reference distribution at the value's rank fraction. Mirrors the standard
/// quantile normalization used by limma `normalizeQuantiles` / preprocessCore:
/// columns of differing observed lengths (missing values) are handled by
/// interpolating the rank fraction in [0, 1] into the mean reference. Tied
/// values share their average-rank fraction. Returns the normalized intensity
/// per (row, sample); callers overwrite the original intensity with it.
pub(crate) fn quantile_normalized_assignments<K>(
    measurements: impl IntoIterator<Item = (K, SampleId, f64)>,
) -> HashMap<(K, SampleId), f64>
where
    K: Copy + Eq + Ord + std::hash::Hash,
{
    let mut by_sample = std::collections::BTreeMap::<SampleId, Vec<(K, f64)>>::new();
    let mut rows = HashSet::new();
    for (row, sample, intensity) in measurements {
        rows.insert(row);
        if intensity.is_finite() {
            by_sample.entry(sample).or_default().push((row, intensity));
        }
    }

    let grid_size = rows.len();
    for values in by_sample.values_mut() {
        values.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
    }
    if grid_size == 0 {
        return HashMap::new();
    }

    let reference = quantile_target(&by_sample, grid_size);

    let mut assignments = HashMap::new();
    for (sample, values) in by_sample {
        let n = values.len();
        let mut index = 0;
        while index < n {
            let mut end = index + 1;
            while end < n && values[end].1 == values[index].1 {
                end += 1;
            }
            // 1-based average rank of the tie group, mapped to a [0, 1] fraction.
            let average_rank = (index + 1 + end) as f64 / 2.0;
            let fraction = if n == 1 {
                0.0
            } else {
                (average_rank - 1.0) / (n - 1) as f64
            };
            let normalized = interpolate_sorted(&reference, fraction);
            for (row, _) in &values[index..end] {
                assignments.insert((*row, sample), normalized);
            }
            index = end;
        }
    }
    assignments
}

/// The target grid has the original row count, including entirely missing rows.
fn quantile_target<K>(
    by_sample: &std::collections::BTreeMap<SampleId, Vec<(K, f64)>>,
    grid_size: usize,
) -> Vec<f64> {
    let mut reference = vec![0.0; grid_size];
    for values in by_sample.values() {
        let sorted = values.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        for (j, slot) in reference.iter_mut().enumerate() {
            *slot += interpolate_sorted(&sorted, grid_fraction(j, grid_size));
        }
    }
    for slot in &mut reference {
        *slot /= by_sample.len() as f64;
    }
    reference
}

/// Fraction in [0, 1] of the `j`-th point on a uniform grid of `size` points.
fn grid_fraction(j: usize, size: usize) -> f64 {
    if size <= 1 {
        0.0
    } else {
        j as f64 / (size - 1) as f64
    }
}

/// Linear interpolation of an ascending-sorted slice at `fraction` in [0, 1],
/// where `fraction` maps to fractional index `fraction * (len - 1)`.
fn interpolate_sorted(sorted: &[f64], fraction: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        len => {
            let position = fraction.clamp(0.0, 1.0) * (len - 1) as f64;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            if lower == upper {
                sorted[lower]
            } else {
                let weight = position - lower as f64;
                sorted[lower] + (sorted[upper] - sorted[lower]) * weight
            }
        }
    }
}
