//! Precision-weighted DEqMS (`--de-precision-weights`).

use mokume_core::stats::{median_finite, quantile_linear};

use super::super::limma::run_weighted_limma_moderation;
use super::{aligned_counts, deqms_results, spectra_count_ebayes};
use crate::de::DeResult;

/// Two-group log2 rows, group A in the first `n_a` columns and group B in the
/// next `n_b`, with the quantification precision (1/SE^2) of each cell.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionRows<'a> {
    pub rows: &'a [&'a [f64]],
    pub precision: &'a [&'a [f64]],
    pub n_a: usize,
    pub n_b: usize,
}

/// [`deqms_two_group`](super::deqms_two_group) with precision-weighted protein values.
///
/// A cell without a precision (zero or NaN, e.g. an imputed value) takes the
/// 5th percentile of the contrast's observed precisions. Each weight is then
/// floored as `1/(1/w + tau2)`, `tau2` being the median observed `1/w`, so the
/// biological variance bounds how much one well-measured cell can count. The
/// weighted cell-means fit feeds the same count-aware moderation.
pub fn deqms_two_group_weighted(
    proteins: &[String],
    data: PrecisionRows<'_>,
    counts: &[f64],
    fdr_threshold: f64,
    log2fc_threshold: f64,
) -> Vec<DeResult> {
    let weights = precision_weights(data.rows, data.precision);
    let weight_refs = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (moderations, df_prior) =
        run_weighted_limma_moderation(data.rows, &weight_refs, data.n_a, data.n_b);
    if moderations.is_empty() {
        return Vec::new();
    }
    let kept_counts = aligned_counts(&moderations, counts);
    let stats = spectra_count_ebayes(&moderations, &kept_counts, df_prior);
    deqms_results(
        proteins,
        moderations.into_iter().zip(stats),
        fdr_threshold,
        log2fc_threshold,
    )
}

/// Observation weights of [`deqms_two_group_weighted`].
fn precision_weights(rows: &[&[f64]], precision: &[&[f64]]) -> Vec<Vec<f64>> {
    let measured =
        |value: f64, weight: f64| value.is_finite() && weight.is_finite() && weight > 0.0;
    let mut observed = rows
        .iter()
        .zip(precision)
        .flat_map(|(row, weights)| row.iter().zip(weights.iter()))
        .filter(|(value, weight)| measured(**value, **weight))
        .map(|(_, weight)| *weight)
        .collect::<Vec<_>>();
    let mut variances = observed
        .iter()
        .map(|weight| 1.0 / weight)
        .collect::<Vec<_>>();
    let tau2 = median_finite(&mut variances).unwrap_or(0.0);
    let unmeasured = quantile_linear(&mut observed, 0.05).unwrap_or(1.0);
    rows.iter()
        .zip(precision)
        .map(|(row, weights)| {
            row.iter()
                .zip(weights.iter())
                .map(|(value, weight)| {
                    let weight = if measured(*value, *weight) {
                        *weight
                    } else {
                        unmeasured
                    };
                    1.0 / (1.0 / weight + tau2)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{deqms_two_group_weighted, precision_weights, PrecisionRows};
    use crate::de::deqms::deqms_two_group;
    use crate::de::{DEFAULT_FDR_THRESHOLD, DEFAULT_LOG2FC_THRESHOLD};

    fn fixture() -> (Vec<String>, Vec<Vec<f64>>, Vec<f64>) {
        let rows = (0..15)
            .map(|index| {
                let base = 5.0 + index as f64;
                let shift = if index % 3 == 0 { 1.5 } else { 0.0 };
                (0..6)
                    .map(|column| {
                        let noise = 0.1 * (((index * 7 + column * 5) % 11) as f64 - 5.0);
                        base + noise + if column < 3 { shift } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        let proteins = (0..15).map(|index| format!("P{index:02}")).collect();
        let counts = (0..15).map(|index| (1 + index % 7) as f64).collect();
        (proteins, rows, counts)
    }

    // Precision 2 everywhere floors to weight 1/(1/2 + 1/2) = 1 exactly, so the
    // weighted fit must reproduce unweighted DEqMS bit for bit.
    #[test]
    fn unit_precision_weights_reproduce_deqms() {
        let (proteins, mut rows, counts) = fixture();
        rows[3][1] = f64::NAN;
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let precision = vec![vec![2.0; 6]; rows.len()];
        let precision_refs = precision.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let thresholds = (DEFAULT_FDR_THRESHOLD, DEFAULT_LOG2FC_THRESHOLD);

        let plain = deqms_two_group(&proteins, &refs, 3, 3, &counts, thresholds.0, thresholds.1);
        let data = PrecisionRows {
            rows: &refs,
            precision: &precision_refs,
            n_a: 3,
            n_b: 3,
        };
        let weighted =
            deqms_two_group_weighted(&proteins, data, &counts, thresholds.0, thresholds.1);
        assert_eq!(format!("{plain:?}"), format!("{weighted:?}"));
    }

    // Unmeasured cells (no precision, or a missing value) take the 5th
    // percentile of the observed precisions; every weight is then floored by
    // the median observed variance.
    #[test]
    fn precision_weights_fill_unmeasured_cells_and_floor() {
        let rows = [vec![1.0, 2.0, 3.0, f64::NAN]];
        let precision = [vec![1.0, 4.0, 0.0, 4.0]];
        let rows = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let precision = precision.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let tau2 = (1.0 + 0.25) / 2.0;
        let unmeasured = 1.0 + (4.0 - 1.0) * 0.05;
        let expected = [1.0, 4.0, unmeasured, unmeasured].map(|w: f64| 1.0 / (1.0 / w + tau2));
        let weights = precision_weights(&rows, &precision);
        for (weight, expected) in weights[0].iter().zip(expected) {
            assert!((weight - expected).abs() < 1e-12, "{weight} != {expected}");
        }
    }
}
