//! DEqMS count-aware moderated t-test for an unpaired two-group design.
//!
//! Count variance moderation follows DEqMS::spectraCounteBayes(fit.method="loess"):
//! R Gaussian degree-2 LOESS (span .75, interpolated surface), prior-df grid,
//! posterior variance and two-sided Student-t probabilities. Mokume applies BH
//! to the count-moderated p-values. Positive varying counts exercise DEqMS;
//! constant counts retain the existing explicitly documented limma fallback.

#[path = "deqms_loess.rs"]
mod loess;

use super::correct::bh_adjust;
use super::limma::{run_limma_moderation, LimmaModeration};
use super::special::{digamma, trigamma};
use super::student_t::{student_t_sf, student_t_two_sided_log_pvalue};
use super::{classify, DeResult};

/// Minimum number of valid (finite `log_var`, `x`, `df`) proteins required for
/// the quadratic count path (floor(.75*n) must contain at least three points).
const MIN_VALID_POINTS: usize = 4;

/// Minimum spread of `log2(counts)` over the valid points for the count path;
/// below this all counts are effectively identical and the method falls back to
/// plain eBayes (deqms.py:76).
const MIN_COUNT_SPREAD: f64 = 1e-10;

/// Two-group DEqMS differential expression.
///
/// `proteins[i]` labels `rows[i]`; each row holds log2 intensities with the
/// first `n_a` columns in condition A and the next `n_b` in condition B.
/// `counts[i]` is the per-protein peptide/spectra count aligned to `rows[i]`
/// (Python's `_build_count_vector` default is all-ones, which triggers the
/// all-equal fallback to plain eBayes). Returns one [`DeResult`] per testable
/// protein, sorted by adjusted p-value ascending (BH), reproducing mokume's
/// `DifferentialExpression(method="deqms")` output contract: `pvalue` and
/// `adj_pvalue` carry the spectra-count-moderated `sca_pvalue` / its BH
/// adjustment, and `t_statistic` carries `sca_t`.
pub fn deqms_two_group(
    proteins: &[String],
    rows: &[&[f64]],
    n_a: usize,
    n_b: usize,
    counts: &[f64],
    fdr_threshold: f64,
    log2fc_threshold: f64,
) -> Vec<DeResult> {
    let (moderations, df_prior) = run_limma_moderation(rows, n_a, n_b);
    if moderations.is_empty() {
        return Vec::new();
    }

    // Align counts to the kept proteins by original row index (defaulting to 1,
    // matching `_build_count_vector`'s default fill).
    let kept_counts = moderations
        .iter()
        .map(|(index, _)| counts.get(*index).copied().unwrap_or(1.0))
        .collect::<Vec<_>>();

    let stats = spectra_count_ebayes(&moderations, &kept_counts, df_prior);

    let p_values = stats.iter().map(|stat| stat.sca_pvalue).collect::<Vec<_>>();
    let adjusted = bh_adjust(&p_values);

    let mut results = moderations
        .iter()
        .zip(&stats)
        .zip(adjusted)
        .filter_map(|(((index, moderation), stat), adj_p_value)| {
            let protein = proteins.get(*index)?.clone();
            let significance = classify(
                adj_p_value,
                moderation.log2_fold_change,
                fdr_threshold,
                log2fc_threshold,
            );
            Some(DeResult {
                protein,
                log2_fold_change: moderation.log2_fold_change,
                p_value: stat.sca_pvalue,
                log_p_value: stat.log_pvalue,
                adj_p_value,
                t_statistic: stat.sca_t,
                // DEqMS emits no AveExpr/B columns, so these limma-only quirks
                // are filled with NaN/0.0 and must not be written for deqms.
                ave_expr: f64::NAN,
                b: 0.0,
                mean_a: moderation.mean_a,
                mean_b: moderation.mean_b,
                n_a: moderation.count_a,
                n_b: moderation.count_b,
                significance,
            })
        })
        .collect::<Vec<_>>();

    results.sort_by(|left, right| left.adj_p_value.total_cmp(&right.adj_p_value));
    results
}

/// One protein's spectraCounteBayes output: the count-moderated t-statistic and
/// its two-sided p-value (deqms.py returns `(sca_t, sca_pvalue)` as columns).
struct ScaStat {
    sca_t: f64,
    sca_pvalue: f64,
    log_pvalue: f64,
}

struct SpectraInputs {
    sigma2: Vec<f64>,
    log_var: Vec<f64>,
    df: Vec<f64>,
    df_valid: Vec<f64>,
    x: Vec<f64>,
    valid: Vec<bool>,
}

/// Port of `_spectra_count_ebayes` (deqms.py:43). Applies the count-aware Bayes
/// moderation to the kept-protein limma fits; falls back to the standard eBayes
/// statistics (`t_statistic`, `p_value`) when the count path is not exercised.
fn spectra_count_ebayes(
    moderations: &[(usize, LimmaModeration)],
    counts: &[f64],
    df_prior: f64,
) -> Vec<ScaStat> {
    let n_genes = moderations.len();
    let inputs = spectra_inputs(moderations, counts);
    let valid_count = inputs.valid.iter().filter(|flag| **flag).count();
    if valid_count < MIN_VALID_POINTS {
        return fallback(moderations, df_prior);
    }
    let x_valid = (0..n_genes)
        .filter(|i| inputs.valid[*i])
        .map(|i| inputs.x[i])
        .collect::<Vec<_>>();
    if ptp(&x_valid) < MIN_COUNT_SPREAD {
        return fallback(moderations, df_prior);
    }

    let y_pred = loess::predict(&inputs.log_var, &inputs.x, &inputs.valid);
    let (egpred, d0) = count_prior(&inputs, &y_pred);
    moderated_sca_stats(moderations, &inputs, &egpred, d0)
}

fn spectra_inputs(moderations: &[(usize, LimmaModeration)], counts: &[f64]) -> SpectraInputs {
    let n_genes = moderations.len();
    let sigma2 = moderations
        .iter()
        .map(|(_, fit)| fit.sigma * fit.sigma)
        .collect::<Vec<_>>();
    let log_var = sigma2.iter().map(|value| value.ln()).collect::<Vec<_>>();
    let df = moderations
        .iter()
        .map(|(_, fit)| fit.df_residual)
        .collect::<Vec<_>>();
    let df_valid = df
        .iter()
        .map(|value| if *value == 0.0 { f64::NAN } else { *value })
        .collect::<Vec<_>>();

    // counts_f += 1 if any count is 0 (deqms.py:60-61); x = log2(counts_f).
    let min_count = counts
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .fold(f64::INFINITY, f64::min);
    let counts_f = counts
        .iter()
        .map(|value| {
            if min_count == 0.0 {
                value + 1.0
            } else {
                *value
            }
        })
        .collect::<Vec<_>>();
    let x = counts_f
        .iter()
        .map(|value| value.log2())
        .collect::<Vec<_>>();
    let valid = (0..n_genes)
        .map(|i| log_var[i].is_finite() && x[i].is_finite() && df_valid[i].is_finite())
        .collect::<Vec<_>>();
    SpectraInputs {
        sigma2,
        log_var,
        df,
        df_valid,
        x,
        valid,
    }
}

fn count_prior(inputs: &SpectraInputs, y_pred: &[f64]) -> (Vec<f64>, f64) {
    let mut myfct_sum = 0.0;
    let mut myfct_count = 0usize;
    let mut egpred = vec![f64::NAN; inputs.df.len()];
    for i in 0..inputs.df.len() {
        if !inputs.df_valid[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        let half_df = inputs.df_valid[i] / 2.0;
        let shift = -digamma(half_df) + (half_df).ln();
        let eg_i = inputs.log_var[i] + shift;
        let egpred_i = y_pred[i] + shift;
        egpred[i] = egpred_i;
        let residual = eg_i - egpred_i;
        let myfct_i = residual * residual - trigamma(half_df);
        if myfct_i.is_finite() {
            myfct_sum += myfct_i;
            myfct_count += 1;
        }
    }
    let mean_myfct = if myfct_count > 0 {
        myfct_sum / myfct_count as f64
    } else {
        f64::NAN
    };
    (egpred, grid_search_d0(mean_myfct, inputs.df.len()))
}

fn moderated_sca_stats(
    moderations: &[(usize, LimmaModeration)],
    inputs: &SpectraInputs,
    egpred: &[f64],
    d0: f64,
) -> Vec<ScaStat> {
    let half_d0 = d0 / 2.0;
    let d0_shift = digamma(half_d0) - half_d0.ln();
    moderations
        .iter()
        .enumerate()
        .map(|(i, (_, fit))| {
            let s02 = (egpred[i] + d0_shift).exp();
            let post_var = (d0 * s02 + inputs.df[i] * inputs.sigma2[i]) / (d0 + inputs.df[i]);
            let post_df = d0 + inputs.df[i];
            let sca_t = fit.log2_fold_change / (fit.stdev_unscaled * post_var.sqrt());
            let sca_pvalue = 2.0 * student_t_sf(sca_t.abs(), post_df);
            let log_pvalue = student_t_two_sided_log_pvalue(sca_t, post_df);
            ScaStat {
                sca_t,
                sca_pvalue,
                log_pvalue,
            }
        })
        .collect()
}

/// The standard-eBayes fallback (deqms.py:72): `sca_t = t_stat`,
/// `sca_pvalue = p_value`. Used when too few valid points or all-equal counts.
fn fallback(moderations: &[(usize, LimmaModeration)], df_prior: f64) -> Vec<ScaStat> {
    let df_pooled = moderations
        .iter()
        .map(|(_, fit)| fit.df_residual)
        .sum::<f64>();
    moderations
        .iter()
        .map(|(_, fit)| ScaStat {
            sca_t: fit.t_statistic,
            sca_pvalue: fit.p_value,
            log_pvalue: student_t_two_sided_log_pvalue(
                fit.t_statistic,
                (fit.df_residual + df_prior).min(df_pooled),
            ),
        })
        .collect()
}

/// Peak-to-peak (`np.ptp`) of a slice: max minus min, or 0 when empty.
fn ptp(values: &[f64]) -> f64 {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        if *value < min {
            min = *value;
        }
        if *value > max {
            max = *value;
        }
    }
    if min.is_finite() && max.is_finite() {
        max - min
    } else {
        0.0
    }
}

/// Grid search for the prior df `d0` (port of `_grid_search_d0`, deqms.py:122).
/// Scans `test_d0 = i/10` for `i` in `1..=n_genes*10`, tracking the `test_d0`
/// whose `trigamma(test_d0/2)` is closest to `mean_myfct`, with the same
/// early break: stop once the previous-previous diff is below the previous diff
/// (after at least two steps). Defaults to `0.1` when no step improves.
fn grid_search_d0(mean_myfct: f64, n_genes: usize) -> f64 {
    let max_iter = n_genes * 10;
    let mut best_d0 = 0.1;
    let mut best_diff = f64::INFINITY;
    let mut prev_prev = f64::INFINITY;
    let mut prev = f64::INFINITY;
    for i in 1..=max_iter {
        let test_d0 = i as f64 / 10.0;
        let diff = (mean_myfct - trigamma(test_d0 / 2.0)).abs();
        if diff < best_diff {
            best_diff = diff;
            best_d0 = test_d0;
        }
        if i > 2 && prev_prev < prev {
            break;
        }
        prev_prev = prev;
        prev = diff;
    }
    best_d0
}

#[cfg(test)]
mod tests {
    use super::deqms_two_group;
    use crate::de::{
        limma_two_group, Significance, DEFAULT_FDR_THRESHOLD, DEFAULT_LOG2FC_THRESHOLD,
    };

    // 15-protein / 3-vs-3 log2 fixture shared by the oracle scripts in the
    // scratchpad (deqms_oracle.py, deqms_lowess_probe.py).
    fn fixture_rows() -> Vec<Vec<f64>> {
        vec![
            vec![10.00, 10.20, 9.80, 12.00, 12.10, 11.90],
            vec![15.00, 15.10, 14.90, 13.00, 13.20, 12.80],
            vec![8.00, 8.10, 7.90, 8.05, 7.95, 8.00],
            vec![20.00, 22.00, 18.00, 21.00, 19.00, 23.00],
            vec![5.00, 5.01, 4.99, 6.00, 6.01, 5.99],
            vec![9.00, 9.50, 8.50, 9.20, 10.50, 7.50],
            vec![11.00, 11.30, 10.70, 12.50, 12.80, 12.20],
            vec![3.00, 3.20, 2.80, 4.00, 3.90, 4.20],
            vec![7.00, 7.05, 6.95, 7.50, 7.55, 7.45],
            vec![13.00, 13.40, 12.60, 11.00, 11.20, 10.80],
            vec![6.00, 6.30, 5.70, 6.10, 6.40, 5.80],
            vec![18.00, 18.20, 17.80, 16.00, 16.30, 15.70],
            vec![4.00, 4.40, 3.60, 5.00, 5.30, 4.70],
            vec![9.50, 9.70, 9.30, 9.55, 9.45, 9.60],
            vec![12.00, 12.50, 11.50, 14.00, 14.30, 13.70],
        ]
    }

    fn fixture_proteins() -> Vec<String> {
        (0..15).map(|i| format!("P{i:02}")).collect()
    }

    fn fixture_counts() -> Vec<f64> {
        vec![
            1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 4.0, 7.0, 11.0, 6.0, 9.0, 15.0, 2.0, 18.0,
        ]
    }

    // Non-vacuity guard: the oracle sca_t must differ materially from plain limma
    // t on the SAME matrix, proving the spectraCounteBayes count path is
    // exercised (not silently equal to eBayes). The largest divergence in the
    // Python oracle is on P04 (|sca_t - limma_t| ~= 6.85).
    #[test]
    fn deqms_oracle_differs_from_plain_limma() {
        let rows = fixture_rows();
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let proteins = fixture_proteins();
        let counts = fixture_counts();

        let deqms = deqms_two_group(
            &proteins,
            &refs,
            3,
            3,
            &counts,
            DEFAULT_FDR_THRESHOLD,
            DEFAULT_LOG2FC_THRESHOLD,
        );
        let limma = limma_two_group(
            &proteins,
            &refs,
            3,
            3,
            DEFAULT_FDR_THRESHOLD,
            DEFAULT_LOG2FC_THRESHOLD,
        );

        let mut max_diff = 0.0_f64;
        for d in &deqms {
            let Some(l) = limma.iter().find(|r| r.protein == d.protein) else {
                panic!("{} missing from limma results", d.protein);
            };
            max_diff = max_diff.max((d.t_statistic - l.t_statistic).abs());
        }
        // Python oracle: max |sca_t - limma_t| ~= 6.85 (P04). Require a large
        // divergence so the count path is unmistakably exercised.
        assert!(
            max_diff > 5.0,
            "deqms vs limma max |t| divergence too small ({max_diff}); count path not exercised"
        );
    }

    // Fallback proof: all-equal counts make `ptp(log2 counts) < 1e-10`, so DEqMS
    // falls back to plain eBayes and is cell-exact (1e-9) against limma_two_group
    // on the same matrix (deqms.py:76-80).
    #[test]
    fn deqms_all_equal_counts_falls_back_to_ebayes() {
        let rows = fixture_rows();
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let proteins = fixture_proteins();
        let counts = vec![1.0; rows.len()];

        let deqms = deqms_two_group(
            &proteins,
            &refs,
            3,
            3,
            &counts,
            DEFAULT_FDR_THRESHOLD,
            DEFAULT_LOG2FC_THRESHOLD,
        );
        let limma = limma_two_group(
            &proteins,
            &refs,
            3,
            3,
            DEFAULT_FDR_THRESHOLD,
            DEFAULT_LOG2FC_THRESHOLD,
        );
        assert_eq!(deqms.len(), limma.len());

        for d in &deqms {
            let Some(l) = limma.iter().find(|r| r.protein == d.protein) else {
                panic!("{} missing from limma results", d.protein);
            };
            assert!(
                (d.log2_fold_change - l.log2_fold_change).abs() <= 1e-9,
                "{} log2FC deqms={} limma={}",
                d.protein,
                d.log2_fold_change,
                l.log2_fold_change
            );
            assert!(
                (d.t_statistic - l.t_statistic).abs() <= 1e-9,
                "{} t deqms={} limma={}",
                d.protein,
                d.t_statistic,
                l.t_statistic
            );
            assert!(
                (d.p_value - l.p_value).abs() <= 1e-9,
                "{} pvalue deqms={} limma={}",
                d.protein,
                d.p_value,
                l.p_value
            );
            assert!(
                (d.adj_p_value - l.adj_p_value).abs() <= 1e-9,
                "{} adj_pvalue deqms={} limma={}",
                d.protein,
                d.adj_p_value,
                l.adj_p_value
            );
            assert_eq!(
                d.significance, l.significance,
                "{} significance deqms vs limma mismatch",
                d.protein
            );
        }

        // Sanity: the fallback still produces a non-trivial significance call.
        assert!(deqms
            .iter()
            .any(|r| r.significance != Significance::Unchanged));
    }
}

#[cfg(test)]
#[path = "deqms_official_tests.rs"]
mod official_tests;
