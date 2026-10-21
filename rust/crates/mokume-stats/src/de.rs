//! Differential expression methods.
//!
//! Each method is ported into its own file here to mirror
//! `mokume/analysis/*.py` and keep ports independently reviewable. The first
//! method to land is `limma` (the empirical-Bayes moderated t-test); `rots`,
//! `deqms`, `limrots`, `proda`, and `ensemble` follow, plus a
//! dispatcher carrying the `auto` selector. Multiple-testing correction lives
//! in `correct`, special functions in `special`, and the Student-t survival
//! function in `student_t`.

mod correct;
mod deqms;
mod effect_size;
mod ensemble;
mod ihw;
mod limma;
mod limrots;
mod port;
mod proda;
mod rots;
mod special;
mod student_t;

use correct::bh_adjust;
pub use correct::{adaptive_adjust, AdaptiveFdrMethod, AppliedFdrMethod};
pub use deqms::deqms_two_group;
pub use effect_size::{estimate_effect_size_gate, EffectSizeGateMethod};
pub use ensemble::{combine_de_results, EnsembleResult};
pub use ihw::{ihw_correction, ihw_with_options, IhwOptions, IhwResult};
use limma::run_limma;
pub use limrots::{
    limrots_two_group, limrots_two_group_with_options, LimRotsOptions, LimRotsOutput,
};
use proda::proda_run;
pub use rots::rots_two_group;

/// Direction call matching mokume's `significance` column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Significance {
    Up,
    Down,
    Unchanged,
    NotTested,
}

/// One differential-expression result row, mirroring the columns mokume's
/// `_finalize_results` emits (`B` is always 0 and `ave_expr == log2_fold_change`
/// for the limma port, reproducing the Python quirks).
#[derive(Debug, Clone)]
pub struct DeResult {
    pub protein: String,
    pub log2_fold_change: f64,
    pub p_value: f64,
    pub log_p_value: f64,
    pub adj_p_value: f64,
    pub t_statistic: f64,
    pub ave_expr: f64,
    pub b: f64,
    pub mean_a: f64,
    pub mean_b: f64,
    pub n_a: usize,
    pub n_b: usize,
    pub significance: Significance,
}

/// Default FDR and log2-fold-change thresholds for the significance call,
/// matching mokume's `_DE_OPTION_DEFAULTS`.
pub const DEFAULT_FDR_THRESHOLD: f64 = 0.05;
pub const DEFAULT_LOG2FC_THRESHOLD: f64 = 0.5;

/// Two-group limma differential expression.
///
/// `proteins[i]` labels `rows[i]`; each row holds log2 intensities with the
/// first `n_a` columns in condition A and the next `n_b` in condition B.
/// Returns one [`DeResult`] per testable protein, sorted by adjusted p-value
/// ascending (BH), reproducing mokume's `DifferentialExpression(method="limma")`
/// output contract.
pub fn limma_two_group(
    proteins: &[String],
    rows: &[&[f64]],
    n_a: usize,
    n_b: usize,
    fdr_threshold: f64,
    log2fc_threshold: f64,
) -> Vec<DeResult> {
    let stats = run_limma(rows, n_a, n_b);
    let p_values = stats
        .iter()
        .map(|(_, stat)| stat.p_value)
        .collect::<Vec<_>>();
    let adjusted = bh_adjust(&p_values);

    let mut results = stats
        .iter()
        .zip(adjusted)
        .filter_map(|((index, stat), adj_p_value)| {
            let protein = proteins.get(*index)?.clone();
            let significance = classify(
                adj_p_value,
                stat.log2_fold_change,
                fdr_threshold,
                log2fc_threshold,
            );
            Some(DeResult {
                protein,
                log2_fold_change: stat.log2_fold_change,
                p_value: stat.p_value,
                log_p_value: stat.p_value.ln(),
                adj_p_value,
                t_statistic: stat.t_statistic,
                ave_expr: stat.ave_expr,
                b: 0.0,
                mean_a: stat.mean_a,
                mean_b: stat.mean_b,
                n_a: stat.n_a,
                n_b: stat.n_b,
                significance,
            })
        })
        .collect::<Vec<_>>();

    results.sort_by(|left, right| left.adj_p_value.total_cmp(&right.adj_p_value));
    results
}

/// Two-group proDA differential expression.
///
/// Same call shape as [`limma_two_group`]: `proteins[i]` labels `rows[i]`; the
/// first `n_a` columns are condition A and the next `n_b` condition B. Returns
/// one [`DeResult`] per testable protein (>= 2 finite observations in each
/// group, matching `filter_testable`), sorted by adjusted p-value ascending
/// (BH), reproducing mokume's `DifferentialExpression(method="proda")` output.
///
/// The proDA `t_stat` (the Wald statistic) is carried in `t_statistic` as the
/// single extra column the pipeline writes; `ave_expr`/`b` are limma-only and
/// filled with `NaN`/`0.0`. `mean_a`/`mean_b` are the per-group nan-means and
/// `n_a`/`n_b` the per-group finite counts (mokume's `per_group_summary`), which
/// can differ from the EM `n_obs`.
pub fn proda_two_group(
    proteins: &[String],
    rows: &[&[f64]],
    n_a: usize,
    n_b: usize,
    fdr_threshold: f64,
    log2fc_threshold: f64,
) -> Vec<DeResult> {
    let stats = proda_run(rows, n_a, n_b);

    // Apply the testability filter and collect the surviving rows' p-values for
    // a SINGLE BH adjustment over the kept set, matching Python's
    // `raw[raw.ProteinName.isin(testable.index)]` before `bh_adjust`.
    let kept: Vec<usize> = (0..rows.len())
        .filter(|&i| {
            let row = rows[i];
            let (_, na) = group_summary(row.get(..n_a).unwrap_or(&[]));
            let (_, nb) = group_summary(row.get(n_a..n_a + n_b).unwrap_or(&[]));
            na >= MIN_PER_GROUP_PRODA && nb >= MIN_PER_GROUP_PRODA
        })
        .collect();

    let p_values: Vec<f64> = kept.iter().map(|&i| stats[i].p_value).collect();
    let adjusted = bh_adjust(&p_values);

    let mut results = kept
        .iter()
        .zip(adjusted)
        .filter_map(|(&index, adj_p_value)| {
            let protein = proteins.get(index)?.clone();
            let stat = &stats[index];
            let row = rows[index];
            let (sum_a, na) = group_summary(row.get(..n_a).unwrap_or(&[]));
            let (sum_b, nb) = group_summary(row.get(n_a..n_a + n_b).unwrap_or(&[]));
            let mean_a = if na > 0 { sum_a / na as f64 } else { f64::NAN };
            let mean_b = if nb > 0 { sum_b / nb as f64 } else { f64::NAN };
            let significance = classify(
                adj_p_value,
                stat.log2_fold_change,
                fdr_threshold,
                log2fc_threshold,
            );
            Some(DeResult {
                protein,
                log2_fold_change: stat.log2_fold_change,
                p_value: stat.p_value,
                log_p_value: stat.p_value.ln(),
                adj_p_value,
                t_statistic: stat.t_stat,
                ave_expr: f64::NAN,
                b: 0.0,
                mean_a,
                mean_b,
                n_a: na,
                n_b: nb,
                significance,
            })
        })
        .collect::<Vec<_>>();

    results.sort_by(|left, right| left.adj_p_value.total_cmp(&right.adj_p_value));
    results
}

/// Minimum finite observations per group for a protein to be tested by proDA
/// (`filter_testable(min_per_group=2)`).
const MIN_PER_GROUP_PRODA: usize = 2;

/// `(sum, count)` over the finite entries of a group slice, for the per-group
/// nan-mean and finite-count columns.
fn group_summary(values: &[f64]) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in values {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    (sum, count)
}

fn classify(
    adj_p_value: f64,
    log2_fold_change: f64,
    fdr_threshold: f64,
    log2fc_threshold: f64,
) -> Significance {
    if !adj_p_value.is_finite() || !log2_fold_change.is_finite() {
        return Significance::NotTested;
    }
    if adj_p_value < fdr_threshold && log2_fold_change > log2fc_threshold {
        Significance::Up
    } else if adj_p_value < fdr_threshold && log2_fold_change < -log2fc_threshold {
        Significance::Down
    } else {
        Significance::Unchanged
    }
}

#[cfg(test)]
mod tests {
    use super::{
        classify, limma_two_group, proda_two_group, Significance, DEFAULT_FDR_THRESHOLD,
        DEFAULT_LOG2FC_THRESHOLD,
    };

    // Independent Bioconductor proDA 1.24.0 fixture. Complete input avoids
    // cross-language initialization RNG differences while testing the public API.
    #[test]
    fn proda_public_entrypoint_matches_official_complete_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/data/proda_official.json"))
                .unwrap_or_else(|e| panic!("invalid official fixture: {e}"));
        let case = &fixture["cases"][0];
        let data = case["matrix"]
            .as_array()
            .unwrap_or_else(|| panic!("fixture array missing"))
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap_or_else(|| panic!("fixture array missing"))
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .unwrap_or_else(|| panic!("fixture number missing"))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let ids = (0..data.len()).map(|i| i.to_string()).collect::<Vec<_>>();
        let refs = data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let results = proda_two_group(&ids, &refs, 4, 4, 0.05, 0.5);
        assert_eq!(results.len(), data.len());
        for row in results {
            let i = row
                .protein
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("fixture index: {e}"));
            for (key, a) in [
                ("log2FC", row.log2_fold_change),
                ("statistic", row.t_statistic),
                ("pvalue", row.p_value),
                ("adjusted_pvalue", row.adj_p_value),
            ] {
                let b = case["expected"][key][i]
                    .as_f64()
                    .unwrap_or_else(|| panic!("fixture number missing"));
                assert!(
                    (a - b).abs() <= 1e-10 + 1e-6 * b.abs(),
                    "protein {i}, {key}: {a} != {b}"
                );
            }
        }
    }

    // A non-estimable fold change or adjusted p-value is NotTested, not evidence
    // that the protein is unchanged.
    #[test]
    fn classify_marks_non_finite_results_not_tested() {
        for log2fc in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                classify(1e-9, log2fc, 0.05, 0.5),
                Significance::NotTested,
                "non-finite log2FC {log2fc} should be NotTested"
            );
        }
        assert_eq!(classify(f64::NAN, 2.0, 0.05, 0.5), Significance::NotTested);
        // Sanity: a finite, significant, above-threshold fold change still calls UP.
        assert_eq!(classify(1e-9, 2.0, 0.05, 0.5), Significance::Up);
    }

    // End-to-end limma DE on the oracle matrix: checks BH adjustment,
    // significance calls, and the adjusted-p sort order against mokume's
    // `DifferentialExpression(method="limma").run()` final table.
    #[test]
    fn limma_two_group_matches_oracle_table() {
        let proteins = ["P1", "P2", "P3", "P4", "P5", "P6"]
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        let rows: Vec<Vec<f64>> = vec![
            vec![10.0, 10.2, 9.8, 12.0, 12.1, 11.9],
            vec![15.0, 15.1, 14.9, 13.0, 13.2, 12.8],
            vec![8.0, 8.1, 7.9, 8.05, 7.95, 8.0],
            vec![20.0, 22.0, 18.0, 21.0, 19.0, 23.0],
            vec![5.0, 5.01, 4.99, 6.0, 6.01, 5.99],
            vec![9.0, 9.5, 8.5, 9.2, 10.5, 7.5],
        ];
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let results = limma_two_group(
            &proteins,
            &refs,
            3,
            3,
            DEFAULT_FDR_THRESHOLD,
            DEFAULT_LOG2FC_THRESHOLD,
        );

        // Sorted by adjusted p-value: P5, P1, P2, then the unchanged proteins.
        assert_eq!(results[0].protein, "P5");
        assert_eq!(results[0].significance, Significance::Down);
        assert!((results[0].adj_p_value - 1.0275419169449564e-06).abs() <= 1e-12);

        let Some(p1) = results.iter().find(|r| r.protein == "P1") else {
            panic!("P1 missing from results");
        };
        assert_eq!(p1.significance, Significance::Down);
        assert!((p1.adj_p_value - 6.346861792017283e-05).abs() <= 1e-12);

        let Some(p2) = results.iter().find(|r| r.protein == "P2") else {
            panic!("P2 missing from results");
        };
        assert_eq!(p2.significance, Significance::Up);

        let Some(p3) = results.iter().find(|r| r.protein == "P3") else {
            panic!("P3 missing from results");
        };
        assert_eq!(p3.significance, Significance::Unchanged);
    }

    // End-to-end limma DE on an NA-CONTAINING matrix where the per-protein
    // residual df varies (df in {2, 3, 4}), exercising the unequal-df1 maximum
    // likelihood `fitFDist` (with its bounded Brent optimiser) rather than the
    // legacy equal-df path. Asserts log2FC / pvalue / adj_pvalue / significance
    // per protein against mokume's
    // `DifferentialExpression(method="limma", skip_log2=True).run()`.
    //
    // Oracle (verbatim):
    //   conda run -n Bigbio python  # see scratchpad oracle.py
    // on the 8-protein, 3-vs-3 log2 matrix below with NaN cells scattered so
    // n_a/n_b differ across proteins. The whole pipeline (log2FC, t, p, BH
    // adjust, significance) is cell-exact to 1e-9 because the bounded Brent port
    // reproduces SciPy's `_minimize_scalar_bounded` step for step and the
    // objective's `math.lgamma`/`math.log1p` agree bit-for-bit with SciPy's
    // `gammaln`/`np.log1p` on these inputs (verified: optimiser `par` differs by
    // exactly 0.0).
    #[test]
    fn limma_two_group_with_missing_values_matches_oracle_table() {
        const NAN: f64 = f64::NAN;
        let proteins = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        let rows: Vec<Vec<f64>> = vec![
            vec![10.0, 10.2, 9.8, 12.0, 12.1, 11.9], // df 4, full
            vec![15.0, 15.1, NAN, 13.0, 13.2, 12.8], // df 3, missing in A
            vec![8.0, 8.1, 7.9, 8.05, NAN, 8.0],     // df 3, missing in B
            vec![20.0, 22.0, NAN, 21.0, NAN, 23.0],  // df 2, one missing each
            vec![5.0, 5.01, 4.99, 6.0, 6.01, 5.99],  // df 4, full
            vec![9.0, 9.5, 8.5, 9.2, 10.5, 7.5],     // df 4, full
            vec![11.0, NAN, 10.5, 12.0, 12.3, NAN],  // df 2, one missing each
            vec![3.0, 3.2, 2.8, 4.0, NAN, 4.2],      // df 3, missing in B
        ];
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let results = limma_two_group(
            &proteins,
            &refs,
            3,
            3,
            DEFAULT_FDR_THRESHOLD,
            DEFAULT_LOG2FC_THRESHOLD,
        );
        assert_eq!(results.len(), 8);

        // (protein, log2FC, pvalue, adj_pvalue, significance).
        let oracle: [(&str, f64, f64, f64, Significance); 8] = [
            (
                "P0",
                -2.0,
                3.716_779_595_593_482_4e-6,
                2.236_455_780_286_270_5e-5,
                Significance::Down,
            ),
            (
                "P1",
                2.050_000_000_000_000_7,
                3.100_836_190_996_106_4e-5,
                8.268_896_509_322_95e-5,
                Significance::Up,
            ),
            (
                "P2",
                -0.025_000_000_000_000_355,
                0.814_039_992_891_240_4,
                0.930_331_420_447_131_9,
                Significance::Unchanged,
            ),
            (
                "P3",
                -1.0,
                0.376_020_843_169_737_66,
                0.501_361_124_226_316_8,
                Significance::Unchanged,
            ),
            (
                "P4",
                -1.0,
                5.591_139_450_715_676e-6,
                2.236_455_780_286_270_5e-5,
                Significance::Down,
            ),
            (
                "P5",
                -0.066_666_666_666_666_43,
                0.932_088_898_901_693_8,
                0.932_088_898_901_693_8,
                Significance::Unchanged,
            ),
            (
                "P6",
                -1.400_000_000_000_000_4,
                0.003_623_737_568_954_826_5,
                0.005_797_980_110_327_722,
                Significance::Down,
            ),
            (
                "P7",
                -1.1,
                0.000_801_088_550_466_308_3,
                0.001_602_177_100_932_616_6,
                Significance::Down,
            ),
        ];

        for (protein, log2fc, pvalue, adj_pvalue, significance) in oracle {
            let Some(row) = results.iter().find(|r| r.protein == protein) else {
                panic!("{protein} missing from results");
            };
            assert!(
                (row.log2_fold_change - log2fc).abs() <= 1e-9,
                "{protein} log2FC actual={} expected={log2fc}",
                row.log2_fold_change
            );
            assert!(
                (row.p_value - pvalue).abs() <= 1e-9,
                "{protein} pvalue actual={} expected={pvalue}",
                row.p_value
            );
            assert!(
                (row.adj_p_value - adj_pvalue).abs() <= 1e-9,
                "{protein} adj_pvalue actual={} expected={adj_pvalue}",
                row.adj_p_value
            );
            assert_eq!(
                row.significance, significance,
                "{protein} significance mismatch"
            );
        }
    }
}
