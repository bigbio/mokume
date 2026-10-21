//! Multi-method ensemble for two-group differential expression.
//!
//! This is a port of mokume's `analysis/ensemble.py`. The deterministic core is
//! `combine_de_results` (ensemble.py:94): it runs several DE methods on the same
//! contrast, then fuses their per-protein calls with a top-k consensus rule and
//! Fisher's combined p-value. Only this combine step lives here; the member
//! methods (`limrots`/`deqms`/`proda`/...) are run by the caller and passed in.
//!
//! DETERMINISM: the combine step is fully deterministic and cell-exact against
//! the Python `combine_de_results` oracle (see the golden test). The end-to-end
//! ensemble inherits the RNG/optimizer non-determinism of whichever members it
//! runs (rots/limrots are RNG-based, proda is optimizer-driven), exactly as
//! documented for those methods; a fully-deterministic member subset
//! (limma/deqms) keeps the whole chain reproducible.
//!
//! The flow mirrors `combine_de_results` operation-by-operation:
//!   - union of all proteins across members, iterated in sorted order
//!     (ensemble.py:120-125);
//!   - per protein, collect each member's `log2FC` and (finite, > 0) `pvalue`,
//!     count `n_up`/`n_down` from each member's `significance`, and append the
//!     contributing method name (ensemble.py:126-148);
//!   - `median_fc = median(fc_values)` (numpy median: mean of the two middle
//!     values for an even count), or 0.0 if no member reported the protein
//!     (ensemble.py:149);
//!   - `combined_p`: Fisher's method over the kept p-values when >= 2 remain,
//!     the single p-value when exactly 1 remains, else NaN (ensemble.py:151-156);
//!   - BH-adjust the combined p over the finite entries only (ensemble.py:174-182);
//!   - consensus significance: UP iff `n_up >= min_k AND adj_p < fdr_threshold
//!     AND log2FC > log2fc_threshold`; DOWN symmetric (ensemble.py:185-197);
//!   - sort by adjusted p-value ascending (ensemble.py:199).

use super::correct::bh_adjust;
use super::special::chi2_sf;
use super::{DeResult, Significance};

/// One combined ensemble result row. Mirrors the columns mokume's
/// `combine_de_results` emits, adding the consensus-specific `n_methods_up`,
/// `n_methods_down`, and `methods_significant` columns on top of the shared
/// `Protein`/`log2FC`/`pvalue`/`adj_pvalue`/`significance` set.
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    pub protein: String,
    /// Median of the members' `log2FC` (numpy median; 0.0 if none reported it).
    pub log2_fold_change: f64,
    /// Fisher-combined p-value (the single member p when only one is finite/> 0,
    /// NaN when none are).
    pub p_value: f64,
    /// BH adjustment of `p_value` over the finite-p subset.
    pub adj_p_value: f64,
    /// Number of members that called this protein UP.
    pub n_methods_up: usize,
    /// Number of members that called this protein DOWN.
    pub n_methods_down: usize,
    /// Comma-joined names of the members that called this protein UP or DOWN,
    /// in member-iteration order (Python's `",".join(contributing)`).
    pub methods_significant: String,
    /// Consensus direction call.
    pub significance: Significance,
}

/// Combine per-method DE results with top-k consensus, porting
/// `combine_de_results` (ensemble.py:94).
///
/// `members` is the ordered list of `(method_name, results)` pairs (Python's
/// `dict[str, DataFrame]`, whose insertion order `combine_de_results` iterates);
/// only members that produced a non-empty result should be passed, matching
/// `run_ensemble`'s `if not result.empty` filter (ensemble.py:76). `min_k`,
/// `log2fc_threshold`, and `fdr_threshold` are the consensus thresholds. Returns
/// the combined rows sorted by adjusted p-value ascending.
pub fn combine_de_results(
    members: &[(String, Vec<DeResult>)],
    min_k: usize,
    log2fc_threshold: f64,
    fdr_threshold: f64,
) -> Vec<EnsembleResult> {
    // Union of all proteins across members, iterated in sorted order
    // (Python `sorted(all_proteins)`).
    let mut proteins: Vec<&str> = members
        .iter()
        .flat_map(|(_, rows)| rows.iter().map(|row| row.protein.as_str()))
        .collect();
    proteins.sort_unstable();
    proteins.dedup();

    // Per-protein accumulation, in the sorted protein order.
    let mut combined: Vec<Combined> = Vec::with_capacity(proteins.len());
    for protein in proteins {
        combined.push(accumulate_protein(protein, members));
    }

    // BH adjustment over the finite combined p-values only (Python masks with
    // `np.isfinite` before `multipletests`); non-finite entries stay NaN.
    let p_values: Vec<f64> = combined.iter().map(|c| c.combined_p).collect();
    let adjusted = bh_adjust(&p_values);

    let thresholds = Thresholds {
        min_k,
        log2fc_threshold,
        fdr_threshold,
    };
    let mut results: Vec<EnsembleResult> = combined
        .into_iter()
        .zip(adjusted)
        .map(|(c, adj_p_value)| {
            let significance = consensus(c.n_up, c.n_down, c.median_fc, adj_p_value, &thresholds);
            EnsembleResult {
                protein: c.protein,
                log2_fold_change: c.median_fc,
                p_value: c.combined_p,
                adj_p_value,
                n_methods_up: c.n_up,
                n_methods_down: c.n_down,
                methods_significant: c.contributing.join(","),
                significance,
            }
        })
        .collect();

    // Sort by adjusted p-value ascending (Python `sort_values("adj_pvalue")`).
    // `total_cmp` keeps NaN entries last and stable, matching pandas' default
    // `na_position="last"`.
    results.sort_by(|left, right| left.adj_p_value.total_cmp(&right.adj_p_value));
    results
}

/// Per-protein accumulation across members, before BH adjustment.
struct Combined {
    protein: String,
    median_fc: f64,
    combined_p: f64,
    n_up: usize,
    n_down: usize,
    contributing: Vec<String>,
}

/// Collect one protein's fold changes, p-values, direction counts, and combine
/// the p-values (ensemble.py:126-167).
fn accumulate_protein(protein: &str, members: &[(String, Vec<DeResult>)]) -> Combined {
    let mut fc_values: Vec<f64> = Vec::new();
    let mut p_values: Vec<f64> = Vec::new();
    let mut n_up = 0usize;
    let mut n_down = 0usize;
    let mut contributing: Vec<String> = Vec::new();

    for (method, rows) in members {
        // Python takes `match.iloc[0]`: the FIRST row for this protein. Member
        // results are unique per protein, but mirror the first-match rule.
        let Some(row) = rows.iter().find(|row| row.protein == protein) else {
            continue;
        };
        if row.significance == Significance::NotTested || !row.log2_fold_change.is_finite() {
            continue;
        }
        fc_values.push(row.log2_fold_change);
        let pval = row.p_value;
        if pval.is_finite() && pval > 0.0 {
            p_values.push(pval);
        }
        match row.significance {
            Significance::Up => {
                n_up += 1;
                contributing.push(method.clone());
            }
            Significance::Down => {
                n_down += 1;
                contributing.push(method.clone());
            }
            Significance::Unchanged | Significance::NotTested => {}
        }
    }

    let median_fc = if fc_values.is_empty() {
        f64::NAN
    } else {
        median(&mut fc_values)
    };
    let combined_p = combine_p_values(&p_values);

    Combined {
        protein: protein.to_owned(),
        median_fc,
        combined_p,
        n_up,
        n_down,
        contributing,
    }
}

/// Combine kept p-values (ensemble.py:151-156): Fisher's method when >= 2
/// remain, passthrough when exactly 1, NaN when none. Fisher's statistic is
/// `-2 sum ln p`, chi-squared on `2k` df, so the combined p is `chi2.sf(stat,
/// 2k)` -- exactly what `scipy.stats.combine_pvalues(..., method="fisher")`
/// returns as its second element.
fn combine_p_values(p_values: &[f64]) -> f64 {
    match p_values.len() {
        0 => f64::NAN,
        1 => p_values[0],
        k => {
            let stat: f64 = -2.0 * p_values.iter().map(|p| p.ln()).sum::<f64>();
            chi2_sf(stat, 2.0 * k as f64)
        }
    }
}

/// numpy-compatible median over a non-empty slice: for an even count it is the
/// mean of the two middle order statistics (Python `np.median`). Sorts in place
/// with `total_cmp`; the member fold changes are finite in every wired path.
pub(super) fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mid = n / 2;
    if n % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

/// The three consensus thresholds, grouped so [`consensus`] stays a small,
/// readable signature.
struct Thresholds {
    min_k: usize,
    log2fc_threshold: f64,
    fdr_threshold: f64,
}

/// Consensus direction call (ensemble.py:185-197). UP requires at least `min_k`
/// members UP AND a significant adjusted p AND a fold change above the positive
/// threshold; DOWN is the symmetric negative-threshold rule. A row without a
/// finite fold change and adjusted p-value is NotTested; other rows are
/// Unchanged when neither direction passes.
fn consensus(
    n_up: usize,
    n_down: usize,
    log2_fold_change: f64,
    adj_p_value: f64,
    thresholds: &Thresholds,
) -> Significance {
    let Thresholds {
        min_k,
        log2fc_threshold,
        fdr_threshold,
    } = *thresholds;
    if !adj_p_value.is_finite() || !log2_fold_change.is_finite() {
        Significance::NotTested
    } else if n_up >= min_k && adj_p_value < fdr_threshold && log2_fold_change > log2fc_threshold {
        Significance::Up
    } else if n_down >= min_k && adj_p_value < fdr_threshold && log2_fold_change < -log2fc_threshold
    {
        Significance::Down
    } else {
        Significance::Unchanged
    }
}

#[cfg(test)]
mod tests {
    use super::{combine_de_results, median};
    use crate::de::{DeResult, Significance};

    /// One hardcoded oracle row: protein, median log2FC, Fisher combined p, BH
    /// adjusted p, n_up, n_down, methods_significant, consensus significance.
    type OracleRow = (
        &'static str,
        f64,
        f64,
        f64,
        usize,
        usize,
        &'static str,
        &'static str,
    );

    fn de_row(protein: &str, log2fc: f64, pvalue: f64, sig: Significance) -> DeResult {
        DeResult {
            protein: protein.to_owned(),
            log2_fold_change: log2fc,
            p_value: pvalue,
            log_p_value: pvalue.ln(),
            adj_p_value: f64::NAN, // unused by the combiner (it reads pvalue).
            t_statistic: f64::NAN,
            ave_expr: f64::NAN,
            b: 0.0,
            mean_a: f64::NAN,
            mean_b: f64::NAN,
            n_a: 0,
            n_b: 0,
            significance: sig,
        }
    }

    fn sig_label(sig: Significance) -> &'static str {
        match sig {
            Significance::Up => "UP",
            Significance::Down => "DOWN",
            Significance::Unchanged => "Unchanged",
            Significance::NotTested => "NotTested",
        }
    }

    // numpy median quirk: even count -> mean of the two middle values.
    #[test]
    fn median_matches_numpy_even_and_odd() {
        let mut odd = [-0.8, -1.0, -0.9];
        assert!((median(&mut odd) - (-0.9)).abs() <= 1e-12);
        let mut even = [-2.5, -2.0];
        assert!((median(&mut even) - (-2.25)).abs() <= 1e-12);
        let mut single = [1.9];
        assert!((median(&mut single) - 1.9).abs() <= 1e-12);
    }

    #[test]
    fn ensemble_preserves_not_tested_semantics() {
        let members = vec![(
            "limma".to_owned(),
            vec![de_row("P1", f64::NAN, f64::NAN, Significance::NotTested)],
        )];

        let result = combine_de_results(&members, 1, 0.5, 0.05);

        assert_eq!(result.len(), 1);
        assert!(result[0].log2_fold_change.is_nan());
        assert!(result[0].p_value.is_nan());
        assert_eq!(result[0].significance, Significance::NotTested);
    }

    // CELL-EXACT golden test for the deterministic combine_de_results core.
    //
    // Three fixed member tables (m1/m2/m3) over 8 proteins, constructed to
    // exercise every Python branch:
    //   * P8 missing from m2, P1/P3 missing from m3 (protein-union handling);
    //   * P6 has a ZERO pvalue in m1 (dropped from Fisher; combined over m2/m3);
    //   * P7 has a NaN pvalue in m1 (dropped from Fisher);
    //   * P5 has 3 fold changes (odd-count median) but only 1 DOWN call, so
    //     consensus stays Unchanged despite adj_p < 0.05 and log2FC < -0.5
    //     (exercises the min_k AND-condition);
    //   * P8 has an even-count fold-change median (-2.25);
    //   * P7/P3 contribute no significant member (empty methods_significant,
    //     NaN-free combined p but Unchanged).
    //
    // The oracle (median log2FC, Fisher combined_p, BH adj_pvalue, n_up/n_down,
    // methods_significant, consensus significance, and the adj_p sort order) is
    // a hardcoded Python `combine_de_results` run captured verbatim from
    // `conda run -n Bigbio python` (scratchpad ensemble_oracle.py). All numeric
    // cells match at 1e-9; this is the RNG-independent correctness anchor.
    #[test]
    fn combine_de_results_matches_python_oracle() {
        use Significance::{Down, Unchanged, Up};
        let m1 = vec![
            de_row("P1", 2.0, 0.001, Up),
            de_row("P2", -1.5, 0.002, Down),
            de_row("P3", 0.1, 0.5, Unchanged),
            de_row("P4", 1.2, 0.01, Up),
            de_row("P5", -0.8, 0.04, Unchanged),
            de_row("P6", 3.0, 0.0, Up),
            de_row("P7", 0.3, f64::NAN, Unchanged),
            de_row("P8", -2.5, 0.0001, Down),
        ];
        let m2 = vec![
            de_row("P1", 1.8, 0.003, Up),
            de_row("P2", -1.7, 0.001, Down),
            de_row("P3", -0.2, 0.4, Unchanged),
            de_row("P5", -1.0, 0.03, Down),
            de_row("P6", 2.5, 0.005, Up),
            de_row("P7", 0.5, 0.2, Unchanged),
        ];
        let m3 = vec![
            de_row("P2", -1.6, 0.002, Down),
            de_row("P4", 1.0, 0.02, Up),
            de_row("P5", -0.9, 0.045, Unchanged),
            de_row("P6", 2.8, 0.001, Up),
            de_row("P7", 0.2, 0.6, Unchanged),
            de_row("P8", -2.0, 0.001, Down),
        ];
        let members = vec![
            ("m1".to_owned(), m1),
            ("m2".to_owned(), m2),
            ("m3".to_owned(), m3),
        ];

        let combined = combine_de_results(&members, 2, 0.5, 0.05);

        // (protein, log2FC, pvalue, adj_pvalue, n_up, n_down, methods_sig, sig),
        // in the Python adj_pvalue-sorted order.
        let oracle: [OracleRow; 8] = [
            (
                "P2",
                -1.6,
                8.291848176171641e-07,
                6.633478540937313e-06,
                0,
                3,
                "m1,m2,m3",
                "DOWN",
            ),
            (
                "P8",
                -2.25,
                1.711809565095837e-06,
                6.847238260383348e-06,
                0,
                2,
                "m1,m3",
                "DOWN",
            ),
            (
                "P1",
                1.9,
                4.1150694807888515e-05,
                0.00010973518615436938,
                2,
                0,
                "m1,m2",
                "UP",
            ),
            (
                "P6",
                2.8,
                6.603036322765084e-05,
                0.00013206072645530168,
                3,
                0,
                "m1,m2,m3",
                "UP",
            ),
            (
                "P4",
                1.1,
                0.0019034386382832487,
                0.0030455018212531978,
                2,
                0,
                "m1,m3",
                "UP",
            ),
            (
                "P5",
                -0.9,
                0.0031917692601552707,
                0.004255692346873694,
                0,
                1,
                "m2",
                "Unchanged",
            ),
            (
                "P7",
                0.3,
                0.3744316243440109,
                0.42792185639315533,
                0,
                0,
                "",
                "Unchanged",
            ),
            (
                "P3",
                -0.05,
                0.5218875824868203,
                0.5218875824868203,
                0,
                0,
                "",
                "Unchanged",
            ),
        ];

        assert_eq!(combined.len(), oracle.len(), "row count must match oracle");
        for (row, (protein, log2fc, pvalue, adj, n_up, n_down, methods_sig, sig)) in
            combined.iter().zip(oracle)
        {
            assert_eq!(row.protein, protein, "protein order");
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
                (row.adj_p_value - adj).abs() <= 1e-9,
                "{protein} adj_pvalue actual={} expected={adj}",
                row.adj_p_value
            );
            assert_eq!(row.n_methods_up, n_up, "{protein} n_up");
            assert_eq!(row.n_methods_down, n_down, "{protein} n_down");
            assert_eq!(
                row.methods_significant, methods_sig,
                "{protein} methods_significant"
            );
            assert_eq!(sig_label(row.significance), sig, "{protein} significance");
        }
    }

    // A single finite p-value passes through unchanged (no Fisher), and a
    // protein with no finite/positive p-value gets NaN combined p and is
    // NotTested (ensemble.py:198-203, the len==1 and len==0 branches).
    #[test]
    fn single_and_zero_pvalue_branches() -> Result<(), String> {
        use Significance::{NotTested, Unchanged, Up};
        let m1 = vec![
            de_row("Q1", 2.0, 0.001, Up), // only member reporting Q1
            de_row("Q2", 1.0, 0.0, Up),   // zero p dropped -> no finite p
        ];
        let members = vec![("m1".to_owned(), m1)];
        let combined = combine_de_results(&members, 2, 0.5, 0.05);

        let q1 = combined
            .iter()
            .find(|r| r.protein == "Q1")
            .expect("Q1 present");
        // Single finite p passes through; BH over a single value leaves it.
        assert!((q1.p_value - 0.001).abs() <= 1e-12);
        // Only 1 UP member, min_k=2 -> Unchanged despite a tiny p.
        assert_eq!(q1.significance, Unchanged);

        let q2 = combined
            .iter()
            .find(|r| r.protein == "Q2")
            .expect("Q2 present");
        assert!(q2.p_value.is_nan(), "Q2 combined p should be NaN");
        assert!(q2.adj_p_value.is_nan(), "Q2 adj p should be NaN");
        assert_eq!(q2.significance, NotTested);
        Ok(())
    }
}
