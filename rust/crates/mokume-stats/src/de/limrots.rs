//! Bioconductor LimROTS 1.2.8 for a single unpaired two-group contrast.
//!
//! The supported contract is a complete log2 protein-by-sample matrix, a
//! cell-means design, and limma moderation with `robust=false, trend=false`.
//! Both bootstrap and permutation fits contain 2B draws; optimization includes
//! the SLR (a1=1,a2=0) branch and uses permutation/permutation reproducibility.
//! The returned statistic is nonnegative, the p-value follows pooled
//! qvalue::empPvals, and adj_p_value is the native LimROTS FDR
//! (calculateFalseDiscoveryRate, the median over permutations). The official
//! BH.pvalue and Storey q-values are not part of this output contract.
//!
//! SplitMix64 supplies reproducible sampling, but a matching R seed does not
//! generate the same indices. Fixed-index official fixtures validate the
//! deterministic stages separately from random-stream differences.

mod fit;
mod optimization;
#[cfg(test)]
mod tests;

use super::correct::permutation_fdr;
use super::rots::{build_n_grid, SplitMix64};
use super::{classify, DeResult};
use fit::{fit, Fit};
use mokume_core::{MokumeError, Result};
use optimization::{empirical_pvalues, optimize, Optimum};
use rayon::prelude::*;

/// Sampling and search settings for the supported two-group LimROTS model.
#[derive(Clone, Copy, Debug)]
pub struct LimRotsOptions {
    /// B, corresponding to official `niter`; generates 2B draws of each kind.
    pub n_iterations: usize,
    /// Exclusive top-list upper bound; None uses floor(number of proteins/4).
    pub k_max: Option<usize>,
    pub seed: u64,
}

impl Default for LimRotsOptions {
    fn default() -> Self {
        Self {
            n_iterations: 100,
            k_max: None,
            seed: 42,
        }
    }
}

/// Results plus the selected optimization parameters, for reproducible audits.
pub struct LimRotsOutput {
    pub results: Vec<DeResult>,
    pub a1: f64,
    pub a2: f64,
    pub k: usize,
    pub reproducibility: f64,
    pub z_score: f64,
}

/// Two-group LimROTS with B=100, K=floor(proteins/4), seed=42.
///
/// Returns an error for unsupported missing values, an empty optimization grid
/// (fewer than 24 proteins with default K), or no finite reproducibility score.
pub fn limrots_two_group(
    proteins: &[String],
    rows: &[&[f64]],
    n_a: usize,
    n_b: usize,
    fdr_threshold: f64,
    log2fc_threshold: f64,
) -> Result<Vec<DeResult>> {
    limrots_two_group_with_options(
        proteins,
        rows,
        n_a,
        n_b,
        fdr_threshold,
        log2fc_threshold,
        LimRotsOptions::default(),
    )
    .map(|output| output.results)
}

/// Run the same model with explicit bootstrap count, K and random seed.
pub fn limrots_two_group_with_options(
    proteins: &[String],
    rows: &[&[f64]],
    n_a: usize,
    n_b: usize,
    fdr_threshold: f64,
    log2fc_threshold: f64,
    options: LimRotsOptions,
) -> Result<LimRotsOutput> {
    let k_values = validate_input(proteins, rows, n_a, n_b, options)?;
    let data = rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
    let plan = SamplingPlan::generate(n_a, n_b, options.n_iterations, options.seed);
    let analysis = analyze(&data, n_a, &plan, &k_values)?;
    Ok(finalize(
        proteins,
        rows,
        n_a,
        n_b,
        analysis,
        fdr_threshold,
        log2fc_threshold,
    ))
}

fn finalize(
    proteins: &[String],
    rows: &[&[f64]],
    n_a: usize,
    n_b: usize,
    analysis: Analysis,
    fdr_threshold: f64,
    log2fc_threshold: f64,
) -> LimRotsOutput {
    let adjusted = analysis.fdr;
    let mut results = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            // Official logfc is rowMeans(group1)-rowMeans(group2). QR contrast
            // coefficients are used separately to build the unsigned statistic.
            let mean_a = row[..n_a].iter().sum::<f64>() / n_a as f64;
            let mean_b = row[n_a..].iter().sum::<f64>() / n_b as f64;
            let log2_fold_change = mean_a - mean_b;
            DeResult {
                protein: proteins[i].clone(),
                log2_fold_change,
                p_value: analysis.pvalues[i],
                log_p_value: analysis.pvalues[i].ln(),
                adj_p_value: adjusted[i],
                t_statistic: analysis.statistics[i],
                ave_expr: f64::NAN,
                b: 0.0,
                mean_a,
                mean_b,
                n_a,
                n_b,
                significance: classify(
                    adjusted[i],
                    log2_fold_change,
                    fdr_threshold,
                    log2fc_threshold,
                ),
            }
        })
        .collect::<Vec<_>>();
    results.sort_by(|a, b| a.adj_p_value.total_cmp(&b.adj_p_value));
    LimRotsOutput {
        results,
        a1: analysis.optimum.a1,
        a2: analysis.optimum.a2,
        k: analysis.optimum.k,
        reproducibility: analysis.optimum.reproducibility,
        z_score: analysis.optimum.z,
    }
}

fn invalid(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

fn validate_input(
    proteins: &[String],
    rows: &[&[f64]],
    n_a: usize,
    n_b: usize,
    options: LimRotsOptions,
) -> Result<Vec<usize>> {
    if proteins.len() != rows.len()
        || n_a < 2
        || n_b < 2
        || rows.iter().any(|r| r.len() != n_a + n_b)
    {
        return Err(invalid(
            "LimROTS requires aligned protein IDs and a rectangular two-group matrix with at least two samples per group",
        ));
    }
    if rows.iter().any(|r| r.iter().any(|v| !v.is_finite())) {
        return Err(invalid(
            "LimROTS currently supports complete finite log2 matrices; missing-value fits are not aligned with the official package",
        ));
    }
    if options.n_iterations < 2 {
        return Err(invalid(
            "LimROTS requires at least two bootstrap iterations",
        ));
    }
    let k_max = options.k_max.unwrap_or(rows.len() / 4).min(rows.len());
    let grid = build_n_grid(k_max);
    if grid.is_empty() {
        return Err(invalid(format!(
            "LimROTS has an empty search grid: {} proteins, K={k_max}; require K>5 (at least 24 proteins for default K)",
            rows.len()
        )));
    }
    Ok(grid)
}

struct SamplingPlan {
    bootstraps: Vec<Vec<usize>>,
    permutations: Vec<Vec<usize>>,
}

impl SamplingPlan {
    fn generate(n_a: usize, n_b: usize, niter: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let bootstraps = (0..2 * niter)
            .map(|_| {
                let mut indices = rng.choice_with_replacement(n_a, n_a);
                indices.extend(
                    rng.choice_with_replacement(n_b, n_b)
                        .into_iter()
                        .map(|i| i + n_a),
                );
                indices
            })
            .collect();
        let permutations = (0..2 * niter).map(|_| rng.permutation(n_a + n_b)).collect();
        Self {
            bootstraps,
            permutations,
        }
    }
}

struct Analysis {
    optimum: Optimum,
    statistics: Vec<f64>,
    pvalues: Vec<f64>,
    fdr: Vec<f64>,
}

fn analyze(
    data: &[Vec<f64>],
    n_a: usize,
    plan: &SamplingPlan,
    k_values: &[usize],
) -> Result<Analysis> {
    let n_samples = data[0].len();
    let labels = (0..n_samples).map(|i| i < n_a).collect::<Vec<_>>();
    let boots = plan
        .bootstraps
        .par_iter()
        .map(|indices| {
            let sampled = data
                .iter()
                .map(|row| indices.iter().map(|&i| row[i]).collect())
                .collect::<Vec<_>>();
            fit(&sampled, &labels)
        })
        .collect::<Vec<_>>();
    let null = plan
        .permutations
        .par_iter()
        .map(|indices| {
            let permuted = indices.iter().map(|&i| i < n_a).collect::<Vec<_>>();
            fit(data, &permuted)
        })
        .collect::<Vec<_>>();
    analyze_fits(fit(data, &labels), &boots, &null, k_values)
}

fn analyze_fits(full: Fit, boots: &[Fit], null: &[Fit], k_values: &[usize]) -> Result<Analysis> {
    let optimum = optimize(boots, null, k_values).ok_or_else(|| {
        invalid("LimROTS has no finite reproducibility score; no default alpha is substituted")
    })?;
    let statistics = full.statistics(optimum.a1, optimum.a2);
    let null_statistics = null
        .iter()
        .map(|f| f.statistics(optimum.a1, optimum.a2))
        .collect::<Vec<_>>();
    let pvalues = empirical_pvalues(&statistics, &null_statistics);
    let fdr = permutation_fdr(&statistics, &null_statistics);
    Ok(Analysis {
        optimum,
        statistics,
        pvalues,
        fdr,
    })
}
