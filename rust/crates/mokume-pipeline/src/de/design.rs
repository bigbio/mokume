//! Fixed-effects designs for `--de-covariate`.
//!
//! Every matrix sample with a condition enters one limma model: one cell-mean
//! column per condition plus the encoded SDRF covariates (numeric columns as
//! is, categorical columns as k-1 indicators), as R's
//! `model.matrix(~0 + condition + covariates)`. Each contrast is the difference
//! of its two condition coefficients. A paired design passes the subject column
//! as a covariate.
//!
//! `--de-block` makes an SDRF column's levels random blocks instead, as limma's
//! `duplicateCorrelation` + `lmFit(block =, correlation =)`: one intra-block
//! correlation is estimated over all proteins and design samples, then every
//! contrast uses the generalized least-squares fit.

use std::collections::HashMap;
use std::path::Path;

use mokume_core::{DifferentialExpressionConfig, Result, SampleId};
use mokume_io::{normalize_file_key, SdrfRawTable};
use mokume_stats::de::{
    deqms_design, duplicate_correlation, limma_design, validate_design, DesignContrast, RandomBlock,
};
use tracing::{info, warn};

use super::{
    available_conditions, build_count_vector, de_output_path, finalize_member_results,
    sample_condition, write_de_csv, Contrast, DeMethod, MemberMethod, Prepared,
};
use crate::{
    encode_covariate, invalid_input, match_sdrf_column, ProteinMatrix, SDRF_SAMPLE_COLUMNS,
};

/// SDRF column holding the run file, used for run-level matrix columns.
const SDRF_DATA_FILE_COLUMN: &str = "comment[data file]";

pub(super) struct ModelDesign<'a> {
    /// Model samples in matrix column order.
    samples: Vec<SampleId>,
    sample_conditions: Vec<&'a str>,
    /// Sorted condition levels; level `i` is design column `i`.
    conditions: Vec<&'a str>,
    /// One row per model sample.
    rows: Vec<Vec<f64>>,
    /// Random-block label per model sample, numbered by first appearance.
    blocks: Option<Vec<usize>>,
    block_correlation: f64,
}

impl<'a> ModelDesign<'a> {
    pub(super) fn new(
        raw: &SdrfRawTable,
        samples: &[(SampleId, &str)],
        condition_by_sample: &'a HashMap<String, String>,
        covariates: &[String],
        block: Option<&str>,
    ) -> Result<Self> {
        let mut ids = Vec::new();
        let mut names = Vec::new();
        let mut sample_conditions = Vec::new();
        for (sample, name) in samples {
            if let Some(condition) = sample_condition(condition_by_sample, name) {
                ids.push(*sample);
                names.push(*name);
                sample_conditions.push(condition);
            }
        }
        let mut conditions = sample_conditions.clone();
        conditions.sort_unstable();
        conditions.dedup();
        let rows = design_rows(raw, &names, &sample_conditions, &conditions, covariates)?;
        let blocks = block
            .map(|column| {
                covariate_values(raw, column, &names, "block").map(|values| block_labels(&values))
            })
            .transpose()?;
        Ok(Self {
            samples: ids,
            sample_conditions,
            conditions,
            rows,
            blocks,
            block_correlation: 0.0,
        })
    }

    fn random_block(&self) -> Option<RandomBlock<'_>> {
        self.blocks.as_deref().map(|blocks| RandomBlock {
            blocks,
            correlation: self.block_correlation,
        })
    }

    fn level(
        &self,
        condition: &str,
        condition_by_sample: &HashMap<String, String>,
    ) -> Result<usize> {
        self.conditions
            .iter()
            .position(|level| *level == condition)
            .ok_or_else(|| {
                invalid_input(format!(
                    "No samples for '{condition}'. Available: {}",
                    available_conditions(condition_by_sample)
                ))
            })
    }

    fn group(&self, level: usize) -> Vec<usize> {
        (0..self.samples.len())
            .filter(|&sample| self.sample_conditions[sample] == self.conditions[level])
            .collect()
    }
}

/// One design row per model sample: the condition indicators, then each
/// encoded covariate.
fn design_rows(
    raw: &SdrfRawTable,
    names: &[&str],
    sample_conditions: &[&str],
    conditions: &[&str],
    covariates: &[String],
) -> Result<Vec<Vec<f64>>> {
    let mut rows = sample_conditions
        .iter()
        .map(|condition| {
            conditions
                .iter()
                .map(|level| f64::from(level == condition))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for column in covariates {
        let values = covariate_values(raw, column, names, "covariate")?;
        let encoded = encode_covariate(column, &values)?;
        info!(
            covariate = %column,
            design_columns = encoded.len(),
            "DE covariate"
        );
        for (sample, row) in rows.iter_mut().enumerate() {
            row.extend(encoded.iter().map(|values| values[sample]));
        }
    }
    validate_design(&rows)?;
    Ok(rows)
}

/// Values of one covariate or block column per model sample, keyed like the
/// conditions: the source name first, then the matrix column normalized as a
/// data file. `role` names the column in errors.
fn covariate_values(
    raw: &SdrfRawTable,
    column: &str,
    names: &[&str],
    role: &str,
) -> Result<Vec<String>> {
    let value_col = match_sdrf_column(raw.headers(), column).ok_or_else(|| {
        invalid_input(format!("DE {role} column `{column}` was not found in SDRF"))
    })?;
    let by_sample = values_by_sample(raw, value_col);
    let values = names
        .iter()
        .map(|name| {
            by_sample
                .get(*name)
                .or_else(|| by_sample.get(&normalize_file_key(name)))
                .map(|value| (*value).to_owned())
                .ok_or_else(|| {
                    invalid_input(format!(
                        "DE {role} column `{column}` has no SDRF row for sample `{name}`"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    if values.iter().all(|value| *value == values[0]) {
        return Err(invalid_input(format!(
            "DE {role} column `{column}` is constant across the DE samples"
        )));
    }
    Ok(values)
}

/// One SDRF column's values keyed by source name and by normalized data file;
/// the first row of a key wins.
fn values_by_sample(raw: &SdrfRawTable, value_col: usize) -> HashMap<String, &str> {
    let mut by_sample = HashMap::new();
    if let Some(source_col) = SDRF_SAMPLE_COLUMNS
        .iter()
        .find_map(|name| raw.column_index(name))
    {
        for row in 0..raw.row_count() {
            by_sample
                .entry(raw.cell(row, source_col).to_owned())
                .or_insert_with(|| raw.cell(row, value_col));
        }
    }
    if let Some(file_col) = raw.column_index(SDRF_DATA_FILE_COLUMN) {
        for row in 0..raw.row_count() {
            let key = normalize_file_key(raw.cell(row, file_col));
            if !key.is_empty() {
                by_sample
                    .entry(key)
                    .or_insert_with(|| raw.cell(row, value_col));
            }
        }
    }
    by_sample
}

/// Number block values by first appearance.
fn block_labels(values: &[String]) -> Vec<usize> {
    let mut levels: Vec<&str> = Vec::new();
    values
        .iter()
        .map(|value| {
            levels
                .iter()
                .position(|level| level == value)
                .unwrap_or_else(|| {
                    levels.push(value);
                    levels.len() - 1
                })
        })
        .collect()
}

/// limma's consensus intra-block correlation over every protein of the design
/// samples.
fn block_correlation(
    matrix: &ProteinMatrix,
    design: &ModelDesign<'_>,
    column: &str,
) -> Result<f64> {
    let Some(blocks) = design.blocks.as_deref() else {
        return Ok(0.0);
    };
    let (_, rows) = matrix.log2_rows(&design.samples)?;
    let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let estimate = duplicate_correlation(&refs, &design.rows, blocks)?;
    if let Some(warning) = estimate.warning {
        warn!(block = %column, "{warning}");
    }
    info!(
        block = %column,
        correlation = estimate.consensus,
        "DE block correlation"
    );
    Ok(estimate.consensus)
}

/// Build the design once and run every contrast on it.
pub(super) fn run_contrasts(
    matrix: &ProteinMatrix,
    raw_sdrf: Option<&SdrfRawTable>,
    samples: &[(SampleId, &str)],
    condition_by_sample: &HashMap<String, String>,
    contrasts: &[Contrast],
    method: DeMethod,
    config: &DifferentialExpressionConfig,
) -> Result<()> {
    let DeMethod::Member(method @ (MemberMethod::Limma | MemberMethod::Deqms)) = method else {
        return Err(invalid_input(
            "--de-covariate and --de-block require --de-method limma or deqms",
        ));
    };
    let raw_sdrf = raw_sdrf
        .ok_or_else(|| invalid_input("differential expression requires an SDRF file (--sdrf)"))?;
    let mut design = ModelDesign::new(
        raw_sdrf,
        samples,
        condition_by_sample,
        &config.covariates,
        config.block.as_deref(),
    )?;
    if let Some(column) = config.block.as_deref() {
        design.block_correlation = block_correlation(matrix, &design, column)?;
    }
    for contrast in contrasts {
        let output = de_output_path(config, contrast, contrasts.len());
        run_contrast(
            matrix,
            &design,
            condition_by_sample,
            contrast,
            method,
            output.as_deref(),
            config,
        )?;
    }
    Ok(())
}

/// Fit the design for one contrast and write its table.
fn run_contrast(
    matrix: &ProteinMatrix,
    design: &ModelDesign<'_>,
    condition_by_sample: &HashMap<String, String>,
    contrast: &Contrast,
    method: MemberMethod,
    output: Option<&Path>,
    config: &DifferentialExpressionConfig,
) -> Result<()> {
    let level_a = design.level(&contrast.cond_a, condition_by_sample)?;
    let level_b = design.level(&contrast.cond_b, condition_by_sample)?;
    let mut weights = vec![0.0; design.rows[0].len()];
    weights[level_a] = 1.0;
    weights[level_b] = -1.0;
    let group_a = design.group(level_a);
    let group_b = design.group(level_b);
    let spec = DesignContrast {
        design: &design.rows,
        weights: &weights,
        group_a: &group_a,
        group_b: &group_b,
        block: design.random_block(),
    };

    let (proteins, rows) = matrix.log2_rows(&design.samples)?;
    let row_refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let peptide_counts = build_count_vector(matrix, &proteins, &design.samples);
    let (fdr, log2fc) = (config.fdr_threshold, config.log2fc_threshold);
    // `run_contrasts` admits only limma and DEqMS.
    let results = if method == MemberMethod::Deqms {
        deqms_design(&proteins, &row_refs, &spec, &peptide_counts, fdr, log2fc)?
    } else {
        limma_design(&proteins, &row_refs, &spec, fdr, log2fc)?
    };
    let results = finalize_member_results(method, results, config)?;
    let prepared = Prepared {
        proteins: &proteins,
        rows: &row_refs,
        peptide_counts: &peptide_counts,
        precision: None,
        n_a: group_a.len(),
        n_b: group_b.len(),
    };
    output.map_or(Ok(()), |path| {
        write_de_csv(path, &results, contrast, method, &prepared)
    })
}
