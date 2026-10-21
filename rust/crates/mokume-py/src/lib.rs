//! PyO3 bindings exposing the mokume Rust compute kernel to Python.
//!
//! maturin builds this crate into the `mokume._mokume` extension module that the
//! `rust/python/mokume/` package imports. It is the FFI boundary between the Rust
//! compute crates and the Python periphery (plotting / tissue maps / reports).
//!
//! The compute commands are reached through [`mokume_command::run_from_args`], so
//! clap parsing and dispatch stay single-sourced. The Python layer in
//! `rust/python/mokume/` builds the argument vector from ergonomic keyword arguments.

use std::collections::{HashMap, HashSet};

use mokume_core::{DifferentialExpressionConfig, ImputationConfig};
use mokume_pipeline::MatrixDifferentialExpressionResults;
use mokume_pipeline::{
    run_pibaq_from_mapping, PeptideObservation, PibaqDigest, PibaqDigestProvenance,
};
use mokume_stats::de::{DeResult, EnsembleResult, Significance};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The mokume version string (the crate's compile-time version).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Run a mokume command in-process from an argument vector (no subprocess).
///
/// `args` is the subcommand and its flags WITHOUT the program name, for example
/// `["quantify", "features2proteins", "--parquet", "x.parquet", "--output", "y.csv"]`. A
/// parse or runtime error is raised as a Python `RuntimeError` rather than
/// exiting the interpreter.
#[pyfunction]
fn run(args: Vec<String>) -> PyResult<()> {
    // clap's `parse_from` expects argv[0] to be the program name.
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("mokume".to_string());
    argv.extend(args);
    mokume_command::run_from_args(argv).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// Return the machine-readable Rust command schema as JSON.
#[pyfunction]
fn command_schema() -> PyResult<String> {
    mokume_command::command_schema_json()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// Parse an argv vector without dispatching or executing the command.
#[pyfunction]
fn validate_args(args: Vec<String>) -> PyResult<()> {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("mokume".to_string());
    argv.extend(args);
    mokume_command::validate_args(argv).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// Return the runtime pyOpenMS digestion request for a parsed piBAQ command.
#[pyfunction]
fn pibaq_digest_request(args: Vec<String>) -> Option<(String, String, usize, usize, usize)> {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("mokume".to_string());
    argv.extend(args);
    mokume_command::pibaq_digest_request_from_args(argv).map(|request| {
        (
            request.fasta.to_string_lossy().into_owned(),
            request.enzyme,
            request.min_aa,
            request.max_aa,
            request.missed_cleavages,
        )
    })
}

type PibaqDigestProvenanceTuple = (String, String, String, usize, usize, usize);

fn runtime_pibaq_digest(
    accession_peptides: HashMap<String, HashSet<String>>,
    provenance: PibaqDigestProvenanceTuple,
) -> PibaqDigest {
    let (pyopenms_version, enzyme, catalog_hash, min_aa, max_aa, missed_cleavages) = provenance;
    PibaqDigest {
        accession_peptides,
        provenance: PibaqDigestProvenance {
            pyopenms_version,
            enzyme,
            catalog_hash,
            min_aa,
            max_aa,
            missed_cleavages,
        },
    }
}

/// Run a parsed command with a complete runtime pyOpenMS theoretical-peptide map.
#[pyfunction]
fn run_with_pibaq_digest(
    args: Vec<String>,
    accession_peptides: HashMap<String, HashSet<String>>,
    provenance: PibaqDigestProvenanceTuple,
) -> PyResult<()> {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("mokume".to_string());
    argv.extend(args);
    let digest = runtime_pibaq_digest(accession_peptides, provenance);
    mokume_command::run_from_args_with_pibaq_digest(argv, digest)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// Run a mokume command as a CLI and return the process exit code.
///
/// This backs the `mokume` console script (`mokume.__main__:main`). Unlike
/// [`run`], it never raises: clap help/version are printed to stdout (exit code
/// 0), usage errors to stderr (exit code 2), and a runtime failure prints the
/// error and returns 1 without exiting the process, so the caller decides via
/// `sys.exit`.
#[pyfunction]
fn run_cli(args: Vec<String>) -> i32 {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("mokume".to_string());
    argv.extend(args);
    mokume_command::run_cli_from_args(argv)
}

/// Run a CLI command with a complete runtime pyOpenMS theoretical-peptide map.
#[pyfunction]
fn run_cli_with_pibaq_digest(
    args: Vec<String>,
    accession_peptides: HashMap<String, HashSet<String>>,
    provenance: PibaqDigestProvenanceTuple,
) -> i32 {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("mokume".to_string());
    argv.extend(args);
    let digest = runtime_pibaq_digest(accession_peptides, provenance);
    mokume_command::run_cli_from_args_with_pibaq_digest(argv, digest)
}

type PibaqObservationTuple = (String, String, f64);
type PibaqFamilyTuple = (String, Vec<String>);
type PibaqOptionsTuple = (Option<HashMap<String, f64>>, usize, usize);
type PibaqRowTuple = (
    String,
    String,
    f64,
    f64,
    String,
    String,
    usize,
    Option<f64>,
    Option<f64>,
);

/// Compute in-memory piBAQ rows with the native shared-peptide allocator.
#[pyfunction(name = "compute_pibaq")]
fn compute_pibaq_py(
    py: Python<'_>,
    observations: Vec<PibaqObservationTuple>,
    accession_peptides: HashMap<String, HashSet<String>>,
    peptide_accessions: HashMap<String, HashSet<String>>,
    families: Vec<PibaqFamilyTuple>,
    options: PibaqOptionsTuple,
) -> PyResult<Vec<PibaqRowTuple>> {
    let (mw_map, min_anchors, high_anchor_threshold) = options;
    let observations = observations
        .into_iter()
        .map(|(peptide, sample, intensity)| PeptideObservation {
            peptide,
            sample,
            intensity,
        })
        .collect::<Vec<_>>();
    let rows = py.detach(move || {
        run_pibaq_from_mapping(
            &observations,
            accession_peptides,
            peptide_accessions,
            families,
            min_anchors,
            high_anchor_threshold,
            mw_map,
        )
    });
    rows.map(|rows| {
        rows.into_iter()
            .map(|row| {
                (
                    row.protein,
                    row.sample,
                    row.norm_intensity,
                    row.pibaq,
                    row.family_id,
                    row.evidence_level.to_owned(),
                    row.family_size,
                    row.molecular_weight,
                    row.tpa,
                )
            })
            .collect()
    })
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// Normalize a row-major linear-intensity matrix with the Rust kernel.
#[pyfunction(name = "normalize_matrix", signature = (values, method, sample_names=None, threads=None))]
fn normalize_matrix_py(
    py: Python<'_>,
    values: Vec<Vec<Option<f64>>>,
    method: String,
    sample_names: Option<Vec<String>>,
    threads: Option<usize>,
) -> PyResult<Vec<Vec<Option<f64>>>> {
    let matrix = decode_matrix(values);
    let width = matrix.first().map_or(0, Vec::len);
    let names = sample_names.unwrap_or_else(|| {
        (0..width)
            .map(|column| format!("sample_{column}"))
            .collect()
    });
    let normalized =
        py.detach(move || mokume_pipeline::normalize_matrix(&matrix, &names, &method, threads));
    normalized
        .map(encode_matrix)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction(
    name = "impute_matrix",
    signature = (values, method, options=None)
)]
/// Impute a row-major linear-intensity matrix with the Rust kernel.
fn impute_matrix_py(
    py: Python<'_>,
    values: Vec<Vec<Option<f64>>>,
    method: String,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<Vec<Option<f64>>>> {
    let options = ImputationOptions::from_dict(options)?;
    options.validate_for_method(&method)?;
    let matrix = decode_matrix(values);
    let config = ImputationConfig {
        enabled: !matches!(method.trim().to_ascii_lowercase().as_str(), "" | "none"),
        method,
        quantile: options.quantile,
        n_neighbors: options.n_neighbors,
        seed: options.seed,
        tune_sigma: options.tune_sigma,
    };
    let imputed =
        py.detach(move || mokume_pipeline::impute_matrix(&matrix, &config, options.threads));
    imputed
        .map(encode_matrix)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

struct ImputationOptions {
    quantile: f64,
    n_neighbors: usize,
    seed: u64,
    tune_sigma: f64,
    threads: Option<usize>,
    quantile_supplied: bool,
    shift_supplied: bool,
    scale_supplied: bool,
    n_neighbors_supplied: bool,
    stochastic_supplied: bool,
}

impl ImputationOptions {
    fn from_dict(options: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let defaults = ImputationConfig::default();
        let mut parsed = Self {
            quantile: defaults.quantile,
            n_neighbors: defaults.n_neighbors,
            seed: defaults.seed,
            tune_sigma: defaults.tune_sigma,
            threads: None,
            quantile_supplied: false,
            shift_supplied: false,
            scale_supplied: false,
            n_neighbors_supplied: false,
            stochastic_supplied: false,
        };
        let Some(options) = options else {
            return Ok(parsed);
        };
        for (key, value) in options.iter() {
            parsed.set(&key.extract::<String>()?, &value)?;
        }
        Ok(parsed)
    }

    fn set(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match key {
            "quantile" => {
                self.quantile = value.extract()?;
                self.quantile_supplied = true;
            }
            "shift" => {
                value.extract::<f64>()?;
                self.shift_supplied = true;
            }
            "scale" => {
                value.extract::<f64>()?;
                self.scale_supplied = true;
            }
            "n_neighbors" => {
                self.n_neighbors = value.extract()?;
                self.n_neighbors_supplied = true;
            }
            "seed" => {
                self.seed = value.extract()?;
                self.stochastic_supplied = true;
            }
            "tune_sigma" => {
                self.tune_sigma = value.extract()?;
                self.stochastic_supplied = true;
            }
            "threads" => self.threads = value.extract()?,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unknown imputation option `{key}`"
                )))
            }
        }
        Ok(())
    }

    fn validate_for_method(&self, method: &str) -> PyResult<()> {
        let method = method.trim().to_ascii_lowercase();
        if self.quantile_supplied && !matches!(method.as_str(), "mindet" | "minprob") {
            return Err(PyTypeError::new_err(
                "`quantile` only applies to mindet/minprob imputation",
            ));
        }
        if self.shift_supplied || self.scale_supplied {
            return Err(PyTypeError::new_err(
                "MinProb now follows imputeLCMD; use `tune_sigma` instead of legacy `shift`/`scale`",
            ));
        }
        if self.stochastic_supplied && !matches!(method.as_str(), "minprob" | "qrilc") {
            return Err(PyTypeError::new_err(
                "`seed` and `tune_sigma` only apply to minprob/qrilc imputation",
            ));
        }
        if self.n_neighbors_supplied && !matches!(method.as_str(), "knn" | "seqknn") {
            return Err(PyTypeError::new_err(
                "`n_neighbors` only applies to knn/seqknn imputation",
            ));
        }
        Ok(())
    }
}

#[pyfunction(
    name = "differential_expression",
    signature = (proteins, values, n_a, n_b, method, options=None)
)]
/// Run one two-group DE comparison on a row-major linear-intensity matrix.
fn differential_expression_py(
    py: Python<'_>,
    proteins: Vec<String>,
    values: Vec<Vec<Option<f64>>>,
    n_a: usize,
    n_b: usize,
    method: String,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<Py<PyDict>>> {
    let options = DifferentialExpressionOptions::from_dict(options)?;
    options.validate_for_method(&method)?;
    let matrix = decode_matrix(values);
    let count_by_protein = peptide_count_map(&proteins, options.peptide_counts.as_deref());
    let config = differential_expression_config(&method, &options);
    let peptide_counts = options.peptide_counts;
    let threads = options.threads;
    let condition_a = options.condition_a;
    let condition_b = options.condition_b;
    let results = py.detach(move || {
        mokume_pipeline::differential_expression_matrix(
            &proteins,
            &matrix,
            n_a,
            n_b,
            peptide_counts.as_deref(),
            &config,
            threads,
        )
    });
    match results.map_err(|error| PyRuntimeError::new_err(error.to_string()))? {
        MatrixDifferentialExpressionResults::Standard(rows) => standard_rows_to_python(
            py,
            rows,
            &method,
            count_by_protein.as_ref(),
            &condition_a,
            &condition_b,
        ),
        MatrixDifferentialExpressionResults::Ensemble(rows) => ensemble_rows_to_python(py, rows),
    }
}

fn peptide_count_map(proteins: &[String], counts: Option<&[f64]>) -> Option<HashMap<String, f64>> {
    counts.map(|counts| {
        proteins
            .iter()
            .cloned()
            .zip(counts.iter().copied())
            .collect()
    })
}

fn differential_expression_config(
    method: &str,
    options: &DifferentialExpressionOptions,
) -> DifferentialExpressionConfig {
    DifferentialExpressionConfig {
        enabled: true,
        contrasts: None,
        contrasts_file: None,
        method: method.to_owned(),
        ensemble_methods: options.ensemble_methods.clone(),
        ensemble_min_k: options.ensemble_min_k,
        log2fc_threshold: options.log2fc_threshold,
        effect_size_gate: options.effect_size_gate.clone(),
        fdr_threshold: options.fdr_threshold,
        fdr_method: options.fdr_method.clone(),
        output: None,
    }
}

struct DifferentialExpressionOptions {
    peptide_counts: Option<Vec<f64>>,
    ensemble_methods: Option<Vec<String>>,
    ensemble_min_k: usize,
    log2fc_threshold: f64,
    effect_size_gate: Option<String>,
    fdr_threshold: f64,
    fdr_method: String,
    condition_a: String,
    condition_b: String,
    threads: Option<usize>,
    ensemble_min_k_supplied: bool,
    fdr_method_supplied: bool,
}

impl DifferentialExpressionOptions {
    fn from_dict(options: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let defaults = DifferentialExpressionConfig::default();
        let mut parsed = Self {
            peptide_counts: None,
            ensemble_methods: None,
            ensemble_min_k: defaults.ensemble_min_k,
            log2fc_threshold: defaults.log2fc_threshold,
            effect_size_gate: None,
            fdr_threshold: defaults.fdr_threshold,
            fdr_method: defaults.fdr_method,
            condition_a: "A".to_owned(),
            condition_b: "B".to_owned(),
            threads: None,
            ensemble_min_k_supplied: false,
            fdr_method_supplied: false,
        };
        let Some(options) = options else {
            return Ok(parsed);
        };
        for (key, value) in options.iter() {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "peptide_counts" => parsed.peptide_counts = value.extract()?,
                "ensemble_methods" => parsed.ensemble_methods = value.extract()?,
                "ensemble_min_k" => {
                    parsed.ensemble_min_k = value.extract()?;
                    parsed.ensemble_min_k_supplied = true;
                }
                "log2fc_threshold" => parsed.log2fc_threshold = value.extract()?,
                "effect_size_gate" => parsed.effect_size_gate = value.extract()?,
                "fdr_threshold" => parsed.fdr_threshold = value.extract()?,
                "fdr_method" => {
                    parsed.fdr_method = value.extract()?;
                    parsed.fdr_method_supplied = true;
                }
                "condition_a" => parsed.condition_a = value.extract()?,
                "condition_b" => parsed.condition_b = value.extract()?,
                "threads" => parsed.threads = value.extract()?,
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "unknown differential-expression option `{key}`"
                    )))
                }
            }
        }
        Ok(parsed)
    }

    fn validate_for_method(&self, method: &str) -> PyResult<()> {
        let method = method.trim().to_ascii_lowercase();
        if self.ensemble_min_k_supplied && method != "ensemble" {
            return Err(PyTypeError::new_err(
                "`ensemble_min_k` only applies to ensemble DE",
            ));
        }
        if self.fdr_method_supplied && matches!(method.as_str(), "rots" | "limrots") {
            return Err(PyTypeError::new_err(format!(
                "`fdr_method` does not apply to {method}, which retains its permutation FDR"
            )));
        }
        Ok(())
    }
}

fn decode_matrix(values: Vec<Vec<Option<f64>>>) -> Vec<Vec<f64>> {
    values
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect()
        })
        .collect()
}

fn encode_matrix(values: Vec<Vec<f64>>) -> Vec<Vec<Option<f64>>> {
    values
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| value.is_finite().then_some(value))
                .collect()
        })
        .collect()
}

fn standard_rows_to_python(
    py: Python<'_>,
    rows: Vec<DeResult>,
    method: &str,
    count_by_protein: Option<&HashMap<String, f64>>,
    condition_a: &str,
    condition_b: &str,
) -> PyResult<Vec<Py<PyDict>>> {
    let method = method.trim().to_ascii_lowercase();
    rows.into_iter()
        .map(|row| {
            let output = PyDict::new(py);
            output.set_item("ProteinName", &row.protein)?;
            output.set_item("log2FC", row.log2_fold_change)?;
            output.set_item("pvalue", row.p_value)?;
            output.set_item("adj_pvalue", row.adj_p_value)?;
            set_standard_method_fields(&output, &row, &method, count_by_protein)?;
            output.set_item(format!("mean_{condition_a}"), row.mean_a)?;
            output.set_item(format!("mean_{condition_b}"), row.mean_b)?;
            output.set_item("n_a", row.n_a)?;
            output.set_item("n_b", row.n_b)?;
            output.set_item("significance", significance_label(row.significance))?;
            Ok(output.unbind())
        })
        .collect()
}

fn set_standard_method_fields(
    output: &Bound<'_, PyDict>,
    row: &DeResult,
    method: &str,
    count_by_protein: Option<&HashMap<String, f64>>,
) -> PyResult<()> {
    match method {
        "limma" => {
            output.set_item("t_stat", row.t_statistic)?;
            output.set_item("AveExpr", row.ave_expr)?;
            output.set_item("B", row.b)
        }
        "deqms" => {
            output.set_item("sca_t", row.t_statistic)?;
            output.set_item("sca_pvalue", row.p_value)?;
            output.set_item("sca_adj_pvalue", row.adj_p_value)?;
            output.set_item(
                "peptide_count",
                count_by_protein
                    .and_then(|counts| counts.get(&row.protein))
                    .copied()
                    .unwrap_or(1.0),
            )?;
            output.set_item("log_pvalue", row.log_p_value)
        }
        "rots" => output.set_item("d_stat", row.t_statistic),
        _ => output.set_item("t_stat", row.t_statistic),
    }
}

fn ensemble_rows_to_python(py: Python<'_>, rows: Vec<EnsembleResult>) -> PyResult<Vec<Py<PyDict>>> {
    rows.into_iter()
        .map(|row| {
            let output = PyDict::new(py);
            output.set_item("ProteinName", row.protein)?;
            output.set_item("log2FC", row.log2_fold_change)?;
            output.set_item("pvalue", row.p_value)?;
            output.set_item("n_methods_up", row.n_methods_up)?;
            output.set_item("n_methods_down", row.n_methods_down)?;
            output.set_item("methods_significant", row.methods_significant)?;
            output.set_item("adj_pvalue", row.adj_p_value)?;
            output.set_item("significance", significance_label(row.significance))?;
            Ok(output.unbind())
        })
        .collect()
}

fn significance_label(significance: Significance) -> &'static str {
    match significance {
        Significance::Up => "UP",
        Significance::Down => "DOWN",
        Significance::Unchanged => "Unchanged",
        Significance::NotTested => "NotTested",
    }
}

fn register_pibaq_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pibaq_digest_request, module)?)?;
    module.add_function(wrap_pyfunction!(run_with_pibaq_digest, module)?)?;
    module.add_function(wrap_pyfunction!(run_cli_with_pibaq_digest, module)?)?;
    module.add_function(wrap_pyfunction!(compute_pibaq_py, module)?)?;
    Ok(())
}

fn register_command_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    module.add_function(wrap_pyfunction!(run_cli, module)?)?;
    module.add_function(wrap_pyfunction!(command_schema, module)?)?;
    module.add_function(wrap_pyfunction!(validate_args, module)?)?;
    Ok(())
}

/// The `mokume._mokume` extension module.
#[pymodule]
fn _mokume(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_command_functions(module)?;
    register_pibaq_functions(module)?;
    module.add_function(wrap_pyfunction!(normalize_matrix_py, module)?)?;
    module.add_function(wrap_pyfunction!(impute_matrix_py, module)?)?;
    module.add_function(wrap_pyfunction!(differential_expression_py, module)?)?;
    Ok(())
}
