//! `correct-batches` command: parametric ComBat batch-effect correction for
//! long-format piBAQ TSV files.
//!
//! This mirrors `mokume.commands.batch_correct.run_batch_correction` in the
//! Python package. The flow:
//!   1. glob `folder` for `pattern`, concatenate the long-format TSV files
//!      (honoring the comment character and the separator);
//!   2. pivot to a protein (row) x sample (column) matrix of raw piBAQ and reject
//!      any missing or non-finite cell before ComBat;
//!   3. validate sample IDs and derive integer batch labels from the prefix
//!      before the first '-' using first-seen (`pd.factorize`) order;
//!   4. run parametric ComBat (`combat_parametric` with default `ComBatParams`,
//!      equivalent to `apply_batch_correction(df, batch, kwargs={})`);
//!   5. merge the corrected values back onto the raw long rows by
//!      (sample, protein) and write the long-format TSV with the corrected
//!      column appended.
//!
//! When `--export-anndata` is set, an AnnData `.h5ad` file is written alongside
//! the TSV (the output path with its extension replaced by `.h5ad`), mirroring
//! `mokume.io.parquet.create_anndata` + `adata.write`. See [`export_anndata`].
//!
//! Determinism note: Python's output row order follows `glob.glob` (filesystem
//! order). Here the matched files are sorted by path and rows are emitted in
//! file/append order, which is deterministic. The corrected values are keyed by
//! (sample, protein), so they are independent of that ordering. Both the matrix
//! columns (samples) and rows (proteins) are sorted before ComBat, matching the
//! pandas `pivot_table` ordering that drives the Python batch labels.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use mokume_core::{MokumeError, Result};
use mokume_stats::batch::{combat_parametric, ComBatParams};

use crate::CorrectBatchesArgs;

/// A single parsed long-format input row, retaining every column so the output
/// reproduces the input plus the appended corrected column.
struct LongRow {
    values: Vec<String>,
}

/// The combined long-format table read from all matched files.
struct LongTable {
    headers: Vec<String>,
    rows: Vec<LongRow>,
    sample_index: usize,
    protein_index: usize,
    pibaq_index: usize,
}

/// Entry point for the `correct-batches` command.
pub fn run_correct_batches(args: &CorrectBatchesArgs) -> Result<()> {
    let separator = single_byte(&args.sep, "sep")?;
    let comment = optional_single_byte(&args.comment, "comment")?;
    let paths = matched_input_files(&args.input, &args.pattern, &args.output)?;

    let table = load_long_table(
        &paths,
        separator,
        comment,
        &args.sample_id_column,
        &args.protein_id_column,
        &args.pibaq_raw_column,
    )?;
    validate_corrected_column(&table.headers, &args.pibaq_corrected_column)?;

    // Sorted, unique proteins (rows) and samples (columns), matching the pandas
    // `pivot_table` ordering the Python code relies on.
    let mut proteins = BTreeSet::new();
    let mut samples = BTreeSet::new();
    for row in &table.rows {
        proteins.insert(cell(row, table.protein_index)?.to_owned());
        samples.insert(cell(row, table.sample_index)?.to_owned());
    }
    let proteins = proteins.into_iter().collect::<Vec<_>>();
    let samples = samples.into_iter().collect::<Vec<_>>();

    validate_sample_ids(&samples)?;
    let batch = batch_labels(&samples)?;
    ensure_batch_sizes(&batch)?;

    let matrix = build_matrix(&table, &proteins, &samples)?;
    let corrected = combat_parametric(&matrix, &batch, ComBatParams::default());
    let protein_pos = index_map(&proteins);
    let sample_pos = index_map(&samples);

    write_output(
        &args.output,
        &table,
        separator,
        &args.pibaq_corrected_column,
        |row| {
            let sample = cell(row, table.sample_index)?;
            let protein = cell(row, table.protein_index)?;
            let (Some(&protein_idx), Some(&sample_idx)) =
                (protein_pos.get(protein), sample_pos.get(sample))
            else {
                return Ok(None);
            };
            let Some(corrected_row) = corrected.get(protein_idx) else {
                return Err(MokumeError::InvalidInput {
                    message: format!("Corrected matrix is missing protein row {protein_idx}."),
                });
            };
            let Some(&value) = corrected_row.get(sample_idx) else {
                return Err(MokumeError::InvalidInput {
                    message: format!(
                        "Corrected matrix row {protein_idx} is missing sample column {sample_idx}."
                    ),
                });
            };
            Ok(Some(value))
        },
    )?;

    if args.export_anndata {
        export_anndata(args, &proteins, &samples, &matrix, &corrected)?;
    }

    Ok(())
}

/// Write the AnnData `.h5ad` export alongside the TSV, mirroring the Python
/// `create_anndata(df_pibaq, obs_col=SampleID, var_col=ProteinName,
/// value_col=PiBAQ, layer_cols=[PiBAQBec])` path.
///
/// The matrices supplied here are protein (row) x sample (column); AnnData wants
/// observation (sample) x variable (protein), so both are transposed. `X` holds
/// the complete raw piBAQ matrix and the `PiBAQBec` layer holds the matching
/// corrected matrix.
fn export_anndata(
    args: &CorrectBatchesArgs,
    proteins: &[String],
    samples: &[String],
    raw: &[Vec<f64>],
    corrected: &[Vec<f64>],
) -> Result<()> {
    // Transpose protein x sample into sample (obs) x protein (var) layout.
    let n_obs = samples.len();
    let n_var = proteins.len();
    let mut x = vec![vec![0.0_f64; n_var]; n_obs];
    let mut layer = vec![vec![0.0_f64; n_var]; n_obs];
    for (protein_idx, _) in proteins.iter().enumerate() {
        for (sample_idx, _) in samples.iter().enumerate() {
            x[sample_idx][protein_idx] = raw[protein_idx][sample_idx];
            layer[sample_idx][protein_idx] = corrected[protein_idx][sample_idx];
        }
    }

    let output_path = anndata_path(&args.output);
    let layers = vec![(args.pibaq_corrected_column.clone(), layer)];
    let export = crate::h5ad::AnnDataExport {
        obs_names: samples,
        obs_index_name: &args.sample_id_column,
        var_names: proteins,
        var_index_name: &args.protein_id_column,
        x: &x,
        layers: &layers,
    };
    crate::h5ad::write_h5ad(&output_path, &export)
}

/// Replace the output file's extension with `.h5ad`, matching Python's
/// `Path(output).with_suffix(".h5ad")`.
fn anndata_path(output: &Path) -> PathBuf {
    output.with_extension("h5ad")
}

/// Read every matched file and concatenate rows in the first file's schema.
/// Files are visited in sorted path order for determinism. Later files may
/// reorder the same unique columns, but their values are canonicalized before
/// they enter the combined table.
fn load_long_table(
    paths: &[PathBuf],
    separator: u8,
    comment: Option<u8>,
    sample_column: &str,
    protein_column: &str,
    pibaq_column: &str,
) -> Result<LongTable> {
    let mut headers: Option<Vec<String>> = None;
    let mut rows = Vec::new();

    for path in paths {
        let (file_headers, file_rows) = read_long_file(path, separator, comment)?;
        match &headers {
            None => {
                validate_unique_headers(path, &file_headers)?;
                headers = Some(file_headers.clone());
                rows.extend(file_rows.into_iter().map(|values| LongRow { values }));
            }
            Some(expected) => {
                let positions = canonical_positions(path, expected, &file_headers)?;
                if positions_are_identity(&positions) {
                    rows.extend(file_rows.into_iter().map(|values| LongRow { values }));
                } else {
                    for values in file_rows {
                        rows.push(LongRow {
                            values: reorder_values(path, values, &positions)?,
                        });
                    }
                }
            }
        }
    }

    let headers = headers.unwrap_or_default();
    let sample_index = column_index(&headers, sample_column)?;
    let protein_index = column_index(&headers, protein_column)?;
    let pibaq_index = column_index(&headers, pibaq_column)?;
    validate_source_columns(sample_column, protein_column, pibaq_column)?;

    Ok(LongTable {
        headers,
        rows,
        sample_index,
        protein_index,
        pibaq_index,
    })
}

/// Read a single long-format TSV, returning `(headers, rows)`.
fn read_long_file(
    path: &Path,
    separator: u8,
    comment: Option<u8>,
) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut builder = csv::ReaderBuilder::new();
    builder.delimiter(separator).flexible(false);
    if let Some(comment) = comment {
        builder.comment(Some(comment));
    }
    let mut reader = builder
        .from_path(path)
        .map_err(|source| csv_error(path, source))?;

    let headers = reader
        .headers()
        .map_err(|source| csv_error(path, source))?
        .iter()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|source| csv_error(path, source))?;
        rows.push(record.iter().map(ToOwned::to_owned).collect::<Vec<_>>());
    }
    Ok((headers, rows))
}

/// Build the complete protein (row) x sample (column) matrix of raw piBAQ
/// values. Missing cells, blank/non-finite values, and duplicate combinations
/// are rejected before ComBat; none are silently converted to biological zero.
fn build_matrix(
    table: &LongTable,
    proteins: &[String],
    samples: &[String],
) -> Result<Vec<Vec<f64>>> {
    let protein_pos = index_map(proteins);
    let sample_pos = index_map(samples);
    let mut matrix = vec![vec![0.0_f64; samples.len()]; proteins.len()];
    let mut seen: HashSet<(&str, &str)> = HashSet::new();

    for row in &table.rows {
        let protein = cell(row, table.protein_index)?;
        let sample = cell(row, table.sample_index)?;
        if !seen.insert((protein, sample)) {
            return Err(MokumeError::InvalidInput {
                message: format!(
                    "Found duplicate combination of protein '{protein}' and sample '{sample}'."
                ),
            });
        }
        let raw = cell(row, table.pibaq_index)?;
        let value = parse_value(raw, table.pibaq_index)?;
        let (Some(&r), Some(&c)) = (protein_pos.get(protein), sample_pos.get(sample)) else {
            // Unreachable: proteins/samples were derived from these same rows.
            continue;
        };
        matrix[r][c] = value;
    }
    let mut missing_count = 0;
    let mut missing_examples = Vec::new();
    for protein in proteins {
        for sample in samples {
            if !seen.contains(&(protein.as_str(), sample.as_str())) {
                missing_count += 1;
                if missing_examples.len() < 5 {
                    missing_examples.push(format!("({protein}, {sample})"));
                }
            }
        }
    }
    if missing_count > 0 {
        let examples = missing_examples.join(", ");
        return Err(MokumeError::InvalidInput {
            message: format!(
                "ComBat requires a complete protein x sample matrix; found {} missing cells (examples: {examples}). Handle missing values explicitly before correct-batches.",
                missing_count
            ),
        });
    }
    Ok(matrix)
}

/// Write the long-format output: every original column followed by the corrected
/// column. `lookup` returns the corrected value for a row (None leaves the cell
/// empty, matching a left-merge miss).
fn write_output(
    output: &Path,
    table: &LongTable,
    separator: u8,
    corrected_column: &str,
    lookup: impl Fn(&LongRow) -> Result<Option<f64>>,
) -> Result<()> {
    let file = File::create(output).map_err(|source| MokumeError::Io {
        path: output.to_path_buf(),
        source,
    })?;
    let mut writer = csv::WriterBuilder::new()
        .delimiter(separator)
        .from_writer(BufWriter::new(file));

    let mut header = table.headers.iter().map(String::as_str).collect::<Vec<_>>();
    header.push(corrected_column);
    writer
        .write_record(header)
        .map_err(|source| csv_write_error(output, source))?;

    for row in &table.rows {
        let corrected = lookup(row)?;
        let corrected = corrected.map(format_value);
        writer
            .write_record(
                row.values
                    .iter()
                    .map(String::as_str)
                    .chain(std::iter::once(corrected.as_deref().unwrap_or(""))),
            )
            .map_err(|source| csv_write_error(output, source))?;
    }
    writer.flush().map_err(|source| MokumeError::Io {
        path: output.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Derive integer batch labels from sample names: the prefix before the first
/// '-' is the batch ID; labels follow first-seen order (`pd.factorize`).
fn batch_labels(samples: &[String]) -> Result<Vec<usize>> {
    let mut order: Vec<String> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut labels = Vec::with_capacity(samples.len());
    for sample in samples {
        let prefix = batch_prefix(sample)?;
        let label = match seen.get(&prefix) {
            Some(&label) => label,
            None => {
                let label = order.len();
                order.push(prefix.clone());
                seen.insert(prefix, label);
                label
            }
        };
        labels.push(label);
    }
    Ok(labels)
}

/// Extract and validate the batch prefix (text before the first '-').
fn batch_prefix(sample: &str) -> Result<String> {
    let prefix = sample.split('-').next().unwrap_or("");
    if prefix.is_empty() {
        return Err(MokumeError::InvalidInput {
            message: format!("Invalid sample name format: {sample}"),
        });
    }
    if !prefix.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(MokumeError::InvalidInput {
            message: format!("Invalid batch ID format: {prefix}"),
        });
    }
    Ok(prefix.to_owned())
}

/// Validate sample IDs against `SAMPLE_ID_REGEX`:
/// `^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$` (alphanumeric segments joined by '-').
fn validate_sample_ids(samples: &[String]) -> Result<()> {
    let invalid = samples
        .iter()
        .filter(|sample| !is_valid_sample_id(sample))
        .cloned()
        .collect::<Vec<_>>();
    if invalid.is_empty() {
        Ok(())
    } else {
        Err(MokumeError::InvalidInput {
            message: format!("Invalid sample IDs found in the data: {invalid:?}"),
        })
    }
}

/// True when `sample` matches `^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$`.
fn is_valid_sample_id(sample: &str) -> bool {
    if sample.is_empty() {
        return false;
    }
    sample
        .split('-')
        .all(|segment| !segment.is_empty() && segment.chars().all(|c| c.is_ascii_alphanumeric()))
}

/// Every batch must contain at least two samples (matching
/// `TooFewSamplesInBatch` in the Python `apply_batch_correction`).
fn ensure_batch_sizes(batch: &[usize]) -> Result<()> {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for &label in batch {
        *counts.entry(label).or_insert(0) += 1;
    }
    let short = counts
        .iter()
        .filter(|&(_, &count)| count < 2)
        .map(|(&label, _)| label)
        .collect::<Vec<_>>();
    if short.is_empty() {
        Ok(())
    } else {
        Err(MokumeError::InvalidInput {
            message: format!(
                "Batches must contain at least two samples, the following batch factors did not: {short:?}"
            ),
        })
    }
}

/// Return the matched files for `folder/pattern`, sorted by path. Supports the
/// shell-style wildcards `*` and `?` used by mokume's default `*pibaq.tsv`.
fn matched_files(folder: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(folder).map_err(|source| MokumeError::Io {
        path: folder.to_path_buf(),
        source,
    })?;
    let mut matches = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| MokumeError::Io {
            path: folder.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            if glob_match(pattern, name) {
                matches.push(path);
            }
        }
    }
    matches.sort();
    Ok(matches)
}

/// Minimal glob matcher for `*` (any run, including empty) and `?` (single
/// character). Sufficient for the `*pibaq.tsv` style patterns mokume uses.
fn glob_match(pattern: &str, name: &str) -> bool {
    let pattern = pattern.chars().collect::<Vec<_>>();
    let name = name.chars().collect::<Vec<_>>();
    glob_match_at(&pattern, &name)
}

fn glob_match_at(pattern: &[char], name: &[char]) -> bool {
    match pattern.first() {
        None => name.is_empty(),
        Some('*') => {
            // Match zero or more characters.
            glob_match_at(&pattern[1..], name)
                || (!name.is_empty() && glob_match_at(pattern, &name[1..]))
        }
        Some('?') => !name.is_empty() && glob_match_at(&pattern[1..], &name[1..]),
        Some(&c) => name.first() == Some(&c) && glob_match_at(&pattern[1..], &name[1..]),
    }
}

/// Map values to their positions for fast lookup.
fn index_map(values: &[String]) -> HashMap<&str, usize> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| (value.as_str(), index))
        .collect()
}

/// Return the positions that reorder a file into the first file's schema.
fn canonical_positions(path: &Path, expected: &[String], actual: &[String]) -> Result<Vec<usize>> {
    validate_unique_headers(path, actual)?;
    let actual_positions = actual
        .iter()
        .enumerate()
        .map(|(index, header)| (header.as_str(), index))
        .collect::<HashMap<_, _>>();

    let missing = expected
        .iter()
        .filter(|header| !actual_positions.contains_key(header.as_str()))
        .collect::<Vec<_>>();
    let extra = actual
        .iter()
        .filter(|header| !expected.iter().any(|expected| expected == *header))
        .collect::<Vec<_>>();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "Schema mismatch in file '{}'. missing columns: {missing:?}; extra columns: {extra:?}; expected columns: {expected:?}; got: {actual:?}",
                path.display()
            ),
        });
    }

    let mut positions = Vec::with_capacity(expected.len());
    for header in expected {
        let Some(&position) = actual_positions.get(header.as_str()) else {
            return Err(MokumeError::InvalidInput {
                message: format!(
                    "Schema mismatch in file '{}': column '{header}' has no source position.",
                    path.display()
                ),
            });
        };
        positions.push(position);
    }
    Ok(positions)
}

fn positions_are_identity(positions: &[usize]) -> bool {
    positions
        .iter()
        .enumerate()
        .all(|(index, &position)| index == position)
}

fn reorder_values(path: &Path, values: Vec<String>, positions: &[usize]) -> Result<Vec<String>> {
    let mut slots = values.into_iter().map(Some).collect::<Vec<_>>();
    let mut reordered = Vec::with_capacity(positions.len());
    for &position in positions {
        let Some(slot) = slots.get_mut(position) else {
            return Err(MokumeError::InvalidInput {
                message: format!(
                    "Schema row in file '{}' is missing column position {position}.",
                    path.display()
                ),
            });
        };
        let Some(value) = slot.take() else {
            return Err(MokumeError::InvalidInput {
                message: format!(
                    "Schema row in file '{}' maps column position {position} more than once.",
                    path.display()
                ),
            });
        };
        reordered.push(value);
    }
    Ok(reordered)
}

fn validate_unique_headers(path: &Path, headers: &[String]) -> Result<()> {
    let mut seen = BTreeSet::new();
    let duplicates = headers
        .iter()
        .filter(|header| !seen.insert(header.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(MokumeError::InvalidInput {
            message: format!(
                "Duplicate columns in file '{}': {duplicates:?}.",
                path.display()
            ),
        })
    }
}

fn validate_source_columns(
    sample_column: &str,
    protein_column: &str,
    pibaq_column: &str,
) -> Result<()> {
    let roles = [
        ("sample", sample_column),
        ("protein", protein_column),
        ("raw piBAQ", pibaq_column),
    ];
    for (left_index, (left_name, left_column)) in roles.iter().enumerate() {
        for (right_name, right_column) in roles.iter().skip(left_index + 1) {
            if left_column == right_column {
                return Err(MokumeError::InvalidInput {
                    message: format!(
                        "The {left_name} column and {right_name} column both use '{left_column}'; source columns must be distinct."
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_corrected_column(headers: &[String], corrected_column: &str) -> Result<()> {
    if corrected_column.is_empty() {
        return Err(MokumeError::InvalidInput {
            message: "The corrected column name must not be empty.".to_string(),
        });
    }
    if headers.iter().any(|header| header == corrected_column) {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "Corrected column '{corrected_column}' already exists in the input schema."
            ),
        });
    }
    Ok(())
}

fn matched_input_files(folder: &Path, pattern: &str, output: &Path) -> Result<Vec<PathBuf>> {
    let paths = matched_files(folder, pattern)?;
    if paths.is_empty() {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "No files found in the directory '{}' matching the pattern '{pattern}'.",
                folder.display()
            ),
        });
    }
    if let Some(input) = paths.iter().find(|input| paths_alias(input, output)) {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "Output path '{}' is also an input file '{}'; choose a separate output path.",
                output.display(),
                input.display()
            ),
        });
    }
    if output_matches_input_domain(folder, pattern, output) {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "Output path '{}' matches the input pattern '{pattern}' in '{}'; choose a separate output path.",
                output.display(),
                folder.display()
            ),
        });
    }
    Ok(paths)
}

fn output_matches_input_domain(folder: &Path, pattern: &str, output: &Path) -> bool {
    let Some(name) = output.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let output_parent = output.parent().unwrap_or_else(|| Path::new("."));
    match (
        std::fs::canonicalize(folder),
        std::fs::canonicalize(output_parent),
    ) {
        (Ok(folder), Ok(output_parent)) => folder == output_parent && glob_match(pattern, name),
        _ => false,
    }
}

fn paths_alias(left: &Path, right: &Path) -> bool {
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn csv_write_error(path: &Path, source: csv::Error) -> MokumeError {
    match source.into_kind() {
        csv::ErrorKind::Io(source) => MokumeError::Io {
            path: path.to_path_buf(),
            source,
        },
        source => MokumeError::InvalidInput {
            message: format!("Error writing file '{}': {source:?}", path.display()),
        },
    }
}

fn column_index(headers: &[String], column: &str) -> Result<usize> {
    headers
        .iter()
        .position(|header| header == column)
        .ok_or_else(|| MokumeError::InvalidInput {
            message: format!("Column '{column}' not found in the input files."),
        })
}

fn cell(row: &LongRow, index: usize) -> Result<&str> {
    row.values
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| MokumeError::InvalidInput {
            message: format!("Row is missing column {index}."),
        })
}

fn parse_value(raw: &str, index: usize) -> Result<f64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "Column {index} contains a missing value; ComBat requires complete input."
            ),
        });
    }
    let value = trimmed
        .parse::<f64>()
        .map_err(|_| MokumeError::InvalidInput {
            message: format!("Column {index} value '{raw}' is not numeric."),
        })?;
    if !value.is_finite() {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "Column {index} value '{raw}' is not finite; ComBat requires complete finite input."
            ),
        });
    }
    Ok(value)
}

/// Format a corrected value. `{}` on `f64` gives Rust's shortest round-trip
/// representation, matching Python's `repr(float)` for the precision the golden
/// test compares (relative 1e-6).
fn format_value(value: f64) -> String {
    format!("{value}")
}

fn single_byte(value: &str, name: &str) -> Result<u8> {
    let bytes = value.as_bytes();
    if bytes.len() == 1 {
        Ok(bytes[0])
    } else {
        Err(MokumeError::InvalidInput {
            message: format!("--{name} must be a single byte, got '{value}'."),
        })
    }
}

fn optional_single_byte(value: &str, name: &str) -> Result<Option<u8>> {
    if value.is_empty() {
        Ok(None)
    } else {
        single_byte(value, name).map(Some)
    }
}

fn csv_error(path: &Path, source: csv::Error) -> MokumeError {
    MokumeError::InvalidInput {
        message: format!("Error reading file '{}': {source}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    // `super::*` brings the crate's single-parameter `Result` alias into scope,
    // so the test helpers spell out the two-parameter standard `Result`.
    type TestResult<T> = std::result::Result<T, Box<dyn Error>>;

    fn temp_dir(tag: &str) -> TestResult<(tempfile::TempDir, PathBuf)> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = tempfile::Builder::new()
            .prefix(&format!("mokume-correct-batches-{tag}-{nanos}-"))
            .tempdir()?;
        let path = directory.path().to_path_buf();
        Ok((directory, path))
    }

    fn write_file(dir: &Path, name: &str, contents: &str) -> TestResult<PathBuf> {
        let path = dir.join(name);
        let mut file = File::create(&path)?;
        file.write_all(contents.as_bytes())?;
        Ok(path)
    }

    fn reorder_first_two_columns(contents: &str) -> String {
        let mut lines = contents
            .lines()
            .map(|line| {
                if line.starts_with('#') {
                    return line.to_string();
                }
                let mut fields = line.split('\t').collect::<Vec<_>>();
                fields.swap(0, 1);
                fields.join("\t")
            })
            .collect::<Vec<_>>();
        lines.push(String::new());
        lines.join("\n")
    }

    fn corrected_by_identity(path: &Path) -> TestResult<HashMap<(String, String), f64>> {
        let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let sample_col = column_index(&headers, "SampleID")?;
        let protein_col = column_index(&headers, "ProteinName")?;
        let corrected_col = column_index(&headers, "PiBAQBec")?;
        let mut values = HashMap::new();
        for record in reader.records() {
            let record = record?;
            values.insert(
                (
                    field(&record, sample_col)?.to_string(),
                    field(&record, protein_col)?.to_string(),
                ),
                field(&record, corrected_col)?.parse::<f64>()?,
            );
        }
        Ok(values)
    }

    fn error_message(result: Result<()>, context: &str) -> TestResult<String> {
        match result {
            Ok(()) => Err(context.to_string().into()),
            Err(error) => Ok(error.to_string()),
        }
    }

    #[test]
    fn combat_matrix_rejects_structural_and_value_missingness() {
        let table = LongTable {
            headers: vec!["ProteinName".into(), "SampleID".into(), "PiBAQ".into()],
            rows: vec![
                LongRow {
                    values: vec!["P1".into(), "S1".into(), "1".into()],
                },
                LongRow {
                    values: vec!["P1".into(), "S2".into(), "2".into()],
                },
                LongRow {
                    values: vec!["P2".into(), "S1".into(), "3".into()],
                },
            ],
            sample_index: 1,
            protein_index: 0,
            pibaq_index: 2,
        };
        let proteins = ["P1".to_owned(), "P2".to_owned()];
        let samples = ["S1".to_owned(), "S2".to_owned()];
        let Err(error) = build_matrix(&table, &proteins, &samples) else {
            panic!("missing cell was accepted");
        };
        assert!(error
            .to_string()
            .contains("complete protein x sample matrix"));
        assert!(parse_value("", 2).is_err());
        assert!(parse_value("NaN", 2).is_err());
    }

    fn assert_rejected_without_output(
        args: &CorrectBatchesArgs,
        output: &Path,
        expected: &str,
    ) -> TestResult<()> {
        let error = error_message(run_correct_batches(args), expected)?;
        assert!(error.contains(expected), "unexpected error: {error}");
        assert!(
            !output.exists(),
            "output must not be created: {}",
            output.display()
        );
        Ok(())
    }

    #[test]
    fn glob_match_handles_wildcards() {
        assert!(glob_match("*pibaq.tsv", "batchA_pibaq.tsv"));
        assert!(glob_match("*pibaq.tsv", "pibaq.tsv"));
        assert!(!glob_match("*pibaq.tsv", "proteins.tsv"));
        assert!(glob_match("sample?.tsv", "sample1.tsv"));
        assert!(!glob_match("sample?.tsv", "sample12.tsv"));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn batch_labels_use_first_seen_order() -> TestResult<()> {
        let samples = vec![
            "B2-s1".to_string(),
            "B1-s1".to_string(),
            "B2-s2".to_string(),
            "B1-s2".to_string(),
        ];
        assert_eq!(batch_labels(&samples)?, vec![0, 1, 0, 1]);
        Ok(())
    }

    #[test]
    fn invalid_batch_prefix_is_rejected() {
        assert!(batch_prefix("good-s1").is_ok());
        assert!(batch_prefix("-s1").is_err());
    }

    #[test]
    fn sample_id_validation_matches_regex() {
        assert!(is_valid_sample_id("B1-s1"));
        assert!(is_valid_sample_id("PXD000001"));
        assert!(!is_valid_sample_id("B1--s1"));
        assert!(!is_valid_sample_id("B1-"));
        assert!(!is_valid_sample_id("B1_s1"));
        assert!(!is_valid_sample_id("bad sample"));
    }

    fn args_for(folder: &Path, output: &Path) -> CorrectBatchesArgs {
        CorrectBatchesArgs {
            input: folder.to_path_buf(),
            pattern: "*pibaq.tsv".to_string(),
            comment: "#".to_string(),
            sep: "\t".to_string(),
            output: output.to_path_buf(),
            sample_id_column: "SampleID".to_string(),
            protein_id_column: "ProteinName".to_string(),
            pibaq_raw_column: "PiBAQ".to_string(),
            pibaq_corrected_column: "PiBAQBec".to_string(),
            export_anndata: false,
        }
    }

    const BATCH_A: &str = "# synthetic pibaq fixture for correct-batches golden test\n\
ProteinName\tSampleID\tCondition\tPiBAQ\n\
P1\tB1-s1\tctrl\t10.0\n\
P2\tB1-s1\tctrl\t5.0\n\
P3\tB1-s1\tctrl\t1.0\n\
P4\tB1-s1\tctrl\t50.0\n\
P5\tB1-s1\tctrl\t3.0\n\
P1\tB1-s2\tcase\t11.0\n\
P2\tB1-s2\tcase\t6.0\n\
P3\tB1-s2\tcase\t2.0\n\
P4\tB1-s2\tcase\t52.0\n\
P5\tB1-s2\tcase\t3.5\n\
P1\tB1-s3\tcase\t9.5\n\
P2\tB1-s3\tcase\t4.0\n\
P3\tB1-s3\tcase\t1.5\n\
P4\tB1-s3\tcase\t48.0\n\
P5\tB1-s3\tcase\t0.0\n";

    const BATCH_B: &str = "# synthetic pibaq fixture for correct-batches golden test\n\
ProteinName\tSampleID\tCondition\tPiBAQ\n\
P1\tB2-s1\tctrl\t20.0\n\
P2\tB2-s1\tctrl\t8.0\n\
P3\tB2-s1\tctrl\t3.0\n\
P4\tB2-s1\tctrl\t30.0\n\
P5\tB2-s1\tctrl\t6.0\n\
P1\tB2-s2\tcase\t21.0\n\
P2\tB2-s2\tcase\t7.5\n\
P3\tB2-s2\tcase\t2.5\n\
P4\tB2-s2\tcase\t31.0\n\
P5\tB2-s2\tcase\t5.5\n\
P1\tB2-s3\tcase\t19.0\n\
P2\tB2-s3\tcase\t9.0\n\
P3\tB2-s3\tcase\t4.0\n\
P4\tB2-s3\tcase\t29.0\n\
P5\tB2-s3\tcase\t6.5\n";

    const DELIMITED_A: &str = r#"# comma-delimited correct-batches fixture
ProteinName,SampleID,Metadata,PiBAQ
P1,B1-s1,"alpha,beta",10.0
P2,B1-s1,"line one
line two",5.0
P1,B1-s2,plain,11.0
P2,B1-s2,plain,6.0
"#;

    const DELIMITED_B: &str = r#"# comma-delimited correct-batches fixture
ProteinName,SampleID,Metadata,PiBAQ
P1,B2-s1,"quoted ""value""",20.0
P2,B2-s1,plain,8.0
P1,B2-s2,plain,21.0
P2,B2-s2,plain,7.5
"#;

    const COLLISION_A: &str = "ProteinName\tSampleID\tPiBAQ\tPiBAQBec\n\
P1\tB1-s1\t10.0\told\n\
P2\tB1-s1\t5.0\told\n\
P1\tB1-s2\t11.0\told\n\
P2\tB1-s2\t6.0\told\n";

    const COLLISION_B: &str = "ProteinName\tSampleID\tPiBAQ\tPiBAQBec\n\
P1\tB2-s1\t20.0\told\n\
P2\tB2-s1\t8.0\told\n\
P1\tB2-s2\t21.0\told\n\
P2\tB2-s2\t7.5\told\n";

    // Expected PiBAQBec values keyed by (SampleID, ProteinName), from
    // sva 3.58.0: ComBat(X, batch=rep(1:2,each=3), par.prior=TRUE,
    // prior.plots=FALSE, BPPARAM=BiocParallel::SerialParam()). X is the
    // 5-by-6 matrix in BATCH_A/BATCH_B; no covariates or log transform.
    // B1-s3/P5 is an observed zero, distinct from a structurally absent cell.
    const EXPECTED: &[(&str, &str, f64)] = &[
        ("B1-s1", "P1", 14.888611859089949),
        ("B1-s1", "P2", 6.5690968224600965),
        ("B1-s1", "P3", 1.9297166147721312),
        ("B1-s1", "P4", 40.173786157439345),
        ("B1-s1", "P5", 4.684966644482776),
        ("B1-s2", "P1", 15.677789476902493),
        ("B1-s2", "P2", 7.334035336187576),
        ("B1-s2", "P3", 2.726342258417554),
        ("B1-s2", "P4", 41.67158412295979),
        ("B1-s2", "P5", 5.0545060403878574),
        ("B1-s3", "P1", 14.494023050183676),
        ("B1-s3", "P2", 5.804158308732618),
        ("B1-s3", "P3", 2.3280294365948424),
        ("B1-s3", "P4", 38.675988191918904),
        ("B1-s3", "P5", 2.467730269052283),
        ("B2-s1", "P1", 15.138421089351253),
        ("B2-s1", "P2", 6.430633041171619),
        ("B2-s1", "P3", 2.189092103326806),
        ("B2-s1", "P4", 39.873424456324024),
        ("B2-s1", "P5", 4.093086754577451),
        ("B2-s2", "P1", 16.047926019136252),
        ("B2-s2", "P2", 5.9390395451127045),
        ("B2-s2", "P3", 1.7421344916475499),
        ("B2-s2", "P4", 40.911988439474825),
        ("B2-s2", "P5", 3.546244078657096),
        ("B2-s3", "P1", 14.228916159566253),
        ("B2-s3", "P2", 7.41382003328945),
        ("B2-s3", "P3", 3.0830073266853177),
        ("B2-s3", "P4", 38.83486047317322),
        ("B2-s3", "P5", 4.639929430497806),
    ];

    /// Golden test: the Rust command reproduces official sva::ComBat PiBAQBec
    /// values to relative 1e-6 on a synthetic 2-batch / 6-sample / 5-protein
    /// complete dataset (including one explicit observed zero).
    #[test]
    fn correct_batches_matches_sva_oracle() -> TestResult<()> {
        let (_dir_guard, dir) = temp_dir("oracle")?;
        write_file(&dir, "batchA_pibaq.tsv", BATCH_A)?;
        write_file(&dir, "batchB_pibaq.tsv", BATCH_B)?;
        let output = dir.join("corrected.tsv");

        run_correct_batches(&args_for(&dir, &output))?;

        let actual = corrected_by_identity(&output)?;

        assert_eq!(
            actual.len(),
            EXPECTED.len(),
            "row count must match the oracle"
        );
        for &(sample, protein, expected) in EXPECTED {
            let Some(&got) = actual.get(&(sample.to_string(), protein.to_string())) else {
                panic!("missing corrected value for ({sample}, {protein})");
            };
            let tolerance = expected.abs() * 1e-6;
            assert!(
                (got - expected).abs() <= tolerance,
                "({sample}, {protein}): got {got}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn reordered_input_columns_preserve_identity() -> TestResult<()> {
        let (_canonical_dir_guard, canonical_dir) = temp_dir("canonical-order")?;
        write_file(&canonical_dir, "batchA_pibaq.tsv", BATCH_A)?;
        write_file(&canonical_dir, "batchB_pibaq.tsv", BATCH_B)?;
        let canonical_output = canonical_dir.join("corrected.tsv");
        run_correct_batches(&args_for(&canonical_dir, &canonical_output))?;

        let (_reordered_dir_guard, reordered_dir) = temp_dir("reordered-order")?;
        write_file(&reordered_dir, "batchA_pibaq.tsv", BATCH_A)?;
        write_file(
            &reordered_dir,
            "batchB_pibaq.tsv",
            &reorder_first_two_columns(BATCH_B),
        )?;
        let reordered_output = reordered_dir.join("corrected.tsv");
        run_correct_batches(&args_for(&reordered_dir, &reordered_output))?;

        assert_eq!(
            corrected_by_identity(&canonical_output)?,
            corrected_by_identity(&reordered_output)?,
            "column order must not change corrected values by identity"
        );
        Ok(())
    }

    #[test]
    fn selected_delimiter_and_quoted_fields_round_trip() -> TestResult<()> {
        let (_dir_guard, dir) = temp_dir("delimited-output")?;
        write_file(&dir, "batchA_pibaq.csv", DELIMITED_A)?;
        write_file(&dir, "batchB_pibaq.csv", DELIMITED_B)?;
        let output = dir.join("corrected.csv");
        let mut args = args_for(&dir, &output);
        args.pattern = "*pibaq.csv".to_string();
        args.sep = ",".to_string();
        run_correct_batches(&args)?;

        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b',')
            .from_path(&output)?;
        assert_eq!(
            reader.headers()?.iter().collect::<Vec<_>>(),
            vec!["ProteinName", "SampleID", "Metadata", "PiBAQ", "PiBAQBec"]
        );
        let mut metadata = HashMap::new();
        for record in reader.records() {
            let record = record?;
            assert_eq!(record.len(), 5, "every output row must match the header");
            metadata.insert(
                (
                    field(&record, 1)?.to_string(),
                    field(&record, 0)?.to_string(),
                ),
                field(&record, 2)?.to_string(),
            );
        }
        assert_eq!(
            metadata[&(String::from("B1-s1"), String::from("P1"))],
            "alpha,beta"
        );
        assert_eq!(
            metadata[&(String::from("B1-s1"), String::from("P2"))],
            "line one\nline two"
        );
        assert_eq!(
            metadata[&(String::from("B2-s1"), String::from("P1"))],
            "quoted \"value\""
        );
        Ok(())
    }

    #[test]
    fn duplicate_headers_fail_before_output_creation() -> TestResult<()> {
        let (_duplicate_dir_guard, duplicate_dir) = temp_dir("duplicate-header")?;
        write_file(
            &duplicate_dir,
            "batchA_pibaq.tsv",
            "ProteinName\tSampleID\tPiBAQ\tPiBAQ\nP1\tB1-s1\t10\t10\nP1\tB1-s2\t11\t11\n",
        )?;
        write_file(&duplicate_dir, "batchB_pibaq.tsv", BATCH_B)?;
        let duplicate_output = duplicate_dir.join("corrected.tsv");
        let args = args_for(&duplicate_dir, &duplicate_output);
        assert_rejected_without_output(&args, &duplicate_output, "Duplicate columns")
    }

    #[test]
    fn extra_schema_column_fails_before_output_creation() -> TestResult<()> {
        let (_mismatch_dir_guard, mismatch_dir) = temp_dir("schema-mismatch")?;
        write_file(&mismatch_dir, "batchA_pibaq.tsv", BATCH_A)?;
        write_file(
            &mismatch_dir,
            "batchB_pibaq.tsv",
            "ProteinName\tSampleID\tPiBAQ\tExtra\nP1\tB2-s1\t20\textra\nP2\tB2-s1\t8\textra\nP1\tB2-s2\t21\textra\nP2\tB2-s2\t7.5\textra\n",
        )?;
        let mismatch_output = mismatch_dir.join("corrected.tsv");
        let args = args_for(&mismatch_dir, &mismatch_output);
        assert_rejected_without_output(&args, &mismatch_output, "extra columns")
    }

    #[test]
    fn missing_schema_column_fails_before_output_creation() -> TestResult<()> {
        let (_missing_dir_guard, missing_dir) = temp_dir("schema-missing")?;
        write_file(&missing_dir, "batchA_pibaq.tsv", BATCH_A)?;
        write_file(
            &missing_dir,
            "batchB_pibaq.tsv",
            "ProteinName\tSampleID\nP1\tB2-s1\nP2\tB2-s1\nP1\tB2-s2\nP2\tB2-s2\n",
        )?;
        let missing_output = missing_dir.join("corrected.tsv");
        let args = args_for(&missing_dir, &missing_output);
        assert_rejected_without_output(&args, &missing_output, "missing columns")
    }

    #[test]
    fn duplicate_source_roles_fail_before_output_creation() -> TestResult<()> {
        let (_role_dir_guard, role_dir) = temp_dir("source-role")?;
        write_file(&role_dir, "batchA_pibaq.tsv", BATCH_A)?;
        write_file(&role_dir, "batchB_pibaq.tsv", BATCH_B)?;
        let role_output = role_dir.join("corrected.tsv");
        let mut role_args = args_for(&role_dir, &role_output);
        role_args.sample_id_column = "ProteinName".to_string();
        assert_rejected_without_output(&role_args, &role_output, "source columns must be distinct")
    }

    #[test]
    fn corrected_column_collision_fails_before_output_creation() -> TestResult<()> {
        let (_collision_dir_guard, collision_dir) = temp_dir("corrected-column")?;
        write_file(&collision_dir, "batchA_pibaq.tsv", COLLISION_A)?;
        write_file(&collision_dir, "batchB_pibaq.tsv", COLLISION_B)?;
        let collision_output = collision_dir.join("corrected.tsv");
        let args = args_for(&collision_dir, &collision_output);
        assert_rejected_without_output(&args, &collision_output, "already exists")
    }

    #[test]
    fn input_output_collision_does_not_truncate_input() -> TestResult<()> {
        let (_dir_guard, dir) = temp_dir("input-output-collision")?;
        let input = write_file(&dir, "batchA_pibaq.tsv", BATCH_A)?;
        write_file(&dir, "batchB_pibaq.tsv", BATCH_B)?;
        let before = std::fs::read(&input)?;
        let error = error_message(
            run_correct_batches(&args_for(&dir, &input)),
            "an input path must not also be the output path",
        )?;
        assert!(error.contains("also an input file"));
        assert_eq!(std::fs::read(&input)?, before);

        let future_output = dir.join("future_pibaq.tsv");
        let mut args = args_for(&dir, &future_output);
        args.pattern = "*pibaq.tsv".to_string();
        let future_error = error_message(
            run_correct_batches(&args),
            "an output matching the input pattern must be rejected",
        )?;
        assert!(future_error.contains("matches the input pattern"));
        assert!(!future_output.exists());
        Ok(())
    }

    fn field(record: &csv::StringRecord, index: usize) -> TestResult<&str> {
        record
            .get(index)
            .ok_or_else(|| format!("record missing column {index}").into())
    }

    #[test]
    fn export_anndata_writes_h5ad_file() -> TestResult<()> {
        let (_dir_guard, dir) = temp_dir("anndata")?;
        write_file(&dir, "batchA_pibaq.tsv", BATCH_A)?;
        write_file(&dir, "batchB_pibaq.tsv", BATCH_B)?;
        let output = dir.join("corrected.tsv");
        let mut args = args_for(&dir, &output);
        args.export_anndata = true;

        run_correct_batches(&args)?;

        // The `.h5ad` lands next to the TSV with the suffix replaced.
        let h5ad = dir.join("corrected.h5ad");
        assert!(
            h5ad.is_file(),
            "expected AnnData file at {}",
            h5ad.display()
        );

        // Read X / PiBAQBec back with the hdf5 crate and confirm the corrected
        // layer matches the oracle (samples x proteins).
        let file = hdf5_metno::File::open(&h5ad)?;
        let obs: Vec<hdf5_metno::types::VarLenUnicode> =
            file.group("obs")?.dataset("SampleID")?.read_raw()?;
        let obs: Vec<String> = obs.iter().map(|s| s.as_str().to_string()).collect();
        let var: Vec<hdf5_metno::types::VarLenUnicode> =
            file.group("var")?.dataset("ProteinName")?.read_raw()?;
        let var: Vec<String> = var.iter().map(|s| s.as_str().to_string()).collect();

        let x: ndarray::Array2<f64> = file.dataset("X")?.read_2d()?;
        let layer: ndarray::Array2<f64> = file.dataset("layers/PiBAQBec")?.read_2d()?;
        assert_eq!(x.shape(), &[obs.len(), var.len()]);
        assert_eq!(layer.shape(), &[obs.len(), var.len()]);

        let obs_idx = |name: &str| obs.iter().position(|s| s == name);
        let var_idx = |name: &str| var.iter().position(|s| s == name);

        // Corrected layer values match the official sva oracle where present.
        for &(sample, protein, expected) in EXPECTED {
            let (Some(i), Some(j)) = (obs_idx(sample), var_idx(protein)) else {
                panic!("missing ({sample}, {protein}) in AnnData index");
            };
            let got = layer[[i, j]];
            let tolerance = expected.abs() * 1e-6;
            assert!(
                (got - expected).abs() <= tolerance,
                "layer ({sample}, {protein}): got {got}, expected {expected}"
            );
        }

        // An explicit input zero remains present in raw X and receives a
        // corrected ComBat value in the layer.
        let (Some(i), Some(j)) = (obs_idx("B1-s3"), var_idx("P5")) else {
            panic!("missing B1-s3/P5 in AnnData index");
        };
        assert_eq!(x[[i, j]], 0.0, "explicit raw zero must be preserved");

        // Raw X reflects the input piBAQ values.
        let (Some(i), Some(j)) = (obs_idx("B1-s1"), var_idx("P4")) else {
            panic!("missing B1-s1/P4 in AnnData index");
        };
        assert_eq!(x[[i, j]], 50.0);
        Ok(())
    }
}
