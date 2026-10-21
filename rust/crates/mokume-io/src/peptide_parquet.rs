//! Writer for the `features2peptides --save_parquet` peptide intensity matrix.
//!
//! Mirrors Python `mokume.core.write_queue.WriteParquetTask`, which serialises
//! the exact `dataset_df` that `WriteCSVTask` also writes (the
//! `sum_peptidoform_intensities` output). The Arrow schema reproduces what
//! `pa.Schema.from_pandas` infers from that frame:
//!   * `ProteinName`, `PeptideCanonical`: `string`,
//!   * `SampleID`, `Condition`: `dictionary<int8, string>` (pandas Categorical),
//!   * `BioReplicate`: `int32`,
//!   * `NormIntensity`: `float` (float32).
//!
//! For `aggregation_level == "run"`, two extra columns are appended before
//! `NormIntensity`:
//!   * `Run`: `string`,
//!   * `TechReplicate`: `int64`.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, DictionaryArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeStringArray, StringArray, StringDictionaryBuilder,
};
use arrow::datatypes::{
    ArrowDictionaryKeyType, DataType, Field, Int16Type, Int32Type, Int8Type, Schema,
};
use arrow::record_batch::RecordBatch;
use mokume_core::{MokumeError, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

/// Column header for the protein identifier (Python `PROTEIN_NAME`).
const PROTEIN_NAME: &str = "ProteinName";
/// Preferred peptide-sequence column (Python `PEPTIDE_CANONICAL`).
const PEPTIDE_CANONICAL: &str = "PeptideCanonical";
/// Fallback peptide-sequence column (Python `PEPTIDE_SEQUENCE`).
const PEPTIDE_SEQUENCE: &str = "PeptideSequence";
/// Column header for the sample identifier (Python `SAMPLE_ID`).
const SAMPLE_ID: &str = "SampleID";
/// Optional condition column (Python `CONDITION`).
const CONDITION: &str = "Condition";
/// Column header for the normalized intensity (Python `NORM_INTENSITY`).
const NORM_INTENSITY: &str = "NormIntensity";

/// One row of the peptide intensity matrix shared by the CSV and parquet
/// writers, so both serialisations are byte-for-byte consistent on the values.
#[derive(Debug, Clone, PartialEq)]
pub struct PeptideParquetRow {
    pub protein_name: String,
    pub peptide_canonical: String,
    pub sample_id: String,
    pub bio_replicate: i32,
    pub condition: String,
    /// Run identifier and technical replicate, populated only when the matrix
    /// is aggregated at run level (`aggregation_level == "run"`).
    pub run: Option<String>,
    pub tech_replicate: Option<i64>,
    pub norm_intensity: f64,
}

/// One raw row read from a peptide-matrix parquet, before the `peptides2protein`
/// intensity filtering / condition-default logic is applied. Mirrors what
/// `pd.read_parquet` yields per row: present cells carry values, null cells are
/// `None`. The caller applies the same `dropna` + `> 0` filter and `"Empty"`
/// condition default as the CSV path so both inputs behave identically.
#[derive(Debug, Clone, PartialEq)]
pub struct RawPeptideRow {
    pub protein: String,
    pub sample: String,
    pub peptide: Option<String>,
    pub condition: Option<String>,
    pub intensity: Option<f64>,
}

/// A peptide-matrix parquet parsed into raw rows plus the column-presence flags
/// the caller needs (whether `Condition` and a peptide-sequence column existed).
#[derive(Debug, Clone, PartialEq)]
pub struct RawPeptideTable {
    pub rows: Vec<RawPeptideRow>,
    pub has_condition: bool,
    pub has_peptide: bool,
}

/// Read a peptide-level matrix from a parquet file, mirroring Python's
/// `pd.read_parquet` on the `peptides2protein` input. Required columns are
/// `ProteinName`, `SampleID` and `NormIntensity`; the peptide column is
/// `PeptideCanonical` (preferred) or `PeptideSequence`; `Condition` is optional.
///
/// `SampleID` and `Condition` are dictionary-encoded (pandas `Categorical`) when
/// mokume itself wrote the parquet, so string extraction transparently decodes
/// `Dictionary(int*, Utf8)` as well as plain `Utf8` / `LargeUtf8`. The intensity
/// is read from `Float32` / `Float64` / `Int32` / `Int64` (or a numeric string,
/// matching Python's `astype(float)`); null cells become `None`.
pub fn read_peptide_parquet(path: &Path) -> Result<RawPeptideTable> {
    read_peptide_parquet_impl(path, false)
}

/// Preserve modification and charge when both species columns are available.
/// Canonical-only matrices retain their supplied peptide identifiers.
pub fn read_peptide_parquet_with_species(path: &Path) -> Result<RawPeptideTable> {
    read_peptide_parquet_impl(path, true)
}

fn read_peptide_parquet_impl(path: &Path, preserve_species: bool) -> Result<RawPeptideTable> {
    let file = File::open(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|source| parquet_read_error(path, &source.to_string()))?;

    let schema = Arc::clone(builder.schema());
    let protein_index = require_column(&schema, &[PROTEIN_NAME])?;
    let sample_index = require_column(&schema, &[SAMPLE_ID])?;
    let intensity_index = require_column(&schema, &[NORM_INTENSITY])?;
    let condition_index = find_column(&schema, &[CONDITION]);
    let peptide_index = find_column(&schema, &[PEPTIDE_CANONICAL, PEPTIDE_SEQUENCE]);
    let species_indices = preserve_species
        .then(|| {
            find_column(&schema, &[PEPTIDE_SEQUENCE])
                .zip(find_column(&schema, &["PrecursorCharge"]))
        })
        .flatten();
    let has_condition = condition_index.is_some();
    let has_peptide = peptide_index.is_some();

    let reader = builder
        .build()
        .map_err(|source| parquet_read_error(path, &source.to_string()))?;

    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|source| parquet_read_error(path, &source.to_string()))?;
        let protein = batch.column(protein_index).as_ref();
        let sample = batch.column(sample_index).as_ref();
        let intensity = batch.column(intensity_index).as_ref();
        let condition = condition_index.map(|index| batch.column(index).as_ref());

        for row in 0..batch.num_rows() {
            rows.push(RawPeptideRow {
                protein: string_value(protein, row, PROTEIN_NAME)?.unwrap_or_default(),
                sample: string_value(sample, row, SAMPLE_ID)?.unwrap_or_default(),
                peptide: parquet_peptide_id(&batch, row, peptide_index, species_indices)?,
                condition: match condition {
                    Some(array) => string_value(array, row, CONDITION)?,
                    None => None,
                },
                intensity: f64_value(intensity, row, NORM_INTENSITY)?,
            });
        }
    }

    Ok(RawPeptideTable {
        rows,
        has_condition,
        has_peptide,
    })
}

fn parquet_peptide_id(
    batch: &RecordBatch,
    row: usize,
    peptide_index: Option<usize>,
    species_indices: Option<(usize, usize)>,
) -> Result<Option<String>> {
    if let Some((sequence, charge)) = species_indices {
        Ok(
            string_value(batch.column(sequence).as_ref(), row, PEPTIDE_SEQUENCE)?
                .zip(f64_value(
                    batch.column(charge).as_ref(),
                    row,
                    "PrecursorCharge",
                )?)
                .filter(|(_, charge)| charge.is_finite() && *charge > 0.0 && charge.fract() == 0.0)
                .map(|(sequence, charge)| format!("{sequence}|z{charge}")),
        )
    } else {
        peptide_index
            .map(|index| string_value(batch.column(index).as_ref(), row, PEPTIDE_CANONICAL))
            .transpose()
            .map(Option::flatten)
    }
}

/// Resolve the first present column among `candidates`, returning its index.
fn find_column(schema: &Schema, candidates: &[&str]) -> Option<usize> {
    candidates
        .iter()
        .find_map(|name| schema.index_of(name).ok())
}

/// Like [`find_column`] but errors when none of the candidates is present,
/// mirroring Python's "Could not find ... column" `UsageError`.
fn require_column(schema: &Schema, candidates: &[&str]) -> Result<usize> {
    find_column(schema, candidates).ok_or_else(|| {
        invalid_input(format!(
            "could not find required column `{}` in the peptide parquet",
            candidates.join(" | ")
        ))
    })
}

/// Extract a UTF-8 string cell, decoding `Utf8`, `LargeUtf8`, and dictionary
/// (`Categorical`) encodings. Null cells return `None`.
fn string_value(array: &dyn Array, row: usize, column: &str) -> Result<Option<String>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Utf8 => Ok(Some(
            downcast::<StringArray>(array, column)?
                .value(row)
                .to_owned(),
        )),
        DataType::LargeUtf8 => Ok(Some(
            downcast::<LargeStringArray>(array, column)?
                .value(row)
                .to_owned(),
        )),
        DataType::Dictionary(key, value) if value.as_ref() == &DataType::Utf8 => {
            dictionary_string_value(array, row, key.as_ref(), column)
        }
        _ => Err(unsupported_column(column, array)),
    }
}

/// Decode a single value from a `Dictionary(int*, Utf8)` array, dispatching on
/// the integer key type. A pandas `Categorical`'s codes are signed `int8` /
/// `int16` / `int32` chosen by category count (`int8` for the 120-sample matrix
/// here), which is what `pyarrow` carries into the parquet dictionary index.
fn dictionary_string_value(
    array: &dyn Array,
    row: usize,
    key_type: &DataType,
    column: &str,
) -> Result<Option<String>> {
    match key_type {
        DataType::Int8 => decode_dictionary::<Int8Type>(array, row, column),
        DataType::Int16 => decode_dictionary::<Int16Type>(array, row, column),
        DataType::Int32 => decode_dictionary::<Int32Type>(array, row, column),
        _ => Err(unsupported_column(column, array)),
    }
}

/// Look up `array[row]` in a dictionary array whose keys are `K` and whose values
/// are a `StringArray`. Returns `None` when the key cell is null.
fn decode_dictionary<K>(array: &dyn Array, row: usize, column: &str) -> Result<Option<String>>
where
    K: ArrowDictionaryKeyType,
    K::Native: TryInto<usize>,
{
    let dictionary = downcast::<DictionaryArray<K>>(array, column)?;
    if dictionary.is_null(row) {
        return Ok(None);
    }
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            invalid_input(format!("dictionary column `{column}` values are not utf8"))
        })?;
    let key = dictionary.keys().value(row);
    let index: usize = key
        .try_into()
        .map_err(|_| invalid_input(format!("dictionary column `{column}` key is out of range")))?;
    Ok(Some(values.value(index).to_owned()))
}

/// Extract a numeric cell as `f64`, accepting the float/int encodings a peptide
/// matrix may use, plus a numeric string (matching Python's `astype(float)`).
/// Null cells return `None`.
fn f64_value(array: &dyn Array, row: usize, column: &str) -> Result<Option<f64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Float32 => Ok(Some(f64::from(
            downcast::<Float32Array>(array, column)?.value(row),
        ))),
        DataType::Float64 => Ok(Some(downcast::<Float64Array>(array, column)?.value(row))),
        DataType::Int8 => Ok(Some(f64::from(
            downcast::<Int8Array>(array, column)?.value(row),
        ))),
        DataType::Int16 => Ok(Some(f64::from(
            downcast::<Int16Array>(array, column)?.value(row),
        ))),
        DataType::Int32 => Ok(Some(f64::from(
            downcast::<Int32Array>(array, column)?.value(row),
        ))),
        DataType::Int64 => Ok(Some(
            downcast::<Int64Array>(array, column)?.value(row) as f64
        )),
        DataType::Utf8 => {
            parse_numeric_string(downcast::<StringArray>(array, column)?.value(row), column)
        }
        DataType::LargeUtf8 => parse_numeric_string(
            downcast::<LargeStringArray>(array, column)?.value(row),
            column,
        ),
        _ => Err(unsupported_column(column, array)),
    }
}

/// Parse a numeric cell stored as text; an empty cell is treated as missing.
fn parse_numeric_string(raw: &str, column: &str) -> Result<Option<f64>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed
        .parse::<f64>()
        .map(Some)
        .map_err(|_| invalid_input(format!("`{trimmed}` in column `{column}` is not numeric")))
}

fn downcast<'a, T: 'static>(array: &'a dyn Array, column: &str) -> Result<&'a T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| unsupported_column(column, array))
}

fn unsupported_column(column: &str, array: &dyn Array) -> MokumeError {
    invalid_input(format!(
        "unsupported peptide-parquet column `{column}` type: {}",
        array.data_type()
    ))
}

fn parquet_read_error(path: &Path, message: &str) -> MokumeError {
    invalid_input(format!("failed to read `{}`: {message}", path.display()))
}

/// Write the peptide intensity matrix to a parquet file, matching Python's
/// `WriteParquetTask` schema and column order. The `run_mode` flag selects the
/// run-level schema (with `Run` / `TechReplicate` columns) over the sample-level
/// schema, mirroring Python's `aggregation_level`.
pub fn write_peptide_parquet(
    path: &Path,
    rows: &[PeptideParquetRow],
    run_mode: bool,
) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|source| MokumeError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let schema = Arc::new(peptide_schema(run_mode));
    let batch = peptide_batch(&schema, rows, run_mode)?;

    let file = File::create(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|source| parquet_error(path, &source.to_string()))?;
    writer
        .write(&batch)
        .map_err(|source| parquet_error(path, &source.to_string()))?;
    writer
        .close()
        .map_err(|source| parquet_error(path, &source.to_string()))?;
    Ok(())
}

fn dictionary_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        false,
    )
}

fn peptide_schema(run_mode: bool) -> Schema {
    let mut fields = vec![
        Field::new(PROTEIN_NAME, DataType::Utf8, false),
        Field::new(PEPTIDE_CANONICAL, DataType::Utf8, false),
        dictionary_field(SAMPLE_ID),
        Field::new("BioReplicate", DataType::Int32, false),
        dictionary_field(CONDITION),
    ];
    if run_mode {
        fields.push(Field::new("Run", DataType::Utf8, false));
        fields.push(Field::new("TechReplicate", DataType::Int64, false));
    }
    fields.push(Field::new(NORM_INTENSITY, DataType::Float32, false));
    Schema::new(fields)
}

fn dictionary_array(values: impl Iterator<Item = String>) -> Result<ArrayRef> {
    let mut builder = StringDictionaryBuilder::<Int8Type>::new();
    for value in values {
        builder
            .append(value)
            .map_err(|source| invalid_input(format!("dictionary encode failed: {source}")))?;
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn peptide_batch(
    schema: &Arc<Schema>,
    rows: &[PeptideParquetRow],
    run_mode: bool,
) -> Result<RecordBatch> {
    let protein = Arc::new(StringArray::from(
        rows.iter()
            .map(|row| row.protein_name.as_str())
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let peptide = Arc::new(StringArray::from(
        rows.iter()
            .map(|row| row.peptide_canonical.as_str())
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let sample = dictionary_array(rows.iter().map(|row| row.sample_id.clone()))?;
    let bio = Arc::new(Int32Array::from(
        rows.iter().map(|row| row.bio_replicate).collect::<Vec<_>>(),
    )) as ArrayRef;
    let condition = dictionary_array(rows.iter().map(|row| row.condition.clone()))?;
    let intensity = Arc::new(Float32Array::from(
        rows.iter()
            .map(|row| row.norm_intensity as f32)
            .collect::<Vec<_>>(),
    )) as ArrayRef;

    let mut columns = vec![protein, peptide, sample, bio, condition];
    if run_mode {
        let run = Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.run.as_deref().unwrap_or_default())
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let tech = Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.tech_replicate.unwrap_or(1))
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        columns.push(run);
        columns.push(tech);
    }
    columns.push(intensity);

    RecordBatch::try_new(Arc::clone(schema), columns)
        .map_err(|source| invalid_input(format!("failed to build peptide batch: {source}")))
}

fn parquet_error(path: &Path, message: &str) -> MokumeError {
    invalid_input(format!("failed to write `{}`: {message}", path.display()))
}

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, DictionaryArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    type TestError = Box<dyn std::error::Error>;
    type TestResult = std::result::Result<(), TestError>;

    fn read_back(path: &Path) -> std::result::Result<RecordBatch, TestError> {
        let file = File::open(path)?;
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        reader.next().expect("empty parquet").map_err(Into::into)
    }

    fn schema_names(batch: &RecordBatch) -> Vec<String> {
        batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    fn dictionary_strings(array: &dyn Array) -> std::result::Result<Vec<String>, TestError> {
        let dict = array
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .expect("expected dictionary array");
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("dictionary values are not utf8");
        let mut out = Vec::with_capacity(dict.len());
        for index in 0..dict.len() {
            let key = dict.keys().value(index);
            out.push(values.value(key as usize).to_owned());
        }
        Ok(out)
    }

    #[test]
    fn sample_mode_schema_and_values_match_python() -> TestResult {
        let dir = tempfile::Builder::new()
            .prefix(&format!("mokume_pq_sample_{}_", std::process::id()))
            .tempdir()?;
        let path = dir.path().join("pep.parquet");
        let rows = vec![PeptideParquetRow {
            protein_name: "P1".to_owned(),
            peptide_canonical: "APEPTIDECK".to_owned(),
            sample_id: "run1".to_owned(),
            bio_replicate: 1,
            condition: "run1".to_owned(),
            run: None,
            tech_replicate: None,
            norm_intensity: 200.0,
        }];
        write_peptide_parquet(&path, &rows, false)?;

        let batch = read_back(&path)?;
        assert_eq!(
            schema_names(&batch),
            vec![
                "ProteinName",
                "PeptideCanonical",
                "SampleID",
                "BioReplicate",
                "Condition",
                "NormIntensity",
            ]
        );
        assert_eq!(batch.column(3).data_type(), &DataType::Int32);
        assert_eq!(batch.column(5).data_type(), &DataType::Float32);
        assert_eq!(
            dictionary_strings(batch.column(2))?,
            vec!["run1".to_owned()]
        );
        let intensity = batch
            .column(5)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("NormIntensity is not float32");
        assert_eq!(intensity.value(0), 200.0_f32);
        Ok(())
    }

    #[test]
    fn run_mode_appends_run_and_tech_replicate_before_intensity() -> TestResult {
        let dir = tempfile::Builder::new()
            .prefix(&format!("mokume_pq_run_{}_", std::process::id()))
            .tempdir()?;
        let path = dir.path().join("pep_run.parquet");
        let rows = vec![PeptideParquetRow {
            protein_name: "P1".to_owned(),
            peptide_canonical: "PEPTIDEAK".to_owned(),
            sample_id: "run2".to_owned(),
            bio_replicate: 1,
            condition: "run2".to_owned(),
            run: Some("run2".to_owned()),
            tech_replicate: Some(1),
            norm_intensity: 300.0,
        }];
        write_peptide_parquet(&path, &rows, true)?;

        let batch = read_back(&path)?;
        assert_eq!(
            schema_names(&batch),
            vec![
                "ProteinName",
                "PeptideCanonical",
                "SampleID",
                "BioReplicate",
                "Condition",
                "Run",
                "TechReplicate",
                "NormIntensity",
            ]
        );
        assert_eq!(batch.column(5).data_type(), &DataType::Utf8);
        assert_eq!(batch.column(6).data_type(), &DataType::Int64);
        assert_eq!(batch.column(7).data_type(), &DataType::Float32);
        Ok(())
    }

    #[test]
    fn read_peptide_parquet_round_trips_sample_mode() -> TestResult {
        let dir = tempfile::Builder::new()
            .prefix(&format!("mokume_pq_read_{}_", std::process::id()))
            .tempdir()?;
        let path = dir.path().join("read_sample.parquet");
        let rows = vec![
            PeptideParquetRow {
                protein_name: "P1".to_owned(),
                peptide_canonical: "APEPTIDECK".to_owned(),
                sample_id: "S1".to_owned(),
                bio_replicate: 1,
                condition: "A".to_owned(),
                run: None,
                tech_replicate: None,
                norm_intensity: 200.0,
            },
            PeptideParquetRow {
                protein_name: "P2".to_owned(),
                peptide_canonical: "ALYAAEK".to_owned(),
                sample_id: "S2".to_owned(),
                bio_replicate: 2,
                condition: "B".to_owned(),
                run: None,
                tech_replicate: None,
                norm_intensity: 50.0,
            },
        ];
        write_peptide_parquet(&path, &rows, false)?;

        let table = read_peptide_parquet(&path)?;
        assert_eq!(read_peptide_parquet_with_species(&path)?, table);
        assert!(table.has_condition);
        assert!(table.has_peptide);
        assert_eq!(
            table.rows,
            vec![
                RawPeptideRow {
                    protein: "P1".to_owned(),
                    sample: "S1".to_owned(),
                    peptide: Some("APEPTIDECK".to_owned()),
                    condition: Some("A".to_owned()),
                    intensity: Some(200.0),
                },
                RawPeptideRow {
                    protein: "P2".to_owned(),
                    sample: "S2".to_owned(),
                    peptide: Some("ALYAAEK".to_owned()),
                    condition: Some("B".to_owned()),
                    intensity: Some(50.0),
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn read_peptide_parquet_ignores_run_level_extra_columns() -> TestResult {
        let dir = tempfile::Builder::new()
            .prefix(&format!("mokume_pq_read_run_{}_", std::process::id()))
            .tempdir()?;
        let path = dir.path().join("read_run.parquet");
        let rows = vec![PeptideParquetRow {
            protein_name: "P1".to_owned(),
            peptide_canonical: "PEPTIDEAK".to_owned(),
            sample_id: "S1".to_owned(),
            bio_replicate: 1,
            condition: "A".to_owned(),
            run: Some("run2".to_owned()),
            tech_replicate: Some(1),
            norm_intensity: 300.0,
        }];
        write_peptide_parquet(&path, &rows, true)?;

        let table = read_peptide_parquet(&path)?;
        assert!(table.has_condition);
        assert!(table.has_peptide);
        assert_eq!(
            table.rows,
            vec![RawPeptideRow {
                protein: "P1".to_owned(),
                sample: "S1".to_owned(),
                peptide: Some("PEPTIDEAK".to_owned()),
                condition: Some("A".to_owned()),
                intensity: Some(300.0),
            }]
        );
        Ok(())
    }

    #[test]
    fn read_peptide_species_retains_modifications_and_int16_charge() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("species.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("ProteinName", DataType::Utf8, false),
            Field::new("SampleID", DataType::Utf8, false),
            Field::new("NormIntensity", DataType::Float64, false),
            Field::new("PeptideCanonical", DataType::Utf8, false),
            Field::new("PeptideSequence", DataType::Utf8, false),
            Field::new("PrecursorCharge", DataType::Int16, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["P", "P", "P"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["S1", "S1", "S2"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(StringArray::from(vec!["PEPTIDEK", "PEPTIDEK", "PEPTIDEK"])),
                Arc::new(StringArray::from(vec![
                    "PEPTIDEK",
                    "PEP[UNIMOD:35]TIDEK",
                    "PEPTIDEK",
                ])),
                Arc::new(Int16Array::from(vec![Some(2), Some(3), None])),
            ],
        )?;
        let mut writer = ArrowWriter::try_new(File::create(&path)?, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        let species = read_peptide_parquet_with_species(&path)?;
        assert_eq!(species.rows[0].peptide.as_deref(), Some("PEPTIDEK|z2"));
        assert_eq!(
            species.rows[1].peptide.as_deref(),
            Some("PEP[UNIMOD:35]TIDEK|z3")
        );
        assert_eq!(species.rows[2].peptide, None);
        assert_eq!(
            read_peptide_parquet(&path)?.rows[1].peptide.as_deref(),
            Some("PEPTIDEK")
        );
        Ok(())
    }

    #[test]
    fn read_peptide_parquet_errors_on_missing_required_column() -> TestResult {
        use arrow::array::Float32Array;

        let dir = tempfile::Builder::new()
            .prefix(&format!("mokume_pq_read_bad_{}_", std::process::id()))
            .tempdir()?;
        let path = dir.path().join("read_bad.parquet");

        // A parquet without `SampleID` must be rejected, not silently mis-read.
        let schema = Arc::new(Schema::new(vec![
            Field::new("ProteinName", DataType::Utf8, false),
            Field::new("NormIntensity", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["P1"])) as ArrayRef,
                Arc::new(Float32Array::from(vec![10.0_f32])) as ArrayRef,
            ],
        )?;
        let file = File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        assert!(read_peptide_parquet(&path).is_err());
        Ok(())
    }
}
