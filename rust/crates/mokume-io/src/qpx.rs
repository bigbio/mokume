use std::fs::File;
use std::path::Path;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, LargeListArray, LargeStringArray, ListArray, StringArray, StructArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use mokume_core::{MokumeError, Result};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

#[derive(Debug, Clone, PartialEq)]
pub struct QpxFeatureRecord {
    pub sequence: String,
    pub peptidoform: String,
    pub charge: i32,
    pub run_file_name: String,
    pub sample_accession: Option<String>,
    pub protein_accessions: Vec<String>,
    pub anchor_protein: Option<String>,
    pub unique: Option<bool>,
    pub is_decoy: Option<bool>,
    pub peptide_qvalue: Option<f64>,
    pub pg_global_qvalue: Option<f64>,
    /// The requested `additional_scores` entry, populated only by
    /// [`flatten_qpx_batch_with_score`].
    pub selected_score: Option<QpxScoreValue>,
    pub label: Option<String>,
    pub intensity: f64,
    /// First record flattened from one parquet row (rows of TMT channels follow it).
    pub row_start: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpxScoreValue {
    pub value: f64,
    pub higher_better: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QpxIntensityEntry {
    pub sample_accession: Option<String>,
    pub label: Option<String>,
    pub intensity: f64,
}

pub struct QpxParquetReader {
    inner: ParquetRecordBatchReader,
}

impl QpxParquetReader {
    /// Returns the next decoded Arrow record batch without flattening it into
    /// `QpxFeatureRecord`s. Callers can run [`flatten_qpx_batch`] on the result,
    /// optionally in parallel, while preserving batch order.
    pub fn next_raw_batch(&mut self) -> Option<Result<RecordBatch>> {
        self.inner.next().map(|batch| {
            batch.map_err(|source| invalid_input(format!("failed to read QPX batch: {source}")))
        })
    }

    pub fn open(path: impl AsRef<Path>, batch_size: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(MokumeError::MissingInput { path });
        }

        let file = File::open(&path).map_err(|source| MokumeError::Io {
            path: path.clone(),
            source,
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|source| {
            invalid_input(format!(
                "failed to open QPX parquet `{}`: {source}",
                path.display()
            ))
        })?;
        let inner = builder
            .with_batch_size(batch_size)
            .build()
            .map_err(|source| {
                invalid_input(format!(
                    "failed to build QPX batch reader `{}`: {source}",
                    path.display()
                ))
            })?;

        Ok(Self { inner })
    }
}

impl Iterator for QpxParquetReader {
    type Item = Result<Vec<QpxFeatureRecord>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|batch| {
            batch
                .map_err(|source| invalid_input(format!("failed to read QPX batch: {source}")))
                .and_then(|batch| flatten_qpx_batch(&batch))
        })
    }
}

pub fn flatten_qpx_batch(batch: &RecordBatch) -> Result<Vec<QpxFeatureRecord>> {
    flatten_qpx_batch_inner(batch, None)
}

/// Flatten a QPX batch while selecting one exact `additional_scores.score_name`.
/// Rows without that score retain `selected_score = None`; the pipeline decides
/// whether missing values are filtered or constitute an unavailable score.
pub fn flatten_qpx_batch_with_score(
    batch: &RecordBatch,
    score_name: &str,
) -> Result<Vec<QpxFeatureRecord>> {
    flatten_qpx_batch_inner(batch, Some(score_name))
}

fn flatten_qpx_batch_inner(
    batch: &RecordBatch,
    score_name: Option<&str>,
) -> Result<Vec<QpxFeatureRecord>> {
    let columns = QpxBatchColumns::from_batch(batch, score_name)?;
    let mut records = Vec::new();
    for row in 0..batch.num_rows() {
        flatten_qpx_row(&columns, score_name, row, &mut records)?;
    }
    Ok(records)
}

struct QpxBatchColumns<'a> {
    sequence: &'a dyn Array,
    peptidoform: &'a dyn Array,
    charge: &'a dyn Array,
    run_file_name: &'a dyn Array,
    intensities: &'a dyn Array,
    protein_groups: &'a dyn Array,
    anchor_protein: Option<&'a dyn Array>,
    unique: Option<&'a dyn Array>,
    is_decoy: Option<&'a dyn Array>,
    peptide_qvalue: Option<&'a dyn Array>,
    pg_global_qvalue: Option<&'a dyn Array>,
    additional_scores: Option<&'a dyn Array>,
}

impl<'a> QpxBatchColumns<'a> {
    fn from_batch(batch: &'a RecordBatch, score_name: Option<&str>) -> Result<Self> {
        Ok(Self {
            sequence: column_by_names(batch, &["sequence", "Sequence"])?,
            peptidoform: column_by_names(batch, &["peptidoform", "modified_sequence"])?,
            charge: column_by_names(batch, &["charge", "precursor_charge"])?,
            run_file_name: column_by_names(
                batch,
                &["run_file_name", "reference_file_name", "run", "raw_file"],
            )?,
            intensities: column_by_names(batch, &["intensities", "primary_intensities"])?,
            protein_groups: column_by_names(
                batch,
                &["pg_accessions", "protein_accessions", "proteins"],
            )?,
            anchor_protein: optional_column_by_names(batch, &["anchor_protein", "protein"]),
            unique: optional_column_by_names(batch, &["unique", "is_unique"]),
            is_decoy: optional_column_by_names(batch, &["is_decoy", "decoy"]),
            peptide_qvalue: optional_column_by_names(batch, &["peptide_qvalue", "peptide_q_value"]),
            pg_global_qvalue: optional_column_by_names(
                batch,
                &["pg_global_qvalue", "protein_qvalue"],
            ),
            additional_scores: requested_score_column(batch, score_name)?,
        })
    }
}

struct QpxRowMetadata {
    sequence: String,
    peptidoform: String,
    charge: i32,
    run_file_name: String,
    protein_accessions: Vec<String>,
    anchor_protein: Option<String>,
    unique: Option<bool>,
    is_decoy: Option<bool>,
    peptide_qvalue: Option<f64>,
    pg_global_qvalue: Option<f64>,
    selected_score: Option<QpxScoreValue>,
}

impl QpxRowMetadata {
    fn from_columns(
        columns: &QpxBatchColumns<'_>,
        score_name: Option<&str>,
        row: usize,
    ) -> Result<Self> {
        let anchor_protein = columns
            .anchor_protein
            .map(|column| optional_string_value(column, row, "anchor_protein"))
            .transpose()?
            .flatten();
        let mut protein_accessions = list_string_values(
            columns.protein_groups,
            row,
            "pg_accessions",
            &["accession", "protein_accession", "protein"],
        )?;
        if protein_accessions.is_empty() {
            protein_accessions.extend(anchor_protein.iter().cloned());
        }
        let (peptide_qvalue, pg_global_qvalue) =
            optional_qvalue_values(columns.peptide_qvalue, columns.pg_global_qvalue, row)?;
        Ok(Self {
            sequence: required_string_value(columns.sequence, row, "sequence")?,
            peptidoform: required_string_value(columns.peptidoform, row, "peptidoform")?,
            charge: required_i32_value(columns.charge, row, "charge")?,
            run_file_name: required_string_value(columns.run_file_name, row, "run_file_name")?,
            protein_accessions,
            anchor_protein,
            unique: optional_bool_column_value(columns.unique, row, "unique")?,
            is_decoy: optional_bool_column_value(columns.is_decoy, row, "is_decoy")?,
            peptide_qvalue,
            pg_global_qvalue,
            selected_score: selected_score_value(columns.additional_scores, score_name, row)?,
        })
    }
}

fn optional_bool_column_value(
    column: Option<&dyn Array>,
    row: usize,
    name: &str,
) -> Result<Option<bool>> {
    column
        .map(|column| optional_bool_value(column, row, name))
        .transpose()
        .map(Option::flatten)
}

fn flatten_qpx_row(
    columns: &QpxBatchColumns<'_>,
    score_name: Option<&str>,
    row: usize,
    records: &mut Vec<QpxFeatureRecord>,
) -> Result<()> {
    let metadata = QpxRowMetadata::from_columns(columns, score_name, row)?;
    let entries = intensity_entries(columns.intensities, row)?;
    let labels_are_runs = intensity_labels_are_runs(&entries, &metadata.run_file_name);
    records.extend(
        entries
            .into_iter()
            .filter(|entry| entry.intensity.is_finite())
            .enumerate()
            .map(|(index, entry)| {
                qpx_feature_record(&metadata, entry, labels_are_runs, index == 0)
            }),
    );
    Ok(())
}

/// LFQ intensities are labeled by run file; isobaric intensities are labeled
/// by reporter channel. Matching the anchor run distinguishes the two layouts.
fn intensity_labels_are_runs(entries: &[QpxIntensityEntry], run_file_name: &str) -> bool {
    let row_key = crate::sdrf::normalize_file_key(run_file_name);
    entries.iter().any(|entry| {
        entry
            .label
            .as_deref()
            .is_some_and(|label| crate::sdrf::normalize_file_key(label) == row_key)
    })
}

fn qpx_feature_record(
    metadata: &QpxRowMetadata,
    entry: QpxIntensityEntry,
    labels_are_runs: bool,
    row_start: bool,
) -> QpxFeatureRecord {
    let (run_file_name, label) = if labels_are_runs {
        (
            entry
                .label
                .unwrap_or_else(|| metadata.run_file_name.clone()),
            None,
        )
    } else {
        (metadata.run_file_name.clone(), entry.label)
    };
    QpxFeatureRecord {
        sequence: metadata.sequence.clone(),
        peptidoform: metadata.peptidoform.clone(),
        charge: metadata.charge,
        run_file_name,
        sample_accession: entry.sample_accession,
        protein_accessions: metadata.protein_accessions.clone(),
        anchor_protein: metadata.anchor_protein.clone(),
        unique: metadata.unique,
        is_decoy: metadata.is_decoy,
        peptide_qvalue: metadata.peptide_qvalue,
        pg_global_qvalue: metadata.pg_global_qvalue,
        selected_score: metadata.selected_score,
        label,
        intensity: entry.intensity,
        row_start,
    }
}

fn requested_score_column<'a>(
    batch: &'a RecordBatch,
    score_name: Option<&str>,
) -> Result<Option<&'a dyn Array>> {
    score_name
        .map(|_| column_by_names(batch, &["additional_scores"]))
        .transpose()
}

fn selected_score_value(
    scores: Option<&dyn Array>,
    name: Option<&str>,
    row: usize,
) -> Result<Option<QpxScoreValue>> {
    scores
        .zip(name)
        .map(|(scores, name)| named_score_value(scores, row, name))
        .transpose()
        .map(Option::flatten)
}

fn named_score_value(array: &dyn Array, row: usize, name: &str) -> Result<Option<QpxScoreValue>> {
    let Some(values) = list_row_values(array, row, "additional_scores")? else {
        return Ok(None);
    };
    let scores = downcast_array::<StructArray>(values.as_ref(), "additional_scores")?;
    let names = struct_column_by_names(scores, &["score_name"])
        .ok_or_else(|| invalid_input("missing `additional_scores.score_name` field"))?;
    let values = struct_column_by_names(scores, &["score_value"])
        .ok_or_else(|| invalid_input("missing `additional_scores.score_value` field"))?;
    let directions = struct_column_by_names(scores, &["higher_better"])
        .ok_or_else(|| invalid_input("missing `additional_scores.higher_better` field"))?;

    let Some(index) = named_score_index(names, scores.len(), name)? else {
        return Ok(None);
    };
    let value = required_score_value(values, index, name, row)?;
    let higher_better = required_score_direction(directions, index, name, row)?;
    Ok(Some(QpxScoreValue {
        value,
        higher_better,
    }))
}

fn named_score_index(names: &dyn Array, len: usize, name: &str) -> Result<Option<usize>> {
    for index in 0..len {
        if optional_string_value(names, index, "additional_scores.score_name")?.as_deref()
            == Some(name)
        {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

fn required_score_value(array: &dyn Array, index: usize, name: &str, row: usize) -> Result<f64> {
    optional_f64_value(array, index, "additional_scores.score_value")?
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            invalid_input(format!(
                "QPX score `{name}` has a null or non-finite score_value at row {row}"
            ))
        })
}

fn required_score_direction(
    array: &dyn Array,
    index: usize,
    name: &str,
    row: usize,
) -> Result<bool> {
    optional_bool_value(array, index, "additional_scores.higher_better")?.ok_or_else(|| {
        invalid_input(format!(
            "QPX score `{name}` has a null higher_better flag at row {row}"
        ))
    })
}

fn optional_qvalue_values(
    peptide: Option<&dyn Array>,
    protein: Option<&dyn Array>,
    row: usize,
) -> Result<(Option<f64>, Option<f64>)> {
    let read = |column: Option<&dyn Array>, name: &str| {
        column
            .map(|values| optional_f64_value(values, row, name))
            .transpose()
            .map(Option::flatten)
    };
    Ok((
        read(peptide, "peptide_qvalue")?,
        read(protein, "pg_global_qvalue")?,
    ))
}

fn column_by_names<'a>(batch: &'a RecordBatch, candidates: &[&str]) -> Result<&'a dyn Array> {
    optional_column_by_names(batch, candidates)
        .ok_or_else(|| invalid_input(format!("missing QPX column: {}", candidates.join(" | "))))
}

fn optional_column_by_names<'a>(
    batch: &'a RecordBatch,
    candidates: &[&str],
) -> Option<&'a dyn Array> {
    let schema = batch.schema();
    candidates.iter().find_map(|name| {
        schema
            .index_of(name)
            .ok()
            .map(|index| batch.column(index).as_ref())
    })
}

fn required_string_value(array: &dyn Array, row: usize, column: &str) -> Result<String> {
    optional_string_value(array, row, column)?.ok_or_else(|| {
        invalid_input(format!(
            "required QPX string column `{column}` is null at row {row}"
        ))
    })
}

fn optional_string_value(array: &dyn Array, row: usize, column: &str) -> Result<Option<String>> {
    if array.is_null(row) {
        return Ok(None);
    }

    match array.data_type() {
        DataType::Utf8 => downcast_array::<StringArray>(array, column)
            .map(|values| Some(values.value(row).to_owned())),
        DataType::LargeUtf8 => downcast_array::<LargeStringArray>(array, column)
            .map(|values| Some(values.value(row).to_owned())),
        _ => Err(unsupported_column(column, array)),
    }
}

fn required_i32_value(array: &dyn Array, row: usize, column: &str) -> Result<i32> {
    optional_i32_value(array, row, column)?.ok_or_else(|| {
        invalid_input(format!(
            "required QPX integer column `{column}` is null at row {row}"
        ))
    })
}

fn optional_i32_value(array: &dyn Array, row: usize, column: &str) -> Result<Option<i32>> {
    if array.is_null(row) {
        return Ok(None);
    }

    match array.data_type() {
        DataType::Int8 => downcast_array::<Int8Array>(array, column)
            .map(|values| Some(i32::from(values.value(row)))),
        DataType::Int16 => downcast_array::<Int16Array>(array, column)
            .map(|values| Some(i32::from(values.value(row)))),
        DataType::Int32 => {
            downcast_array::<Int32Array>(array, column).map(|values| Some(values.value(row)))
        }
        DataType::Int64 => checked_i32(
            downcast_array::<Int64Array>(array, column)?.value(row),
            column,
            row,
        )
        .map(Some),
        DataType::UInt8 => downcast_array::<UInt8Array>(array, column)
            .map(|values| Some(i32::from(values.value(row)))),
        DataType::UInt16 => downcast_array::<UInt16Array>(array, column)
            .map(|values| Some(i32::from(values.value(row)))),
        DataType::UInt32 => checked_i32(
            downcast_array::<UInt32Array>(array, column)?.value(row),
            column,
            row,
        )
        .map(Some),
        DataType::UInt64 => checked_i32(
            downcast_array::<UInt64Array>(array, column)?.value(row),
            column,
            row,
        )
        .map(Some),
        _ => Err(unsupported_column(column, array)),
    }
}

fn optional_bool_value(array: &dyn Array, row: usize, column: &str) -> Result<Option<bool>> {
    if array.is_null(row) {
        return Ok(None);
    }

    match array.data_type() {
        DataType::Boolean => {
            downcast_array::<BooleanArray>(array, column).map(|values| Some(values.value(row)))
        }
        // QPX writers often store boolean flags (`unique`, `is_decoy`) as integer
        // 0/1 columns; treat any non-zero integer as true, matching how pandas
        // reads such a column as truthy.
        _ => optional_i32_value(array, row, column).map(|value| value.map(|value| value != 0)),
    }
}

fn optional_f64_value(array: &dyn Array, row: usize, column: &str) -> Result<Option<f64>> {
    if array.is_null(row) {
        return Ok(None);
    }

    match array.data_type() {
        DataType::Float32 => downcast_array::<Float32Array>(array, column)
            .map(|values| Some(f64::from(values.value(row)))),
        DataType::Float64 => {
            downcast_array::<Float64Array>(array, column).map(|values| Some(values.value(row)))
        }
        DataType::Int32 => downcast_array::<Int32Array>(array, column)
            .map(|values| Some(f64::from(values.value(row)))),
        DataType::Int64 => {
            downcast_array::<Int64Array>(array, column).map(|values| Some(values.value(row) as f64))
        }
        _ => Err(unsupported_column(column, array)),
    }
}

fn list_string_values(
    array: &dyn Array,
    row: usize,
    column: &str,
    struct_field_candidates: &[&str],
) -> Result<Vec<String>> {
    let Some(values) = list_row_values(array, row, column)? else {
        return Ok(Vec::new());
    };

    match values.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 => collect_string_values(values.as_ref(), column),
        DataType::Struct(_) => {
            let structs = downcast_array::<StructArray>(values.as_ref(), column)?;
            let accession_column = struct_column_by_names(structs, struct_field_candidates)
                .ok_or_else(|| invalid_input(format!("missing struct field in `{column}`")))?;
            collect_string_values(accession_column, column)
        }
        _ => Err(unsupported_column(column, values.as_ref())),
    }
}

fn collect_string_values(array: &dyn Array, column: &str) -> Result<Vec<String>> {
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if let Some(value) = optional_string_value(array, row, column)? {
            values.push(value);
        }
    }
    Ok(values)
}

fn intensity_entries(array: &dyn Array, row: usize) -> Result<Vec<QpxIntensityEntry>> {
    let Some(values) = list_row_values(array, row, "intensities")? else {
        return Ok(Vec::new());
    };

    let structs = downcast_array::<StructArray>(values.as_ref(), "intensities")?;
    let sample_accession = struct_column_by_names(structs, &["sample_accession", "sample"]);
    let label = struct_column_by_names(structs, &["label", "channel"]);
    let intensity = struct_column_by_names(structs, &["intensity", "intensity_value", "value"])
        .ok_or_else(|| invalid_input("missing `intensities` intensity field"))?;

    let mut entries = Vec::with_capacity(structs.len());
    for index in 0..structs.len() {
        if let Some(intensity) = optional_f64_value(intensity, index, "intensity")? {
            let sample_accession = sample_accession
                .map(|column| optional_string_value(column, index, "sample_accession"))
                .transpose()?
                .flatten();
            let label = label
                .map(|column| optional_string_value(column, index, "label"))
                .transpose()?
                .flatten();
            entries.push(QpxIntensityEntry {
                sample_accession,
                label,
                intensity,
            });
        }
    }
    Ok(entries)
}

fn list_row_values(array: &dyn Array, row: usize, column: &str) -> Result<Option<ArrayRef>> {
    if array.is_null(row) {
        return Ok(None);
    }

    match array.data_type() {
        DataType::List(_) => {
            downcast_array::<ListArray>(array, column).map(|values| Some(values.value(row)))
        }
        DataType::LargeList(_) => {
            downcast_array::<LargeListArray>(array, column).map(|values| Some(values.value(row)))
        }
        _ => Err(unsupported_column(column, array)),
    }
}

fn struct_column_by_names<'a>(
    array: &'a StructArray,
    candidates: &[&str],
) -> Option<&'a dyn Array> {
    candidates
        .iter()
        .find_map(|name| array.column_by_name(name).map(|column| column.as_ref()))
}

fn downcast_array<'a, T: 'static>(array: &'a dyn Array, column: &str) -> Result<&'a T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| unsupported_column(column, array))
}

fn checked_i32<T>(value: T, column: &str, row: usize) -> Result<i32>
where
    i32: TryFrom<T>,
    T: Copy + std::fmt::Display,
{
    i32::try_from(value).map_err(|_| {
        invalid_input(format!(
            "QPX integer column `{column}` value {value} overflows i32 at row {row}"
        ))
    })
}

fn unsupported_column(column: &str, array: &dyn Array) -> MokumeError {
    invalid_input(format!(
        "unsupported QPX column `{column}` type: {}",
        array.data_type()
    ))
}

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, BooleanArray, Float32Builder, Float64Array, Float64Builder, Int16Array,
        Int32Array, Int32Builder, ListBuilder, StringArray, StringBuilder, StructBuilder,
    };
    use arrow::datatypes::{DataType, Field, Fields, Schema};

    use super::flatten_qpx_batch;
    use crate::sdrf::SdrfTable;

    #[test]
    fn flattens_legacy_qpx_sample_accession() -> Result<(), Box<dyn std::error::Error>> {
        let intensities = legacy_intensities()?;
        let proteins = legacy_proteins();
        let schema = Arc::new(Schema::new(vec![
            Field::new("sequence", DataType::Utf8, false),
            Field::new("modified_sequence", DataType::Utf8, false),
            Field::new("precursor_charge", DataType::Int32, false),
            Field::new("reference_file_name", DataType::Utf8, false),
            Field::new("unique", DataType::Boolean, false),
            Field::new("intensities", intensities.data_type().clone(), true),
            Field::new("protein_accessions", proteins.data_type().clone(), true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["PEPTIDEA"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["PEPTIDEA"])) as ArrayRef,
                Arc::new(Int32Array::from(vec![2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["file.raw"])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                intensities,
                proteins,
            ],
        )?;

        let records = flatten_qpx_batch(&batch)?;

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].run_file_name, "file.raw");
        assert_eq!(records[0].sample_accession.as_deref(), Some("sample-a"));
        assert_eq!(records[0].label.as_deref(), Some("channel-a"));
        assert_eq!(records[0].protein_accessions, vec!["sp|P12345|TEST"]);
        assert_eq!(records[0].intensity, 42.0);
        Ok(())
    }

    #[test]
    fn flattens_new_qpx_struct_accessions_and_label_intensities(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let intensities = new_qpx_intensities()?;
        let proteins = new_qpx_proteins()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("sequence", DataType::Utf8, false),
            Field::new("peptidoform", DataType::Utf8, false),
            Field::new("charge", DataType::Int16, false),
            Field::new("run_file_name", DataType::Utf8, false),
            Field::new("anchor_protein", DataType::Utf8, false),
            Field::new("unique", DataType::Boolean, false),
            Field::new("is_decoy", DataType::Boolean, false),
            Field::new("peptide_qvalue", DataType::Float64, true),
            Field::new("pg_global_qvalue", DataType::Float64, true),
            Field::new("intensities", intensities.data_type().clone(), true),
            Field::new("pg_accessions", proteins.data_type().clone(), true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["PEPTIDEK"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["PEPTIDEK"])) as ArrayRef,
                Arc::new(Int16Array::from(vec![2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["run1"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["sp|P12345|PROT_HUMAN"])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![false])) as ArrayRef,
                Arc::new(Float64Array::from(vec![Some(0.005)])) as ArrayRef,
                Arc::new(Float64Array::from(vec![Some(0.008)])) as ArrayRef,
                intensities,
                proteins,
            ],
        )?;

        let records = flatten_qpx_batch(&batch)?;

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].charge, 2);
        assert_eq!(records[0].run_file_name, "run1");
        assert_eq!(records[0].protein_accessions, vec!["sp|P12345|PROT_HUMAN"]);
        assert_eq!(
            records[0].anchor_protein.as_deref(),
            Some("sp|P12345|PROT_HUMAN")
        );
        assert_eq!(records[0].unique, Some(true));
        assert_eq!(records[0].is_decoy, Some(false));
        assert_eq!(records[0].peptide_qvalue, Some(0.005));
        assert_eq!(records[0].pg_global_qvalue, Some(0.008));
        assert_eq!(records[0].sample_accession, None);
        assert_eq!(records[0].label.as_deref(), Some("TMT126"));
        assert_eq!(records[0].intensity, 1000.0);
        assert_eq!(records[1].label.as_deref(), Some("TMT127"));
        assert_eq!(records[1].intensity, 2000.0);

        let sdrf = concat!(
            "source name\tcomment[data file]\tcomment[label]\tfactor value[group]\n",
            "sample-126\tRUN1.raw\tTMT126\tcontrol\n",
            "sample-127\trun1.mzML\ttmt127\ttreated\n",
        );
        let table = SdrfTable::from_reader(sdrf.as_bytes())?;
        let mapped: Vec<_> = records
            .iter()
            .map(|record| {
                table
                    .lookup(&record.run_file_name, record.label.as_deref())
                    .map(|row| (row.sample_accession.as_str(), row.condition.as_deref()))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        assert_eq!(
            mapped,
            [
                ("sample-126", Some("control")),
                ("sample-127", Some("treated"))
            ]
        );
        Ok(())
    }

    #[test]
    fn routes_lfq_run_labels_to_distinct_runs() -> Result<(), Box<dyn std::error::Error>> {
        // LFQ: the intensity labels are run files (the row's anchor run appears
        // among them), so each intensity must map to its own run/sample instead of
        // collapsing onto the anchor `run_file_name`.
        let intensities = lfq_run_intensities()?;
        let proteins = new_qpx_proteins()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("sequence", DataType::Utf8, false),
            Field::new("peptidoform", DataType::Utf8, false),
            Field::new("charge", DataType::Int16, false),
            Field::new("run_file_name", DataType::Utf8, false),
            Field::new("anchor_protein", DataType::Utf8, false),
            Field::new("unique", DataType::Boolean, false),
            Field::new("is_decoy", DataType::Boolean, false),
            Field::new("intensities", intensities.data_type().clone(), true),
            Field::new("pg_accessions", proteins.data_type().clone(), true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["PEPTIDEK"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["PEPTIDEK"])) as ArrayRef,
                Arc::new(Int16Array::from(vec![2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["Run_A_01.mzML"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["sp|P12345|PROT_HUMAN"])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![false])) as ArrayRef,
                intensities,
                proteins,
            ],
        )?;

        let records = flatten_qpx_batch(&batch)?;

        assert_eq!(records.len(), 3);
        let runs: Vec<_> = records.iter().map(|r| r.run_file_name.as_str()).collect();
        assert_eq!(runs, ["Run_A_01.mzML", "Run_B_01.mzML", "Run_B_02.mzML"]);
        // LFQ has no reporter channel; run-file labels must not leak into `label`.
        assert!(records.iter().all(|record| record.label.is_none()));
        assert_eq!(records[1].intensity, 20.0);

        let sdrf = concat!(
            "source name\tcomment[data file]\tcomment[label]\n",
            "sample-a\trun_a_01.raw\tLabel free sample\n",
            "sample-b1\tRUN_B_01.wiff\tLFQ\n",
            "sample-b2\trun_b_02.d\tlabel-free\n",
        );
        let table = SdrfTable::from_reader(sdrf.as_bytes())?;
        let samples: Vec<_> = records
            .iter()
            .map(|record| {
                table
                    .lookup(&record.run_file_name, record.label.as_deref())
                    .map(|row| row.sample_accession.as_str())
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        assert_eq!(samples, ["sample-a", "sample-b1", "sample-b2"]);
        Ok(())
    }

    #[test]
    fn reads_integer_unique_and_decoy_flags() -> Result<(), Box<dyn std::error::Error>> {
        let intensities = new_qpx_intensities()?;
        let proteins = new_qpx_proteins()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("sequence", DataType::Utf8, false),
            Field::new("peptidoform", DataType::Utf8, false),
            Field::new("charge", DataType::Int16, false),
            Field::new("run_file_name", DataType::Utf8, false),
            Field::new("anchor_protein", DataType::Utf8, false),
            Field::new("unique", DataType::Int32, false),
            Field::new("is_decoy", DataType::Int32, false),
            Field::new("intensities", intensities.data_type().clone(), true),
            Field::new("pg_accessions", proteins.data_type().clone(), true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["PEPTIDEK"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["PEPTIDEK"])) as ArrayRef,
                Arc::new(Int16Array::from(vec![2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["run1"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["sp|P12345|PROT_HUMAN"])) as ArrayRef,
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
                Arc::new(Int32Array::from(vec![0])) as ArrayRef,
                intensities,
                proteins,
            ],
        )?;

        let records = flatten_qpx_batch(&batch)?;

        assert_eq!(records[0].unique, Some(true));
        assert_eq!(records[0].is_decoy, Some(false));
        Ok(())
    }

    fn legacy_intensities() -> Result<ArrayRef, Box<dyn std::error::Error>> {
        let fields = Fields::from(vec![
            Field::new("sample_accession", DataType::Utf8, false),
            Field::new("channel", DataType::Utf8, false),
            Field::new("intensity", DataType::Float64, false),
        ]);
        let struct_builder = StructBuilder::new(
            fields,
            vec![
                Box::new(StringBuilder::new()),
                Box::new(StringBuilder::new()),
                Box::new(Float64Builder::new()),
            ],
        );
        let mut builder = ListBuilder::new(struct_builder);
        append_string_field(builder.values(), 0, "sample-a")?;
        append_string_field(builder.values(), 1, "channel-a")?;
        append_f64_field(builder.values(), 2, 42.0)?;
        builder.values().append(true);
        builder.append(true);
        Ok(Arc::new(builder.finish()) as ArrayRef)
    }

    fn legacy_proteins() -> ArrayRef {
        let mut builder = ListBuilder::new(StringBuilder::new());
        builder.values().append_value("sp|P12345|TEST");
        builder.append(true);
        Arc::new(builder.finish()) as ArrayRef
    }

    fn new_qpx_intensities() -> Result<ArrayRef, Box<dyn std::error::Error>> {
        let fields = Fields::from(vec![
            Field::new("label", DataType::Utf8, false),
            Field::new("intensity", DataType::Float32, false),
        ]);
        let struct_builder = StructBuilder::new(
            fields,
            vec![
                Box::new(StringBuilder::new()),
                Box::new(Float32Builder::new()),
            ],
        );
        let mut builder = ListBuilder::new(struct_builder);
        append_string_field(builder.values(), 0, "TMT126")?;
        append_f32_field(builder.values(), 1, 1000.0)?;
        builder.values().append(true);
        append_string_field(builder.values(), 0, "TMT127")?;
        append_f32_field(builder.values(), 1, 2000.0)?;
        builder.values().append(true);
        builder.append(true);
        Ok(Arc::new(builder.finish()) as ArrayRef)
    }

    fn lfq_run_intensities() -> Result<ArrayRef, Box<dyn std::error::Error>> {
        // Labels are run files; the anchor run (`Run_A_01.mzML`) appears among them.
        let fields = Fields::from(vec![
            Field::new("label", DataType::Utf8, false),
            Field::new("intensity", DataType::Float32, false),
        ]);
        let struct_builder = StructBuilder::new(
            fields,
            vec![
                Box::new(StringBuilder::new()),
                Box::new(Float32Builder::new()),
            ],
        );
        let mut builder = ListBuilder::new(struct_builder);
        for (label, value) in [
            ("Run_A_01.mzML", 10.0f32),
            ("Run_B_01.mzML", 20.0),
            ("Run_B_02.mzML", 30.0),
        ] {
            append_string_field(builder.values(), 0, label)?;
            append_f32_field(builder.values(), 1, value)?;
            builder.values().append(true);
        }
        builder.append(true);
        Ok(Arc::new(builder.finish()) as ArrayRef)
    }

    fn new_qpx_proteins() -> Result<ArrayRef, Box<dyn std::error::Error>> {
        let fields = Fields::from(vec![
            Field::new("accession", DataType::Utf8, false),
            Field::new("start", DataType::Int32, true),
            Field::new("end", DataType::Int32, true),
            Field::new("pre", DataType::Utf8, true),
            Field::new("post", DataType::Utf8, true),
        ]);
        let struct_builder = StructBuilder::new(
            fields,
            vec![
                Box::new(StringBuilder::new()),
                Box::new(Int32Builder::new()),
                Box::new(Int32Builder::new()),
                Box::new(StringBuilder::new()),
                Box::new(StringBuilder::new()),
            ],
        );
        let mut builder = ListBuilder::new(struct_builder);
        append_string_field(builder.values(), 0, "sp|P12345|PROT_HUMAN")?;
        append_i32_field(builder.values(), 1, 10)?;
        append_i32_field(builder.values(), 2, 18)?;
        append_string_field(builder.values(), 3, "K")?;
        append_string_field(builder.values(), 4, "A")?;
        builder.values().append(true);
        builder.append(true);
        Ok(Arc::new(builder.finish()) as ArrayRef)
    }

    fn append_string_field(
        builder: &mut StructBuilder,
        index: usize,
        value: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(builder) = builder.field_builder::<StringBuilder>(index) else {
            return Err(format!("missing string field builder {index}").into());
        };
        builder.append_value(value);
        Ok(())
    }

    fn append_f64_field(
        builder: &mut StructBuilder,
        index: usize,
        value: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(builder) = builder.field_builder::<Float64Builder>(index) else {
            return Err(format!("missing f64 field builder {index}").into());
        };
        builder.append_value(value);
        Ok(())
    }

    fn append_f32_field(
        builder: &mut StructBuilder,
        index: usize,
        value: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(builder) = builder.field_builder::<Float32Builder>(index) else {
            return Err(format!("missing f32 field builder {index}").into());
        };
        builder.append_value(value);
        Ok(())
    }

    fn append_i32_field(
        builder: &mut StructBuilder,
        index: usize,
        value: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(builder) = builder.field_builder::<Int32Builder>(index) else {
            return Err(format!("missing i32 field builder {index}").into());
        };
        builder.append_value(value);
        Ok(())
    }
}
