use std::collections::HashMap;
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
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::ParquetMetaData;
use parquet::schema::types::SchemaDescriptor;

// Column-name candidates in priority order. The reader decodes exactly these
// columns, so a QPX file's other (often large nested) columns are skipped.
const SEQUENCE_COLUMNS: &[&str] = &["sequence", "Sequence"];
const PEPTIDOFORM_COLUMNS: &[&str] = &["peptidoform", "modified_sequence"];
const CHARGE_COLUMNS: &[&str] = &["charge", "precursor_charge"];
const RUN_COLUMNS: &[&str] = &["run_file_name", "reference_file_name", "run", "raw_file"];
const INTENSITY_COLUMNS: &[&str] = &["intensities", "primary_intensities"];
const PROTEIN_GROUP_COLUMNS: &[&str] = &["pg_accessions", "protein_accessions", "proteins"];
const ANCHOR_PROTEIN_COLUMNS: &[&str] = &["anchor_protein", "protein"];
const UNIQUE_COLUMNS: &[&str] = &["unique", "is_unique"];
const DECOY_COLUMNS: &[&str] = &["is_decoy", "decoy"];
const PEPTIDE_QVALUE_COLUMNS: &[&str] = &["peptide_qvalue", "peptide_q_value"];
const PG_QVALUE_COLUMNS: &[&str] = &["pg_global_qvalue", "protein_qvalue"];
const SCORE_COLUMNS: &[&str] = &["additional_scores"];
const ID_RUN_COLUMNS: &[&str] = &["id_run_file_name"];
const FEATURE_COLUMNS: [&[&str]; 11] = [
    SEQUENCE_COLUMNS,
    PEPTIDOFORM_COLUMNS,
    CHARGE_COLUMNS,
    RUN_COLUMNS,
    INTENSITY_COLUMNS,
    PROTEIN_GROUP_COLUMNS,
    ANCHOR_PROTEIN_COLUMNS,
    UNIQUE_COLUMNS,
    DECOY_COLUMNS,
    PEPTIDE_QVALUE_COLUMNS,
    PG_QVALUE_COLUMNS,
];

#[derive(Debug, Clone, PartialEq, Default)]
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
    /// The requested named score, from a list, a struct field, or a complete
    /// top-level q-value column; populated by [`flatten_qpx_batch_with_score`].
    pub selected_score: Option<QpxScoreValue>,
    pub modifications: Option<Vec<String>>,
    /// Explicit transferred identification, or the legacy identification-run
    /// fallback; false when transfer columns were not requested.
    pub transferred: bool,
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
    struct_layout: Option<String>,
}

impl QpxParquetReader {
    /// Returns the next decoded Arrow record batch without flattening it into
    /// `QpxFeatureRecord`s. Callers can run [`flatten_qpx_batch`] on the result,
    /// optionally in parallel, while preserving batch order.
    pub fn next_raw_batch(&mut self) -> Option<Result<RecordBatch>> {
        self.inner.next().map(|batch| {
            let batch = batch
                .map_err(|source| invalid_input(format!("failed to read QPX batch: {source}")))?;
            let Some(layout) = &self.struct_layout else {
                return Ok(batch);
            };
            let mut metadata = batch.schema().metadata().clone();
            metadata.insert("qpx.struct_lists", layout.clone());
            let schema = batch.schema().as_ref().clone().with_metadata(metadata);
            batch
                .with_schema(std::sync::Arc::new(schema))
                .map_err(|source| invalid_input(format!("failed to attach QPX footer: {source}")))
        })
    }

    /// Opens `path`, decoding only the columns [`flatten_qpx_batch`] reads.
    /// `score_name` selects a score leaf or a proven non-null top-level q value.
    /// `with_id_runs` requests transfer classification; `with_modifications`
    /// carries structured names and accessions for modification exclusion.
    pub fn open(
        path: impl AsRef<Path>,
        batch_size: usize,
        score_name: Option<&str>,
        with_id_runs: bool,
        with_modifications: bool,
    ) -> Result<Self> {
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
        let schema = builder.parquet_schema();
        validate_transfer_columns(schema, with_id_runs, &path)?;
        let score = score_projection(&builder, score_name)?;
        let struct_layout = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .into_iter()
            .flatten()
            .find(|kv| kv.key == "qpx.struct_lists")
            .and_then(|kv| kv.value.clone());
        let projection = feature_projection(schema, &score, with_id_runs, with_modifications);
        let inner = builder
            .with_projection(projection)
            .with_batch_size(batch_size)
            .build()
            .map_err(|source| {
                invalid_input(format!(
                    "failed to build QPX batch reader `{}`: {source}",
                    path.display()
                ))
            })?;

        Ok(Self {
            inner,
            struct_layout,
        })
    }
}

fn validate_transfer_columns(schema: &SchemaDescriptor, needed: bool, path: &Path) -> Result<()> {
    if needed
        && !schema.root_schema().get_fields().iter().any(|field| {
            ID_RUN_COLUMNS.contains(&field.name()) || field.name() == "identification_type"
        })
    {
        return Err(invalid_input(format!(
            "QPX parquet `{}` has no id_run_file_name or identification_type column to mark transferred features",
            path.display()
        )));
    }
    Ok(())
}

fn feature_projection(
    schema: &SchemaDescriptor,
    score: &[usize],
    with_id_runs: bool,
    with_modifications: bool,
) -> ProjectionMask {
    let leaves = schema
        .columns()
        .iter()
        .enumerate()
        .filter(|(index, column)| {
            let root = column.path().parts()[0].as_str();
            FEATURE_COLUMNS
                .iter()
                .copied()
                .flatten()
                .any(|name| *name == root)
                || root == "imputed_labels"
                || (with_id_runs
                    && (ID_RUN_COLUMNS.contains(&root) || root == "identification_type"))
                || (with_modifications && root == "modifications")
                || score.contains(index)
        })
        .map(|(index, _)| index);
    ProjectionMask::leaves(schema, leaves)
}

fn complete_qvalue(metadata: &ParquetMetaData, name: &str) -> bool {
    matches!(name, "precursor_qvalue" | "global_qvalue")
        && metadata.row_groups().iter().all(|group| {
            group
                .columns()
                .iter()
                .find(|column| column.column_path().parts() == [name])
                .and_then(|column| column.statistics())
                .and_then(|stats| stats.null_count_opt())
                == Some(0)
        })
}

fn struct_score_layout(raw: Option<&str>, name: &str) -> Result<(String, bool)> {
    let raw = raw.ok_or_else(|| {
        invalid_input("STRUCT additional_scores requires qpx.struct_lists footer")
    })?;
    let layout: serde_json::Value = serde_json::from_str(raw)
        .map_err(|error| invalid_input(format!("invalid qpx.struct_lists footer: {error}")))?;
    let (field, entry) = layout
        .get("additional_scores")
        .and_then(|value| value.as_array())
        .and_then(|entries| {
            entries
                .iter()
                .filter_map(|entry| {
                    entry
                        .get("name")
                        .and_then(|value| value.as_str())
                        .map(|field| (field, entry))
                })
                .find(|(field, _)| *field == name)
        })
        .ok_or_else(|| {
            invalid_input(format!("score `{name}` is not in qpx.struct_lists layout"))
        })?;
    let direction = entry
        .get("higher_better")
        .and_then(|value| value.as_bool())
        .ok_or_else(|| {
            invalid_input(format!(
                "score `{name}` has no boolean higher_better in footer"
            ))
        })?;
    Ok((field.to_owned(), direction))
}

fn score_projection(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    name: Option<&str>,
) -> Result<Vec<usize>> {
    let Some(name) = name else {
        return Ok(Vec::new());
    };
    let top = complete_qvalue(builder.metadata(), name);
    let field = builder.schema().field_with_name("additional_scores");
    let structured =
        !top && field.is_ok_and(|field| matches!(field.data_type(), DataType::Struct(_)));
    let field_name = if structured {
        let metadata = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .into_iter()
            .flatten()
            .find(|kv| kv.key == "qpx.struct_lists")
            .and_then(|kv| kv.value.as_deref());
        Some(struct_score_layout(metadata, name)?.0)
    } else {
        None
    };
    Ok(builder
        .parquet_schema()
        .columns()
        .iter()
        .enumerate()
        .filter(|(_, column)| {
            let parts = column.path().parts();
            if top {
                parts == [name]
            } else {
                parts[0] == "additional_scores"
                    && field_name
                        .as_ref()
                        .is_none_or(|field| parts.get(1) == Some(field))
            }
        })
        .map(|(index, _)| index)
        .collect())
}

impl Iterator for QpxParquetReader {
    type Item = Result<Vec<QpxFeatureRecord>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_raw_batch()
            .map(|batch| batch.and_then(|batch| flatten_qpx_batch(&batch)))
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
    let mut run_keys = RunKeys::default();
    let mut records = Vec::new();
    for row in 0..batch.num_rows() {
        flatten_qpx_row(&columns, score_name, row, &mut run_keys, &mut records)?;
    }
    Ok(records)
}

/// Normalized run keys for one batch. Rows repeat a handful of run names and
/// labels, so each distinct value is normalized once.
#[derive(Default)]
struct RunKeys(HashMap<String, String>);

impl RunKeys {
    fn same_run(&mut self, label: &str, run_file_name: &str) -> bool {
        if label == run_file_name {
            return true;
        }
        self.cache(label);
        self.cache(run_file_name);
        self.0.get(label) == self.0.get(run_file_name)
    }

    fn cache(&mut self, value: &str) {
        if !self.0.contains_key(value) {
            self.0
                .insert(value.to_owned(), crate::sdrf::normalize_file_key(value));
        }
    }
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
    additional_scores: Option<ScoreColumn<'a>>,
    modifications: Option<&'a dyn Array>,
    imputed_labels: Option<&'a dyn Array>,
    identification_type: Option<&'a dyn Array>,
    id_run_file_name: Option<&'a dyn Array>,
}

impl<'a> QpxBatchColumns<'a> {
    fn from_batch(batch: &'a RecordBatch, score_name: Option<&str>) -> Result<Self> {
        Ok(Self {
            sequence: column_by_names(batch, SEQUENCE_COLUMNS)?,
            peptidoform: column_by_names(batch, PEPTIDOFORM_COLUMNS)?,
            charge: column_by_names(batch, CHARGE_COLUMNS)?,
            run_file_name: column_by_names(batch, RUN_COLUMNS)?,
            intensities: column_by_names(batch, INTENSITY_COLUMNS)?,
            protein_groups: column_by_names(batch, PROTEIN_GROUP_COLUMNS)?,
            anchor_protein: optional_column_by_names(batch, ANCHOR_PROTEIN_COLUMNS),
            unique: optional_column_by_names(batch, UNIQUE_COLUMNS),
            is_decoy: optional_column_by_names(batch, DECOY_COLUMNS),
            peptide_qvalue: optional_column_by_names(batch, PEPTIDE_QVALUE_COLUMNS),
            pg_global_qvalue: optional_column_by_names(batch, PG_QVALUE_COLUMNS),
            additional_scores: requested_score_column(batch, score_name)?,
            id_run_file_name: optional_column_by_names(batch, ID_RUN_COLUMNS),
            modifications: optional_column_by_names(batch, &["modifications"]),
            imputed_labels: optional_column_by_names(batch, &["imputed_labels"]),
            identification_type: optional_column_by_names(batch, &["identification_type"]),
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
    modifications: Option<Vec<String>>,
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
            modifications: modification_names(columns.modifications, row)?,
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
    run_keys: &mut RunKeys,
    records: &mut Vec<QpxFeatureRecord>,
) -> Result<()> {
    let metadata = QpxRowMetadata::from_columns(columns, score_name, row)?;
    let mut entries = intensity_entries(columns.intensities, row)?;
    let labels_are_runs = intensity_labels_are_runs(&entries, &metadata.run_file_name, run_keys);
    let transferred =
        row_transferred(columns, row, &metadata, &entries, labels_are_runs, run_keys)?;
    drop_imputed_entries(&mut entries, columns.imputed_labels, row)?;
    records.extend(
        entries
            .into_iter()
            .filter(|entry| entry.intensity.is_finite())
            .enumerate()
            .map(|(index, entry)| {
                qpx_feature_record(&metadata, entry, labels_are_runs, transferred, index == 0)
            }),
    );
    Ok(())
}

/// Prefer an explicit identification type. Null or absent types use the legacy
/// identification-run rule. A multi-run row cannot use that fallback because it
/// has only one identification run for all its intensities.
fn row_transferred(
    columns: &QpxBatchColumns<'_>,
    row: usize,
    metadata: &QpxRowMetadata,
    entries: &[QpxIntensityEntry],
    labels_are_runs: bool,
    run_keys: &mut RunKeys,
) -> Result<bool> {
    if let Some(column) = columns.identification_type {
        if let Some(value) = optional_string_value(column, row, "identification_type")? {
            return Ok(value == "transferred");
        }
    }
    let Some(column) = columns.id_run_file_name else {
        return Ok(false);
    };
    let run = metadata.run_file_name.as_str();
    if labels_are_runs
        && entries.iter().any(|entry| {
            entry
                .label
                .as_deref()
                .is_some_and(|label| !run_keys.same_run(label, run))
        })
    {
        return Err(invalid_input(
            "QPX rows list several runs, so id_run_file_name cannot mark transferred features",
        ));
    }
    Ok(optional_string_value(column, row, "id_run_file_name")?
        .is_none_or(|id_run| !run_keys.same_run(&id_run, run)))
}

/// LFQ intensities are labeled by run file; isobaric intensities are labeled
/// by reporter channel. Matching the anchor run distinguishes the two layouts.
fn intensity_labels_are_runs(
    entries: &[QpxIntensityEntry],
    run_file_name: &str,
    run_keys: &mut RunKeys,
) -> bool {
    entries.iter().any(|entry| {
        entry
            .label
            .as_deref()
            .is_some_and(|label| run_keys.same_run(label, run_file_name))
    })
}

fn qpx_feature_record(
    metadata: &QpxRowMetadata,
    entry: QpxIntensityEntry,
    labels_are_runs: bool,
    transferred: bool,
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
        modifications: metadata.modifications.clone(),
        transferred,
        label,
        intensity: entry.intensity,
        row_start,
    }
}

#[derive(Clone, Copy)]
enum ScoreColumn<'a> {
    List(&'a dyn Array),
    Scalar(&'a dyn Array, bool, Option<&'a StructArray>),
}

fn requested_score_column<'a>(
    batch: &'a RecordBatch,
    score_name: Option<&str>,
) -> Result<Option<ScoreColumn<'a>>> {
    let Some(name) = score_name else {
        return Ok(None);
    };
    if matches!(name, "precursor_qvalue" | "global_qvalue") {
        if let Some(column) = optional_column_by_names(batch, &[name]) {
            return Ok(Some(ScoreColumn::Scalar(column, false, None)));
        }
    }
    let column = column_by_names(batch, SCORE_COLUMNS)?;
    if let Some(values) = column.as_any().downcast_ref::<StructArray>() {
        let (field, direction) = struct_score_layout(
            batch
                .schema()
                .metadata()
                .get("qpx.struct_lists")
                .map(String::as_str),
            name,
        )?;
        let value = struct_column_by_names(values, &[field.as_str()])
            .ok_or_else(|| invalid_input(format!("missing additional_scores field `{name}`")))?;
        return Ok(Some(ScoreColumn::Scalar(value, direction, Some(values))));
    }
    Ok(Some(ScoreColumn::List(column)))
}

fn selected_score_value(
    scores: Option<ScoreColumn<'_>>,
    name: Option<&str>,
    row: usize,
) -> Result<Option<QpxScoreValue>> {
    let Some((scores, name)) = scores.zip(name) else {
        return Ok(None);
    };
    match scores {
        ScoreColumn::List(array) => named_score_value(array, row, name),
        ScoreColumn::Scalar(array, higher_better, parent) => {
            if parent.is_some_and(|parent| parent.is_null(row)) {
                return Ok(None);
            }
            Ok(Some(QpxScoreValue {
                value: required_score_value(array, row, name, row)?,
                higher_better,
            }))
        }
    }
}

fn drop_imputed_entries(
    entries: &mut Vec<QpxIntensityEntry>,
    labels: Option<&dyn Array>,
    row: usize,
) -> Result<()> {
    let Some(labels) = labels else {
        return Ok(());
    };
    let labels = list_string_values(labels, row, "imputed_labels", &[])?;
    entries.retain(|entry| {
        !entry
            .label
            .as_ref()
            .is_some_and(|label| labels.contains(label))
    });
    Ok(())
}

fn modification_names(column: Option<&dyn Array>, row: usize) -> Result<Option<Vec<String>>> {
    let Some(column) = column else {
        return Ok(None);
    };
    let Some(values) = list_row_values(column, row, "modifications")? else {
        return Ok(Some(Vec::new()));
    };
    let values = downcast_array::<StructArray>(values.as_ref(), "modifications")?;
    let mut names = Vec::new();
    for key in ["name", "accession"] {
        if let Some(column) = struct_column_by_names(values, &[key]) {
            for index in 0..values.len() {
                names.extend(optional_string_value(column, index, key)?);
            }
        }
    }
    Ok(Some(names))
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

    use super::{flatten_qpx_batch, QpxParquetReader};
    use crate::sdrf::SdrfTable;
    use arrow::array::{Float32Array, StructArray};
    use arrow::record_batch::RecordBatch;
    use std::fs::File;
    use std::path::Path;

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
        let intensities = labeled_intensities(&LFQ_RUNS)?;
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
        // Without id_run_file_name every feature reads as identified.
        assert!(!records[0].transferred);
        Ok(())
    }

    #[test]
    fn marks_features_without_an_identification_in_their_run(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // quantms leaves id_run_file_name null for match-between-runs features.
        for (id_run, transferred) in [
            (Some("Run_A_01.mzML"), false),
            (Some("run_a_01.raw"), false),
            (None, true),
            (Some("Run_B_01.mzML"), true),
        ] {
            let batch = id_run_batch(id_run, labeled_intensities(&[("LFQ", 10.0)])?)?;
            let records = flatten_qpx_batch(&batch)?;
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].transferred, transferred, "{id_run:?}");
        }
        Ok(())
    }

    #[test]
    fn rejects_identification_runs_for_rows_with_several_runs(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batch = id_run_batch(Some("Run_A_01.mzML"), labeled_intensities(&LFQ_RUNS)?)?;
        assert!(flatten_qpx_batch(&batch).is_err());
        // A row labeled with its own run is still one run.
        let batch = id_run_batch(None, labeled_intensities(&LFQ_RUNS[..1])?)?;
        assert!(flatten_qpx_batch(&batch)?[0].transferred);
        Ok(())
    }

    fn with_column(
        batch: RecordBatch,
        name: &str,
        values: ArrayRef,
    ) -> Result<RecordBatch, arrow::error::ArrowError> {
        let mut fields = batch.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(name, values.data_type().clone(), true)));
        let mut columns = batch.columns().to_vec();
        columns.push(values);
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
    }

    fn score_batch(value: Option<f32>) -> Result<RecordBatch, Box<dyn std::error::Error>> {
        let base = id_run_batch(None, labeled_intensities(&[("LFQ", 10.0)])?)?;
        let scores = StructArray::from(vec![
            (
                Arc::new(Field::new("quality", DataType::Float32, true)),
                Arc::new(Float32Array::from(vec![value])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("global_qvalue", DataType::Float64, true)),
                Arc::new(Float64Array::from(vec![0.02])) as ArrayRef,
            ),
        ]);
        Ok(with_column(base, "additional_scores", Arc::new(scores))?)
    }

    fn write_scores(
        path: &Path,
        batches: &[RecordBatch],
        footer: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use parquet::arrow::ArrowWriter;
        use parquet::file::metadata::KeyValue;
        use parquet::file::properties::WriterProperties;
        let layout = serde_json::json!({"additional_scores": [
            {"name": "quality", "higher_better": true},
            {"name": "global_qvalue", "higher_better": false}
        ]})
        .to_string();
        let metadata = footer.then(|| vec![KeyValue::new("qpx.struct_lists".to_owned(), layout)]);
        let properties = WriterProperties::builder()
            .set_key_value_metadata(metadata)
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(path)?, batches[0].schema(), Some(properties))?;
        for batch in batches {
            writer.write(batch)?;
            writer.flush()?;
        }
        writer.close()?;
        Ok(())
    }

    #[test]
    fn struct_scores_use_footer_and_project_one_leaf() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("scores.parquet");
        write_scores(&path, &[score_batch(Some(0.75))?], true)?;
        for (name, expected, higher) in [("quality", 0.75, true), ("global_qvalue", 0.02, false)] {
            let mut reader = QpxParquetReader::open(&path, 8, Some(name), false, false)?;
            let batch = reader.next_raw_batch().ok_or("missing batch")??;
            let scores = batch
                .column_by_name("additional_scores")
                .ok_or("missing scores")?;
            assert_eq!(
                scores
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or("not struct")?
                    .num_columns(),
                1
            );
            let value = super::flatten_qpx_batch_with_score(&batch, name)?[0]
                .selected_score
                .ok_or("missing score")?;
            assert_eq!(
                value,
                super::QpxScoreValue {
                    value: expected,
                    higher_better: higher
                }
            );
        }
        assert!(QpxParquetReader::open(&path, 8, Some("missing"), false, false).is_err());
        write_scores(&path, &[score_batch(Some(0.75))?], false)?;
        assert!(QpxParquetReader::open(&path, 8, Some("quality"), false, false).is_err());
        Ok(())
    }

    #[test]
    fn struct_scores_distinguish_null_parent_and_invalid_field(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("scores.parquet");
        for value in [None, Some(f32::INFINITY), Some(f32::NAN)] {
            write_scores(&path, &[score_batch(value)?], true)?;
            let mut reader = QpxParquetReader::open(&path, 8, Some("quality"), false, false)?;
            let batch = reader.next_raw_batch().ok_or("missing batch")??;
            assert!(super::flatten_qpx_batch_with_score(&batch, "quality").is_err());
            let schema = batch.schema();
            let values = batch
                .column_by_name("additional_scores")
                .ok_or("missing scores")?;
            let scores = values
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or("not struct")?;
            let null_scores = StructArray::new(
                scores.fields().clone(),
                scores.columns().to_vec(),
                Some(arrow::buffer::NullBuffer::from(vec![false])),
            );
            let columns = batch
                .columns()
                .iter()
                .zip(schema.fields())
                .map(|(column, field)| {
                    if field.name() == "additional_scores" {
                        Arc::new(null_scores.clone()) as ArrayRef
                    } else {
                        column.clone()
                    }
                })
                .collect();
            let batch = RecordBatch::try_new(schema, columns)?;
            assert!(super::flatten_qpx_batch_with_score(&batch, "quality")?[0]
                .selected_score
                .is_none());
        }
        Ok(())
    }

    fn top_qvalue_batch(value: Option<f64>) -> Result<RecordBatch, Box<dyn std::error::Error>> {
        Ok(with_column(
            score_batch(Some(0.9))?,
            "global_qvalue",
            Arc::new(Float64Array::from(vec![value])),
        )?)
    }

    #[test]
    fn top_qvalue_requires_every_row_group_nonnull() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("top.parquet");
        for partial in [false, true] {
            let first = top_qvalue_batch(Some(0.01))?;
            let second = top_qvalue_batch(if partial { None } else { Some(0.2) })?;
            write_scores(&path, &[first, second], true)?;
            let mut reader = QpxParquetReader::open(&path, 8, Some("global_qvalue"), false, false)?;
            let batch = reader.next_raw_batch().ok_or("missing batch")??;
            assert_eq!(batch.column_by_name("global_qvalue").is_some(), !partial);
            assert_eq!(batch.column_by_name("additional_scores").is_some(), partial);
            let records = super::flatten_qpx_batch_with_score(&batch, "global_qvalue")?;
            assert_eq!(
                records[0].selected_score.ok_or("missing score")?.value,
                if partial { 0.02 } else { 0.01 }
            );
        }
        Ok(())
    }

    #[test]
    fn identification_type_overrides_id_run_and_null_falls_back(
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (kind, id_run, expected) in [
            (Some("transferred"), Some("Run_A_01.mzML"), true),
            (Some("secondary"), None, false),
            (Some("direct"), None, false),
            (None, None, true),
            (None, Some("Run_A_01.mzML"), false),
        ] {
            let batch = id_run_batch(id_run, labeled_intensities(&[("LFQ", 10.0)])?)?;
            let batch = with_column(
                batch,
                "identification_type",
                Arc::new(StringArray::from(vec![kind])),
            )?;
            assert_eq!(flatten_qpx_batch(&batch)?[0].transferred, expected);
        }
        Ok(())
    }

    #[test]
    fn imputed_labels_drop_entries_and_modifications_are_optional(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batch = id_run_batch(None, new_qpx_intensities()?)?;
        let mut labels = ListBuilder::new(StringBuilder::new());
        labels.values().append_value("TMT126");
        labels.append(true);
        let batch = with_column(batch, "imputed_labels", Arc::new(labels.finish()))?;
        let batch = with_column(batch, "modifications", new_qpx_proteins()?)?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("imputed.parquet");
        write_scores(&path, &[batch], false)?;
        for with_modifications in [false, true] {
            let mut reader = QpxParquetReader::open(&path, 8, None, false, with_modifications)?;
            let batch = reader.next_raw_batch().ok_or("missing batch")??;
            let records = flatten_qpx_batch(&batch)?;
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].label.as_deref(), Some("TMT127"));
            assert_eq!(records[0].modifications.is_some(), with_modifications);
        }
        Ok(())
    }

    /// One `Run_A_01.mzML` row with an `id_run_file_name` column.
    fn id_run_batch(
        id_run: Option<&str>,
        intensities: ArrayRef,
    ) -> Result<arrow::record_batch::RecordBatch, Box<dyn std::error::Error>> {
        let proteins = new_qpx_proteins()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("sequence", DataType::Utf8, false),
            Field::new("peptidoform", DataType::Utf8, false),
            Field::new("charge", DataType::Int16, false),
            Field::new("run_file_name", DataType::Utf8, false),
            Field::new("id_run_file_name", DataType::Utf8, true),
            Field::new("intensities", intensities.data_type().clone(), true),
            Field::new("pg_accessions", proteins.data_type().clone(), true),
        ]));
        Ok(arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["PEPTIDEK"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["PEPTIDEK"])) as ArrayRef,
                Arc::new(Int16Array::from(vec![2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["Run_A_01.mzML"])) as ArrayRef,
                Arc::new(StringArray::from(vec![id_run])) as ArrayRef,
                intensities,
                proteins,
            ],
        )?)
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

    /// LFQ labels that are run files; the anchor run (`Run_A_01.mzML`) is first.
    const LFQ_RUNS: [(&str, f32); 3] = [
        ("Run_A_01.mzML", 10.0),
        ("Run_B_01.mzML", 20.0),
        ("Run_B_02.mzML", 30.0),
    ];

    /// One row whose intensities carry the given labels.
    fn labeled_intensities(
        entries: &[(&str, f32)],
    ) -> Result<ArrayRef, Box<dyn std::error::Error>> {
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
        for &(label, value) in entries {
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
