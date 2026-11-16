use std::fs::File;
use std::path::Path;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeListArray, LargeStringArray,
    ListArray, StringArray, StructArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use mokume_core::{MokumeError, Result};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;

const PSM_COLUMNS: &[&str] = &[
    "psm_id",
    "sequence",
    "run_file_name",
    "reference_file_name",
    "run",
    "raw_file",
    "scan",
    "feature_id",
    "is_decoy",
    "decoy",
    "additional_scores",
];

/// The PSM evidence required for true spectral counting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QpxPsmRecord {
    pub psm_id: i64,
    pub sequence: String,
    pub run_file_name: String,
    pub scan: Vec<i64>,
    pub feature_id: Option<i64>,
    pub is_decoy: bool,
}

pub struct QpxPsmParquetReader {
    inner: ParquetRecordBatchReader,
}

impl QpxPsmParquetReader {
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
                "failed to open QPX PSM parquet `{}`: {source}",
                path.display()
            ))
        })?;
        let roots = builder
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(index, field)| {
                PSM_COLUMNS
                    .contains(&field.name().as_str())
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        let projection = ProjectionMask::roots(builder.parquet_schema(), roots);
        let inner = builder
            .with_projection(projection)
            .with_batch_size(batch_size)
            .build()
            .map_err(|source| {
                invalid_input(format!(
                    "failed to build QPX PSM reader `{}`: {source}",
                    path.display()
                ))
            })?;
        Ok(Self { inner })
    }
}

impl Iterator for QpxPsmParquetReader {
    type Item = Result<Vec<QpxPsmRecord>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|batch| {
            batch
                .map_err(|source| invalid_input(format!("failed to read QPX PSM batch: {source}")))
                .and_then(|batch| flatten_psm_batch(&batch))
        })
    }
}

pub fn flatten_psm_batch(batch: &RecordBatch) -> Result<Vec<QpxPsmRecord>> {
    let psm_id = required_column(batch, &["psm_id"])?;
    let sequence = required_column(batch, &["sequence"])?;
    let run = required_column(
        batch,
        &["run_file_name", "reference_file_name", "run", "raw_file"],
    )?;
    let scan = required_column(batch, &["scan"])?;
    let feature_id = optional_column(batch, &["feature_id"]);
    let is_decoy = optional_column(batch, &["is_decoy", "decoy"]);
    let scores = optional_column(batch, &["additional_scores"]);

    (0..batch.num_rows())
        .filter_map(|row| match rank_one(scores, row) {
            Ok(true) => Some(Ok(row)),
            Ok(false) => None,
            Err(error) => Some(Err(error)),
        })
        .map(|row| {
            let row = row?;
            Ok(QpxPsmRecord {
                psm_id: required_i64(psm_id, row, "psm_id")?,
                sequence: required_string(sequence, row, "sequence")?,
                run_file_name: required_string(run, row, "run_file_name")?,
                scan: list_i64(scan, row, "scan")?,
                feature_id: feature_id
                    .map(|column| optional_i64(column, row, "feature_id"))
                    .transpose()?
                    .flatten(),
                is_decoy: is_decoy
                    .map(|column| optional_bool(column, row, "is_decoy"))
                    .transpose()?
                    .flatten()
                    .unwrap_or(false),
            })
        })
        .collect()
}

fn rank_one(scores: Option<&dyn Array>, row: usize) -> Result<bool> {
    let Some(scores) = scores.filter(|array| !array.is_null(row)) else {
        return Ok(true);
    };
    if let Some(fields) = scores.as_any().downcast_ref::<StructArray>() {
        return fields.column_by_name("rank").map_or(Ok(true), |rank| {
            crate::qpx::optional_f64_value(rank.as_ref(), row, "rank")
                .map(|value| value == Some(1.0))
        });
    }
    let Some(values) = list_values(scores, row, "additional_scores")? else {
        return Ok(true);
    };
    let fields = downcast::<StructArray>(values.as_ref(), "additional_scores")?;
    let names = required_score_field(fields, "score_name")?;
    for index in 0..fields.len() {
        if crate::qpx::optional_string_value(names, index, "score_name")?.as_deref() == Some("rank")
        {
            let values = required_score_field(fields, "score_value")?;
            if crate::qpx::optional_f64_value(values, index, "rank")? != Some(1.0) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn required_score_field<'a>(fields: &'a StructArray, name: &str) -> Result<&'a dyn Array> {
    fields
        .column_by_name(name)
        .map(|column| column.as_ref())
        .ok_or_else(|| invalid_input(format!("missing QPX PSM additional_scores.{name}")))
}

fn required_column<'a>(batch: &'a RecordBatch, names: &[&str]) -> Result<&'a dyn Array> {
    optional_column(batch, names)
        .ok_or_else(|| invalid_input(format!("missing QPX PSM column: {}", names.join(" | "))))
}

fn optional_column<'a>(batch: &'a RecordBatch, names: &[&str]) -> Option<&'a dyn Array> {
    let schema = batch.schema();
    names.iter().find_map(|name| {
        schema
            .index_of(name)
            .ok()
            .map(|index| batch.column(index).as_ref())
    })
}

fn required_string(array: &dyn Array, row: usize, name: &str) -> Result<String> {
    if array.is_null(row) {
        return Err(invalid_input(format!(
            "required QPX PSM column `{name}` is null at row {row}"
        )));
    }
    match array.data_type() {
        DataType::Utf8 => {
            downcast::<StringArray>(array, name).map(|values| values.value(row).to_owned())
        }
        DataType::LargeUtf8 => {
            downcast::<LargeStringArray>(array, name).map(|values| values.value(row).to_owned())
        }
        _ => Err(unsupported(name, array)),
    }
}

fn optional_bool(array: &dyn Array, row: usize, name: &str) -> Result<Option<bool>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Boolean => {
            downcast::<BooleanArray>(array, name).map(|values| Some(values.value(row)))
        }
        DataType::Int32 => {
            downcast::<Int32Array>(array, name).map(|values| Some(values.value(row) != 0))
        }
        DataType::Int64 => {
            downcast::<Int64Array>(array, name).map(|values| Some(values.value(row) != 0))
        }
        _ => Err(unsupported(name, array)),
    }
}

fn list_i64(array: &dyn Array, row: usize, name: &str) -> Result<Vec<i64>> {
    let Some(values) = list_values(array, row, name)? else {
        return Ok(Vec::new());
    };
    match values.data_type() {
        DataType::Int32 => Ok(downcast::<Int32Array>(values.as_ref(), name)?
            .iter()
            .flatten()
            .map(i64::from)
            .collect()),
        DataType::Int64 => Ok(downcast::<Int64Array>(values.as_ref(), name)?
            .iter()
            .flatten()
            .collect()),
        DataType::UInt32 => Ok(downcast::<UInt32Array>(values.as_ref(), name)?
            .iter()
            .flatten()
            .map(i64::from)
            .collect()),
        DataType::UInt64 => downcast::<UInt64Array>(values.as_ref(), name)?
            .iter()
            .flatten()
            .map(|value| {
                i64::try_from(value).map_err(|_| {
                    invalid_input(format!("QPX PSM `{name}` value {value} overflows i64"))
                })
            })
            .collect(),
        _ => Err(unsupported(name, values.as_ref())),
    }
}

fn optional_i64(array: &dyn Array, row: usize, name: &str) -> Result<Option<i64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Int32 => {
            downcast::<Int32Array>(array, name).map(|values| Some(i64::from(values.value(row))))
        }
        DataType::Int64 => {
            downcast::<Int64Array>(array, name).map(|values| Some(values.value(row)))
        }
        DataType::UInt32 => {
            downcast::<UInt32Array>(array, name).map(|values| Some(i64::from(values.value(row))))
        }
        DataType::UInt64 => downcast::<UInt64Array>(array, name)?
            .value(row)
            .try_into()
            .map(Some)
            .map_err(|_| invalid_input(format!("QPX PSM `{name}` value overflows i64"))),
        _ => Err(unsupported(name, array)),
    }
}

fn required_i64(array: &dyn Array, row: usize, name: &str) -> Result<i64> {
    optional_i64(array, row, name)?.ok_or_else(|| {
        invalid_input(format!(
            "required QPX PSM column `{name}` is null at row {row}"
        ))
    })
}

fn list_values(array: &dyn Array, row: usize, name: &str) -> Result<Option<ArrayRef>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::List(_) => {
            downcast::<ListArray>(array, name).map(|values| Some(values.value(row)))
        }
        DataType::LargeList(_) => {
            downcast::<LargeListArray>(array, name).map(|values| Some(values.value(row)))
        }
        _ => Err(unsupported(name, array)),
    }
}

fn downcast<'a, T: 'static>(array: &'a dyn Array, name: &str) -> Result<&'a T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| unsupported(name, array))
}

fn unsupported(name: &str, array: &dyn Array) -> MokumeError {
    invalid_input(format!(
        "unsupported QPX PSM column `{name}` type: {}",
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
    use std::fs::File;
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, BooleanArray, Float64Builder, Int32Builder, Int64Array, ListBuilder, StringArray,
        StringBuilder, StructBuilder,
    };
    use arrow::datatypes::{DataType, Field};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use super::{flatten_psm_batch, QpxPsmParquetReader};

    fn psm_batch() -> Result<RecordBatch, Box<dyn std::error::Error>> {
        let mut scans = ListBuilder::new(Int32Builder::new());
        scans.values().append_value(101);
        scans.append(true);
        scans.values().append_value(202);
        scans.values().append_value(203);
        scans.append(true);

        Ok(RecordBatch::try_from_iter([
            (
                "psm_id",
                Arc::new(Int64Array::from(vec![1001, 1002])) as ArrayRef,
            ),
            (
                "sequence",
                Arc::new(StringArray::from(vec!["PEPTIDE", "DECOYPEP"])) as ArrayRef,
            ),
            (
                "run_file_name",
                Arc::new(StringArray::from(vec!["run-a", "run-b"])) as ArrayRef,
            ),
            ("scan", Arc::new(scans.finish()) as ArrayRef),
            (
                "feature_id",
                Arc::new(Int64Array::from(vec![Some(10), None])) as ArrayRef,
            ),
            (
                "is_decoy",
                Arc::new(BooleanArray::from(vec![false, true])) as ArrayRef,
            ),
        ])?)
    }

    #[test]
    fn flattens_psm_identity_and_feature_link() -> Result<(), Box<dyn std::error::Error>> {
        let records = flatten_psm_batch(&psm_batch()?)?;
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].psm_id, 1001);
        assert_eq!(records[0].scan, vec![101]);
        assert_eq!(records[0].feature_id, Some(10));
        assert_eq!(records[1].feature_id, None);
        assert!(!records[0].is_decoy);
        assert!(records[1].is_decoy);
        Ok(())
    }

    fn rank_scores(ranks: [Option<f64>; 2]) -> Result<ArrayRef, Box<dyn std::error::Error>> {
        let fields = vec![
            Field::new("score_name", DataType::Utf8, false),
            Field::new("score_value", DataType::Float64, false),
        ];
        let mut scores = ListBuilder::new(StructBuilder::from_fields(fields, 2));
        for rank in ranks {
            if let Some(rank) = rank {
                scores
                    .values()
                    .field_builder::<StringBuilder>(0)
                    .ok_or("score_name builder")?
                    .append_value("rank");
                scores
                    .values()
                    .field_builder::<Float64Builder>(1)
                    .ok_or("score_value builder")?
                    .append_value(rank);
                scores.values().append(true);
            }
            scores.append(true);
        }
        Ok(Arc::new(scores.finish()))
    }

    #[test]
    fn reads_rank_and_keeps_rank_one_or_absent() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("ranked.psm.parquet");
        for (ranks, expected) in [
            ([Some(1.0), Some(2.0)], vec![1001]),
            ([None, Some(1.0)], vec![1001, 1002]),
        ] {
            let batch = psm_batch()?;
            let mut columns = batch
                .schema()
                .fields()
                .iter()
                .zip(batch.columns())
                .map(|(field, values)| (field.name().to_owned(), values.clone()))
                .collect::<Vec<_>>();
            columns.push(("additional_scores".to_owned(), rank_scores(ranks)?));
            let batch = RecordBatch::try_from_iter(columns)?;
            let mut writer = ArrowWriter::try_new(File::create(&path)?, batch.schema(), None)?;
            writer.write(&batch)?;
            writer.close()?;
            let records = QpxPsmParquetReader::open(&path, 1)?.collect::<Result<Vec<_>, _>>()?;
            assert_eq!(
                records
                    .into_iter()
                    .flatten()
                    .map(|record| record.psm_id)
                    .collect::<Vec<_>>(),
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn streams_psm_parquet_batches() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("test.psm.parquet");
        let batch = psm_batch()?;
        let file = File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
        writer.write(&batch)?;
        writer.close()?;

        let mut reader = QpxPsmParquetReader::open(&path, 1)?;
        let records = reader.next().expect("missing first PSM batch")?;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].run_file_name, "run-a");
        Ok(())
    }
}
