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

const FEATURE_GROUP_COLUMNS: &[&str] = &[
    "feature_id",
    "pg_accessions",
    "protein_accessions",
    "anchor_protein",
    "protein",
    "is_decoy",
    "decoy",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QpxFeatureProteinGroup {
    pub feature_id: i64,
    pub protein_accessions: Vec<String>,
    pub is_decoy: bool,
}

pub struct QpxFeatureProteinGroupReader {
    inner: ParquetRecordBatchReader,
}

impl QpxFeatureProteinGroupReader {
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
                "failed to open QPX feature parquet `{}`: {source}",
                path.display()
            ))
        })?;
        let roots = builder
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(index, field)| {
                FEATURE_GROUP_COLUMNS
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
                    "failed to build QPX feature mapping reader `{}`: {source}",
                    path.display()
                ))
            })?;
        Ok(Self { inner })
    }
}

impl Iterator for QpxFeatureProteinGroupReader {
    type Item = Result<Vec<QpxFeatureProteinGroup>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|batch| {
            batch
                .map_err(|source| {
                    invalid_input(format!(
                        "failed to read QPX feature mapping batch: {source}"
                    ))
                })
                .and_then(|batch| flatten_feature_protein_groups(&batch))
        })
    }
}

pub fn flatten_feature_protein_groups(batch: &RecordBatch) -> Result<Vec<QpxFeatureProteinGroup>> {
    let feature_ids = required_column(batch, &["feature_id"])?;
    let protein_groups = optional_column(batch, &["pg_accessions", "protein_accessions"]);
    let anchors = optional_column(batch, &["anchor_protein", "protein"]);
    let decoys = optional_column(batch, &["is_decoy", "decoy"]);
    if protein_groups.is_none() && anchors.is_none() {
        return Err(invalid_input(
            "QPX feature mapping requires `pg_accessions` or `anchor_protein`",
        ));
    }

    (0..batch.num_rows())
        .filter_map(|row| {
            optional_i64(feature_ids, row, "feature_id")
                .transpose()
                .map(|feature_id| feature_id.map(|feature_id| (row, feature_id)))
        })
        .map(|result| {
            let (row, feature_id) = result?;
            let mut protein_accessions = protein_groups
                .map(|column| list_strings(column, row, "pg_accessions"))
                .transpose()?
                .unwrap_or_default();
            if protein_accessions.is_empty() {
                if let Some(anchor) = anchors
                    .map(|column| optional_string(column, row, "anchor_protein"))
                    .transpose()?
                    .flatten()
                {
                    protein_accessions.push(anchor);
                }
            }
            let is_decoy = decoys
                .map(|column| optional_bool(column, row, "is_decoy"))
                .transpose()?
                .flatten()
                .unwrap_or(false);
            Ok(QpxFeatureProteinGroup {
                feature_id,
                protein_accessions,
                is_decoy,
            })
        })
        .collect()
}

fn list_strings(array: &dyn Array, row: usize, name: &str) -> Result<Vec<String>> {
    let Some(values) = list_values(array, row, name)? else {
        return Ok(Vec::new());
    };
    match values.data_type() {
        DataType::Struct(_) => struct_strings(&values, name),
        DataType::Utf8 | DataType::LargeUtf8 => (0..values.len())
            .filter(|index| !values.is_null(*index))
            .map(|index| required_string(values.as_ref(), index, name))
            .collect(),
        _ => Err(unsupported(name, values.as_ref())),
    }
}

fn struct_strings(values: &ArrayRef, name: &str) -> Result<Vec<String>> {
    let entries = downcast::<StructArray>(values.as_ref(), name)?;
    let accessions = ["accession", "protein_accession", "protein"]
        .iter()
        .find_map(|field| entries.column_by_name(field))
        .ok_or_else(|| invalid_input("missing accession field inside `pg_accessions`"))?;
    (0..entries.len())
        .filter(|index| !entries.is_null(*index) && !accessions.is_null(*index))
        .map(|index| required_string(accessions.as_ref(), index, name))
        .collect()
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
            .map_err(|_| invalid_input(format!("QPX `{name}` value overflows i64 at row {row}"))),
        _ => Err(unsupported(name, array)),
    }
}

fn optional_string(array: &dyn Array, row: usize, name: &str) -> Result<Option<String>> {
    if array.is_null(row) {
        return Ok(None);
    }
    required_string(array, row, name).map(Some)
}

fn required_string(array: &dyn Array, row: usize, name: &str) -> Result<String> {
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

fn required_column<'a>(batch: &'a RecordBatch, names: &[&str]) -> Result<&'a dyn Array> {
    optional_column(batch, names)
        .ok_or_else(|| invalid_input(format!("missing QPX feature column: {}", names.join(" | "))))
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

fn downcast<'a, T: 'static>(array: &'a dyn Array, name: &str) -> Result<&'a T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| unsupported(name, array))
}

fn unsupported(name: &str, array: &dyn Array) -> MokumeError {
    invalid_input(format!(
        "unsupported QPX feature column `{name}` type: {}",
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
        ArrayRef, BooleanArray, Int32Builder, Int64Array, ListBuilder, StringArray, StringBuilder,
        StructBuilder,
    };
    use arrow::datatypes::{DataType, Field, Fields, Schema};

    use super::flatten_feature_protein_groups;

    #[test]
    fn reads_feature_groups_from_qpx_struct_accessions() -> Result<(), Box<dyn std::error::Error>> {
        let fields = Fields::from(vec![
            Field::new("accession", DataType::Utf8, false),
            Field::new("start", DataType::Int32, true),
        ]);
        let struct_builder = StructBuilder::new(
            fields,
            vec![
                Box::new(StringBuilder::new()),
                Box::new(Int32Builder::new()),
            ],
        );
        let mut proteins = ListBuilder::new(struct_builder);
        proteins
            .values()
            .field_builder::<StringBuilder>(0)
            .expect("missing accession builder")
            .append_value("sp|P12345|PROT_HUMAN");
        proteins
            .values()
            .field_builder::<Int32Builder>(1)
            .expect("missing start builder")
            .append_value(1);
        proteins.values().append(true);
        proteins.append(true);
        let proteins = Arc::new(proteins.finish()) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![
            Field::new("feature_id", DataType::Int64, false),
            Field::new("pg_accessions", proteins.data_type().clone(), true),
            Field::new("anchor_protein", DataType::Utf8, true),
            Field::new("is_decoy", DataType::Boolean, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![42])) as ArrayRef,
                proteins,
                Arc::new(StringArray::from(vec![Some("P12345")])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![false])) as ArrayRef,
            ],
        )?;

        let groups = flatten_feature_protein_groups(&batch)?;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].feature_id, 42);
        assert_eq!(groups[0].protein_accessions, ["sp|P12345|PROT_HUMAN"]);
        assert!(!groups[0].is_decoy);
        Ok(())
    }
}
