use std::fs::File;
use std::path::Path;

use arrow::array::{Array, StructArray};
use arrow::record_batch::RecordBatch;
use mokume_core::{MokumeError, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;

use crate::qpx::{
    list_row_values, optional_i32_value, optional_string_value, required_string_value,
};
use crate::SdrfRecord;

pub(crate) fn read_run(path: &Path) -> Result<Vec<SdrfRecord>> {
    let file = File::open(path).map_err(|source| MokumeError::Io {
        path: path.to_owned(),
        source,
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(invalid_input)?;
    let roots = builder
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| {
            ["run_file_name", "samples", "fraction"]
                .contains(&field.name().as_str())
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let projection = ProjectionMask::roots(builder.parquet_schema(), roots);
    let reader = builder
        .with_projection(projection)
        .build()
        .map_err(invalid_input)?;
    let mut records = Vec::new();
    for batch in reader {
        append_run_records(&batch.map_err(invalid_input)?, &mut records)?;
    }
    Ok(records)
}

fn append_run_records(batch: &RecordBatch, records: &mut Vec<SdrfRecord>) -> Result<()> {
    let runs = batch
        .column_by_name("run_file_name")
        .ok_or_else(|| invalid_input("missing run_file_name"))?;
    let samples = batch
        .column_by_name("samples")
        .ok_or_else(|| invalid_input("missing run.samples"))?;
    for row in 0..batch.num_rows() {
        let run = required_string_value(runs.as_ref(), row, "run_file_name")?;
        let fraction = batch
            .column_by_name("fraction")
            .map(|column| optional_string_value(column.as_ref(), row, "fraction"))
            .transpose()?
            .flatten();
        if let Some(values) = list_row_values(samples.as_ref(), row, "samples")? {
            let values = values
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| invalid_input("run.samples must be a list of structs"))?;
            for index in 0..values.len() {
                records.push(sample_record(values, index, &run, fraction.clone())?);
            }
        }
    }
    Ok(())
}

fn sample_record(
    samples: &StructArray,
    index: usize,
    run: &str,
    fraction: Option<String>,
) -> Result<SdrfRecord> {
    let required = |name| {
        let column = samples
            .column_by_name(name)
            .ok_or_else(|| invalid_input(format!("missing run.samples.{name}")))?;
        required_string_value(column.as_ref(), index, name)
    };
    let sample_accession = required("sample_accession")?;
    Ok(SdrfRecord {
        sample_accession: sample_accession.clone(),
        run_accession: Some(run.to_owned()),
        data_file: run.to_owned(),
        file_uri: None,
        label: Some(required("label")?),
        fraction,
        biological_replicate: replicate(samples, index, "biological_replicate")?,
        technical_replicate: replicate(samples, index, "technical_replicate")?,
        condition: Some(sample_accession),
        pooled_sample: None,
    })
}

fn replicate(samples: &StructArray, row: usize, name: &str) -> Result<Option<u32>> {
    samples
        .column_by_name(name)
        .map(|column| {
            optional_i32_value(column.as_ref(), row, name)?
                .map(|value| {
                    u32::try_from(value)
                        .map_err(|_| invalid_input(format!("invalid run.samples.{name}: {value}")))
                })
                .transpose()
        })
        .transpose()
        .map(Option::flatten)
}

fn invalid_input(error: impl std::fmt::Display) -> MokumeError {
    MokumeError::InvalidInput {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SdrfTable;
    use arrow::array::{ArrayRef, Int32Array, ListArray, StringArray};
    use arrow::buffer::{OffsetBuffer, ScalarBuffer};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn sample_array(labels: &[&str], tmt: bool) -> StructArray {
        let names = if tmt {
            vec!["S0", "S1"]
        } else {
            vec!["S", "S"]
        };
        StructArray::from(vec![
            (
                Arc::new(Field::new("sample_accession", DataType::Utf8, false)),
                Arc::new(StringArray::from(names)) as ArrayRef,
            ),
            (
                Arc::new(Field::new("label", DataType::Utf8, false)),
                Arc::new(StringArray::from(labels.to_vec())) as ArrayRef,
            ),
            (
                Arc::new(Field::new("biological_replicate", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![2, 2])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("technical_replicate", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![3, 3])) as ArrayRef,
            ),
        ])
    }

    fn write_run_fixture(path: &Path, tmt: bool) -> Result<()> {
        let labels = if tmt {
            vec!["TMT126", "TMT127"]
        } else {
            vec!["LFQ", "LFQ"]
        };
        let samples = sample_array(&labels, tmt);
        let offsets = if tmt { vec![0, 2] } else { vec![0, 1, 2] };
        let samples = ListArray::new(
            Arc::new(Field::new("item", samples.data_type().clone(), true)),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(samples),
            None,
        );
        let runs = if tmt {
            vec!["run_a"]
        } else {
            vec!["run_a", "run_b"]
        };
        let fractions = if tmt { vec!["1"] } else { vec!["1", "2"] };
        let fields = vec![
            Field::new("run_file_name", DataType::Utf8, false),
            Field::new("samples", samples.data_type().clone(), true),
            Field::new("fraction", DataType::Utf8, true),
        ];
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            vec![
                Arc::new(StringArray::from(runs)),
                Arc::new(samples),
                Arc::new(StringArray::from(fractions)),
            ],
        )
        .map_err(invalid_input)?;
        let mut writer = ArrowWriter::try_new(
            File::create(path).map_err(invalid_input)?,
            batch.schema(),
            None,
        )
        .map_err(invalid_input)?;
        writer.write(&batch).map_err(invalid_input)?;
        writer.close().map_err(invalid_input)?;
        Ok(())
    }

    #[test]
    fn run_view_maps_channels_and_fractionated_samples_exactly(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("run.parquet");
        for tmt in [false, true] {
            write_run_fixture(&path, tmt)?;
            let table = SdrfTable::from_run_path(&path)?;
            let a = table.lookup("run_a", Some(if tmt { "TMT126" } else { "LFQ" }))?;
            let b = table.lookup(
                if tmt { "run_a" } else { "run_b" },
                Some(if tmt { "TMT127" } else { "LFQ" }),
            )?;
            assert_eq!(a.sample_accession, if tmt { "S0" } else { "S" });
            assert_eq!(b.sample_accession, if tmt { "S1" } else { "S" });
            assert_eq!(b.fraction.as_deref(), Some(if tmt { "1" } else { "2" }));
            assert_eq!(a.biological_replicate, Some(2));
            assert_eq!(a.technical_replicate, Some(3));
            assert_eq!(a.condition.as_deref(), Some(a.sample_accession.as_str()));
            assert_eq!(b.condition.as_deref(), Some(b.sample_accession.as_str()));
            assert!(table.lookup("RUN_A", Some("LFQ")).is_err());
        }
        Ok(())
    }
}
