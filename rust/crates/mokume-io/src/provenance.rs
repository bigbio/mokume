//! Check upstream isotope correction before applying another impurity table.

use std::fs::File;
use std::path::Path;

use arrow::array::{Array, StructArray};
use arrow::record_batch::RecordBatch;
use mokume_core::{MokumeError, Result};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::qpx::{list_row_values, optional_string_value};

/// Reject another isotope correction when companion provenance records one.
pub fn reject_isotope_corrected(feature: &Path) -> Result<()> {
    let Some(prefix) = feature
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".feature.parquet"))
    else {
        return Ok(());
    };
    let path = feature.with_file_name(format!("{prefix}.provenance.parquet"));
    if !path.exists() {
        return Ok(());
    }
    let file = File::open(&path).map_err(|source| MokumeError::Io {
        path: path.clone(),
        source,
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| invalid_input(error.to_string()))?;
    let roots = builder
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| {
            ["tool_name", "parameters"]
                .contains(&field.name().as_str())
                .then_some(index)
        });
    let projection = ProjectionMask::roots(builder.parquet_schema(), roots);
    let reader = builder
        .with_projection(projection)
        .build()
        .map_err(|error| invalid_input(error.to_string()))?;
    for batch in reader {
        let batch = batch.map_err(|error| invalid_input(error.to_string()))?;
        if let Some(tool) = corrected_tool(&batch)? {
            return Err(invalid_input(format!(
                "--tmt-impurity cannot correct intensities again: {} records isotope_corrected=true for tool `{tool}`; another correction would subtract isotope impurities twice",
                path.display()
            )));
        }
    }
    Ok(())
}

fn corrected_tool(batch: &RecordBatch) -> Result<Option<String>> {
    let Some(parameters) = batch.column_by_name("parameters") else {
        return Ok(None);
    };
    for row in 0..batch.num_rows() {
        let Some(values) = list_row_values(parameters.as_ref(), row, "parameters")? else {
            continue;
        };
        let fields = values
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                invalid_input("QPX provenance parameters must contain key/value structs")
            })?;
        if corrected_parameters(fields)? {
            let tools = batch
                .column_by_name("tool_name")
                .ok_or_else(|| invalid_input("QPX provenance has no tool_name"))?;
            return optional_string_value(tools.as_ref(), row, "tool_name")?
                .map(Some)
                .ok_or_else(|| invalid_input("QPX provenance tool_name is null"));
        }
    }
    Ok(None)
}

fn corrected_parameters(fields: &StructArray) -> Result<bool> {
    let keys = parameter_field(fields, "key")?;
    let values = parameter_field(fields, "value")?;
    for index in 0..fields.len() {
        if optional_string_value(keys, index, "parameters.key")?.as_deref()
            == Some("isotope_corrected")
            && optional_string_value(values, index, "parameters.value")?.as_deref() == Some("true")
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn parameter_field<'a>(fields: &'a StructArray, name: &str) -> Result<&'a dyn Array> {
    fields
        .column_by_name(name)
        .map(|array| array.as_ref())
        .ok_or_else(|| invalid_input(format!("QPX provenance parameters.{name} is missing")))
}

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, ListBuilder, StringArray, StringBuilder, StructBuilder};
    use arrow::datatypes::{DataType, Field};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use super::reject_isotope_corrected;

    fn write_provenance(
        path: &Path,
        value: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fields = vec![
            Field::new("key", DataType::Utf8, true),
            Field::new("value", DataType::Utf8, true),
        ];
        let mut parameters = ListBuilder::new(StructBuilder::from_fields(fields, 2));
        for (key, value) in [
            ("normalization", "none"),
            (
                if value.is_some() {
                    "isotope_corrected"
                } else {
                    "other"
                },
                value.unwrap_or("unknown"),
            ),
        ] {
            parameters
                .values()
                .field_builder::<StringBuilder>(0)
                .ok_or("key builder")?
                .append_value(key);
            parameters
                .values()
                .field_builder::<StringBuilder>(1)
                .ok_or("value builder")?
                .append_value(value);
            parameters.values().append(true);
            parameters.append(true);
        }
        let batch = RecordBatch::try_from_iter([
            (
                "tool_name",
                Arc::new(StringArray::from(vec!["qpx", "upstream-tool"])) as ArrayRef,
            ),
            ("parameters", Arc::new(parameters.finish()) as ArrayRef),
        ])?;
        let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    #[test]
    fn rejects_only_explicit_upstream_isotope_correction() -> Result<(), Box<dyn std::error::Error>>
    {
        let directory = tempfile::tempdir()?;
        let feature = directory.path().join("sample.feature.parquet");
        let provenance = directory.path().join("sample.provenance.parquet");
        assert!(reject_isotope_corrected(&feature).is_ok());
        for value in [None, Some("false"), Some("true")] {
            write_provenance(&provenance, value)?;
            let result = reject_isotope_corrected(&feature);
            if value == Some("true") {
                let message = result
                    .err()
                    .ok_or("corrected input was accepted")?
                    .to_string();
                assert!(message.contains("sample.provenance.parquet"));
                assert!(message.contains("upstream-tool"));
                assert!(message.contains("twice"));
            } else {
                assert!(result.is_ok());
            }
        }
        Ok(())
    }
}
