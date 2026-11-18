use super::*;

/// Look up the `NormIntensity` cell of the peptide long table for a given
/// (ProteinName, PeptideCanonical, SampleID), returning the parsed value.
pub(crate) fn peptide_norm_intensity(
    table: &CsvTable,
    protein: &str,
    peptide: &str,
    sample: &str,
) -> Result<f64, Box<dyn Error>> {
    let column = |name: &str| {
        table
            .headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| -> Box<dyn Error> { format!("column '{name}' is missing").into() })
    };
    let protein_idx = column("ProteinName")?;
    let peptide_idx = column("PeptideCanonical")?;
    let sample_idx = column("SampleID")?;
    let value_idx = column("NormIntensity")?;
    let row = table
        .rows
        .iter()
        .find(|row| {
            row.get(protein_idx).is_some_and(|value| value == protein)
                && row.get(peptide_idx).is_some_and(|value| value == peptide)
                && row.get(sample_idx).is_some_and(|value| value == sample)
        })
        .ok_or_else(|| -> Box<dyn Error> {
            format!("peptide row ({protein}, {peptide}, {sample}) is missing").into()
        })?;
    let value = row
        .get(value_idx)
        .ok_or_else(|| -> Box<dyn Error> { "NormIntensity cell is missing".into() })?;
    value
        .parse::<f64>()
        .map_err(|_| format!("'{value}' is not numeric").into())
}

pub(crate) fn assert_peptide_cell(
    table: &CsvTable,
    protein: &str,
    peptide: &str,
    sample: &str,
    expected: f64,
) -> Result<(), Box<dyn Error>> {
    let actual = peptide_norm_intensity(table, protein, peptide, sample)?;
    assert!(
        (actual - expected).abs() <= 1e-9,
        "({protein}, {peptide}, {sample}): got {actual}, expected {expected}"
    );
    Ok(())
}

/// Assert the `Condition` cell of the peptide long table for a given
/// (ProteinName, PeptideCanonical, SampleID). On a no-SDRF / SDRF-miss path
/// Python's QPX view defaults `Condition` to the run/`sample_accession`
/// fallback (`feature.py:_create_unnest_view`, `sa_default = run_file_name`),
/// so the expected value is the SampleID. Asserting it here guards the Rust
/// fallback against regressing to the old literal "Empty".
pub(crate) fn assert_peptide_condition(
    table: &CsvTable,
    protein: &str,
    peptide: &str,
    sample: &str,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    let column = |name: &str| {
        table
            .headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| -> Box<dyn Error> { format!("column '{name}' is missing").into() })
    };
    let protein_idx = column("ProteinName")?;
    let peptide_idx = column("PeptideCanonical")?;
    let sample_idx = column("SampleID")?;
    let condition_idx = column("Condition")?;
    let row = table
        .rows
        .iter()
        .find(|row| {
            row.get(protein_idx).is_some_and(|value| value == protein)
                && row.get(peptide_idx).is_some_and(|value| value == peptide)
                && row.get(sample_idx).is_some_and(|value| value == sample)
        })
        .ok_or_else(|| -> Box<dyn Error> {
            format!("peptide row ({protein}, {peptide}, {sample}) is missing").into()
        })?;
    let actual = row
        .get(condition_idx)
        .ok_or_else(|| -> Box<dyn Error> { "Condition cell is missing".into() })?;
    assert_eq!(
        actual, expected,
        "({protein}, {peptide}, {sample}) Condition: got {actual:?}, expected {expected:?}"
    );
    Ok(())
}

pub(crate) fn assert_sum_table(actual: &CsvTable) -> Result<(), Box<dyn Error>> {
    let expected = read_csv(Path::new(
        "fixtures/python/features2proteins_sum_python_compatible.csv",
    ))?;

    assert_eq!(actual.headers, expected.headers);
    assert_eq!(actual.rows.len(), expected.rows.len());

    for (actual, expected) in actual.rows.iter().zip(expected.rows.iter()) {
        assert_eq!(actual.first(), expected.first());
        for column in 1..actual.len() {
            assert_cell_close(actual.get(column), expected.get(column));
        }
    }

    assert!(
        actual
            .rows
            .iter()
            .all(|record| record.first().is_none_or(|value| value != "P2")),
        "single-unique-peptide protein must be filtered"
    );
    assert!(
        actual.rows.iter().all(|record| record
            .first()
            .is_none_or(|value| value != "sp|CONTAMINANT_P00761|TRYP_PIG")),
        "contaminant protein must be filtered"
    );
    assert!(
        actual.rows.iter().all(|record| record
            .first()
            .is_none_or(|value| value != "sp|DECOY_P11111|REV_HUMAN")),
        "decoy protein must be filtered"
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CsvTable {
    pub(crate) headers: Vec<String>,
    pub(crate) rows: Vec<Vec<String>>,
}

pub(crate) fn read_csv(path: &Path) -> Result<CsvTable, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.iter().map(ToOwned::to_owned).collect();
    let rows = reader
        .records()
        .map(|record| record.map(|record| record.iter().map(ToOwned::to_owned).collect::<Vec<_>>()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CsvTable { headers, rows })
}

/// Decoded peptide parquet matrix for golden-test assertions: column names and
/// their Arrow dtype labels (compared against Python's WriteParquetTask schema),
/// plus the protein/peptide/sample/intensity values used for cell comparisons.
pub(crate) struct PeptideParquetTable {
    pub(crate) column_names: Vec<String>,
    pub(crate) dtypes: Vec<String>,
    pub(crate) rows: Vec<(String, String, String, f64)>,
}

/// Read one Utf8-or-Dictionary(Int8, Utf8) string cell from `array` at `row`,
/// transparently decoding the pandas-Categorical dictionary columns that
/// Python's WriteParquetTask emits for SampleID/Condition.
pub(crate) fn parquet_string_cell(
    array: &dyn arrow::array::Array,
    row: usize,
) -> Result<String, Box<dyn Error>> {
    use arrow::array::{DictionaryArray, StringArray};
    use arrow::datatypes::Int8Type;

    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row).to_owned());
    }
    let dict = array
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .expect("expected utf8 or dictionary column");
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary values are not utf8");
    let key = dict.keys().value(row);
    Ok(values.value(key as usize).to_owned())
}

pub(crate) fn header_index(names: &[String], name: &str) -> Result<usize, Box<dyn Error>> {
    names
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| format!("parquet column '{name}' missing").into())
}

pub(crate) fn read_peptide_parquet(path: &Path) -> Result<PeptideParquetTable, Box<dyn Error>> {
    use arrow::array::Float32Array;

    let file = File::open(path)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let batch = reader.next().expect("empty parquet")?;
    let schema = batch.schema();
    let column_names = schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();
    let dtypes = schema
        .fields()
        .iter()
        .map(|field| format!("{:?}", field.data_type()))
        .collect::<Vec<_>>();

    let protein_col = batch.column(header_index(&column_names, "ProteinName")?);
    let peptide_col = batch.column(header_index(&column_names, "PeptideCanonical")?);
    let sample_col = batch.column(header_index(&column_names, "SampleID")?);
    let intensity_col = batch
        .column(header_index(&column_names, "NormIntensity")?)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("NormIntensity is not float32");

    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        rows.push(read_peptide_row(
            protein_col.as_ref(),
            peptide_col.as_ref(),
            sample_col.as_ref(),
            intensity_col,
            row,
        )?);
    }
    Ok(PeptideParquetTable {
        column_names,
        dtypes,
        rows,
    })
}

pub(crate) fn read_peptide_row(
    protein_col: &dyn arrow::array::Array,
    peptide_col: &dyn arrow::array::Array,
    sample_col: &dyn arrow::array::Array,
    intensity_col: &arrow::array::Float32Array,
    row: usize,
) -> Result<(String, String, String, f64), Box<dyn Error>> {
    Ok((
        parquet_string_cell(protein_col, row)?,
        parquet_string_cell(peptide_col, row)?,
        parquet_string_cell(sample_col, row)?,
        f64::from(intensity_col.value(row)),
    ))
}

pub(crate) fn parquet_norm_intensity(
    table: &PeptideParquetTable,
    protein: &str,
    peptide: &str,
    sample: &str,
) -> Result<f64, Box<dyn Error>> {
    table
        .rows
        .iter()
        .find(|(p, pep, s, _)| p == protein && pep == peptide && s == sample)
        .map(|(_, _, _, value)| *value)
        .ok_or_else(|| format!("parquet row ({protein}, {peptide}, {sample}) is missing").into())
}

pub(crate) fn numeric_cell(
    table: &CsvTable,
    protein: &str,
    sample: &str,
) -> Result<f64, Box<dyn Error>> {
    let column = table
        .headers
        .iter()
        .position(|header| header == sample)
        .ok_or_else(|| format!("sample column is missing: {sample}"))?;
    let row = table
        .rows
        .iter()
        .find(|row| row.first().is_some_and(|value| value == protein))
        .ok_or_else(|| format!("protein row is missing: {protein}"))?;
    Ok(row
        .get(column)
        .ok_or_else(|| format!("cell {protein}/{sample} is missing"))?
        .parse::<f64>()?)
}

pub(crate) fn assert_numeric_cell_close(
    table: &CsvTable,
    protein: &str,
    sample: &str,
    expected: f64,
) {
    let Some(sample_index) = table.headers.iter().position(|header| header == sample) else {
        panic!("sample column is missing: {sample}");
    };
    let Some(row) = table
        .rows
        .iter()
        .find(|row| row.first().is_some_and(|value| value == protein))
    else {
        panic!("protein row is missing: {protein}");
    };
    let Some(value) = row.get(sample_index) else {
        panic!("sample column {sample} is missing from row {protein}");
    };
    let Ok(actual) = value.parse::<f64>() else {
        panic!("numeric cell is not a valid float: {value}");
    };
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1e-9,
        "actual {actual} differs from expected {expected} by {delta}"
    );
}

/// Like `assert_numeric_cell_close` but allows a relative tolerance. Used for
/// LOESS, where the Rust Cleveland LOWESS reimplementation differs from
/// statsmodels' `lowess` by sub-0.2% on the smoothed bias curve (delta
/// interpolation / window-edge arithmetic), so cell-exact 1e-9 is not achievable.
pub(crate) fn assert_numeric_cell_relative(
    table: &CsvTable,
    protein: &str,
    sample: &str,
    expected: f64,
    relative_tolerance: f64,
) {
    let Some(sample_index) = table.headers.iter().position(|header| header == sample) else {
        panic!("sample column is missing: {sample}");
    };
    let Some(row) = table
        .rows
        .iter()
        .find(|row| row.first().is_some_and(|value| value == protein))
    else {
        panic!("protein row is missing: {protein}");
    };
    let Some(value) = row.get(sample_index) else {
        panic!("sample column {sample} is missing from row {protein}");
    };
    let Ok(actual) = value.parse::<f64>() else {
        panic!("numeric cell is not a valid float: {value}");
    };
    let relative = (actual - expected).abs() / expected.abs();
    assert!(
        relative <= relative_tolerance,
        "actual {actual} differs from expected {expected} by relative {relative} (> {relative_tolerance})"
    );
}

pub(crate) fn assert_cell_close(actual: Option<&String>, expected: Option<&String>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) if actual.is_empty() && expected.is_empty() => {}
        (Some(actual), Some(expected)) => assert_float_close(actual, expected),
        _ => panic!("CSV row width mismatch: actual={actual:?}, expected={expected:?}"),
    }
}

pub(crate) fn assert_float_close(actual: &str, expected: &str) {
    let Ok(actual) = actual.parse::<f64>() else {
        panic!("numeric cell is not a valid float: {actual}");
    };
    let Ok(expected) = expected.parse::<f64>() else {
        panic!("oracle numeric cell is not a valid float: {expected}");
    };
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1e-9,
        "actual {actual} differs from expected {expected} by {delta}"
    );
}

pub(crate) fn temp_root() -> Result<(tempfile::TempDir, PathBuf), Box<dyn Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = tempfile::Builder::new()
        .prefix(&format!("mokume-golden-{timestamp}-"))
        .tempdir()?;
    let path = directory.path().to_path_buf();
    Ok((directory, path))
}
