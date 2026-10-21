//! Minimal AnnData `.h5ad` writer used by `correct-batches --export-anndata`.
//!
//! This reproduces the on-disk layout that Python's
//! `anndata.AnnData(...).write(...)` (via `mokume.io.parquet.create_anndata`)
//! emits for the batch-correction export, so the resulting file round-trips
//! through `anndata.read_h5ad`. The Python path builds the object as:
//!
//! ```python
//! adata = an.AnnData(
//!     X=df_matrix.to_numpy(),       # obs (samples) x var (proteins), float64
//!     obs=df_matrix.index.to_frame(),   # one column: SampleID
//!     var=df_matrix.columns.to_frame(), # one column: ProteinName
//! )
//! adata.layers["PiBAQBec"] = df_layer.to_numpy()
//! ```
//!
//! The AnnData HDF5 schema we mirror (captured from a reference file produced by
//! anndata 0.x / h5py with libhdf5 2.0.0) is:
//!
//! ```text
//! /                         @encoding-type=anndata   @encoding-version=0.1.0
//! /X        f8 (obs,var)    @encoding-type=array     @encoding-version=0.2.0
//! /layers                   @encoding-type=dict      @encoding-version=0.1.0
//! /layers/PiBAQBec  f8       @encoding-type=array     @encoding-version=0.2.0
//! /obs                      @encoding-type=dataframe @encoding-version=0.2.0
//!                           @_index=SampleID         @column-order=[SampleID]
//! /obs/SampleID  str        @encoding-type=string-array  @encoding-version=0.2.0
//! /var                      @encoding-type=dataframe @encoding-version=0.2.0
//!                           @_index=ProteinName      @column-order=[ProteinName]
//! /var/ProteinName  str     @encoding-type=string-array  @encoding-version=0.2.0
//! /obsm /obsp /uns /varm /varp   (empty)  @encoding-type=dict  @encoding-version=0.1.0
//! ```
//!
//! Strings are written as variable-length UTF-8, matching what h5py emits and
//! what `anndata.read_h5ad` expects for the dataframe index columns.
//!
//! Build prerequisite: this uses `hdf5-metno` with the `static` feature, which
//! compiles a vendored libhdf5 from source at build time. That needs a C
//! compiler (`cc`/`gcc`) and `cmake` on the build host. There is no system
//! `libhdf5` requirement at build or run time.

use std::path::Path;
use std::str::FromStr;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group, Location};
use mokume_core::{MokumeError, Result};
use ndarray::Array2;

/// AnnData encoding-version strings, kept as constants so the layout stays in
/// lock-step with the reference file documented above.
const ENC_VERSION_ANNDATA: &str = "0.1.0";
const ENC_VERSION_ARRAY: &str = "0.2.0";
const ENC_VERSION_DICT: &str = "0.1.0";
const ENC_VERSION_DATAFRAME: &str = "0.2.0";
const ENC_VERSION_STRING_ARRAY: &str = "0.2.0";

/// The data needed to materialise the AnnData export. `x` and each layer are
/// row-major `obs` (samples) x `var` (proteins) matrices; their column count
/// must equal `var_names.len()` and their row count `obs_names.len()`.
pub struct AnnDataExport<'a> {
    /// Observation (sample) identifiers, one per matrix row.
    pub obs_names: &'a [String],
    /// The name of the obs index column (e.g. `SampleID`).
    pub obs_index_name: &'a str,
    /// Variable (protein) identifiers, one per matrix column.
    pub var_names: &'a [String],
    /// The name of the var index column (e.g. `ProteinName`).
    pub var_index_name: &'a str,
    /// The main `X` matrix, `obs` rows by `var` columns.
    pub x: &'a [Vec<f64>],
    /// Additional named layers, each the same shape as `x`.
    pub layers: &'a [(String, Vec<Vec<f64>>)],
}

/// Wrap an `hdf5_metno::Error` in a `MokumeError` with file context.
fn h5_error(path: &Path, message: impl AsRef<str>) -> MokumeError {
    MokumeError::InvalidInput {
        message: format!(
            "Failed to write AnnData file '{}': {}",
            path.display(),
            message.as_ref()
        ),
    }
}

/// Convert a `&str` into a variable-length unicode string for HDF5.
fn vlen(path: &Path, value: &str) -> Result<VarLenUnicode> {
    VarLenUnicode::from_str(value)
        .map_err(|source| h5_error(path, format!("invalid UTF-8 string '{value}': {source}")))
}

/// Write a scalar variable-length string attribute onto `location`.
fn write_str_attr(path: &Path, location: &Location, name: &str, value: &str) -> Result<()> {
    let attr = location
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)
        .map_err(|source| h5_error(path, format!("creating attribute '{name}': {source}")))?;
    attr.as_writer()
        .write_scalar(&vlen(path, value)?)
        .map_err(|source| h5_error(path, format!("writing attribute '{name}': {source}")))
}

/// Write a 1-D variable-length string array attribute (used for `column-order`).
fn write_str_array_attr(
    path: &Path,
    location: &Location,
    name: &str,
    values: &[&str],
) -> Result<()> {
    let data = values
        .iter()
        .map(|value| vlen(path, value))
        .collect::<Result<Vec<_>>>()?;
    let view = ndarray::ArrayView1::from(&data);
    location
        .new_attr::<VarLenUnicode>()
        .shape([values.len()])
        .create(name)
        .map_err(|source| h5_error(path, format!("creating attribute '{name}': {source}")))?
        .as_writer()
        .write(view)
        .map_err(|source| h5_error(path, format!("writing attribute '{name}': {source}")))
}

/// Tag a group/dataset with the AnnData `encoding-type` / `encoding-version`
/// attribute pair.
fn write_encoding(
    path: &Path,
    location: &Location,
    enc_type: &str,
    enc_version: &str,
) -> Result<()> {
    write_str_attr(path, location, "encoding-type", enc_type)?;
    write_str_attr(path, location, "encoding-version", enc_version)
}

/// Create an empty AnnData "dict" group (obsm/obsp/uns/varm/varp).
fn create_empty_dict_group(path: &Path, parent: &Group, name: &str) -> Result<()> {
    let group = parent
        .create_group(name)
        .map_err(|source| h5_error(path, format!("creating group '{name}': {source}")))?;
    write_encoding(path, &group, "dict", ENC_VERSION_DICT)
}

/// Flatten a row-major `obs x var` matrix into an owned `ndarray::Array2<f64>`
/// for the HDF5 writer. Validates that every row matches the var count.
fn to_array2(path: &Path, rows: &[Vec<f64>], n_var: usize, label: &str) -> Result<Array2<f64>> {
    let n_obs = rows.len();
    let mut flat = Vec::with_capacity(n_obs * n_var);
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != n_var {
            return Err(h5_error(
                path,
                format!(
                    "{label} row {row_idx} has {} columns, expected {n_var}",
                    row.len()
                ),
            ));
        }
        flat.extend_from_slice(row);
    }
    Array2::from_shape_vec((n_obs, n_var), flat)
        .map_err(|source| h5_error(path, format!("reshaping {label} matrix: {source}")))
}

/// Write a dense float64 matrix dataset with AnnData `array` encoding.
fn write_array_dataset(path: &Path, parent: &Group, name: &str, data: &Array2<f64>) -> Result<()> {
    let dataset = parent
        .new_dataset_builder()
        .with_data(data)
        .create(name)
        .map_err(|source| h5_error(path, format!("creating dataset '{name}': {source}")))?;
    write_encoding(path, &dataset, "array", ENC_VERSION_ARRAY)
}

/// Write an AnnData dataframe group holding a single string index column.
/// This mirrors `df.index.to_frame()` / `df.columns.to_frame()` on the Python
/// side, which produces a one-column frame whose only column is the index.
fn write_index_dataframe(
    path: &Path,
    parent: &Group,
    group_name: &str,
    index_name: &str,
    values: &[String],
) -> Result<()> {
    let group = parent
        .create_group(group_name)
        .map_err(|source| h5_error(path, format!("creating group '{group_name}': {source}")))?;
    write_encoding(path, &group, "dataframe", ENC_VERSION_DATAFRAME)?;
    write_str_attr(path, &group, "_index", index_name)?;
    write_str_array_attr(path, &group, "column-order", &[index_name])?;

    let strings = values
        .iter()
        .map(|value| vlen(path, value))
        .collect::<Result<Vec<_>>>()?;
    let dataset = group
        .new_dataset_builder()
        .with_data(&ndarray::ArrayView1::from(&strings))
        .create(index_name)
        .map_err(|source| h5_error(path, format!("creating dataset '{index_name}': {source}")))?;
    write_encoding(path, &dataset, "string-array", ENC_VERSION_STRING_ARRAY)
}

/// Write the AnnData export to `path`, overwriting any existing file.
pub fn write_h5ad(path: &Path, export: &AnnDataExport<'_>) -> Result<()> {
    let n_obs = export.obs_names.len();
    let n_var = export.var_names.len();
    if export.x.len() != n_obs {
        return Err(h5_error(
            path,
            format!(
                "X has {} rows but there are {n_obs} observations",
                export.x.len()
            ),
        ));
    }

    let x = to_array2(path, export.x, n_var, "X")?;

    let file =
        File::create(path).map_err(|source| h5_error(path, format!("creating file: {source}")))?;
    write_encoding(path, &file, "anndata", ENC_VERSION_ANNDATA)?;

    write_array_dataset(path, &file, "X", &x)?;

    write_index_dataframe(path, &file, "obs", export.obs_index_name, export.obs_names)?;
    write_index_dataframe(path, &file, "var", export.var_index_name, export.var_names)?;

    let layers = file
        .create_group("layers")
        .map_err(|source| h5_error(path, format!("creating group 'layers': {source}")))?;
    write_encoding(path, &layers, "dict", ENC_VERSION_DICT)?;
    for (layer_name, layer_rows) in export.layers {
        let layer = to_array2(path, layer_rows, n_var, layer_name)?;
        write_array_dataset(path, &layers, layer_name, &layer)?;
    }

    for name in ["obsm", "obsp", "uns", "varm", "varp"] {
        create_empty_dict_group(path, &file, name)?;
    }

    file.close()
        .map_err(|source| h5_error(path, format!("closing file: {source}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::time::{SystemTime, UNIX_EPOCH};

    type TestResult<T> = std::result::Result<T, Box<dyn Error>>;

    fn temp_path(tag: &str) -> TestResult<(tempfile::TempDir, std::path::PathBuf)> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = tempfile::Builder::new()
            .prefix(&format!("mokume-h5ad-{tag}-{nanos}-"))
            .tempdir()?;
        let path = directory.path().join("out.h5ad");
        Ok((directory, path))
    }

    #[test]
    fn writes_anndata_layout_that_reads_back() -> TestResult<()> {
        let (_tempdir, path) = temp_path("layout")?;
        let obs = vec!["B1-s1".to_string(), "B1-s2".to_string()];
        let var = vec!["P1".to_string(), "P2".to_string(), "P3".to_string()];
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let layer = vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]];
        let layers = vec![("PiBAQBec".to_string(), layer)];
        let export = AnnDataExport {
            obs_names: &obs,
            obs_index_name: "SampleID",
            var_names: &var,
            var_index_name: "ProteinName",
            x: &x,
            layers: &layers,
        };
        write_h5ad(&path, &export)?;

        // Read the file back with the hdf5 crate and assert the schema/values.
        let file = File::open(&path)?;
        assert_eq!(
            file.attr("encoding-type")?
                .as_reader()
                .read_scalar::<VarLenUnicode>()?
                .as_str(),
            "anndata"
        );

        let x_ds = file.dataset("X")?;
        assert_eq!(x_ds.shape(), [2, 3]);
        let x_read: Array2<f64> = x_ds.read_2d()?;
        assert_eq!(
            x_read,
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?
        );
        assert_eq!(
            x_ds.attr("encoding-type")?
                .as_reader()
                .read_scalar::<VarLenUnicode>()?
                .as_str(),
            "array"
        );

        let obs_group = file.group("obs")?;
        assert_eq!(
            obs_group
                .attr("_index")?
                .as_reader()
                .read_scalar::<VarLenUnicode>()?
                .as_str(),
            "SampleID"
        );
        let sample_ids: Vec<VarLenUnicode> = obs_group.dataset("SampleID")?.read_raw()?;
        let sample_ids: Vec<String> = sample_ids.iter().map(|s| s.as_str().to_string()).collect();
        assert_eq!(sample_ids, vec!["B1-s1".to_string(), "B1-s2".to_string()]);

        let var_group = file.group("var")?;
        let protein_names: Vec<VarLenUnicode> = var_group.dataset("ProteinName")?.read_raw()?;
        let protein_names: Vec<String> = protein_names
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();
        assert_eq!(
            protein_names,
            vec!["P1".to_string(), "P2".to_string(), "P3".to_string()]
        );

        let layer_ds = file.dataset("layers/PiBAQBec")?;
        let layer_read: Array2<f64> = layer_ds.read_2d()?;
        assert_eq!(
            layer_read,
            Array2::from_shape_vec((2, 3), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])?
        );

        for name in ["obsm", "obsp", "uns", "varm", "varp"] {
            assert!(file.group(name).is_ok(), "missing empty group {name}");
        }
        Ok(())
    }
}
