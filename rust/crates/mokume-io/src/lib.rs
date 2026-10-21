pub mod feature_mapping;
pub mod msstats;
pub mod peptide_parquet;
pub mod psm;
pub mod qpx;
pub mod sdrf;

pub use arrow::record_batch::RecordBatch;
pub use feature_mapping::{
    flatten_feature_protein_groups, QpxFeatureProteinGroup, QpxFeatureProteinGroupReader,
};
pub use msstats::MsstatsReader;
pub use peptide_parquet::{
    read_peptide_parquet, read_peptide_parquet_with_species, write_peptide_parquet,
    PeptideParquetRow, RawPeptideRow, RawPeptideTable,
};
pub use psm::{flatten_psm_batch, QpxPsmParquetReader, QpxPsmRecord};
pub use qpx::{
    flatten_qpx_batch, flatten_qpx_batch_with_score, QpxFeatureRecord, QpxIntensityEntry,
    QpxParquetReader, QpxScoreValue,
};
pub use sdrf::{
    normalize_file_key, normalize_label_key, SdrfLookupCache, SdrfRawTable, SdrfRecord, SdrfTable,
};

pub const DEFAULT_QPX_BATCH_SIZE: usize = 65_536;
