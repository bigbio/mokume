pub mod feature_mapping;
pub mod msstats;
pub mod peptide_parquet;
mod provenance;
pub mod psm;
pub mod qpx;
pub mod sdrf;

pub use arrow::record_batch::RecordBatch;
pub use feature_mapping::{
    QpxFeatureProteinGroup, QpxFeatureProteinGroupReader, flatten_feature_protein_groups,
};
pub use msstats::MsstatsReader;
pub use peptide_parquet::{
    PeptideParquetRow, RawPeptideRow, RawPeptideTable, read_peptide_parquet,
    read_peptide_parquet_with_species, read_peptide_parquet_with_weights, write_peptide_parquet,
};
pub use provenance::reject_isotope_corrected;
pub use psm::{QpxPsmParquetReader, QpxPsmRecord, flatten_psm_batch};
pub use qpx::{
    QpxFeatureRecord, QpxIntensityEntry, QpxParquetReader, QpxScoreValue, flatten_qpx_batch,
    flatten_qpx_batch_with_score,
};
pub use sdrf::{
    SdrfLookupCache, SdrfRawTable, SdrfRecord, SdrfTable, SharedSourceName, normalize_file_key,
    normalize_label_key,
};

pub const DEFAULT_QPX_BATCH_SIZE: usize = 65_536;

mod protein_groups;
pub use protein_groups::{ProteinGroupOrders, parse_protein_accession};

mod run;
