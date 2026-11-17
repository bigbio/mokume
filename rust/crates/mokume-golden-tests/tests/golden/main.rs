#![cfg(test)]

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::{File, create_dir_all};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    ArrayRef, BooleanArray, BooleanBuilder, Float32Builder, Float64Array, Float64Builder,
    Int16Array, Int32Builder, ListBuilder, StringArray, StringBuilder, StructBuilder,
};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use mokume_core::{
    AggregationLevel, BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig,
    FeatureToPeptidesConfig, FeatureToProteinsConfig, FilterConfig, ImputationConfig, InputConfig,
    IntensityFilterConfig, IrsChannelConfig, IrsConfig, IrsScope, IrsStat, MaxLfqConfig,
    MokumeError, NamedScoreFilterConfig, NormalizationConfig, OutputConfig, OutputFormat,
    PeptideFilterConfig, PibaqConfig, PibaqSharedMode, PreprocessingFilterConfig,
    ProteinFilterConfig, QuantMethod, RatioConfig, RunQcFilterConfig, RuntimeConfig, TmtConfig,
    TransferredFeatures,
};
use mokume_pipeline::{
    PibaqDigest, PibaqDigestProvenance, run_features_to_peptides, run_features_to_proteins,
    run_features_to_proteins_with_pibaq_digest,
};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

mod assertions;
mod exports;
mod fixtures;
mod peptides_filters;
mod peptides_normalization_irs;
mod peptides_oracles;
mod peptides_scores_qc;
mod proteins_basics;
mod proteins_fdr;
mod proteins_imputation_de;
mod proteins_methods;
mod proteins_normalization;
mod runners;

use assertions::*;
use fixtures::*;
use runners::*;
