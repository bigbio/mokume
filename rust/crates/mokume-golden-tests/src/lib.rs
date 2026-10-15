#![cfg(test)]

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::{create_dir_all, File};
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
    PeptideFilterConfig, PibaqConfig, PreprocessingFilterConfig, ProteinFilterConfig, QuantMethod,
    RatioConfig, RunQcFilterConfig, RuntimeConfig, TmtConfig,
};
use mokume_pipeline::{
    run_features_to_peptides, run_features_to_proteins, run_features_to_proteins_with_pibaq_digest,
    PibaqDigest, PibaqDigestProvenance,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

#[test]
fn features2proteins_sum_matches_synthetic_golden_matrix() -> Result<(), Box<dyn Error>> {
    let table = run_synthetic_quantification(QuantMethod::Sum, 3)?;
    assert_sum_table(&table)
}

#[test]
fn features2proteins_preserves_biological_replicate_peptide_rows() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("bioreplicate.features.parquet");
    let sdrf = root.join("bioreplicate.sdrf.tsv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 300.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 400.0, &["P1"]),
        ],
    )?;
    std::fs::write(
        &sdrf,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcharacteristics[biological replicate]\tfactor value[cell line]\n",
            "sample-1\trun 1\trun1.raw\tAC=MS:1002038;NT=label free sample\t1\tA\n",
            "sample-1\trun 2\trun2.raw\tAC=MS:1002038;NT=label free sample\t2\tA\n",
        ),
    )?;

    for (method, expected) in [
        (QuantMethod::Sum, 1000.0),
        (QuantMethod::Median, 250.0),
        (QuantMethod::TopN, 300.0),
        (QuantMethod::PeptideCount, 4.0),
    ] {
        let output = root.join(format!("{}.csv", method.as_str()));
        let mut config = default_sum_config(parquet.clone(), sdrf.clone(), output.clone());
        config.quantification = method;
        config.topn_peptides = 3;
        run_features_to_proteins(&config)?;
        let table = read_csv(&output)?;
        let actual = table
            .rows
            .first()
            .and_then(|row| row.get(1))
            .ok_or("missing protein matrix value")?
            .parse::<f64>()?;
        assert!(
            (actual - expected).abs() <= 1e-12,
            "{} collapsed biological-replicate peptide rows: {actual} != {expected}",
            method.as_str()
        );
    }
    Ok(())
}

#[test]
fn features2proteins_exports_python_compatible_peptide_intermediates() -> Result<(), Box<dyn Error>>
{
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("export.features.parquet");
    let sdrf = root.join("export.sdrf.tsv");
    let output = root.join("export.protein.csv");
    let peptides = root.join("export.peptides.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run1.raw", 150.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 300.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 400.0, &["P1"]),
            QpxRow::new("ONEPEPTK", "run1.raw", 999.0, &["P2"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output);
    config.output.export_peptides = Some(peptides.clone());
    run_features_to_proteins(&config)?;

    let table = read_csv(&peptides)?;
    assert_eq!(
        table.headers,
        vec![
            "ProteinName".to_owned(),
            "PeptideCanonical".to_owned(),
            "SampleID".to_owned(),
            "BioReplicate".to_owned(),
            "Condition".to_owned(),
            "NormIntensity".to_owned(),
        ]
    );
    assert_eq!(
        table.rows,
        vec![
            vec![
                "P1".to_owned(),
                "APEPTIDECK".to_owned(),
                "sample-1".to_owned(),
                "1".to_owned(),
                "A".to_owned(),
                "200".to_owned(),
            ],
            vec![
                "P1".to_owned(),
                "APEPTIDECK".to_owned(),
                "sample-2".to_owned(),
                "1".to_owned(),
                "B".to_owned(),
                "400".to_owned(),
            ],
            vec![
                "P1".to_owned(),
                "PEPTIDEAK".to_owned(),
                "sample-1".to_owned(),
                "1".to_owned(),
                "A".to_owned(),
                "150".to_owned(),
            ],
            vec![
                "P1".to_owned(),
                "PEPTIDEAK".to_owned(),
                "sample-2".to_owned(),
                "1".to_owned(),
                "B".to_owned(),
                "300".to_owned(),
            ],
        ]
    );
    Ok(())
}

// `--export-peptides` with a dataset-level sample normalization must carry that
// normalization into the exported peptides (Python applies it in
// `load_for_mokume` before writing the CSV). Real-data parity against Python is
// verified to machine epsilon on PXD003539 (quantile / mediancenter); this
// guards the wiring with quantile's defining property: when the per-sample ranks
// agree, quantile normalization equalizes the two samples' values.
#[test]
fn features2proteins_export_peptides_applies_dataset_normalization() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("qn.features.parquet");
    let sdrf = root.join("qn.sdrf.tsv");
    let output = root.join("qn.protein.csv");
    let peptides = root.join("qn.peptides.csv");

    // sample-1: APEPTIDECK=200 > PEPTIDEAK=150; sample-2: APEPTIDECK=400 >
    // PEPTIDEAK=300. Ranks agree across samples, so quantile normalization gives
    // each peptide the same value in both samples, distinct from the raw sums.
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 150.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 300.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 400.0, &["P1"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output);
    config.normalization.sample_method = "quantile".to_owned();
    config.output.export_peptides = Some(peptides.clone());
    run_features_to_proteins(&config)?;

    let table = read_csv(&peptides)?;
    // Dataset-level normalization pivots+melts in Python (stages.py:961-1047),
    // dropping BioReplicate/Condition, so the export is exactly four columns.
    assert_eq!(
        table.headers,
        vec![
            "ProteinName",
            "PeptideCanonical",
            "SampleID",
            "NormIntensity"
        ],
        "dataset-level normalization export must drop BioReplicate/Condition"
    );
    let value = |protein: &str, canonical: &str, sample: &str| -> Result<f64, Box<dyn Error>> {
        let row = table
            .rows
            .iter()
            .find(|row| {
                row.first().map(String::as_str) == Some(protein)
                    && row.get(1).map(String::as_str) == Some(canonical)
                    && row.get(2).map(String::as_str) == Some(sample)
            })
            .ok_or_else(|| -> Box<dyn Error> {
                format!("missing row ({protein}, {canonical}, {sample})").into()
            })?;
        let raw = row
            .get(3)
            .ok_or_else(|| -> Box<dyn Error> { "missing NormIntensity column".into() })?;
        raw.parse::<f64>()
            .map_err(|_| -> Box<dyn Error> { format!("'{raw}' is not numeric").into() })
    };

    let apc_s1 = value("P1", "APEPTIDECK", "sample-1")?;
    let apc_s2 = value("P1", "APEPTIDECK", "sample-2")?;
    let pep_s1 = value("P1", "PEPTIDEAK", "sample-1")?;
    let pep_s2 = value("P1", "PEPTIDEAK", "sample-2")?;

    // Quantile equalizes the two samples (ranks agree).
    assert!(
        (apc_s1 - apc_s2).abs() <= 1e-6,
        "APEPTIDECK not equalized: {apc_s1} vs {apc_s2}"
    );
    assert!(
        (pep_s1 - pep_s2).abs() <= 1e-6,
        "PEPTIDEAK not equalized: {pep_s1} vs {pep_s2}"
    );
    // Rank preserved, and the values differ from the raw sums (200 / 150),
    // proving the dataset-level normalization reached the export.
    assert!(apc_s1 > pep_s1, "rank not preserved: {apc_s1} !> {pep_s1}");
    assert!(
        (apc_s1 - 200.0).abs() > 1e-6,
        "export looks un-normalized (still the raw sum 200): {apc_s1}"
    );
    Ok(())
}

// ===========================================================================
// features2peptides (standalone command) golden tests vs the Python oracle.
//
// The synthetic feature set mirrors `scratchpad/gen_features.py`, which is the
// exact input fed to:
//   conda run -n Bigbio python -m mokume.mokume_cli features2peptides \
//       -p feat.parquet --skip_normalization -o pep.csv
// (run WITHOUT --sdrf; see the divergence note below). The peptide matrix:
//   P1/APEPTIDECK/run1 = 200, P1/PEPTIDEAK/run1 = 100 (z2) + 150 (z3) = 250,
//   P1/APEPTIDECK/run2 = 400, P1/PEPTIDEAK/run2 = 300.
//   P2/ONEPEPTK is dropped: a single canonical peptide fails min_unique=2.
//
// Divergence (flagged): Python's `peptide_normalization` computes its sample
// list from the *pre-SDRF-enrichment* view, so when an SDRF join would remap
// `sample_accession`, the per-sample filter matches nothing and the output is
// empty -- a latent Python quirk. The faithful, non-empty deterministic path
// is therefore run *without* --sdrf, where both engines key on run_file_name.
// On that path every column matches cell-for-cell, `Condition` included:
// Python's QPX view defaults it to run_file_name and the Rust peptide export
// now applies the same fallback (`sample_condition`, formerly the literal
// "Empty"). Verified on real PXD003539 (no/SDRF-miss): Condition mismatches
// drop from 100% to 0 while NormIntensity/BioReplicate/row set stay aligned.
fn synthetic_peptide_rows() -> Vec<QpxRow<'static>> {
    vec![
        QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
        QpxRow::new_with_charge("PEPTIDEAK", "run1", 150.0, 3, &["P1"]),
        QpxRow::new_with_charge("APEPTIDECK", "run1", 200.0, 2, &["P1"]),
        QpxRow::new_with_charge("PEPTIDEAK", "run2", 300.0, 2, &["P1"]),
        QpxRow::new_with_charge("APEPTIDECK", "run2", 400.0, 2, &["P1"]),
        QpxRow::new_with_charge("ONEPEPTK", "run1", 999.0, 2, &["P2"]),
    ]
}

fn default_peptides_config(parquet: PathBuf, output: PathBuf) -> FeatureToPeptidesConfig {
    FeatureToPeptidesConfig {
        input: InputConfig {
            parquet: Some(parquet),
            msstats: None,
            psm: None,
            sdrf: None,
            fasta: None,
        },
        output,
        filtering: FilterConfig {
            min_aa: 7,
            min_unique_peptides: 2,
            remove_contaminants: false,
        },
        remove_ids: None,
        remove_low_frequency_peptides: false,
        keep_shared_peptides: false,
        skip_normalization: true,
        run_normalization: "median".to_owned(),
        sample_normalization: "globalmedian".to_owned(),
        log2: false,
        save_parquet: false,
        aggregation_level: AggregationLevel::Sample,
        filter_pipeline: None,
        irs: None,
    }
}

#[test]
fn features2peptides_disabled_filter_pipeline_is_inert() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.disabled-filter.features.parquet");
    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;

    let baseline_output = root.join("peptides.disabled-filter.baseline.csv");
    let baseline = default_peptides_config(parquet.clone(), baseline_output.clone());
    run_features_to_peptides(&baseline)?;

    let disabled_output = root.join("peptides.disabled-filter.disabled.csv");
    let mut disabled = default_peptides_config(parquet, disabled_output.clone());
    disabled.filter_pipeline = Some(PreprocessingFilterConfig {
        enabled: false,
        intensity: IntensityFilterConfig {
            min_intensity: 10_000.0,
            cv_threshold: Some(0.0),
            min_replicate_agreement: 2,
            quantile_lower: 0.49,
            quantile_upper: 0.51,
            remove_zero_intensity: true,
        },
        peptide: PeptideFilterConfig {
            allowed_charge_states: Some(vec![9]),
            max_missed_cleavages: Some(0),
            min_peptide_length: 30,
            max_peptide_length: 31,
            exclude_sequence_patterns: vec!["PEPTIDE".to_owned()],
            ..PeptideFilterConfig::default()
        },
        protein: ProteinFilterConfig {
            min_unique_peptides: 99,
            razor_peptide_handling: "remove".to_owned(),
            contaminant_patterns: vec!["P1".to_owned()],
            ..ProteinFilterConfig::default()
        },
        run_qc: RunQcFilterConfig {
            min_total_intensity: 1_000_000.0,
            min_identified_features: 99,
            min_identified_proteins: 99,
            max_missing_rate: 0.0,
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&disabled)?;

    let baseline_table = read_csv(&baseline_output)?;
    let disabled_table = read_csv(&disabled_output)?;
    assert_eq!(disabled_table.headers, baseline_table.headers);
    assert_eq!(
        disabled_table.rows, baseline_table.rows,
        "enabled=false must make every nested preprocessing setting inert"
    );
    Ok(())
}

/// Look up the `NormIntensity` cell of the peptide long table for a given
/// (ProteinName, PeptideCanonical, SampleID), returning the parsed value.
fn peptide_norm_intensity(
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

fn assert_peptide_cell(
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
fn assert_peptide_condition(
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

// Oracle (captured from the command above; --skip_normalization, no --sdrf):
//   ProteinName,PeptideCanonical,SampleID,BioReplicate,Condition,NormIntensity
//   P1,APEPTIDECK,run2,1,run2,400.0
//   P1,PEPTIDEAK,run2,1,run2,300.0
//   P1,APEPTIDECK,run1,1,run1,200.0
//   P1,PEPTIDEAK,run1,1,run1,250.0
#[test]
fn features2peptides_skip_normalization_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.skip.features.parquet");
    let output = root.join("peptides.skip.csv");

    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;
    let config = default_peptides_config(parquet, output.clone());
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(
        table.headers,
        vec![
            "ProteinName",
            "PeptideCanonical",
            "SampleID",
            "BioReplicate",
            "Condition",
            "NormIntensity",
        ]
    );
    // The matrix has exactly the four P1 cells; P2 is filtered by min_unique=2.
    assert_eq!(table.rows.len(), 4, "unexpected rows:\n{:#?}", table.rows);
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 200.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 250.0)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run2", 400.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run2", 300.0)?;
    // Condition matches the Python oracle cell-for-cell: with no SDRF the QPX
    // view defaults it to run_file_name (= SampleID), not the literal "Empty".
    assert_peptide_condition(&table, "P1", "APEPTIDECK", "run1", "run1")?;
    assert_peptide_condition(&table, "P1", "PEPTIDEAK", "run1", "run1")?;
    assert_peptide_condition(&table, "P1", "APEPTIDECK", "run2", "run2")?;
    assert_peptide_condition(&table, "P1", "PEPTIDEAK", "run2", "run2")?;
    assert!(
        table
            .rows
            .iter()
            .all(|row| row.first().is_none_or(|value| value != "P2")),
        "single-unique-peptide protein P2 must be filtered"
    );
    Ok(())
}

// Channel IRS (TMT, global scope, --skip_normalization, no --sdrf). Verified
// cell-for-cell against the Python CLI:
//   mokume features2peptides -p tmt.parquet --keep-shared-peptides
//     --skip_normalization --irs_channel TMT126 --irs_scope global
//     --irs_stat {median,mean} -o py.csv
// Runs M1_1 / M2_2 / M3_3 carry reference channel TMT126 (irs_value 200/100/400)
// and sample channel TMT127 (the surviving peptidoform per cell, picked by max).
//   median: global_center = median(200,100,400) = 200 ->
//           scale = {techrep1: 1.0, techrep2: 2.0, techrep3: 0.5}
//   mean:   global_center = mean(200,100,400)   = 233.33... ->
//           scale = {techrep1: 1.1667, techrep2: 2.3333, techrep3: 0.5833}
// The reference channel TMT126 also survives as its own (lower-intensity) cell,
// scaled by the same per-techreplicate factor.
fn run_tmt_irs_global(stat: IrsStat) -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("tmt.irs.features.parquet");
    let output = root.join("tmt.irs.csv");

    write_qpx_tmt_rows(
        &parquet,
        &[
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "M1_1",
                channels: &[("TMT126", 150.0), ("TMT127", 1000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "M1_1",
                channels: &[("TMT126", 250.0), ("TMT127", 2000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "M2_2",
                channels: &[("TMT126", 80.0), ("TMT127", 3000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "M2_2",
                channels: &[("TMT126", 120.0), ("TMT127", 4000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "M3_3",
                channels: &[("TMT126", 350.0), ("TMT127", 5000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "M3_3",
                channels: &[("TMT126", 450.0), ("TMT127", 6000.0)],
                accessions: &["P1"],
            },
        ],
    )?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.keep_shared_peptides = true;
    config.skip_normalization = true;
    config.irs = Some(IrsChannelConfig {
        channel: "TMT126".to_owned(),
        stat,
        scope: IrsScope::Global,
    });
    run_features_to_peptides(&config)?;
    read_csv(&output)
}

#[test]
fn features2peptides_tmt_irs_global_median_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_tmt_irs_global(IrsStat::Median)?;
    // Sample-channel TMT127 cells (the surviving peptidoform), scaled by techrep:
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "M1_1", 2000.0)?; // 2000 * 1.0
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "M1_1", 1000.0)?; // 1000 * 1.0
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "M2_2", 8000.0)?; // 4000 * 2.0
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "M2_2", 6000.0)?; // 3000 * 2.0
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "M3_3", 3000.0)?; // 6000 * 0.5
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "M3_3", 2500.0)?; // 5000 * 0.5
    Ok(())
}

#[test]
fn features2peptides_tmt_irs_global_mean_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_tmt_irs_global(IrsStat::Mean)?;
    let center = (200.0 + 100.0 + 400.0) / 3.0;
    let s1 = center / 200.0;
    let s2 = center / 100.0;
    let s3 = center / 400.0;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "M1_1", 2000.0 * s1)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "M1_1", 1000.0 * s1)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "M2_2", 4000.0 * s2)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "M2_2", 3000.0 * s2)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "M3_3", 6000.0 * s3)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "M3_3", 5000.0 * s3)?;
    Ok(())
}

// Channel IRS, non-global scopes (`by_mixture` / `two_stage`), median stat,
// --skip_normalization, no --sdrf. Verified cell-for-cell against the Python
// `peptide_normalization` path on the matching pyarrow fixture (scratchpad
// oracle `run_e2e.py`). Four runs across two mixtures, two techreplicates each;
// the run name is `<mixture>_<techrep>` so collect-side (`split_part(_,2)`) and
// apply-side (last token) techreplicate agree:
//   mixA_1: TMT126 {100,300} -> irs_value median 200 ; sample TMT127 = 1000
//   mixA_2: TMT126 {100,200} -> irs_value median 150 ; sample TMT127 = 2000
//   mixB_3: TMT126 {400,600} -> irs_value median 500 ; sample TMT127 = 3000
//   mixB_4: TMT126 { 80,160} -> irs_value median 120 ; sample TMT127 = 4000
// Each surviving (peptide, run) cell is the sample channel (max over channels),
// scaled by the per-techreplicate factor below.
fn run_tmt_irs_multimix(scope: IrsScope) -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("tmt.irs.multimix.features.parquet");
    let output = root.join("tmt.irs.multimix.csv");

    write_qpx_tmt_rows(
        &parquet,
        &[
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "mixA_1",
                channels: &[("TMT126", 100.0), ("TMT127", 1000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "mixA_1",
                channels: &[("TMT126", 300.0), ("TMT127", 1000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "mixA_2",
                channels: &[("TMT126", 100.0), ("TMT127", 2000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "mixA_2",
                channels: &[("TMT126", 200.0), ("TMT127", 2000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "mixB_3",
                channels: &[("TMT126", 400.0), ("TMT127", 3000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "mixB_3",
                channels: &[("TMT126", 600.0), ("TMT127", 3000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "mixB_4",
                channels: &[("TMT126", 80.0), ("TMT127", 4000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "mixB_4",
                channels: &[("TMT126", 160.0), ("TMT127", 4000.0)],
                accessions: &["P1"],
            },
        ],
    )?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.keep_shared_peptides = true;
    config.skip_normalization = true;
    config.irs = Some(IrsChannelConfig {
        channel: "TMT126".to_owned(),
        stat: IrsStat::Median,
        scope,
    });
    run_features_to_peptides(&config)?;
    read_csv(&output)
}

#[test]
fn features2peptides_tmt_irs_by_mixture_median_matches_python_oracle() -> Result<(), Box<dyn Error>>
{
    let table = run_tmt_irs_multimix(IrsScope::ByMixture)?;
    // mixA center = median(200,150) = 175; mixB center = median(500,120) = 310.
    // scale = {1: 175/200, 2: 175/150, 3: 310/500, 4: 310/120}.
    for peptide in ["PEPTIDEAAK", "APEPTIDECK"] {
        assert_peptide_cell(&table, "P1", peptide, "mixA_1", 1000.0 * (175.0 / 200.0))?;
        assert_peptide_cell(&table, "P1", peptide, "mixA_2", 2000.0 * (175.0 / 150.0))?;
        assert_peptide_cell(&table, "P1", peptide, "mixB_3", 3000.0 * (310.0 / 500.0))?;
        assert_peptide_cell(&table, "P1", peptide, "mixB_4", 4000.0 * (310.0 / 120.0))?;
    }
    Ok(())
}

#[test]
fn features2peptides_tmt_irs_two_stage_median_matches_python_oracle() -> Result<(), Box<dyn Error>>
{
    let table = run_tmt_irs_multimix(IrsScope::TwoStage)?;
    // Stage-1 centers mixA 175 / mixB 310; global center over the distinct
    // mixture centers = median(175,310) = 242.5. scale = stage1 * stage2.
    let global = 242.5;
    let s1 = (175.0 / 200.0) * (global / 175.0);
    let s2 = (175.0 / 150.0) * (global / 175.0);
    let s3 = (310.0 / 500.0) * (global / 310.0);
    let s4 = (310.0 / 120.0) * (global / 310.0);
    for peptide in ["PEPTIDEAAK", "APEPTIDECK"] {
        assert_peptide_cell(&table, "P1", peptide, "mixA_1", 1000.0 * s1)?;
        assert_peptide_cell(&table, "P1", peptide, "mixA_2", 2000.0 * s2)?;
        assert_peptide_cell(&table, "P1", peptide, "mixB_3", 3000.0 * s3)?;
        assert_peptide_cell(&table, "P1", peptide, "mixB_4", 4000.0 * s4)?;
    }
    Ok(())
}

// Channel IRS by_mixture WITH --sdrf: the mixture key is the first `_`-token of
// the SDRF `source name` (Python `split_part(sdrf_sample_accession, '_', 1)`),
// NOT the run name. All four runs share the run-name prefix `g`, so without an
// SDRF they would collapse into a single mixture (by_mixture == global); the
// SDRF source names regroup them into POOLX = {g_1, g_2} and POOLY = {g_3, g_4}.
// Locked against the Python function-level oracle (`feature.enrich_with_sdrf`
// then `get_irs_scaling_factors`, scratchpad `sdrf_mixture_fn.py`), which yields
// the same per-mixture centers (POOLX 175, POOLY 310) as the no-SDRF fixture and
// thus the same scale {1: 0.875, 2: 1.1667, 3: 0.62, 4: 2.5833}. The output
// SampleID is the SDRF source name; the techreplicate still comes from the run
// name (`g_1` -> 1, ...).
#[test]
fn features2peptides_tmt_irs_by_mixture_uses_sdrf_source_name_mixture() -> Result<(), Box<dyn Error>>
{
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("tmt.irs.sdrfmix.features.parquet");
    let sdrf = root.join("tmt.irs.sdrfmix.sdrf.tsv");
    let output = root.join("tmt.irs.sdrfmix.csv");

    write_qpx_tmt_rows(
        &parquet,
        &[
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "g_1",
                channels: &[("TMT126", 100.0), ("TMT127", 1000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "g_1",
                channels: &[("TMT126", 300.0), ("TMT127", 1000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "g_2",
                channels: &[("TMT126", 100.0), ("TMT127", 2000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "g_2",
                channels: &[("TMT126", 200.0), ("TMT127", 2000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "g_3",
                channels: &[("TMT126", 400.0), ("TMT127", 3000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "g_3",
                channels: &[("TMT126", 600.0), ("TMT127", 3000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: "g_4",
                channels: &[("TMT126", 80.0), ("TMT127", 4000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "APEPTIDECK",
                run_file_name: "g_4",
                channels: &[("TMT126", 160.0), ("TMT127", 4000.0)],
                accessions: &["P1"],
            },
        ],
    )?;

    // SDRF regroups runs into mixtures POOLX (g_1,g_2) and POOLY (g_3,g_4) via the
    // `source name` first token. Each run/channel pair has its own exact SDRF row.
    std::fs::write(
        &sdrf,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "POOLX_a_ref\trun 1 ref\tg_1\tTMT126\tA\n",
            "POOLX_a\trun 1 sample\tg_1\tTMT127\tA\n",
            "POOLX_b_ref\trun 2 ref\tg_2\tTMT126\tA\n",
            "POOLX_b\trun 2 sample\tg_2\tTMT127\tA\n",
            "POOLY_a_ref\trun 3 ref\tg_3\tTMT126\tB\n",
            "POOLY_a\trun 3 sample\tg_3\tTMT127\tB\n",
            "POOLY_b_ref\trun 4 ref\tg_4\tTMT126\tB\n",
            "POOLY_b\trun 4 sample\tg_4\tTMT127\tB\n",
        ),
    )?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.input.sdrf = Some(sdrf);
    config.keep_shared_peptides = true;
    config.skip_normalization = true;
    config.irs = Some(IrsChannelConfig {
        channel: "TMT126".to_owned(),
        stat: IrsStat::Median,
        scope: IrsScope::ByMixture,
    });
    run_features_to_peptides(&config)?;
    let table = read_csv(&output)?;

    // POOLX center = median(200,150) = 175; POOLY center = median(500,120) = 310.
    // The SampleID is the SDRF source name; cell = sample TMT127 * mixture scale.
    for peptide in ["PEPTIDEAAK", "APEPTIDECK"] {
        assert_peptide_cell(&table, "P1", peptide, "POOLX_a", 1000.0 * (175.0 / 200.0))?;
        assert_peptide_cell(&table, "P1", peptide, "POOLX_b", 2000.0 * (175.0 / 150.0))?;
        assert_peptide_cell(&table, "P1", peptide, "POOLY_a", 3000.0 * (310.0 / 500.0))?;
        assert_peptide_cell(&table, "P1", peptide, "POOLY_b", 4000.0 * (310.0 / 120.0))?;
    }
    Ok(())
}

// Opt-in filter pipeline (`--filter-charge-states 2 --filter-max-missed-cleavages 0`,
// --skip_normalization, no --sdrf). The per-row filters run during ingest, before
// the per-(protein, sample) unique-peptide gate, matching Python's chain order:
//   - charge filter [2] drops PEPTIDEAK z3 (150) -> P1/PEPTIDEAK/run1 = 100 (not 250),
//     and drops SHORTBBK z9 -> P3 keeps only SHORTAAK (1 canonical).
//   - missed-cleavage filter (max 0) drops PEPTKIDEAK (1 miss) -> P4 keeps only CLEANAK.
//   - P2 (ONEPEPTK), P3, P4 each fall below min_unique=2 after row filtering and are
//     dropped -- proving the filters apply before the unique gate. Only P1 survives.
#[test]
fn features2peptides_filter_pipeline_applies_per_row_before_unique_gate(
) -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.filter.features.parquet");
    let output = root.join("peptides.filter.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 150.0, 3, &["P1"]),
            QpxRow::new_with_charge("APEPTIDECK", "run1", 200.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2", 300.0, 2, &["P1"]),
            QpxRow::new_with_charge("APEPTIDECK", "run2", 400.0, 2, &["P1"]),
            QpxRow::new_with_charge("ONEPEPTK", "run1", 999.0, 2, &["P2"]),
            QpxRow::new_with_charge("SHORTAAK", "run1", 500.0, 2, &["P3"]),
            QpxRow::new_with_charge("SHORTBBK", "run1", 600.0, 9, &["P3"]),
            QpxRow::new_with_charge("PEPTKIDEAK", "run1", 700.0, 2, &["P4"]),
            QpxRow::new_with_charge("CLEANAK", "run1", 800.0, 2, &["P4"]),
        ],
    )?;

    let filter = PreprocessingFilterConfig {
        peptide: PeptideFilterConfig {
            allowed_charge_states: Some(vec![2]),
            max_missed_cleavages: Some(0),
            ..PeptideFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    };
    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(filter);
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    // Only P1 survives; charge filter cut PEPTIDEAK z3 so run1 = 100, not 250.
    assert_eq!(table.rows.len(), 4, "unexpected rows:\n{:#?}", table.rows);
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 200.0)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run2", 400.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 100.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run2", 300.0)?;
    for dropped in ["P2", "P3", "P4"] {
        assert!(
            table
                .rows
                .iter()
                .all(|row| row.first().is_none_or(|value| value != dropped)),
            "{dropped} must be dropped: it falls below min_unique=2 after row filtering"
        );
    }
    Ok(())
}

// `exclude_sequence_patterns` (Python `SequencePatternFilter`, peptide.py:415-418)
// drops canonical sequences matching any user regex during ingest, before the
// per-protein unique gate, and does NOT touch the normalization median. This
// differential test excludes `^BPEPTIDE`: BPEPTIDEDK is dropped while the two
// surviving canonicals keep P1 above min_unique=2. Real-data parity vs Python is
// verified on PXD003539 through a `--filter-config` exclude pattern.
#[test]
fn features2peptides_exclude_sequence_patterns_drops_matching_canonicals(
) -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.exclude.features.parquet");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2", 300.0, 2, &["P1"]),
            QpxRow::new_with_charge("APEPTIDECK", "run1", 200.0, 2, &["P1"]),
            QpxRow::new_with_charge("APEPTIDECK", "run2", 400.0, 2, &["P1"]),
            QpxRow::new_with_charge("BPEPTIDEDK", "run1", 250.0, 2, &["P1"]),
            QpxRow::new_with_charge("BPEPTIDEDK", "run2", 350.0, 2, &["P1"]),
        ],
    )?;

    // Baseline: no exclusion, all three canonicals survive (3 x 2 samples).
    let base_out = root.join("exclude.base.csv");
    let mut base = default_peptides_config(parquet.clone(), base_out.clone());
    base.filter_pipeline = Some(PreprocessingFilterConfig::default());
    run_features_to_peptides(&base)?;
    assert_eq!(
        read_csv(&base_out)?.rows.len(),
        6,
        "baseline keeps 3 canonicals x 2 samples"
    );

    // Excluding `^BPEPTIDE` drops BPEPTIDEDK; P1 still has 2 canonicals.
    let out = root.join("exclude.set.csv");
    let mut config = default_peptides_config(parquet, out.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        peptide: PeptideFilterConfig {
            exclude_sequence_patterns: vec!["^BPEPTIDE".to_string()],
            ..PeptideFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&out)?;
    assert_eq!(
        table.rows.len(),
        4,
        "exclusion drops BPEPTIDEDK:\n{:#?}",
        table.rows
    );
    assert!(
        table
            .rows
            .iter()
            .all(|row| row.get(1).is_none_or(|value| value != "BPEPTIDEDK")),
        "BPEPTIDEDK must be excluded:\n{:#?}",
        table.rows
    );
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 200.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run2", 300.0)?;
    Ok(())
}

// QuantileFilter (Python `intensity.py:293-297`) drops per-sample rows whose raw
// intensity is outside the `[lower, upper]` linear-quantile bounds (double-closed).
// The bounds are computed over the `min_unique`-gated set (here P1's five canonicals
// span 100..500): numpy linear quantile [0.2, 0.8] -> bounds [180, 420], so 100 and
// 500 are dropped and 200/300/400 are kept. Real-data bit-exact row-set parity
// (only_py=0/only_rs=0) is verified on PXD003539 (quantile [0.05, 0.95]).
#[test]
fn features2peptides_quantile_filter_drops_out_of_bound_rows() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.quantile.features.parquet");
    let output = root.join("peptides.quantile.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEBK", "run1", 200.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDECK", "run1", 300.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEDK", "run1", 400.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEEK", "run1", 500.0, 2, &["P1"]),
        ],
    )?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        intensity: IntensityFilterConfig {
            quantile_lower: 0.2,
            quantile_upper: 0.8,
            ..IntensityFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(
        table.rows.len(),
        3,
        "quantile [0.2,0.8] keeps the middle three:\n{:#?}",
        table.rows
    );
    for kept in ["PEPTIDEBK", "PEPTIDECK", "PEPTIDEDK"] {
        assert!(
            table
                .rows
                .iter()
                .any(|row| row.get(1).is_some_and(|v| v == kept)),
            "{kept} (inside bounds) must be kept"
        );
    }
    for dropped in ["PEPTIDEAK", "PEPTIDEEK"] {
        assert!(
            table
                .rows
                .iter()
                .all(|row| row.get(1).is_none_or(|v| v != dropped)),
            "{dropped} (outside bounds) must be dropped:\n{:#?}",
            table.rows
        );
    }
    assert_peptide_cell(&table, "P1", "PEPTIDECK", "run1", 300.0)?;
    Ok(())
}

// CVThresholdFilter (Python `intensity.py:111-154`) drops every row of a
// `(sample, ProteinName, PeptideCanonical)` group whose within-sample coefficient
// of variation (pandas std ddof=1 / mean) exceeds the threshold. A group with a
// single measurement (or non-positive mean) has CV `NaN`, which Python keeps
// (intensity.py:136), so it always survives.
//
// Fixture (one sample, run1, P1):
//   PEPTIDEAK: 3 measurements 100/120/140 -> mean 120, std 20 -> CV 0.16667 (low)
//   PEPTIDEBK: 2 measurements 10/990      -> CV 1.38593 (high)
//   PEPTIDECK: 1 measurement 300          -> CV NaN -> kept
// At cv_threshold=0.2 PEPTIDEBK is dropped; PEPTIDEAK (CV 0.167 <= 0.2) and
// PEPTIDECK (NaN) survive. The surviving rows are summed per canonical
// (`sum_peptidoform_intensities`): PEPTIDEAK = 100+120+140 = 360, PEPTIDECK = 300.
//
// Cross-language parity (validated out-of-band against the `mokume` CLI on the
// byte-identical parquet: `features2peptides --skip_normalization --filter-config
// <cv>.yaml`, Python logs "CVThresholdFilter removed 2 items"): Python and Rust
// emit exactly these two rows, identical on ProteinName/PeptideCanonical/SampleID/
// NormIntensity; only the `Condition` column differs (the documented no-SDRF
// default "run1" vs "Empty"), which the cell helpers ignore.
#[test]
fn features2peptides_cv_filter_drops_high_cv_canonical() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.cv.features.parquet");
    let output = root.join("peptides.cv.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 120.0, 3, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 140.0, 4, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEBK", "run1", 10.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEBK", "run1", 990.0, 3, &["P1"]),
            QpxRow::new_with_charge("PEPTIDECK", "run1", 300.0, 2, &["P1"]),
        ],
    )?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        intensity: IntensityFilterConfig {
            cv_threshold: Some(0.2),
            ..IntensityFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(
        table.rows.len(),
        2,
        "CV 0.2 keeps the low-CV and single-measurement canonicals:\n{:#?}",
        table.rows
    );
    assert!(
        table
            .rows
            .iter()
            .all(|row| row.get(1).is_none_or(|value| value != "PEPTIDEBK")),
        "PEPTIDEBK (CV 1.386 > 0.2) must be dropped:\n{:#?}",
        table.rows
    );
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 360.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDECK", "run1", 300.0)?;
    Ok(())
}

// Boundary guard for the float32 CV decision: Python computes the CV on the
// float32 `NormIntensity` column, so the Rust pre-pass rounds its f64 CV to f32
// before comparing to the threshold. This fixture's BOUNDARYAK group has a
// float32 CV of 0.19999966 -- ~3.4e-7 *below* the 0.2 threshold -- so it must be
// KEPT. The f32 rounding's value is that a 30k-group fuzz against pandas showed
// zero decision flips, of which near-boundary groups like this one are the
// tightest. Cross-checked against the `mokume` CLI (Python keeps BOUNDARYAK;
// "CVThresholdFilter removed 0 items").
// The summed BOUNDARYAK intensity matches Python to f32 tolerance (the standard
// f32-input/f64-sum behaviour shared by every peptide cell).
#[test]
fn features2peptides_cv_filter_float32_boundary_keep() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.cvboundary.features.parquet");
    let output = root.join("peptides.cvboundary.csv");
    write_qpx_rows(
        &parquet,
        &[
            // float32 CV of this triple == 0.19999969 (< 0.2): kept.
            QpxRow::new_with_charge("BOUNDARYAK", "run1", 128.701_7, 2, &["P1"]),
            QpxRow::new_with_charge("BOUNDARYAK", "run1", 128.769_46, 3, &["P1"]),
            QpxRow::new_with_charge("BOUNDARYAK", "run1", 179.152_4, 4, &["P1"]),
            // Second low-CV canonical so P1 clears min_unique=2.
            QpxRow::new_with_charge("STABLEAAK", "run1", 500.0, 2, &["P1"]),
            QpxRow::new_with_charge("STABLEAAK", "run1", 505.0, 3, &["P1"]),
        ],
    )?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        intensity: IntensityFilterConfig {
            cv_threshold: Some(0.2),
            ..IntensityFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(
        table.rows.len(),
        2,
        "the boundary canonical (CV 0.19999969 < 0.2) must be kept:\n{:#?}",
        table.rows
    );
    assert!(
        table
            .rows
            .iter()
            .any(|row| row.get(1).is_some_and(|value| value == "BOUNDARYAK")),
        "BOUNDARYAK must survive the float32 CV comparison:\n{:#?}",
        table.rows
    );
    // Sum of the three float32 intensities; equals Python's 436.62354 to f32
    // tolerance (the cell helper allows 1e-9, so assert the value within a
    // float32-grade epsilon explicitly instead).
    let summed = peptide_norm_intensity(&table, "P1", "BOUNDARYAK", "run1")?;
    let expected = f64::from(128.701_7_f32) + f64::from(128.769_46_f32) + f64::from(179.152_4_f32);
    assert!(
        (summed - expected).abs() / expected.abs() <= 2.5e-7,
        "BOUNDARYAK sum {summed} must match {expected} to f32 tolerance"
    );
    Ok(())
}

// CV runs BEFORE QuantileFilter in Python's intensity block (factory.py wires
// MinIntensity -> CV -> ReplicateAgreement -> Quantile), so the quantile bounds
// are taken over the post-CV survivors. This guards that the Rust pre-pass shares
// the buffer between the two filters.
//
// Fixture (one sample, run1, P1):
//   PEPTIDEAK: 100/120 -> CV 0.1178 (kept at cv=0.5)
//   PEPTIDEBK: 10/5000 -> CV 1.40 (dropped at cv=0.5)
//   PEPTIDECK: 300 ; PEPTIDEDK: 400 (single, CV NaN -> kept)
// After CV the surviving rows are {100, 120, 300, 400}. numpy linear quantile
// [0.1, 0.9] over those four -> bounds [106, 370], so 100 (<106) and 400 (>370)
// are dropped. Survivors: PEPTIDEAK=120 (only the 120 row), PEPTIDECK=300.
//
// Cross-language parity (validated against the `mokume` CLI: "CVThresholdFilter
// removed 2 items" then "QuantileFilter removed 2 items"): Python and Rust emit
// exactly these two rows, cell-identical on the parity key.
#[test]
fn features2peptides_cv_then_quantile_share_buffer() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.cvquant.features.parquet");
    let output = root.join("peptides.cvquant.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 120.0, 3, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEBK", "run1", 10.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEBK", "run1", 5000.0, 3, &["P1"]),
            QpxRow::new_with_charge("PEPTIDECK", "run1", 300.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEDK", "run1", 400.0, 2, &["P1"]),
        ],
    )?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        intensity: IntensityFilterConfig {
            cv_threshold: Some(0.5),
            quantile_lower: 0.1,
            quantile_upper: 0.9,
            ..IntensityFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(
        table.rows.len(),
        2,
        "CV(0.5) then quantile [0.1,0.9] over post-CV survivors:\n{:#?}",
        table.rows
    );
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 120.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDECK", "run1", 300.0)?;
    Ok(())
}

// CV is computed before the per-row peptide filters (charge/length/modification/
// missed cleavage), which Python applies only after the intensity block. So a row
// the ChargeStateFilter would later drop still feeds the CV computation.
//
// Fixture (one sample, run1, P1): PEPTIDEAK has charge2(100) and charge9(5000) ->
// CV 1.371 over both rows. With cv_threshold=0.5 AND allowed_charge_states=[2],
// the CV pre-pass (which precedes the charge filter) sees both charges and drops
// PEPTIDEAK entirely. PEPTIDECK(300)/PEPTIDEDK(400) survive (single -> CV NaN,
// charge 2 allowed). If CV were computed after the charge filter it would see only
// the charge-2 row (CV NaN) and keep PEPTIDEAK -- so this row count distinguishes
// the orderings. Validated against the `mokume` CLI (Python keeps exactly
// PEPTIDECK/PEPTIDEDK).
#[test]
fn features2peptides_cv_precedes_charge_filter() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.cvcharge.features.parquet");
    let output = root.join("peptides.cvcharge.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 5000.0, 9, &["P1"]),
            QpxRow::new_with_charge("PEPTIDECK", "run1", 300.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEDK", "run1", 400.0, 2, &["P1"]),
        ],
    )?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        intensity: IntensityFilterConfig {
            cv_threshold: Some(0.5),
            ..IntensityFilterConfig::default()
        },
        peptide: PeptideFilterConfig {
            allowed_charge_states: Some(vec![2]),
            ..PeptideFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(
        table.rows.len(),
        2,
        "CV (seeing the charge-9 row) drops PEPTIDEAK before the charge filter:\n{:#?}",
        table.rows
    );
    assert!(
        table
            .rows
            .iter()
            .all(|row| row.get(1).is_none_or(|value| value != "PEPTIDEAK")),
        "PEPTIDEAK must be dropped by CV (which precedes the charge filter):\n{:#?}",
        table.rows
    );
    assert_peptide_cell(&table, "P1", "PEPTIDECK", "run1", 300.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEDK", "run1", 400.0)?;
    Ok(())
}

// ReplicateAgreementFilter (Python `intensity.py:194-242`) is DEGENERATE under
// mokume's per-sample pipeline application: it counts `nunique(SampleID)` within
// `dataset_df`, which holds exactly one sample (`peptide.py:266`), so the count is
// always 1. Any `min_replicate_agreement >= 2` therefore drops every row of every
// sample, leaving a header-only output. This is not an approximation -- it is
// Python's exact behaviour (verified against the `mokume` CLI: "ReplicateAgreement
// Filter removed N items (100.0%)", empty CSV body). Rust reproduces it by
// rejecting every feature at ingest and emits the same header-only table.
#[test]
fn features2peptides_replicate_agreement_empties_output() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.rep.features.parquet");
    let output = root.join("peptides.rep.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 120.0, 3, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEBK", "run1", 200.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDECK", "run1", 300.0, 2, &["P1"]),
        ],
    )?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        intensity: IntensityFilterConfig {
            min_replicate_agreement: 2,
            ..IntensityFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert!(
        table.rows.is_empty(),
        "min_replicate_agreement=2 must empty the output (per-sample nunique==1):\n{:#?}",
        table.rows
    );
    // The header is still written, matching Python's header-only CSV.
    assert_eq!(
        table.headers,
        vec![
            "ProteinName",
            "PeptideCanonical",
            "SampleID",
            "BioReplicate",
            "Condition",
            "NormIntensity"
        ]
    );
    Ok(())
}

// Under `--keep-shared-peptides`, Python skips the per-protein `min_unique` gate
// (peptide.py:278-281, guarded by `if not keep_shared_peptides`) but STILL runs the
// filter pipeline's `MinPeptideFilter` (peptide.py:291-293), which is unconditional.
// So a protein with fewer than `min_unique_peptides` distinct canonicals is still
// dropped. Here P2 has one canonical and must be removed even with keep-shared on;
// P1 (two canonicals) survives. This closes the ~88767-row keep-shared gap measured
// on PXD003539 (Rust previously zeroed the threshold and kept the singletons).
#[test]
fn features2peptides_keep_shared_still_applies_min_peptide_filter() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.keepshared.features.parquet");
    let output = root.join("peptides.keepshared.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("APEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("BPEPTIDEBK", "run1", 200.0, 2, &["P1"]),
            QpxRow::new_with_charge("CPEPTIDECK", "run1", 300.0, 2, &["P2"]),
        ],
    )?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.keep_shared_peptides = true;
    // Default protein.min_unique_peptides = 2, so MinPeptideFilter still runs.
    config.filter_pipeline = Some(PreprocessingFilterConfig::default());
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(
        table.rows.len(),
        2,
        "P2 (one canonical) must be dropped by MinPeptideFilter:\n{:#?}",
        table.rows
    );
    assert!(
        table
            .rows
            .iter()
            .all(|row| row.first().is_none_or(|v| v != "P2")),
        "P2 must be removed even with keep-shared on:\n{:#?}",
        table.rows
    );
    assert_peptide_cell(&table, "P1", "APEPTIDEAK", "run1", 100.0)?;
    assert_peptide_cell(&table, "P1", "BPEPTIDEBK", "run1", 200.0)?;
    Ok(())
}

// Under `--keep-shared-peptides`, the globalmedian sample-normalization median
// pre-pass must INCLUDE shared (non-unique) peptides, mirroring Python's
// `SQLFilterBuilder(require_unique = not keep_shared)` (`peptide.py:168/176`,
// consumed by `get_median_map` at `peptide.py:197`). Previously the Rust median
// collector derived `keep_shared = (quantification == Pibaq)`, but the peptide
// path pins quantification to `Sum` (`peptide_export_config`), so shared rows
// were always excluded -- biasing every per-sample factor and scaling every
// NormIntensity in the sample by a constant (~0.41% on PXD003539, max_rel
// 1.28e-2).
//
// Fixture: two samples (run1/run2), each with three P1 uniques, three P2
// uniques, and one shared peptide SHAREDPEPTIDEK ([P1, P2]) whose large
// intensity (5000/8000) shifts the per-sample median. The oracle values below
// come from the Python CLI on the byte-identical fixture:
//   mokume features2peptides -p fixture.parquet --keep-shared-peptides
//     --min_unique 2 -o py.csv
// They differ from the no-keep-shared run (e.g. P1/APEPTIDECK/run1 = 342.857
// here vs 346.341 when shared peptides are excluded from the median), so this
// test fails on the pre-fix binary.
#[test]
fn features2peptides_keep_shared_median_includes_shared_peptides() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.keepshared.median.features.parquet");
    let output = root.join("peptides.keepshared.median.csv");
    write_qpx_rows(
        &parquet,
        &[
            // run1: P1 uniques
            QpxRow::new("PEPTIDEAAK", "run1", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1", 200.0, &["P1"]),
            QpxRow::new("KPEPTIDEMK", "run1", 300.0, &["P1"]),
            // run1: P2 uniques
            QpxRow::new("DPEPTIDEFK", "run1", 110.0, &["P2"]),
            QpxRow::new("EPEPTIDEGK", "run1", 210.0, &["P2"]),
            QpxRow::new("FPEPTIDEHK", "run1", 310.0, &["P2"]),
            // run1: shared peptide moves the per-sample median
            QpxRow::new_shared("SHAREDPEPTIDEK", "run1", 5000.0, &["P1", "P2"]),
            // run2: P1 uniques
            QpxRow::new("PEPTIDEAAK", "run2", 400.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2", 500.0, &["P1"]),
            QpxRow::new("KPEPTIDEMK", "run2", 600.0, &["P1"]),
            // run2: P2 uniques
            QpxRow::new("DPEPTIDEFK", "run2", 410.0, &["P2"]),
            QpxRow::new("EPEPTIDEGK", "run2", 510.0, &["P2"]),
            QpxRow::new("FPEPTIDEHK", "run2", 610.0, &["P2"]),
            // run2: shared peptide
            QpxRow::new_shared("SHAREDPEPTIDEK", "run2", 8000.0, &["P1", "P2"]),
        ],
    )?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.keep_shared_peptides = true;
    // Exercise the globalmedian sample-normalization median pre-pass.
    config.skip_normalization = false;
    config.run_normalization = "median".to_owned();
    config.sample_normalization = "globalmedian".to_owned();
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    // Oracle (Python CLI, keep-shared): unique-peptide cells reflect the median
    // pre-pass that INCLUDES the shared peptide.
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "run1", 171.42857142857142)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 342.85714285714283)?;
    assert_peptide_cell(&table, "P1", "KPEPTIDEMK", "run1", 514.2857142857142)?;
    assert_peptide_cell(&table, "P2", "DPEPTIDEFK", "run1", 188.57142857142856)?;
    assert_peptide_cell(&table, "P2", "EPEPTIDEGK", "run1", 360.0)?;
    assert_peptide_cell(&table, "P2", "FPEPTIDEHK", "run1", 531.4285714285714)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "run2", 282.3529411764706)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run2", 352.94117647058823)?;
    assert_peptide_cell(&table, "P1", "KPEPTIDEMK", "run2", 423.5294117647059)?;
    assert_peptide_cell(&table, "P2", "DPEPTIDEFK", "run2", 289.4117647058824)?;
    assert_peptide_cell(&table, "P2", "EPEPTIDEGK", "run2", 360.0)?;
    assert_peptide_cell(&table, "P2", "FPEPTIDEHK", "run2", 430.5882352941177)?;
    // The shared peptide itself survives (kept) and carries the joined name.
    assert_peptide_cell(&table, "P1;P2", "SHAREDPEPTIDEK", "run1", 8571.42857142857)?;
    assert_peptide_cell(&table, "P1;P2", "SHAREDPEPTIDEK", "run2", 5647.058823529412)?;
    Ok(())
}

// RazorPeptideFilter `remove` (Python `preprocessing/filters/protein.py:348-357`).
// A razor peptide is a canonical that maps to more than one `ProteinName` within
// a sample. `ProteinName` is the joined `pg_accessions` (Python
// aggregation.py:154 `.str.join(";")`, Rust `protein_group_name`), so the same
// canonical observed in two single-protein rows ([P1] and [P2]) yields two
// distinct ProteinName values "P1"/"P2" in one sample -> nunique=2 -> razor.
//
// Fixture (one sample, run1, `--keep-shared-peptides` so the shared peptide is
// not dropped by the unique gate before razor runs):
//   P1 anchors ANCHORPONEK(200)/ANCHORPTWOK(210); P2 anchors ANCHORQONEK(300)/
//   ANCHORQTWOK(310); RAZORSHAREDK shared as [P1](500) + [P2](505).
// The razor block isolation (min_unique_peptides=0, no
// contaminant/decoy removal) keeps the anchors and exercises only razor.
//
// Cross-language parity (validated out-of-band against the `mokume` CLI on the
// byte-identical parquet: `features2peptides --keep-shared-peptides
// --skip_normalization --filter-config <razor>.yaml`):
//   * razor=remove -> Python and Rust both emit the 4 anchor rows, identical
//     cell-for-cell on ProteinName/PeptideCanonical/SampleID/NormIntensity, and
//     now also on `Condition` (both fall the no-SDRF default back to
//     run_file_name; Python logs "RazorPeptideFilter removed 2 items").
//   * razor=remove output equals razor=keep output minus exactly the razor rows;
//     no anchor row is touched (remove is a strict subset of keep).
// (razor=keep's row count differs across engines by the pre-existing keep-shared
// peptidoform-selection divergence, which is orthogonal to razor and not
// asserted here.)
fn razor_filter_pipeline(handling: &str) -> PreprocessingFilterConfig {
    PreprocessingFilterConfig {
        protein: ProteinFilterConfig {
            // Isolate razor: no unique/peptide-count gate, no contaminant or
            // decoy removal, so the only protein-level effect is razor handling.
            min_unique_peptides: 0,
            remove_contaminants: false,
            remove_decoys: false,
            razor_peptide_handling: handling.to_owned(),
            ..ProteinFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    }
}

#[test]
fn features2peptides_razor_remove_drops_only_the_razor_peptide() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.razor.features.parquet");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("ANCHORPONEK", "run1", 200.0, &["P1"]),
            QpxRow::new("ANCHORPTWOK", "run1", 210.0, &["P1"]),
            QpxRow::new("ANCHORQONEK", "run1", 300.0, &["P2"]),
            QpxRow::new("ANCHORQTWOK", "run1", 310.0, &["P2"]),
            // RAZORSHAREDK maps to P1 in one row and P2 in another -> razor.
            QpxRow::new("RAZORSHAREDK", "run1", 500.0, &["P1"]),
            QpxRow::new("RAZORSHAREDK", "run1", 505.0, &["P2"]),
        ],
    )?;

    // razor=keep: the razor peptide survives under both proteins (no-op filter).
    let keep_out = root.join("razor.keep.csv");
    let mut keep = default_peptides_config(parquet.clone(), keep_out.clone());
    keep.keep_shared_peptides = true;
    keep.filter_pipeline = Some(razor_filter_pipeline("keep"));
    run_features_to_peptides(&keep)?;
    let keep_table = read_csv(&keep_out)?;
    assert_eq!(
        keep_table.rows.len(),
        6,
        "razor=keep keeps 4 anchors + 2 razor cells:\n{:#?}",
        keep_table.rows
    );
    // Both razor cells present under razor=keep (non-vacuity vs remove).
    assert_peptide_cell(&keep_table, "P1", "RAZORSHAREDK", "run1", 500.0)?;
    assert_peptide_cell(&keep_table, "P2", "RAZORSHAREDK", "run1", 505.0)?;

    // razor=remove: the razor peptide is dropped from BOTH proteins.
    let remove_out = root.join("razor.remove.csv");
    let mut remove = default_peptides_config(parquet, remove_out.clone());
    remove.keep_shared_peptides = true;
    remove.filter_pipeline = Some(razor_filter_pipeline("remove"));
    run_features_to_peptides(&remove)?;
    let remove_table = read_csv(&remove_out)?;
    assert_eq!(
        remove_table.rows.len(),
        4,
        "razor=remove drops both RAZORSHAREDK cells, leaving 4 anchors:\n{:#?}",
        remove_table.rows
    );
    assert!(
        remove_table
            .rows
            .iter()
            .all(|row| row.get(1).is_none_or(|value| value != "RAZORSHAREDK")),
        "RAZORSHAREDK must be removed from every protein:\n{:#?}",
        remove_table.rows
    );
    // The four anchor cells are untouched by razor removal (cell-for-cell).
    assert_peptide_cell(&remove_table, "P1", "ANCHORPONEK", "run1", 200.0)?;
    assert_peptide_cell(&remove_table, "P1", "ANCHORPTWOK", "run1", 210.0)?;
    assert_peptide_cell(&remove_table, "P2", "ANCHORQONEK", "run1", 300.0)?;
    assert_peptide_cell(&remove_table, "P2", "ANCHORQTWOK", "run1", 310.0)?;

    // razor=remove is exactly razor=keep minus the razor peptide: every remove
    // row appears verbatim in keep, and the only keep rows absent from remove are
    // the RAZORSHAREDK cells.
    assert!(
        remove_table
            .rows
            .iter()
            .all(|row| keep_table.rows.contains(row)),
        "every razor=remove row must appear in razor=keep (anchors untouched)"
    );
    let only_in_keep = keep_table
        .rows
        .iter()
        .filter(|row| !remove_table.rows.contains(row))
        .collect::<Vec<_>>();
    assert!(
        only_in_keep
            .iter()
            .all(|row| row.get(1).is_some_and(|value| value == "RAZORSHAREDK")),
        "the only keep-minus-remove rows must be the razor peptide:\n{only_in_keep:#?}"
    );
    Ok(())
}

// RazorPeptideFilter `assign_to_top` (Python `protein.py:358-378`). For a razor
// canonical (mapping to >1 ProteinName in the sample), keep only the rows of the
// protein with the most unique canonicals in the sample. Ties are broken by
// first-appearance order: pandas `group[ProteinName].unique()` preserves
// first-occurrence order and `max(..., key=...)` keeps the first maximum, so the
// protein observed earliest in parquet row order wins. The Rust side records
// that order in `PeptideExport.push` (called in parquet row order) because the
// materialization `HashMap` cannot recover it.
//
// This golden test reuses the two cross-language fixtures (validated out-of-band
// against the `mokume` CLI on byte-identical parquet, `features2peptides
// --keep-shared-peptides --skip_normalization --filter-config <assign>.yaml`,
// `razor_peptide_handling: assign_to_top`, all other gates disabled):
//   * Fixture A: SHAREDXYK shared by PX/PY (each 2 unique anchors -> tie). The
//     [PY] row precedes the [PX] row -> PY wins. Python and Rust both emit
//     `PY,SHAREDXYK`.
//   * Fixture B (row order flipped): the [PX] row precedes the [PY] row, and PX's
//     SHAREDXYK intensity is HIGHER than PY's (999 vs 500) to rule out intensity
//     -> PX wins purely on order. The winner flips to `PX,SHAREDXYK` in both
//     engines.
//   * Non-tie control SHAREDABK shared by PA (4 unique anchors) / PB (2 unique).
//     PA wins on count regardless of order, even though PB's SHAREDABK row has
//     the HIGHER intensity (800 vs 400) -> PA always keeps the row.
// Each fixture removes exactly the 2 losing razor rows (Python logs
// "RazorPeptideFilter removed 2 items"); every emitted column matches across
// engines, `Condition` included (both default the no-SDRF value to
// run_file_name).

/// Tie razor `SHAREDXYK` plus non-tie control `SHAREDABK`, parameterised by which
/// of PX/PY appears first so the same builder produces fixtures A and B. `py_first`
/// true = fixture A (PY before PX -> PY wins); false = fixture B (PX before PY,
/// PX intensity higher -> PX wins).
fn assign_to_top_rows(py_first: bool) -> Vec<QpxRow<'static>> {
    // PX/PY each carry 2 unique anchor canonicals -> SHAREDXYK is a true tie.
    let anchors = [
        QpxRow::new("PXANCHORONEK", "run1", 100.0, &["PX"]),
        QpxRow::new("PXANCHORTWOK", "run1", 110.0, &["PX"]),
        QpxRow::new("PYANCHORONEK", "run1", 120.0, &["PY"]),
        QpxRow::new("PYANCHORTWOK", "run1", 130.0, &["PY"]),
        // PA has 4 unique anchors, PB only 2 -> SHAREDABK is a non-tie (PA wins).
        QpxRow::new("PAANCHORONEK", "run1", 200.0, &["PA"]),
        QpxRow::new("PAANCHORTWOK", "run1", 210.0, &["PA"]),
        QpxRow::new("PAANCHORTREK", "run1", 220.0, &["PA"]),
        QpxRow::new("PAANCHORFORK", "run1", 230.0, &["PA"]),
        QpxRow::new("PBANCHORONEK", "run1", 300.0, &["PB"]),
        QpxRow::new("PBANCHORTWOK", "run1", 310.0, &["PB"]),
    ];
    // The tie razor rows, ordered so the winner is decided purely by row order.
    let shared_xy = if py_first {
        [
            QpxRow::new("SHAREDXYK", "run1", 500.0, &["PY"]),
            QpxRow::new("SHAREDXYK", "run1", 505.0, &["PX"]),
        ]
    } else {
        // PX first AND higher intensity (999) -> still PX, proving order beats
        // intensity.
        [
            QpxRow::new("SHAREDXYK", "run1", 999.0, &["PX"]),
            QpxRow::new("SHAREDXYK", "run1", 500.0, &["PY"]),
        ]
    };
    // Non-tie razor: PB has the HIGHER intensity (800) but fewer unique peptides,
    // so PA (4 unique) wins on count.
    let shared_ab = [
        QpxRow::new("SHAREDABK", "run1", 400.0, &["PA"]),
        QpxRow::new("SHAREDABK", "run1", 800.0, &["PB"]),
    ];
    anchors
        .into_iter()
        .chain(shared_xy)
        .chain(shared_ab)
        .collect()
}

/// Run `assign_to_top` over the parameterised fixture and return the long table.
fn run_assign_to_top(root: &Path, tag: &str, py_first: bool) -> Result<CsvTable, Box<dyn Error>> {
    let parquet = root.join(format!("assign.{tag}.features.parquet"));
    write_qpx_rows(&parquet, &assign_to_top_rows(py_first))?;
    let out = root.join(format!("assign.{tag}.csv"));
    let mut config = default_peptides_config(parquet, out.clone());
    config.keep_shared_peptides = true;
    config.filter_pipeline = Some(razor_filter_pipeline("assign_to_top"));
    run_features_to_peptides(&config)?;
    read_csv(&out)
}

/// Look up which `ProteinName` keeps a given razor canonical, asserting exactly
/// one survives (assign_to_top collapses a razor peptide to a single protein).
fn razor_winner(table: &CsvTable, peptide: &str) -> Result<String, Box<dyn Error>> {
    let protein_idx = table
        .headers
        .iter()
        .position(|header| header == "ProteinName")
        .ok_or_else(|| -> Box<dyn Error> { "ProteinName column missing".into() })?;
    let peptide_idx = table
        .headers
        .iter()
        .position(|header| header == "PeptideCanonical")
        .ok_or_else(|| -> Box<dyn Error> { "PeptideCanonical column missing".into() })?;
    let winners = table
        .rows
        .iter()
        .filter(|row| row.get(peptide_idx).is_some_and(|value| value == peptide))
        .filter_map(|row| row.get(protein_idx).cloned())
        .collect::<Vec<_>>();
    assert_eq!(
        winners.len(),
        1,
        "razor peptide {peptide} must survive under exactly one protein, got {winners:?}"
    );
    Ok(winners[0].clone())
}

#[test]
fn features2peptides_razor_assign_to_top_flips_winner_on_row_order() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;

    // Fixture A: PY row precedes PX -> the tied razor goes to PY.
    let table_a = run_assign_to_top(&root, "a", true)?;
    // Fixture B: PX row precedes PY (and is louder) -> the winner flips to PX,
    // locking the tie-break to first-appearance order rather than intensity.
    let table_b = run_assign_to_top(&root, "b", false)?;

    // 10 anchors + 1 surviving tie row + 1 surviving non-tie row = 12 rows each.
    assert_eq!(
        table_a.rows.len(),
        12,
        "fixture A keeps 10 anchors + 2 razor winners:\n{:#?}",
        table_a.rows
    );
    assert_eq!(
        table_b.rows.len(),
        12,
        "fixture B keeps 10 anchors + 2 razor winners:\n{:#?}",
        table_b.rows
    );

    // Core assertion: the tie peptide's winner flips with row order.
    assert_eq!(
        razor_winner(&table_a, "SHAREDXYK")?,
        "PY",
        "fixture A: PY appears first, so PY must win the tie"
    );
    assert_eq!(
        razor_winner(&table_b, "SHAREDXYK")?,
        "PX",
        "fixture B: PX appears first, so the winner must flip to PX"
    );

    // Non-tie control: PA (4 unique) beats PB (2 unique) in BOTH fixtures, even
    // though PB's razor row carries the higher intensity -> count, not strength.
    assert_eq!(razor_winner(&table_a, "SHAREDABK")?, "PA");
    assert_eq!(razor_winner(&table_b, "SHAREDABK")?, "PA");

    // The winner keeps its own intensity, cell-for-cell (mirrors the Python CLI).
    assert_peptide_cell(&table_a, "PY", "SHAREDXYK", "run1", 500.0)?;
    assert_peptide_cell(&table_b, "PX", "SHAREDXYK", "run1", 999.0)?;
    assert_peptide_cell(&table_a, "PA", "SHAREDABK", "run1", 400.0)?;
    assert_peptide_cell(&table_b, "PA", "SHAREDABK", "run1", 400.0)?;

    // The losing proteins keep their own (non-razor) anchors untouched: PX still
    // has its anchors in fixture A, PY in fixture B, PB in both.
    assert_peptide_cell(&table_a, "PX", "PXANCHORONEK", "run1", 100.0)?;
    assert_peptide_cell(&table_b, "PY", "PYANCHORONEK", "run1", 120.0)?;
    assert_peptide_cell(&table_a, "PB", "PBANCHORONEK", "run1", 300.0)?;

    Ok(())
}

// The synthetic QPX lacks the requested score and any coverage column. Reject
// both instead of accepting filters that cannot affect output.
#[test]
fn features2peptides_rejects_unavailable_search_score_and_coverage_filters(
) -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.unsupported.features.parquet");
    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;

    let mut search_score = default_peptides_config(parquet.clone(), root.join("search-score.csv"));
    search_score.filter_pipeline = Some(PreprocessingFilterConfig {
        peptide: PeptideFilterConfig {
            score: Some(NamedScoreFilterConfig {
                name: "missing_score".to_owned(),
                threshold: 0.9,
            }),
            ..PeptideFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    assert!(matches!(
        run_features_to_peptides(&search_score),
        Err(MokumeError::InvalidInput { message })
            if message.contains("missing_score")
    ));

    let mut coverage = default_peptides_config(parquet, root.join("coverage.csv"));
    coverage.filter_pipeline = Some(PreprocessingFilterConfig {
        protein: ProteinFilterConfig {
            min_coverage: 0.5,
            ..ProteinFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    assert!(matches!(
        run_features_to_peptides(&coverage),
        Err(MokumeError::NotImplemented {
            stage: "features2peptides filter min-coverage"
        })
    ));
    Ok(())
}

#[test]
fn features2peptides_applies_named_score_direction() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;

    for (case, higher_better) in [("higher", true), ("lower", false)] {
        let mut rows = synthetic_peptide_rows();
        for (index, row) in rows.iter_mut().enumerate() {
            let passes = index != 1;
            let value = if passes == higher_better { 0.9 } else { 0.1 };
            *row = row.with_score("engine_score", value, higher_better);
        }
        let parquet = root.join(format!("score-{case}.features.parquet"));
        let output = root.join(format!("score-{case}.csv"));
        write_qpx_rows(&parquet, &rows)?;

        let mut config = default_peptides_config(parquet, output.clone());
        config.filter_pipeline = Some(PreprocessingFilterConfig {
            peptide: PeptideFilterConfig {
                score: Some(NamedScoreFilterConfig {
                    name: "engine_score".to_owned(),
                    threshold: 0.5,
                }),
                ..PeptideFilterConfig::default()
            },
            ..PreprocessingFilterConfig::default()
        });
        run_features_to_peptides(&config)?;

        let table = read_csv(&output)?;
        assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 100.0)?;
    }
    Ok(())
}

#[test]
fn features2peptides_applies_explicit_qpx_fdr_filters() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.fdr.features.parquet");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1", 100.0, &["P1"]).with_qvalues(Some(0.005), Some(0.02)),
            QpxRow::new("APEPTIDECK", "run1", 200.0, &["P1"]).with_qvalues(Some(0.02), Some(0.02)),
            QpxRow::new("PEPTIDEBK", "run1", 300.0, &["P2"]).with_qvalues(Some(0.005), Some(0.02)),
            QpxRow::new("BPEPTIDECK", "run1", 400.0, &["P2"])
                .with_qvalues(Some(0.005), Some(0.005)),
        ],
    )?;

    let peptide_output = root.join("peptides.peptide-fdr.csv");
    let mut peptide = default_peptides_config(parquet.clone(), peptide_output.clone());
    peptide.filter_pipeline = Some(PreprocessingFilterConfig {
        peptide: PeptideFilterConfig {
            fdr_threshold: Some(0.01),
            ..PeptideFilterConfig::default()
        },
        protein: ProteinFilterConfig {
            min_unique_peptides: 1,
            ..ProteinFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&peptide)?;
    let peptide_table = read_csv(&peptide_output)?;
    assert!(peptide_table
        .rows
        .iter()
        .all(|row| row.get(1).is_none_or(|sequence| sequence != "APEPTIDECK")));

    let protein_output = root.join("peptides.protein-fdr.csv");
    let mut protein = default_peptides_config(parquet, protein_output.clone());
    protein.filter_pipeline = Some(PreprocessingFilterConfig {
        protein: ProteinFilterConfig {
            fdr_threshold: Some(0.01),
            min_unique_peptides: 1,
            ..ProteinFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&protein)?;
    let protein_table = read_csv(&protein_output)?;
    assert!(protein_table
        .rows
        .iter()
        .all(|row| row.first().is_none_or(|value| value == "P2")));
    assert_eq!(protein_table.rows.len(), 2);
    Ok(())
}

#[test]
fn features2peptides_fdr_request_rejects_unpopulated_qvalue() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.missing-fdr.features.parquet");
    let output = root.join("peptides.missing-fdr.csv");
    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        peptide: PeptideFilterConfig {
            fdr_threshold: Some(0.01),
            ..PeptideFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });

    let error = match run_features_to_peptides(&config) {
        Ok(()) => return Err("unpopulated peptide q-value was accepted".into()),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("requires a populated QPX `peptide_qvalue` column"));
    assert!(!output.exists());
    Ok(())
}

// Run-QC MinFeaturesFilter counts distinct `(protein, canonical)` features per
// technical run after the per-sample protein min_unique gate. Duplicate charge
// states must not inflate the count: run1 has three canonicals, while run2 has
// two, so min_features=3 keeps only run1.
#[test]
fn features2peptides_run_qc_drops_low_feature_samples() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.runqc.features.parquet");
    let output = root.join("peptides.runqc.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 150.0, 3, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1", 200.0, &["P1"]),
            QpxRow::new("THIDPEAK", "run1", 250.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run2", 300.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2", 400.0, &["P1"]),
        ],
    )?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        run_qc: RunQcFilterConfig {
            min_identified_features: 3,
            ..RunQcFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    let sample_of = |row: &Vec<String>| row.get(2).cloned();
    assert!(
        table
            .rows
            .iter()
            .all(|row| sample_of(row).as_deref() != Some("run2")),
        "run2 (2 features < 3) must be dropped by Run-QC:\n{:#?}",
        table.rows
    );
    assert!(
        table
            .rows
            .iter()
            .any(|row| sample_of(row).as_deref() == Some("run1")),
        "run1 (3 features >= 3) must survive Run-QC:\n{:#?}",
        table.rows
    );
    Ok(())
}

// MissingRateFilter uses the complete `(protein, canonical)` universe of the
// surviving technical runs within a sample. run_a detects A/B/C; run_b detects
// only A, so its missing rate is 2/3 and a 0.4 cutoff removes run_b only. The
// shared sample remains, and A's sample-level intensity proves the rejected run
// did not leak into aggregation (10 rather than 10 + 9).
#[test]
fn features2peptides_missing_rate_drops_incomplete_technical_run() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.missing-rate.features.parquet");
    let sdrf = root.join("peptides.missing-rate.sdrf.tsv");
    let output = root.join("peptides.missing-rate.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run_a.raw", 10.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run_a.raw", 11.0, &["P1"]),
            QpxRow::new("THIDPEAK", "run_a.raw", 12.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run_b.raw", 9.0, &["P1"]),
        ],
    )?;
    write_run_replicate_sdrf(&sdrf)?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.input.sdrf = Some(sdrf);
    config.filtering.min_unique_peptides = 1;
    config.aggregation_level = AggregationLevel::Run;
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        run_qc: RunQcFilterConfig {
            max_missing_rate: 0.4,
            ..RunQcFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(table.rows.len(), 3, "unexpected filtered table: {table:#?}");
    assert_eq!(
        peptide_norm_intensity(&table, "P1", "PEPTIDEAK", "sample-run")?,
        10.0,
        "run_b must be excluded before peptide aggregation"
    );
    let run = header_index(&table.headers, "Run")?;
    let technical_replicate = header_index(&table.headers, "TechReplicate")?;
    assert!(table.rows.iter().all(|row| row[run] == "run_a.raw"));
    assert!(table.rows.iter().all(|row| row[technical_replicate] == "1"));
    Ok(())
}

// Default and custom contaminant patterns (`ProteinFilterConfig`).
#[test]
fn features2peptides_default_patterns_drop_contam_prefix() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.contam.features.parquet");
    write_qpx_rows(
        &parquet,
        &[
            // Genuine protein P1: two distinct canonicals -> passes min_unique=2.
            QpxRow::new("PEPTIDEAK", "run1", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1", 200.0, &["P1"]),
            // `CONTAM_` protein: two distinct canonicals so it survives the
            // min_unique gate and is only removed by the contaminant filter.
            QpxRow::new("CPEPTIDEAK", "run1", 300.0, &["CONTAM_P00001"]),
            QpxRow::new("DPEPTIDECK", "run1", 400.0, &["CONTAM_P00001"]),
        ],
    )?;

    // The default patterns remove the common DIA-NN `CONTAM_` prefix.
    let base_out = root.join("contam.base.csv");
    let mut base = default_peptides_config(parquet.clone(), base_out.clone());
    base.filter_pipeline = Some(PreprocessingFilterConfig::default());
    run_features_to_peptides(&base)?;
    let base_table = read_csv(&base_out)?;
    assert_eq!(
        base_table.rows.len(),
        2,
        "default patterns must drop CONTAM_ protein:\n{:#?}",
        base_table.rows
    );

    // An explicit custom list still replaces the defaults. The historical
    // three-pattern list therefore retains `CONTAM_` while preserving P1.
    let out = root.join("contam.custom.csv");
    let mut config = default_peptides_config(parquet, out.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig {
        protein: ProteinFilterConfig {
            contaminant_patterns: vec![
                "CONTAMINANT".to_owned(),
                "ENTRAP".to_owned(),
                "DECOY".to_owned(),
            ],
            ..ProteinFilterConfig::default()
        },
        ..PreprocessingFilterConfig::default()
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&out)?;
    assert_eq!(
        table.rows.len(),
        4,
        "explicit custom patterns must replace defaults:\n{:#?}",
        table.rows
    );
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 100.0)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 200.0)?;
    Ok(())
}

#[test]
fn features2peptides_keeps_human_protein_with_contaminant_twin() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.twin.features.parquet");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("EPEPTIDEAK", "run1", 100.0, &["CONTAM_P05787", "P05787"]),
            QpxRow::new("FPEPTIDECK", "run1", 200.0, &["P05787", "CONTAM_P05787"]),
            QpxRow::new("CPEPTIDEAK", "run1", 300.0, &["CONTAM_P00761"]),
            QpxRow::new("DPEPTIDECK", "run1", 400.0, &["CONTAM_P00761"]),
        ],
    )?;
    let out = root.join("twin.csv");
    let mut config = default_peptides_config(parquet, out.clone());
    config.filter_pipeline = Some(PreprocessingFilterConfig::default());
    run_features_to_peptides(&config)?;
    let table = read_csv(&out)?;
    assert_eq!(
        table.rows.len(),
        2,
        "twin kept, trypsin dropped:\n{:#?}",
        table.rows
    );
    assert_peptide_cell(&table, "P05787", "EPEPTIDEAK", "run1", 100.0)?;
    assert_peptide_cell(&table, "P05787", "FPEPTIDECK", "run1", 200.0)?;
    Ok(())
}

// Oracle (--skip_normalization --log2, no --sdrf): NormIntensity = log2(sum).
//   P1,APEPTIDECK,run2 -> log2(400) = 8.643856...
//   P1,PEPTIDEAK,run2  -> log2(300) = 8.228818...
//   P1,APEPTIDECK,run1 -> log2(200) = 7.643856...
//   P1,PEPTIDEAK,run1  -> log2(250) = 7.965784...
#[test]
fn features2peptides_log2_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.log2.features.parquet");
    let output = root.join("peptides.log2.csv");

    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.log2 = true;
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 200.0_f64.log2())?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 250.0_f64.log2())?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run2", 400.0_f64.log2())?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run2", 300.0_f64.log2())?;
    Ok(())
}

// `--keep-shared-peptides` (peptide.py:268-281). On the synth fixture P2 has a
// single canonical peptide (ONEPEPTK) that fails the default min_unique=2 gate,
// so the baseline drops it. With keep-shared, Python skips the `unique == 1`
// filter AND the per-protein min_unique gate, so the P2/ONEPEPTK/run1 cell is
// kept (999.0). Python oracle (`keep_shared_peptides=True --skip_normalization`,
// no sdrf) -> 5 rows: the four P1 cells plus P2,ONEPEPTK,run1,1,run1,999.0. The
// extra P2 row is the non-vacuity proof: the baseline (4 rows) lacks it.
#[test]
fn features2peptides_keep_shared_peptides_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.keep.features.parquet");
    let output = root.join("peptides.keep.csv");
    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.keep_shared_peptides = true;
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(table.rows.len(), 5, "unexpected rows:\n{:#?}", table.rows);
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 200.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 250.0)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run2", 400.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run2", 300.0)?;
    assert_peptide_cell(&table, "P2", "ONEPEPTK", "run1", 999.0)?;
    Ok(())
}

// `--save_parquet` (WriteParquetTask). The parquet sibling (extension swapped to
// `.parquet`) must equal the CSV matrix value-for-value AND match Python's
// schema: ProteinName/PeptideCanonical = Utf8, SampleID/Condition =
// Dictionary(Int8, Utf8), BioReplicate = Int32, NormIntensity = Float32.
#[test]
fn features2peptides_save_parquet_matches_csv_and_python_schema() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.sp.features.parquet");
    let output = root.join("peptides.sp.csv");
    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.save_parquet = true;
    run_features_to_peptides(&config)?;

    let csv = read_csv(&output)?;
    let pq_path = output.with_extension("parquet");
    let pq = read_peptide_parquet(&pq_path)?;

    // Schema (names, order, dtypes) matches Python's WriteParquetTask exactly.
    assert_eq!(
        pq.column_names,
        vec![
            "ProteinName".to_owned(),
            "PeptideCanonical".to_owned(),
            "SampleID".to_owned(),
            "BioReplicate".to_owned(),
            "Condition".to_owned(),
            "NormIntensity".to_owned(),
        ]
    );
    assert_eq!(pq.dtypes[0], "Utf8");
    assert_eq!(pq.dtypes[1], "Utf8");
    assert_eq!(pq.dtypes[2], "Dictionary(Int8, Utf8)");
    assert_eq!(pq.dtypes[3], "Int32");
    assert_eq!(pq.dtypes[4], "Dictionary(Int8, Utf8)");
    assert_eq!(pq.dtypes[5], "Float32");

    // Parquet values equal the CSV matrix cell-for-cell.
    assert_eq!(
        pq.rows.len(),
        csv.rows.len(),
        "parquet/csv row count diverged"
    );
    for (protein, peptide, sample, expected) in [
        ("P1", "APEPTIDECK", "run1", 200.0_f64),
        ("P1", "PEPTIDEAK", "run1", 250.0),
        ("P1", "APEPTIDECK", "run2", 400.0),
        ("P1", "PEPTIDEAK", "run2", 300.0),
    ] {
        let from_csv = peptide_norm_intensity(&csv, protein, peptide, sample)?;
        assert!((from_csv - expected).abs() <= 1e-9, "csv cell mismatch");
        let from_pq = parquet_norm_intensity(&pq, protein, peptide, sample)?;
        assert!(
            (from_pq - from_csv).abs() <= 1e-9,
            "({protein},{peptide},{sample}) parquet {from_pq} != csv {from_csv}"
        );
    }
    Ok(())
}

// `--aggregation_level run` (sum_peptidoform_intensities run branch). On the
// no-SDRF synth fixture each sample maps 1:1 to a run, so the NormIntensity
// matrix is identical to the sample-level oracle (200/250/400/300) but the CSV
// gains `Run` and `TechReplicate` columns (TechReplicate = 1: "run1"/"run2" are
// neither underscore-suffixed integers nor plain integers, so Python's
// per-sample run-index fallback yields 1 for a single-run sample). Python oracle
// (`aggregation_level="run" --skip_normalization`, no sdrf) confirms the
// 7-column header and Run = run_file_name. The extra columns are the non-vacuity
// proof against the sample-level (6-column) path.
#[test]
fn features2peptides_aggregation_level_run_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.aggrun.features.parquet");
    let output = root.join("peptides.aggrun.csv");
    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.aggregation_level = AggregationLevel::Run;
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(
        table.headers,
        vec![
            "ProteinName",
            "PeptideCanonical",
            "SampleID",
            "BioReplicate",
            "Condition",
            "Run",
            "TechReplicate",
            "NormIntensity",
        ]
    );
    assert_eq!(table.rows.len(), 4, "unexpected rows:\n{:#?}", table.rows);
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 200.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 250.0)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run2", 400.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run2", 300.0)?;
    // Run == run_file_name and TechReplicate == 1 for every row.
    let run_idx = header_index(&table.headers, "Run")?;
    let tech_idx = header_index(&table.headers, "TechReplicate")?;
    let sample_idx = header_index(&table.headers, "SampleID")?;
    for row in &table.rows {
        assert_eq!(row[run_idx], row[sample_idx], "Run must equal SampleID");
        assert_eq!(row[tech_idx], "1", "TechReplicate must be 1");
    }
    Ok(())
}

// `--aggregation_level run --save_parquet`: the run-mode parquet appends `Run`
// (Utf8) and `TechReplicate` (Int64) before `NormIntensity`, matching Python's
// run-level WriteParquetTask schema, and its values equal the run-level CSV.
#[test]
fn features2peptides_aggregation_run_parquet_matches_python_schema() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.aggrunpq.features.parquet");
    let output = root.join("peptides.aggrunpq.csv");
    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.aggregation_level = AggregationLevel::Run;
    config.save_parquet = true;
    run_features_to_peptides(&config)?;

    let pq = read_peptide_parquet(&output.with_extension("parquet"))?;
    assert_eq!(
        pq.column_names,
        vec![
            "ProteinName".to_owned(),
            "PeptideCanonical".to_owned(),
            "SampleID".to_owned(),
            "BioReplicate".to_owned(),
            "Condition".to_owned(),
            "Run".to_owned(),
            "TechReplicate".to_owned(),
            "NormIntensity".to_owned(),
        ]
    );
    assert_eq!(pq.dtypes[5], "Utf8");
    assert_eq!(pq.dtypes[6], "Int64");
    assert_eq!(pq.dtypes[7], "Float32");
    assert_eq!(pq.rows.len(), 4);
    let value = parquet_norm_intensity(&pq, "P1", "APEPTIDECK", "run2")?;
    assert!((value - 400.0).abs() <= 1e-9);
    Ok(())
}

// Dataset-level sample-normalization methods are not implemented by the
// standalone peptide command. Reject them instead of silently returning the
// same matrix as `--sample-normalization none`.
#[test]
fn features2peptides_rejects_dataset_level_sample_normalization() -> Result<(), Box<dyn Error>> {
    for method in [
        "quantile",
        "mediancenter",
        "meancenter",
        "rlr",
        "loess",
        "hierarchical",
    ] {
        let root = temp_root()?;
        create_dir_all(&root)?;
        let parquet = root.join(format!("peptides.noop.{method}.features.parquet"));
        let output = root.join(format!("peptides.unsupported.{method}.csv"));
        write_qpx_rows(&parquet, &synthetic_peptide_rows())?;

        let mut config = default_peptides_config(parquet, output);
        config.skip_normalization = false;
        config.sample_normalization = method.to_owned();
        assert!(
            matches!(
                run_features_to_peptides(&config),
                Err(MokumeError::NotImplemented {
                    stage: "features2peptides sample-normalization-method"
                })
            ),
            "{method} must be rejected instead of ignored"
        );
    }
    Ok(())
}

// Stage 2 (peptide normalization). Same fixture and no-SDRF path as
// `features2peptides_skip_normalization_matches_python_oracle`, but with the
// default run=median / sample=globalMedian normalization. The run/sample factor
// pipeline that backs `features2proteins` is reused; this proves it also drives
// `features2peptides`. Python oracle (run=median, sample=globalMedian, no sdrf):
//   P1,APEPTIDECK,run2 -> 300.0   P1,PEPTIDEAK,run2 -> 225.0
//   P1,APEPTIDECK,run1 -> 300.0   P1,PEPTIDEAK,run1 -> 375.0
// These differ from the un-normalized matrix (200/250/400/300), so the assert
// is non-vacuous: a no-op normalization would fail every cell.
#[test]
fn features2peptides_normalization_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.norm.features.parquet");
    let output = root.join("peptides.norm.csv");

    write_qpx_rows(&parquet, &synthetic_peptide_rows())?;
    let mut config = default_peptides_config(parquet, output.clone());
    config.skip_normalization = false;
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(table.rows.len(), 4, "unexpected rows:\n{:#?}", table.rows);
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 300.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 375.0)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run2", 300.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run2", 225.0)?;
    Ok(())
}

// Stage 1a (`--remove_ids`). Fixture: P1 (two canonical peptides, kept) and P9
// (two canonical peptides, both removed by the ID file listing "P9"). The IDs
// file has a trailing blank line, exercising the blank-line skip. Python oracle
// (`--remove_ids ids.txt --skip_normalization`, no sdrf):
//   P1,APEPTIDECK,run2 -> 400   P1,PEPTIDEAK,run2 -> 300
//   P1,APEPTIDECK,run1 -> 200   P1,PEPTIDEAK,run1 -> 250
// (no P9 rows; baseline without --remove_ids keeps two extra P9 rows).
#[test]
fn features2peptides_remove_ids_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.rid.features.parquet");
    let output = root.join("peptides.rid.csv");
    let ids = root.join("peptides.rid.ids.txt");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 150.0, 3, &["P1"]),
            QpxRow::new_with_charge("APEPTIDECK", "run1", 200.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2", 300.0, 2, &["P1"]),
            QpxRow::new_with_charge("APEPTIDECK", "run2", 400.0, 2, &["P1"]),
            QpxRow::new_with_charge("REMOVEMEK", "run1", 500.0, 2, &["P9"]),
            QpxRow::new_with_charge("OTHERREMK", "run1", 600.0, 2, &["P9"]),
        ],
    )?;
    std::fs::write(&ids, "P9\n\n")?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.remove_ids = Some(ids);
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(table.rows.len(), 4, "unexpected rows:\n{:#?}", table.rows);
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run1", 200.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 250.0)?;
    assert_peptide_cell(&table, "P1", "APEPTIDECK", "run2", 400.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run2", 300.0)?;
    assert!(
        table
            .rows
            .iter()
            .all(|row| row.first().is_none_or(|value| value != "P9")),
        "--remove_ids must drop every P9 row"
    );
    Ok(())
}

// Stage 1b (`--remove_low_frequency_peptides`). Fixture: P1 with three peptides
// across six runs. PEPTIDEAK and STABLE2PEPTK appear in all six runs (count 6);
// RAREPEPTK appears in only run1 (count 1). Threshold = 0.2 * 6 = 1.2, so
// RAREPEPTK (count 1 < 1.2) is removed while the others are kept. Python oracle
// (`--remove_low_frequency_peptides --skip_normalization`, no sdrf) drops the
// single RAREPEPTK/run1 cell (12 rows kept; baseline keeps 13). min_unique=2 is
// satisfied per run by PEPTIDEAK + STABLE2PEPTK.
#[test]
fn features2peptides_remove_low_frequency_peptides_matches_python_oracle(
) -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.lf.features.parquet");
    let output = root.join("peptides.lf.csv");

    let mut rows = Vec::new();
    for run in 1..=6 {
        let run_name: &'static str = match run {
            1 => "run1",
            2 => "run2",
            3 => "run3",
            4 => "run4",
            5 => "run5",
            _ => "run6",
        };
        rows.push(QpxRow::new_with_charge(
            "PEPTIDEAK",
            run_name,
            100.0 + run as f32,
            2,
            &["P1"],
        ));
        rows.push(QpxRow::new_with_charge(
            "STABLE2PEPTK",
            run_name,
            200.0 + run as f32,
            2,
            &["P1"],
        ));
    }
    rows.push(QpxRow::new_with_charge(
        "RAREPEPTK",
        "run1",
        999.0,
        2,
        &["P1"],
    ));
    write_qpx_rows(&parquet, &rows)?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.remove_low_frequency_peptides = true;
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_eq!(table.rows.len(), 12, "unexpected rows:\n{:#?}", table.rows);
    assert_peptide_cell(&table, "P1", "PEPTIDEAK", "run1", 101.0)?;
    assert_peptide_cell(&table, "P1", "STABLE2PEPTK", "run6", 206.0)?;
    assert!(
        table
            .rows
            .iter()
            .all(|row| row.get(1).is_none_or(|value| value != "RAREPEPTK")),
        "--remove_low_frequency_peptides must drop the single-sample RAREPEPTK"
    );
    Ok(())
}

#[test]
fn features2proteins_sums_charge_states_per_sample() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("charge_states.features.parquet");
    let sdrf = root.join("charge_states.sdrf.tsv");
    let output = root.join("charge_states.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAR", "run1.raw", 101.0, 2, &["P25"]),
            QpxRow::new_with_charge("PEPTIDEAR", "run1.raw", 302.0, 3, &["P25"]),
            QpxRow::new("PEPTIDEBR", "run1.raw", 203.0, &["P25"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let config = default_sum_config(parquet, sdrf, output.clone());
    run_features_to_proteins(&config)?;

    let table = read_csv(&output)?;
    assert_numeric_cell_close(&table, "P25", "sample-1", 606.0);
    Ok(())
}

#[test]
fn features2proteins_min_unique_counts_canonical_peptides() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("canonical_min_unique.features.parquet");
    let sdrf = root.join("canonical_min_unique.sdrf.tsv");
    let output = root.join("canonical_min_unique.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAR", "run1.raw", 101.0, 2, &["P26"]),
            QpxRow::new_with_charge("PEPTIDEAR", "run1.raw", 302.0, 3, &["P26"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let config = default_sum_config(parquet, sdrf, output.clone());
    run_features_to_proteins(&config)?;

    let table = read_csv(&output)?;
    assert!(
        table.rows.is_empty(),
        "one canonical peptide with multiple charge states must not satisfy min_unique=2"
    );
    Ok(())
}

#[test]
fn features2proteins_extra_quant_methods_match_synthetic_oracles() -> Result<(), Box<dyn Error>> {
    let topn = run_synthetic_quantification(QuantMethod::TopN, 2)?;
    assert_numeric_cell_close(&topn, "P1", "sample-1", 175.0);
    assert_numeric_cell_close(&topn, "P3", "sample-1", 800.0);
    assert_numeric_cell_close(&topn, "P4A;P4B", "sample-1", 15.0);

    let abd = run_synthetic_quantification(QuantMethod::Abd, 3)?;
    assert_numeric_cell_close(
        &abd,
        "P1",
        "sample-1",
        (150.0_f64.log2() + 200.0_f64.log2()) / 2.0,
    );
    assert_numeric_cell_close(
        &abd,
        "P3",
        "sample-1",
        (700.0_f64.log2() + 900.0_f64.log2()) / 2.0,
    );

    let intensity = run_synthetic_quantification(QuantMethod::Intensity, 3)?;
    assert_numeric_cell_close(&intensity, "P1", "sample-1", 350.0);
    assert_numeric_cell_close(&intensity, "P4A;P4B", "sample-1", 30.0);

    let peptide_count = run_synthetic_quantification(QuantMethod::PeptideCount, 3)?;
    assert_numeric_cell_close(&peptide_count, "P1", "sample-1", 2.0);
    assert_numeric_cell_close(&peptide_count, "P3", "sample-1", 2.0);
    assert_numeric_cell_close(&peptide_count, "P4A;P4B", "sample-1", 2.0);

    let pibaq = run_synthetic_quantification(QuantMethod::Pibaq, 3)?;
    assert_numeric_cell_close(&pibaq, "P1", "sample-1", 3450.0);
    assert_numeric_cell_close(&pibaq, "P1", "sample-2", 100.0);
    assert_numeric_cell_close(&pibaq, "P2", "sample-1", 500.0);
    assert_numeric_cell_close(&pibaq, "P3", "sample-1", 800.0);
    // P4A/P4B have no unique anchors, so their shared signal is split once.
    assert_numeric_cell_close(&pibaq, "P4A", "sample-1", 7.5);
    assert_numeric_cell_close(&pibaq, "P4B", "sample-1", 7.5);
    Ok(())
}

#[test]
fn features2proteins_collapses_charges_into_canonical_before_rollup() -> Result<(), Box<dyn Error>>
{
    // PEPTIDEK canonical = max(100, 80) [z2] + 40 [z3] = 140; ANOTHERK = 30.
    // Sum is unchanged by the collapse (associative); the other methods diverge
    // from a per-(peptidoform, charge) rollup and must match the Python oracle.
    let sum = run_canonical_collapse_quantification(QuantMethod::Sum, 3)?;
    assert_numeric_cell_close(&sum, "P50", "sample-1", 170.0);

    let median = run_canonical_collapse_quantification(QuantMethod::Median, 3)?;
    assert_numeric_cell_close(&median, "P50", "sample-1", 85.0);

    let top1 = run_canonical_collapse_quantification(QuantMethod::TopN, 1)?;
    assert_numeric_cell_close(&top1, "P50", "sample-1", 140.0);

    let top2 = run_canonical_collapse_quantification(QuantMethod::TopN, 2)?;
    assert_numeric_cell_close(&top2, "P50", "sample-1", 85.0);

    let abd = run_canonical_collapse_quantification(QuantMethod::Abd, 3)?;
    assert_numeric_cell_close(
        &abd,
        "P50",
        "sample-1",
        (140.0_f64.log2() + 30.0_f64.log2()) / 2.0,
    );

    let peptide_count = run_canonical_collapse_quantification(QuantMethod::PeptideCount, 3)?;
    assert_numeric_cell_close(&peptide_count, "P50", "sample-1", 2.0);

    let pibaq = run_canonical_collapse_quantification(QuantMethod::Pibaq, 3)?;
    assert_numeric_cell_close(&pibaq, "P50", "sample-1", 85.0);
    Ok(())
}

#[test]
fn features2proteins_ratio_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_ratio_quantification()?;

    assert_eq!(
        table.headers,
        vec!["ProteinName".to_owned(), "plexA_2".to_owned()]
    );
    assert_numeric_cell_close(&table, "P1", "plexA_2", 1.5);
    Ok(())
}

#[test]
fn features2proteins_ratio_sdrf_pooled_multiplex_matches_python_oracle(
) -> Result<(), Box<dyn Error>> {
    // SDRF-driven ratio quantification across two plexes (p1, p2). The reference
    // channel in each plex is marked by `characteristics[pooled sample] = pooled`
    // (Python's `detect_pooled_from_sdrf`), and the plex is derived from the
    // `source name` prefix (`p1_1` -> plex `p1`). Reference samples are excluded
    // from the output. Cell-exact Python oracle (1e-9):
    //   ProteinName  p1_2  p1_3  p2_2  p2_3
    //            P1   1.5   0.5   1.0   0.5
    let table = run_ratio_multiplex_quantification()?;

    assert_eq!(
        table.headers,
        vec![
            "ProteinName".to_owned(),
            "p1_2".to_owned(),
            "p1_3".to_owned(),
            "p2_2".to_owned(),
            "p2_3".to_owned(),
        ]
    );
    assert_numeric_cell_close(&table, "P1", "p1_2", 1.5);
    assert_numeric_cell_close(&table, "P1", "p1_3", 0.5);
    assert_numeric_cell_close(&table, "P1", "p2_2", 1.0);
    assert_numeric_cell_close(&table, "P1", "p2_3", 0.5);
    Ok(())
}

// Non-vacuity guard for the per-plex reference grouping. The pipeline groups the
// reference intensity by plex (`source name` prefix: p1_1 -> p1), so each plex's
// samples are normalised against THAT plex's pooled channel. This test pins the
// per-plex oracle and proves it is genuinely different from a GLOBAL-reference
// computation that pools p1_1 and p2_1 together (a single `plex1`). Both oracles
// come from `mokume.quantification.ratio.RatioQuantification` on the identical
// fixture, differing only in `sample_to_plex`:
//   per-plex {p1_*:p1, p2_*:p2}  -> P1: p1_2=1.5      p1_3=0.5      p2_2=1.0      p2_3=0.5
//   global   {all -> plex1}      -> P1: p1_2=-1.273447 p1_3=-2.273447 p2_2=1.887517 p2_3=1.387517
// If the per-plex grouping regressed to a single global reference, the per-plex
// asserts above (and the per-cell deltas below) would fail, so the parity test is
// not vacuously satisfied by any global-reference implementation.
#[test]
fn features2proteins_ratio_per_plex_differs_from_global_reference() -> Result<(), Box<dyn Error>> {
    let table = run_ratio_multiplex_quantification()?;

    // Per-plex oracle (the value the pipeline must produce), cell-exact.
    let per_plex = [
        ("p1_2", 1.5_f64),
        ("p1_3", 0.5_f64),
        ("p2_2", 1.0_f64),
        ("p2_3", 0.5_f64),
    ];
    // Global-reference oracle on the same fixture (single pooled reference).
    let global = [
        ("p1_2", -1.273_447_010_476_287_3_f64),
        ("p1_3", -2.273_447_010_476_287_f64),
        ("p2_2", 1.887_516_834_894_859_4_f64),
        ("p2_3", 1.387_516_834_894_859_4_f64),
    ];

    for ((sample, expected_per_plex), (_, global_value)) in per_plex.iter().zip(global.iter()) {
        assert_numeric_cell_close(&table, "P1", sample, *expected_per_plex);
        let delta = (expected_per_plex - global_value).abs();
        assert!(
            delta > 1e-3,
            "per-plex and global-reference oracles must differ at {sample}: per_plex={expected_per_plex}, global={global_value}, delta={delta}"
        );
    }
    Ok(())
}

#[test]
fn features2proteins_pibaq_allocates_family_shared_peptides() -> Result<(), Box<dyn Error>> {
    let table = run_family_pibaq_quantification()?;

    assert_numeric_cell_close(&table, "P5A", "sample-1", 250.0);
    assert_numeric_cell_close(&table, "P5B", "sample-1", 250.0 / 3.0);
    Ok(())
}

#[test]
fn features2proteins_lfq_methods_match_synthetic_oracles() -> Result<(), Box<dyn Error>> {
    // DirectLFQ oracle from the Mann-Labs `directlfq` package on the same 2-ion
    // x 2-sample P6 matrix: sample normalization removes the global 2x factor
    // between the samples, so both report the same intensity (total preserved at
    // 1000 = 500 + 500). The previous 200/400 oracle reflected the old native
    // approximation, not the real algorithm.
    let directlfq = run_lfq_quantification(QuantMethod::DirectLfq)?;
    assert_numeric_cell_close(&directlfq, "P6", "sample-1", 500.0);
    assert_numeric_cell_close(&directlfq, "P6", "sample-2", 500.0);

    // Python's `--quant-method maxlfq` delegates to DirectLFQ whenever the
    // directlfq package is installed (the default in the reference environment),
    // so its default output equals the directlfq output. Confirmed empirically:
    // Python maxlfq and directlfq both yield [500.0, 500.0] on this dataset, while
    // the built-in fallback (directlfq absent) yields [500.0, 1000.0]. The Rust
    // port routes maxlfq through the parity-verified DirectLFQ-aligned solver.
    let maxlfq = run_lfq_quantification(QuantMethod::MaxLfq)?;
    assert_numeric_cell_close(&maxlfq, "P6", "sample-1", 500.0);
    assert_numeric_cell_close(&maxlfq, "P6", "sample-2", 500.0);

    // The built-in MaxLFQ fallback (force_builtin) reproduces Python's not-installed
    // path: [500.0, 1000.0] on the same matrix.
    let maxlfq_builtin = run_lfq_quantification_builtin_maxlfq()?;
    assert_numeric_cell_close(&maxlfq_builtin, "P6", "sample-1", 500.0);
    assert_numeric_cell_close(&maxlfq_builtin, "P6", "sample-2", 1000.0);
    Ok(())
}

#[test]
fn features2proteins_maxlfq_maxes_contextual_ions_then_sums_canonical_peptides(
) -> Result<(), Box<dyn Error>> {
    // Python's loader first keeps the maximum intensity for each
    // (peptidoform, charge, sample, condition, biological-replicate) ion, then
    // sums those contextual ions into the canonical peptide. For sample-1 this
    // yields PEPTIDEK = max(100, 80) + 20 + 40 = 160 and ANOTHERK = 30;
    // sample-2 is exactly 2x. The delegated DirectLFQ solver normalizes that 2x
    // sample shift to [190, 190], while the forced built-in solver returns the
    // unnormalized canonical totals [190, 380].
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("maxlfq_dup.features.parquet");
    let sdrf = root.join("maxlfq_dup.sdrf.tsv");
    let delegated_output = root.join("maxlfq_dup.delegated.csv");
    let builtin_output = root.join("maxlfq_dup.builtin.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEK", "run1.raw", 100.0, 2, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEK", "run1.raw", 80.0, 2, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEK", "run1.raw", 40.0, 3, &["P6"])
                .with_peptidoform("PEP[UNIMOD:35]TIDEK"),
            QpxRow::new("ANOTHERK", "run1.raw", 30.0, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEK", "run2.raw", 20.0, 2, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEK", "run3.raw", 200.0, 2, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEK", "run3.raw", 160.0, 2, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEK", "run3.raw", 80.0, 3, &["P6"])
                .with_peptidoform("PEP[UNIMOD:35]TIDEK"),
            QpxRow::new("ANOTHERK", "run3.raw", 60.0, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEK", "run4.raw", 40.0, 2, &["P6"]),
        ],
    )?;
    std::fs::write(
        &sdrf,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcharacteristics[biological replicate]\tfactor value[cell line]\n",
            "sample-1\trun 1\trun1.raw\tAC=MS:1002038;NT=label free sample\t1\tA\n",
            "sample-1\trun 2\trun2.raw\tAC=MS:1002038;NT=label free sample\t2\tA\n",
            "sample-2\trun 3\trun3.raw\tAC=MS:1002038;NT=label free sample\t1\tB\n",
            "sample-2\trun 4\trun4.raw\tAC=MS:1002038;NT=label free sample\t2\tB\n",
        ),
    )?;

    let mut config = default_sum_config(parquet, sdrf, delegated_output.clone());
    config.quantification = QuantMethod::MaxLfq;
    run_features_to_proteins(&config)?;
    let delegated = read_csv(&delegated_output)?;
    assert_numeric_cell_close(&delegated, "P6", "sample-1", 190.0);
    assert_numeric_cell_close(&delegated, "P6", "sample-2", 190.0);

    config.output.protein_matrix = builtin_output.clone();
    config.maxlfq.force_builtin = true;
    run_features_to_proteins(&config)?;
    let builtin = read_csv(&builtin_output)?;
    assert_numeric_cell_close(&builtin, "P6", "sample-1", 190.0);
    assert_numeric_cell_close(&builtin, "P6", "sample-2", 380.0);
    Ok(())
}

#[test]
fn features2proteins_delegated_maxlfq_is_invariant_to_feature_order() -> Result<(), Box<dyn Error>>
{
    let root = temp_root()?;
    create_dir_all(&root)?;
    let forward_parquet = root.join("maxlfq_order.forward.features.parquet");
    let reversed_parquet = root.join("maxlfq_order.reversed.features.parquet");
    let sdrf = root.join("maxlfq_order.sdrf.tsv");
    let forward_output = root.join("maxlfq_order.forward.csv");
    let reversed_output = root.join("maxlfq_order.reversed.csv");

    // Minimal observations from PXD002099 that exposed encounter-order IDs in
    // the delegated DirectLFQ solver. The two parquets carry identical data in
    // opposite row order.
    let rows = vec![
        QpxRow::new("LRTDETLR", "run1.raw", 1_661_357.0, &["A5Z2X5"]),
        QpxRow::new("LRTDETLR", "run2.raw", 2_011_746.0, &["A5Z2X5"]),
        QpxRow::new("LRTDETLR", "run3.raw", 771_094.9, &["A5Z2X5"]),
        QpxRow::new("LTGNPELSSLDEVLAK", "run1.raw", 3_310_899.0, &["A5Z2X5"]),
        QpxRow::new("LTGNPELSSLDEVLAK", "run2.raw", 2_914_053.0, &["A5Z2X5"]),
        QpxRow::new("LTGNPELSSLDEVLAK", "run3.raw", 4_332_763.0, &["A5Z2X5"]),
        QpxRow::new("LTGNPELSSLDEVLAK", "run4.raw", 5_018_000.0, &["A5Z2X5"]),
        QpxRow::new("LTGNPELSSLDEVLAK", "run5.raw", 4_421_422.0, &["A5Z2X5"]),
    ];
    write_qpx_rows(&forward_parquet, &rows)?;
    let mut reversed_rows = rows;
    reversed_rows.reverse();
    write_qpx_rows(&reversed_parquet, &reversed_rows)?;
    write_maxlfq_order_sdrf(&sdrf)?;

    let mut config = default_sum_config(forward_parquet, sdrf.clone(), forward_output.clone());
    config.quantification = QuantMethod::MaxLfq;
    run_features_to_proteins(&config)?;
    config.input.parquet = Some(reversed_parquet);
    config.output.protein_matrix = reversed_output.clone();
    run_features_to_proteins(&config)?;

    let forward = read_csv(&forward_output)?;
    let reversed = read_csv(&reversed_output)?;
    let protein_row = forward
        .rows
        .iter()
        .find(|row| row.first().is_some_and(|protein| protein == "A5Z2X5"))
        .ok_or("forward MaxLFQ protein row is missing")?;
    for sample in ["Sample 1", "Sample 2", "Sample 3"] {
        let column = forward
            .headers
            .iter()
            .position(|header| header == sample)
            .ok_or("forward MaxLFQ sample column is missing")?;
        let expected = protein_row[column].parse::<f64>()?;
        assert_numeric_cell_close(&reversed, "A5Z2X5", sample, expected);
    }
    Ok(())
}

#[test]
fn features2proteins_directlfq_exports_python_shaped_ion_matrix() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("directlfq_ions.features.parquet");
    let sdrf = root.join("directlfq_ions.sdrf.tsv");
    let output = root.join("directlfq_ions.protein.csv");
    let ions = root.join("directlfq_ions.ions.csv");

    write_qpx_rows(&parquet, &directlfq_export_rows())?;
    write_directlfq_export_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = QuantMethod::DirectLfq;
    config.output.export_ions = Some(ions.clone());
    run_features_to_proteins(&config)?;

    assert_directlfq_ion_matrix(&read_csv(&ions)?)?;

    let protein = read_csv(&output)?;
    assert_numeric_cell_close(&protein, "P6;Q6", "sample-1", 190.0);
    assert_numeric_cell_close(&protein, "P6;Q6", "sample-2", 190.0);
    Ok(())
}

fn directlfq_export_rows() -> Vec<QpxRow<'static>> {
    vec![
        QpxRow::new_with_charge("PEPTIDEK", "run1.raw", 100.0, 2, &["P6", "Q6"]),
        QpxRow::new_with_charge("PEPTIDEK", "run1.raw", 80.0, 2, &["P6", "Q6"]),
        QpxRow::new_with_charge("PEPTIDEK", "run1.raw", 40.0, 3, &["P6", "Q6"])
            .with_peptidoform("PEP[UNIMOD:35]TIDEK"),
        QpxRow::new("ANOTHERK", "run1.raw", 30.0, &["P6", "Q6"]),
        QpxRow::new_with_charge("PEPTIDEK", "run2.raw", 20.0, 2, &["P6", "Q6"]),
        QpxRow::new_with_charge("PEPTIDEK", "run3.raw", 200.0, 2, &["P6", "Q6"]),
        QpxRow::new_with_charge("PEPTIDEK", "run3.raw", 160.0, 2, &["P6", "Q6"]),
        QpxRow::new_with_charge("PEPTIDEK", "run3.raw", 80.0, 3, &["P6", "Q6"])
            .with_peptidoform("PEP[UNIMOD:35]TIDEK"),
        QpxRow::new("ANOTHERK", "run3.raw", 60.0, &["P6", "Q6"]),
        QpxRow::new_with_charge("PEPTIDEK", "run4.raw", 40.0, 2, &["P6", "Q6"]),
    ]
}

fn write_directlfq_export_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcharacteristics[biological replicate]\tfactor value[cell line]\n",
            "sample-1\trun 1\trun1.raw\tAC=MS:1002038;NT=label free sample\t1\tA\n",
            "sample-1\trun 2\trun2.raw\tAC=MS:1002038;NT=label free sample\t2\tA\n",
            "sample-2\trun 3\trun3.raw\tAC=MS:1002038;NT=label free sample\t1\tB\n",
            "sample-2\trun 4\trun4.raw\tAC=MS:1002038;NT=label free sample\t2\tB\n",
        ),
    )?;
    Ok(())
}

fn assert_directlfq_ion_matrix(table: &CsvTable) -> Result<(), Box<dyn Error>> {
    assert_eq!(table.headers, ["protein", "ion", "sample-1", "sample-2"]);
    assert_eq!(table.rows.len(), 2);
    for ion in ["ANOTHERK", "PEPTIDEK"] {
        for sample in ["sample-1", "sample-2"] {
            assert_directlfq_ion_value(table, ion, sample, 30.0)?;
        }
    }
    Ok(())
}

fn assert_directlfq_ion_value(
    table: &CsvTable,
    ion: &str,
    sample: &str,
    expected: f64,
) -> Result<(), Box<dyn Error>> {
    let sample_index = table
        .headers
        .iter()
        .position(|header| header == sample)
        .ok_or_else(|| format!("sample column '{sample}' is missing"))?;
    let row = table
        .rows
        .iter()
        .find(|row| {
            row.first().is_some_and(|protein| protein == "P6;Q6")
                && row.get(1).is_some_and(|value| value == ion)
        })
        .ok_or_else(|| format!("DirectLFQ ion row P6;Q6/{ion} is missing"))?;
    let actual = row
        .get(sample_index)
        .ok_or_else(|| format!("sample column '{sample}' is missing from ion row"))?
        .parse::<f64>()?;
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1e-9,
        "aligned ion P6;Q6/{ion}/{sample} is {actual}, expected {expected} (delta {delta})"
    );
    Ok(())
}

#[test]
fn features2proteins_normalization_and_irs_match_synthetic_oracles() -> Result<(), Box<dyn Error>> {
    let global_median = run_global_median_quantification()?;
    assert_numeric_cell_close(&global_median, "P20", "sample-1", 2200.0);
    assert_numeric_cell_close(&global_median, "P20", "sample-2", 2200.0);

    let condition_median = run_condition_median_quantification()?;
    assert_numeric_cell_close(&condition_median, "P21", "condA_1", 2200.0);
    assert_numeric_cell_close(&condition_median, "P21", "condA_2", 2200.0);
    assert_numeric_cell_close(&condition_median, "P21", "condB_1", 100.0);
    assert_numeric_cell_close(&condition_median, "P21", "condB_2", 100.0);

    // Both samples share the same sorted distribution, so quantile normalization
    // is the identity and the Sum rollup equals the raw per-sample peptide sums.
    let quantile = run_quantile_sample_normalization()?;
    assert_numeric_cell_close(&quantile, "P50", "sample-1", 30.0);
    assert_numeric_cell_close(&quantile, "P50", "sample-2", 3000.0);
    assert_numeric_cell_close(&quantile, "P51", "sample-1", 300.0);
    assert_numeric_cell_close(&quantile, "P51", "sample-2", 30.0);
    assert_numeric_cell_close(&quantile, "P52", "sample-1", 3000.0);
    assert_numeric_cell_close(&quantile, "P52", "sample-2", 300.0);

    // Different sample distributions are both remapped onto the mean reference
    // distribution [55, 210, 465] (one peptide per protein, min_unique=1).
    let quantile_remap = run_quantile_cross_distribution_normalization()?;
    assert_numeric_cell_close(&quantile_remap, "P60", "sample-1", 55.0);
    assert_numeric_cell_close(&quantile_remap, "P60", "sample-2", 55.0);
    assert_numeric_cell_close(&quantile_remap, "P61", "sample-1", 210.0);
    assert_numeric_cell_close(&quantile_remap, "P61", "sample-2", 210.0);
    assert_numeric_cell_close(&quantile_remap, "P62", "sample-1", 465.0);
    assert_numeric_cell_close(&quantile_remap, "P62", "sample-2", 465.0);

    // LOESS needs >=10 valid points to fit; on this 3-row matrix it falls
    // back to identity (factor 1.0 / "0 samples corrected"), matching Python.
    let loess = run_loess_cross_distribution_normalization()?;
    assert_numeric_cell_close(&loess, "P60", "sample-2", 100.0);
    assert_numeric_cell_close(&loess, "P61", "sample-1", 20.0);
    assert_numeric_cell_close(&loess, "P62", "sample-2", 900.0);

    // Hierarchical needs a pairwise overlap >= 10 rows; on this 3-row matrix the
    // shift is 0.0 (identity), matching Python.
    let hierarchical = run_named_cross_distribution_normalization("hierarchical")?;
    assert_numeric_cell_close(&hierarchical, "P60", "sample-1", 10.0);
    assert_numeric_cell_close(&hierarchical, "P61", "sample-2", 400.0);
    assert_numeric_cell_close(&hierarchical, "P62", "sample-1", 30.0);

    // RLR (robust linear regression vs the per-row median) fits on all 6 rows,
    // exercising the real algorithm. Oracle from the Python reference.
    let rlr = run_rlr_cross_distribution_normalization()?;
    assert_numeric_cell_close(&rlr, "P60", "sample-1", 32.769_585_527_426_57);
    assert_numeric_cell_close(&rlr, "P60", "sample-2", 31.024274739381042);
    assert_numeric_cell_close(&rlr, "P62", "sample-1", 158.04803873053925);
    assert_numeric_cell_close(&rlr, "P65", "sample-1", 531.855_889_076_295_2);
    assert_numeric_cell_close(&rlr, "P65", "sample-2", 486.67655066607495);

    let run_median = run_run_median_quantification()?;
    assert_numeric_cell_close(&run_median, "P22", "sample-run", 2200.0);

    let run_max_min = run_run_max_min_quantification()?;
    assert_numeric_cell_close(&run_max_min, "P24", "sample-run", 2200.0);

    let irs = run_irs_quantification()?;
    assert_eq!(
        irs.headers,
        vec![
            "ProteinName".to_owned(),
            "plexA_2".to_owned(),
            "plexB_2".to_owned(),
        ]
    );
    assert_numeric_cell_close(&irs, "P23", "plexA_2", 1600.0);
    assert_numeric_cell_close(&irs, "P23", "plexB_2", 1600.0);
    Ok(())
}

/// Hierarchical (DirectLFQ-style) real-path parity. Two samples take the n==2
/// branch: with a pairwise overlap of 12 (>= min_overlap 10) the shift is the
/// real `median(sample-1) - median(sample-2)` over the overlap in log2 space
/// (here -0.17581643490674992), not the zero-shift fallback. Pure medians and a
/// single shift make this cell-exact at 1e-9. Oracle:
/// `HierarchicalSampleNormalizer(num_samples_quadratic=50).fit_transform` on the
/// log2 matrix, exponentiated back.
#[test]
fn features2proteins_hierarchical_real_path_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let hierarchical = run_sample_normalization_real_path("hierarchical")?;
    assert_numeric_cell_close(&hierarchical, "P70", "sample-1", 1000.0);
    assert_numeric_cell_close(&hierarchical, "P71", "sample-1", 1499.9999999999998);
    assert_numeric_cell_close(&hierarchical, "P72", "sample-1", 2299.9999999999995);
    assert_numeric_cell_close(&hierarchical, "P73", "sample-1", 3100.0000000000005);
    assert_numeric_cell_close(&hierarchical, "P74", "sample-1", 4399.999999999999);
    assert_numeric_cell_close(&hierarchical, "P75", "sample-1", 5999.999999999999);
    assert_numeric_cell_close(&hierarchical, "P76", "sample-1", 8299.999999999996);
    assert_numeric_cell_close(&hierarchical, "P77", "sample-1", 10999.999999999993);
    assert_numeric_cell_close(&hierarchical, "P78", "sample-1", 14999.999999999993);
    assert_numeric_cell_close(&hierarchical, "P79", "sample-1", 20999.999999999993);
    assert_numeric_cell_close(&hierarchical, "P80", "sample-1", 29000.00000000001);
    assert_numeric_cell_close(&hierarchical, "P81", "sample-1", 39999.99999999998);
    assert_numeric_cell_close(&hierarchical, "P70", "sample-2", 1088.877667829437);
    assert_numeric_cell_close(&hierarchical, "P71", "sample-2", 1425.278898540971);
    assert_numeric_cell_close(&hierarchical, "P72", "sample-2", 2222.018655489339);
    assert_numeric_cell_close(&hierarchical, "P73", "sample-2", 3496.80226660673);
    assert_numeric_cell_close(&hierarchical, "P74", "sample-2", 4240.426039758544);
    assert_numeric_cell_close(&hierarchical, "P75", "sample-2", 6285.391415926021);
    assert_numeric_cell_close(&hierarchical, "P76", "sample-2", 7923.134249653228);
    assert_numeric_cell_close(&hierarchical, "P77", "sample-2", 11242.883236938096);
    assert_numeric_cell_close(&hierarchical, "P78", "sample-2", 15846.268499306456);
    assert_numeric_cell_close(&hierarchical, "P79", "sample-2", 20626.707040996665);
    assert_numeric_cell_close(&hierarchical, "P80", "sample-2", 30541.69068302079);
    assert_numeric_cell_close(&hierarchical, "P81", "sample-2", 38951.72145080913);
    Ok(())
}

/// Hierarchical column-order parity. The four samples register in the order
/// [c, a, d, b] (parquet/SDRF insertion) but Python's pivot orders columns by name
/// [a, b, c, d]; the leaf order, the cluster anchor, and the cumulative shift
/// chain all depend on that column order. This matrix is genuinely order-sensitive
/// (name order anchors hsample-b at shift 0; insertion order would anchor
/// hsample-c, moving every column), so the protein matrix only matches Python when
/// the Rust hierarchical sorts its columns by sample NAME rather than by
/// `SampleId`. This is the guard for that fix; the two-sample real-path test could
/// not catch it because there name-sort and insertion order coincide. Oracle:
/// `HierarchicalSampleNormalizer().fit_transform` on the name-sorted log2 matrix.
#[test]
fn features2proteins_hierarchical_orders_columns_by_sample_name() -> Result<(), Box<dyn Error>> {
    let hierarchical = run_hier_order_real_path()?;
    // Oracle values are computed from the f32-quantized intensities (the QPX
    // parquet stores intensity as f32), so the comparison holds at 1e-9.
    let oracle: [(&str, [(&str, f64); 4]); 12] = [
        (
            "H90",
            [
                ("hsample-a", 157.1363517371411),
                ("hsample-b", 1346.5999755859382),
                ("hsample-c", 1020.3879303023602),
                ("hsample-d", 369.6212233109585),
            ],
        ),
        (
            "H91",
            [
                ("hsample-a", 235.7045276057116),
                ("hsample-b", 2019.9000244140636),
                ("hsample-c", 1530.5820017144852),
                ("hsample-d", 554.4056136920551),
            ],
        ),
        (
            "H92",
            [
                ("hsample-a", 361.41360899542445),
                ("hsample-b", 3097.199951171876),
                ("hsample-c", 2346.9272420504276),
                ("hsample-d", 850.1026051316876),
            ],
        ),
        (
            "H93",
            [
                ("hsample-a", 487.12269038513745),
                ("hsample-b", 4174.499999999998),
                ("hsample-c", 3163.185348412179),
                ("hsample-d", 1145.7995965713192),
            ],
        ),
        (
            "H94",
            [
                ("hsample-a", 691.3999476434207),
                ("hsample-b", 5924.999999999999),
                ("hsample-c", 4489.724660160245),
                ("hsample-d", 1626.333356986486),
            ],
        ),
        (
            "H95",
            [
                ("hsample-a", 942.8181104228464),
                ("hsample-b", 8079.600097656255),
                ("hsample-c", 6122.328006857941),
                ("hsample-d", 2217.7273398657508),
            ],
        ),
        (
            "H96",
            [
                ("hsample-a", 1304.2317194182706),
                ("hsample-b", 11176.7998046875),
                ("hsample-c", 8469.25482386459),
                ("hsample-d", 3067.882451500532),
            ],
        ),
        (
            "H97",
            [
                ("hsample-a", 1728.499869108551),
                ("hsample-b", 14812.599609375),
                ("hsample-c", 11224.267870891634),
                ("hsample-d", 4065.833456420541),
            ],
        ),
        (
            "H98",
            [
                ("hsample-a", 2357.045276057115),
                ("hsample-b", 20199.000000000004),
                ("hsample-c", 15305.81959210108),
                ("hsample-d", 5544.318349664374),
            ],
        ),
        (
            "H99",
            [
                ("hsample-a", 3299.8633864799617),
                ("hsample-b", 28278.59960937499),
                ("hsample-c", 21428.146748871448),
                ("hsample-d", 7762.045689530126),
            ],
        ),
        (
            "H100",
            [
                ("hsample-a", 4556.954200377088),
                ("hsample-b", 39051.39843749999),
                ("hsample-c", 29591.251891465436),
                ("hsample-d", 10719.015476017788),
            ],
        ),
        (
            "H101",
            [
                ("hsample-a", 6285.4540694856405),
                ("hsample-b", 53864.000000000015),
                ("hsample-c", 40815.518912269545),
                ("hsample-d", 14784.848932438334),
            ],
        ),
    ];
    for (protein, samples) in oracle {
        for (sample, expected) in samples {
            assert_numeric_cell_close(&hierarchical, protein, sample, expected);
        }
    }
    Ok(())
}

/// LOESS real-path parity. The 12-row matrix clears the >=10 MA-points guard, so
/// each sample column gets a real LOWESS bias curve subtracted (frac=0.75, it=3)
/// in log2 space rather than passing through unchanged. The Rust LOWESS is a
/// faithful Cleveland reimplementation but differs from statsmodels' `lowess` by
/// sub-0.2% on the smoothed curve (delta interpolation and window-edge
/// arithmetic), so this method is tolerance-aligned (relative 2e-3), not
/// cell-exact. The non-trivial correction is confirmed: every cell moves off its
/// raw value and tracks the Python oracle within tolerance. Oracle:
/// `LOESSNormalizer(frac=0.75, reference="median").fit_transform` on the log2
/// matrix, exponentiated back.
#[test]
fn features2proteins_loess_real_path_matches_python_oracle_within_tolerance(
) -> Result<(), Box<dyn Error>> {
    const RELATIVE_TOLERANCE: f64 = 2e-3;
    let loess = run_sample_normalization_real_path("loess")?;
    let oracle: [(&str, f64, f64); 12] = [
        ("P70", 1076.7736028666823, 1145.61937986298),
        ("P71", 1610.0202075409704, 1502.3120227807826),
        ("P72", 2458.5126090340837, 2349.1549291324277),
        ("P73", 3300.475263712687, 3710.704646105187),
        ("P74", 4690.472415121416, 4492.431325297136),
        ("P75", 6414.491518775941, 6638.182605497903),
        ("P76", 8867.485800189006, 8368.537919740635),
        ("P77", 11758.838636099168, 11877.422292329597),
        ("P78", 16043.02613012259, 16731.212201943723),
        ("P79", 22442.244925638, 21793.439087101913),
        ("P80", 30954.44430098106, 32312.584288539383),
        ("P81", 42638.064465495, 41259.769516195236),
    ];
    for (protein, sample1, sample2) in oracle {
        assert_numeric_cell_relative(&loess, protein, "sample-1", sample1, RELATIVE_TOLERANCE);
        assert_numeric_cell_relative(&loess, protein, "sample-2", sample2, RELATIVE_TOLERANCE);
    }
    Ok(())
}

/// Median-center real-path parity, exercising the per-sample-scale centering bug
/// that the one-peptide-per-protein 12-row fixtures could not catch. Each of the
/// 12 proteins carries TWO canonical peptides, each with TWO charge states (z2,
/// z3), so the summed-canonical-peptide matrix the centering must run on (24 rows
/// per sample) differs from BOTH the protein-sum matrix and the raw per-feature
/// intensities (48 features per sample). The per-sample log2 median therefore
/// lands at a value the raw-feature center never would (peptide-sum median
/// log2 = 13.07/13.25 for sample-1/2 vs raw-feature 12.12/12.28), and the two
/// samples have DISTINCT non-uniform scale factors, so a per-sample scalar bug
/// is visible. Python is dataset-level here (`apply_median_center` ->
/// `_apply_dataset_normalizer(log_space=True)` -> `median_center(axis=0)`):
/// pivot to `(protein, canonical) x sample`, log2, subtract each column's median,
/// `2 **`, then Sum-rollup to proteins. Oracle: the real
/// `mokume.normalization.protein.median_center` on the log2 peptide matrix.
#[test]
fn features2proteins_median_center_real_path_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let median_center = run_center_real_path("mediancenter")?;
    let oracle: [(&str, f64, f64); 12] = [
        ("P70", 0.28983680985010524, 0.3157953791511622),
        ("P71", 0.44055195097215993, 0.4175687842694715),
        ("P72", 0.6260475092762272, 0.6044785696532492),
        ("P73", 0.9274777915203368, 1.043408472252295),
        ("P74", 1.3216558529164792, 1.2761214248235255),
        ("P75", 1.7622078038886395, 1.8419856651886986),
        ("P76", 2.4810030923169, 2.373548879005418),
        ("P77", 3.42007435623124, 3.484018939090464),
        ("P78", 4.44029992690361, 4.680652356283748),
        ("P79", 6.04019911727619, 5.9391097794116625),
        ("P80", 8.799445547049192, 9.275757541564921),
        ("P81", 12.509356713130543, 12.189188146813478),
    ];
    for (protein, sample1, sample2) in oracle {
        assert_numeric_cell_close(&median_center, protein, "sample-1", sample1);
        assert_numeric_cell_close(&median_center, protein, "sample-2", sample2);
    }
    Ok(())
}

/// Mean-center real-path parity on the same two-peptide-per-protein matrix. Python
/// is dataset-level (`apply_mean_center` -> `_apply_dataset_normalizer` ->
/// `x - x.mean(axis=0)` in log2). Oracle: `MeanCenterNormalizer().fit_transform`
/// on the log2 peptide matrix.
#[test]
fn features2proteins_mean_center_real_path_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let mean_center = run_center_real_path("meancenter")?;
    let oracle: [(&str, f64, f64); 12] = [
        ("P70", 0.3003601015570483, 0.32289956078791465),
        ("P71", 0.45654735436671334, 0.42696247615078387),
        ("P72", 0.6487778193632242, 0.6180770129423304),
        ("P73", 0.9611523249825545, 1.0668811504407194),
        ("P74", 1.3696420631001396, 1.3048292495449194),
        ("P75", 1.8261894174668531, 1.8834232592819644),
        ("P76", 2.5710824693283327, 2.426944601278141),
        ("P77", 3.5442491983731683, 3.562395967391871),
        ("P78", 4.601516755853978, 4.785949034806761),
        ("P79", 6.259504516448882, 6.072716910545006),
        ("P80", 9.118932683271984, 9.484426416235856),
        ("P81", 12.963541983202205, 12.463398006477266),
    ];
    for (protein, sample1, sample2) in oracle {
        assert_numeric_cell_close(&mean_center, protein, "sample-1", sample1);
        assert_numeric_cell_close(&mean_center, protein, "sample-2", sample2);
    }
    Ok(())
}

#[test]
fn features2proteins_normalization_proteins_select_factor_inputs_only() -> Result<(), Box<dyn Error>>
{
    let table = run_normalization_proteins_quantification()?;

    assert_numeric_cell_close(&table, "P40", "sample-1", 2200.0);
    assert_numeric_cell_close(&table, "P40", "sample-2", 2200.0);
    assert_numeric_cell_close(&table, "P41", "sample-1", 11000.0);
    assert_numeric_cell_close(&table, "P41", "sample-2", 11.0);
    Ok(())
}

#[test]
fn features2proteins_coverage_filter_matches_condition_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_coverage_quantification()?;

    assert_numeric_cell_close(&table, "P7", "sample-1", 300.0);
    assert_numeric_cell_close(&table, "P7", "sample-2", 700.0);
    assert!(
        table
            .rows
            .iter()
            .all(|row| row.first().is_none_or(|protein| protein != "P8")),
        "protein missing one condition must be filtered"
    );
    Ok(())
}

#[test]
fn features2proteins_mindet_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_mindet_quantification()?;

    assert_numeric_cell_close(&table, "P11", "sample-2", 1000.0);
    assert_numeric_cell_close(&table, "P12", "sample-1", 400.0);
    Ok(())
}

#[test]
fn features2proteins_deterministic_imputation_methods_match_synthetic_oracles(
) -> Result<(), Box<dyn Error>> {
    let minprob = run_imputation_quantification("minprob", |config| {
        config.imputation.quantile = 0.0;
        config.imputation.shift = 1.0;
        config.imputation.scale = 0.0;
    })?;
    // Imputation runs in log2 space (matching the Python pipeline). Python's
    // MinProb is stochastic (draws ~ N(mu, sd)), so cell-exact cross-language
    // parity is not applicable; Rust deterministically fills the distribution
    // mean mu = q_low - shift*sd. With quantile=0, shift=1, scale=0 (sd floored
    // to 0.1): fill = 2**(min(log2 observed) - 0.1) = observed_min * 2**-0.1.
    let shift = 2.0f64.powf(-0.1);
    assert_numeric_cell_close(&minprob, "P31", "sample-2", 20.0 * shift);
    assert_numeric_cell_close(&minprob, "P32", "sample-1", 10.0 * shift);
    assert_numeric_cell_close(&minprob, "P32", "sample-3", 30.0 * shift);

    // mean/median/constant are Rust-only extras (no Python counterpart). In
    // log2 space mean becomes the geometric mean and constant fills 2**0 = 1.
    let mean = run_imputation_quantification("mean", |_| {})?;
    assert_numeric_cell_close(
        &mean,
        "P31",
        "sample-2",
        (20.0f64 * 200.0 * 1000.0).powf(1.0 / 3.0),
    );
    assert_numeric_cell_close(
        &mean,
        "P32",
        "sample-1",
        (10.0f64 * 100.0 * 1000.0).powf(1.0 / 3.0),
    );
    assert_numeric_cell_close(
        &mean,
        "P32",
        "sample-3",
        (30.0f64 * 300.0 * 1000.0).powf(1.0 / 3.0),
    );

    // Median is invariant under the monotone log2 transform.
    let median = run_imputation_quantification("median", |_| {})?;
    assert_numeric_cell_close(&median, "P31", "sample-2", 200.0);
    assert_numeric_cell_close(&median, "P32", "sample-1", 100.0);
    assert_numeric_cell_close(&median, "P32", "sample-3", 300.0);

    let constant = run_imputation_quantification("constant", |_| {})?;
    assert_numeric_cell_close(&constant, "P31", "sample-2", 1.0);
    assert_numeric_cell_close(&constant, "P32", "sample-1", 1.0);
    assert_numeric_cell_close(&constant, "P32", "sample-3", 1.0);

    let zero = run_imputation_quantification("zero", |_| {})?;
    assert_numeric_cell_close(&zero, "P31", "sample-2", 1.0);
    assert_numeric_cell_close(&zero, "P32", "sample-1", 1.0);
    assert_numeric_cell_close(&zero, "P32", "sample-3", 1.0);

    // KNN (k=2) over nan-euclidean distances in log2 space matches the Python
    // reference (sklearn KNNImputer on the log2 matrix, then 2**) cell-for-cell.
    // Averaging two donors in log2 space is the geometric mean: e.g. P31@S2 =
    // 2**((log2 100 + log2 200)/2) = sqrt(100*200) = sqrt(20000).
    let knn = run_imputation_quantification("knn", |config| {
        config.imputation.n_neighbors = 2;
    })?;
    assert_numeric_cell_close(&knn, "P31", "sample-2", 20000.0f64.sqrt());
    assert_numeric_cell_close(&knn, "P32", "sample-1", 100.0);
    assert_numeric_cell_close(&knn, "P32", "sample-3", 30000.0f64.sqrt());
    Ok(())
}

#[test]
fn features2proteins_seqknn_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_seqknn_quantification()?;

    assert_numeric_cell_close(&table, "P62", "sample-2", 20.0);
    assert_numeric_cell_close(&table, "P63", "sample-3", 32.0);
    Ok(())
}

fn run_synthetic_quantification(
    quantification: QuantMethod,
    topn_peptides: usize,
) -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("features.parquet");
    let sdrf = root.join("test.sdrf.tsv");
    let fasta = root.join("test.fasta");
    let output = root.join("protein.csv");

    write_synthetic_qpx(&parquet)?;
    write_synthetic_sdrf(&sdrf)?;
    let fasta = if quantification == QuantMethod::Pibaq {
        write_synthetic_fasta(&fasta)?;
        Some(fasta)
    } else {
        None
    };

    let config = FeatureToProteinsConfig {
        input: InputConfig {
            parquet: Some(parquet),
            msstats: None,
            psm: None,
            sdrf: Some(sdrf),
            fasta,
        },
        output: OutputConfig {
            protein_matrix: output.clone(),
            export_peptides: None,
            export_ions: None,
            format: OutputFormat::PythonCompatible,
        },
        filtering: FilterConfig {
            min_aa: 7,
            min_unique_peptides: 2,
            remove_contaminants: true,
        },
        normalization: NormalizationConfig {
            run_method: "none".to_owned(),
            sample_method: "none".to_owned(),
            normalization_proteins: None,
        },
        quantification,
        topn_peptides,
        maxlfq: MaxLfqConfig::default(),
        pibaq: PibaqConfig::default(),
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig::default(),
        tmt: TmtConfig::default(),
        coverage_threshold: None,
        sample_correlation_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: Some(24),
        },
    };
    if quantification == QuantMethod::Pibaq {
        run_features_to_proteins_with_pibaq_digest(
            &config,
            test_pibaq_digest(&[
                ("P1", &["PEPTIDEAK", "APEPTIDECK", "ASHAEDPEPK"]),
                ("P2", &["ALYAAEK"]),
                ("P3", &["THIDPEAK", "ATHIDPECK"]),
                ("P4A", &["GAAAPEAK", "AGAAAPECK"]),
                ("P4B", &["GAAAPEAK", "AGAAAPECK"]),
            ]),
        )?;
    } else {
        run_features_to_proteins(&config)?;
    }

    read_csv(&output)
}

/// Two canonical peptides where one (PEPTIDEK) is observed at two charge states
/// (z2 with a duplicate run, plus z3). Mirrors the Python loader: max per ion
/// across runs, then sum charges into the canonical peptide, before the
/// per-method rollup. Canonical values for (P50, sample-1): PEPTIDEK = max(100,
/// 80) + 40 = 140, ANOTHERK = 30.
fn run_canonical_collapse_quantification(
    quantification: QuantMethod,
    topn_peptides: usize,
) -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("canonical_collapse.features.parquet");
    let sdrf = root.join("canonical_collapse.sdrf.tsv");
    let output = root.join("canonical_collapse.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEK", "run1.raw", 100.0, 2, &["P50"]),
            QpxRow::new_with_charge("PEPTIDEK", "run2.raw", 80.0, 2, &["P50"]),
            QpxRow::new_with_charge("PEPTIDEK", "run1.raw", 40.0, 3, &["P50"]),
            QpxRow::new_with_charge("ANOTHERK", "run1.raw", 30.0, 2, &["P50"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = quantification;
    config.topn_peptides = topn_peptides;
    if quantification == QuantMethod::Pibaq {
        let fasta = root.join("canonical_collapse.fasta");
        std::fs::write(&fasta, ">P50\nPEPTIDEKANOTHERK\n")?;
        config.input.fasta = Some(fasta);
        run_features_to_proteins_with_pibaq_digest(
            &config,
            test_pibaq_digest(&[("P50", &["PEPTIDEK", "ANOTHERK"])]),
        )?;
    } else {
        run_features_to_proteins(&config)?;
    }

    read_csv(&output)
}

fn run_ratio_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("ratio.features.parquet");
    let sdrf = root.join("ratio.sdrf.tsv");
    let output = root.join("ratio.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "ref.raw", 100.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "case.raw", 200.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "ref.raw", 50.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "case.raw", 200.0, &["P1"]),
        ],
    )?;
    write_ratio_sdrf(&sdrf)?;

    let config = FeatureToProteinsConfig {
        input: InputConfig {
            parquet: Some(parquet),
            msstats: None,
            psm: None,
            sdrf: Some(sdrf),
            fasta: None,
        },
        output: OutputConfig {
            protein_matrix: output.clone(),
            export_peptides: None,
            export_ions: None,
            format: OutputFormat::PythonCompatible,
        },
        filtering: FilterConfig {
            min_aa: 7,
            min_unique_peptides: 2,
            remove_contaminants: true,
        },
        normalization: NormalizationConfig {
            run_method: "none".to_owned(),
            sample_method: "none".to_owned(),
            normalization_proteins: None,
        },
        quantification: QuantMethod::Ratio,
        topn_peptides: 3,
        maxlfq: MaxLfqConfig::default(),
        pibaq: PibaqConfig::default(),
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig {
            reference_samples: Some(vec!["plexA_1".to_owned()]),
            ..IrsConfig::default()
        },
        tmt: TmtConfig::default(),
        coverage_threshold: None,
        sample_correlation_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: Some(24),
        },
    };
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_ratio_multiplex_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("ratio_mplex.features.parquet");
    let sdrf = root.join("ratio_mplex.sdrf.tsv");
    let output = root.join("ratio_mplex.protein.csv");

    // Two peptides per protein (passes min_unique_peptides=2), two plexes with a
    // pooled reference channel each. Matches the Python oracle dataset exactly.
    write_qpx_rows(
        &parquet,
        &[
            // plex p1 (reference channel = p1_ref.raw -> sample p1_1)
            QpxRow::new("PEPTIDEAK", "p1_ref.raw", 100.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "p1_a.raw", 200.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "p1_b.raw", 400.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "p1_ref.raw", 50.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "p1_a.raw", 200.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "p1_b.raw", 25.0, &["P1"]),
            // plex p2 (reference channel = p2_ref.raw -> sample p2_1)
            QpxRow::new("PEPTIDEAK", "p2_ref.raw", 1000.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "p2_a.raw", 4000.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "p2_b.raw", 500.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "p2_ref.raw", 800.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "p2_a.raw", 800.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "p2_b.raw", 3200.0, &["P1"]),
        ],
    )?;
    write_ratio_multiplex_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = QuantMethod::Ratio;
    // No explicit reference samples: the SDRF pooled-sample column drives detection.
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_family_pibaq_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("family.features.parquet");
    let sdrf = root.join("family.sdrf.tsv");
    let fasta = root.join("family.fasta");
    let output = root.join("family.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("QASTVWK", "run1.raw", 300.0, &["P5A"]),
            QpxRow::new("GHILMVK", "run1.raw", 100.0, &["P5B"]),
            QpxRow::new("ACDEFGK", "run1.raw", 400.0, &["P5A", "P5B"]),
            QpxRow::new("LMNSTYK", "run1.raw", 200.0, &["P5A", "P5B"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;
    write_family_fasta(&fasta)?;

    let config = FeatureToProteinsConfig {
        input: InputConfig {
            parquet: Some(parquet),
            msstats: None,
            psm: None,
            sdrf: Some(sdrf),
            fasta: Some(fasta),
        },
        output: OutputConfig {
            protein_matrix: output.clone(),
            export_peptides: None,
            export_ions: None,
            format: OutputFormat::PythonCompatible,
        },
        filtering: FilterConfig {
            min_aa: 7,
            min_unique_peptides: 2,
            remove_contaminants: true,
        },
        normalization: NormalizationConfig {
            run_method: "none".to_owned(),
            sample_method: "none".to_owned(),
            normalization_proteins: None,
        },
        quantification: QuantMethod::Pibaq,
        topn_peptides: 3,
        maxlfq: MaxLfqConfig::default(),
        pibaq: PibaqConfig::default(),
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig::default(),
        tmt: TmtConfig::default(),
        coverage_threshold: None,
        sample_correlation_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: Some(24),
        },
    };
    run_features_to_proteins_with_pibaq_digest(
        &config,
        test_pibaq_digest(&[
            ("P5A", &["ACDEFGK", "LMNSTYK", "QASTVWK"]),
            ("P5B", &["ACDEFGK", "LMNSTYK", "GHILMVK"]),
        ]),
    )?;

    read_csv(&output)
}

fn run_lfq_quantification(quantification: QuantMethod) -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("lfq.features.parquet");
    let sdrf = root.join("lfq.sdrf.tsv");
    let output = root.join("lfq.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P6"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 200.0, &["P6"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 400.0, &["P6"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 800.0, &["P6"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = quantification;
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

/// Same dataset as `run_lfq_quantification`, but forces the built-in MaxLFQ
/// fallback (Python's directlfq-not-installed path) instead of delegating to the
/// DirectLFQ-aligned solver.
fn run_lfq_quantification_builtin_maxlfq() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("lfq_builtin.features.parquet");
    let sdrf = root.join("lfq_builtin.sdrf.tsv");
    let output = root.join("lfq_builtin.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P6"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 200.0, &["P6"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 400.0, &["P6"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 800.0, &["P6"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = QuantMethod::MaxLfq;
    config.maxlfq.force_builtin = true;
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_global_median_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("global_median.features.parquet");
    let sdrf = root.join("global_median.sdrf.tsv");
    let output = root.join("global_median.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P20"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 300.0, &["P20"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 1000.0, &["P20"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 3000.0, &["P20"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.normalization.sample_method = "globalmedian".to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_condition_median_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("condition_median.features.parquet");
    let sdrf = root.join("condition_median.sdrf.tsv");
    let output = root.join("condition_median.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "cond_a1.raw", 100.0, &["P21"]),
            QpxRow::new("APEPTIDECK", "cond_a1.raw", 300.0, &["P21"]),
            QpxRow::new("PEPTIDEAK", "cond_a2.raw", 1000.0, &["P21"]),
            QpxRow::new("APEPTIDECK", "cond_a2.raw", 3000.0, &["P21"]),
            QpxRow::new("PEPTIDEAK", "cond_b1.raw", 10.0, &["P21"]),
            QpxRow::new("APEPTIDECK", "cond_b1.raw", 10.0, &["P21"]),
            QpxRow::new("PEPTIDEAK", "cond_b2.raw", 90.0, &["P21"]),
            QpxRow::new("APEPTIDECK", "cond_b2.raw", 90.0, &["P21"]),
        ],
    )?;
    write_condition_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.normalization.sample_method = "conditionmedian".to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_quantile_sample_normalization() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("quantile_sample.features.parquet");
    let sdrf = root.join("quantile_sample.sdrf.tsv");
    let output = root.join("quantile_sample.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("P50PEPAK", "run1.raw", 10.0, &["P50"]),
            QpxRow::new("P50PEPBK", "run1.raw", 20.0, &["P50"]),
            QpxRow::new("P51PEPAK", "run1.raw", 100.0, &["P51"]),
            QpxRow::new("P51PEPBK", "run1.raw", 200.0, &["P51"]),
            QpxRow::new("P52PEPAK", "run1.raw", 1000.0, &["P52"]),
            QpxRow::new("P52PEPBK", "run1.raw", 2000.0, &["P52"]),
            QpxRow::new("P50PEPAK", "run2.raw", 1000.0, &["P50"]),
            QpxRow::new("P50PEPBK", "run2.raw", 2000.0, &["P50"]),
            QpxRow::new("P51PEPAK", "run2.raw", 10.0, &["P51"]),
            QpxRow::new("P51PEPBK", "run2.raw", 20.0, &["P51"]),
            QpxRow::new("P52PEPAK", "run2.raw", 100.0, &["P52"]),
            QpxRow::new("P52PEPBK", "run2.raw", 200.0, &["P52"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.normalization.sample_method = "quantile".to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

/// Quantile normalization where the two samples have different distributions:
/// sample-1 = [10, 20, 30], sample-2 = [100, 400, 900]. Both must be remapped
/// onto the mean reference distribution [55, 210, 465]. One peptide per protein
/// (min_unique=1) so each protein value equals its peptide's normalized value.
fn run_quantile_cross_distribution_normalization() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("quantile_cross.features.parquet");
    let sdrf = root.join("quantile_cross.sdrf.tsv");
    let output = root.join("quantile_cross.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEK", "run1.raw", 10.0, &["P60"]),
            QpxRow::new("SAMPLEAK", "run1.raw", 20.0, &["P61"]),
            QpxRow::new("GENERICK", "run1.raw", 30.0, &["P62"]),
            QpxRow::new("PEPTIDEK", "run2.raw", 100.0, &["P60"]),
            QpxRow::new("SAMPLEAK", "run2.raw", 400.0, &["P61"]),
            QpxRow::new("GENERICK", "run2.raw", 900.0, &["P62"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.filtering.min_unique_peptides = 1;
    config.normalization.sample_method = "quantile".to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

/// LOESS over the 3-row cross-distribution matrix. With < 10 valid MA points
/// LOESS reports "0 samples corrected" (identity), matching Python.
fn run_loess_cross_distribution_normalization() -> Result<CsvTable, Box<dyn Error>> {
    run_named_cross_distribution_normalization("loess")
}

fn run_named_cross_distribution_normalization(method: &str) -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join(format!("{method}_cross.features.parquet"));
    let sdrf = root.join(format!("{method}_cross.sdrf.tsv"));
    let output = root.join(format!("{method}_cross.protein.csv"));

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEK", "run1.raw", 10.0, &["P60"]),
            QpxRow::new("SAMPLEAK", "run1.raw", 20.0, &["P61"]),
            QpxRow::new("GENERICK", "run1.raw", 30.0, &["P62"]),
            QpxRow::new("PEPTIDEK", "run2.raw", 100.0, &["P60"]),
            QpxRow::new("SAMPLEAK", "run2.raw", 400.0, &["P61"]),
            QpxRow::new("GENERICK", "run2.raw", 900.0, &["P62"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.filtering.min_unique_peptides = 1;
    config.normalization.sample_method = method.to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

/// RLR over a 6-row cross-distribution matrix. RLR fits a robust line per sample
/// against the per-row median with no minimum-row guard, so all 6 rows
/// participate and the real algorithm runs. Oracle from the Python reference.
fn run_rlr_cross_distribution_normalization() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("rlr_cross.features.parquet");
    let sdrf = root.join("rlr_cross.sdrf.tsv");
    let output = root.join("rlr_cross.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEK", "run1.raw", 10.0, &["P60"]),
            QpxRow::new("SAMPLEAK", "run1.raw", 20.0, &["P61"]),
            QpxRow::new("GENERICK", "run1.raw", 30.0, &["P62"]),
            QpxRow::new("HELPERAK", "run1.raw", 40.0, &["P63"]),
            QpxRow::new("MARKERPK", "run1.raw", 55.0, &["P64"]),
            QpxRow::new("TRACERLK", "run1.raw", 70.0, &["P65"]),
            QpxRow::new("PEPTIDEK", "run2.raw", 100.0, &["P60"]),
            QpxRow::new("SAMPLEAK", "run2.raw", 400.0, &["P61"]),
            QpxRow::new("GENERICK", "run2.raw", 900.0, &["P62"]),
            QpxRow::new("HELPERAK", "run2.raw", 1600.0, &["P63"]),
            QpxRow::new("MARKERPK", "run2.raw", 2500.0, &["P64"]),
            QpxRow::new("TRACERLK", "run2.raw", 3600.0, &["P65"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.filtering.min_unique_peptides = 1;
    config.normalization.sample_method = "rlr".to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

/// Twelve distinct peptides (>=7 aa, end in K/R), one per protein P70..P81. With
/// `min_unique_peptides = 1` and Sum quantification the pipeline's (protein x
/// sample) wide matrix equals these intensities exactly, so it is the same matrix
/// the Python normalizers are fitted on. Twelve rows clears every fallback
/// threshold (LOESS >=10 MA points, Hierarchical >=10 overlap), exercising
/// the real shift/smoothing math.
const SAMPLE_NORM_PEPTIDES: [&str; 12] = [
    "AAALEPK", "ACDEFGK", "ADEFGHK", "AEFGHIK", "AFGHIKR", "AGHIKLR", "AHIKLMR", "AIKLMNR",
    "AKLMNPR", "ALMNPQR", "AMNPQRK", "ANPQRSK",
];
const SAMPLE_NORM_PROTEINS: [&str; 12] = [
    "P70", "P71", "P72", "P73", "P74", "P75", "P76", "P77", "P78", "P79", "P80", "P81",
];
// Intensities are exact integers (lossless f32<->f64) and every sample-2 /
// sample-1 ratio is distinct with a clear gap between sorted log2 ratios, so the
// trimming/smoothing normalizers fit on unambiguous rows (no argsort tie between
// numpy and the Rust port).
const SAMPLE_NORM_SAMPLE1: [f32; 12] = [
    1000.0, 1500.0, 2300.0, 3100.0, 4400.0, 6000.0, 8300.0, 11000.0, 15000.0, 21000.0, 29000.0,
    40000.0,
];
const SAMPLE_NORM_SAMPLE2: [f32; 12] = [
    1230.0, 1610.0, 2510.0, 3950.0, 4790.0, 7100.0, 8950.0, 12700.0, 17900.0, 23300.0, 34500.0,
    44000.0,
];

/// Run the 12-row two-sample matrix through `features_to_proteins` with the named
/// sample normalization method, returning the protein matrix.
fn run_sample_normalization_real_path(method: &str) -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join(format!("{method}_real.features.parquet"));
    let sdrf = root.join(format!("{method}_real.sdrf.tsv"));
    let output = root.join(format!("{method}_real.protein.csv"));

    let mut rows = Vec::with_capacity(SAMPLE_NORM_PEPTIDES.len() * 2);
    for index in 0..SAMPLE_NORM_PEPTIDES.len() {
        rows.push(QpxRow::new(
            SAMPLE_NORM_PEPTIDES[index],
            "run1.raw",
            SAMPLE_NORM_SAMPLE1[index],
            std::slice::from_ref(&SAMPLE_NORM_PROTEINS[index]),
        ));
    }
    for index in 0..SAMPLE_NORM_PEPTIDES.len() {
        rows.push(QpxRow::new(
            SAMPLE_NORM_PEPTIDES[index],
            "run2.raw",
            SAMPLE_NORM_SAMPLE2[index],
            std::slice::from_ref(&SAMPLE_NORM_PROTEINS[index]),
        ));
    }
    write_qpx_rows(&parquet, &rows)?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.filtering.min_unique_peptides = 1;
    config.normalization.sample_method = method.to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

// Two-peptide-per-protein, two-charge-per-peptide centering fixture. Twelve
// proteins P70..P81, each with canonical peptides PEPA<nn>AK and PEPB<nn>AK; each
// peptide is observed at charge z2 and z3 in both samples (run1 -> sample-1,
// run2 -> sample-2). This is what makes the centering value-set non-degenerate:
//   * raw features per sample = 48 (the OLD, buggy ingest-time center used these),
//   * summed-canonical peptides per sample = 24 (what Python centers on),
//   * proteins after Sum rollup = 12.
// Their per-sample log2 medians/means differ, so a center taken on the wrong set
// produces a different per-sample scalar and the protein matrix diverges. The
// sample-2 intensities are per-protein non-uniform multiples of sample-1, giving
// the two samples DISTINCT centering factors (the bug's signature).
const CENTER_PEPTIDE_Z2_SAMPLE1: [[f32; 2]; 12] = [
    [1000.0, 500.0],
    [1500.0, 800.0],
    [2300.0, 1100.0],
    [3100.0, 1900.0],
    [4400.0, 2500.0],
    [6000.0, 3700.0],
    [8300.0, 4900.0],
    [11000.0, 6300.0],
    [15000.0, 8100.0],
    [21000.0, 10700.0],
    [29000.0, 15300.0],
    [40000.0, 22900.0],
];
const CENTER_PEPTIDE_Z3_SAMPLE1: [[f32; 2]; 12] = [
    [700.0, 300.0],
    [900.0, 600.0],
    [1300.0, 700.0],
    [1700.0, 1300.0],
    [2900.0, 1600.0],
    [3300.0, 2200.0],
    [5100.0, 3100.0],
    [7700.0, 4500.0],
    [9300.0, 5900.0],
    [13100.0, 7300.0],
    [19700.0, 11900.0],
    [27300.0, 17700.0],
];
const CENTER_PEPTIDE_Z2_SAMPLE2: [[f32; 2]; 12] = [
    [1230.0, 615.0],
    [1605.0, 856.0],
    [2507.0, 1199.0],
    [3937.0, 2413.0],
    [4796.0, 2725.0],
    [7080.0, 4366.0],
    [8964.0, 5292.0],
    [12650.0, 7245.0],
    [17850.0, 9639.0],
    [23310.0, 11877.0],
    [34510.0, 18207.0],
    [44000.0, 25190.0],
];
const CENTER_PEPTIDE_Z3_SAMPLE2: [[f32; 2]; 12] = [
    [861.0, 369.0],
    [963.0, 642.0],
    [1417.0, 763.0],
    [2159.0, 1651.0],
    [3161.0, 1744.0],
    [3894.0, 2596.0],
    [5508.0, 3348.0],
    [8855.0, 5175.0],
    [11067.0, 7021.0],
    [14541.0, 8103.0],
    [23443.0, 14161.0],
    [30030.0, 19470.0],
];

/// Run the two-peptide-per-protein centering matrix through `features_to_proteins`
/// with the named centering method (`mediancenter` / `meancenter`), returning the
/// protein matrix.
fn run_center_real_path(method: &str) -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join(format!("{method}_center.features.parquet"));
    let sdrf = root.join(format!("{method}_center.sdrf.tsv"));
    let output = root.join(format!("{method}_center.protein.csv"));

    let canonical_a: [String; 12] = std::array::from_fn(|index| format!("PEPA{index:02}AK"));
    let canonical_b: [String; 12] = std::array::from_fn(|index| format!("PEPB{index:02}AK"));

    let mut rows = Vec::with_capacity(SAMPLE_NORM_PROTEINS.len() * 8);
    for (run, z2, z3) in [
        (
            "run1.raw",
            &CENTER_PEPTIDE_Z2_SAMPLE1,
            &CENTER_PEPTIDE_Z3_SAMPLE1,
        ),
        (
            "run2.raw",
            &CENTER_PEPTIDE_Z2_SAMPLE2,
            &CENTER_PEPTIDE_Z3_SAMPLE2,
        ),
    ] {
        for index in 0..SAMPLE_NORM_PROTEINS.len() {
            let protein = std::slice::from_ref(&SAMPLE_NORM_PROTEINS[index]);
            rows.push(QpxRow::new_with_charge(
                &canonical_a[index],
                run,
                z2[index][0],
                2,
                protein,
            ));
            rows.push(QpxRow::new_with_charge(
                &canonical_a[index],
                run,
                z3[index][0],
                3,
                protein,
            ));
            rows.push(QpxRow::new_with_charge(
                &canonical_b[index],
                run,
                z2[index][1],
                2,
                protein,
            ));
            rows.push(QpxRow::new_with_charge(
                &canonical_b[index],
                run,
                z3[index][1],
                3,
                protein,
            ));
        }
    }
    write_qpx_rows(&parquet, &rows)?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.filtering.min_unique_peptides = 1;
    config.normalization.sample_method = method.to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

// Four-sample hierarchical matrix whose sample REGISTRATION order (parquet/SDRF
// insertion: run-c, run-a, run-d, run-b) differs from the lexicographic NAME
// order (hsample-a..d). The hierarchical leaf order, the cluster anchor (shift 0),
// and the shift propagation all depend on the column order, so this fixture is
// order-sensitive: under the name order hsample-b anchors at shift 0, but under
// the insertion order hsample-c would anchor instead, shifting every column by a
// different amount. The protein matrix therefore only matches the Python oracle
// when the Rust hierarchical sorts its columns by sample NAME (Python's
// `pivot_table(columns=SAMPLE_ID)` order), not by `SampleId`. Twelve
// single-peptide proteins clear the >=10 pairwise-overlap guard so the real
// median-shift solver runs on every pair. Oracle:
// `HierarchicalSampleNormalizer().fit_transform` on the name-sorted log2 matrix.
const HIER_ORDER_PROTEINS: [&str; 12] = [
    "H90", "H91", "H92", "H93", "H94", "H95", "H96", "H97", "H98", "H99", "H100", "H101",
];
const HIER_ORDER_SAMPLE_A: [f32; 12] = [
    460.0, 690.0, 1058.0, 1426.0, 2024.0, 2760.0, 3818.0, 5060.0, 6900.0, 9660.0, 13340.0, 18400.0,
];
const HIER_ORDER_SAMPLE_B: [f32; 12] = [
    1346.6, 2019.9, 3097.2, 4174.5, 5925.0, 8079.6, 11176.8, 14812.6, 20199.0, 28278.6, 39051.4,
    53864.0,
];
const HIER_ORDER_SAMPLE_C: [f32; 12] = [
    1172.2, 1758.3, 2696.1, 3633.8, 5157.7, 7033.2, 9729.3, 12894.2, 17583.0, 24616.2, 33993.8,
    46888.0,
];
const HIER_ORDER_SAMPLE_D: [f32; 12] = [
    705.5, 1058.2, 1622.6, 2187.0, 3104.2, 4233.0, 5855.7, 7760.5, 10582.5, 14815.5, 20459.5,
    28220.0,
];

/// Run the four-sample hierarchical matrix whose registration order differs from
/// the name order, returning the protein matrix. Rows are emitted run-c, run-a,
/// run-d, run-b so `SampleId` insertion order is [c, a, d, b] while names sort
/// [a, b, c, d].
fn run_hier_order_real_path() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("hier_order.features.parquet");
    let sdrf = root.join("hier_order.sdrf.tsv");
    let output = root.join("hier_order.protein.csv");

    let canonical: [String; 12] = std::array::from_fn(|index| format!("HIERPEP{index:02}AK"));

    let mut rows = Vec::with_capacity(HIER_ORDER_PROTEINS.len() * 4);
    // run-c first so hsample-c registers as SampleId 0 (insertion order != names).
    for (run, intensities) in [
        ("run_c.raw", &HIER_ORDER_SAMPLE_C),
        ("run_a.raw", &HIER_ORDER_SAMPLE_A),
        ("run_d.raw", &HIER_ORDER_SAMPLE_D),
        ("run_b.raw", &HIER_ORDER_SAMPLE_B),
    ] {
        for index in 0..HIER_ORDER_PROTEINS.len() {
            rows.push(QpxRow::new(
                &canonical[index],
                run,
                intensities[index],
                std::slice::from_ref(&HIER_ORDER_PROTEINS[index]),
            ));
        }
    }
    write_qpx_rows(&parquet, &rows)?;
    write_hier_order_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.filtering.min_unique_peptides = 1;
    config.normalization.sample_method = "hierarchical".to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_run_median_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("run_median.features.parquet");
    let sdrf = root.join("run_median.sdrf.tsv");
    let output = root.join("run_median.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run_a.raw", 100.0, &["P22"]),
            QpxRow::new("APEPTIDECK", "run_a.raw", 300.0, &["P22"]),
            QpxRow::new("PEPTIDEAK", "run_b.raw", 1000.0, &["P22"]),
            QpxRow::new("APEPTIDECK", "run_b.raw", 3000.0, &["P22"]),
        ],
    )?;
    write_run_replicate_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.normalization.run_method = "median".to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_run_max_min_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("run_max_min.features.parquet");
    let sdrf = root.join("run_max_min.sdrf.tsv");
    let output = root.join("run_max_min.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run_a.raw", 100.0, &["P24"]),
            QpxRow::new("APEPTIDECK", "run_a.raw", 300.0, &["P24"]),
            QpxRow::new("PEPTIDEAK", "run_b.raw", 1000.0, &["P24"]),
            QpxRow::new("APEPTIDECK", "run_b.raw", 3000.0, &["P24"]),
        ],
    )?;
    write_run_replicate_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.normalization.run_method = "max_min".to_owned();
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_irs_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("irs.features.parquet");
    let sdrf = root.join("irs.sdrf.tsv");
    let output = root.join("irs.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "plex_a_ref.raw", 50.0, &["P23"]),
            QpxRow::new("APEPTIDECK", "plex_a_ref.raw", 50.0, &["P23"]),
            QpxRow::new("PEPTIDEAK", "plex_a_case.raw", 400.0, &["P23"]),
            QpxRow::new("APEPTIDECK", "plex_a_case.raw", 400.0, &["P23"]),
            QpxRow::new("PEPTIDEAK", "plex_b_ref.raw", 200.0, &["P23"]),
            QpxRow::new("APEPTIDECK", "plex_b_ref.raw", 200.0, &["P23"]),
            QpxRow::new("PEPTIDEAK", "plex_b_case.raw", 1600.0, &["P23"]),
            QpxRow::new("APEPTIDECK", "plex_b_case.raw", 1600.0, &["P23"]),
        ],
    )?;
    write_irs_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.irs = IrsConfig {
        enabled: true,
        reference_samples: Some(vec!["plexA_1".to_owned(), "plexB_1".to_owned()]),
        remove_reference: true,
        ..IrsConfig::default()
    };
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_normalization_proteins_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("normalization_proteins.features.parquet");
    let sdrf = root.join("normalization_proteins.sdrf.tsv");
    let selected = root.join("normalization_proteins.txt");
    let output = root.join("normalization_proteins.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P40"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 300.0, &["P40"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 1000.0, &["P40"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 3000.0, &["P40"]),
            QpxRow::new("QASTVWK", "run1.raw", 1000.0, &["P41"]),
            QpxRow::new("GHILMVK", "run1.raw", 1000.0, &["P41"]),
            QpxRow::new("QASTVWK", "run2.raw", 10.0, &["P41"]),
            QpxRow::new("GHILMVK", "run2.raw", 10.0, &["P41"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;
    std::fs::write(&selected, "P40\n")?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.normalization.sample_method = "globalmedian".to_owned();
    config.normalization.normalization_proteins = Some(selected);
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn default_sum_config(parquet: PathBuf, sdrf: PathBuf, output: PathBuf) -> FeatureToProteinsConfig {
    FeatureToProteinsConfig {
        input: InputConfig {
            parquet: Some(parquet),
            msstats: None,
            psm: None,
            sdrf: Some(sdrf),
            fasta: None,
        },
        output: OutputConfig {
            protein_matrix: output,
            export_peptides: None,
            export_ions: None,
            format: OutputFormat::PythonCompatible,
        },
        filtering: FilterConfig {
            min_aa: 7,
            min_unique_peptides: 2,
            remove_contaminants: true,
        },
        normalization: NormalizationConfig {
            run_method: "none".to_owned(),
            sample_method: "none".to_owned(),
            normalization_proteins: None,
        },
        quantification: QuantMethod::Sum,
        topn_peptides: 3,
        maxlfq: MaxLfqConfig::default(),
        pibaq: PibaqConfig::default(),
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig::default(),
        tmt: TmtConfig::default(),
        coverage_threshold: None,
        sample_correlation_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: Some(24),
        },
    }
}

fn run_coverage_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("coverage.features.parquet");
    let sdrf = root.join("coverage.sdrf.tsv");
    let output = root.join("coverage.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P7"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P7"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 300.0, &["P7"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 400.0, &["P7"]),
            QpxRow::new("QASTVWK", "run1.raw", 500.0, &["P8"]),
            QpxRow::new("GHILMVK", "run1.raw", 600.0, &["P8"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    run_features_to_proteins(&FeatureToProteinsConfig {
        input: InputConfig {
            parquet: Some(parquet),
            msstats: None,
            psm: None,
            sdrf: Some(sdrf),
            fasta: None,
        },
        output: OutputConfig {
            protein_matrix: output.clone(),
            export_peptides: None,
            export_ions: None,
            format: OutputFormat::PythonCompatible,
        },
        filtering: FilterConfig {
            min_aa: 7,
            min_unique_peptides: 2,
            remove_contaminants: true,
        },
        normalization: NormalizationConfig {
            run_method: "none".to_owned(),
            sample_method: "none".to_owned(),
            normalization_proteins: None,
        },
        quantification: QuantMethod::Sum,
        topn_peptides: 3,
        maxlfq: MaxLfqConfig::default(),
        pibaq: PibaqConfig::default(),
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig::default(),
        tmt: TmtConfig::default(),
        coverage_threshold: Some(1.0),
        sample_correlation_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: Some(24),
        },
    })?;

    read_csv(&output)
}

fn run_mindet_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("mindet.features.parquet");
    let sdrf = root.join("mindet.sdrf.tsv");
    let output = root.join("mindet.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P10"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 300.0, &["P10"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 1000.0, &["P10"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 3000.0, &["P10"]),
            QpxRow::new("QASTVWK", "run1.raw", 200.0, &["P11"]),
            QpxRow::new("GHILMVK", "run1.raw", 300.0, &["P11"]),
            QpxRow::new("THIDPEAK", "run2.raw", 400.0, &["P12"]),
            QpxRow::new("ATHIDPECK", "run2.raw", 600.0, &["P12"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.imputation = ImputationConfig {
        enabled: true,
        method: "mindet".to_owned(),
        quantile: 0.0,
        ..ImputationConfig::default()
    };
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_imputation_quantification<F>(method: &str, configure: F) -> Result<CsvTable, Box<dyn Error>>
where
    F: FnOnce(&mut FeatureToProteinsConfig),
{
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join(format!("{method}.features.parquet"));
    let sdrf = root.join(format!("{method}.sdrf.tsv"));
    let output = root.join(format!("{method}.protein.csv"));

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "imp1.raw", 5.0, &["P30"]),
            QpxRow::new("APEPTIDECK", "imp1.raw", 5.0, &["P30"]),
            QpxRow::new("PEPTIDEAK", "imp2.raw", 10.0, &["P30"]),
            QpxRow::new("APEPTIDECK", "imp2.raw", 10.0, &["P30"]),
            QpxRow::new("PEPTIDEAK", "imp3.raw", 15.0, &["P30"]),
            QpxRow::new("APEPTIDECK", "imp3.raw", 15.0, &["P30"]),
            QpxRow::new("QASTVWK", "imp1.raw", 50.0, &["P31"]),
            QpxRow::new("GHILMVK", "imp1.raw", 50.0, &["P31"]),
            QpxRow::new("QASTVWK", "imp3.raw", 150.0, &["P31"]),
            QpxRow::new("GHILMVK", "imp3.raw", 150.0, &["P31"]),
            QpxRow::new("THIDPEAK", "imp2.raw", 100.0, &["P32"]),
            QpxRow::new("ATHIDPECK", "imp2.raw", 100.0, &["P32"]),
            QpxRow::new("ALWAYSAK", "imp1.raw", 500.0, &["P33"]),
            QpxRow::new("BALWAYSCK", "imp1.raw", 500.0, &["P33"]),
            QpxRow::new("ALWAYSAK", "imp2.raw", 500.0, &["P33"]),
            QpxRow::new("BALWAYSCK", "imp2.raw", 500.0, &["P33"]),
            QpxRow::new("ALWAYSAK", "imp3.raw", 500.0, &["P33"]),
            QpxRow::new("BALWAYSCK", "imp3.raw", 500.0, &["P33"]),
        ],
    )?;
    write_imputation_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.imputation = ImputationConfig {
        enabled: true,
        method: method.to_owned(),
        ..ImputationConfig::default()
    };
    configure(&mut config);
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn run_seqknn_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("seqknn.features.parquet");
    let sdrf = root.join("seqknn.sdrf.tsv");
    let output = root.join("seqknn.protein.csv");

    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("ACDEFGK", "imp1.raw", 5.0, &["P60"]),
            QpxRow::new("LMNPQRK", "imp1.raw", 5.0, &["P60"]),
            QpxRow::new("ACDEFGK", "imp2.raw", 10.0, &["P60"]),
            QpxRow::new("LMNPQRK", "imp2.raw", 10.0, &["P60"]),
            QpxRow::new("ACDEFGK", "imp3.raw", 15.0, &["P60"]),
            QpxRow::new("LMNPQRK", "imp3.raw", 15.0, &["P60"]),
            QpxRow::new("QRSTVWK", "imp1.raw", 500.0, &["P61"]),
            QpxRow::new("AACDEGK", "imp1.raw", 500.0, &["P61"]),
            QpxRow::new("QRSTVWK", "imp2.raw", 500.0, &["P61"]),
            QpxRow::new("AACDEGK", "imp2.raw", 500.0, &["P61"]),
            QpxRow::new("QRSTVWK", "imp3.raw", 500.0, &["P61"]),
            QpxRow::new("AACDEGK", "imp3.raw", 500.0, &["P61"]),
            QpxRow::new("GHILMVK", "imp1.raw", 6.0, &["P62"]),
            QpxRow::new("QASTVWK", "imp1.raw", 6.0, &["P62"]),
            QpxRow::new("GHILMVK", "imp3.raw", 16.0, &["P62"]),
            QpxRow::new("QASTVWK", "imp3.raw", 16.0, &["P62"]),
            QpxRow::new("THIDPEAK", "imp1.raw", 6.0, &["P63"]),
            QpxRow::new("ATHIDPECK", "imp1.raw", 6.0, &["P63"]),
            QpxRow::new("THIDPEAK", "imp2.raw", 10.0, &["P63"]),
            QpxRow::new("ATHIDPECK", "imp2.raw", 10.0, &["P63"]),
        ],
    )?;
    write_imputation_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.imputation = ImputationConfig {
        enabled: true,
        method: "seqknn".to_owned(),
        n_neighbors: 1,
        ..ImputationConfig::default()
    };
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

fn write_synthetic_qpx(path: &Path) -> Result<(), Box<dyn Error>> {
    let sequence = Arc::new(StringArray::from(vec![
        "PEPTIDEAK",
        "APEPTIDECK",
        "PEPTIDEAK",
        "ALYAAEK",
        "SHORT",
        "CONTAMPEPK",
        "DECOYPEPK",
        "THIDPEAK",
        "ATHIDPECK",
        "PEPTIDEAK",
        "ASHAEDPEPK",
        "GAAAPEAK",
        "AGAAAPECK",
    ])) as ArrayRef;
    let peptidoform = Arc::new(StringArray::from(vec![
        "PEPTIDEAK",
        "APEPTIDECK",
        "PEPTIDEAK",
        "ALYAAEK",
        "SHORT",
        "CONTAMPEPK",
        "DECOYPEPK",
        "THIDPEAK",
        "ATHIDPECK",
        "PEPTIDEAK",
        "ASHAEDPEPK",
        "GAAAPEAK",
        "AGAAAPECK",
    ])) as ArrayRef;
    let charge = Arc::new(Int16Array::from(vec![
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ])) as ArrayRef;
    let run_file_name = Arc::new(StringArray::from(vec![
        "run1.raw", "run1.raw", "run2.raw", "run1.raw", "run1.raw", "run1.raw", "run1.raw",
        "run1.raw", "run1.raw", "run1.raw", "run1.raw", "run1.raw", "run1.raw",
    ])) as ArrayRef;
    let is_decoy = Arc::new(BooleanArray::from(vec![
        false, false, false, false, false, false, true, false, false, false, false, false, false,
    ])) as ArrayRef;
    let unique = Arc::new(BooleanArray::from(vec![
        true, true, true, true, true, true, true, true, true, true, false, true, true,
    ])) as ArrayRef;
    let intensities = synthetic_intensities()?;
    let proteins = synthetic_proteins()?;
    let intensities_field = Field::new("intensities", intensities.data_type().clone(), true);
    let proteins_field = Field::new("pg_accessions", proteins.data_type().clone(), true);

    let schema = Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::Utf8, false),
        Field::new("peptidoform", DataType::Utf8, false),
        Field::new("charge", DataType::Int16, false),
        Field::new("run_file_name", DataType::Utf8, false),
        Field::new("is_decoy", DataType::Boolean, false),
        Field::new("unique", DataType::Boolean, false),
        intensities_field,
        proteins_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            sequence,
            peptidoform,
            charge,
            run_file_name,
            is_decoy,
            unique,
            intensities,
            proteins,
        ],
    )?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn synthetic_intensities() -> Result<ArrayRef, Box<dyn Error>> {
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
    for intensity in [
        100.0_f32, 200.0, 300.0, 500.0, 999.0, 800.0, 50.0, 700.0, 900.0, 150.0, 10_000.0, 10.0,
        20.0,
    ] {
        append_string_field(builder.values(), 0, "label free sample")?;
        append_f32_field(builder.values(), 1, intensity)?;
        builder.values().append(true);
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn synthetic_proteins() -> Result<ArrayRef, Box<dyn Error>> {
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
    let rows = vec![
        (vec!["P1"], 1, 8),
        (vec!["P1"], 9, 16),
        (vec!["P1"], 1, 8),
        (vec!["P2"], 1, 8),
        (vec!["P1"], 17, 21),
        (vec!["sp|CONTAMINANT_P00761|TRYP_PIG"], 1, 9),
        (vec!["sp|DECOY_P11111|REV_HUMAN"], 1, 8),
        (vec!["P3"], 1, 8),
        (vec!["P3"], 9, 16),
        (vec!["P1"], 1, 8),
        (vec!["P1"], 22, 30),
        (vec!["sp|P4A|GROUP_A", "tr|P4B|GROUP_B"], 1, 8),
        (vec!["sp|P4A|GROUP_A", "tr|P4B|GROUP_B"], 9, 16),
    ];

    for (accessions, start, end) in rows {
        for accession in accessions {
            append_string_field(builder.values(), 0, accession)?;
            append_i32_field(builder.values(), 1, start)?;
            append_i32_field(builder.values(), 2, end)?;
            append_string_field(builder.values(), 3, "K")?;
            append_string_field(builder.values(), 4, "R")?;
            builder.values().append(true);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn append_string_field(
    builder: &mut StructBuilder,
    index: usize,
    value: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(builder) = builder.field_builder::<StringBuilder>(index) else {
        return Err(format!("missing string field builder {index}").into());
    };
    builder.append_value(value);
    Ok(())
}

fn append_f32_field(
    builder: &mut StructBuilder,
    index: usize,
    value: f32,
) -> Result<(), Box<dyn Error>> {
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
) -> Result<(), Box<dyn Error>> {
    let Some(builder) = builder.field_builder::<Int32Builder>(index) else {
        return Err(format!("missing i32 field builder {index}").into());
    };
    builder.append_value(value);
    Ok(())
}

fn write_synthetic_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "sample-1\trun 1\trun1.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "sample-2\trun 2\trun2.raw\tAC=MS:1002038;NT=label free sample\tB\n",
        ),
    )?;
    Ok(())
}

fn write_maxlfq_order_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "Sample 1\trun 1\trun1.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "Sample 2\trun 2\trun2.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "Sample 3\trun 3\trun3.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "Sample 4\trun 4\trun4.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "Sample 5\trun 5\trun5.raw\tAC=MS:1002038;NT=label free sample\tA\n",
        ),
    )?;
    Ok(())
}

fn write_hier_order_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    // Names sort [hsample-a..d]; the parquet emits run-c first, so registration
    // order is [c, a, d, b]. The data-file -> source-name mapping is what makes
    // the two orders disagree.
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "hsample-a\trun a\trun_a.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "hsample-b\trun b\trun_b.raw\tAC=MS:1002038;NT=label free sample\tB\n",
            "hsample-c\trun c\trun_c.raw\tAC=MS:1002038;NT=label free sample\tC\n",
            "hsample-d\trun d\trun_d.raw\tAC=MS:1002038;NT=label free sample\tD\n",
        ),
    )?;
    Ok(())
}

fn write_condition_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "condA_1\tcond a1\tcond_a1.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "condA_2\tcond a2\tcond_a2.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "condB_1\tcond b1\tcond_b1.raw\tAC=MS:1002038;NT=label free sample\tB\n",
            "condB_2\tcond b2\tcond_b2.raw\tAC=MS:1002038;NT=label free sample\tB\n",
        ),
    )?;
    Ok(())
}

fn write_run_replicate_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcomment[technical replicate]\tfactor value[cell line]\n",
            "sample-run\trun a\trun_a.raw\tAC=MS:1002038;NT=label free sample\t1\tA\n",
            "sample-run\trun b\trun_b.raw\tAC=MS:1002038;NT=label free sample\t2\tA\n",
        ),
    )?;
    Ok(())
}

fn write_irs_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\tcomment[fraction identifier]\n",
            "plexA_1\tplex a ref\tplex_a_ref.raw\tAC=MS:1002038;NT=label free sample\treference\tplexA\n",
            "plexA_2\tplex a case\tplex_a_case.raw\tAC=MS:1002038;NT=label free sample\tcase\tplexA\n",
            "plexB_1\tplex b ref\tplex_b_ref.raw\tAC=MS:1002038;NT=label free sample\treference\tplexB\n",
            "plexB_2\tplex b case\tplex_b_case.raw\tAC=MS:1002038;NT=label free sample\tcase\tplexB\n",
        ),
    )?;
    Ok(())
}

fn write_imputation_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "sample-1\timp 1\timp1.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "sample-2\timp 2\timp2.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "sample-3\timp 3\timp3.raw\tAC=MS:1002038;NT=label free sample\tA\n",
        ),
    )?;
    Ok(())
}

fn write_synthetic_fasta(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            ">P1\n",
            "PEPTIDEAKAPEPTIDECKASHAEDPEPK\n",
            ">P2\n",
            "ALYAAEK\n",
            ">P3\n",
            "THIDPEAKATHIDPECK\n",
            ">sp|P4A|GROUP_A\n",
            "GAAAPEAKAGAAAPECK\n",
            ">tr|P4B|GROUP_B\n",
            "GAAAPEAKAGAAAPECK\n",
        ),
    )?;
    Ok(())
}

fn test_pibaq_digest(entries: &[(&str, &[&str])]) -> PibaqDigest {
    let accession_peptides = entries
        .iter()
        .map(|(accession, peptides)| {
            (
                (*accession).to_owned(),
                peptides
                    .iter()
                    .map(|peptide| (*peptide).to_owned())
                    .collect::<HashSet<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    PibaqDigest {
        accession_peptides,
        provenance: PibaqDigestProvenance {
            pyopenms_version: "test".to_owned(),
            enzyme: "Trypsin".to_owned(),
            catalog_hash: "test".to_owned(),
            min_aa: 7,
            max_aa: 30,
            missed_cleavages: 0,
        },
    }
}

#[derive(Debug, Clone, Copy)]
struct QpxRow<'a> {
    sequence: &'a str,
    peptidoform: &'a str,
    run_file_name: &'a str,
    intensity: f32,
    charge: i16,
    accessions: &'a [&'a str],
    /// Maps to the parquet `unique` column. `false` marks a shared peptide,
    /// which Python's `SQLFilterBuilder` keeps in the median pre-pass only when
    /// `--keep-shared-peptides` is set (`require_unique = not keep_shared`).
    unique: bool,
    peptide_qvalue: Option<f64>,
    protein_qvalue: Option<f64>,
    score: Option<(&'a str, f64, bool)>,
}

impl<'a> QpxRow<'a> {
    const fn new(
        sequence: &'a str,
        run_file_name: &'a str,
        intensity: f32,
        accessions: &'a [&'a str],
    ) -> Self {
        Self {
            sequence,
            peptidoform: sequence,
            run_file_name,
            intensity,
            charge: 2,
            accessions,
            unique: true,
            peptide_qvalue: None,
            protein_qvalue: None,
            score: None,
        }
    }

    const fn new_with_charge(
        sequence: &'a str,
        run_file_name: &'a str,
        intensity: f32,
        charge: i16,
        accessions: &'a [&'a str],
    ) -> Self {
        Self {
            sequence,
            peptidoform: sequence,
            run_file_name,
            intensity,
            charge,
            accessions,
            unique: true,
            peptide_qvalue: None,
            protein_qvalue: None,
            score: None,
        }
    }

    const fn new_shared(
        sequence: &'a str,
        run_file_name: &'a str,
        intensity: f32,
        accessions: &'a [&'a str],
    ) -> Self {
        Self {
            sequence,
            peptidoform: sequence,
            run_file_name,
            intensity,
            charge: 2,
            accessions,
            unique: false,
            peptide_qvalue: None,
            protein_qvalue: None,
            score: None,
        }
    }

    const fn with_qvalues(mut self, peptide: Option<f64>, protein: Option<f64>) -> Self {
        self.peptide_qvalue = peptide;
        self.protein_qvalue = protein;
        self
    }

    const fn with_score(mut self, name: &'a str, value: f64, higher_better: bool) -> Self {
        self.score = Some((name, value, higher_better));
        self
    }

    const fn with_peptidoform(mut self, peptidoform: &'a str) -> Self {
        self.peptidoform = peptidoform;
        self
    }
}

fn write_qpx_rows(path: &Path, rows: &[QpxRow<'_>]) -> Result<(), Box<dyn Error>> {
    let sequence = Arc::new(StringArray::from(
        rows.iter().map(|row| row.sequence).collect::<Vec<_>>(),
    )) as ArrayRef;
    let peptidoform = Arc::new(StringArray::from(
        rows.iter().map(|row| row.peptidoform).collect::<Vec<_>>(),
    )) as ArrayRef;
    let charge = Arc::new(Int16Array::from(
        rows.iter().map(|row| row.charge).collect::<Vec<_>>(),
    )) as ArrayRef;
    let run_file_name = Arc::new(StringArray::from(
        rows.iter().map(|row| row.run_file_name).collect::<Vec<_>>(),
    )) as ArrayRef;
    let is_decoy = Arc::new(BooleanArray::from(vec![false; rows.len()])) as ArrayRef;
    let unique = Arc::new(BooleanArray::from(
        rows.iter().map(|row| row.unique).collect::<Vec<_>>(),
    )) as ArrayRef;
    let peptide_qvalue = Arc::new(Float64Array::from(
        rows.iter()
            .map(|row| row.peptide_qvalue)
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let protein_qvalue = Arc::new(Float64Array::from(
        rows.iter()
            .map(|row| row.protein_qvalue)
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let additional_scores = qpx_row_scores(rows)?;
    let intensities = qpx_row_intensities(rows)?;
    let proteins = qpx_row_proteins(rows)?;
    let intensities_field = Field::new("intensities", intensities.data_type().clone(), true);
    let proteins_field = Field::new("pg_accessions", proteins.data_type().clone(), true);
    let scores_field = Field::new(
        "additional_scores",
        additional_scores.data_type().clone(),
        true,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::Utf8, false),
        Field::new("peptidoform", DataType::Utf8, false),
        Field::new("charge", DataType::Int16, false),
        Field::new("run_file_name", DataType::Utf8, false),
        Field::new("is_decoy", DataType::Boolean, false),
        Field::new("unique", DataType::Boolean, false),
        Field::new("peptide_qvalue", DataType::Float64, true),
        Field::new("pg_global_qvalue", DataType::Float64, true),
        scores_field,
        intensities_field,
        proteins_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            sequence,
            peptidoform,
            charge,
            run_file_name,
            is_decoy,
            unique,
            peptide_qvalue,
            protein_qvalue,
            additional_scores,
            intensities,
            proteins,
        ],
    )?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn qpx_row_scores(rows: &[QpxRow<'_>]) -> Result<ArrayRef, Box<dyn Error>> {
    let fields = Fields::from(vec![
        Field::new("score_name", DataType::Utf8, false),
        Field::new("score_value", DataType::Float64, false),
        Field::new("higher_better", DataType::Boolean, false),
    ]);
    let struct_builder = StructBuilder::new(
        fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Float64Builder::new()),
            Box::new(BooleanBuilder::new()),
        ],
    );
    let mut builder = ListBuilder::new(struct_builder);
    for row in rows {
        if let Some((name, value, higher_better)) = row.score {
            append_string_field(builder.values(), 0, name)?;
            builder
                .values()
                .field_builder::<Float64Builder>(1)
                .ok_or("missing score_value builder")?
                .append_value(value);
            builder
                .values()
                .field_builder::<BooleanBuilder>(2)
                .ok_or("missing higher_better builder")?
                .append_value(higher_better);
            builder.values().append(true);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn qpx_row_intensities(rows: &[QpxRow<'_>]) -> Result<ArrayRef, Box<dyn Error>> {
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
    for row in rows {
        append_string_field(builder.values(), 0, "label free sample")?;
        append_f32_field(builder.values(), 1, row.intensity)?;
        builder.values().append(true);
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn qpx_row_proteins(rows: &[QpxRow<'_>]) -> Result<ArrayRef, Box<dyn Error>> {
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
    for row in rows {
        for accession in row.accessions {
            append_string_field(builder.values(), 0, accession)?;
            append_i32_field(builder.values(), 1, 1)?;
            append_i32_field(builder.values(), 2, row.sequence.len() as i32)?;
            append_string_field(builder.values(), 3, "K")?;
            append_string_field(builder.values(), 4, "R")?;
            builder.values().append(true);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

/// One synthetic TMT feature: a peptide in a run carrying several channels, each
/// an `(label, intensity)` pair in the `intensities[]` list. The QPX flattener
/// emits one `QpxFeatureRecord` per channel, so the IRS reference-channel filter
/// (`label == irs_channel`) sees the per-channel `label` directly.
struct TmtRow<'a> {
    sequence: &'a str,
    run_file_name: &'a str,
    channels: &'a [(&'a str, f32)],
    accessions: &'a [&'a str],
}

/// Writer for multi-channel TMT QPX parquets (channel-aware sibling of
/// [`write_qpx_rows`], whose rows are single label-free intensities). The schema
/// matches the new-QPX layout the Python `Feature` loader expects.
fn write_qpx_tmt_rows(path: &Path, rows: &[TmtRow<'_>]) -> Result<(), Box<dyn Error>> {
    let sequence = Arc::new(StringArray::from(
        rows.iter().map(|row| row.sequence).collect::<Vec<_>>(),
    )) as ArrayRef;
    let peptidoform = Arc::new(StringArray::from(
        rows.iter().map(|row| row.sequence).collect::<Vec<_>>(),
    )) as ArrayRef;
    let charge = Arc::new(Int16Array::from(vec![2_i16; rows.len()])) as ArrayRef;
    let run_file_name = Arc::new(StringArray::from(
        rows.iter().map(|row| row.run_file_name).collect::<Vec<_>>(),
    )) as ArrayRef;
    let is_decoy = Arc::new(BooleanArray::from(vec![false; rows.len()])) as ArrayRef;
    let unique = Arc::new(BooleanArray::from(vec![true; rows.len()])) as ArrayRef;

    let intensity_fields = Fields::from(vec![
        Field::new("label", DataType::Utf8, false),
        Field::new("intensity", DataType::Float32, false),
    ]);
    let intensity_struct = StructBuilder::new(
        intensity_fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Float32Builder::new()),
        ],
    );
    let mut intensity_builder = ListBuilder::new(intensity_struct);
    for row in rows {
        for (label, value) in row.channels {
            append_string_field(intensity_builder.values(), 0, label)?;
            append_f32_field(intensity_builder.values(), 1, *value)?;
            intensity_builder.values().append(true);
        }
        intensity_builder.append(true);
    }
    let intensities = Arc::new(intensity_builder.finish()) as ArrayRef;

    let proteins = qpx_tmt_proteins(rows)?;
    let intensities_field = Field::new("intensities", intensities.data_type().clone(), true);
    let proteins_field = Field::new("pg_accessions", proteins.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::Utf8, false),
        Field::new("peptidoform", DataType::Utf8, false),
        Field::new("charge", DataType::Int16, false),
        Field::new("run_file_name", DataType::Utf8, false),
        Field::new("is_decoy", DataType::Boolean, false),
        Field::new("unique", DataType::Boolean, false),
        intensities_field,
        proteins_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            sequence,
            peptidoform,
            charge,
            run_file_name,
            is_decoy,
            unique,
            intensities,
            proteins,
        ],
    )?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn qpx_tmt_proteins(rows: &[TmtRow<'_>]) -> Result<ArrayRef, Box<dyn Error>> {
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
    for row in rows {
        for accession in row.accessions {
            append_string_field(builder.values(), 0, accession)?;
            append_i32_field(builder.values(), 1, 1)?;
            append_i32_field(builder.values(), 2, row.sequence.len() as i32)?;
            append_string_field(builder.values(), 3, "K")?;
            append_string_field(builder.values(), 4, "R")?;
            builder.values().append(true);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn write_ratio_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "plexA_1\tref run\tref.raw\tAC=MS:1002038;NT=label free sample\treference\n",
            "plexA_2\tcase run\tcase.raw\tAC=MS:1002038;NT=label free sample\tcase\n",
        ),
    )?;
    Ok(())
}

fn write_ratio_multiplex_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcharacteristics[pooled sample]\tfactor value[cell line]\n",
            "p1_1\tp1 ref\tp1_ref.raw\tAC=MS:1002038;NT=label free sample\tpooled\treference\n",
            "p1_2\tp1 a\tp1_a.raw\tAC=MS:1002038;NT=label free sample\tnot pooled\tcase\n",
            "p1_3\tp1 b\tp1_b.raw\tAC=MS:1002038;NT=label free sample\tnot pooled\tcase\n",
            "p2_1\tp2 ref\tp2_ref.raw\tAC=MS:1002038;NT=label free sample\tpooled\treference\n",
            "p2_2\tp2 a\tp2_a.raw\tAC=MS:1002038;NT=label free sample\tnot pooled\tcase\n",
            "p2_3\tp2 b\tp2_b.raw\tAC=MS:1002038;NT=label free sample\tnot pooled\tcase\n",
        ),
    )?;
    Ok(())
}

fn write_family_fasta(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            ">P5A\n",
            "ACDEFGKLMNSTYKQASTVWK\n",
            ">P5B\n",
            "ACDEFGKLMNSTYKGHILMVK\n",
        ),
    )?;
    Ok(())
}

fn assert_sum_table(actual: &CsvTable) -> Result<(), Box<dyn Error>> {
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
struct CsvTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn read_csv(path: &Path) -> Result<CsvTable, Box<dyn Error>> {
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
struct PeptideParquetTable {
    column_names: Vec<String>,
    dtypes: Vec<String>,
    rows: Vec<(String, String, String, f64)>,
}

/// Read one Utf8-or-Dictionary(Int8, Utf8) string cell from `array` at `row`,
/// transparently decoding the pandas-Categorical dictionary columns that
/// Python's WriteParquetTask emits for SampleID/Condition.
fn parquet_string_cell(
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
        .ok_or("expected utf8 or dictionary column")?;
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("dictionary values are not utf8")?;
    let key = dict.keys().value(row);
    Ok(values.value(key as usize).to_owned())
}

fn header_index(names: &[String], name: &str) -> Result<usize, Box<dyn Error>> {
    names
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| format!("parquet column '{name}' missing").into())
}

fn read_peptide_parquet(path: &Path) -> Result<PeptideParquetTable, Box<dyn Error>> {
    use arrow::array::Float32Array;

    let file = File::open(path)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let batch = reader.next().ok_or("empty parquet")??;
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
        .ok_or("NormIntensity is not float32")?;

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

fn read_peptide_row(
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

fn parquet_norm_intensity(
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

fn assert_numeric_cell_close(table: &CsvTable, protein: &str, sample: &str, expected: f64) {
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
fn assert_numeric_cell_relative(
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

fn assert_cell_close(actual: Option<&String>, expected: Option<&String>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) if actual.is_empty() && expected.is_empty() => {}
        (Some(actual), Some(expected)) => assert_float_close(actual, expected),
        _ => panic!("CSV row width mismatch: actual={actual:?}, expected={expected:?}"),
    }
}

fn assert_float_close(actual: &str, expected: &str) {
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

fn temp_root() -> Result<PathBuf, Box<dyn Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    Ok(tempfile::Builder::new()
        .prefix(&format!("mokume-golden-{timestamp}-"))
        .tempdir()?
        .keep())
}

// impSeq: with only 2 complete rows (P30, P33) < max(2, p=3), the algorithm
// takes the column-mean fallback, so each missing log2 cell is the mean of its
// column's finite log2 values -> the geometric mean of the column's observed
// raw values after 2**. Deterministic, matching the Python reference.
#[test]
fn features2proteins_impseq_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let impseq = run_imputation_quantification("impseq", |_| {})?;
    assert_numeric_cell_close(
        &impseq,
        "P32",
        "sample-1",
        (10.0_f64 * 100.0 * 1000.0).powf(1.0 / 3.0),
    );
    assert_numeric_cell_close(
        &impseq,
        "P31",
        "sample-2",
        (20.0_f64 * 200.0 * 1000.0).powf(1.0 / 3.0),
    );
    assert_numeric_cell_close(
        &impseq,
        "P32",
        "sample-3",
        (30.0_f64 * 300.0 * 1000.0).powf(1.0 / 3.0),
    );
    assert_numeric_cell_close(&impseq, "P31", "sample-1", 100.0);
    assert_numeric_cell_close(&impseq, "P32", "sample-2", 200.0);
    Ok(())
}

// GMS is stochastic in Python (GMM + random draw), so cell-exact cross-language
// parity is not applicable. The Rust port deterministically fills the lower
// mixture component's mean (like MinProb). For this 3-observations-per-column
// matrix the lower component collapses onto the minimum observation, and every
// fill lies in [min_observed, geometric_mean_observed].
#[test]
fn features2proteins_gms_imputation_matches_stochastic_oracle_property(
) -> Result<(), Box<dyn Error>> {
    let gms = run_imputation_quantification("gms", |_| {})?;
    assert_numeric_cell_close(&gms, "P31", "sample-2", 20.0);
    assert_numeric_cell_close(&gms, "P32", "sample-1", 10.0);
    assert_numeric_cell_close(&gms, "P32", "sample-3", 30.0);
    assert_gms_in_band(
        &gms,
        "P31",
        "sample-2",
        20.0,
        geometric_mean(&[20.0, 200.0, 1000.0]),
    );
    assert_gms_in_band(
        &gms,
        "P32",
        "sample-1",
        10.0,
        geometric_mean(&[10.0, 100.0, 1000.0]),
    );
    assert_gms_in_band(
        &gms,
        "P32",
        "sample-3",
        30.0,
        geometric_mean(&[30.0, 300.0, 1000.0]),
    );
    Ok(())
}

fn geometric_mean(values: &[f64]) -> f64 {
    let sum_log = values.iter().map(|value| value.ln()).sum::<f64>();
    (sum_log / values.len() as f64).exp()
}

fn assert_gms_in_band(table: &CsvTable, protein: &str, sample: &str, lower: f64, upper: f64) {
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
    let Some(cell) = row.get(sample_index) else {
        panic!("sample column {sample} is missing from row {protein}");
    };
    let Ok(actual) = cell.parse::<f64>() else {
        panic!("numeric cell is not a valid float: {cell}");
    };
    assert!(
        actual >= lower - 1e-9 && actual <= upper + 1e-9,
        "GMS fill {actual} for {protein}@{sample} outside band [{lower}, {upper}]"
    );
}

// BPCA (Bayesian PCA EM) is deterministic in Python (numpy svd/inv/cov only, no
// RNG). On this 4x3 matrix k = min(2, min(4,3)-1) = 2 so the full EM branch
// runs; oracle from the Python reference (impute_bpca on log2, then 2**).
#[test]
fn features2proteins_bpca_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_imputation_quantification("bpca", |_| {})?;
    assert_numeric_cell_close(&table, "P31", "sample-2", 2.0f64.powf(7.522275951523529));
    assert_numeric_cell_close(&table, "P32", "sample-1", 2.0f64.powf(7.014180587257054));
    assert_numeric_cell_close(&table, "P32", "sample-3", 2.0f64.powf(7.986058602267095));
    assert_numeric_cell_close(&table, "P30", "sample-1", 10.0);
    assert_numeric_cell_close(&table, "P33", "sample-2", 1000.0);
    Ok(())
}

// impSeqRob (robust sequential) is deterministic. On this matrix (2 complete
// rows < ceil(p/alpha)=4) the conditional-MVN pre-impute branch fills both
// incomplete rows; oracle from the Python reference (impute_impseqrob on log2,
// then 2**).
#[test]
fn features2proteins_impseqrob_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_imputation_quantification("impseqrob", |_| {})?;
    assert_numeric_cell_close(&table, "P31", "sample-2", 2.0f64.powf(7.569660028762611));
    assert_numeric_cell_close(&table, "P32", "sample-1", 2.0f64.powf(6.967599172815016));
    assert_numeric_cell_close(&table, "P32", "sample-3", 2.0f64.powf(7.947008510825019));
    assert_numeric_cell_close(&table, "P30", "sample-1", 10.0);
    assert_numeric_cell_close(&table, "P33", "sample-2", 1000.0);
    Ok(())
}

// QRILC is stochastic in Python (truncated-normal draw). The Rust port fills the
// deterministic center mu_imp = trunc - sd_obs (trunc = min observed, sd ddof=1
// floored to 0.1), like MinProb/GMS. Oracle = 2**(log2-space center); every fill
// lies at or below the column minimum (left-censored tail).
#[test]
fn features2proteins_qrilc_imputation_matches_stochastic_oracle_property(
) -> Result<(), Box<dyn Error>> {
    let qrilc = run_imputation_quantification("qrilc", |_| {})?;
    assert_numeric_cell_close(
        &qrilc,
        "P32",
        "sample-1",
        2.0f64.powf(-4.440892098500626e-16),
    );
    assert_numeric_cell_close(&qrilc, "P31", "sample-2", 2.0f64.powf(1.4852731095132246));
    assert_numeric_cell_close(&qrilc, "P32", "sample-3", 2.0f64.powf(2.336395795637601));
    assert_qrilc_below_min(&qrilc, "P32", "sample-1", 10.0);
    assert_qrilc_below_min(&qrilc, "P31", "sample-2", 20.0);
    assert_qrilc_below_min(&qrilc, "P32", "sample-3", 30.0);
    Ok(())
}

fn assert_qrilc_below_min(table: &CsvTable, protein: &str, sample: &str, min_observed: f64) {
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
    let Some(cell) = row.get(sample_index) else {
        panic!("sample column {sample} is missing from row {protein}");
    };
    let Ok(actual) = cell.parse::<f64>() else {
        panic!("numeric cell is not a valid float: {cell}");
    };
    assert!(
        actual > 0.0 && actual <= min_observed + 1e-9,
        "QRILC fill {actual} for {protein}@{sample} not in (0, min_observed={min_observed}]"
    );
}

// ===========================================================================
// features2proteins --de-method deqms: PER-PROTEIN PEPTIDE-COUNT wiring.
//
// This is the pipeline-level proof that the DEqMS DE path consumes the real
// per-protein unique-canonical-peptide counts (captured at ingest) rather than
// the all-ones fallback. The synthetic parquet has 12 proteins, each with a
// DISTINCT number of unique peptide sequences (1,2,3,4,5,6,7,8,9,11,13,15), so
// the spectraCounteBayes count moderation genuinely runs (>= 10 testable
// proteins and a real log2(count) spread), NOT the limma fallback.
//
// Each protein's per-(sample) protein-level Sum is fixed to a controlled target:
// one "anchor" peptide carries `target - (count-1)` and the remaining `count-1`
// peptides each carry 1.0, so the Sum rollup recovers the integer target exactly
// (f32-exact for these magnitudes). The log2 protein matrix is therefore a clean,
// known matrix; the only thing the count vector changes is the deqms moderation.
//
// Oracle (captured verbatim from `conda run -n Bigbio python` on mokume's
// `run_deqms(log2_matrix, A, B, ("groupA","groupB"), peptide_counts=counts)`,
// where `counts = parquet.groupby("anchor_protein")["sequence"].nunique()`, on
// the exact log2 matrix this Rust run produces; see scratchpad
// deqms_pipeline_oracle.py). Tuple: (protein, log2FC, sca_t, sca_pvalue,
// peptide_count). DEqMS is LOWESS-bounded, so sca_t is asserted at relative 6e-3
// (the established mokume-stats tolerance) and the adj-p rank is asserted exactly.
const DEQMS_PIPELINE_ORACLE: &[(&str, f64, f64, f64, usize)] = &[
    ("PROT05", -1.0, -98.5279709372, 3.82786439473e-10, 8),
    ("PROT09", -1.0, -52.9074128499, 1.16613161144e-08, 11),
    ("PROT08", -1.0, -37.1125038157, 8.1589378105e-08, 7),
    ("PROT07", 1.0, 30.7917066547, 2.26854357277e-07, 4),
    ("PROT04", -1.0, -23.5060652956, 9.9169228319e-07, 5),
    ("PROT10", 1.0, 19.3560071397, 2.85460418766e-06, 6),
    ("PROT12", -1.0, -17.2840417306, 5.27452228571e-06, 15),
    ("PROT02", 2.00906160978, 16.5837158474, 6.59733867016e-06, 2),
    ("PROT01", -2.00048684329, -15.8911018093, 8.307394376e-06, 1),
    ("PROT03", 0.0159106916777, 0.345686569053, 0.742412704892, 3),
    (
        "PROT06",
        0.0917559508266,
        0.274768056767,
        0.793520719233,
        13,
    ),
    (
        "PROT11",
        -0.00166445565951,
        -0.120709050091,
        0.908209010135,
        9,
    ),
];

// The deqms adj-p rank above, in order. The count moderation reorders the middle
// of the table relative to the all-ones limma fallback (which ranks
// [..PROT01,PROT02,PROT12..]); the count path ranks [..PROT12,PROT02,PROT01..].
const DEQMS_PIPELINE_RANK: &[&str] = &[
    "PROT05", "PROT09", "PROT08", "PROT07", "PROT04", "PROT10", "PROT12", "PROT02", "PROT01",
    "PROT03", "PROT06", "PROT11",
];

// All-ones limma fallback sca_t (Python `run_deqms(..., peptide_counts=None)` on
// the same matrix). Used ONLY to prove non-vacuity: the deqms-with-counts sca_t
// must differ from this fallback by a large margin, so the test cannot be passed
// by the old all-ones code path. Tuple: (protein, fallback_sca_t).
const DEQMS_FALLBACK_SCA_T: &[(&str, f64)] = &[
    ("PROT05", -65.11281129),
    ("PROT09", -46.68051845),
    ("PROT08", -33.8387029),
    ("PROT07", 31.91915491),
    ("PROT04", -22.73149498),
    ("PROT10", 18.51554925),
    ("PROT01", -17.31471545),
    ("PROT02", 16.53307016),
    ("PROT12", -16.76645805),
    ("PROT03", 0.3754350973),
    ("PROT06", 0.2646723684),
    ("PROT11", -0.09359488618),
];

#[test]
fn features2proteins_deqms_wires_per_protein_peptide_counts() -> Result<(), Box<dyn Error>> {
    let root = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("deqms.features.parquet");
    let sdrf = root.join("deqms.sdrf.tsv");
    let output = root.join("deqms.protein.csv");
    let de_output = root.join("deqms.de.csv");

    write_deqms_count_features(&parquet)?;
    write_deqms_count_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output);
    // One unique peptide is enough to keep every protein (PROT01 has a single
    // sequence); the count spread is what drives the deqms moderation.
    config.filtering.min_unique_peptides = 1;
    config.differential_expression = DifferentialExpressionConfig {
        enabled: true,
        contrasts: Some(vec!["groupA vs groupB".to_owned()]),
        method: "deqms".to_owned(),
        output: Some(de_output.clone()),
        ..DifferentialExpressionConfig::default()
    };
    run_features_to_proteins(&config)?;

    let table = read_csv(&de_output)?;
    assert_eq!(
        table.headers,
        vec![
            "ProteinName",
            "log2FC",
            "pvalue",
            "adj_pvalue",
            "sca_t",
            "sca_pvalue",
            "sca_adj_pvalue",
            "mean_groupA",
            "mean_groupB",
            "n_a",
            "n_b",
            "peptide_count",
            "log_pvalue",
            "significance",
        ]
    );
    assert_eq!(table.rows.len(), DEQMS_PIPELINE_ORACLE.len());

    assert_deqms_oracle_rows(&table)?;
    assert_deqms_rank(&table)?;
    assert_deqms_differs_from_fallback(&table)?;
    Ok(())
}

/// Assert every protein row matches the count-aware Python deqms oracle. Field
/// layout: ProteinName(0), log2FC(1), pvalue(2), adj_pvalue(3), sca_t(4),
/// sca_pvalue(5), sca_adj_pvalue(6), mean_A(7), mean_B(8), n_a(9), n_b(10),
/// peptide_count(11), log_pvalue(12), significance(13).
fn assert_deqms_oracle_rows(table: &CsvTable) -> Result<(), Box<dyn Error>> {
    for &(protein, log2fc, sca_t, sca_p, count) in DEQMS_PIPELINE_ORACLE {
        let row = find_de_row(table, protein)?;
        // log2FC is the untouched limma coefficient: cell-exact (1e-9).
        assert_de_cell_abs(row, 1, log2fc, 1e-9, protein, "log2FC")?;
        // sca_t / sca_pvalue: LOWESS-bounded relative 6e-3 (matches mokume-stats'
        // deqms oracle tolerance), with an absolute floor so near-zero t / large p
        // are not over-constrained.
        assert_de_cell_rel(row, 4, sca_t, 6e-3, 1.0, protein, "sca_t")?;
        assert_de_cell_rel(row, 5, sca_p, 6e-3, 1e-4, protein, "sca_pvalue")?;
        // pvalue == sca_pvalue and adj_pvalue == sca_adj_pvalue, exactly as Python.
        assert_de_columns_equal(row, 2, 5, protein, "pvalue==sca_pvalue")?;
        assert_de_columns_equal(row, 3, 6, protein, "adj_pvalue==sca_adj_pvalue")?;
        // The per-protein peptide_count is the exact unique-sequence count: this is
        // the core of the fix (Rust nunique == Python nunique).
        let actual_count = de_field(row, 11)?;
        assert_eq!(
            actual_count,
            count.to_string(),
            "peptide_count for {protein}: got {actual_count}, expected {count}"
        );
    }
    Ok(())
}

/// Assert the emitted adj-p rank matches the count-aware oracle exactly (the
/// count moderation reorders the middle of the table vs the all-ones fallback).
fn assert_deqms_rank(table: &CsvTable) -> Result<(), Box<dyn Error>> {
    let rank = table
        .rows
        .iter()
        .map(|row| {
            row.first()
                .cloned()
                .ok_or_else(|| -> Box<dyn Error> { "empty deqms row".into() })
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        rank, DEQMS_PIPELINE_RANK,
        "deqms adj-p rank must match the count-aware Python oracle"
    );
    Ok(())
}

/// Non-vacuity: the emitted sca_t must differ materially from the all-ones limma
/// fallback, proving the count path is genuinely exercised (the old all-ones code
/// could not have produced these numbers). The worst mover (PROT05) shifts ~33.
fn assert_deqms_differs_from_fallback(table: &CsvTable) -> Result<(), Box<dyn Error>> {
    let mut max_fallback_diff = 0.0_f64;
    for &(protein, fallback_t) in DEQMS_FALLBACK_SCA_T {
        let row = find_de_row(table, protein)?;
        let actual_t = de_field(row, 4)?.parse::<f64>()?;
        max_fallback_diff = max_fallback_diff.max((actual_t - fallback_t).abs());
    }
    assert!(
        max_fallback_diff > 5.0,
        "deqms sca_t too close to the all-ones limma fallback (max diff {max_fallback_diff}); \
         the count path is not being exercised"
    );
    Ok(())
}

/// Find a DE result row by protein name, erroring if it is missing.
fn find_de_row<'a>(table: &'a CsvTable, protein: &str) -> Result<&'a [String], Box<dyn Error>> {
    table
        .rows
        .iter()
        .find(|row| row.first().is_some_and(|value| value == protein))
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing deqms row {protein}").into())
}

/// Look up a field of a DE CSV row by index.
fn de_field(row: &[String], index: usize) -> Result<&str, Box<dyn Error>> {
    row.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("DE row field {index} is missing").into())
}

/// Assert a DE cell matches `expected` at an absolute tolerance.
fn assert_de_cell_abs(
    row: &[String],
    index: usize,
    expected: f64,
    tolerance: f64,
    protein: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let actual = de_field(row, index)?.parse::<f64>()?;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{protein}.{label}: got {actual}, expected {expected} (abs tol {tolerance})"
    );
    Ok(())
}

/// Assert a DE cell matches `expected` at a relative tolerance with an absolute
/// floor (so tiny / near-zero magnitudes are not over-constrained).
fn assert_de_cell_rel(
    row: &[String],
    index: usize,
    expected: f64,
    relative: f64,
    absolute_floor: f64,
    protein: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let actual = de_field(row, index)?.parse::<f64>()?;
    let tolerance = (expected.abs() * relative).max(absolute_floor);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{protein}.{label}: got {actual}, expected {expected} (tol {tolerance})"
    );
    Ok(())
}

/// Assert two DE cells are bit-identical (used for `pvalue == sca_pvalue`).
fn assert_de_columns_equal(
    row: &[String],
    left: usize,
    right: usize,
    protein: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let left_value = de_field(row, left)?;
    let right_value = de_field(row, right)?;
    assert_eq!(
        left_value, right_value,
        "{protein}.{label}: columns {left} and {right} differ ({left_value} vs {right_value})"
    );
    Ok(())
}

/// Per-protein intended per-sample protein-level Sum (linear), columns
/// A1,A2,A3,B1,B2,B3. Mirrors scratchpad/gen_deqms_parquet.py.
const DEQMS_TARGETS: &[(&str, [f32; 6])] = &[
    ("PROT01", [1000.0, 1149.0, 870.0, 4000.0, 4290.0, 3732.0]),
    (
        "PROT02",
        [32000.0, 34000.0, 30000.0, 8000.0, 9200.0, 6800.0],
    ),
    ("PROT03", [256.0, 270.0, 244.0, 257.0, 248.0, 256.0]),
    ("PROT04", [5000.0, 5200.0, 4800.0, 10000.0, 10400.0, 9600.0]),
    ("PROT05", [320.0, 322.0, 318.0, 640.0, 644.0, 636.0]),
    ("PROT06", [512.0, 724.0, 362.0, 480.0, 660.0, 350.0]),
    ("PROT07", [2048.0, 2100.0, 1990.0, 1024.0, 1050.0, 995.0]),
    ("PROT08", [800.0, 820.0, 780.0, 1600.0, 1640.0, 1560.0]),
    ("PROT09", [128.0, 130.0, 126.0, 256.0, 260.0, 252.0]),
    ("PROT10", [6000.0, 6300.0, 5700.0, 3000.0, 3150.0, 2850.0]),
    ("PROT11", [1500.0, 1520.0, 1480.0, 1490.0, 1510.0, 1505.0]),
    ("PROT12", [900.0, 950.0, 850.0, 1800.0, 1900.0, 1700.0]),
];

/// Per-protein number of UNIQUE peptide sequences (the count vector). Index
/// aligns with `DEQMS_TARGETS`.
const DEQMS_COUNTS: [usize; 12] = [1, 2, 3, 5, 8, 13, 4, 7, 11, 6, 9, 15];

const DEQMS_SAMPLE_RUNS: [&str; 6] = ["A1.raw", "A2.raw", "A3.raw", "B1.raw", "B2.raw", "B3.raw"];

/// Write the 12-protein deqms feature parquet. Each protein gets `count` distinct
/// peptide sequences: one anchor carrying `target - (count-1)` and `count-1`
/// extras carrying 1.0, so the Sum rollup per (protein, sample) recovers the
/// integer target. The single `pg_accessions` entry equals the protein name, so
/// the parsed ProteinName == the anchor_protein grouping Python counts over.
fn write_deqms_count_features(path: &Path) -> Result<(), Box<dyn Error>> {
    // Owned storage that outlives the borrowed `QpxRow` slice.
    let mut sequences: Vec<String> = Vec::new();
    let mut accession_storage: Vec<Vec<String>> = Vec::new();
    // (sequence_index, accession_storage_index, run_index, intensity)
    let mut plan: Vec<(usize, usize, usize, f32)> = Vec::new();

    for (protein_index, (protein, targets)) in DEQMS_TARGETS.iter().enumerate() {
        let count = DEQMS_COUNTS[protein_index];
        let accession_index = accession_storage.len();
        accession_storage.push(vec![(*protein).to_owned()]);
        let first_sequence = sequences.len();
        for sequence_number in 0..count {
            sequences.push(format!("{protein}PEPSEQ{sequence_number:03}K"));
        }
        let extras = count - 1;
        for (run_index, target) in targets.iter().enumerate() {
            // Anchor sequence carries target - extras; each extra carries 1.0.
            let anchor_intensity = target - extras as f32;
            plan.push((first_sequence, accession_index, run_index, anchor_intensity));
            for extra in 1..count {
                plan.push((first_sequence + extra, accession_index, run_index, 1.0));
            }
        }
    }

    let accession_refs: Vec<Vec<&str>> = accession_storage
        .iter()
        .map(|group| group.iter().map(String::as_str).collect())
        .collect();
    let rows = plan
        .iter()
        .map(|&(sequence_index, accession_index, run_index, intensity)| {
            QpxRow::new(
                sequences[sequence_index].as_str(),
                DEQMS_SAMPLE_RUNS[run_index],
                intensity,
                accession_refs[accession_index].as_slice(),
            )
        })
        .collect::<Vec<_>>();
    write_qpx_rows(path, &rows)
}

/// Write the 3-vs-3 SDRF for the deqms count fixture (A1..A3 -> groupA, B1..B3 ->
/// groupB), one source name / data file per sample.
fn write_deqms_count_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut tsv = String::from(
        "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[phenotype]\n",
    );
    for (index, run) in DEQMS_SAMPLE_RUNS.iter().enumerate() {
        let sample = run.trim_end_matches(".raw");
        let condition = if sample.starts_with('A') {
            "groupA"
        } else {
            "groupB"
        };
        tsv.push_str(&format!(
            "{sample}\trun {index}\t{run}\tAC=MS:1002038;NT=label free sample\t{condition}\n"
        ));
    }
    std::fs::write(path, tsv)?;
    Ok(())
}
