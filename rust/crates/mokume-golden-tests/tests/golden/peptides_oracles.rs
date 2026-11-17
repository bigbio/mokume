use super::*;

// Oracle (--skip_normalization --log2, no --sdrf): NormIntensity = log2(sum).
//   P1,APEPTIDECK,run2 -> log2(400) = 8.643856...
//   P1,PEPTIDEAK,run2  -> log2(300) = 8.228818...
//   P1,APEPTIDECK,run1 -> log2(200) = 7.643856...
//   P1,PEPTIDEAK,run1  -> log2(250) = 7.965784...
#[test]
fn features2peptides_log2_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
        let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
fn features2peptides_remove_low_frequency_peptides_matches_python_oracle()
-> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
