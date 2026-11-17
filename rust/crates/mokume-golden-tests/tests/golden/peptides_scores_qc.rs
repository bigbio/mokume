use super::*;

// The synthetic QPX lacks the requested score and any coverage column. Reject
// both instead of accepting filters that cannot affect output.
#[test]
fn features2peptides_rejects_unavailable_search_score_and_coverage_filters()
-> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    assert!(
        peptide_table
            .rows
            .iter()
            .all(|row| row.get(1).is_none_or(|sequence| sequence != "APEPTIDECK"))
    );

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
    assert!(
        protein_table
            .rows
            .iter()
            .all(|row| row.first().is_none_or(|value| value == "P2"))
    );
    assert_eq!(protein_table.rows.len(), 2);
    Ok(())
}

#[test]
fn features2peptides_fdr_request_rejects_unpopulated_qvalue() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    assert!(
        error
            .to_string()
            .contains("requires a populated QPX `peptide_qvalue` column")
    );
    assert!(!output.exists());
    Ok(())
}

// Run-QC MinFeaturesFilter counts distinct `(protein, canonical)` features per
// technical run after the per-sample protein min_unique gate. Duplicate charge
// states must not inflate the count: run1 has three canonicals, while run2 has
// two, so min_features=3 keeps only run1.
#[test]
fn features2peptides_run_qc_drops_low_feature_samples() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
