use super::*;

#[test]
fn tmm_peptide_export_matches_canonical_protein_rollup() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    let parquet = root.join("features.parquet");
    let sdrf = root.join("samples.tsv");
    write_tmm_export_features(&parquet)?;
    write_synthetic_sdrf(&sdrf)?;
    let output = root.join("proteins.csv");
    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.normalization.sample_method = "tmm".to_owned();
    run_features_to_proteins(&config)?;
    let baseline = std::fs::read(&output)?;
    config.output.export_peptides = Some(root.join("peptides.csv"));
    run_features_to_proteins(&config)?;
    assert_eq!(baseline, std::fs::read(&output)?);
    assert_tmm_export_rollup(&root)?;
    let expected = std::fs::read(root.join("peptides.csv"))?;
    for method in [QuantMethod::Median, QuantMethod::TopN] {
        config.quantification = method;
        run_features_to_proteins(&config)?;
        assert_eq!(expected, std::fs::read(root.join("peptides.csv"))?);
    }
    Ok(())
}

fn assert_tmm_export_rollup(root: &Path) -> Result<(), Box<dyn Error>> {
    let peptides = read_csv(&root.join("peptides.csv"))?;
    let proteins = read_csv(&root.join("proteins.csv"))?;
    let mut sums = HashMap::<(String, String), f64>::new();
    for row in peptides.rows {
        *sums.entry((row[0].clone(), row[2].clone())).or_default() += row[3].parse::<f64>()?;
    }
    for row in proteins.rows {
        for (sample, value) in proteins.headers.iter().skip(1).zip(row.iter().skip(1)) {
            let expected = value.parse::<f64>()?;
            let actual = sums[&(row[0].clone(), sample.clone())];
            assert!(
                (actual - expected).abs() < 1e-9 * expected.abs(),
                "{} {sample}: {actual} != {expected}",
                row[0]
            );
        }
    }
    Ok(())
}

fn write_tmm_export_features(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut rows = Vec::new();
    let proteins = (0..12)
        .map(|index| format!("P{index:02}"))
        .collect::<Vec<_>>();
    let accessions = proteins
        .iter()
        .map(|protein| [protein.as_str()])
        .collect::<Vec<_>>();
    let sequences = "ACDEFGHIKLMN"
        .chars()
        .map(|letter| [format!("{letter}PEPTIDEAK"), format!("{letter}APEPTIDECK")])
        .collect::<Vec<_>>();
    for (index, peptides) in sequences.iter().enumerate() {
        for (run, scale) in [("run1.raw", 1.0), ("run2.raw", 2.0 + index as f64 / 3.0)] {
            for sequence in peptides {
                for charge in [2, 3] {
                    rows.push(QpxRow::new_with_charge(
                        sequence,
                        run,
                        (100.0 * (index + 1) as f64 * scale * f64::from(charge)) as f32,
                        charge,
                        &accessions[index],
                    ));
                }
            }
        }
    }
    write_qpx_rows(path, &rows)
}

#[test]
fn maxlfq_exports_normalized_species_collapsed_to_canonical() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    let parquet = root.join("features.parquet");
    let sdrf = root.join("runs.sdrf.tsv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1.raw", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1.raw", 300.0, 3, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2.raw", 1000.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2.raw", 100.0, 3, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 2000.0, &["P1"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;
    let mut config = default_sum_config(parquet, sdrf, root.join("proteins.csv"));
    config.quantification = QuantMethod::MaxLfq;
    config.normalization.sample_method = "quantile".to_owned();
    config.output.export_peptides = Some(root.join("normalized.csv"));
    config.output.export_unnormalized_peptides = Some(root.join("unnormalized.csv"));
    run_features_to_proteins(&config)?;
    assert_peptide_exports_align(&root)?;
    let normalized = read_csv(&root.join("normalized.csv"))?;
    let unnormalized = read_csv(&root.join("unnormalized.csv"))?;
    for (row, expected) in normalized.rows.iter().zip([600.0, 1150.0, 1250.0, 700.0]) {
        assert_eq!(row[3].parse::<f64>()?, expected);
    }
    for (row, expected) in unnormalized.rows.iter().zip([200.0, 2000.0, 400.0, 1100.0]) {
        assert_eq!(row[3].parse::<f64>()?, expected);
    }
    Ok(())
}

#[test]
fn peptide_exports_align_before_and_after_sample_normalization() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    let parquet = root.join("features.parquet");
    let sdrf = root.join("runs.sdrf.tsv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 400.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 800.0, &["P1"]),
            QpxRow::new("SINGLETONK", "run1.raw", 10000.0, &["P2"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;
    let mut config = default_sum_config(parquet, sdrf, root.join("proteins.csv"));
    config.output.export_peptides = Some(root.join("normalized.csv"));
    config.output.export_unnormalized_peptides = Some(root.join("unnormalized.csv"));
    for method in [
        "none",
        "globalmedian",
        "conditionmedian",
        "quantile",
        "hierarchical",
    ] {
        config.normalization.sample_method = method.to_owned();
        run_features_to_proteins(&config)?;
        assert_peptide_exports_align(&root)?;
        let table = read_csv(&root.join("unnormalized.csv"))?;
        assert_eq!(table.rows.len(), 4);
        for (row, expected) in table.rows.iter().zip([200.0, 800.0, 100.0, 400.0]) {
            assert_eq!(row[row.len() - 1].parse::<f64>()?, expected);
        }
        if method == "none" {
            assert_eq!(
                std::fs::read(root.join("normalized.csv"))?,
                std::fs::read(root.join("unnormalized.csv"))?
            );
        }
    }
    Ok(())
}

#[test]
fn exports_preserve_maxlfq_proteins_and_de_bytes() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    let parquet = root.join("features.parquet");
    let sdrf = root.join("runs.sdrf.tsv");
    write_export_features(&parquet)?;
    write_deqms_count_sdrf(&sdrf)?;
    for impute in [false, true] {
        let output = root.join("proteins.csv");
        let de_output = root.join("de.csv");
        let mut config = default_sum_config(parquet.clone(), sdrf.clone(), output.clone());
        config.quantification = QuantMethod::MaxLfq;
        config.normalization.run_method = "median".to_owned();
        config.normalization.sample_method = "hierarchical".to_owned();
        config.imputation.enabled = impute;
        config.imputation.method = "minprob".to_owned();
        config.differential_expression = DifferentialExpressionConfig {
            enabled: true,
            contrasts: Some(vec!["groupA vs groupB".to_owned()]),
            method: "limma".to_owned(),
            output: Some(de_output.clone()),
            ..DifferentialExpressionConfig::default()
        };
        run_features_to_proteins(&config)?;
        let proteins = std::fs::read(&output)?;
        let de = std::fs::read(&de_output)?;
        config.output.export_peptides = Some(root.join("normalized.csv"));
        config.output.export_unnormalized_peptides = Some(root.join("unnormalized.csv"));
        config.output.export_unimputed = impute.then(|| root.join("unimputed.csv"));
        run_features_to_proteins(&config)?;
        assert_eq!(proteins, std::fs::read(&output)?);
        assert_eq!(de, std::fs::read(&de_output)?);
        assert_peptide_exports_align(&root)?;
    }
    assert_unimputed_values(&root)
}

fn assert_peptide_exports_align(root: &Path) -> Result<(), Box<dyn Error>> {
    let normalized = read_csv(&root.join("normalized.csv"))?;
    let unnormalized = read_csv(&root.join("unnormalized.csv"))?;
    assert_eq!(normalized.headers, unnormalized.headers);
    assert_eq!(normalized.rows.len(), unnormalized.rows.len());
    for (before, after) in unnormalized.rows.iter().zip(&normalized.rows) {
        assert_eq!(before[..before.len() - 1], after[..after.len() - 1]);
    }
    Ok(())
}

fn assert_unimputed_values(root: &Path) -> Result<(), Box<dyn Error>> {
    let before = read_csv(&root.join("unimputed.csv"))?;
    let after = read_csv(&root.join("proteins.csv"))?;
    assert_eq!(before.headers, after.headers);
    assert_eq!(before.rows.len(), after.rows.len());
    let mut filled = 0;
    for (before, after) in before.rows.iter().zip(&after.rows) {
        for (before, after) in before.iter().zip(after) {
            if before.is_empty() {
                assert!(after.parse::<f64>()?.is_finite());
                filled += 1;
            } else {
                assert_eq!(before, after);
            }
        }
    }
    assert!(filled > 0);
    Ok(())
}

fn write_export_features(path: &Path) -> Result<(), Box<dyn Error>> {
    let sequences = DEQMS_TARGETS
        .iter()
        .map(|(protein, _)| {
            [
                format!("{protein}PEPTIDEAK"),
                format!("{protein}APEPTIDECK"),
            ]
        })
        .collect::<Vec<_>>();
    let accessions = DEQMS_TARGETS
        .iter()
        .map(|(protein, _)| [*protein])
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for (index, (_, targets)) in DEQMS_TARGETS.iter().enumerate() {
        for (run, target) in targets.iter().enumerate() {
            if index == 0 && run == 5 {
                continue;
            }
            for sequence in &sequences[index] {
                rows.push(QpxRow::new(
                    sequence,
                    DEQMS_SAMPLE_RUNS[run],
                    target / 2.0,
                    &accessions[index],
                ));
            }
        }
    }
    write_qpx_rows(path, &rows)
}
