use super::*;

#[test]
fn features2proteins_sum_matches_synthetic_golden_matrix() -> Result<(), Box<dyn Error>> {
    let table = run_synthetic_quantification(QuantMethod::Sum, 3)?;
    assert_sum_table(&table)
}

#[test]
fn features2proteins_preserves_biological_replicate_peptide_rows() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
            .expect("missing protein matrix value")
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
