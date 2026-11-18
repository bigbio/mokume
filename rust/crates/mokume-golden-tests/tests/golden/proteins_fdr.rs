use super::*;

fn protein_fdr_rows() -> Vec<QpxRow<'static>> {
    vec![
        // P1: best q 0.005, two peptides in both runs -> kept.
        QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P1"]).with_qvalues(None, Some(0.005)),
        QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]).with_qvalues(None, Some(0.005)),
        QpxRow::new("PEPTIDEAK", "run2.raw", 300.0, &["P1"]).with_qvalues(None, Some(0.005)),
        QpxRow::new("APEPTIDECK", "run2.raw", 400.0, &["P1"]).with_qvalues(None, Some(0.005)),
        // P2: q 0.03 (between 1% and 5%) -> removed by a 1% protein FDR.
        QpxRow::new("PEPTIDEBK", "run1.raw", 500.0, &["P2"]).with_qvalues(None, Some(0.03)),
        QpxRow::new("BPEPTIDECK", "run1.raw", 600.0, &["P2"]).with_qvalues(None, Some(0.03)),
        // P3: q exactly at the threshold -> kept (`<=`).
        QpxRow::new("PEPTIDEDK", "run1.raw", 700.0, &["P3"]).with_qvalues(None, Some(0.01)),
        QpxRow::new("DPEPTIDECK", "run1.raw", 800.0, &["P3"]).with_qvalues(None, Some(0.01)),
        // P4: confident but a single peptide -> removed by --min-unique 2 only.
        QpxRow::new("PEPTIDEEK", "run1.raw", 900.0, &["P4"]).with_qvalues(None, Some(0.001)),
        // P5: group minimum rule -- one row at 0.04, one at 0.008 -> kept.
        QpxRow::new("PEPTIDEFK", "run2.raw", 110.0, &["P5"]).with_qvalues(None, Some(0.04)),
        QpxRow::new("FPEPTIDECK", "run2.raw", 120.0, &["P5"]).with_qvalues(None, Some(0.008)),
        // P6: no protein q-value -> can never pass an explicit protein FDR.
        QpxRow::new("PEPTIDEGK", "run2.raw", 130.0, &["P6"]),
        QpxRow::new("GPEPTIDECK", "run2.raw", 140.0, &["P6"]),
    ]
}

fn protein_names(table: &CsvTable) -> Vec<String> {
    let mut names = table
        .rows
        .iter()
        .filter_map(|row| row.first().cloned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[test]
fn features2proteins_filter_protein_fdr_keeps_groups_at_or_below_threshold()
-> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("proteins.protein-fdr.features.parquet");
    let sdrf = root.join("proteins.protein-fdr.sdrf.tsv");
    write_qpx_rows(&parquet, &protein_fdr_rows())?;
    write_synthetic_sdrf(&sdrf)?;

    // Default (flag off): only --min-unique 2 applies, so P4 is the one loss.
    let unfiltered = root.join("proteins.no-fdr.csv");
    run_features_to_proteins(&default_sum_config(
        parquet.clone(),
        sdrf.clone(),
        unfiltered.clone(),
    ))?;
    assert_eq!(
        protein_names(&read_csv(&unfiltered)?),
        ["P1", "P2", "P3", "P5", "P6"]
    );

    // 1% protein FDR on top of --min-unique 2: P2 (q 0.03) and P6 (no q) go,
    // P3 (q == threshold) and P5 (best row 0.008) stay; P4 still fails min-unique.
    let filtered = root.join("proteins.fdr.csv");
    let mut config = default_sum_config(parquet.clone(), sdrf.clone(), filtered.clone());
    config.protein_fdr_threshold = Some(0.01);
    run_features_to_proteins(&config)?;
    let filtered_table = read_csv(&filtered)?;
    assert_eq!(protein_names(&filtered_table), ["P1", "P3", "P5"]);
    // Kept groups keep every row: P1 intensities are untouched by the filter.
    assert_numeric_cell_close(&filtered_table, "P1", "sample-1", 300.0);
    assert_numeric_cell_close(&filtered_table, "P1", "sample-2", 700.0);

    // The two gates are independent: relaxing --min-unique to 1 brings back the
    // confident single-peptide P4 but never the FDR-failing groups.
    let relaxed = root.join("proteins.fdr-min-unique-1.csv");
    let mut config = default_sum_config(parquet, sdrf, relaxed.clone());
    config.protein_fdr_threshold = Some(0.01);
    config.filtering.min_unique_peptides = 1;
    run_features_to_proteins(&config)?;
    assert_eq!(
        protein_names(&read_csv(&relaxed)?),
        ["P1", "P3", "P4", "P5"]
    );
    Ok(())
}

// quantms leaves `id_run_file_name` null for features transferred by match
// between runs; `TransferredFeatures::Drop` removes them before quantification.
#[test]
fn features2proteins_drops_transferred_features() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("proteins.transferred.features.parquet");
    let sdrf = root.join("proteins.transferred.sdrf.tsv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]),
            QpxRow::new("THIRDPEPK", "run1.raw", 500.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 300.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 400.0, &["P1"]),
            QpxRow::new("THIRDPEPK", "run2.raw", 600.0, &["P1"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let output = root.join("proteins.transferred.csv");
    let mut config = default_sum_config(parquet.clone(), sdrf, output.clone());
    config.input.transferred_features = TransferredFeatures::Drop;
    let Err(error) = run_features_to_proteins(&config) else {
        return Err("drop without id_run_file_name must fail".into());
    };
    assert!(error.to_string().contains("id_run_file_name"), "{error}");

    add_id_run_column(
        &parquet,
        &[
            Some("run1.raw"),
            Some("run1.raw"),
            Some("run1.raw"),
            Some("run2.raw"),
            None,
            Some("run2.raw"),
        ],
    )?;
    run_features_to_proteins(&config)?;
    let dropped = read_csv(&output)?;
    assert_numeric_cell_close(&dropped, "P1", "sample-1", 800.0);
    assert_numeric_cell_close(&dropped, "P1", "sample-2", 300.0 + 600.0);

    config.input.transferred_features = TransferredFeatures::Keep;
    run_features_to_proteins(&config)?;
    assert_numeric_cell_close(&read_csv(&output)?, "P1", "sample-2", 1300.0);
    Ok(())
}

#[test]
fn dropping_all_transferred_features_is_an_explicit_error() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    let parquet = root.join("features.parquet");
    let sdrf = root.join("samples.tsv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]),
        ],
    )?;
    add_id_run_column(&parquet, &[None; 2])?;
    write_synthetic_sdrf(&sdrf)?;
    let mut config = default_sum_config(parquet, sdrf, root.join("proteins.csv"));
    config.input.transferred_features = TransferredFeatures::Drop;
    let Err(error) = run_features_to_proteins(&config) else {
        return Err("dropping every transferred feature must fail explicitly".into());
    };
    assert!(
        error.to_string().contains("removed all QPX features"),
        "{error}"
    );
    Ok(())
}

// A transferred weight keeps match-between-runs features but lets top-N take
// identified peptides before more intense transferred ones, also after a
// dataset normalization rekeys the cells from charge states to peptides.
#[test]
fn features2proteins_top_n_takes_identified_peptides_first() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    let parquet = root.join("features.parquet");
    let sdrf = root.join("samples.tsv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1.raw", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1.raw", 100.0, 3, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2.raw", 100.0, 2, &["P1"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2.raw", 100.0, 3, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 900.0, &["P1"]),
        ],
    )?;
    let identified = [Some("run1.raw"), Some("run1.raw"), Some("run1.raw")];
    let run2 = [Some("run2.raw"), Some("run2.raw"), None];
    add_id_run_column(&parquet, &[identified, run2].concat())?;
    write_synthetic_sdrf(&sdrf)?;
    let output = root.join("proteins.csv");
    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = QuantMethod::TopN;
    config.topn_peptides = 1;
    for (weight, expected) in [(None, 900.0), (Some(0.5), 200.0)] {
        config.input.transferred_weight = weight;
        run_features_to_proteins(&config)?;
        assert_numeric_cell_close(&read_csv(&output)?, "P1", "sample-2", expected);
    }

    config.normalization.sample_method = "hierarchical".to_owned();
    let mut selected = Vec::new();
    for weight in [None, Some(0.5)] {
        config.input.transferred_weight = weight;
        run_features_to_proteins(&config)?;
        selected.push(numeric_cell(&read_csv(&output)?, "P1", "sample-2")?);
    }
    let ratio = selected[1] / selected[0];
    assert!((ratio - 200.0 / 900.0).abs() < 1e-12, "{selected:?}");
    Ok(())
}

// The pair's log2 ratios are 0 and 2, the second transferred: their median
// is 1, and with weight 0.5 the identified ratio carries the weighted median.
#[test]
fn features2proteins_maxlfq_weighs_transferred_ratios() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    let parquet = root.join("features.parquet");
    let sdrf = root.join("samples.tsv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new("PEPTIDEAK", "run2.raw", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run2.raw", 400.0, &["P1"]),
        ],
    )?;
    add_id_run_column(
        &parquet,
        &[Some("run1.raw"), Some("run1.raw"), Some("run2.raw"), None],
    )?;
    write_synthetic_sdrf(&sdrf)?;
    let output = root.join("proteins.csv");
    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = QuantMethod::MaxLfq;
    for (weight, expected) in [(None, 2.0), (Some(0.5), 1.0)] {
        config.input.transferred_weight = weight;
        run_features_to_proteins(&config)?;
        let table = read_csv(&output)?;
        let ratio =
            numeric_cell(&table, "P1", "sample-2")? / numeric_cell(&table, "P1", "sample-1")?;
        assert!((ratio - expected).abs() < 1e-12, "{weight:?}: {ratio}");
    }
    Ok(())
}

#[test]
fn features2proteins_filter_protein_fdr_applies_to_maxlfq() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("proteins.protein-fdr-maxlfq.features.parquet");
    let sdrf = root.join("proteins.protein-fdr-maxlfq.sdrf.tsv");
    let output = root.join("proteins.protein-fdr-maxlfq.csv");
    write_qpx_rows(&parquet, &protein_fdr_rows())?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = QuantMethod::MaxLfq;
    config.protein_fdr_threshold = Some(0.01);
    run_features_to_proteins(&config)?;
    let names = protein_names(&read_csv(&output)?);
    assert!(
        names
            .iter()
            .all(|name| ["P1", "P3", "P5"].contains(&name.as_str()))
    );
    assert!(names.iter().any(|name| name == "P1"));
    Ok(())
}

#[test]
fn features2proteins_filter_protein_fdr_rejects_unpopulated_qvalue() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("proteins.missing-fdr.features.parquet");
    let sdrf = root.join("proteins.missing-fdr.sdrf.tsv");
    let output = root.join("proteins.missing-fdr.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new("PEPTIDEAK", "run1.raw", 100.0, &["P1"]),
            QpxRow::new("APEPTIDECK", "run1.raw", 200.0, &["P1"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.protein_fdr_threshold = Some(0.01);
    let error = match run_features_to_proteins(&config) {
        Ok(()) => return Err("unpopulated protein q-value was accepted".into()),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("requires a populated QPX `pg_global_qvalue` column")
    );
    assert!(!output.exists());
    Ok(())
}
