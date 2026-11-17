use super::*;

#[test]
fn features2proteins_sums_charge_states_per_sample() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
fn features2proteins_ratio_sdrf_pooled_multiplex_matches_python_oracle()
-> Result<(), Box<dyn Error>> {
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
fn features2proteins_pibaq_default_uses_unique_peptides_and_family_rows()
-> Result<(), Box<dyn Error>> {
    // P5A/P5B share two peptides; each member keeps only its own unique peptide.
    let table = run_family_pibaq_quantification_with(PibaqConfig::default())?;
    assert_numeric_cell_close(&table, "P5A", "sample-1", 300.0);
    assert_numeric_cell_close(&table, "P5B", "sample-1", 100.0);

    // P4A/P4B have no unique peptide: one family row instead of invented member values.
    let synthetic =
        run_synthetic_quantification_with(QuantMethod::Pibaq, 3, PibaqConfig::default())?;
    assert_numeric_cell_close(&synthetic, "P4A;P4B", "sample-1", 15.0);
    assert!(!synthetic.rows.iter().any(|row| {
        row.first()
            .is_some_and(|name| name == "P4A" || name == "P4B")
    }));
    assert_numeric_cell_close(&synthetic, "P1", "sample-1", 3450.0);
    Ok(())
}

#[test]
fn features2proteins_pibaq_stable_ratio_matches_legacy_on_one_anchored_sample()
-> Result<(), Box<dyn Error>> {
    let table = run_family_pibaq_quantification_with(PibaqConfig {
        shared_mode: PibaqSharedMode::StableRatio,
        ..PibaqConfig::default()
    })?;
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

    // Cox Eq. 3 preserves the shared 2x peptide ratio. With external
    // normalization disabled, total input intensity is 1500 => [500, 1000].
    let maxlfq = run_lfq_quantification(QuantMethod::MaxLfq)?;
    assert_numeric_cell_close(&maxlfq, "P6", "sample-1", 500.0);
    assert_numeric_cell_close(&maxlfq, "P6", "sample-2", 1000.0);

    // The legacy flag is a no-op: neither path delegates to DirectLFQ.
    let maxlfq_builtin = run_lfq_quantification_builtin_maxlfq()?;
    assert_numeric_cell_close(&maxlfq_builtin, "P6", "sample-1", 500.0);
    assert_numeric_cell_close(&maxlfq_builtin, "P6", "sample-2", 1000.0);
    Ok(())
}

#[test]
fn features2proteins_maxlfq_preserves_species_after_contextual_maxima() -> Result<(), Box<dyn Error>>
{
    // Keep contextual maxima, then sum repeated observations of each species:
    // sample-1 has intensities [120, 40, 30], sample-2 is exactly 2x.
    // Cox's profile-total rescaling therefore gives [190, 380].
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("maxlfq_dup.features.parquet");
    let sdrf = root.join("maxlfq_dup.sdrf.tsv");
    let default_output = root.join("maxlfq_dup.default.csv");

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

    let mut config = default_sum_config(parquet, sdrf, default_output.clone());
    config.quantification = QuantMethod::MaxLfq;
    run_features_to_proteins(&config)?;
    let default = read_csv(&default_output)?;
    assert_numeric_cell_close(&default, "P6", "sample-1", 190.0);
    assert_numeric_cell_close(&default, "P6", "sample-2", 380.0);
    Ok(())
}

#[test]
fn features2proteins_maxlfq_keeps_charge_specific_ratios() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("species.features.parquet");
    let sdrf = root.join("species.sdrf.tsv");
    let output = root.join("species.proteins.csv");
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1.raw", 1.0, 2, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run1.raw", 1.0, 3, &["P6"]),
            QpxRow::new("ANOTHERAK", "run1.raw", 1.0, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2.raw", 1.0, 2, &["P6"]),
            QpxRow::new_with_charge("PEPTIDEAK", "run2.raw", 9.0, 3, &["P6"]),
            QpxRow::new("ANOTHERAK", "run2.raw", 3.0, &["P6"]),
        ],
    )?;
    write_synthetic_sdrf(&sdrf)?;
    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = QuantMethod::MaxLfq;
    run_features_to_proteins(&config)?;
    let table = read_csv(&output)?;
    // Three species ratios (1, 9, 3) imply 3-fold change with total 16.
    assert_numeric_cell_close(&table, "P6", "sample-1", 4.0);
    assert_numeric_cell_close(&table, "P6", "sample-2", 12.0);
    Ok(())
}

#[test]
fn features2proteins_maxlfq_stabilization_is_opt_in() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("stabilization.features.parquet");
    let sdrf = root.join("stabilization.sdrf.tsv");
    let output = root.join("stabilization.proteins.csv");
    let sequences = (0..20)
        .map(|i| format!("{}K", "A".repeat(i + 7)))
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for (i, sequence) in sequences.iter().enumerate() {
        rows.push(QpxRow::new(sequence, "run1.raw", 10.0, &["P6"]));
        if i < 2 {
            rows.push(QpxRow::new(sequence, "run2.raw", 1.0, &["P6"]));
        }
    }
    write_qpx_rows(&parquet, &rows)?;
    write_synthetic_sdrf(&sdrf)?;
    let mut config = default_sum_config(parquet, sdrf, output.clone());
    config.quantification = QuantMethod::MaxLfq;
    for (enabled, first, second) in [(false, 2020.0 / 11.0, 202.0 / 11.0), (true, 200.0, 2.0)] {
        config.maxlfq.stabilize = enabled;
        run_features_to_proteins(&config)?;
        let table = read_csv(&output)?;
        assert_numeric_cell_close(&table, "P6", "sample-1", first);
        assert_numeric_cell_close(&table, "P6", "sample-2", second);
    }
    Ok(())
}

#[test]
fn features2proteins_maxlfq_is_invariant_to_feature_order() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
        .expect("forward MaxLFQ protein row is missing");
    for sample in ["Sample 1", "Sample 2", "Sample 3"] {
        let column = forward
            .headers
            .iter()
            .position(|header| header == sample)
            .expect("forward MaxLFQ sample column is missing");
        let expected = protein_row[column].parse::<f64>()?;
        assert_numeric_cell_close(&reversed, "A5Z2X5", sample, expected);
    }
    Ok(())
}
