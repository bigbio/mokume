use super::*;

// Opt-in filter pipeline (`--filter-charge-states 2 --filter-max-missed-cleavages 0`,
// --skip_normalization, no --sdrf). The per-row filters run during ingest, before
// the per-(protein, sample) unique-peptide gate, matching Python's chain order:
//   - charge filter [2] drops PEPTIDEAK z3 (150) -> P1/PEPTIDEAK/run1 = 100 (not 250),
//     and drops SHORTBBK z9 -> P3 keeps only SHORTAAK (1 canonical).
//   - missed-cleavage filter (max 0) drops PEPTKIDEAK (1 miss) -> P4 keeps only CLEANAK.
//   - P2 (ONEPEPTK), P3, P4 each fall below min_unique=2 after row filtering and are
//     dropped -- proving the filters apply before the unique gate. Only P1 survives.
#[test]
fn features2peptides_filter_pipeline_applies_per_row_before_unique_gate()
-> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
fn features2peptides_exclude_sequence_patterns_drops_matching_canonicals()
-> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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

// The CV pre-pass keys a group by its raw name, while ingest renames a mixed
// `CONTAM_<ACC>;<ACC>` group to `<ACC>`; the high-CV canonical must still drop.
#[test]
fn features2peptides_cv_filter_drops_high_cv_canonical_in_contaminant_twin_group()
-> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("peptides.cv-twin.features.parquet");
    let output = root.join("peptides.cv-twin.csv");
    let group = ["CONTAM_P05787", "P05787"];
    write_qpx_rows(
        &parquet,
        &[
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &group),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 120.0, 3, &group),
            QpxRow::new_with_charge("PEPTIDEAK", "run1", 140.0, 4, &group),
            QpxRow::new_with_charge("PEPTIDEBK", "run1", 10.0, 2, &group),
            QpxRow::new_with_charge("PEPTIDEBK", "run1", 990.0, 3, &group),
            QpxRow::new_with_charge("PEPTIDECK", "run1", 300.0, 2, &group),
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
        "CV 0.2 drops PEPTIDEBK from the renamed twin group:\n{:#?}",
        table.rows
    );
    assert_peptide_cell(&table, "P05787", "PEPTIDEAK", "run1", 360.0)?;
    assert_peptide_cell(&table, "P05787", "PEPTIDECK", "run1", 300.0)?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
