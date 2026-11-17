use super::*;

#[test]
fn features2peptides_disabled_filter_pipeline_is_inert() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

// Oracle (captured from the command above; --skip_normalization, no --sdrf):
//   ProteinName,PeptideCanonical,SampleID,BioReplicate,Condition,NormIntensity
//   P1,APEPTIDECK,run2,1,run2,400.0
//   P1,PEPTIDEAK,run2,1,run2,300.0
//   P1,APEPTIDECK,run1,1,run1,200.0
//   P1,PEPTIDEAK,run1,1,run1,250.0
#[test]
fn features2peptides_skip_normalization_matches_python_oracle() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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

#[test]
fn features2peptides_irs_keys_scales_by_run_identity() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("tmt.irs.run-key.features.parquet");
    let sdrf = root.join("tmt.irs.run-key.sdrf.tsv");
    let output = root.join("tmt.irs.run-key.csv");
    let run1 = "161122_SILAC_Mixture1_01";
    let run2 = "161122_SILAC_Mixture2_01";

    write_qpx_tmt_rows(
        &parquet,
        &[
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: run1,
                channels: &[("TMT126", 100.0), ("TMT127", 1000.0)],
                accessions: &["P1"],
            },
            TmtRow {
                sequence: "PEPTIDEAAK",
                run_file_name: run2,
                channels: &[("TMT126", 400.0), ("TMT127", 2000.0)],
                accessions: &["P1"],
            },
        ],
    )?;
    std::fs::write(
        &sdrf,
        format!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcomment[technical replicate]\tfactor value[group]\n\
             POOLX_ref\trun 1 ref\t{run1}\tTMT126\t1\tA\n\
             POOLX_sample\trun 1 sample\t{run1}\tTMT127\t1\tA\n\
             POOLY_ref\trun 2 ref\t{run2}\tTMT126\t1\tB\n\
             POOLY_sample\trun 2 sample\t{run2}\tTMT127\t1\tB\n"
        ),
    )?;

    let mut config = default_peptides_config(parquet, output.clone());
    config.input.sdrf = Some(sdrf);
    config.filtering.min_unique_peptides = 1;
    config.irs = Some(IrsChannelConfig {
        channel: "TMT126".to_owned(),
        stat: IrsStat::Median,
        scope: IrsScope::Global,
    });
    run_features_to_peptides(&config)?;

    let table = read_csv(&output)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "POOLX_sample", 2500.0)?;
    assert_peptide_cell(&table, "P1", "PEPTIDEAAK", "POOLY_sample", 1250.0)?;
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
    let (_tempdir, root) = temp_root()?;
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
    let (_tempdir, root) = temp_root()?;
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
