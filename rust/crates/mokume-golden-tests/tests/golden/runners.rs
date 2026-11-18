use super::*;

pub(crate) fn run_synthetic_quantification(
    quantification: QuantMethod,
    topn_peptides: usize,
) -> Result<CsvTable, Box<dyn Error>> {
    let pibaq = if quantification == QuantMethod::Pibaq {
        legacy_pibaq_config()
    } else {
        PibaqConfig::default()
    };
    run_synthetic_quantification_with(quantification, topn_peptides, pibaq)
}

pub(crate) fn run_synthetic_quantification_with(
    quantification: QuantMethod,
    topn_peptides: usize,
    pibaq: PibaqConfig,
) -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
            run: None,
            fasta,
            transferred_features: TransferredFeatures::Keep,
            transferred_weight: None,
        },
        output: OutputConfig {
            protein_matrix: output.clone(),
            export_peptides: None,
            export_unnormalized_peptides: None,
            export_unimputed: None,
            export_ions: None,
            export_precision: None,
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
        pibaq,
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig::default(),
        tmt: TmtConfig::default(),
        coverage_threshold: None,
        sample_correlation_threshold: None,
        protein_fdr_threshold: None,
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
pub(crate) fn run_canonical_collapse_quantification(
    quantification: QuantMethod,
    topn_peptides: usize,
) -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_ratio_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
            run: None,
            fasta: None,
            transferred_features: TransferredFeatures::Keep,
            transferred_weight: None,
        },
        output: OutputConfig {
            protein_matrix: output.clone(),
            export_peptides: None,
            export_unnormalized_peptides: None,
            export_unimputed: None,
            export_ions: None,
            export_precision: None,
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
        protein_fdr_threshold: None,
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

pub(crate) fn run_ratio_multiplex_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_family_pibaq_quantification() -> Result<CsvTable, Box<dyn Error>> {
    run_family_pibaq_quantification_with(legacy_pibaq_config())
}

pub(crate) fn run_family_pibaq_quantification_with(
    pibaq: PibaqConfig,
) -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
            run: None,
            fasta: Some(fasta),
            transferred_features: TransferredFeatures::Keep,
            transferred_weight: None,
        },
        output: OutputConfig {
            protein_matrix: output.clone(),
            export_peptides: None,
            export_unnormalized_peptides: None,
            export_unimputed: None,
            export_ions: None,
            export_precision: None,
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
        pibaq,
        directlfq: DirectLfqConfig::default(),
        batch: BatchCorrectionConfig::default(),
        irs: IrsConfig::default(),
        tmt: TmtConfig::default(),
        coverage_threshold: None,
        sample_correlation_threshold: None,
        protein_fdr_threshold: None,
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

pub(crate) fn run_lfq_quantification(
    quantification: QuantMethod,
) -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

/// The compatibility flag must not select a different quantification algorithm.
pub(crate) fn run_lfq_quantification_builtin_maxlfq() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    run_features_to_proteins(&config)?;

    read_csv(&output)
}

pub(crate) fn run_global_median_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_condition_median_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_quantile_sample_normalization() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
pub(crate) fn run_quantile_cross_distribution_normalization() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
pub(crate) fn run_loess_cross_distribution_normalization() -> Result<CsvTable, Box<dyn Error>> {
    run_named_cross_distribution_normalization("loess")
}

pub(crate) fn run_named_cross_distribution_normalization(
    method: &str,
) -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
pub(crate) fn run_rlr_cross_distribution_normalization() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
/// threshold (including LOESS >=10 MA points), exercising
/// the real shift/smoothing math.
pub(crate) const SAMPLE_NORM_PEPTIDES: [&str; 12] = [
    "AAALEPK", "ACDEFGK", "ADEFGHK", "AEFGHIK", "AFGHIKR", "AGHIKLR", "AHIKLMR", "AIKLMNR",
    "AKLMNPR", "ALMNPQR", "AMNPQRK", "ANPQRSK",
];
pub(crate) const SAMPLE_NORM_PROTEINS: [&str; 12] = [
    "P70", "P71", "P72", "P73", "P74", "P75", "P76", "P77", "P78", "P79", "P80", "P81",
];
// Intensities are exact integers (lossless f32<->f64) and every sample-2 /
// sample-1 ratio is distinct with a clear gap between sorted log2 ratios, so the
// trimming/smoothing normalizers fit on unambiguous rows (no argsort tie between
// numpy and the Rust port).
pub(crate) const SAMPLE_NORM_SAMPLE1: [f32; 12] = [
    1000.0, 1500.0, 2300.0, 3100.0, 4400.0, 6000.0, 8300.0, 11000.0, 15000.0, 21000.0, 29000.0,
    40000.0,
];
pub(crate) const SAMPLE_NORM_SAMPLE2: [f32; 12] = [
    1230.0, 1610.0, 2510.0, 3950.0, 4790.0, 7100.0, 8950.0, 12700.0, 17900.0, 23300.0, 34500.0,
    44000.0,
];

/// Run the 12-row two-sample matrix through `features_to_proteins` with the named
/// sample normalization method, returning the protein matrix.
pub(crate) fn run_sample_normalization_real_path(
    method: &str,
    quantification: QuantMethod,
) -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
    config.quantification = quantification;
    // Each protein has one peptide species, so MaxLFQ needs single-species ratios.
    config.maxlfq.min_ratio_count = 1;
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
pub(crate) const CENTER_PEPTIDE_Z2_SAMPLE1: [[f32; 2]; 12] = [
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
pub(crate) const CENTER_PEPTIDE_Z3_SAMPLE1: [[f32; 2]; 12] = [
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
pub(crate) const CENTER_PEPTIDE_Z2_SAMPLE2: [[f32; 2]; 12] = [
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
pub(crate) const CENTER_PEPTIDE_Z3_SAMPLE2: [[f32; 2]; 12] = [
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
pub(crate) fn run_center_real_path(method: &str) -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

// Four-sample fixture with registration order [c,a,d,b] and the pipeline's
// lexical sample order [a,b,c,d]. Official expected values above were computed
// by directlfq 0.3.3 on these f32-quantized intensities in lexical sample order.
pub(crate) const HIER_ORDER_PROTEINS: [&str; 12] = [
    "H90", "H91", "H92", "H93", "H94", "H95", "H96", "H97", "H98", "H99", "H100", "H101",
];
pub(crate) const HIER_ORDER_SAMPLE_A: [f32; 12] = [
    460.0, 690.0, 1058.0, 1426.0, 2024.0, 2760.0, 3818.0, 5060.0, 6900.0, 9660.0, 13340.0, 18400.0,
];
pub(crate) const HIER_ORDER_SAMPLE_B: [f32; 12] = [
    1346.6, 2019.9, 3097.2, 4174.5, 5925.0, 8079.6, 11176.8, 14812.6, 20199.0, 28278.6, 39051.4,
    53864.0,
];
pub(crate) const HIER_ORDER_SAMPLE_C: [f32; 12] = [
    1172.2, 1758.3, 2696.1, 3633.8, 5157.7, 7033.2, 9729.3, 12894.2, 17583.0, 24616.2, 33993.8,
    46888.0,
];
pub(crate) const HIER_ORDER_SAMPLE_D: [f32; 12] = [
    705.5, 1058.2, 1622.6, 2187.0, 3104.2, 4233.0, 5855.7, 7760.5, 10582.5, 14815.5, 20459.5,
    28220.0,
];

/// Run the four-sample hierarchical matrix whose registration order differs from
/// the name order, returning the protein matrix. Rows are emitted run-c, run-a,
/// run-d, run-b so `SampleId` insertion order is [c, a, d, b] while names sort
/// [a, b, c, d].
pub(crate) fn run_hier_order_real_path() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_run_median_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_run_max_min_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_irs_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_normalization_proteins_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn default_sum_config(
    parquet: PathBuf,
    sdrf: PathBuf,
    output: PathBuf,
) -> FeatureToProteinsConfig {
    FeatureToProteinsConfig {
        input: InputConfig {
            parquet: Some(parquet),
            sdrf: Some(sdrf),
            ..InputConfig::default()
        },
        output: OutputConfig {
            protein_matrix: output,
            export_peptides: None,
            export_unnormalized_peptides: None,
            export_unimputed: None,
            export_ions: None,
            export_precision: None,
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
        protein_fdr_threshold: None,
        ratio: RatioConfig::default(),
        imputation: ImputationConfig::default(),
        differential_expression: DifferentialExpressionConfig::default(),
        runtime: RuntimeConfig {
            memory: None,
            threads: Some(24),
        },
    }
}

pub(crate) fn run_coverage_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
            run: None,
            fasta: None,
            transferred_features: TransferredFeatures::Keep,
            transferred_weight: None,
        },
        output: OutputConfig {
            protein_matrix: output.clone(),
            export_peptides: None,
            export_unnormalized_peptides: None,
            export_unimputed: None,
            export_ions: None,
            export_precision: None,
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
        protein_fdr_threshold: None,
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

pub(crate) fn run_mindet_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

pub(crate) fn run_imputation_quantification<F>(
    method: &str,
    configure: F,
) -> Result<CsvTable, Box<dyn Error>>
where
    F: FnOnce(&mut FeatureToProteinsConfig),
{
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join(format!("{method}.features.parquet"));
    let sdrf = root.join(format!("{method}.sdrf.tsv"));
    let output = root.join(format!("{method}.protein.csv"));

    let rows = imputation_qpx_rows(method);
    write_qpx_rows(&parquet, &rows)?;
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

pub(crate) fn imputation_qpx_rows(method: &str) -> Vec<QpxRow<'static>> {
    let mut rows = vec![
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
    ];
    if method == "impseqrob" {
        rows.extend([
            QpxRow::new("NTERMPEPK", "imp1.raw", 20.0, &["P34"]),
            QpxRow::new("CTERMPEPK", "imp1.raw", 20.0, &["P34"]),
            QpxRow::new("NTERMPEPK", "imp2.raw", 200.0, &["P34"]),
            QpxRow::new("CTERMPEPK", "imp2.raw", 200.0, &["P34"]),
            QpxRow::new("NTERMPEPK", "imp3.raw", 200.0, &["P34"]),
            QpxRow::new("CTERMPEPK", "imp3.raw", 200.0, &["P34"]),
        ]);
    }
    rows
}

pub(crate) fn run_seqknn_quantification() -> Result<CsvTable, Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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
