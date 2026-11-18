use super::*;

#[test]
fn features2proteins_directlfq_exports_python_shaped_ion_matrix() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
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

    // directlfq 0.3.3 aligns the three paired rows; the median log2 ratio
    // is log2(20), so sample-2 is divided by 20.
    let hierarchical = run_named_cross_distribution_normalization("hierarchical")?;
    assert_numeric_cell_close(&hierarchical, "P60", "sample-1", 10.0);
    assert_numeric_cell_close(&hierarchical, "P61", "sample-2", 20.0);
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

/// Official directlfq 0.3.3 `NormalizationManagerSamples(..., 50)` on the
/// log2 input matrix, exponentiated back. The shift is the median of paired
/// differences, not the difference between the two column medians.
#[test]
fn features2proteins_hierarchical_real_path_matches_directlfq() -> Result<(), Box<dyn Error>> {
    let hierarchical = run_sample_normalization_real_path("hierarchical", QuantMethod::Sum)?;
    let expected = [
        [1000.0, 1086.7540826452525],
        [1499.9999999999998, 1422.4992463893152],
        [2299.9999999999995, 2217.6851605199845],
        [3100.0000000000005, 3489.9826231290645],
        [4399.999999999999, 4232.156142984357],
        [5999.999999999999, 6273.133322586412],
        [8299.999999999996, 7907.682146077251],
        [10999.999999999993, 11220.956788288388],
        [14999.999999999993, 15815.364292154502],
        [20999.999999999993, 20586.47977693851],
        [29000.00000000001, 30482.126708342446],
        [39999.99999999998, 38875.75580194394],
    ];
    for (index, values) in expected.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            assert_numeric_cell_close(
                &hierarchical,
                SAMPLE_NORM_PROTEINS[index],
                &format!("sample-{}", column + 1),
                value,
            );
        }
    }
    Ok(())
}

/// MaxLFQ applies hierarchical normalization to its species traces. With one
/// species per protein those traces are the canonical-peptide cells, so MaxLFQ
/// reproduces the Sum path's directlfq alignment above; without normalization it
/// keeps the raw sample-2 scale.
#[test]
fn features2proteins_maxlfq_hierarchical_aligns_species_traces() -> Result<(), Box<dyn Error>> {
    let sum = run_sample_normalization_real_path("hierarchical", QuantMethod::Sum)?;
    let maxlfq = run_sample_normalization_real_path("hierarchical", QuantMethod::MaxLfq)?;
    let raw = run_sample_normalization_real_path("none", QuantMethod::MaxLfq)?;
    assert_eq!(maxlfq.headers, sum.headers);
    assert_eq!(maxlfq.rows.len(), SAMPLE_NORM_PROTEINS.len());
    let mut shifted = 0;
    for ((normalized, expected), unnormalized) in maxlfq.rows.iter().zip(&sum.rows).zip(&raw.rows) {
        assert_eq!(normalized.first(), expected.first());
        for ((value, target), before) in normalized.iter().zip(expected).zip(unnormalized).skip(1) {
            let (value, target, before) = (
                value.parse::<f64>()?,
                target.parse::<f64>()?,
                before.parse::<f64>()?,
            );
            assert!(
                (value - target).abs() <= 1e-9 * target,
                "{value} != {target}"
            );
            shifted += usize::from((value - before).abs() > 1e-6 * before);
        }
    }
    assert!(
        shifted > 0,
        "hierarchical normalization left MaxLFQ unchanged"
    );
    Ok(())
}

/// The pipeline pivot orders sample names [a,b,c,d], independently of their
/// registration order [c,a,d,b]. Official directlfq 0.3.3 expected values use
/// that name-sorted matrix after f32 quantization at the QPX boundary.
#[test]
fn features2proteins_hierarchical_orders_columns_by_sample_name() -> Result<(), Box<dyn Error>> {
    let hierarchical = run_hier_order_real_path()?;
    let expected = [
        [
            460.00000000000017,
            459.9999916601302,
            459.99997084436984,
            459.9999958080811,
        ],
        [
            690.0000000000001,
            690.00000833987,
            690.0000041699354,
            689.9673608833084,
        ],
        [
            1058.0000000000002,
            1058.0068153416473,
            1058.0157123155623,
            1057.9673734482278,
        ],
        [
            1426.0000000000007,
            1426.0136640427731,
            1425.9921396892423,
            1425.9673860131434,
        ],
        [
            2024.0000000000002,
            2023.9863359572244,
            2024.0078811604344,
            2023.999949718653,
        ],
        [
            2760.0000000000005,
            2760.00003335948,
            2760.0000166797417,
            2759.9999748484897,
        ],
        [
            3818.0,
            3818.0067653024266,
            3818.0155373817815,
            3818.032693546606,
        ],
        [
            5059.999999999999,
            5059.999866562081,
            5059.999966708354,
            5059.99995388889,
        ],
        [
            6899.999999999999,
            6899.999999999999,
            6899.999850085835,
            6899.999937121222,
        ],
        [
            9660.0,
            9659.999866562075,
            9659.999483538526,
            9659.9999119697,
        ],
        [
            13339.999999999995,
            13339.99946624832,
            13340.00001674757,
            13339.999878434359,
        ],
        [
            18399.999999999996,
            18399.999999999996,
            18399.999600228894,
            18399.999832323236,
        ],
    ];
    for (index, values) in expected.iter().enumerate() {
        for (sample, &value) in ["hsample-a", "hsample-b", "hsample-c", "hsample-d"]
            .iter()
            .zip(values)
        {
            assert_numeric_cell_close(&hierarchical, HIER_ORDER_PROTEINS[index], sample, value);
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
fn features2proteins_loess_real_path_matches_python_oracle_within_tolerance()
-> Result<(), Box<dyn Error>> {
    const RELATIVE_TOLERANCE: f64 = 2e-3;
    let loess = run_sample_normalization_real_path("loess", QuantMethod::Sum)?;
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
