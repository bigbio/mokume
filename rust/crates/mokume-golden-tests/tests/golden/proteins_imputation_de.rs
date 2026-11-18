use super::*;

#[test]
fn features2proteins_mindet_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_mindet_quantification()?;

    assert_numeric_cell_close(&table, "P11", "sample-2", 1000.0);
    assert_numeric_cell_close(&table, "P12", "sample-1", 400.0);
    Ok(())
}

#[test]
fn features2proteins_deterministic_imputation_methods_match_synthetic_oracles()
-> Result<(), Box<dyn Error>> {
    let minprob = run_imputation_quantification("minprob", |config| {
        config.imputation.quantile = 0.0;
    })?;
    // imputeLCMD 2.1 with shared seed-42 standard-normal draws. The public
    // method is stochastic; this pins its fitted SD and pipeline log/exp scale.
    assert_numeric_cell_close(&minprob, "P31", "sample-2", 2.0f64.powf(2.8123942601584764));
    assert_numeric_cell_close(&minprob, "P32", "sample-1", 2.0f64.powf(2.6070930014911515));
    assert_numeric_cell_close(&minprob, "P32", "sample-3", 2.0f64.powf(3.4811671996414963));

    // mean/median/constant are Rust-only extras (no Python counterpart). In
    // log2 space mean becomes the geometric mean and constant fills 2**0 = 1.
    let mean = run_imputation_quantification("mean", |_| {})?;
    assert_numeric_cell_close(
        &mean,
        "P31",
        "sample-2",
        (20.0f64 * 200.0 * 1000.0).powf(1.0 / 3.0),
    );
    assert_numeric_cell_close(
        &mean,
        "P32",
        "sample-1",
        (10.0f64 * 100.0 * 1000.0).powf(1.0 / 3.0),
    );
    assert_numeric_cell_close(
        &mean,
        "P32",
        "sample-3",
        (30.0f64 * 300.0 * 1000.0).powf(1.0 / 3.0),
    );

    // Median is invariant under the monotone log2 transform.
    let median = run_imputation_quantification("median", |_| {})?;
    assert_numeric_cell_close(&median, "P31", "sample-2", 200.0);
    assert_numeric_cell_close(&median, "P32", "sample-1", 100.0);
    assert_numeric_cell_close(&median, "P32", "sample-3", 300.0);

    let constant = run_imputation_quantification("constant", |_| {})?;
    assert_numeric_cell_close(&constant, "P31", "sample-2", 1.0);
    assert_numeric_cell_close(&constant, "P32", "sample-1", 1.0);
    assert_numeric_cell_close(&constant, "P32", "sample-3", 1.0);

    let zero = run_imputation_quantification("zero", |_| {})?;
    assert_numeric_cell_close(&zero, "P31", "sample-2", 1.0);
    assert_numeric_cell_close(&zero, "P32", "sample-1", 1.0);
    assert_numeric_cell_close(&zero, "P32", "sample-3", 1.0);

    // KNN (k=2) over nan-euclidean distances in log2 space matches the Python
    // reference (sklearn KNNImputer on the log2 matrix, then 2**) cell-for-cell.
    // Averaging two donors in log2 space is the geometric mean: e.g. P31@S2 =
    // 2**((log2 100 + log2 200)/2) = sqrt(100*200) = sqrt(20000).
    let knn = run_imputation_quantification("knn", |config| {
        config.imputation.n_neighbors = 2;
    })?;
    assert_numeric_cell_close(&knn, "P31", "sample-2", 20000.0f64.sqrt());
    assert_numeric_cell_close(&knn, "P32", "sample-1", 100.0);
    assert_numeric_cell_close(&knn, "P32", "sample-3", 30000.0f64.sqrt());
    Ok(())
}

#[test]
fn features2proteins_seqknn_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_seqknn_quantification()?;

    assert_numeric_cell_close(&table, "P62", "sample-2", 20.0);
    assert_numeric_cell_close(&table, "P63", "sample-3", 32.0);
    Ok(())
}

// rrcovNA 0.5-3: two complete rows trigger covariance + 0.01 I,
// followed by sequential conditional means, not column-mean imputation.
#[test]
fn features2proteins_impseq_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let impseq = run_imputation_quantification("impseq", |_| {})?;
    assert_numeric_cell_close(&impseq, "P32", "sample-1", 2.0f64.powf(7.102566092730076));
    assert_numeric_cell_close(&impseq, "P31", "sample-2", 2.0f64.powf(7.468237752931963));
    assert_numeric_cell_close(&impseq, "P32", "sample-3", 2.0f64.powf(8.053970647172795));
    assert_numeric_cell_close(&impseq, "P31", "sample-1", 100.0);
    assert_numeric_cell_close(&impseq, "P32", "sample-2", 200.0);
    Ok(())
}

// GMS fills the lower mixture component's mean, as the pure-Python impute_gms
// does. For this 3-observations-per-column matrix the lower component collapses
// onto the minimum observation, and every fill lies in [min_observed,
// geometric_mean_observed].
#[test]
fn features2proteins_gms_imputation_matches_stochastic_oracle_property()
-> Result<(), Box<dyn Error>> {
    let gms = run_imputation_quantification("gms", |_| {})?;
    assert_numeric_cell_close(&gms, "P31", "sample-2", 20.0);
    assert_numeric_cell_close(&gms, "P32", "sample-1", 10.0);
    assert_numeric_cell_close(&gms, "P32", "sample-3", 30.0);
    assert_gms_in_band(
        &gms,
        "P31",
        "sample-2",
        20.0,
        geometric_mean(&[20.0, 200.0, 1000.0]),
    );
    assert_gms_in_band(
        &gms,
        "P32",
        "sample-1",
        10.0,
        geometric_mean(&[10.0, 100.0, 1000.0]),
    );
    assert_gms_in_band(
        &gms,
        "P32",
        "sample-3",
        30.0,
        geometric_mean(&[30.0, 300.0, 1000.0]),
    );
    Ok(())
}

fn geometric_mean(values: &[f64]) -> f64 {
    let sum_log = values.iter().map(|value| value.ln()).sum::<f64>();
    (sum_log / values.len() as f64).exp()
}

fn assert_gms_in_band(table: &CsvTable, protein: &str, sample: &str, lower: f64, upper: f64) {
    let Some(sample_index) = table.headers.iter().position(|header| header == sample) else {
        panic!("sample column is missing: {sample}");
    };
    let Some(row) = table
        .rows
        .iter()
        .find(|row| row.first().is_some_and(|value| value == protein))
    else {
        panic!("protein row is missing: {protein}");
    };
    let Some(cell) = row.get(sample_index) else {
        panic!("sample column {sample} is missing from row {protein}");
    };
    let Ok(actual) = cell.parse::<f64>() else {
        panic!("numeric cell is not a valid float: {cell}");
    };
    assert!(
        actual >= lower - 1e-9 && actual <= upper + 1e-9,
        "GMS fill {actual} for {protein}@{sample} outside band [{lower}, {upper}]"
    );
}

// pcaMethods 2.2.0 pca(method="bpca", nPcs=2, maxSteps=100,
// threshold=1e-4, center=TRUE, scale="none") on the log2 matrix.
#[test]
fn features2proteins_bpca_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_imputation_quantification("bpca", |_| {})?;
    assert_numeric_cell_close(&table, "P31", "sample-2", 2.0f64.powf(7.522275951523529));
    assert_numeric_cell_close(&table, "P32", "sample-1", 2.0f64.powf(7.014180587257054));
    assert_numeric_cell_close(&table, "P32", "sample-3", 2.0f64.powf(7.986058602267095));
    assert_numeric_cell_close(&table, "P30", "sample-1", 10.0);
    assert_numeric_cell_close(&table, "P33", "sample-2", 1000.0);
    Ok(())
}

// rrcovNA 0.5-3 / norm 1.0-11.1, with a third complete row so the
// initialized support leaves one row for the official sequential loop.
#[test]
fn features2proteins_impseqrob_imputation_matches_synthetic_oracle() -> Result<(), Box<dyn Error>> {
    let table = run_imputation_quantification("impseqrob", |_| {})?;
    assert_imputation_log2_close(&table, "P31", "sample-2", 8.058466477798845);
    assert_imputation_log2_close(&table, "P32", "sample-1", 6.209673798985398);
    assert_imputation_log2_close(&table, "P32", "sample-3", 7.844203324508301);
    assert_numeric_cell_close(&table, "P30", "sample-1", 10.0);
    assert_numeric_cell_close(&table, "P33", "sample-2", 1000.0);
    Ok(())
}

// Use the same predeclared log2 atol as the imputation-crate official fixtures.
fn assert_imputation_log2_close(table: &CsvTable, protein: &str, sample: &str, expected: f64) {
    let Some(column) = table.headers.iter().position(|header| header == sample) else {
        panic!("missing sample {sample}");
    };
    let Some(row) = table
        .rows
        .iter()
        .find(|row| row.first().is_some_and(|id| id == protein))
    else {
        panic!("missing protein {protein}");
    };
    let Ok(actual) = row[column].parse::<f64>() else {
        panic!("nonnumeric abundance");
    };
    let error = (actual.log2() - expected).abs();
    assert!(
        error <= 1e-8,
        "{protein}/{sample}: absolute log2 error {error}"
    );
}

// imputeLCMD 2.1 QRILC with shared seed-42 uniforms. Its fitted truncation
// limit may exceed the observed minimum; the old minimum-SD oracle was wrong.
#[test]
fn features2proteins_qrilc_imputation_matches_official_shared_draws() -> Result<(), Box<dyn Error>>
{
    let qrilc = run_imputation_quantification("qrilc", |_| {})?;
    assert_numeric_cell_close(&qrilc, "P32", "sample-1", 2.0f64.powf(1.6186427548835307));
    assert_numeric_cell_close(&qrilc, "P31", "sample-2", 2.0f64.powf(4.730141539459813));
    assert_numeric_cell_close(&qrilc, "P32", "sample-3", 2.0f64.powf(3.7726840325126347));
    assert_numeric_cell_close(&qrilc, "P30", "sample-1", 10.0);
    Ok(())
}

// ===========================================================================
// features2proteins --de-method deqms: PER-PROTEIN PEPTIDE-COUNT wiring.
//
// This is the pipeline-level proof that the DEqMS DE path consumes the real
// per-sample unique-canonical-peptide counts (captured at ingest; the smallest
// over the contrast's samples, plus 1) rather than the all-ones fallback. The
// synthetic parquet has 12 proteins, each with a
// DISTINCT number of unique peptide sequences (1,2,3,4,5,6,7,8,9,11,13,15), so
// the spectraCounteBayes count moderation genuinely runs (>= 10 testable
// proteins and a real log2(count) spread), NOT the limma fallback.
//
// Each protein's per-(sample) protein-level Sum is fixed to a controlled target:
// one "anchor" peptide carries `target - (count-1)` and the remaining `count-1`
// peptides each carry 1.0, so the Sum rollup recovers the integer target exactly
// (f32-exact for these magnitudes). The log2 protein matrix is therefore a clean,
// known matrix; the only thing the count vector changes is the deqms moderation.
//
// Independent Bioconductor DEqMS 1.28.0, R 4.5.3: lmFit(log2(DEQMS_TARGETS),
// ~0+group), contrast A-B, eBayes, then spectraCounteBayes(fit.method="loess")
// with DEQMS_COUNTS + 1 (every peptide is in every sample, so the smallest
// per-sample count is the total). This uses the official Gaussian quadratic
// LOESS surface.
// Tuple: (protein, log2FC, sca_t, sca_pvalue, BH, peptide_count).
const DEQMS_PIPELINE_ORACLE: &[(&str, f64, f64, f64, f64, usize)] = &[
    (
        "PROT05",
        -0.9999999999999982,
        -109.51940472414849,
        7.302063232188464e-12,
        8.762475878626156e-11,
        9,
    ),
    (
        "PROT09",
        -0.9999999999999991,
        -47.323521066550306,
        1.6951257200029422e-09,
        1.0170754320017653e-08,
        12,
    ),
    (
        "PROT08",
        -1.0000000000000018,
        -38.80926849955069,
        6.128701301104685e-09,
        2.451480520441874e-08,
        8,
    ),
    (
        "PROT07",
        1.0,
        27.116203656122753,
        6.221031193079203e-08,
        1.8663093579237608e-07,
        5,
    ),
    (
        "PROT04",
        -1.0,
        -21.580991911897772,
        2.704484155453108e-07,
        6.49076197308746e-07,
        6,
    ),
    (
        "PROT10",
        0.9999999999999964,
        19.869256756540995,
        4.595047291347392e-07,
        9.190094582694784e-07,
        7,
    ),
    (
        "PROT02",
        2.0090616097754097,
        17.152970197769925,
        1.1759423721963774e-06,
        2.015901209479504e-06,
        3,
    ),
    (
        "PROT01",
        -2.00048684328549,
        -15.20198554297494,
        2.534570967347522e-06,
        3.801856451021283e-06,
        2,
    ),
    (
        "PROT12",
        -0.9999999999999982,
        -8.981209839393513,
        6.72251409468691e-05,
        8.963352126249214e-05,
        16,
    ),
    (
        "PROT03",
        0.015910691677653688,
        0.3338625791472756,
        0.7489935554533345,
        0.8463792080005441,
        4,
    ),
    (
        "PROT06",
        0.09175595082658283,
        0.29684043988169595,
        0.7758476073338322,
        0.8463792080005441,
        14,
    ),
    (
        "PROT11",
        -0.0016644556595100113,
        -0.1302860212054248,
        0.9002796786684337,
        0.9002796786684337,
        10,
    ),
];

const DEQMS_PIPELINE_RANK: &[&str] = &[
    "PROT05", "PROT09", "PROT08", "PROT07", "PROT04", "PROT10", "PROT02", "PROT01", "PROT12",
    "PROT03", "PROT06", "PROT11",
];

// Independent limma eBayes t statistics on the same matrix. Used only to
// verify that variable peptide counts exercise DEqMS rather than the fallback.
const DEQMS_FALLBACK_SCA_T: &[(&str, f64)] = &[
    ("PROT05", -65.11281128871813),
    ("PROT09", -46.680518447285614),
    ("PROT08", -33.83870290280209),
    ("PROT07", 31.919154914696882),
    ("PROT04", -22.731494984546792),
    ("PROT10", 18.515549252570654),
    ("PROT02", 16.53307015530404),
    ("PROT01", -17.31471545499555),
    ("PROT12", -16.76645805143682),
    ("PROT03", 0.375435097346578),
    ("PROT06", 0.2646723684393925),
    ("PROT11", -0.09359488617639883),
];

#[test]
fn features2proteins_deqms_wires_per_protein_peptide_counts() -> Result<(), Box<dyn Error>> {
    let (_tempdir, root) = temp_root()?;
    create_dir_all(&root)?;
    let parquet = root.join("deqms.features.parquet");
    let sdrf = root.join("deqms.sdrf.tsv");
    let output = root.join("deqms.protein.csv");
    let de_output = root.join("deqms.de.csv");

    write_deqms_count_features(&parquet)?;
    write_deqms_count_sdrf(&sdrf)?;

    let mut config = default_sum_config(parquet, sdrf, output);
    // One unique peptide is enough to keep every protein (PROT01 has a single
    // sequence); the count spread is what drives the deqms moderation.
    config.filtering.min_unique_peptides = 1;
    config.differential_expression = DifferentialExpressionConfig {
        enabled: true,
        contrasts: Some(vec!["groupA vs groupB".to_owned()]),
        method: "deqms".to_owned(),
        output: Some(de_output.clone()),
        ..DifferentialExpressionConfig::default()
    };
    run_features_to_proteins(&config)?;

    let table = read_csv(&de_output)?;
    assert_eq!(
        table.headers,
        vec![
            "ProteinName",
            "log2FC",
            "pvalue",
            "adj_pvalue",
            "sca_t",
            "sca_pvalue",
            "sca_adj_pvalue",
            "mean_groupA",
            "mean_groupB",
            "n_a",
            "n_b",
            "peptide_count",
            "log_pvalue",
            "significance",
        ]
    );
    assert_eq!(table.rows.len(), DEQMS_PIPELINE_ORACLE.len());

    assert_deqms_oracle_rows(&table)?;
    assert_deqms_rank(&table)?;
    assert_deqms_differs_from_fallback(&table)?;
    Ok(())
}

/// Assert every protein row matches independent official DEqMS. Field
/// layout: ProteinName(0), log2FC(1), pvalue(2), adj_pvalue(3), sca_t(4),
/// sca_pvalue(5), sca_adj_pvalue(6), mean_A(7), mean_B(8), n_a(9), n_b(10),
/// peptide_count(11), log_pvalue(12), significance(13).
fn assert_deqms_oracle_rows(table: &CsvTable) -> Result<(), Box<dyn Error>> {
    for &(protein, log2fc, sca_t, sca_p, bh, count) in DEQMS_PIPELINE_ORACLE {
        let row = find_de_row(table, protein)?;
        // log2FC is the untouched limma coefficient: cell-exact (1e-9).
        assert_de_cell_abs(row, 1, log2fc, 1e-9, protein, "log2FC")?;
        // Original official audit tolerances: relative 1e-6, absolute 1e-10.
        assert_de_cell_rel(row, 4, sca_t, 1e-6, 1e-10, protein, "sca_t")?;
        assert_de_cell_rel(row, 5, sca_p, 1e-6, 1e-10, protein, "sca_pvalue")?;
        assert_de_cell_rel(row, 6, bh, 1e-6, 1e-10, protein, "sca_adj_pvalue")?;
        // Public p-value columns must preserve the count-aware results.
        assert_de_columns_equal(row, 2, 5, protein, "pvalue==sca_pvalue")?;
        assert_de_columns_equal(row, 3, 6, protein, "adj_pvalue==sca_adj_pvalue")?;
        // peptide_count is the smallest per-sample unique-sequence count plus 1,
        // the count the DEqMS vignette and OpDEA pass to spectraCounteBayes.
        let actual_count = de_field(row, 11)?;
        assert_eq!(
            actual_count,
            count.to_string(),
            "peptide_count for {protein}: got {actual_count}, expected {count}"
        );
    }
    Ok(())
}

/// Assert the emitted adj-p rank matches the count-aware oracle exactly (the
/// count moderation changes the statistics relative to the limma fallback).
fn assert_deqms_rank(table: &CsvTable) -> Result<(), Box<dyn Error>> {
    let rank = table
        .rows
        .iter()
        .map(|row| {
            row.first()
                .cloned()
                .ok_or_else(|| -> Box<dyn Error> { "empty deqms row".into() })
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        rank, DEQMS_PIPELINE_RANK,
        "deqms adj-p rank must match the independent official oracle"
    );
    Ok(())
}

/// Non-vacuity: the emitted sca_t must differ materially from the all-ones limma
/// fallback, proving the count path is genuinely exercised (the old all-ones code
/// could not have produced these numbers). PROT05 shifts by more than 43.
fn assert_deqms_differs_from_fallback(table: &CsvTable) -> Result<(), Box<dyn Error>> {
    let mut max_fallback_diff = 0.0_f64;
    for &(protein, fallback_t) in DEQMS_FALLBACK_SCA_T {
        let row = find_de_row(table, protein)?;
        let actual_t = de_field(row, 4)?.parse::<f64>()?;
        max_fallback_diff = max_fallback_diff.max((actual_t - fallback_t).abs());
    }
    assert!(
        max_fallback_diff > 5.0,
        "deqms sca_t too close to the all-ones limma fallback (max diff {max_fallback_diff}); \
         the count path is not being exercised"
    );
    Ok(())
}

/// Find a DE result row by protein name, erroring if it is missing.
fn find_de_row<'a>(table: &'a CsvTable, protein: &str) -> Result<&'a [String], Box<dyn Error>> {
    table
        .rows
        .iter()
        .find(|row| row.first().is_some_and(|value| value == protein))
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing deqms row {protein}").into())
}

/// Look up a field of a DE CSV row by index.
fn de_field(row: &[String], index: usize) -> Result<&str, Box<dyn Error>> {
    row.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("DE row field {index} is missing").into())
}

/// Assert a DE cell matches `expected` at an absolute tolerance.
fn assert_de_cell_abs(
    row: &[String],
    index: usize,
    expected: f64,
    tolerance: f64,
    protein: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let actual = de_field(row, index)?.parse::<f64>()?;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{protein}.{label}: got {actual}, expected {expected} (abs tol {tolerance})"
    );
    Ok(())
}

/// Assert a DE cell matches `expected` at a relative tolerance with an absolute
/// floor (so tiny / near-zero magnitudes are not over-constrained).
fn assert_de_cell_rel(
    row: &[String],
    index: usize,
    expected: f64,
    relative: f64,
    absolute_floor: f64,
    protein: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let actual = de_field(row, index)?.parse::<f64>()?;
    let tolerance = (expected.abs() * relative).max(absolute_floor);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{protein}.{label}: got {actual}, expected {expected} (tol {tolerance})"
    );
    Ok(())
}

/// Assert two DE cells are bit-identical (used for `pvalue == sca_pvalue`).
fn assert_de_columns_equal(
    row: &[String],
    left: usize,
    right: usize,
    protein: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let left_value = de_field(row, left)?;
    let right_value = de_field(row, right)?;
    assert_eq!(
        left_value, right_value,
        "{protein}.{label}: columns {left} and {right} differ ({left_value} vs {right_value})"
    );
    Ok(())
}
