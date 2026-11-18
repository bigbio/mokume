use clap::Parser;

use super::config::features_to_proteins_args;
use crate::Cli;

const OPTION_CONFLICTS: &[(&str, &str)] = &[
    (
        "--msstats in.csv -s s.tsv --quant-method ratio",
        "Ratio quantification requires PSM-level QPX input",
    ),
    (
        "--msstats in.csv -s s.tsv --tmt-row-merge best-row",
        "TMT reporter corrections (--tmt-row-merge best-row, --tmt-impurity, --tmt-interference-floor) require QPX input",
    ),
    (
        "--msstats in.csv -s s.tsv --filter-protein-fdr 0.01",
        "--filter-protein-fdr requires QPX --parquet input",
    ),
    (
        "-p in.parquet --psm in.psm.parquet -s s.tsv --quant-method spectral-count --filter-protein-fdr 0.01",
        "--filter-protein-fdr is not supported with --quant-method spectral-count",
    ),
    (
        "--msstats in.csv -s s.tsv --transferred-features drop",
        "--transferred-features requires QPX --parquet feature input",
    ),
    (
        "-p in.parquet --psm in.psm.parquet -s s.tsv --quant-method spectral-count --transferred-features drop",
        "--transferred-features requires QPX --parquet feature input",
    ),
    (
        "-p in.parquet -s s.tsv --normalization-proteins np.txt --sample-normalization quantile",
        "--normalization-proteins requires globalmedian or conditionmedian",
    ),
    (
        "-p in.parquet --quant-method pibaq",
        "piBAQ quantification requires --fasta option",
    ),
    (
        "-p in.parquet --pibaq-max-aa 30",
        "piBAQ FASTA/digestion options require --quant-method pibaq",
    ),
    (
        "-p in.parquet -f db.fasta --quant-method pibaq --pibaq-shared proportional --pibaq-family-rows all",
        "--pibaq-family-rows requires --pibaq-shared unique or stable-ratio",
    ),
    (
        "-p in.parquet --quant-method ratio",
        "Ratio quantification requires --sdrf option",
    ),
    (
        "-p in.parquet --quant-method directlfq --run-normalization median",
        "directlfq manages normalization internally",
    ),
    (
        "-p in.parquet -s s.tsv --batch-correction --batch-method column",
        "Batch correction with method 'column' requires --batch-column option",
    ),
    (
        "-p in.parquet -s s.tsv --batch-correction --batch-column batch",
        "--batch-column requires --batch-method column",
    ),
    (
        "-p in.parquet --batch-correction --batch-covariate sex",
        "Batch correction with --batch-column or --batch-covariate requires --sdrf option",
    ),
    (
        "-p in.parquet --quant-method directlfq --export-peptides peptides.csv",
        "export-peptides is not supported by directlfq quantification",
    ),
    ("-p in.parquet --irs", "IRS options require --sdrf option"),
    (
        "-p in.parquet -s s.tsv --quant-method ratio --irs-plex-column comment[plex]",
        "IRS-only options require --irs",
    ),
    (
        "-p in.parquet -s s.tsv --quant-method ratio --irs-missing-reference drop",
        "IRS-only options require --irs",
    ),
    (
        "-p in.parquet --coverage-threshold 0.5",
        "coverage-threshold requires --sdrf option",
    ),
    (
        "-p in.parquet --de-contrast A B --de-output de.csv",
        "differential expression requires an SDRF file (--sdrf)",
    ),
    (
        "-p in.parquet -s s.tsv --de-contrast A B",
        "differential expression requires --de-output",
    ),
];

fn differential_expression_conflicts() -> [(String, &'static str); 5] {
    const SDRF_DE: &str = "-p in.parquet -s s.tsv --de-contrast A B --de-output de.csv";
    [
        (
            format!("{SDRF_DE} --de-method limma --de-ensemble-method deqms"),
            "--de-ensemble-method only applies to --de-method ensemble",
        ),
        (
            format!("{SDRF_DE} --de-method rots --de-fdr-method ihw"),
            "--de-fdr-method does not apply to rots",
        ),
        (
            format!("{SDRF_DE} --de-method limrots --de-covariate batch"),
            "--de-covariate requires --de-method limma or deqms",
        ),
        (
            format!("{SDRF_DE} --de-method ensemble --de-covariate batch"),
            "--de-covariate requires --de-method limma or deqms",
        ),
        (
            format!("{SDRF_DE} --de-method rots --de-block characteristics[individual]"),
            "--de-block requires --de-method limma or deqms",
        ),
    ]
}

#[test]
fn rejects_option_combinations_before_running() {
    let de_conflicts = differential_expression_conflicts();
    let cases = OPTION_CONFLICTS.iter().copied().chain(
        de_conflicts
            .iter()
            .map(|(options, expected)| (options.as_str(), *expected)),
    );
    for (options, expected) in cases {
        let argv = [
            "mokume",
            "quantify",
            "features2proteins",
            "-o",
            "protein.csv",
        ]
        .into_iter()
        .chain(options.split_whitespace());
        let args = features_to_proteins_args(Cli::parse_from(argv));
        let Err(error) = args.into_config() else {
            panic!("{options} was accepted");
        };
        assert!(error.to_string().contains(expected), "{options}: {error}");
    }
}
