use std::{
    collections::{HashMap, HashSet},
    fs::{read_to_string, write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use mokume_command::{run_from_args, run_from_args_with_pibaq_digest};
use mokume_pipeline::{PibaqDigest, PibaqDigestProvenance};

#[test]
fn log_file_option_writes_file_and_preserves_command_errors(
) -> Result<(), Box<dyn std::error::Error>> {
    let (_tempdir, root) = temp_root()?;
    let log_file = root.join("logs").join("mokume.log");
    let error = match run_from_args([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "definitely-missing.feature.parquet",
        "--output",
        "protein.csv",
        "--log-level",
        "info",
        "--log-file",
        path_str(&log_file)?,
    ]) {
        Ok(()) => panic!("command must still fail for a missing input"),
        Err(error) => error,
    };

    let message = error.to_string();
    assert!(
        message.contains("input file does not exist: definitely-missing.feature.parquet"),
        "unexpected error: {message}"
    );
    assert!(
        !message.contains("stage `log-file` is not implemented yet"),
        "log-file option must not block command execution: {message}"
    );
    assert!(
        log_file.exists(),
        "log file was not created: {}",
        log_file.display()
    );

    let fasta = root.join("proteome.fasta");
    let peptides = root.join("peptides.csv");
    let output = root.join("proteins.tsv");
    write(&fasta, ">P1\nPEPTIDEAK\n")?;
    write(
        &peptides,
        "ProteinName,PeptideCanonical,SampleID,Condition,NormIntensity\n\
P1,PEPTIDEAK,S1,A,100.0\n",
    )?;
    run_from_args_with_pibaq_digest(
        [
            "mokume",
            "quantify",
            "peptides2protein",
            "--quant-method",
            "pibaq",
            "--peptides",
            path_str(&peptides)?,
            "--fasta",
            path_str(&fasta)?,
            "--min-aa",
            "1",
            "--max-aa",
            "100",
            "--output",
            path_str(&output)?,
            "--log-level",
            "info",
            "--log-file",
            path_str(&log_file)?,
        ],
        PibaqDigest {
            accession_peptides: HashMap::from([(
                "P1".to_owned(),
                HashSet::from(["PEPTIDEAK".to_owned()]),
            )]),
            provenance: PibaqDigestProvenance {
                pyopenms_version: "3.5.0-test".to_owned(),
                enzyme: "Trypsin".to_owned(),
                catalog_hash: "catalog-test-hash".to_owned(),
                min_aa: 1,
                max_aa: 100,
                missed_cleavages: 0,
            },
        },
    )?;

    let log = read_to_string(&log_file)?;
    for expected in [
        "pyopenms_version=\"3.5.0-test\"",
        "enzyme=\"Trypsin\"",
        "catalog_hash=\"catalog-test-hash\"",
        "min_aa=1",
        "max_aa=100",
        "missed_cleavages=0",
    ] {
        assert!(
            log.contains(expected),
            "missing {expected:?} in log:\n{log}"
        );
    }

    let conflicting_log = root.join("logs").join("conflicting.log");
    let conflict = match run_from_args([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "another-missing.feature.parquet",
        "--output",
        "other.csv",
        "--log-level",
        "warn",
        "--log-file",
        path_str(&conflicting_log)?,
    ]) {
        Ok(()) => panic!("a later logging configuration must not be silently ignored"),
        Err(error) => error,
    };
    assert!(
        conflict
            .to_string()
            .contains("logging is already initialized with log_level=info"),
        "unexpected conflict error: {conflict}"
    );
    assert!(
        !conflicting_log.exists(),
        "conflicting logging configuration must not create {}",
        conflicting_log.display()
    );
    Ok(())
}

fn temp_root() -> Result<(tempfile::TempDir, PathBuf), Box<dyn std::error::Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = tempfile::Builder::new()
        .prefix(&format!("mokume-command-test-{timestamp}-"))
        .tempdir()?;
    let path = directory.path().to_path_buf();
    Ok((directory, path))
}

fn path_str(path: &Path) -> Result<&str, Box<dyn std::error::Error>> {
    path.to_str()
        .ok_or_else(|| format!("path is not UTF-8: {}", path.display()).into())
}
