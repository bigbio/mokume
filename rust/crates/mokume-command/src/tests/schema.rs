use std::path::Path;

use crate::{command_schema, command_schema_json, validate_args, CommandSpec, FlagSpec};

#[test]
fn schema_lists_only_executable_leaf_commands() {
    let schema = command_schema();
    let paths = schema
        .iter()
        .map(|command| command.path.join(" "))
        .collect::<Vec<_>>();

    assert_eq!(
        paths,
        [
            "quantify features2proteins",
            "quantify features2peptides",
            "quantify peptides2protein",
            "correct-batches",
        ]
    );
    assert!(schema.iter().all(|command| command.help.is_some()));
}

#[test]
fn schema_reflects_clap_flag_contract() {
    let schema = command_schema();
    let command = find_command(&schema, &["quantify", "features2proteins"]);

    let parquet = find_flag(command, "parquet");
    assert_eq!(parquet.long.as_deref(), Some("parquet"));
    assert_eq!(parquet.short, Some('p'));
    assert_eq!(parquet.value_names, ["FILE"]);
    assert_eq!(parquet.value_arity.min, 1);
    assert_eq!(parquet.value_arity.max, Some(1));
    assert!(parquet.conflicts.contains(&"msstats".to_owned()));

    let output = find_flag(command, "output");
    assert!(output.required);

    let normalization = find_flag(command, "run_normalization");
    assert_eq!(normalization.default, Vec::<String>::new());
    assert_eq!(
        normalization.possible_values,
        ["none", "mean", "median", "max", "global", "max-min", "iqr"]
    );

    let contrast = find_flag(command, "de_contrast");
    assert!(contrast.repeat);
    assert_eq!(contrast.value_names, ["GROUP_A", "GROUP_B"]);
    assert_eq!(contrast.value_arity.min, 2);
    assert_eq!(contrast.value_arity.max, Some(2));

    let log_level = find_flag(command, "log_level");
    assert!(log_level.global);
    assert_eq!(log_level.default, ["debug"]);
    assert_eq!(log_level.possible_values, ["debug", "info", "warn"]);
}

#[test]
fn schema_serializes_as_json() {
    let encoded = command_schema_json();
    let Ok(encoded) = encoded else {
        panic!("command schema did not serialize");
    };
    let decoded = serde_json::from_str::<serde_json::Value>(&encoded);
    let Ok(decoded) = decoded else {
        panic!("serialized command schema was not valid JSON");
    };

    assert_eq!(decoded[0]["path"][0], "quantify");
    assert!(decoded[0]["flags"].is_array());
}

#[test]
fn validate_args_parses_without_executing() {
    let output = Path::new("schema-validation-must-not-exist.csv");
    assert!(!output.exists());

    let valid = validate_args([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "missing-but-parseable.parquet",
        "--output",
        output.to_str().unwrap_or_default(),
    ]);
    assert!(valid.is_ok());
    assert!(!output.exists());

    let invalid = validate_args(["mokume", "not-a-command"]);
    assert!(invalid.is_err());

    let semantically_invalid = validate_args([
        "mokume",
        "quantify",
        "peptides2protein",
        "--peptides",
        "missing-but-parseable.csv",
        "--quant-method",
        "sum",
        "--threads",
        "2",
        "--output",
        "unused.tsv",
    ]);
    assert!(semantically_invalid.is_err());

    let export_ions_without_directlfq = validate_args([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "missing-but-parseable.parquet",
        "--output",
        "unused.csv",
        "--quant-method",
        "sum",
        "--export-ions",
        "ions.csv",
    ]);
    assert_eq!(
        export_ions_without_directlfq
            .err()
            .map(|error| error.to_string()),
        Some("invalid input: --export-ions requires --quant-method directlfq".to_owned())
    );
}

fn find_command<'a>(schema: &'a [CommandSpec], path: &[&str]) -> &'a CommandSpec {
    schema
        .iter()
        .find(|command| {
            command
                .path
                .iter()
                .map(String::as_str)
                .eq(path.iter().copied())
        })
        .unwrap_or_else(|| panic!("missing command path: {}", path.join(" ")))
}

fn find_flag<'a>(command: &'a CommandSpec, id: &str) -> &'a FlagSpec {
    command
        .flags
        .iter()
        .find(|flag| flag.id == id)
        .unwrap_or_else(|| panic!("missing flag `{id}` for {}", command.path.join(" ")))
}
