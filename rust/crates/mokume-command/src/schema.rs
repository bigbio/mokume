use clap::{Arg, ArgAction, Command, CommandFactory};
use serde::Serialize;

use crate::Cli;

/// Machine-readable description of one executable leaf command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommandSpec {
    pub path: Vec<String>,
    pub help: Option<String>,
    pub flags: Vec<FlagSpec>,
}

/// Machine-readable description of one command-line flag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FlagSpec {
    pub id: String,
    pub long: Option<String>,
    pub short: Option<char>,
    pub help: Option<String>,
    pub default: Vec<String>,
    pub possible_values: Vec<String>,
    pub required: bool,
    pub repeat: bool,
    pub value_names: Vec<String>,
    pub value_arity: ValueArity,
    pub conflicts: Vec<String>,
    pub global: bool,
}

/// Number of values accepted by one occurrence of a flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ValueArity {
    pub min: usize,
    /// `None` represents an unbounded maximum.
    pub max: Option<usize>,
}

/// Return the schema for every executable Rust-backed leaf command.
pub fn command_schema() -> Vec<CommandSpec> {
    let mut root = Cli::command();
    root.build();

    let mut commands = Vec::new();
    collect_leaf_commands(&root, &mut Vec::new(), &mut commands);
    commands
}

/// Serialize [`command_schema`] for language bindings.
pub fn command_schema_json() -> serde_json::Result<String> {
    serde_json::to_string(&command_schema())
}

fn collect_leaf_commands(
    command: &Command,
    path: &mut Vec<String>,
    commands: &mut Vec<CommandSpec>,
) {
    let subcommands = command
        .get_subcommands()
        .filter(|subcommand| subcommand.get_name() != "help" && !subcommand.is_hide_set())
        .collect::<Vec<_>>();

    if subcommands.is_empty() {
        if !path.is_empty() {
            commands.push(CommandSpec {
                path: path.clone(),
                help: command_help(command),
                flags: command
                    .get_arguments()
                    .filter(|arg| !arg.is_hide_set() && !is_meta_action(arg.get_action()))
                    .map(|arg| flag_spec(command, arg))
                    .collect(),
            });
        }
        return;
    }

    for subcommand in subcommands {
        path.push(subcommand.get_name().to_owned());
        collect_leaf_commands(subcommand, path, commands);
        path.pop();
    }
}

fn command_help(command: &Command) -> Option<String> {
    command
        .get_long_about()
        .or_else(|| command.get_about())
        .map(ToString::to_string)
}

fn flag_spec(command: &Command, arg: &Arg) -> FlagSpec {
    let range = arg.get_num_args();
    let min_values = range.map_or_else(
        || usize::from(arg.get_action().takes_values()),
        |arity| arity.min_values(),
    );
    let max_values = range.map_or(min_values, |arity| arity.max_values());
    let mut conflicts = command
        .get_arg_conflicts_with(arg)
        .into_iter()
        .map(|conflict| conflict.get_id().to_string())
        .collect::<Vec<_>>();
    conflicts.sort();
    conflicts.dedup();

    FlagSpec {
        id: arg.get_id().to_string(),
        long: arg.get_long().map(str::to_owned),
        short: arg.get_short(),
        help: arg
            .get_long_help()
            .or_else(|| arg.get_help())
            .map(ToString::to_string),
        default: arg
            .get_default_values()
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect(),
        possible_values: arg
            .get_possible_values()
            .into_iter()
            .filter(|value| !value.is_hide_set())
            .map(|value| value.get_name().to_owned())
            .collect(),
        required: arg.is_required_set(),
        repeat: matches!(arg.get_action(), ArgAction::Append | ArgAction::Count),
        value_names: arg
            .get_value_names()
            .unwrap_or_default()
            .iter()
            .map(ToString::to_string)
            .collect(),
        value_arity: ValueArity {
            min: min_values,
            max: (max_values != usize::MAX).then_some(max_values),
        },
        conflicts,
        global: arg.is_global_set(),
    }
}

fn is_meta_action(action: &ArgAction) -> bool {
    matches!(
        action,
        ArgAction::Help | ArgAction::HelpShort | ArgAction::HelpLong | ArgAction::Version
    )
}
