use std::{
    ffi::OsString,
    fs::{create_dir_all, File},
    path::PathBuf,
    sync::Mutex,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use mokume_core::{FeatureToProteinsConfig, MokumeError};
use mokume_pipeline::{
    run_features_to_proteins, run_features_to_proteins_with_pibaq_digest, PibaqDigest,
};
use tracing_subscriber::EnvFilter;

mod correct_batches;
mod features_to_peptides;
mod features_to_proteins;
mod filter_config_examples;
mod h5ad;
mod other_args;
mod parsers;
mod peptides2protein;
mod schema;

pub use schema::{command_schema, command_schema_json, CommandSpec, FlagSpec, ValueArity};

use features_to_peptides::Features2PeptidesArgs;
use features_to_proteins::Features2ProteinsArgs;
use other_args::{CorrectBatchesArgs, Peptides2ProteinArgs};

#[derive(Debug, Parser)]
#[command(name = "mokume")]
#[command(version)]
#[command(about = "Rust-native proteomics quantification toolkit for quantms/QPX data")]
struct Cli {
    #[arg(
        short = 'v',
        long = "log-level",
        value_name = "LEVEL",
        default_value = "debug",
        ignore_case = true,
        global = true
    )]
    log_level: LogLevel,

    #[arg(long = "log-file", value_name = "FILE", global = true)]
    log_file: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LogLevel {
    Debug,
    Info,
    Warn,
}

impl LogLevel {
    const fn as_filter(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
        }
    }
}

/// The FASTA digestion requested by a parsed piBAQ command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PibaqDigestRequest {
    pub fasta: PathBuf,
    pub enzyme: String,
    pub min_aa: usize,
    pub max_aa: usize,
    pub missed_cleavages: usize,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(about = "Build peptide and protein expression matrices")]
    Quantify(QuantifyArgs),

    #[command(name = "correct-batches")]
    #[command(about = "Correct batch effects in protein quantification output")]
    CorrectBatches(Box<CorrectBatchesArgs>),
}

#[derive(Debug, Args)]
struct QuantifyArgs {
    #[command(subcommand)]
    command: QuantifyCommands,
}

#[derive(Debug, Subcommand)]
enum QuantifyCommands {
    #[command(name = "features2proteins")]
    #[command(about = "Build a protein expression matrix")]
    Features2Proteins(Box<Features2ProteinsArgs>),

    #[command(name = "features2peptides")]
    #[command(about = "Convert features to peptide-level output")]
    Features2Peptides(Box<Features2PeptidesArgs>),

    #[command(name = "peptides2protein")]
    #[command(about = "Compute protein quantities from peptide-level input")]
    Peptides2Protein(Box<Peptides2ProteinArgs>),
}

/// Dispatch a fully-built [`Cli`] to its subcommand.
#[allow(dead_code)]
fn dispatch(cli: Cli, pibaq_digest: Option<PibaqDigest>) -> mokume_core::Result<()> {
    init_logging(cli.log_level, cli.log_file).and_then(|()| match cli.command {
        Commands::Quantify(args) => match args.command {
            QuantifyCommands::Features2Proteins(args) => args
                .into_config()
                .and_then(|config| dispatch_features_to_proteins(config, pibaq_digest)),
            QuantifyCommands::Features2Peptides(args) => features_to_peptides::dispatch(&args),
            QuantifyCommands::Peptides2Protein(args) => {
                peptides2protein::run_peptides_to_protein_with_digest(&args, pibaq_digest)
            }
        },
        Commands::CorrectBatches(args) => correct_batches::run_correct_batches(&args),
    })
}

/// Parse an argv vector and return the runtime pyOpenMS digest request when the
/// selected command is piBAQ. Parse/help errors return `None`; the normal CLI
/// entry point remains authoritative for rendering those errors.
pub fn pibaq_digest_request_from_args<I, T>(args: I) -> Option<PibaqDigestRequest>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = Cli::try_parse_from(args).ok()?;
    match cli.command {
        Commands::Quantify(QuantifyArgs {
            command: QuantifyCommands::Features2Proteins(args),
        }) => args.into_pibaq_digest_request(),
        Commands::Quantify(QuantifyArgs {
            command: QuantifyCommands::Peptides2Protein(args),
        }) if args.quant_method.eq_ignore_ascii_case("pibaq") && args.output.is_some() => {
            let (enzyme, min_aa, max_aa) = (
                args.pibaq_enzyme(),
                args.pibaq_min_aa(),
                args.pibaq_max_aa(),
            );
            let fasta = args.fasta?;
            fasta.is_file().then_some(PibaqDigestRequest {
                fasta,
                enzyme,
                min_aa,
                max_aa,
                missed_cleavages: 0,
            })
        }
        _ => None,
    }
}

/// Library entry point used by the internal `mokume-py` PyO3 crate: parse an
/// explicit argument vector and return the dispatch result. A parse error is
/// returned as `Err` rather than exiting the process, so it never tears down a
/// hosting Python interpreter.
pub fn run_from_args<I, T>(args: I) -> mokume_core::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = Cli::try_parse_from(args).map_err(|error| MokumeError::InvalidInput {
        message: error.to_string(),
    })?;
    dispatch(cli, None)
}

/// Parse an argument vector without dispatching or executing the selected command.
pub fn validate_args<I, T>(args: I) -> mokume_core::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = Cli::try_parse_from(args).map_err(|error| MokumeError::InvalidInput {
        message: error.to_string(),
    })?;
    match cli.command {
        Commands::Quantify(args) => match args.command {
            QuantifyCommands::Features2Proteins(args) => args.into_config().map(|_| ()),
            QuantifyCommands::Features2Peptides(args) => {
                features_to_peptides::validate_options(&args)
            }
            QuantifyCommands::Peptides2Protein(args) => peptides2protein::validate_options(&args),
        },
        Commands::CorrectBatches(_) => Ok(()),
    }
}

/// Parse and run a command with a runtime pyOpenMS theoretical-peptide map.
pub fn run_from_args_with_pibaq_digest<I, T>(
    args: I,
    digest: PibaqDigest,
) -> mokume_core::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = Cli::try_parse_from(args).map_err(|error| MokumeError::InvalidInput {
        message: error.to_string(),
    })?;
    dispatch(cli, Some(digest))
}

/// Console-script entry point for the `mokume` wheel: parse an explicit argument
/// vector and return the process exit code WITHOUT calling `process::exit`, so it
/// never tears down a hosting Python interpreter. clap's help/version are printed
/// to stdout (exit 0) and usage errors to stderr (exit 2); a dispatch failure
/// prints the error and returns 1.
pub fn run_cli_from_args<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Cli::try_parse_from(args) {
        Ok(cli) => match dispatch(cli, None) {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("{error}");
                1
            }
        },
        Err(error) => {
            // Reuse clap's own rendering + exit code: help/version go to stdout
            // with code 0, real parse errors to stderr with code 2.
            let _ = error.print();
            error.exit_code()
        }
    }
}

/// Console-script entry point with a runtime pyOpenMS theoretical-peptide map.
pub fn run_cli_from_args_with_pibaq_digest<I, T>(args: I, digest: PibaqDigest) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Cli::try_parse_from(args) {
        Ok(cli) => match dispatch(cli, Some(digest)) {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("{error}");
                1
            }
        },
        Err(error) => {
            let _ = error.print();
            error.exit_code()
        }
    }
}

/// Run `features2proteins`: the Rust pipeline is pure-compute. It writes the
/// protein-matrix CSV and, when `--de-output` is set, one differential-expression
/// result CSV per contrast. Plotting, the interactive HTML report, and the other
/// visualization periphery now live in the Python wheel
/// (`python/mokume/commands/`) and are no longer invoked from here.
fn dispatch_features_to_proteins(
    config: FeatureToProteinsConfig,
    pibaq_digest: Option<PibaqDigest>,
) -> mokume_core::Result<()> {
    match pibaq_digest {
        Some(digest) => run_features_to_proteins_with_pibaq_digest(&config, digest),
        None => run_features_to_proteins(&config),
    }
}

/// Initialize the global tracing subscriber once per process.
///
/// Library entry points may be called repeatedly from one Python process. The
/// same logging configuration reuses the installed subscriber; a different
/// configuration is rejected instead of silently pretending that it took
/// effect.
fn init_logging(level: LogLevel, log_file: Option<PathBuf>) -> mokume_core::Result<()> {
    static CONFIG: Mutex<Option<(LogLevel, Option<PathBuf>)>> = Mutex::new(None);
    let requested = (level, log_file);
    let mut active = CONFIG.lock().map_err(|_| MokumeError::InvalidInput {
        message: "logging configuration lock is poisoned".to_owned(),
    })?;

    if let Some(config) = active.as_ref() {
        if config == &requested {
            return Ok(());
        }
        return Err(MokumeError::InvalidInput {
            message: format!(
                "logging is already initialized with log_level={} and log_file={:?}; \
                 requested log_level={} and log_file={:?}",
                config.0.as_filter(),
                config.1,
                requested.0.as_filter(),
                requested.1,
            ),
        });
    }

    init_logging_once(requested.0, requested.1.clone())?;
    *active = Some(requested);
    Ok(())
}

fn init_logging_once(level: LogLevel, log_file: Option<PathBuf>) -> mokume_core::Result<()> {
    let filter = EnvFilter::new(level.as_filter());

    if let Some(path) = log_file {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).map_err(|source| MokumeError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }
        let file = File::create(&path).map_err(|source| MokumeError::Io {
            path: path.clone(),
            source,
        })?;
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(false)
            .with_writer(Mutex::new(file))
            .try_init()
            .map_err(|error| MokumeError::InvalidInput {
                message: format!("failed to initialize logging: {error}"),
            })?;
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .try_init()
            .map_err(|error| MokumeError::InvalidInput {
                message: format!("failed to initialize logging: {error}"),
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
