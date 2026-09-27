//! Run the mokume CLI without the Python wheel (e.g. on a cluster):
//!
//! `cargo run --release -p mokume-command --example mokume -- correct-batches --method lim ...`

fn main() {
    std::process::exit(mokume_command::run_cli_from_args(std::env::args_os()));
}
