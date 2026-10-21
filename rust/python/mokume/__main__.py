"""Unified entry point for the Rust-backed ``mokume`` wheel."""

import argparse
import importlib
import sys

from mokume.core.logger import configure_logging


_PERIPHERY_COMMANDS = {
    ("tissuemap",): ("tissuemap", None),
    ("interactive-report",): ("interactive_report", None),
    ("plot", "tsne"): ("visualize", None),
    ("plot", "pca"): ("de_plots", "pca"),
    ("plot", "de"): ("de_plots", "de"),
}

_ROOT_HELP = """Mokume proteomics quantification and analysis toolkit

Usage:
  mokume [GLOBAL OPTIONS] <COMMAND> [COMMAND OPTIONS]

Commands:
  quantify            Build peptide and protein expression matrices
  correct-batches     Correct batch effects in protein quantification output
  tissuemap           Build a tissue proteome atlas [requires: mokume[tissuemap]]
  plot                Render PCA, t-SNE, and DE plots [requires: mokume[analysis]]
  interactive-report  Build a DE HTML report [requires: mokume[analysis]]
  help                Print root or command-specific help

Global options:
  -v, --log-level <LEVEL>  [default: debug] [possible values: debug, info, warn]
      --log-file <FILE>
  -h, --help               Print help
  -V, --version            Print version

Run `mokume help <COMMAND> [SUBCOMMAND]` for command-specific options.
"""

_PLOT_HELP = """Render proteomics visualizations

Usage:
  mokume plot <COMMAND> [COMMAND OPTIONS]

Commands:
  pca   Render a PCA-by-condition plot
  tsne  Render a t-SNE plot from protein tables
  de    Render volcano plots and DE heatmaps

Run `mokume help plot <COMMAND>` for command-specific options.
"""


def _print_root_help(stream):
    print(_ROOT_HELP, file=stream, end="")


def _periphery_request(args):
    requested = args[1:] if args[:1] == ["help"] else args
    help_requested = args[:1] == ["help"]
    for path in sorted(_PERIPHERY_COMMANDS, key=len, reverse=True):
        if tuple(requested[: len(path)]) == path:
            command_args = requested[len(path) :]
            if help_requested:
                command_args = ["--help"]
            return _PERIPHERY_COMMANDS[path], command_args
    return None


def _extract_global_options(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-v", "--log-level", choices=("debug", "info", "warn"), default="debug"
    )
    parser.add_argument("--log-file")
    parser.add_argument("-V", "--version", action="store_true")
    return parser.parse_known_args(args)


def _run_periphery(command, args, global_options):
    _configure_global_logging(global_options)
    module_name, mode = command
    module = importlib.import_module(f"mokume.commands.{module_name}")
    command_main = getattr(module, "main")
    result = command_main(args, mode=mode) if mode is not None else command_main(args)
    return 0 if result is None else result


def _configure_global_logging(global_options):
    level = (
        "warning" if global_options.log_level == "warn" else global_options.log_level
    )
    configure_logging(level=level, log_file=global_options.log_file)


def _render_requested_pibaq_qc(args, package):
    """Render the QC PDF after the native command has written its piBAQ table."""
    try:
        command_index = args.index("peptides2protein")
    except ValueError:
        return
    command_args = args[command_index + 1 :]
    qc_requested = any(
        option == "--qc-report" or option.startswith("--qc-report=")
        for option in command_args
    )
    if not qc_requested:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--quant-method", default="pibaq")
    parser.add_argument("--qc-report", required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--tpa", action="store_true")
    parser.add_argument("--ruler", action="store_true")
    parsed, _ = parser.parse_known_args(command_args)
    if parsed.quant_method.lower() != "pibaq":
        return
    package.peptides2protein_qc(
        protein_table=parsed.output,
        qc_report=parsed.qc_report,
        plot_column="PiBAQPpb" if parsed.normalize else "PiBAQ",
        tpa=parsed.tpa,
        ruler=parsed.ruler,
    )


def main():
    """Dispatch wheel periphery, plugin service, or Rust-native computation."""
    args = sys.argv[1:]
    if not args:
        _print_root_help(sys.stderr)
        raise SystemExit(2)
    global_options, routed_args = _extract_global_options(args)
    if global_options.version:
        package = importlib.import_module("mokume")
        print(getattr(package, "__version__"))
        raise SystemExit(0)
    if routed_args in (["-h"], ["--help"], ["help"]):
        _print_root_help(sys.stdout)
        raise SystemExit(0)
    if routed_args in (
        ["plot"],
        ["plot", "-h"],
        ["plot", "--help"],
        ["help", "plot"],
    ):
        print(_PLOT_HELP, file=sys.stdout, end="")
        raise SystemExit(0)
    periphery = _periphery_request(routed_args)
    if periphery is not None:
        command, command_args = periphery
        raise SystemExit(_run_periphery(command, command_args, global_options))
    if routed_args[:2] == ["mcp", "serve"]:
        _configure_global_logging(global_options)
        module = importlib.import_module("mokume.agentic.mcp_server")
        try:
            raise SystemExit(getattr(module, "main")(routed_args[2:]))
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
    if routed_args and routed_args[0] == "mcp":
        raise SystemExit("Usage: mokume mcp serve [--knowledge PATH]")
    package = importlib.import_module("mokume")
    if routed_args[:1] == ["help"]:
        args = [*routed_args[1:], "--help"]
    code = getattr(package, "_run_cli")(args)
    if code == 0:
        try:
            _render_requested_pibaq_qc(args, package)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
    raise SystemExit(code)


if __name__ == "__main__":
    main()
