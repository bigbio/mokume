"""The relocated periphery command modules import and expose ``main(argv)``."""

import importlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

COMMAND_MODULES = [
    "visualize",
    "tissuemap",
    "de_plots",
    "interactive_report",
    "peptides2protein_qc",
    "peptides2protein_pibaq",
]

PUBLIC_CLI_PATHS = (
    ("tissuemap",),
    ("interactive-report",),
    ("plot", "pca"),
    ("plot", "tsne"),
    ("plot", "de"),
)


@pytest.mark.parametrize("name", COMMAND_MODULES)
def test_command_module_imports_with_main(name):
    """Each relocated periphery command must expose a callable entrypoint."""
    module = importlib.import_module(f"mokume.commands.{name}")
    assert callable(getattr(module, "main", None))


def test_console_root_help_covers_the_installed_wheel(monkeypatch, capsys):
    """The PyPI entry point must advertise native and periphery workflows."""
    entrypoint = importlib.import_module("mokume.__main__")
    monkeypatch.setattr(entrypoint.sys, "argv", ["mokume", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "quantify" in output
    assert "plot" in output
    assert "interactive-report" in output
    assert output.count("[requires: mokume[analysis]]") == 2
    assert "mokume[plotting]" not in output
    assert "mokume[reports]" not in output
    assert "--log-level <LEVEL>" in output
    assert "--log-file <FILE>" in output
    assert "mcp serve" not in output


def test_console_without_a_command_prints_unified_help(monkeypatch, capsys):
    """A missing command must not fall back to Rust-only discovery output."""
    entrypoint = importlib.import_module("mokume.__main__")
    monkeypatch.setattr(entrypoint.sys, "argv", ["mokume"])

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 2
    assert "quantify" in capsys.readouterr().err


@pytest.mark.parametrize("path", PUBLIC_CLI_PATHS)
def test_console_periphery_help_uses_public_command_path(path, monkeypatch, capsys):
    """Each advertised workflow must have direct command-specific help."""
    entrypoint = importlib.import_module("mokume.__main__")
    monkeypatch.setattr(entrypoint.sys, "argv", ["mokume", *path, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    assert f"usage: mokume {' '.join(path)}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("path", "expected_value_names"),
    [
        (("tissuemap",), ("--input <DIR>", "--threads <N>")),
        (("plot", "pca"), ("--protein-matrix <FILE>", "--output <FILE>")),
        (("plot", "tsne"), ("--input <DIR>", "--pattern <GLOB>")),
        (
            ("plot", "de"),
            ("--outdir <DIR>", "<GROUP_A> <GROUP_B> <DE_FILE>"),
        ),
        (
            ("interactive-report",),
            ("--output <FILE>", "<GROUP_A> <GROUP_B> <DE_FILE>"),
        ),
    ],
)
def test_console_periphery_help_uses_semantic_value_names(
    path, expected_value_names, monkeypatch, capsys
):
    """Periphery help should match the concise metavar style of the Rust CLI."""
    entrypoint = importlib.import_module("mokume.__main__")
    monkeypatch.setattr(entrypoint.sys, "argv", ["mokume", *path, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for expected in expected_value_names:
        assert expected in output


def test_tissuemap_help_uses_possible_values_label(monkeypatch, capsys):
    """Fixed Python choices should use the same label as the Rust CLI."""
    entrypoint = importlib.import_module("mokume.__main__")
    monkeypatch.setattr(entrypoint.sys, "argv", ["mokume", "tissuemap", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "[possible values: mindet, minprob" in output
    assert "[possible values: tsne, umap]" in output
    assert "choices:" not in output


def test_console_dispatches_periphery_arguments(monkeypatch):
    """Periphery argv must reach its module without being parsed by Rust."""
    observed = {}
    module = SimpleNamespace(
        main=lambda args: observed.update(args=args) or 0,
    )
    entrypoint = importlib.import_module("mokume.__main__")
    monkeypatch.setattr(
        entrypoint.importlib,
        "import_module",
        lambda name: (
            module
            if name == "mokume.commands.tissuemap"
            else pytest.fail(f"unexpected import: {name}")
        ),
    )
    monkeypatch.setattr(
        entrypoint.sys,
        "argv",
        [
            "mokume",
            "--log-level",
            "warn",
            "tissuemap",
            "--input",
            "datasets",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    assert observed["args"] == ["--input", "datasets"]


def test_console_dispatches_mcp_without_knowledge_path(monkeypatch):
    """Global logging options must be consumed before MCP argparse runs."""
    observed = {}
    module = SimpleNamespace(
        main=lambda args: observed.update(args=args) or 0,
    )

    def import_module(name):
        if name == "mokume.agentic.mcp_server":
            return module
        raise AssertionError(f"unexpected import: {name}")

    def configure_logging(**kwargs):
        observed["logging"] = kwargs

    entrypoint = importlib.import_module("mokume.__main__")
    monkeypatch.setattr(
        entrypoint.importlib,
        "import_module",
        import_module,
    )
    monkeypatch.setattr(
        entrypoint,
        "configure_logging",
        configure_logging,
    )
    monkeypatch.setattr(
        entrypoint.sys,
        "argv",
        [
            "mokume",
            "--log-level",
            "info",
            "mcp",
            "serve",
            "--log-file",
            "mokume.log",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    assert observed["args"] == []
    assert observed["logging"] == {"level": "info", "log_file": "mokume.log"}


def test_console_global_version_does_not_enter_mcp(monkeypatch, capsys):
    """The global version flag must not require MCP-specific arguments."""
    entrypoint = importlib.import_module("mokume.__main__")
    package = SimpleNamespace(__version__="0.2.0")
    monkeypatch.setattr(entrypoint.importlib, "import_module", lambda _name: package)
    monkeypatch.setattr(
        entrypoint.sys,
        "argv",
        ["mokume", "mcp", "serve", "--version"],
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == "0.2.0\n"


def test_pibaq_qc_wrapper_accepts_global_options_before_subcommand(monkeypatch):
    """Global options must not prevent post-command piBAQ QC rendering."""
    observed = {}
    package = SimpleNamespace(
        _run_cli=lambda _args: 0,
        peptides2protein_qc=lambda **kwargs: observed.update(kwargs),
    )
    entrypoint = importlib.import_module("mokume.__main__")
    argv = [
        "mokume",
        "--log-level",
        "warn",
        "quantify",
        "peptides2protein",
        "--quant-method",
        "pibaq",
        "--normalize",
        "--qc-report",
        "qc.pdf",
        "-o",
        "proteins.tsv",
    ]
    monkeypatch.setattr(entrypoint.sys, "argv", argv)
    monkeypatch.setattr(entrypoint.importlib, "import_module", lambda _name: package)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    assert observed == {
        "protein_table": "proteins.tsv",
        "qc_report": "qc.pdf",
        "plot_column": "PiBAQPpb",
        "tpa": False,
        "ruler": False,
    }


def test_pibaq_qc_wrapper_accepts_equals_syntax_without_verbose():
    """An explicit QC path requests rendering without a separate verbose flag."""
    observed = {}
    package = SimpleNamespace(
        peptides2protein_qc=lambda **kwargs: observed.update(kwargs)
    )
    entrypoint = importlib.import_module("mokume.__main__")

    render_qc = getattr(entrypoint, "_render_requested_pibaq_qc")
    render_qc(
        [
            "quantify",
            "peptides2protein",
            "--quant-method=pibaq",
            "--qc-report=qc.pdf",
            "--output=proteins.tsv",
        ],
        package,
    )

    assert observed["qc_report"] == "qc.pdf"


def test_python_compute_wrapper_runs_requested_pibaq_qc(monkeypatch):
    """The in-process Python wrapper must not accept qc_report as a no-op."""
    observed = {}
    package = importlib.import_module("mokume")
    monkeypatch.setattr(package, "_prepare_pibaq_digest", lambda _args: None)
    monkeypatch.setattr(package, "_native_run", lambda _args: None)
    monkeypatch.setattr(
        package,
        "peptides2protein_qc",
        lambda **kwargs: observed.update(kwargs),
    )

    package.peptides2protein(
        quant_method="pibaq",
        peptides="peptides.tsv",
        fasta="proteins.fasta",
        output="proteins.tsv",
        qc_report="qc.pdf",
    )

    assert observed["protein_table"] == "proteins.tsv"
    assert observed["qc_report"] == "qc.pdf"


def test_non_pibaq_qc_report_does_not_render_pibaq_qc():
    """A QC path for another method must not invoke the piBAQ renderer."""
    package = SimpleNamespace(
        peptides2protein_qc=lambda **_kwargs: pytest.fail("unexpected piBAQ QC")
    )
    entrypoint = importlib.import_module("mokume.__main__")

    render_qc = getattr(entrypoint, "_render_requested_pibaq_qc")
    render_qc(
        [
            "quantify",
            "peptides2protein",
            "--quant-method",
            "top3",
            "--qc-report",
            "qc.pdf",
            "--output",
            "proteins.tsv",
        ],
        package,
    )


def test_pibaq_compat_command_qc_report_enables_rendering(monkeypatch):
    """The compatibility command treats an explicit QC path as a render request."""
    captured = {}
    module = importlib.import_module("mokume.commands.peptides2protein_pibaq")
    pibaq_module = SimpleNamespace(
        peptides_to_protein=lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: pibaq_module,
    )

    result = module.main(
        [
            "--peptides",
            "peptides.tsv",
            "--fasta",
            "proteins.fasta",
            "--enzyme",
            "Trypsin",
            "--output",
            "proteins.tsv",
            "--qc-report",
            "qc.pdf",
        ]
    )

    assert result == 0
    assert captured["verbose"] is True
    assert captured["qc_report"] == "qc.pdf"


def test_tissuemap_config_values_are_not_overridden_by_parser_defaults(
    tmp_path, monkeypatch
):
    """Omitted CLI options must preserve values loaded from TissueMap YAML."""
    scan_dir = tmp_path / "datasets"
    scan_dir.mkdir()
    output_dir = tmp_path / "configured-output"
    config_path = tmp_path / "tissuemap.yaml"
    config_path.write_text(
        "\n".join(
            (
                "n_jobs: 3",
                "input:",
                f"  scan_dir: {scan_dir}",
                "output:",
                f"  output_dir: {output_dir}",
            )
        ),
        encoding="utf-8",
    )
    captured = {}

    def capture_pipeline(config):
        """Return a no-op pipeline while retaining its resolved configuration."""
        captured["config"] = config
        return SimpleNamespace(run=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "mokume.tissuemap.pipeline",
        SimpleNamespace(TissueMapPipeline=capture_pipeline),
    )
    module = importlib.import_module("mokume.commands.tissuemap")

    assert module.main(["--config", str(config_path)]) == 0

    config = captured["config"]
    assert config.n_jobs == 3
    assert config.input.scan_dir == scan_dir
    assert config.output.output_dir == output_dir


@pytest.mark.parametrize(
    "content,match",
    [
        ("unknown: true\n", "Unknown TissueMap config keys"),
        ("input:\n  unknown: true\n", "Unknown keys in 'input' section"),
    ],
)
def test_tissuemap_config_rejects_unknown_keys(tmp_path, content, match):
    """Unknown top-level and section keys must fail closed."""
    config_path = tmp_path / "tissuemap.yaml"
    config_path.write_text(content, encoding="utf-8")
    config_module = importlib.import_module("mokume.tissuemap.config")

    with pytest.raises(ValueError, match=match):
        config_module.load_config(config_path)


def test_visualize_reads_tab_delimited_protein_tables(tmp_path, monkeypatch):
    """Visualization must honor the delimiter of protein table inputs."""
    protein_path = tmp_path / "demo-proteins.tsv"
    protein_path.write_text(
        "ProteinName\tSampleID\tPiBAQLog\n"
        "P1\tS1\t1.0\n"
        "P2\tS1\t2.0\n"
        "P1\tS2\t1.1\n"
        "P2\tS2\t2.1\n",
        encoding="utf-8",
    )
    module = importlib.import_module("mokume.commands.visualize")
    plotting = importlib.import_module("mokume.plotting")

    monkeypatch.setattr(
        plotting,
        "compute_pca_with_plot",
        lambda frame, n_components: frame.iloc[:, :1],
    )
    monkeypatch.setattr(plotting, "plot_tsne", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_compute_tsne_compat",
        lambda frame: pd.DataFrame(
            {"tSNE1": [0.0] * len(frame), "tSNE2": [0.0] * len(frame)},
            index=frame.index,
        ),
    )

    assert (
        module.main(["--input", str(tmp_path), "--output", str(tmp_path / "tsne.pdf")])
        == 0
    )


def test_visualize_requires_output(tmp_path):
    """The public t-SNE command must not write to an implicit legacy filename."""
    module = importlib.import_module("mokume.commands.visualize")

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--input", str(tmp_path)])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "module_name,args,mode,match",
    [
        (
            "de_plots",
            ["--protein-matrix", "proteins.csv", "--outdir", "plots"],
            "de",
            "select --volcano or --heatmap",
        ),
        (
            "de_plots",
            [
                "--protein-matrix",
                "proteins.csv",
                "--output",
                "pca.png",
            ],
            "pca",
            "required: -s/--sdrf",
        ),
        (
            "interactive_report",
            [
                "--protein-matrix",
                "proteins.csv",
                "--sdrf",
                "data.sdrf.tsv",
                "--output",
                "report.html",
                "--plot-dir",
                "plots",
            ],
            None,
            "unrecognized arguments: --plot-dir plots",
        ),
        (
            "peptides2protein_pibaq",
            [
                "--peptides",
                "peptides.tsv",
                "--fasta",
                "proteins.fasta",
                "--enzyme",
                "Trypsin",
                "--output",
                "proteins.tsv",
                "--ploidy",
                "2",
            ],
            None,
            "--ploidy/--organism/--cpc require --ruler",
        ),
    ],
)
def test_periphery_commands_reject_options_that_would_be_ignored(
    module_name, args, mode, match, capsys
):
    """Periphery commands must reject options outside their active path."""
    module = importlib.import_module(f"mokume.commands.{module_name}")

    with pytest.raises(SystemExit) as exc_info:
        if mode is None:
            module.main(args)
        else:
            module.main(args, mode=mode)
    if isinstance(exc_info.value.code, str):
        assert match in exc_info.value.code
    else:
        assert exc_info.value.code == 2
        assert match in capsys.readouterr().err


def test_tissuemap_generate_config_rejects_run_options(tmp_path):
    """Config generation must reject execution-only overrides."""
    module = importlib.import_module("mokume.commands.tissuemap")

    code = module.main(
        [
            "--generate-config",
            str(tmp_path / "config.yaml"),
            "--threads",
            "24",
        ]
    )

    assert code == 2
    assert not (tmp_path / "config.yaml").exists()
