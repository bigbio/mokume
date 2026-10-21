# Installation

## From PyPI

```bash
pip install mokume
```

`mokume` is the default Rust-kernel distribution. It installs the compiled
`mokume._mokume` extension, its Python API and periphery, and the `mokume`
console command.

!!! warning "Install `mokume` **or** `mokume-py` — never both"

    Both distributions install the same `mokume` import package, so having both
    in one environment makes pip silently overwrite files (it does not detect the
    collision). Keep only one; to switch, `pip uninstall` the other first. Each
    package warns at import time if it finds its sibling installed.

Install the pure-Python implementation explicitly when needed:

```bash
pip install mokume-py
```

`mokume-py` also installs pyOpenMS as a base dependency. Its piBAQ paths call
the installed pyOpenMS `ProteaseDigestion` implementation directly; piBAQ is
included in the base install and has no self-maintained cleavage-rule catalog.

`pip install mokume` gives you the Rust compute kernel (the compiled
`mokume._mokume` extension), the thin Python API that drives it in-process, and
pyOpenMS. The base dependency supplies the runtime protease catalog and FASTA
digestion for both piBAQ commands; pyOpenMS itself depends on numpy, pandas, and
matplotlib. Extras select the additional Python stacks needed by each Mokume
periphery command.

!!! note "Console command"

    The wheel installs one `mokume` console command. Its root help lists the
    `quantify` command group, standalone batch correction, and the optional
    Python periphery workflows supplied by the installed wheel.

### Optional Extras

mokume uses optional dependencies for the periphery commands:

=== "Analysis"

    ```bash
    pip install mokume[analysis]
    ```

    Enables `mokume plot tsne`, `mokume plot pca`, `mokume plot de`, the piBAQ
    QC report, `mokume interactive-report`, the QC / workflow-comparison
    reports, and the pure-Python method fallback the Rust kernel does not
    reproduce — `mokume.impute(method="missforest")` (numpy, pandas, scipy,
    scikit-learn, matplotlib, seaborn, plotly).

=== "TissueMap"

    ```bash
    pip install mokume[tissuemap]
    ```

    Enables `mokume tissuemap` for per-dataset tissue atlas analysis, including
    AdaTiSS tissue-specificity scoring, embeddings, and atlas plots (scanpy,
    anndata, umap-learn, combat, matplotlib, seaborn, pyarrow).

=== "Mokume Plugin"

    ```bash
    pip install "mokume[plugin]"
    ```

    This optional MCP workflow requires Python 3.10 or newer.

    Installs the local MCP service used by the installable Mokume Plugin. The
    plugin host owns the model and credentials; Mokume does not need a model API
    key. Continue with the [plugin installation](user-guide/agentic-plugin.md).

=== "Everything"

    ```bash
    pip install mokume[all]
    ```

    Installs all optional periphery and local MCP dependencies.

## From Source

mokume builds from two project roots, one per distribution:

- **`mokume-py` (pure Python)** — `python/pyproject.toml` uses standard PEP 621
  metadata with the **hatchling** build backend
  (`build-backend = "hatchling.build"`, not Poetry). Install it from source with:

  ```bash
  git clone https://github.com/bigbio/mokume
  cd mokume
  pip install ./python          # builds mokume-py via the hatchling backend
  ```

- **`mokume` (Rust wheel)** — the default distribution is built with maturin,
  which compiles the internal `crates/mokume-py` PyO3 binding crate into the
  `mokume._mokume` extension and packages it
  with the Python periphery. The maturin project lives in `rust/`:

  ```bash
  pip install ./rust            # builds the extension via the maturin backend
  ```

For a development checkout of the Rust wheel, build the extension in place with
`maturin develop` run from `rust/`; the periphery is plain Python and needs no
separate build step.

## Using Conda

```bash
mamba env create -f rust/environment.yaml
conda activate mokume
pip install ./rust
```

## Requirements

- Python >= 3.10 for the Rust-backed ``mokume`` distribution, ``mokume-py``,
  and the optional Plugin/MCP workflow
- Both distributions declare pyOpenMS as a base dependency. In the default
  Rust-backed wheel, piBAQ reads every protease registered by the installed
  runtime, digests the FASTA in Python, and passes the complete theoretical-
  peptide map into the Rust kernel. pyOpenMS's declared dependencies also install
  numpy, pandas, and matplotlib
- Periphery extras select additional packages such as scipy, scikit-learn,
  pyarrow, plotly, scanpy, and friends for the commands that need them
- The quantification methods (DirectLFQ, MaxLFQ, piBAQ), ComBat batch correction,
  DE methods (limma, DEqMS, proDA, LimROTS, ROTS), and most imputation methods
  run in the **native Rust kernel** — no R or rpy2 required. ComBat is
  oracle-verified against inmoose
- A method stays in the Python periphery because the kernel cannot reproduce
  it cross-language: the `missforest` imputer (`mokume.impute`), in the
  `analysis` extra
- The Mokume Plugin requires an agent host plus `mokume[plugin]`; its bundled
  stdio MCP server is started by the host and calls the local Rust-backed wheel
