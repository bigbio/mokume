#!/usr/bin/env python3
"""Vendor the shared Python sidecar subset from ``python/mokume`` into ``rust/python/mokume``.

``rust/python/mokume`` is a hand-maintained VENDORED COPY of a SUBSET of the
canonical ``python/mokume`` package: the plotting / QC / analysis / tissue-atlas
periphery that the Rust compute kernel does not port. ``python/mokume`` is the
canonical, upstream-facing source; the sidecar rides along inside the ``mokume``
wheel so the periphery ships next to the compiled kernel.

Most of that subset is meant to be byte-for-byte identical to its canonical
counterpart. Keeping those files in sync by hand is drift-prone, so this script
copies the identical shared files from ``python/mokume`` into ``rust/python/mokume``
driven by the EXPLICIT ``SHARED_FILES`` allow-list below. A companion CI guard
(``.github/workflows/sidecar-drift.yml``) runs this script in ``--check`` mode and
fails if any allow-listed file has drifted between the two trees.

Modes
-----
- default (apply): copy every allow-listed file from ``python/mokume`` over the
  sidecar copy, so developers can re-sync after editing an upstream file.
- ``--check``: do not write anything; report any allow-listed file that differs
  (or is missing on either side) and exit non-zero. Used by CI.

The allow-list is DELIBERATELY explicit rather than "copy the whole tree": some
sidecar files INTENTIONALLY DIVERGE from canonical and must never be clobbered.
Those are catalogued in ``INTENTIONALLY_EXCLUDED`` (with the reason each one
diverges) purely as living documentation; the script simply never touches any
file that is not in ``SHARED_FILES``.

Intentionally-excluded (divergent) files — NOT vendored, do NOT add to the
allow-list:

  __init__.py
      Rust-wheel package init: imports the compiled ``mokume._mokume`` extension
      (``run`` / ``version``) and builds keyword->CLI-flag wrappers. A different
      public surface from the pure-Python package init.
  commands/__init__.py
      Documents/exposes the periphery commands as ``python -m mokume.commands.*``
      ``main(argv)`` modules instead of the pure-Python Click entrypoints.
  commands/visualize.py
  commands/tissuemap.py
      Re-architected from Click decorators to standalone ``argparse`` +
      ``main(argv) -> int`` to match the wheel's CLI-dispatch model. Same compute,
      different entrypoint wiring.
  core/__init__.py
      Lazy ``__getattr__`` for ``WriteCSVTask`` / ``WriteParquetTask`` so importing
      ``mokume.core`` does not require pyarrow.
  io/__init__.py
  normalization/__init__.py
  quantification/__init__.py
      Trimmed package inits: only the surviving vendored submodules are re-exported
      (FASTA / IRS / piBAQ respectively), via lazy ``__getattr__``; the Rust-ported
      methods and non-vendored submodules are dropped on purpose.
  quantification/pibaq.py
      Rust-wheel compatibility entrypoints adapt DataFrames and legacy arguments
      to the native piBAQ core; the pure-Python calculation and file workflow are
      intentionally not vendored.
  quantification/_pibaq_allocation.py
      The pure-Python package retains the allocation implementation, while the
      Rust-wheel sidecar keeps only DataFrame helpers used around the native core.
  imputation/censored.py
  imputation/methods.py
  imputation/missforest.py
      Eager top-level scikit-learn imports with the optional-dependency
      ``try/except ImportError`` guards removed. scikit-learn comes from the
      periphery extras that expose these imputation modules. Same computation;
      only the import ceremony differs.

Sidecar-only files (exist ONLY in ``rust/python/mokume`` — no canonical
counterpart, so never synced): ``__main__.py``, ``_pibaq_digest.py``,
``commands/de_plots.py``,
``commands/interactive_report.py``, ``commands/peptides2protein_pibaq.py``,
``commands/peptides2protein_qc.py``.

Usage
-----
    # from anywhere; paths are resolved relative to the repo root:
    python rust/scripts/vendor_sidecar.py            # apply: re-sync the sidecar
    python rust/scripts/vendor_sidecar.py --check     # CI: fail on any drift
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

# Repo root = two levels up from this file (rust/scripts/vendor_sidecar.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = REPO_ROOT / "python" / "mokume"
SIDECAR_ROOT = REPO_ROOT / "rust" / "python" / "mokume"

# Explicit allow-list of files that MUST stay byte-for-byte identical between
# ``python/mokume`` (canonical) and ``rust/python/mokume`` (sidecar). Paths are
# relative to each package root. Keep this sorted. Only add a file here once it
# is genuinely identical and meant to stay that way; anything that intentionally
# diverges belongs in the module docstring's excluded list, not here.
SHARED_FILES: tuple[str, ...] = (
    "core/constants.py",
    "core/logger.py",
    "core/logging_config.py",
    "core/write_queue.py",
    "data/__init__.py",
    "data/organisms.py",
    "imputation/__init__.py",
    "imputation/bpca.py",
    "imputation/gms.py",
    "imputation/impseq.py",
    "imputation/impseqrob.py",
    "imputation/qrilc.py",
    "imputation/seqknn.py",
    "imputation/stochastic.py",
    "io/fasta.py",
    "model/__init__.py",
    "model/batch_correction.py",
    "model/labeling.py",
    "model/normalization.py",
    "model/organism.py",
    "model/quantification.py",
    "model/summarization.py",
    "normalization/irs.py",
    "plotting/__init__.py",
    "plotting/differential_expression.py",
    "plotting/distributions.py",
    "plotting/pca.py",
    "quantification/families.py",
    "reports/__init__.py",
    "reports/interactive.py",
    "reports/qc_report.py",
    "reports/workflow_comparison.py",
    "tissuemap/__init__.py",
    "tissuemap/batch_correction.py",
    "tissuemap/config.py",
    "tissuemap/embedding.py",
    "tissuemap/enrichment.py",
    "tissuemap/loader.py",
    "tissuemap/pipeline.py",
    "tissuemap/plotting/__init__.py",
    "tissuemap/plotting/atlas.py",
    "tissuemap/plotting/embedding.py",
    "tissuemap/plotting/markers.py",
    "tissuemap/plotting/specificity.py",
    "tissuemap/preprocessing.py",
    "tissuemap/protein_selection.py",
    "tissuemap/tissue_specificity.py",
)

# Files that INTENTIONALLY diverge and must never be copied. Kept here (not just
# in the docstring) so ``--check`` can assert none of them accidentally leaked
# into ``SHARED_FILES``. The prose reason for each lives in the module docstring.
INTENTIONALLY_EXCLUDED: tuple[str, ...] = (
    "__init__.py",
    "commands/__init__.py",
    "commands/tissuemap.py",
    "commands/visualize.py",
    "core/__init__.py",
    "imputation/censored.py",
    "imputation/methods.py",
    "imputation/missforest.py",
    "io/__init__.py",
    "normalization/__init__.py",
    "quantification/_pibaq_allocation.py",
    "quantification/__init__.py",
    "quantification/pibaq.py",
)


def _validate_lists() -> list[str]:
    """Return configuration errors in the allow-list / exclude-list themselves."""
    errors: list[str] = []

    overlap = set(SHARED_FILES) & set(INTENTIONALLY_EXCLUDED)
    if overlap:
        errors.append(
            "files listed as both shared and intentionally-excluded: "
            + ", ".join(sorted(overlap))
        )

    if list(SHARED_FILES) != sorted(SHARED_FILES):
        errors.append("SHARED_FILES is not sorted; keep it sorted for readability")

    dupes = sorted({p for p in SHARED_FILES if SHARED_FILES.count(p) > 1})
    if dupes:
        errors.append("duplicate entries in SHARED_FILES: " + ", ".join(dupes))

    return errors


def _apply() -> int:
    """Copy every allow-listed file from canonical into the sidecar. Returns exit code."""
    errors = _validate_lists()
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1

    copied = 0
    unchanged = 0
    for rel in SHARED_FILES:
        src = CANONICAL_ROOT / rel
        dst = SIDECAR_ROOT / rel
        if not src.is_file():
            print(f"error: canonical source missing: {src}", file=sys.stderr)
            return 1
        if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
            unchanged += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"vendored: {rel}")
        copied += 1

    print(
        f"done: {copied} file(s) updated, {unchanged} already in sync "
        f"({len(SHARED_FILES)} allow-listed)."
    )
    return 0


def _check() -> int:
    """Report any drift between canonical and sidecar for allow-listed files. Returns exit code."""
    errors = _validate_lists()
    for err in errors:
        print(f"error: {err}", file=sys.stderr)

    drifted: list[str] = []
    missing: list[str] = []
    for rel in SHARED_FILES:
        src = CANONICAL_ROOT / rel
        dst = SIDECAR_ROOT / rel
        if not src.is_file():
            missing.append(f"{rel} (missing in python/mokume)")
            continue
        if not dst.is_file():
            missing.append(f"{rel} (missing in rust/python/mokume)")
            continue
        if not filecmp.cmp(src, dst, shallow=False):
            drifted.append(rel)

    for rel in missing:
        print(f"error: allow-listed file not found: {rel}", file=sys.stderr)
    for rel in drifted:
        print(
            f"error: sidecar drift: rust/python/mokume/{rel} differs from "
            f"python/mokume/{rel}",
            file=sys.stderr,
        )

    if errors or missing or drifted:
        print(
            "\nSidecar is out of sync with the canonical python/mokume package. "
            "Run `python rust/scripts/vendor_sidecar.py` (or `make vendor-sidecar`) "
            "and commit the result. If a file is meant to diverge, remove it from "
            "SHARED_FILES and document it in the script header.",
            file=sys.stderr,
        )
        return 1

    print(f"ok: {len(SHARED_FILES)} allow-listed sidecar file(s) match canonical.")
    return 0


def main(argv: list[str]) -> int:
    """Apply or verify the explicit canonical-to-sidecar file mapping."""
    parser = argparse.ArgumentParser(
        prog="vendor_sidecar.py",
        description=(
            "Sync the identical shared subset of python/mokume into "
            "rust/python/mokume (the vendored Rust-wheel sidecar)."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit non-zero instead of writing (used by CI).",
    )
    args = parser.parse_args(argv)

    if not CANONICAL_ROOT.is_dir():
        print(f"error: canonical package not found: {CANONICAL_ROOT}", file=sys.stderr)
        return 1
    if not SIDECAR_ROOT.is_dir():
        print(f"error: sidecar package not found: {SIDECAR_ROOT}", file=sys.stderr)
        return 1

    return _check() if args.check else _apply()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
