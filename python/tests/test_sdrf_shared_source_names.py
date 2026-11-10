"""
Warnings for SDRF source names that map to more than one data file.

Runs that share a ``source name`` are combined into one sample. That is how
technical replicates and fractions are meant to be combined, but it also
follows from a duplicated name, so the SDRF reader says so. The message text
is shared with the Rust kernel (``SdrfTable::shared_source_name_warnings``).
"""

from __future__ import annotations

from pathlib import Path

from mokume.core.constants import load_sdrf
from mokume.io.feature import Feature
from mokume.preprocessing.sdrf import (
    shared_source_name_warnings,
    shared_source_names,
    warn_shared_source_names,
)

EXAMPLE_DIR = Path(__file__).parent / "example"
LOGGER = "mokume.preprocessing.sdrf"


def _write_sdrf(tmp_path: Path, technical_replicates: tuple[str, str]) -> Path:
    path = tmp_path / "shared.sdrf.tsv"
    path.write_text(
        "source name\tcomment[data file]\tcomment[fraction identifier]\t"
        "comment[technical replicate]\n"
        f"sample-1\trun-a.raw\t1\t{technical_replicates[0]}\n"
        f"sample-1\trun-b.raw\t1\t{technical_replicates[1]}\n"
        "sample-2\trun-c.raw\t1\t1\n"
    )
    return path


def test_unique_source_names_give_no_warning(caplog):
    sdrf = load_sdrf(str(EXAMPLE_DIR / "PXD020192.sdrf.tsv"))
    assert shared_source_names(sdrf) == []
    with caplog.at_level("WARNING", logger=LOGGER):
        warn_shared_source_names(sdrf)
    assert caplog.text == ""


def test_distinct_technical_replicates_are_reported(tmp_path):
    sdrf = load_sdrf(str(_write_sdrf(tmp_path, ("1", "2"))))
    assert shared_source_names(sdrf) == [
        {
            "source_name": "sample-1",
            "data_files": ["run-a.raw", "run-b.raw"],
            "same_replicate_and_fraction": False,
        }
    ]
    assert shared_source_name_warnings(sdrf) == [
        "SDRF source names mapped to more than one data file: `sample-1` "
        "(run-a.raw, run-b.raw). Runs that share a source name are combined into "
        "one sample (technical replicates or fractions), which also affects the "
        "normalization of all other samples."
    ]


def test_same_or_missing_technical_replicate_is_flagged_as_duplicate(tmp_path):
    for replicates in (("1", "1"), ("", "")):
        sdrf = load_sdrf(str(_write_sdrf(tmp_path, replicates)))
        warnings = shared_source_name_warnings(sdrf)
        assert len(warnings) == 1, warnings
        assert warnings[0].startswith(
            "SDRF source names reused across data files with the same technical "
            "replicate and fraction identifier: `sample-1` (run-a.raw, run-b.raw)."
        )
        assert "usually indicate a duplicated source name" in warnings[0]


def test_long_lists_are_truncated(tmp_path):
    path = tmp_path / "many.sdrf.tsv"
    rows = ["source name\tcomment[data file]"]
    rows += [f"sample-{s}\trun-{s}-{r}.raw" for s in range(12) for r in range(7)]
    path.write_text("\n".join(rows) + "\n")
    warnings = shared_source_name_warnings(load_sdrf(str(path)))
    assert len(warnings) == 1
    assert (
        "`sample-0` (run-0-0.raw, run-0-1.raw, run-0-2.raw, run-0-3.raw, "
        "run-0-4.raw, ... 2 more)"
    ) in warnings[0]
    assert "; ... 2 more." in warnings[0]
    assert "`sample-10`" not in warnings[0]


def test_enrich_with_sdrf_warns_on_shared_source_name(tmp_path, caplog):
    """The example data with one run renamed onto another sample's source name."""
    lines = (EXAMPLE_DIR / "PXD020192.sdrf.tsv").read_text().splitlines(keepends=True)
    shared = tmp_path / "shared.sdrf.tsv"
    shared.write_text(
        "".join(
            line.replace("PXD020192-Sample-14\t", "PXD020192-Sample-13\t", 1)
            if line.startswith("PXD020192-Sample-14\t")
            else line
            for line in lines
        )
    )
    feature = Feature(str(EXAMPLE_DIR / "feature_wide.parquet"))
    with caplog.at_level("WARNING", logger=LOGGER):
        feature.enrich_with_sdrf(str(shared))
    assert "`PXD020192-Sample-13` (S13_Brain_1.raw, S14_Brain_2.raw)" in caplog.text
    assert "duplicated source name" in caplog.text
