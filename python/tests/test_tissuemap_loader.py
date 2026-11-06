"""Focused tests for TissueMap QPX metadata handling."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from mokume.commands.tissuemap import _build_config
from mokume.tissuemap import loader
from mokume.tissuemap.loader import (
    _aggregate_tmt_with_duckdb as aggregate_tmt_with_duckdb,
    _detect_gis_accessions as detect_gis_accessions,
    _tmt_label_aliases as tmt_label_aliases,
    TissueLoadOptions,
    load_dataset,
)


def _write_samples(tmp_path, rows: list[dict[str, str]]) -> None:
    qpx_dir = tmp_path / "qpx_output"
    qpx_dir.mkdir()
    pd.DataFrame(rows).to_parquet(qpx_dir / "TEST.sample.parquet", index=False)


def test_cli_defaults_do_not_override_tissuemap_yaml(tmp_path):
    """Omitted CLI options preserve scan_dir, output_dir, and n_jobs from YAML."""
    scan_dir = tmp_path / "yaml-scan"
    scan_dir.mkdir()
    output_dir = tmp_path / "yaml-output"
    config_path = tmp_path / "tissuemap.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "n_jobs": 17,
                "input": {"scan_dir": str(scan_dir)},
                "output": {"output_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )

    config = _build_config(None, None, config_path, (), None, None, None, None)

    assert config.n_jobs == 17
    assert config.input.scan_dir == scan_dir
    assert config.output.output_dir == output_dir


def test_detect_gis_from_qpx_pooled_sample_metadata(tmp_path):
    """A real QPX pooled-sample value identifies the GIS channel."""
    _write_samples(
        tmp_path,
        [
            {
                "sample_accession": "bio",
                "organism_part": "lung",
                "pooled_sample": "not pooled",
            },
            {
                "sample_accession": "gis",
                "organism_part": "not available",
                "pooled_sample": "SN=A_LUNG,B_BREAST",
            },
        ],
    )

    assert detect_gis_accessions(tmp_path, "TEST") == {"gis"}


def test_detect_gis_from_organism_part_without_pool_column(tmp_path):
    """The organism-part fallback identifies a GIS without pool metadata."""
    _write_samples(
        tmp_path,
        [
            {
                "sample_accession": "bio",
                "organism_part": "liver",
            },
            {
                "sample_accession": "gis",
                "organism_part": "global internal standard",
            },
        ],
    )

    assert detect_gis_accessions(tmp_path, "TEST") == {"gis"}


def test_tmt_pg_numeric_channels_use_qpx_sample_order(tmp_path):
    """Numeric reporter labels follow QPX channel order deterministically."""
    _write_samples(
        tmp_path,
        [
            {
                "sample_accession": "bio-a",
                "organism_part": "lung",
                "pooled_sample": "not pooled",
            },
            {
                "sample_accession": "bio-b",
                "organism_part": "liver",
                "pooled_sample": "not pooled",
            },
            {
                "sample_accession": "gis",
                "organism_part": "not available",
                "pooled_sample": "SN=A_LUNG,B_LIVER",
            },
        ],
    )
    pd.DataFrame(
        [
            {
                "anchor_protein": "P1",
                "run_file_name": "run-1",
                "intensities": [
                    {"label": "1", "intensity": 10.0},
                    {"label": "2", "intensity": 20.0},
                    {"label": "3", "intensity": 5.0},
                ],
                "is_decoy": False,
            }
        ]
    ).to_parquet(tmp_path / "qpx_output" / "TEST.feature.parquet", index=False)
    run_map = pd.DataFrame(
        [
            {
                "run_file_name": "run-1",
                "tmt_label": "TMT126",
                "channel_index": "1",
                "sample_accession": "bio-a",
            },
            {
                "run_file_name": "run-1",
                "tmt_label": "TMT127N",
                "channel_index": "2",
                "sample_accession": "bio-b",
            },
            {
                "run_file_name": "run-1",
                "tmt_label": "TMT131",
                "channel_index": "3",
                "sample_accession": "gis",
            },
        ]
    )

    result = aggregate_tmt_with_duckdb(
        tmp_path,
        "TEST",
        None,
        run_map,
        n_jobs=24,
    ).set_index("sample_accession")

    assert result.loc["bio-a", "intensity"] == 2.0
    assert result.loc["bio-b", "intensity"] == 4.0


def test_canonical_numeric_tmt_labels_win_over_positional_aliases():
    """Explicit numeric labels take precedence over positional aliases."""
    run_map = pd.DataFrame(
        [
            {
                "run_file_name": "run-1",
                "tmt_label": "2",
                "channel_index": "1",
                "sample_accession": "bio-a",
            },
            {
                "run_file_name": "run-1",
                "tmt_label": "1",
                "channel_index": "2",
                "sample_accession": "bio-b",
            },
        ]
    )

    aliases = tmt_label_aliases(run_map).set_index(["run_file_name", "tmt_label"])

    assert aliases.index.is_unique
    assert aliases.loc[("run-1", "1"), "sample_accession"] == "bio-b"
    assert aliases.loc[("run-1", "2"), "sample_accession"] == "bio-a"


def test_duckdb_tmt_normalizes_each_run_by_its_reference_channel(tmp_path):
    """Streaming aggregation divides by the GIS channel of the same run."""
    _write_samples(
        tmp_path,
        [
            {
                "sample_accession": "bio-a",
                "organism_part": "lung",
                "pooled_sample": "not pooled",
            },
            {
                "sample_accession": "bio-b",
                "organism_part": "liver",
                "pooled_sample": "not pooled",
            },
            {
                "sample_accession": "gis",
                "organism_part": "not available",
                "pooled_sample": "SN=A_LUNG,B_LIVER",
            },
        ],
    )
    feature_rows = [
        {
            "anchor_protein": "P1",
            "run_file_name": "run-1",
            "intensities": [
                {"label": "TMT126", "intensity": 10.0},
                {"label": "TMT127N", "intensity": 20.0},
                {"label": "TMT131", "intensity": 5.0},
            ],
            "is_decoy": False,
        },
        {
            "anchor_protein": "P1",
            "run_file_name": "run-2",
            "intensities": [
                {"label": "TMT126", "intensity": 30.0},
                {"label": "TMT127N", "intensity": 40.0},
                {"label": "TMT131", "intensity": 10.0},
            ],
            "is_decoy": False,
        },
    ]
    pd.DataFrame(feature_rows).to_parquet(
        tmp_path / "qpx_output" / "TEST.feature.parquet",
        index=False,
    )
    run_map = pd.DataFrame(
        [
            {
                "run_file_name": run_name,
                "tmt_label": label,
                "channel_index": str(channel_index),
                "sample_accession": sample,
            }
            for run_name in ("run-1", "run-2")
            for channel_index, (label, sample) in enumerate(
                (
                    ("TMT126", "bio-a"),
                    ("TMT127N", "bio-b"),
                    ("TMT131", "gis"),
                ),
                start=1,
            )
        ]
    )
    streamed = aggregate_tmt_with_duckdb(
        tmp_path,
        "TEST",
        None,
        run_map,
        n_jobs=24,
    ).sort_values(["protein", "sample_accession"], ignore_index=True)
    # Each run divides by its own reference channel, then the runs add up.
    expected = pd.DataFrame(
        {
            "protein": ["P1", "P1"],
            "sample_accession": ["bio-a", "bio-b"],
            "intensity": [10.0 / 5.0 + 30.0 / 10.0, 20.0 / 5.0 + 40.0 / 10.0],
        }
    )

    pd.testing.assert_frame_equal(streamed, expected)


@pytest.mark.parametrize("legacy", [False, True])
def test_load_dataset_lfq_stays_on_existing_path(tmp_path, monkeypatch, legacy):
    """LFQ input remains on the established non-TMT loader path."""
    _write_samples(
        tmp_path,
        [
            {"sample_accession": "sample-a", "organism_part": "lung"},
            {"sample_accession": "sample-b", "organism_part": "liver"},
        ],
    )
    pd.DataFrame(
        [
            {
                "run_file_name": "run-1",
                "fraction": 1,
                "samples": [
                    {
                        "label": "label free sample",
                        "sample_accession": "sample-a",
                    }
                ],
            },
            {
                "run_file_name": "run-2",
                "fraction": 1,
                "samples": [
                    {
                        "label": "label free sample",
                        "sample_accession": "sample-b",
                    }
                ],
            },
        ]
    ).to_parquet(tmp_path / "qpx_output" / "TEST.run.parquet", index=False)
    pd.DataFrame(
        [
            {
                "anchor_protein": "P1",
                "run_file_name": "run-1",
                "intensities": [_lfq_intensity(10.0, legacy)],
                "is_decoy": False,
            },
            {
                "anchor_protein": "P1",
                "run_file_name": "run-2",
                "intensities": [_lfq_intensity(20.0, legacy)],
                "is_decoy": False,
            },
        ]
    ).to_parquet(tmp_path / "qpx_output" / "TEST.feature.parquet", index=False)

    def fail_if_tmt_path_is_used(*_args, **_kwargs):
        raise AssertionError("LFQ input unexpectedly entered the TMT loader")

    monkeypatch.setattr(
        loader,
        "_aggregate_tmt_with_duckdb",
        fail_if_tmt_path_is_used,
    )
    matrix, metadata = load_dataset(
        tmp_path, "TEST", options=TissueLoadOptions(n_jobs=24)
    )

    assert matrix.loc["P1", "sample-a"] == 10.0
    assert matrix.loc["P1", "sample-b"] == 20.0
    assert set(metadata["quant_type"]) == {"LFQ"}


def _lfq_intensity(value, legacy):
    """Keep the legacy layout without a label field in its Arrow struct."""
    metadata = (
        {"sample_accession": "unused", "channel": "LFQ"} if legacy else {"label": "LFQ"}
    )
    return {**metadata, "intensity": value}


def test_load_dataset_lfq_matches_exploded_sums(tmp_path):
    """Decoys, non-positive and NaN intensities and unmatched runs stay out of the sums."""
    _write_samples(
        tmp_path,
        [
            {"sample_accession": "sample-a", "organism_part": "lung"},
            {"sample_accession": "sample-b", "organism_part": "liver"},
        ],
    )
    pd.DataFrame(
        [
            {
                "run_file_name": run,
                "fraction": 1,
                "samples": [{"label": "label free sample", "sample_accession": sample}],
            }
            for run, sample in (
                ("run-1", "sample-a"),
                ("run-2", "sample-a"),
                ("run-3", "sample-b"),
            )
        ]
    ).to_parquet(tmp_path / "qpx_output" / "TEST.run.parquet", index=False)
    pd.DataFrame(
        [
            ("P1", "run-1", 10.0, False),
            ("P1", "run-1", 1.0, None),
            ("P1", "run-2", 5.0, False),
            ("P1", "run-3", 20.0, False),
            ("P1", "run-3", 7.0, True),
            ("P2", "run-3", 0.0, False),
            ("P2", "run-3", float("nan"), False),
            ("P3", "unknown-run", 9.0, False),
        ],
        columns=["anchor_protein", "run_file_name", "value", "is_decoy"],
    ).assign(
        intensities=lambda rows: [
            [{"label": "LFQ", "intensity": value}] for value in rows["value"]
        ]
    ).drop(columns="value").to_parquet(
        tmp_path / "qpx_output" / "TEST.feature.parquet", index=False
    )

    matrix, metadata = load_dataset(
        tmp_path, "TEST", options=TissueLoadOptions(n_jobs=2)
    )

    assert matrix.to_dict() == {"sample-a": {"P1": 16.0}, "sample-b": {"P1": 20.0}}
    assert set(metadata["quant_type"]) == {"LFQ"}


def test_load_dataset_detects_tmt_from_feature_labels(tmp_path, monkeypatch):
    """TMT labels in the features switch to the TMT path when the run table has none."""
    _write_samples(
        tmp_path, [{"sample_accession": "sample-a", "organism_part": "lung"}]
    )
    pd.DataFrame(
        [
            {
                "run_file_name": "run-1",
                "fraction": 1,
                "samples": [
                    {"label": "label free sample", "sample_accession": "sample-a"}
                ],
            }
        ]
    ).to_parquet(tmp_path / "qpx_output" / "TEST.run.parquet", index=False)
    pd.DataFrame(
        [
            {
                "anchor_protein": "P1",
                "run_file_name": "run-1",
                "intensities": [{"label": "TMT126", "intensity": 10.0}],
                "is_decoy": False,
            }
        ]
    ).to_parquet(tmp_path / "qpx_output" / "TEST.feature.parquet", index=False)
    monkeypatch.setattr(
        loader,
        "_aggregate_tmt_with_duckdb",
        lambda *_args: pd.DataFrame(
            {"protein": ["P1"], "sample_accession": ["sample-a"], "intensity": [3.0]}
        ),
    )

    matrix, metadata = load_dataset(tmp_path, "TEST")

    assert matrix.loc["P1", "sample-a"] == 3.0
    assert set(metadata["quant_type"]) == {"TMT"}
