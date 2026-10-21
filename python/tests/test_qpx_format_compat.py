"""
Test QPX format compatibility: verify mokume can read both new and legacy QPX parquet formats.

New QPX format columns:
  - charge, run_file_name, intensities[{label, intensity}]

Legacy QPX format columns:
  - precursor_charge, reference_file_name, intensities[{sample_accession, channel, intensity}]
"""

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mokume.io import feature as qpx_feature

# Arrow type constants for new QPX format
_NEW_INTENSITIES_TYPE = pa.list_(
    pa.struct([("label", pa.string()), ("intensity", pa.float32())])
)
_PG_PROTEIN_TYPE = pa.list_(
    pa.struct(
        [
            ("accession", pa.string()),
            ("start", pa.int32()),
            ("end", pa.int32()),
            ("pre", pa.string()),
            ("post", pa.string()),
        ]
    )
)
_ADDITIONAL_SCORES_TYPE = pa.list_(
    pa.struct(
        [
            ("score_name", pa.string()),
            ("score_value", pa.float64()),
            ("higher_better", pa.bool_()),
        ]
    )
)
_NEW_QPX_SCHEMA = pa.schema(
    [
        ("sequence", pa.string()),
        ("peptidoform", pa.string()),
        ("pg_accessions", _PG_PROTEIN_TYPE),
        ("anchor_protein", pa.string()),
        ("charge", pa.int16()),
        ("run_file_name", pa.string()),
        ("unique", pa.bool_()),
        ("is_decoy", pa.bool_()),
        ("intensities", _NEW_INTENSITIES_TYPE),
    ]
)


def _make_new_qpx_parquet(
    path: str,
    *,
    reverse_rows: bool = False,
    reverse_intensities: bool = False,
    missing_reporter_label: bool = False,
) -> None:
    """Create a mock parquet file in new QPX format (matches latest QPX schema).

    Key differences from legacy:
    - pg_accessions: list<struct{accession, start, end, pre, post}> (not list<string>)
    - unique: bool (not int)
    - anchor_protein: string (new field)
    - charge: int16 (not precursor_charge)
    - run_file_name: string (not reference_file_name)
    - intensities: list<struct{label, intensity}> (not {sample_accession, channel, intensity})
    - is_decoy: bool (new field)
    """

    data = {
        "sequence": ["PEPTIDEK", "ANOTHERPEPTIDE", "PEPTIDEK"],
        "peptidoform": ["PEPTIDEK", "ANOTHERPEPTIDE", "PEPTIDEK"],
        "pg_accessions": [
            [
                {
                    "accession": "sp|P12345|PROT_HUMAN",
                    "start": 10,
                    "end": 18,
                    "pre": "K",
                    "post": "A",
                }
            ],
            [
                {
                    "accession": "sp|P67890|PROT2_HUMAN",
                    "start": 5,
                    "end": 19,
                    "pre": "R",
                    "post": "L",
                }
            ],
            [
                {
                    "accession": "sp|CONTAMINANT_P99999|CONTAM_HUMAN",
                    "start": 10,
                    "end": 18,
                    "pre": "K",
                    "post": "A",
                }
            ],
        ],
        "anchor_protein": [
            "sp|P12345|PROT_HUMAN",
            "sp|P67890|PROT2_HUMAN",
            "sp|CONTAMINANT_P99999|CONTAM_HUMAN",
        ],
        "charge": [2, 3, 2],
        "run_file_name": ["run1", "run1", "run2"],
        "unique": [True, False, True],
        "is_decoy": [False, False, False],
        "intensities": [
            [
                {"label": "TMT126", "intensity": 1000.0},
                {"label": "TMT127", "intensity": 2000.0},
                {"label": "UNMAPPED_ZERO", "intensity": 0.0},
            ],
            [
                {"label": "TMT126", "intensity": 3000.0},
                {"label": "TMT127", "intensity": 4000.0},
            ],
            [
                {"label": "TMT126", "intensity": 1500.0},
                {"label": "TMT127", "intensity": 2500.0},
            ],
        ],
    }
    if missing_reporter_label:
        data["intensities"][0][0]["label"] = None
    if reverse_intensities:
        data["intensities"] = [list(reversed(values)) for values in data["intensities"]]
    if reverse_rows:
        data = {column: list(reversed(values)) for column, values in data.items()}
    table = pa.table(data, schema=_NEW_QPX_SCHEMA)
    pq.write_table(table, path)


def _make_lfq_qpx_parquet(path: str) -> None:
    protein = {
        "accession": "sp|P12345|PROT_HUMAN",
        "start": 10,
        "end": 18,
        "pre": "K",
        "post": "A",
    }
    table = pa.table(
        {
            "sequence": ["PEPTIDEA", "PEPTIDEB"],
            "peptidoform": ["PEPTIDEA", "PEPTIDEB"],
            "pg_accessions": [[protein], [protein]],
            "anchor_protein": [protein["accession"], protein["accession"]],
            "charge": [2, 2],
            "run_file_name": ["Run_A.raw", "Run_B.mzML"],
            "unique": [True, True],
            "is_decoy": [False, False],
            "intensities": [
                [
                    {"label": "run_a", "intensity": 10.0},
                    {"label": "RUN_B.wiff", "intensity": 20.0},
                    {"label": "UNMAPPED_ZERO", "intensity": 0.0},
                ],
                [
                    {"label": "run_a", "intensity": 30.0},
                    {"label": "RUN_B.wiff", "intensity": 40.0},
                ],
            ],
        },
        schema=_NEW_QPX_SCHEMA,
    )
    pq.write_table(table, path)


def _make_scored_qpx_parquet(path: str) -> None:
    protein = {
        "accession": "sp|P12345|PROT_HUMAN",
        "start": 1,
        "end": 8,
        "pre": "K",
        "post": "R",
    }
    schema = _NEW_QPX_SCHEMA.append(
        pa.field("additional_scores", _ADDITIONAL_SCORES_TYPE)
    )
    table = pa.table(
        {
            "sequence": ["PEPTIDEA", "PEPTIDEB"],
            "peptidoform": ["PEPTIDEA", "PEPTIDEB"],
            "pg_accessions": [[protein], [protein]],
            "anchor_protein": [protein["accession"], protein["accession"]],
            "charge": [2, 2],
            "run_file_name": ["run_a", "run_b"],
            "unique": [True, True],
            "is_decoy": [False, False],
            "intensities": [
                [{"label": "run_a", "intensity": 10.0}],
                [{"label": "run_b", "intensity": 20.0}],
            ],
            "additional_scores": [
                [
                    {
                        "score_name": "quality",
                        "score_value": 0.9,
                        "higher_better": True,
                    },
                    {
                        "score_name": "qvalue",
                        "score_value": 0.01,
                        "higher_better": False,
                    },
                ],
                [
                    {
                        "score_name": "quality",
                        "score_value": 0.1,
                        "higher_better": True,
                    },
                    {
                        "score_name": "qvalue",
                        "score_value": 0.2,
                        "higher_better": False,
                    },
                ],
            ],
        },
        schema=schema,
    )
    pq.write_table(table, path)


def _make_raw_placeholder_lfq_qpx_parquet(path: str) -> None:
    """Create the older QPX LFQ dialect whose only intensity label is ``raw``."""
    protein = {
        "accession": "sp|P12345|PROT_HUMAN",
        "start": 10,
        "end": 18,
        "pre": "K",
        "post": "A",
    }
    table = pa.table(
        {
            "sequence": ["PEPTIDEA", "PEPTIDEB"],
            "peptidoform": ["PEPTIDEA", "PEPTIDEB"],
            "pg_accessions": [[protein], [protein]],
            "anchor_protein": [protein["accession"], protein["accession"]],
            "charge": [2, 2],
            "run_file_name": ["Run_A", "Run_B"],
            "unique": [True, True],
            "is_decoy": [False, False],
            "intensities": [
                [{"label": "raw", "intensity": 10.0}],
                [{"label": "raw", "intensity": 20.0}],
            ],
        },
        schema=_NEW_QPX_SCHEMA,
    )
    pq.write_table(table, path)


def _make_zero_anchor_lfq_parquet(path: str) -> None:
    """LFQ where one run's only anchored row has nothing quantifiable.

    ``A.raw`` anchors a single row whose intensities are all zero -- an
    identification with no integrated peak. ``B.raw`` anchors a row carrying
    match-between-runs quant for both runs. The file is still LFQ: every label
    names a run.
    """
    protein = {
        "accession": "sp|P12345|PROT_HUMAN",
        "start": 10,
        "end": 18,
        "pre": "K",
        "post": "A",
    }
    table = pa.table(
        {
            "sequence": ["PEPTIDEA", "PEPTIDEB"],
            "peptidoform": ["PEPTIDEA", "PEPTIDEB"],
            "pg_accessions": [[protein], [protein]],
            "anchor_protein": [protein["accession"], protein["accession"]],
            "charge": [2, 2],
            "run_file_name": ["A.raw", "B.raw"],
            "unique": [True, True],
            "is_decoy": [False, False],
            "intensities": [
                [{"label": "A.raw", "intensity": 0.0}],
                [
                    {"label": "A.raw", "intensity": 10.0},
                    {"label": "B.raw", "intensity": 20.0},
                ],
            ],
        },
        schema=_NEW_QPX_SCHEMA,
    )
    pq.write_table(table, path)


def _make_legacy_qpx_parquet(path: str) -> None:
    """Create a mock parquet file in legacy QPX format."""
    intensities_type = pa.list_(
        pa.struct(
            [
                ("sample_accession", pa.string()),
                ("channel", pa.string()),
                ("intensity", pa.float64()),
            ]
        )
    )
    schema = pa.schema(
        [
            ("sequence", pa.string()),
            ("peptidoform", pa.string()),
            ("pg_accessions", pa.list_(pa.string())),
            ("precursor_charge", pa.int32()),
            ("reference_file_name", pa.string()),
            ("unique", pa.int32()),
            ("intensities", intensities_type),
        ]
    )

    data = {
        "sequence": ["PEPTIDEK", "ANOTHERPEPTIDE", "PEPTIDEK"],
        "peptidoform": ["PEPTIDEK", "ANOTHERPEPTIDE", "PEPTIDEK"],
        "pg_accessions": [["P12345"], ["P67890"], ["P12345"]],
        "precursor_charge": [2, 3, 2],
        "reference_file_name": ["run1.raw", "run1.raw", "run2.raw"],
        "unique": [1, 1, 1],
        "intensities": [
            [
                {"sample_accession": "S1", "channel": "TMT126", "intensity": 1000.0},
                {"sample_accession": "S2", "channel": "TMT127", "intensity": 2000.0},
            ],
            [
                {"sample_accession": "S1", "channel": "TMT126", "intensity": 3000.0},
                {"sample_accession": "S2", "channel": "TMT127", "intensity": 4000.0},
            ],
            [
                {"sample_accession": "S1", "channel": "TMT126", "intensity": 1500.0},
                {"sample_accession": "S2", "channel": "TMT127", "intensity": 2500.0},
            ],
        ],
    }
    table = pa.table(data, schema=schema)
    pq.write_table(table, path)


def _write_sdrf(path, rows) -> None:
    lines = ["source name\tcomment[data file]\tcomment[label]\tfactor value[group]\n"]
    lines.extend("\t".join(row) + "\n" for row in rows)
    path.write_text("".join(lines), encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "threshold"),
    [("quality", 0.5), ("qvalue", 0.05)],
)
def test_named_score_filter_uses_qpx_direction(tmp_path, name, threshold):
    parquet_file = tmp_path / f"{name}.feature.parquet"
    _make_scored_qpx_parquet(str(parquet_file))
    builder = qpx_feature.SQLFilterBuilder(
        remove_contaminants=False,
        require_unique=False,
        named_score_name=name,
        named_score_threshold=threshold,
    )
    feature = qpx_feature.Feature(str(parquet_file), filter_builder=builder)
    try:
        report = feature.get_report_from_database(feature.samples)
    finally:
        feature.parquet_db.close()

    assert report["sequence"].tolist() == ["PEPTIDEA"]


def test_named_score_filter_rejects_missing_name(tmp_path):
    parquet_file = tmp_path / "missing-score.feature.parquet"
    _make_scored_qpx_parquet(str(parquet_file))
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name="missing",
        named_score_threshold=0.5,
    )

    with pytest.raises(ValueError, match="additional_scores entry `missing`"):
        qpx_feature.Feature(str(parquet_file), filter_builder=builder)


class TestQpxSdrfIdentity:
    """Test that each new-QPX intensity has one SDRF owner."""

    def test_exact_mapping_preserves_identity_under_sdrf_reordering(self, tmp_path):
        rows = [
            ("sample-r1-126", "/data/RUN1.mzML", "AC=MS:1;NT=TMT126", "A"),
            ("sample-r1-127", "run1.raw", "TMT127", "B"),
            ("sample-r2-126", "RUN2.wiff", "tmt126", "C"),
            ("sample-r2-127", "run2.d", "TMT127", "D"),
        ]
        expected = {
            ("sample-r1-126", "A"): 4000.0,
            ("sample-r1-127", "B"): 6000.0,
            ("sample-r2-126", "C"): 1500.0,
            ("sample-r2-127", "D"): 2500.0,
        }

        cases = [
            (rows, False, False),
            (list(reversed(rows)), False, False),
            (rows, True, True),
        ]
        for index, (ordered_rows, reverse_rows, reverse_intensities) in enumerate(
            cases
        ):
            parquet_file = tmp_path / f"new-qpx-{index}.feature.parquet"
            sdrf_file = tmp_path / f"mapping-{index}.sdrf.tsv"
            _make_new_qpx_parquet(
                str(parquet_file),
                reverse_rows=reverse_rows,
                reverse_intensities=reverse_intensities,
            )
            _write_sdrf(sdrf_file, ordered_rows)
            from mokume.io.feature import Feature

            feature = Feature(str(parquet_file))
            feature.enrich_with_sdrf(str(sdrf_file))
            observed = {
                (sample, condition): intensity
                for sample, condition, intensity in feature.parquet_db.execute(
                    "SELECT sample_accession, condition, sum(intensity) FROM parquet_db "
                    "GROUP BY sample_accession, condition"
                ).fetchall()
            }

            assert observed == expected
            assert sum(observed.values()) == 14000.0

    def test_lfq_intensity_labels_retain_their_sdrf_sample(self, tmp_path):
        parquet_file = tmp_path / "lfq.feature.parquet"
        sdrf_file = tmp_path / "lfq.sdrf.tsv"
        _make_lfq_qpx_parquet(str(parquet_file))
        _write_sdrf(
            sdrf_file,
            [
                ("sample-a", "run_a.mzML", "Label free sample", "A"),
                ("sample-b", "RUN_B.raw", "LFQ", "B"),
            ],
        )
        from mokume.io.feature import Feature

        feature = Feature(str(parquet_file))
        feature.enrich_with_sdrf(str(sdrf_file))
        observed = feature.parquet_db.execute(
            "SELECT sample_accession, run, channel, sum(intensity) "
            "FROM parquet_db GROUP BY ALL ORDER BY sample_accession"
        ).fetchall()

        assert observed == [
            ("sample-a", "run_a", None, 40.0),
            ("sample-b", "RUN_B.wiff", None, 60.0),
        ]
        assert set(feature.samples) == {"sample-a", "sample-b"}

    def test_raw_placeholder_lfq_label_matches_canonical_sdrf_label(self, tmp_path):
        """Archived run suffixes and raw LFQ labels map to canonical SDRF rows."""
        parquet_file = tmp_path / "raw-placeholder.feature.parquet"
        sdrf_file = tmp_path / "raw-placeholder.sdrf.tsv"
        _make_raw_placeholder_lfq_qpx_parquet(str(parquet_file))
        _write_sdrf(
            sdrf_file,
            [
                ("sample-a", "Run_A.d.zip", "Label free sample", "A"),
                ("sample-b", "Run_B.mzML.gz", "LFQ", "B"),
            ],
        )
        feature = qpx_feature.Feature(str(parquet_file))
        feature.enrich_with_sdrf(str(sdrf_file))

        assert feature.parquet_db.execute(
            "SELECT sample_accession, run, sum(intensity) "
            "FROM parquet_db GROUP BY ALL ORDER BY sample_accession"
        ).fetchall() == [
            ("sample-a", "Run_A", 10.0),
            ("sample-b", "Run_B", 20.0),
        ]

    def test_lfq_run_rejects_reporter_labeled_sdrf(self, tmp_path):
        parquet_file = tmp_path / "lfq.feature.parquet"
        sdrf_file = tmp_path / "labeled.sdrf.tsv"
        _make_lfq_qpx_parquet(str(parquet_file))
        _write_sdrf(
            sdrf_file,
            [
                ("sample-a", "run_a.mzML", "TMT126", "A"),
                ("sample-b", "run_b.raw", "LFQ", "B"),
            ],
        )
        from mokume.io.feature import Feature

        with pytest.raises(ValueError, match="has no reporter label") as error:
            Feature(str(parquet_file)).enrich_with_sdrf(str(sdrf_file))

        assert "record 1 (`run_a.mzML`, label `TMT126`)" in str(error.value)

    def test_duplicate_normalized_mapping_is_rejected(self, tmp_path):
        parquet_file = tmp_path / "new_qpx.feature.parquet"
        _make_new_qpx_parquet(str(parquet_file))
        rows = [
            ("sample-a", "run1.raw", "TMT126", "A"),
            ("sample-b", "RUN1.mzML", "tmt126", "B"),
        ]
        from mokume.io.feature import Feature

        for index, ordered_rows in enumerate((rows, list(reversed(rows)))):
            sdrf_file = tmp_path / f"duplicate-{index}.sdrf.tsv"
            _write_sdrf(sdrf_file, ordered_rows)
            feature = Feature(str(parquet_file))

            with pytest.raises(ValueError, match="duplicate SDRF mapping") as error:
                feature.enrich_with_sdrf(str(sdrf_file))

            assert "run1" in str(error.value)
            assert "tmt126" in str(error.value)
            assert "record 1" in str(error.value)
            assert "record 2" in str(error.value)

    def test_unknown_channel_is_rejected_with_mapping_context(self, tmp_path):
        parquet_file = tmp_path / "new_qpx.feature.parquet"
        sdrf_file = tmp_path / "missing-channel.sdrf.tsv"
        _make_new_qpx_parquet(str(parquet_file))
        _write_sdrf(
            sdrf_file,
            [
                ("sample-r1-126", "run1.raw", "TMT126", "A"),
                ("sample-r2-126", "run2.raw", "TMT126", "C"),
                ("sample-r2-127", "run2.raw", "TMT127", "D"),
            ],
        )
        from mokume.io.feature import Feature

        feature = Feature(str(parquet_file))
        before = feature.parquet_db.execute(
            "SELECT sample_accession, channel, intensity FROM parquet_db ORDER BY ALL"
        ).fetchall()

        with pytest.raises(ValueError, match="no exact SDRF match") as error:
            feature.enrich_with_sdrf(str(sdrf_file))

        assert "QPX run `run1` and label `TMT127`" in str(error.value)
        assert "record 1 (`run1.raw`, label `TMT126`)" in str(error.value)
        assert (
            feature.parquet_db.execute(
                "SELECT sample_accession, channel, intensity FROM parquet_db ORDER BY ALL"
            ).fetchall()
            == before
        )

    def test_missing_reporter_label_is_rejected(self, tmp_path):
        parquet_file = tmp_path / "missing-label.feature.parquet"
        sdrf_file = tmp_path / "mapping.sdrf.tsv"
        _make_new_qpx_parquet(str(parquet_file), missing_reporter_label=True)
        _write_sdrf(
            sdrf_file,
            [
                ("sample-r1-126", "run1.raw", "TMT126", "A"),
                ("sample-r1-127", "run1.raw", "TMT127", "B"),
                ("sample-r2-126", "run2.raw", "TMT126", "C"),
                ("sample-r2-127", "run2.raw", "TMT127", "D"),
            ],
        )
        from mokume.io.feature import Feature

        feature = Feature(str(parquet_file))

        with pytest.raises(ValueError, match="no reporter label") as error:
            feature.enrich_with_sdrf(str(sdrf_file))

        assert "QPX run `run1`" in str(error.value)

    def test_run_with_only_zero_intensities_stays_in_the_run_domain(self, tmp_path):
        """A run that quantifies nothing must not flip the file onto TMT.

        The run domain decides whether every label names a run (LFQ) or a
        reporter channel (TMT). Building it from positive intensities alone
        drops a run whose rows are all zero, so its label stops looking like a
        run, the file is read as TMT, and every MBR-transferred intensity is
        folded onto its anchor run -- here A's 10.0 would land on B, giving B
        30.0 and losing A entirely.
        """
        from mokume.io.feature import Feature

        parquet_file = tmp_path / "zero-anchor-lfq.feature.parquet"
        _make_zero_anchor_lfq_parquet(str(parquet_file))

        feature = Feature(str(parquet_file))

        assert feature._label_is_run is True
        assert feature.parquet_db.execute(
            "SELECT sample_accession, sum(intensity) FROM parquet_db "
            "GROUP BY 1 ORDER BY 1"
        ).fetchall() == [("A.raw", 10.0), ("B.raw", 20.0)]

    def test_failed_run_label_probe_is_refused_rather_than_mapped_to_null(
        self, tmp_path, monkeypatch
    ):
        """A DuckDB error in the probe must not silently blank every sample.

        The probe is allowed to fail -- ``Feature`` stays usable for callers
        that never touch SDRF. But its pair list is the only thing the new-QPX
        SDRF join has to match on, so an empty list leaves the LEFT JOIN with
        nothing and turns sample_accession, condition and mixture into NULL for
        every row. Before the fallback ``COALESCE(..., run_file_name)`` was
        removed, a failed probe merely lost the LFQ label refinement.
        """
        from mokume.io.feature import Feature

        probe_sql = "SELECT DISTINCT run_file_name, unnest.label,"

        class FailingProbeConnection:
            """Delegates every call, but makes the probe query raise."""

            def __init__(self, inner):
                self._inner = inner

            def execute(self, query, *args, **kwargs):
                """Raise on the probe query, delegate everything else."""
                if isinstance(query, str) and query.startswith(probe_sql):
                    raise duckdb.OutOfMemoryException("Out of Memory Error: probe")
                return self._inner.execute(query, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        parquet_file = tmp_path / "probe-fails.feature.parquet"
        sdrf_file = tmp_path / "mapping.sdrf.tsv"
        _make_new_qpx_parquet(str(parquet_file))
        _write_sdrf(
            sdrf_file,
            [
                ("sample-r1-126", "run1.raw", "TMT126", "A"),
                ("sample-r1-127", "run1.raw", "TMT127", "B"),
                ("sample-r2-126", "run2.raw", "TMT126", "C"),
                ("sample-r2-127", "run2.raw", "TMT127", "D"),
            ],
        )

        real_connect = duckdb.connect
        monkeypatch.setattr(
            duckdb,
            "connect",
            lambda *a, **kw: FailingProbeConnection(real_connect(*a, **kw)),
        )
        feature = Feature(str(parquet_file))

        # Constructing the reader still works; only the SDRF join is refused.
        assert feature._qpx_sdrf_pairs == []
        assert "Out of Memory Error" in feature._qpx_probe_error

        with pytest.raises(ValueError, match="probe failed") as error:
            feature.enrich_with_sdrf(str(sdrf_file))

        assert "Out of Memory Error" in str(error.value)


class TestNewQPXFormat:
    """Test mokume Feature reader with new QPX format."""

    def test_feature_init_new_format(self, tmp_path):
        parquet_file = str(tmp_path / "new_qpx.feature.parquet")
        _make_new_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        assert feat._is_new_qpx is True
        assert feat._charge_col == "charge"
        assert feat._run_col == "run_file_name"

    def test_feature_query_new_format(self, tmp_path):
        parquet_file = str(tmp_path / "new_qpx.feature.parquet")
        _make_new_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        df = feat.parquet_db.execute("SELECT * FROM parquet_db").df()
        assert len(df) > 0
        for col in [
            "charge",
            "run_file_name",
            "intensity",
            "channel",
            "sample_accession",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_feature_samples_new_format(self, tmp_path):
        parquet_file = str(tmp_path / "new_qpx.feature.parquet")
        _make_new_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        samples = feat.get_unique_samples()
        assert len(samples) > 0


class TestLegacyQPXFormat:
    """Test mokume Feature reader with legacy QPX format."""

    def test_feature_init_legacy_format(self, tmp_path):
        parquet_file = str(tmp_path / "legacy_qpx.feature.parquet")
        _make_legacy_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        assert feat._is_new_qpx is False
        assert feat._charge_col == "precursor_charge"
        assert feat._run_col == "reference_file_name"

    def test_feature_query_legacy_format(self, tmp_path):
        parquet_file = str(tmp_path / "legacy_qpx.feature.parquet")
        _make_legacy_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        df = feat.parquet_db.execute("SELECT * FROM parquet_db").df()
        assert len(df) > 0
        for col in [
            "charge",
            "run_file_name",
            "intensity",
            "channel",
            "sample_accession",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_feature_samples_legacy_format(self, tmp_path):
        parquet_file = str(tmp_path / "legacy_qpx.feature.parquet")
        _make_legacy_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        samples = feat.get_unique_samples()
        assert len(samples) > 0


class TestNewQPXDeepCompat:
    """Test deep compatibility: pg_accessions struct parsing, unique bool, etc."""

    def test_pg_accessions_struct_in_pandas(self, tmp_path):
        """Verify pg_accessions list<struct> can be parsed like list<string>."""
        parquet_file = str(tmp_path / "new_qpx.feature.parquet")
        _make_new_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        df = feat.parquet_db.execute("SELECT pg_accessions FROM parquet_db").df()
        # In new QPX, pg_accessions is list<struct{accession,...}>
        # mokume code does: df["pg_accessions"].str[0] then .split("|")
        first_elem = df["pg_accessions"].str[0]
        assert first_elem.map(type).eq(str).all()

        # This is what mokume ratio.py does:
        first_acc = df["pg_accessions"].str[0].fillna("")
        result = np.where(
            first_acc.str.contains("|", regex=False),
            first_acc.str.split("|").str[1],
            first_acc,
        )
        assert sorted(result.tolist()) == ["CONTAMINANT_P99999"] * 2 + [
            "P12345",
            "P12345",
            "P67890",
            "P67890",
        ]

    def test_unique_bool_filter_sql(self, tmp_path):
        """Verify 'unique = 1' SQL filter works with bool column."""
        parquet_file = str(tmp_path / "new_qpx.feature.parquet")
        _make_new_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        # SQLFilterBuilder generates: "unique" = 1
        df = feat.parquet_db.execute('SELECT * FROM parquet_db WHERE "unique" = 1').df()
        assert df["unique"].tolist() == [True] * 4
        assert set(df["pg_accessions"].str[0]) == {
            "sp|P12345|PROT_HUMAN",
            "sp|CONTAMINANT_P99999|CONTAM_HUMAN",
        }

    def test_unique_bool_filter_pandas(self, tmp_path):
        """Verify unique == 1 Pandas filter works with bool column."""
        parquet_file = str(tmp_path / "new_qpx.feature.parquet")
        _make_new_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        df = feat.parquet_db.execute("SELECT * FROM parquet_db").df()
        # stages.py and peptide.py do: dataset_df[dataset_df["unique"] == 1]
        filtered = df[df["unique"] == 1]
        assert filtered["unique"].tolist() == [True] * 4
        assert set(filtered["pg_accessions"].str[0]) == {
            "sp|P12345|PROT_HUMAN",
            "sp|CONTAMINANT_P99999|CONTAM_HUMAN",
        }

    def test_get_low_frequency_peptides(self, tmp_path):
        """Fully observed struct-backed peptides are not marked low frequency."""
        parquet_file = str(tmp_path / "new_qpx.feature.parquet")
        _make_new_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature

        feat = Feature(parquet_file)

        result = feat.get_low_frequency_peptides(percentage=0.2)
        assert isinstance(result, tuple)
        assert not result

    def test_contaminant_filter_with_struct(self, tmp_path):
        """Test SQL contaminant filter (pg_accessions::text LIKE) with struct type."""
        parquet_file = str(tmp_path / "new_qpx.feature.parquet")
        _make_new_qpx_parquet(parquet_file)

        from mokume.io.feature import Feature, SQLFilterBuilder

        fb = SQLFilterBuilder(remove_contaminants=True, require_unique=False)
        feat = Feature(parquet_file, filter_builder=fb)

        where_clause, where_params = fb.build_where_clause()
        sql = "".join(["SELECT * FROM parquet_db WHERE ", where_clause])
        df = feat.parquet_db.execute(sql, where_params).df()
        assert len(df) == 4
        assert set(df["pg_accessions"].str[0]) == {
            "sp|P12345|PROT_HUMAN",
            "sp|P67890|PROT2_HUMAN",
        }


class TestBothFormatsProduceSameSchema:
    """Verify that both formats produce compatible output schemas."""

    def test_same_core_columns(self, tmp_path):
        """Both formats must produce all core columns needed by downstream code."""
        new_file = str(tmp_path / "new.parquet")
        legacy_file = str(tmp_path / "legacy.parquet")
        _make_new_qpx_parquet(new_file)
        _make_legacy_qpx_parquet(legacy_file)

        from mokume.io.feature import Feature

        feat_new = Feature(new_file)
        feat_legacy = Feature(legacy_file)

        df_new = feat_new.parquet_db.execute("SELECT * FROM parquet_db").df()
        df_legacy = feat_legacy.parquet_db.execute("SELECT * FROM parquet_db").df()

        core_columns = {
            "sequence",
            "peptidoform",
            "pg_accessions",
            "charge",
            "run_file_name",
            "unique",
            "sample_accession",
            "channel",
            "intensity",
            "run",
            "condition",
            "biological_replicate",
            "fraction",
            "mixture",
        }
        assert core_columns.issubset(set(df_new.columns)), (
            f"New format missing core columns: {core_columns - set(df_new.columns)}"
        )
        assert core_columns.issubset(set(df_legacy.columns)), (
            f"Legacy format missing core columns: {core_columns - set(df_legacy.columns)}"
        )

    def test_new_format_has_extra_columns(self, tmp_path):
        """New QPX format should expose is_decoy and anchor_protein."""
        new_file = str(tmp_path / "new.parquet")
        _make_new_qpx_parquet(new_file)

        from mokume.io.feature import Feature

        feat = Feature(new_file)
        df = feat.parquet_db.execute("SELECT * FROM parquet_db").df()

        assert "is_decoy" in df.columns, "New QPX should expose is_decoy"
        assert "anchor_protein" in df.columns, "New QPX should expose anchor_protein"
