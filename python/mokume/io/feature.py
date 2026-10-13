"""
Feature data access via DuckDB for proteomics parquet files.

This module provides the Feature class for reading and querying quantms.io/qpx
wide-format parquet files using DuckDB, and the SQLFilterBuilder for constructing
SQL-level filters for normalization pre-computations.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

import pandas as pd
import numpy as np
import duckdb

from mokume.model.labeling import QuantificationCategory, IsobaricLabel
from mokume.core.constants import load_sdrf
from mokume.core.logger import get_logger
from mokume.io.msstats import create_msstats_feature_table

logger = get_logger("mokume.io.feature")


def _first_present(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first candidate present in a parquet schema."""
    return next((name for name in candidates if name in columns), None)


def _optional_qvalue_projection(source: Optional[str], alias: str, indent: int) -> str:
    """Build a canonical q-value projection, using typed null when absent."""
    padding = " " * indent
    expression = f"{source} as {alias}" if source else f"NULL::DOUBLE as {alias}"
    return f",\n{padding}{expression}"


def _optional_passthrough_projection(enabled: bool, column: str, indent: int) -> str:
    """Project an optional source column without inventing a typed-null field."""
    return f",\n{' ' * indent}{column}" if enabled else ""


def _normalize_run_key(value: object) -> str:
    if pd.isna(value):
        return ""
    key = str(value).strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()
    # SDRF data files can name an archived vendor directory/file, for example
    # ``sample.d.zip`` or ``sample.mzML.gz``, while QPX stores ``sample``.
    # Peel a chain of known transport and acquisition suffixes.
    extensions = (".zip", ".gz", ".raw", ".mzml", ".d", ".wiff")
    stripped = True
    while stripped:
        stripped = False
        for extension in extensions:
            if key.endswith(extension):
                key = key[: -len(extension)]
                stripped = True
                break
    return key


def _normalize_label_key(value: object) -> str:
    if pd.isna(value):
        return ""
    label = str(value).strip()
    for part in label.split(";"):
        part = part.strip()
        if part.startswith("NT="):
            label = part[3:].strip()
            break
    key = label.lower()
    # Older quantms.io/QPX feature files use ``raw`` as the sole
    # ``intensities.label`` value for label-free runs. It is a quantification
    # placeholder, not a reporter channel, and must match the SDRF's canonical
    # ``label free sample`` term.
    if key in {"raw", "lfq", "label-free", "label free sample"}:
        return "label free sample"
    return key


def _sdrf_record_context(records: list[dict]) -> str:
    return ", ".join(
        f"record {row['sdrf_record']} (`{row['sdrf_data_file']}`, label `{row['sdrf_label']}`)"
        for row in records
    )


def _resolve_qpx_sdrf_mapping(
    sdrf_records: list[dict],
    qpx_pairs: list[tuple[object, object]],
    labels_are_runs: bool,
) -> list[dict]:
    by_run: dict[str, list[dict]] = {}
    by_run_label: dict[tuple[str, str], dict] = {}
    for record in sdrf_records:
        run_key = record["sdrf_run_key"]
        label_key = record["sdrf_label_key"]
        key = (run_key, label_key)
        if previous := by_run_label.get(key):
            raise ValueError(
                "duplicate SDRF mapping for normalized run "
                f"`{run_key}` and label `{label_key}` at "
                f"{_sdrf_record_context([previous, record])}"
            )
        by_run_label[key] = record
        by_run.setdefault(run_key, []).append(record)

    resolved = []
    for qpx_run, qpx_label_value in sorted(
        qpx_pairs, key=lambda pair: (str(pair[0]), str(pair[1]))
    ):
        run_key = _normalize_run_key(qpx_run)
        label_key = "" if labels_are_runs else _normalize_label_key(qpx_label_value)
        run_matches = by_run.get(run_key, [])
        if not labels_are_runs and not label_key:
            qpx_label = "" if pd.isna(qpx_label_value) else str(qpx_label_value)
            context = _sdrf_record_context(run_matches) or "no SDRF records"
            raise ValueError(
                f"QPX run `{qpx_run}` and label `{qpx_label}` have no reporter "
                f"label; normalized run occurs in {context}"
            )
        exact = by_run_label.get((run_key, label_key))
        matches = run_matches if labels_are_runs else ([exact] if exact else [])
        if len(matches) != 1:
            qpx_label = "" if pd.isna(qpx_label_value) else str(qpx_label_value)
            if not matches:
                context = _sdrf_record_context(run_matches) or "no SDRF records"
                raise ValueError(
                    f"QPX run `{qpx_run}` and label `{qpx_label}` have no "
                    f"exact SDRF match; normalized run occurs in {context}"
                )
            raise ValueError(
                f"QPX run `{qpx_run}` and label `{qpx_label}` are ambiguous "
                f"across {_sdrf_record_context(matches)}"
            )

        mapped = matches[0].copy()
        if labels_are_runs and mapped["sdrf_label_key"] not in {
            "",
            "label free sample",
        }:
            raise ValueError(
                f"QPX run `{qpx_run}` has no reporter label; "
                f"{_sdrf_record_context([mapped])} declares a labeled channel"
            )
        mapped["qpx_run"] = qpx_run
        mapped["qpx_label"] = "" if pd.isna(qpx_label_value) else qpx_label_value
        resolved.append(mapped)

    return resolved


@dataclass
class SQLFilterBuilder:
    """Builds SQL WHERE clauses for filtering parquet data at query time.

    This class is used to ensure that normalization factors (median maps, peptide
    frequencies, IRS scaling) are computed on filtered data that excludes contaminants,
    decoys, and other artifacts.

    Attributes
    ----------
    remove_contaminants : bool
        Whether to exclude rows containing contaminant patterns in pg_accessions.
    contaminant_patterns : list[str]
        List of substring patterns to identify contaminants (e.g., ["CONTAMINANT", "DECOY"]).
    min_intensity : float
        Minimum intensity threshold (0.0 means only filter intensity > 0).
    min_peptide_length : int
        Minimum peptide sequence length.
    require_unique : bool
        Whether to require unique peptides only (unique = 1).
    has_is_decoy : bool
        Whether the parquet has an ``is_decoy`` column. When True, DECOY
        filtering uses ``is_decoy = false`` instead of text pattern matching.
        Automatically set by :class:`Feature` after format detection.
    named_score_name : str, optional
        Exact QPX ``additional_scores.score_name`` to filter.
    named_score_threshold : float, optional
        Threshold interpreted using the score's ``higher_better`` flag.
    """

    remove_contaminants: bool = True
    contaminant_patterns: list[str] = field(
        default_factory=lambda: ["CONTAMINANT", "CONTAM_", "ENTRAP", "DECOY"]
    )
    min_intensity: float = 0.0
    min_peptide_length: int = 7
    require_unique: bool = True
    has_is_decoy: bool = False
    named_score_name: Optional[str] = None
    named_score_threshold: Optional[float] = None

    def build_where_clause(self) -> tuple[str, list]:
        """Build parameterized SQL WHERE clause for DuckDB queries.

        Returns
        -------
        tuple[str, list]
            A tuple of (clause, params) where *clause* is a SQL WHERE fragment
            with ``?`` placeholders and *params* is the list of bind values.
        """
        conditions: list[str] = []
        params: list = []

        # Always filter intensity > 0
        conditions.append("intensity > 0")

        # Min intensity threshold
        if self.min_intensity > 0:
            conditions.append("intensity >= ?")
            params.append(self.min_intensity)

        # Peptide length filter
        if self.min_peptide_length > 0:
            conditions.append('LENGTH("sequence") >= ?')
            params.append(self.min_peptide_length)

        # Unique peptides only
        if self.require_unique:
            conditions.append('"unique" = 1')

        score_clause, score_params = self.build_named_score_clause()
        if score_clause:
            conditions.append(score_clause)
            params.extend(score_params)

        # Contaminant/decoy filter
        if self.remove_contaminants and self.contaminant_patterns:
            cont_conds, cont_params = self.build_contaminant_filter()
            conditions.append("(" + " AND ".join(cont_conds) + ")")
            params.extend(cont_params)

        clause = " AND ".join(conditions) if conditions else "1=1"
        return clause, params

    def build_named_score_clause(self) -> tuple[str, list]:
        """Return the input-level predicate for the configured named score."""
        if self.named_score_name is None:
            return "", []
        clause = (
            "EXISTS (SELECT 1 FROM UNNEST(additional_scores) AS score_item(score) "
            "WHERE score.score_name = ? AND "
            "((score.higher_better AND score.score_value >= ?) OR "
            "(NOT score.higher_better AND score.score_value <= ?)))"
        )
        return clause, [
            self.named_score_name,
            self.named_score_threshold,
            self.named_score_threshold,
        ]

    def build_contaminant_filter(self) -> tuple[list[str], list]:
        """Build contaminant/decoy filter conditions and params."""
        conditions: list[str] = []
        params: list = []

        def any_member(patterns: list[str]) -> str:
            params.extend(patterns)
            tests = " OR ".join("strpos(a::text, ?) > 0" for _ in patterns)
            return f"list_transform(pg_accessions, a -> {tests})"

        decoys = [p for p in self.contaminant_patterns if p.upper() == "DECOY"]
        contams = [p for p in self.contaminant_patterns if p.upper() != "DECOY"]
        if decoys and self.has_is_decoy:
            conditions.append("is_decoy = false")
        if decoys:
            conditions.append(
                f"NOT coalesce(list_bool_or({any_member(decoys)}), false)"
            )
        # A group is a contaminant only when every member matches.
        if contams:
            conditions.append(
                "NOT coalesce(len(pg_accessions) > 0 AND "
                f"list_bool_and({any_member(contams)}), false)"
            )
        return conditions, params


class Feature:
    """
    Represents a feature in a proteomics dataset, providing methods for data manipulation
    and analysis using a DuckDB database connection to a Parquet file.

    This class expects the quantms.io/qpx wide format where intensities are stored
    in a nested array. Supports both new QPX format intensities[{label, intensity}, ...]
    and legacy format intensities[{sample_accession, channel, intensity}, ...]

    Attributes
    ----------
    filter_builder : Optional[SQLFilterBuilder]
        If provided, this filter builder will be used to apply SQL-level filtering
        when computing normalization factors (median maps, peptide frequencies, etc.).
        This ensures normalization is computed on clean data without contaminants/decoys.
    """

    labels: Optional[list[str]]
    label: Optional[QuantificationCategory]
    choice: Optional[IsobaricLabel]
    technical_repetitions: Optional[int]
    filter_builder: Optional[SQLFilterBuilder]

    _MEMORY_LIMIT_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*[KMGT]?B?\s*$", re.IGNORECASE)

    def __init__(
        self,
        database_path: str,
        filter_builder: Optional[SQLFilterBuilder] = None,
        duckdb_memory: Optional[str] = None,
        duckdb_threads: Optional[int] = None,
    ):
        """Open a DuckDB-backed view over the parquet at ``database_path``.

        ``duckdb_memory`` and ``duckdb_threads`` map directly onto DuckDB's
        ``SET memory_limit=...`` / ``SET threads=...`` pragmas. They cap **the
        DuckDB engine only**; the surrounding Python process (PyArrow,
        polars, pandas) is not limited by these values. Use cgroup
        ``MemoryMax``, SLURM ``--mem``, or container ``resources.limits`` for
        a process-level cap in production.
        """
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"the file {database_path} does not exist.")

        self.parquet_db = self._connect(duckdb_memory, duckdb_threads)

        # Use DuckDB Python API to avoid SQL string interpolation for file paths
        self.parquet_db.read_parquet(database_path).create_view("parquet_db_raw")
        self._detect_qpx_format()
        self._initialize_raw_view(filter_builder)

    @classmethod
    def from_msstats(
        cls,
        msstats_path: str,
        sdrf_path: str,
        filter_builder: Optional[SQLFilterBuilder] = None,
        duckdb_limits: tuple[Optional[str], Optional[int]] = (None, None),
    ) -> "Feature":
        """Open a quantms MSstats table as a QPX-compatible feature view."""
        feature = cls.__new__(cls)
        feature.parquet_db = cls._connect(*duckdb_limits)
        try:
            create_msstats_feature_table(
                feature.parquet_db,
                msstats_path=msstats_path,
                sdrf_path=sdrf_path,
            )
            feature._detect_qpx_format()
            feature._initialize_raw_view(filter_builder)
        except Exception:
            feature.parquet_db.close()
            raise
        return feature

    @classmethod
    def _connect(
        cls,
        duckdb_memory: Optional[str],
        duckdb_threads: Optional[int],
    ) -> duckdb.DuckDBPyConnection:
        """Create a DuckDB connection with the configured resource limits."""
        connection = duckdb.connect()
        try:
            # Apply DuckDB resource pragmas before any expensive operation.
            if duckdb_memory is not None:
                if not cls._MEMORY_LIMIT_RE.match(str(duckdb_memory)):
                    raise ValueError(
                        "duckdb_memory must be a DuckDB memory string like "
                        f"'80GB' or '16384MB'; got {duckdb_memory!r}"
                    )
                connection.execute("SET memory_limit=?", [str(duckdb_memory)])
            if duckdb_threads is not None:
                threads = int(duckdb_threads)
                if threads < 1:
                    raise ValueError(
                        f"duckdb_threads must be >= 1; got {duckdb_threads!r}"
                    )
                connection.execute("SET threads=?", [threads])
        except Exception:
            connection.close()
            raise
        return connection

    def _initialize_raw_view(self, filter_builder: Optional[SQLFilterBuilder]) -> None:
        """Initialize format detection and the canonical long feature view."""
        self._create_unnest_view()

        self._samples: Optional[list[str]] = None
        self.filter_builder = filter_builder
        # Propagate is_decoy availability to filter builder for optimized DECOY filtering
        if self.filter_builder is not None and self._has_is_decoy:
            self.filter_builder.has_is_decoy = True
        if self.filter_builder is not None and self.filter_builder.named_score_name:
            self._validate_named_score(self.filter_builder.named_score_name)

    def _named_score_clause(self) -> tuple[str, list]:
        if self.filter_builder is None:
            return "", []
        return self.filter_builder.build_named_score_clause()

    def _append_named_score_condition(
        self, conditions: list[str], params: list
    ) -> None:
        score_clause, score_params = self._named_score_clause()
        if score_clause:
            conditions.append(score_clause)
            params.extend(score_params)

    def _validate_named_score(self, name: str) -> None:
        """Fail before output when a requested score is absent or malformed."""
        if not self._has_additional_scores:
            raise ValueError(
                f"named score filter `{name}` requires a QPX additional_scores column"
            )
        row = self.parquet_db.execute(
            """
            SELECT
                COUNT(*) AS score_count,
                COUNT(DISTINCT score.higher_better) AS direction_count,
                COUNT(*) FILTER (
                    WHERE score.score_value IS NULL
                       OR NOT isfinite(score.score_value)
                       OR score.higher_better IS NULL
                ) AS invalid_count
            FROM parquet_db_raw,
                 UNNEST(additional_scores) AS score_item(score)
            WHERE score.score_name = ?
            """,
            [name],
        ).fetchone()
        score_count, direction_count, invalid_count = row
        if score_count == 0:
            raise ValueError(
                f"--filter-score requires QPX additional_scores entry `{name}`"
            )
        if invalid_count:
            raise ValueError(f"QPX score `{name}` contains null or non-finite values")
        if direction_count != 1:
            raise ValueError(
                f"QPX score `{name}` has inconsistent higher_better values"
            )

    @property
    def samples(self) -> list[str]:
        """Load and cache sample accessions when a caller first requests them."""
        if self._samples is None:
            self._samples = self.get_unique_samples()
        return self._samples

    @samples.setter
    def samples(self, value: list[str]) -> None:
        """Replace the cached sample accessions."""
        self._samples = value

    def _detect_qpx_format(self) -> None:
        """Detect whether the parquet uses new or legacy QPX schema."""
        cols = [
            r[0]
            for r in self.parquet_db.execute(
                "SELECT column_name FROM (DESCRIBE parquet_db_raw)"
            ).fetchall()
        ]
        self._is_new_qpx = "charge" in cols or "run_file_name" in cols
        self._charge_col = "charge" if self._is_new_qpx else "precursor_charge"
        self._run_col = "run_file_name" if self._is_new_qpx else "reference_file_name"

        # Detect if pg_accessions is list<struct{accession,...}> (new QPX)
        # vs list<string> (legacy). If struct, we need to extract .accession.
        self._pg_accessions_is_struct = False
        if "pg_accessions" in cols:
            try:
                type_str = (
                    self.parquet_db.execute(
                        "SELECT typeof(pg_accessions) FROM parquet_db_raw LIMIT 1"
                    )
                    .fetchone()[0]
                    .lower()
                )
                self._pg_accessions_is_struct = "struct" in type_str
            except Exception as exc:
                logger.debug("Could not detect pg_accessions type: %s", exc)

        # Detect new QPX fields for optimized filtering
        self._has_is_decoy = "is_decoy" in cols
        self._has_anchor_protein = "anchor_protein" in cols
        self._has_additional_scores = "additional_scores" in cols
        self._peptide_qvalue_col = _first_present(
            cols, ("peptide_qvalue", "peptide_q_value")
        )
        self._protein_qvalue_col = _first_present(
            cols, ("pg_global_qvalue", "protein_qvalue")
        )

        # New-QPX LFQ vs TMT discriminator. In LFQ the ``intensities.label`` names
        # the run each (possibly MBR-transferred) intensity belongs to, so it — not
        # the row's anchor ``run_file_name`` — is the per-run/per-sample key. In TMT
        # the label is a reporter channel and ``run_file_name`` is the run. Detect by
        # checking whether every distinct label is a member of the run_file_name
        # domain (true for LFQ, never for TMT channels).
        self._label_is_run = False
        self._qpx_sdrf_pairs: list[tuple[object, object]] = []
        self._qpx_probe_error: Optional[str] = None
        if self._is_new_qpx:
            try:
                observed = self.parquet_db.execute(
                    "SELECT DISTINCT run_file_name, unnest.label,"
                    " unnest.intensity IS NOT NULL AND unnest.intensity > 0"
                    " FROM parquet_db_raw, UNNEST(intensities) AS intensity_item(unnest)"
                ).fetchall()
                positive = [(run, label) for run, label, keep in observed if keep]
                # The run domain has to come from every row, not from the
                # positive subset. A run whose rows all carry zero intensity
                # would otherwise drop out of the domain, flipping an LFQ file
                # onto the TMT branch, where ``_create_unnest_view`` keys on
                # ``run_file_name`` and folds every MBR-transferred intensity
                # onto its anchor run. The ``> 0`` filter belongs on the label
                # side alone, where it exists to ignore unmapped zero labels.
                run_keys = {
                    _normalize_run_key(run)
                    for (run,) in self.parquet_db.execute(
                        "SELECT DISTINCT run_file_name FROM parquet_db_raw"
                    ).fetchall()
                }
                self._label_is_run = len(positive) > 0 and all(
                    _normalize_run_key(label) in run_keys for _run, label in positive
                )
                if self._label_is_run:
                    self._qpx_sdrf_pairs = list(
                        {(label, "") for _run, label in positive}
                    )
                else:
                    self._qpx_sdrf_pairs = positive
            except duckdb.Error as exc:
                # Swallowing this used to be survivable: the SDRF join fell back
                # to COALESCE(..., run_file_name), so a failed probe merely lost
                # the LFQ label refinement. That fallback is gone, and an empty
                # pair list now yields an empty mapping table, so the LEFT JOIN
                # makes every sample_accession, condition and mixture NULL. The
                # probe is still allowed to fail -- callers that never touch
                # SDRF stay usable -- but enrich_with_sdrf refuses to run on it.
                logger.debug("label-is-run detection failed: %s", exc)
                self._qpx_probe_error = str(exc)

    def _require_qpx_sdrf_pairs(self, sdrf_path: str) -> None:
        """Refuse the new-QPX SDRF join when the run/label probe found nothing.

        The pair list is the only key the new-QPX join matches on, so an empty
        one registers an empty mapping table and the LEFT JOIN silently turns
        sample_accession, condition and mixture into NULL for every row.
        """
        if self._qpx_sdrf_pairs:
            return
        reason = (
            f"the run/label probe failed with {self._qpx_probe_error}"
            if self._qpx_probe_error
            else "the parquet holds no intensity greater than zero"
        )
        raise ValueError(
            f"cannot map QPX intensities onto {sdrf_path}: {reason}. Every "
            "sample_accession, condition and mixture would be NULL."
        )

    def _create_unnest_view(self) -> None:
        """Create the long-format DuckDB view by unnesting intensities."""
        if self._is_new_qpx:
            # LFQ: the intensity's owning run is its label (keep run_file_name only
            # as provenance). TMT: label is a channel, run stays run_file_name.
            run_key = "unnest.label" if self._label_is_run else "run_file_name"
            unnest_sql = (
                f"{run_key} as sample_accession,\n"
                "                    unnest.label as channel,\n"
                "                    unnest.intensity"
            )
            sa_default = run_key
            run_expr = run_key
        else:
            unnest_sql = (
                "unnest.sample_accession,\n"
                "                    unnest.channel,\n"
                "                    unnest.intensity"
            )
            sa_default = "unnest.sample_accession"
            run_expr = self._run_col

        charge_col, run_col = self._charge_col, self._run_col
        # Normalize pg_accessions: extract accession strings from struct if needed
        pg_expr = (
            "list_transform(pg_accessions, x -> x.accession) as pg_accessions"
            if self._pg_accessions_is_struct
            else "pg_accessions"
        )

        # Optional new QPX columns
        extra_cols = ""
        if self._has_is_decoy:
            extra_cols += ",\n                    is_decoy"
        if self._has_anchor_protein:
            extra_cols += ",\n                    anchor_protein"
        extra_cols += _optional_passthrough_projection(
            self._has_additional_scores, "additional_scores", 20
        )
        extra_cols += _optional_qvalue_projection(
            self._peptide_qvalue_col, "peptide_qvalue", 20
        )
        extra_cols += _optional_qvalue_projection(
            self._protein_qvalue_col, "pg_global_qvalue", 20
        )

        self.parquet_db.execute(
            "".join(
                [
                    "CREATE VIEW parquet_db AS SELECT",
                    " sequence, peptidoform, ",
                    pg_expr,
                    ",",
                    " ",
                    charge_col,
                    " as charge,",
                    " ",
                    run_col,
                    " as run_file_name,",
                    ' "unique",',
                    " ",
                    unnest_sql,
                    ",",
                    " ",
                    run_expr,
                    " as run,",
                    " ",
                    sa_default,
                    " as condition,",
                    " 1 as biological_replicate, '1' as fraction,",
                    " split_part(",
                    sa_default,
                    ", '_', 1) as mixture",
                    extra_cols,
                    " FROM parquet_db_raw, UNNEST(intensities) AS intensity_item(unnest)",
                    " WHERE unnest.intensity IS NOT NULL AND unnest.intensity > 0",
                ]
            )
        )

    def enrich_with_sdrf(self, sdrf_path: str) -> None:
        """Enrich parquet data with SDRF metadata (condition, biological_replicate, etc.).

        Parameters
        ----------
        sdrf_path : str
            Path to the SDRF file containing sample metadata.
        """
        sdrf_df = load_sdrf(sdrf_path)

        # Find the condition column (try factor value first, then characteristics)
        condition_col = None
        for col in sdrf_df.columns:
            if "factor value" in col:
                condition_col = col
                break
        if condition_col is None:
            for col in sdrf_df.columns:
                if "organism part" in col and "characteristics" in col:
                    condition_col = col
                    break

        sdrf_labels = sdrf_df.get("comment[label]", pd.Series("", index=sdrf_df.index))
        sdrf_conditions = (
            sdrf_df[condition_col] if condition_col else sdrf_df["source name"]
        )
        sdrf_bioreplicates = sdrf_df.get(
            "characteristics[biological replicate]", pd.Series(1, index=sdrf_df.index)
        )
        sdrf_fractions = sdrf_df.get(
            "comment[fraction identifier]", pd.Series("1", index=sdrf_df.index)
        )
        sdrf_records = [
            {
                "sdrf_record": record,
                "sdrf_data_file": data_file,
                "sdrf_label": "" if pd.isna(label) else label,
                "sdrf_run_key": _normalize_run_key(data_file),
                "sdrf_label_key": _normalize_label_key(label),
                "sdrf_sample_accession": sample,
                "sdrf_condition": condition,
                "sdrf_biological_replicate": bioreplicate,
                "sdrf_fraction": fraction,
            }
            for record, data_file, label, sample, condition, bioreplicate, fraction in zip(
                sdrf_df.index + 1,
                sdrf_df["comment[data file]"],
                sdrf_labels,
                sdrf_df["source name"],
                sdrf_conditions,
                sdrf_bioreplicates,
                sdrf_fractions,
            )
        ]

        # Build format-aware UNNEST SQL
        charge_col = self._charge_col
        run_col = self._run_col

        if self._is_new_qpx:
            if self._label_is_run:
                unnest_cols = (
                    "NULL::VARCHAR as channel,\n                unnest.intensity"
                )
                mapping_run = "unnest.label"
                mapping_label = "''"
                run_value = mapping_run
            else:
                unnest_cols = (
                    "unnest.label as channel,\n                unnest.intensity"
                )
                mapping_run = run_col
                mapping_label = "unnest.label"
                run_value = run_col
            extra_cols = ""
            join_clause = (
                "ON p._mapping_run = s.qpx_run\n"
                "                AND p._mapping_label = s.qpx_label"
            )
            sample_accession = "s.sdrf_sample_accession"
            condition = "s.sdrf_condition"
        else:
            unnest_cols = "unnest.channel,\n                unnest.intensity"
            extra_cols = ",\n                unnest.sample_accession as _legacy_sa"
            join_clause = "ON p._legacy_sa = s.sdrf_sample_accession"
            sample_accession = "COALESCE(s.sdrf_sample_accession, p._legacy_sa)"
            condition = "COALESCE(s.sdrf_condition, p._legacy_sa)"
            run_value = run_col

        # Normalize pg_accessions: extract accession strings from struct if needed
        pg_expr = (
            "list_transform(pg_accessions, x -> x.accession) as pg_accessions"
            if self._pg_accessions_is_struct
            else "pg_accessions"
        )

        if self._is_new_qpx:
            self._require_qpx_sdrf_pairs(sdrf_path)
            sdrf_records = _resolve_qpx_sdrf_mapping(
                sdrf_records, self._qpx_sdrf_pairs, self._label_is_run
            )
            mapping_columns = [
                "qpx_run",
                "qpx_label",
                "sdrf_sample_accession",
                "sdrf_condition",
                "sdrf_biological_replicate",
                "sdrf_fraction",
            ]
        else:
            mapping_columns = [
                "sdrf_sample_accession",
                "sdrf_condition",
                "sdrf_biological_replicate",
                "sdrf_fraction",
            ]
        sdrf_mapping = pd.DataFrame.from_records(
            sdrf_records, columns=mapping_columns
        ).astype("object")
        self.parquet_db.register("sdrf_mapping", sdrf_mapping)

        # Optional new QPX columns
        opt_cols_raw = ""
        if self._has_is_decoy:
            opt_cols_raw += ",\n                    is_decoy"
        if self._has_anchor_protein:
            opt_cols_raw += ",\n                    anchor_protein"
        opt_cols_raw += _optional_passthrough_projection(
            self._has_additional_scores, "additional_scores", 20
        )
        opt_cols_raw += _optional_qvalue_projection(
            self._peptide_qvalue_col, "peptide_qvalue", 20
        )
        opt_cols_raw += _optional_qvalue_projection(
            self._protein_qvalue_col, "pg_global_qvalue", 20
        )
        if self._is_new_qpx:
            opt_cols_raw += (
                f",\n                    {mapping_run} as _mapping_run"
                f",\n                    {mapping_label} as _mapping_label"
            )

        # Create intermediate view for unnested data
        self.parquet_db.execute(
            "".join(
                [
                    "CREATE OR REPLACE VIEW parquet_db_unnested AS SELECT",
                    " sequence, peptidoform, ",
                    pg_expr,
                    ",",
                    " ",
                    charge_col,
                    " as charge,",
                    " ",
                    run_col,
                    " as run_file_name,",
                    ' "unique",',
                    " ",
                    unnest_cols,
                    ",",
                    " ",
                    run_value,
                    " as run",
                    extra_cols,
                    opt_cols_raw,
                    " FROM parquet_db_raw, UNNEST(intensities) AS intensity_item(unnest)",
                    " WHERE unnest.intensity IS NOT NULL AND unnest.intensity > 0",
                ]
            )
        )

        # Optional new QPX columns for final view
        opt_cols_final = ""
        if self._has_is_decoy:
            opt_cols_final += ",\n                p.is_decoy"
        if self._has_anchor_protein:
            opt_cols_final += ",\n                p.anchor_protein"
        opt_cols_final += _optional_passthrough_projection(
            self._has_additional_scores, "p.additional_scores", 16
        )
        opt_cols_final += ",\n                p.peptide_qvalue"
        opt_cols_final += ",\n                p.pg_global_qvalue"

        # Recreate main view with SDRF data joined
        self.parquet_db.execute("DROP VIEW IF EXISTS parquet_db")
        self.parquet_db.execute(
            "".join(
                [
                    "CREATE VIEW parquet_db AS SELECT",
                    " p.sequence, p.peptidoform, p.pg_accessions,",
                    " p.charge, p.run_file_name,",
                    ' p."unique",',
                    " ",
                    sample_accession,
                    " as sample_accession,",
                    " p.channel, p.intensity, p.run,",
                    " ",
                    condition,
                    " as condition,",
                    " COALESCE(CAST(s.sdrf_biological_replicate AS INTEGER), 1) as biological_replicate,",
                    " COALESCE(CAST(s.sdrf_fraction AS VARCHAR), '1') as fraction,",
                    " split_part(",
                    sample_accession,
                    ", '_', 1) as mixture",
                    opt_cols_final,
                    " FROM parquet_db_unnested p LEFT JOIN sdrf_mapping s ",
                    join_clause,
                ]
            )
        )

        if self._is_new_qpx:
            self.samples = list(
                dict.fromkeys(
                    record["sdrf_sample_accession"] for record in sdrf_records
                )
            )
        else:
            self.samples = self.get_unique_samples()
        logger.info("Enriched parquet data with SDRF metadata from %s", sdrf_path)

    @staticmethod
    def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names in the given DataFrame."""
        return df.rename({"protein_accessions": "pg_accessions"}, axis=1)

    @property
    def experimental_inference(
        self,
    ) -> tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]:
        """Infers experimental details from the dataset."""
        self.labels = self.get_unique_labels()
        self.label, self.choice = QuantificationCategory.classify(self.labels)
        self.technical_repetitions = self.get_unique_tec_reps()
        return len(self.technical_repetitions), self.label, self.samples, self.choice

    def get_low_frequency_peptides(self, percentage: float = 0.2) -> tuple:
        """Identifies peptides that occur with low frequency across samples.

        If a filter_builder is set on this Feature instance, it will be used to
        exclude contaminants, decoys, and other artifacts from the frequency
        calculation.

        Parameters
        ----------
        percentage : float
            The frequency threshold. Peptides appearing in less than this
            fraction of samples are considered low frequency. Default is 0.2 (20%).

        Returns
        -------
        tuple
            A tuple of (protein_accession, sequence) pairs for low frequency peptides.
        """
        if self.filter_builder:
            where_clause, where_params = self.filter_builder.build_where_clause()
        else:
            where_clause, where_params = "1=1", []

        # Use anchor_protein directly when available (new QPX), otherwise parse pg_accessions
        if self._has_anchor_protein:
            sql = "".join(
                [
                    'SELECT "sequence", anchor_protein as protein,',
                    ' COUNT(DISTINCT sample_accession) as "count"',
                    " FROM parquet_db WHERE ",
                    where_clause,
                    ' GROUP BY "sequence", anchor_protein',
                ]
            )
            f_table = self.parquet_db.execute(sql, where_params).df()
            f_table.dropna(subset=["protein"], inplace=True)
        else:
            sql = "".join(
                [
                    'SELECT "sequence", "pg_accessions",',
                    ' COUNT(DISTINCT sample_accession) as "count"',
                    " FROM parquet_db WHERE ",
                    where_clause,
                    ' GROUP BY "sequence", "pg_accessions"',
                ]
            )
            f_table = self.parquet_db.execute(sql, where_params).df()
            f_table.dropna(subset=["pg_accessions"], inplace=True)
            try:
                f_table["protein"] = f_table["pg_accessions"].apply(
                    lambda x: x[0].split("|")[1]
                )
            except IndexError:
                f_table["protein"] = f_table["pg_accessions"].apply(lambda x: x[0])
            except Exception as e:
                raise ValueError(
                    "Some errors occurred when parsing pg_accessions column in feature parquet!"
                ) from e

        f_table.set_index(["sequence", "protein"], inplace=True)
        f_table.drop(
            f_table[f_table["count"] >= (percentage * len(self.samples))].index,
            inplace=True,
        )
        f_table.reset_index(inplace=True)
        return tuple(zip(f_table["protein"], f_table["sequence"]))

    @staticmethod
    def csv2parquet(csv):
        """Converts a CSV file to a Parquet file using DuckDB."""
        parquet_path = os.path.splitext(csv)[0] + ".parquet"
        duckdb.read_csv(csv).to_parquet(parquet_path)

    def _validate_columns(self, columns: list) -> str:
        """Validate and quote column names against the parquet_db view schema."""
        valid = {
            r[0] for r in self.parquet_db.execute("DESCRIBE parquet_db").fetchall()
        }
        for c in columns:
            if c not in valid:
                raise ValueError(f"Invalid column name: {c!r}")
        return ",".join(['"' + c + '"' for c in columns])

    def get_report_from_database(self, samples: list, columns: list = None):
        """Retrieves a standardized report from the database for specified samples."""
        if columns is not None:
            cols = self._validate_columns(columns)
        else:
            cols = (
                "* EXCLUDE (additional_scores)" if self._has_additional_scores else "*"
            )
        placeholders = ",".join(["?"] * len(samples))
        score_clause, score_params = self._named_score_clause()
        sql = "".join(
            [
                "SELECT ",
                cols,
                " FROM parquet_db WHERE sample_accession IN (",
                placeholders,
                ")",
                " AND " + score_clause if score_clause else "",
            ]
        )
        database = self.parquet_db.execute(sql, [*samples, *score_params])
        report = database.df()
        return Feature.standardize_df(report)

    def iter_samples(
        self, sample_num: int = 20, columns: list = None
    ) -> Iterator[tuple[list[str], pd.DataFrame]]:
        """Iterates over samples in batches."""
        ref_list = [
            self.samples[i : i + sample_num]
            for i in range(0, len(self.samples), sample_num)
        ]
        for refs in ref_list:
            batch_df = self.get_report_from_database(refs, columns)
            yield refs, batch_df

    def get_unique_samples(self) -> list[str]:
        """Retrieves a list of unique sample accessions from the Parquet database."""
        unique = self.parquet_db.sql(
            "SELECT DISTINCT sample_accession FROM parquet_db"
        ).df()
        return unique["sample_accession"].tolist()

    def get_unique_labels(self) -> list[str]:
        """Retrieves a list of unique channel labels from the Parquet database."""
        unique = self.parquet_db.sql("SELECT DISTINCT channel FROM parquet_db").df()
        return unique["channel"].tolist()

    def get_unique_tec_reps(self) -> list[int]:
        """Retrieves a list of unique technical repetition identifiers.

        Attempts to extract technical replicate numbers from run names in order:
        1. If run name contains '_', try to extract the last part as an integer
        2. If run name is numeric, use it directly
        3. Fall back to sequential integers (1, 2, 3, ...) based on unique runs
        """
        unique = self.parquet_db.sql("SELECT DISTINCT run FROM parquet_db").df()

        try:
            # Try to extract last part after underscore as tech rep
            if unique["run"].str.contains("_").all():
                # Get the last part after splitting by underscore
                last_parts = unique["run"].str.split("_").str.get(-1)
                unique["run"] = last_parts.astype("int")
            else:
                # Try to convert directly to int
                unique["run"] = unique["run"].astype("int")
        except (ValueError, TypeError):
            # Fall back to sequential integers
            unique["run"] = list(range(1, len(unique) + 1))

        return unique["run"].tolist()

    def get_median_map(self) -> dict[str, float]:
        """Computes a median intensity map for samples.

        If a filter_builder is set on this Feature instance, it will be used to
        exclude contaminants, decoys, and other artifacts from the median
        calculation. This ensures normalization factors are computed on clean data.

        Returns
        -------
        dict[str, float]
            A dictionary mapping sample accessions to their normalization factors
            (sample median / global median).
        """
        if self.filter_builder:
            where_clause, where_params = self.filter_builder.build_where_clause()
        else:
            where_clause, where_params = "1=1", []

        # Use SQL aggregation with filtering for efficiency
        sql = "".join(
            [
                "SELECT sample_accession, MEDIAN(intensity) as median_intensity",
                " FROM parquet_db WHERE ",
                where_clause,
                " GROUP BY sample_accession",
            ]
        )
        result = self.parquet_db.execute(sql, where_params).df()

        med_map = dict(zip(result["sample_accession"], result["median_intensity"]))
        global_med = np.median(list(med_map.values()))

        for sample, med in med_map.items():
            med_map[sample] = med / global_med

        return med_map

    def get_report_condition_from_database(
        self, cons: list, columns: list = None
    ) -> pd.DataFrame:
        """Retrieves a standardized report from the database for specified conditions."""
        if columns is not None:
            cols = self._validate_columns(columns)
        else:
            cols = (
                "* EXCLUDE (additional_scores)" if self._has_additional_scores else "*"
            )
        placeholders = ",".join(["?"] * len(cons))
        score_clause, score_params = self._named_score_clause()
        sql = "".join(
            [
                "SELECT ",
                cols,
                " FROM parquet_db WHERE condition IN (",
                placeholders,
                ")",
                " AND " + score_clause if score_clause else "",
            ]
        )
        database = self.parquet_db.execute(sql, [*cons, *score_params])
        report = database.df()
        return Feature.standardize_df(report)

    def iter_conditions(
        self, conditions: int = 10, columns: list = None
    ) -> Iterator[tuple[list[str], pd.DataFrame]]:
        """Iterates over experimental conditions in batches."""
        condition_list = self.get_unique_conditions()
        ref_list = [
            condition_list[i : i + conditions]
            for i in range(0, len(condition_list), conditions)
        ]
        for refs in ref_list:
            batch_df = self.get_report_condition_from_database(refs, columns)
            yield refs, batch_df

    def get_unique_conditions(self) -> list[str]:
        """Retrieves a list of unique experimental conditions from the Parquet database."""
        unique = self.parquet_db.sql("SELECT DISTINCT condition FROM parquet_db").df()
        return unique["condition"].tolist()

    def get_median_map_to_condition(self) -> dict[str, dict[str, float]]:
        """Computes a median intensity map for each experimental condition.

        If a filter_builder is set on this Feature instance, it will be used to
        exclude contaminants, decoys, and other artifacts from the median
        calculation. This ensures normalization factors are computed on clean data.

        Returns
        -------
        dict[str, dict[str, float]]
            A nested dictionary mapping conditions to sample normalization factors.
            For each condition, samples are normalized to the condition mean.
        """
        if self.filter_builder:
            where_clause, where_params = self.filter_builder.build_where_clause()
        else:
            where_clause, where_params = "1=1", []

        # Use SQL aggregation with filtering for efficiency
        sql = "".join(
            [
                "SELECT condition, sample_accession, MEDIAN(intensity) as median_intensity",
                " FROM parquet_db WHERE ",
                where_clause,
                " GROUP BY condition, sample_accession",
            ]
        )
        result = self.parquet_db.execute(sql, where_params).df()

        med_map = {}
        for condition in result["condition"].unique():
            cond_data = result[result["condition"] == condition]
            meds = pd.Series(
                cond_data["median_intensity"].values,
                index=cond_data["sample_accession"].values,
            )
            meds = meds / meds.mean()
            med_map[condition] = meds.to_dict()

        return med_map

    def get_irs_scaling_factors(
        self,
        irs_channel: str,
        irs_stat: str = "median",
        irs_scope: str = "global",
    ) -> dict[int, float]:
        """Compute IRS (Internal Reference Scaling) factors with filtering applied.

        If a filter_builder is set on this Feature instance, it will be used to
        exclude contaminants, decoys, and other artifacts from the IRS calculation.

        Parameters
        ----------
        irs_channel : str
            The channel label to use as internal reference (e.g., "126").
        irs_stat : str
            Statistic to use for computing reference values: "median" or "mean".
        irs_scope : str
            Scope of normalization: "global", "by_mixture", or "two_stage".

        Returns
        -------
        dict[int, float]
            Dictionary mapping technical replicate indices to scaling factors.
        """
        stat_fn = "median" if (irs_stat or "").lower() == "median" else "avg"

        # Build filter conditions for contaminants only (not unique peptide requirement)
        # since IRS uses specific channel which may have different characteristics
        filter_conditions = ["intensity > 0"]
        irs_params: list = []

        if self.filter_builder and self.filter_builder.remove_contaminants:
            contaminant_filter = self.filter_builder.build_contaminant_filter()
            filter_conditions.extend(contaminant_filter[0])
            irs_params.extend(contaminant_filter[1])

        if self.filter_builder and self.filter_builder.min_intensity > 0:
            filter_conditions.append("intensity >= ?")
            irs_params.append(self.filter_builder.min_intensity)

        self._append_named_score_condition(filter_conditions, irs_params)

        # Add channel filter
        filter_conditions.append("channel = ?")
        irs_params.append(irs_channel)
        where_clause = " AND ".join(filter_conditions)

        sql = "".join(
            [
                "SELECT run, ",
                stat_fn,
                "(intensity) as irs_value,",
                " mixture, techreplicate as techrep_guess FROM (",
                " SELECT *,",
                " CASE WHEN position('_' in run) > 0",
                " THEN CAST(split_part(run, '_', 2) AS INTEGER)",
                " ELSE CAST(run AS INTEGER) END AS techreplicate",
                " FROM parquet_db WHERE ",
                where_clause,
                ") GROUP BY run, mixture, techrep_guess",
            ]
        )
        irs_df = self.parquet_db.execute(sql, irs_params).df()

        irs_scale_by_techrep: dict[int, float] = {}

        if len(irs_df.index) > 0:
            irs_df = irs_df[irs_df["irs_value"] > 0]

            if irs_scope.lower() == "by_mixture":
                transform_fn = "median" if stat_fn == "median" else "mean"
                irs_df["mixture_center"] = irs_df.groupby("mixture")[
                    "irs_value"
                ].transform(transform_fn)
                irs_df["scale"] = irs_df["mixture_center"] / irs_df["irs_value"]
            elif irs_scope.lower() == "two_stage":
                transform_fn = "median" if stat_fn == "median" else "mean"
                irs_df["mixture_center"] = irs_df.groupby("mixture")[
                    "irs_value"
                ].transform(transform_fn)
                irs_df["scale_stage1"] = irs_df["mixture_center"] / irs_df["irs_value"]
                mixture_center_df = irs_df[
                    ["mixture", "mixture_center"]
                ].drop_duplicates()
                if stat_fn == "median":
                    global_center = mixture_center_df["mixture_center"].median()
                else:
                    global_center = mixture_center_df["mixture_center"].mean()
                mixture_center_df["scale_stage2"] = (
                    global_center / mixture_center_df["mixture_center"]
                )
                irs_df = irs_df.merge(
                    mixture_center_df, on="mixture", how="left", suffixes=("", "_mix")
                )
                irs_df["scale"] = irs_df["scale_stage1"] * irs_df["scale_stage2"]
            else:
                # Global scope
                if stat_fn == "median":
                    global_center = irs_df["irs_value"].median()
                else:
                    global_center = irs_df["irs_value"].mean()
                irs_df["scale"] = global_center / irs_df["irs_value"]

            irs_scale_by_techrep = dict(
                zip(irs_df["techrep_guess"].tolist(), irs_df["scale"].tolist())
            )

        return irs_scale_by_techrep
