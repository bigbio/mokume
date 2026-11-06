"""Native loader for quantms ``*_msstats_in.csv`` feature tables."""

from __future__ import annotations

import re
from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from mokume.core.constants import load_sdrf
from mokume.model.labeling import IsobaricLabel, QuantificationCategory


_RUN_ALIAS_COLUMNS = (
    "comment[data file]",
    "comment[file uri]",
    "comment[associated file uri]",
    "assay name",
)


def _run_key_sql(column: str) -> str:
    basename = (
        "regexp_extract(replace(lower(coalesce("
        + column
        + ", '')), '\\', '/'), '([^/]+)$', 1)"
    )
    return (
        "CASE WHEN regexp_matches("
        + basename
        + ", '\\.(raw|mzml|wiff|d)(?:$|[._\\s])') "
        "THEN regexp_extract("
        + basename
        + ", '^(.*)(?:\\.(?:raw|mzml|wiff|d))(?:$|[._\\s])', 1) "
        "ELSE regexp_replace(" + basename + ", '\\.(raw|mzml|wiff|d|scan)$', '') END"
    )


_RESOLVED_SQL = """
    CREATE TEMP VIEW _msstats_resolved AS
    SELECT m.*, a.run_file_name AS _run_file_name
    FROM _msstats_input m
    LEFT JOIN _msstats_run_aliases a ON a.alias_key = m._run_key
"""
_VALIDATION_SQL = """
    SELECT COUNT(*),
           COUNT_IF(_run_file_name IS NULL),
           COUNT_IF(NULLIF(trim(_protein), '') IS NULL
                    OR NULLIF(trim(_peptide), '') IS NULL
                    OR upper(regexp_replace(regexp_replace(regexp_replace(
                        trim(_peptide), '\\([^)]*\\)', '', 'g'),
                        '\\[[^]]*\\]', '', 'g'), '[^A-Za-z]', '', 'g')) = ''
                    OR TRY_CAST(_charge AS SMALLINT) IS NULL
                    OR TRY_CAST(_intensity AS REAL) IS NULL
                    OR NOT isfinite(TRY_CAST(_intensity AS REAL)))
    FROM _msstats_resolved
"""
_RAW_FEATURE_SQL = """
    CREATE TEMP TABLE parquet_db_raw AS
    WITH normalized AS (
        SELECT upper(regexp_replace(regexp_replace(regexp_replace(
                   trim(_peptide), '\\([^)]*\\)', '', 'g'),
                   '\\[[^]]*\\]', '', 'g'), '[^A-Za-z]', '', 'g')) AS sequence,
               trim(_peptide) AS peptidoform,
               trim(_protein) AS protein_name,
               list_transform(string_split(trim(_protein), ';'),
                              lambda accession: trim(accession)) AS pg_accessions,
               TRY_CAST(_charge AS SMALLINT) AS charge,
               r._run_file_name AS run_file_name,
               {label_expression} AS intensity_label,
               TRY_CAST(_intensity AS REAL) AS intensity
        FROM _msstats_resolved r{channel_join}
    ), annotated AS (
        SELECT *, COUNT(DISTINCT protein_name) OVER (PARTITION BY sequence) = 1
                  AND list_count(pg_accessions) = 1 AS is_unique
        FROM normalized
    )
    SELECT sequence, peptidoform, pg_accessions, charge, run_file_name,
           is_unique AS "unique",
           [struct_pack(label := intensity_label, intensity := intensity)] AS intensities
    FROM annotated
"""
_LFQ_FEATURE_SQL = _RAW_FEATURE_SQL.format(
    label_expression="r._run_file_name", channel_join=""
)
_ISOBARIC_FEATURE_SQL = _RAW_FEATURE_SQL.format(
    label_expression="cm.canonical_label",
    channel_join=(
        " LEFT JOIN _msstats_channel_map cm ON "
        "r._channel IS NOT DISTINCT FROM cm.source_channel"
    ),
)


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _resolve_column(
    columns: Iterable[str], aliases: tuple[str, ...], required: bool = True
) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for alias in aliases:
        if column := by_lower.get(alias.lower()):
            return column
    if required:
        raise ValueError(f"MSstats input is missing required column: {aliases[0]}")
    return None


def _label_key(value: object) -> str:
    label = "" if pd.isna(value) else str(value).strip()
    for part in label.split(";"):
        if part.strip().startswith("NT="):
            label = part.strip()[3:].strip()
            break
    key = label.lower()
    return (
        "label free sample"
        if key in {"lfq", "label-free", "label free sample"}
        else key
    )


def _channel_position(value: object) -> int | None:
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if not isfinite(number) or number < 1 or number != int(number):
        return None
    return int(number)


def _isobaric_schemes(
    labels: set[str],
) -> tuple[QuantificationCategory, tuple[IsobaricLabel, ...]]:
    if labels and all(label.startswith("tmt") for label in labels):
        return QuantificationCategory.TMT, (
            IsobaricLabel.TMT6plex,
            IsobaricLabel.TMT10plex,
            IsobaricLabel.TMT11plex,
            IsobaricLabel.TMT16plex,
        )
    if labels and all(label.startswith("itraq") for label in labels):
        return QuantificationCategory.ITRAQ, (
            IsobaricLabel.ITRAQ4plex,
            IsobaricLabel.ITRAQ8plex,
        )
    raise ValueError(f"Cannot infer LFQ, TMT, or iTRAQ labeling from {labels}")


def _position_labels(scheme: IsobaricLabel) -> dict[int, str]:
    return {position: name for name, position in scheme.channels().items()}


def _matches_positions(
    scheme: IsobaricLabel, labels: set[str], positions: list[int]
) -> bool:
    position_labels = _position_labels(scheme)
    return all(
        (label := position_labels.get(position)) is not None
        and _label_key(label) in labels
        for position in positions
    )


def _select_isobaric_scheme(
    labels: set[str], schemes: tuple[IsobaricLabel, ...], positions: list[int]
) -> IsobaricLabel:
    candidates = [
        scheme
        for scheme in schemes
        if labels <= {_label_key(label) for label in scheme.channels()}
    ]
    exact = [scheme for scheme in candidates if len(scheme.channels()) == len(labels)]
    if exact:
        return exact[0]

    candidates = [
        scheme for scheme in candidates if _matches_positions(scheme, labels, positions)
    ]
    if not candidates:
        raise ValueError(
            "Cannot infer the isobaric plex from SDRF labels and MSstats channels"
        )

    signatures = {
        tuple(_position_labels(scheme)[position] for position in positions)
        for scheme in candidates
    }
    if len(signatures) != 1:
        raise ValueError(
            "The isobaric plex is ambiguous for the supplied SDRF and MSstats channels"
        )
    return min(candidates, key=lambda scheme: len(scheme.channels()))


def _infer_labeling(
    connection: duckdb.DuckDBPyConnection, sdrf: pd.DataFrame
) -> tuple[QuantificationCategory, IsobaricLabel | None]:
    labels = {_label_key(value) for value in sdrf["comment[label]"].dropna()}
    labels.discard("")
    if labels == {"label free sample"}:
        return QuantificationCategory.LFQ, None

    category, schemes = _isobaric_schemes(labels)
    raw_channels = [
        value
        for (value,) in connection.execute(
            "SELECT DISTINCT _channel FROM _msstats_input"
        ).fetchall()
    ]
    positions = sorted(
        {
            position
            for value in raw_channels
            if (position := _channel_position(value)) is not None
        }
    )
    return category, _select_isobaric_scheme(labels, schemes, positions)


def _run_key(value: object) -> str:
    if pd.isna(value):
        return ""
    basename = str(value).strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    basename = basename.lower()
    match = re.match(r"^(.*)(?:\.(?:raw|mzml|wiff|d))(?:$|[._\s])", basename)
    if match:
        return match.group(1)
    return re.sub(r"\.(?:raw|mzml|wiff|d|scan)$", "", basename)


def _register_run_aliases(
    connection: duckdb.DuckDBPyConnection, sdrf: pd.DataFrame
) -> None:
    if "comment[data file]" not in sdrf.columns:
        raise ValueError("SDRF input is missing required column: comment[data file]")

    aliases: dict[str, set[str]] = defaultdict(set)
    for _, record in sdrf.iterrows():
        data_file = record["comment[data file]"]
        if pd.isna(data_file) or not str(data_file).strip():
            raise ValueError("SDRF comment[data file] contains an empty value")
        run_file_name = str(data_file).strip()
        for column in _RUN_ALIAS_COLUMNS:
            if column in sdrf.columns and (key := _run_key(record[column])):
                aliases[key].add(run_file_name)

    records = [
        {"alias_key": key, "run_file_name": next(iter(values))}
        for key, values in aliases.items()
        if len(values) == 1
    ]
    if not records:
        raise ValueError("SDRF contains no unambiguous run aliases")
    alias_table = pd.DataFrame.from_records(records).astype("object")
    connection.register("_msstats_run_aliases", alias_table)


def _load_msstats_input(
    connection: duckdb.DuckDBPyConnection, msstats_path: str
) -> bool:
    source = connection.read_csv(msstats_path, header=True, all_varchar=True)
    columns = source.columns
    resolved = {
        "_protein": _resolve_column(columns, ("ProteinName",)),
        "_peptide": _resolve_column(columns, ("PeptideSequence",)),
        "_charge": _resolve_column(columns, ("Charge", "PrecursorCharge")),
        "_intensity": _resolve_column(columns, ("Intensity",)),
        "_run": _resolve_column(columns, ("Run",), required=False),
        "_reference": _resolve_column(columns, ("Reference",), required=False),
        "_channel": _resolve_column(columns, ("Channel",), required=False),
    }
    if resolved["_run"] is None and resolved["_reference"] is None:
        raise ValueError("MSstats input requires a Run or Reference column")

    projection = [
        (
            f"CAST({_quote_identifier(column)} AS VARCHAR) AS {alias}"
            if column
            else f"NULL::VARCHAR AS {alias}"
        )
        for alias, column in resolved.items()
    ]
    # quantms writes Run as an OpenMS design index and the data file in
    # Reference, so Run names the file only when there is no Reference column.
    file_column = resolved["_reference"] or resolved["_run"]
    projection.append(_run_key_sql(_quote_identifier(file_column)) + " AS _run_key")
    source.project(", ".join(projection)).create_view("_msstats_input")
    return resolved["_channel"] is not None


def _channel_labels(
    connection: duckdb.DuckDBPyConnection,
    sdrf: pd.DataFrame,
    label_scheme: IsobaricLabel,
) -> dict[str, str]:
    positions = {position: label for label, position in label_scheme.channels().items()}
    canonical = {label.lower(): label for label in positions.values()}
    declared = {
        key: canonical[key]
        for value in sdrf["comment[label]"].dropna()
        if (key := _label_key(value)) in canonical
    }
    values = connection.execute(
        "SELECT DISTINCT _channel FROM _msstats_input"
    ).fetchall()

    resolved: dict[str, str] = {}
    for (value,) in values:
        text = "" if value is None else str(value).strip()
        position = _channel_position(text)
        label = positions.get(position) if position is not None else None
        if label is None:
            label = declared.get(_label_key(text))
        if label is None:
            raise ValueError(
                f"MSstats channel {text!r} is not declared by the SDRF plex"
            )
        resolved[text] = label
    return resolved


def _validate_resolved_rows(connection: duckdb.DuckDBPyConnection) -> None:
    total, unresolved, invalid = connection.execute(_VALIDATION_SQL).fetchone()
    if total == 0:
        raise ValueError("MSstats input contains no rows")
    if unresolved:
        raise ValueError(
            f"{unresolved} MSstats rows cannot be mapped to an SDRF data file"
        )
    if invalid:
        raise ValueError(f"{invalid} MSstats rows contain invalid required values")


def _prepare_isobaric_channels(
    connection: duckdb.DuckDBPyConnection,
    sdrf: pd.DataFrame,
    label_scheme: IsobaricLabel | None,
    has_channel: bool,
) -> None:
    if label_scheme is None:
        raise ValueError("Cannot determine the isobaric plex from SDRF labels")
    if not has_channel:
        raise ValueError("MSstats input is missing required column: Channel")

    channels = _channel_labels(connection, sdrf, label_scheme)
    channel_table = pd.DataFrame(
        {
            "source_channel": list(channels),
            "canonical_label": list(channels.values()),
        }
    ).astype("object")
    connection.register("_msstats_channel_map", channel_table)
    declared_pairs = {
        (str(row["comment[data file]"]).strip(), _label_key(row["comment[label]"]))
        for _, row in sdrf.iterrows()
    }
    observed_pairs = connection.execute(
        "SELECT DISTINCT _run_file_name, _channel FROM _msstats_resolved"
    ).fetchall()
    missing = [
        (run_file, channels[str(raw_channel).strip()])
        for run_file, raw_channel in observed_pairs
        if (run_file, _label_key(channels[str(raw_channel).strip()]))
        not in declared_pairs
    ]
    if missing:
        run_file, label = missing[0]
        raise ValueError(
            f"MSstats run {run_file!r} and channel {label!r} have no SDRF sample"
        )


def create_msstats_feature_table(
    connection: duckdb.DuckDBPyConnection,
    msstats_path: str,
    sdrf_path: str,
) -> None:
    """Create ``parquet_db_raw`` from quantms MSstats and authoritative SDRF data."""
    if not Path(msstats_path).is_file():
        raise FileNotFoundError(f"MSstats file not found: {msstats_path}")
    if not Path(sdrf_path).is_file():
        raise FileNotFoundError(f"SDRF file not found: {sdrf_path}")

    has_channel = _load_msstats_input(connection, msstats_path)
    sdrf = load_sdrf(sdrf_path)
    if "comment[label]" not in sdrf.columns:
        raise ValueError("SDRF input is missing required column: comment[label]")
    category, label_scheme = _infer_labeling(connection, sdrf)
    _register_run_aliases(connection, sdrf)
    connection.execute(_RESOLVED_SQL)
    _validate_resolved_rows(connection)

    isobaric = category in {QuantificationCategory.TMT, QuantificationCategory.ITRAQ}
    if isobaric:
        _prepare_isobaric_channels(connection, sdrf, label_scheme, has_channel)
    connection.execute(_ISOBARIC_FEATURE_SQL if isobaric else _LFQ_FEATURE_SQL)
