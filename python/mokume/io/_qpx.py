"""Column and footer based compatibility helpers for QPX feature files."""

import json


def measured_intensity_clause(columns, entry="unnest"):
    """Keep measured labels, treating an absent or null imputed list as empty."""
    if "imputed_labels" not in columns:
        return "TRUE"
    return f"NOT COALESCE(list_contains(imputed_labels, {entry}.label), FALSE)"


def _complete_qvalue(connection, path, name):
    """Prefer typed q-value columns, including columns containing nulls."""
    if path is None or name not in {
        "peptide_qvalue",
        "precursor_qvalue",
        "global_qvalue",
    }:
        return False
    columns = connection.execute("DESCRIBE parquet_db_raw").fetchall()
    return any(column[0] == name for column in columns)


def resolve_score_source(connection, path, name):
    """Resolve a scalar score expression; list scores keep the existing reader."""
    if _complete_qvalue(connection, path, name):
        return f'"{name}"', False
    types = dict(
        (row[0], row[1])
        for row in connection.execute("DESCRIBE parquet_db_raw").fetchall()
    )
    score_type = types.get("additional_scores", "").upper()
    if not score_type.startswith("STRUCT(") or score_type.endswith("[]"):
        return None
    rows = connection.execute(
        "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'qpx.struct_lists'::BLOB",
        [path],
    ).fetchall()
    if not rows:
        raise ValueError("STRUCT additional_scores requires qpx.struct_lists footer")
    layout = json.loads(rows[0][0]).get("additional_scores", [])
    entry = next((item for item in layout if item["name"] == name), None)
    if entry is None:
        raise ValueError(f"score `{name}` is not in qpx.struct_lists layout")
    field = entry["name"].replace('"', '""')
    return f'additional_scores."{field}"', entry["higher_better"]
