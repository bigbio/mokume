"""Stable protein-group identities shared by feature and TissueMap readers."""

import pandas as pd
import pyarrow as pa


def accession_name(accession):
    """Return the accession component used by protein matrices."""
    fields = accession.strip().split("|")
    return fields[1] if len(fields) == 3 else accession.strip()


def ordered_members(group):
    """Deduplicate members without changing their stored order."""
    members = {}
    for item in group:
        accession = item["accession"] if isinstance(item, dict) else item
        name = accession_name(accession)
        if name:
            members.setdefault(name, accession)
    return members


def canonical_members(groups):
    """Choose the smallest observed name for each unordered member set."""
    preferred = {}
    parsed = [ordered_members(group) if group is not None else {} for group in groups]
    # Key by stored accession, not by name: a decoy shares its target's name.
    keys = [tuple(sorted(a.strip() for a in members.values())) for members in parsed]
    for key, members in zip(keys, parsed):
        candidate = (";".join(members), tuple(members.values()))
        preferred[key] = min(preferred.get(key, candidate), candidate)
    return [list(preferred[key][1]) for key in keys]


def protein_group_names(groups):
    """Resolve matrix names over all observed orders in a Series."""
    return pd.Series(
        [
            ";".join(map(accession_name, members))
            for members in canonical_members(groups)
        ],
        index=groups.index,
    )


def create_group_view(connection, source, target):
    """Canonicalize only distinct member lists, retaining all other raw columns."""
    # execute() rather than the relation API: DuckDB 2.0 refuses relation calls
    # while an earlier execute() result on the connection is unfinished.
    types = {
        row[0]: row[1] for row in connection.execute(f"DESCRIBE {source}").fetchall()
    }
    expression = "pg_accessions"
    if "STRUCT" in types["pg_accessions"].upper():
        expression = "list_transform(pg_accessions, lambda x: x.accession)"
    groups = [
        row[0]
        for row in connection.execute(
            "".join(["SELECT DISTINCT ", expression, " FROM ", source])
        ).fetchall()
    ]
    mapping = pa.table(
        {
            "stored": pa.array(groups, type=pa.list_(pa.string())),
            "members": pa.array(canonical_members(groups), type=pa.list_(pa.string())),
        }
    )
    connection.register("qpx_group_orders", mapping)
    connection.execute(
        "".join(
            [
                "CREATE OR REPLACE VIEW ",
                target,
                " AS SELECT * EXCLUDE (pg_accessions, stored, members),",
                " members AS pg_accessions FROM ",
                source,
                " LEFT JOIN qpx_group_orders ON stored = ",
                expression,
            ]
        )
    )


def protein_peptide_counts(path):
    """Count canonical sequences per matrix protein-group identity for DEqMS."""
    rows = pd.read_parquet(path, columns=["pg_accessions", "sequence"])
    return rows.groupby(protein_group_names(rows["pg_accessions"]))[
        "sequence"
    ].nunique()
