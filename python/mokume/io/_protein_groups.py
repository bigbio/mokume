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
    for members in parsed:
        key = tuple(sorted(members))
        candidate = (";".join(members), tuple(members.values()))
        preferred[key] = min(preferred.get(key, candidate), candidate)
    return [list(preferred[tuple(sorted(members))][1]) for members in parsed]


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
    relation = connection.table(source)
    types = dict(zip(relation.columns, map(str, relation.types)))
    expression = "pg_accessions"
    if "STRUCT" in types["pg_accessions"].upper():
        expression = "list_transform(pg_accessions, x -> x.accession)"
    groups = [row[0] for row in relation.project(expression).distinct().fetchall()]
    mapping = pa.table(
        {
            "stored": pa.array(groups, type=pa.list_(pa.string())),
            "members": pa.array(canonical_members(groups), type=pa.list_(pa.string())),
        }
    )
    connection.register("qpx_group_orders", mapping)
    relation.join(
        connection.table("qpx_group_orders"), f"stored = {expression}", how="left"
    ).project(
        "* EXCLUDE (pg_accessions, stored, members), members AS pg_accessions"
    ).create_view(target)


def protein_peptide_counts(path):
    """Count canonical sequences per matrix protein-group identity for DEqMS."""
    rows = pd.read_parquet(path, columns=["pg_accessions", "sequence"])
    return rows.groupby(protein_group_names(rows["pg_accessions"]))[
        "sequence"
    ].nunique()
