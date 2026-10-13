"""Group-level contaminant rule: drop only all-contaminant (or decoy) groups."""

import duckdb
import pandas as pd

from mokume.core.contaminants import (
    DEFAULT_CONTAMINANT_PATTERNS,
    contaminant_group_mask,
)
from mokume.io.feature import SQLFilterBuilder
from mokume.preprocessing.aggregation import remove_contaminants_entrapments_decoys
from mokume.preprocessing.filters.protein import ContaminantFilter

GROUPS = [
    "CONTAM_P05787;P05787",  # human twin: kept
    "CONTAM_P00761",  # trypsin: dropped
    "CONTAM_P02769;CONTAM_P02768",  # all contaminant: dropped
    "P02768;CONTAM_P02769",  # mixed: kept
    "P12345;DECOY_P99999",  # decoy member: dropped
    "P12345",
]
KEPT = ["CONTAM_P05787;P05787", "P02768;CONTAM_P02769", "P12345"]


def test_mask_rule():
    mask = contaminant_group_mask(pd.Series(GROUPS), DEFAULT_CONTAMINANT_PATTERNS)
    assert list(pd.Series(GROUPS)[~mask]) == KEPT


def test_contaminant_filter_keeps_twins():
    df = pd.DataFrame({"ProteinName": GROUPS})
    out, _ = ContaminantFilter(DEFAULT_CONTAMINANT_PATTERNS).apply(df)
    assert list(out["ProteinName"]) == KEPT


def test_remove_contaminants_entrapments_decoys_keeps_twins():
    df = pd.DataFrame({"ProteinName": GROUPS})
    out = remove_contaminants_entrapments_decoys(df, "ProteinName")
    assert list(out["ProteinName"]) == KEPT


def test_sql_filter_keeps_twins():
    clause, params = SQLFilterBuilder(
        min_peptide_length=0, require_unique=False
    ).build_where_clause()
    con = duckdb.connect()
    con.execute("CREATE TABLE t (pg_accessions VARCHAR[], intensity DOUBLE)")
    for group in GROUPS:
        con.execute("INSERT INTO t VALUES (?, 1.0)", [group.split(";")])
    rows = con.execute(f"SELECT pg_accessions FROM t WHERE {clause}", params).fetchall()
    assert [";".join(row[0]) for row in rows] == KEPT
