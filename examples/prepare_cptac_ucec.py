#!/usr/bin/env python3
"""Build the PDC000125 analysis SDRF from PDC metadata and a QPX parquet."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests


PDC_GRAPHQL = "https://pdc.cancer.gov/graphql"
EXPECTED_CONDITIONS = Counter(
    {"Primary Tumor": 104, "Solid Tissue Normal": 49, "Reference": 17}
)
CHANNELS = (
    ("tmt_126", "TMT10-126"),
    ("tmt_127n", "TMT10-127N"),
    ("tmt_127c", "TMT10-127C"),
    ("tmt_128n", "TMT10-128N"),
    ("tmt_128c", "TMT10-128C"),
    ("tmt_129n", "TMT10-129N"),
    ("tmt_129c", "TMT10-129C"),
    ("tmt_130n", "TMT10-130N"),
    ("tmt_130c", "TMT10-130C"),
    ("tmt_131", "TMT10-131"),
)
RUN_PATTERN = re.compile(
    r"^(?P<plex>\d{2})CPTAC_UCEC_W_PNNL_\d{8}_B\dS\d_f(?P<fraction>\d{2})$"
)
DESIGN_PATTERN = re.compile(r"^(?P<plex>\d{2})CPTAC_UCEC_Proteome_PNNL_\d{8}$")

EXPERIMENTAL_DESIGN_QUERY = """
{
  studyExperimentalDesign(pdc_study_id: "PDC000125", acceptDUA: true) {
    plex_dataset_name
    number_of_fractions
    tmt_126 { aliquot_id aliquot_submitter_id }
    tmt_127n { aliquot_id aliquot_submitter_id }
    tmt_127c { aliquot_id aliquot_submitter_id }
    tmt_128n { aliquot_id aliquot_submitter_id }
    tmt_128c { aliquot_id aliquot_submitter_id }
    tmt_129n { aliquot_id aliquot_submitter_id }
    tmt_129c { aliquot_id aliquot_submitter_id }
    tmt_130n { aliquot_id aliquot_submitter_id }
    tmt_130c { aliquot_id aliquot_submitter_id }
    tmt_131 { aliquot_id aliquot_submitter_id }
  }
}
"""

BIOSPECIMEN_QUERY = """
{
  biospecimenPerStudy(pdc_study_id: "PDC000125", acceptDUA: true) {
    aliquot_id
    aliquot_submitter_id
    case_submitter_id
    sample_type
    disease_type
    primary_site
    taxon
  }
}
"""

SDRF_COLUMNS = (
    "source name",
    "characteristics[organism]",
    "characteristics[organism part]",
    "characteristics[disease]",
    "characteristics[individual]",
    "characteristics[sample type]",
    "characteristics[aliquot]",
    "characteristics[pooled sample]",
    "assay name",
    "technology type",
    "comment[data file]",
    "comment[fraction identifier]",
    "comment[label]",
    "factor value[condition]",
)


def query_pdc(query: str) -> list[dict[str, object]]:
    """Execute one PDC GraphQL query and return its sole result collection."""
    response = requests.post(PDC_GRAPHQL, json={"query": query}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"PDC GraphQL returned errors: {payload['errors']}")
    return next(iter(payload["data"].values()))


def qpx_runs(path: Path, threads: int) -> dict[int, list[tuple[int, str]]]:
    """Group validated QPX run names by plex and fraction."""
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    runs: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["run_file_name"], use_threads=True):
        runs.update(batch.column(0).to_pylist())

    by_plex: dict[int, list[tuple[int, str]]] = {}
    for run in runs:
        match = RUN_PATTERN.fullmatch(run)
        if match is None:
            raise ValueError(f"unexpected PDC000125 QPX run name: {run}")
        plex = int(match.group("plex"))
        fraction = int(match.group("fraction"))
        by_plex.setdefault(plex, []).append((fraction, run))

    for plex_runs in by_plex.values():
        plex_runs.sort()
    return by_plex


def design_by_plex(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    """Index validated PDC experimental-design records by plex number."""
    result = {}
    for row in rows:
        name = str(row["plex_dataset_name"])
        match = DESIGN_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected PDC000125 plex name: {name}")
        result[int(match.group("plex"))] = row
    return result


def biological_fields(
    channel: dict[str, str], biospecimens: dict[str, dict[str, str]]
) -> tuple[dict[str, str], str]:
    """Translate one TMT channel assignment into SDRF biological fields."""
    if channel["aliquot_submitter_id"] == "Ref":
        return (
            {
                "characteristics[organism]": "Homo sapiens",
                "characteristics[organism part]": "uterus",
                "characteristics[disease]": "Not Reported",
                "characteristics[individual]": "Pool",
                "characteristics[sample type]": "Reference",
                "characteristics[aliquot]": "Ref",
                "characteristics[pooled sample]": "pooled",
            },
            "Reference",
        )

    specimen = biospecimens[channel["aliquot_id"]]
    condition = specimen["sample_type"]
    if condition not in {"Primary Tumor", "Solid Tissue Normal"}:
        raise ValueError(
            f"unexpected sample type for {channel['aliquot_submitter_id']}: {condition}"
        )
    return (
        {
            "characteristics[organism]": specimen["taxon"],
            "characteristics[organism part]": specimen["primary_site"],
            "characteristics[disease]": specimen["disease_type"],
            "characteristics[individual]": specimen["case_submitter_id"],
            "characteristics[sample type]": condition,
            "characteristics[aliquot]": specimen["aliquot_submitter_id"],
            "characteristics[pooled sample]": "not pooled",
        },
        condition,
    )


def channel_assignment(
    design: dict[str, object], channel_key: str, plex: int
) -> dict[str, str]:
    """Return the single PDC assignment for one plex channel."""
    assignments = design[channel_key]
    if not isinstance(assignments, list) or len(assignments) != 1:
        raise ValueError(f"plex {plex:02d} has invalid {channel_key} assignment")
    return assignments[0]


def build_plex_rows(
    plex: int,
    plex_runs: list[tuple[int, str]],
    design: dict[str, object],
    biospecimens: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Build every fraction/channel SDRF row for one validated plex."""
    if len(plex_runs) != int(str(design["number_of_fractions"])):
        raise ValueError(
            f"plex {plex:02d} has {len(plex_runs)} QPX runs; "
            f"PDC declares {design['number_of_fractions']}"
        )
    rows = []
    conditions = {}
    for channel_index, (channel_key, label) in enumerate(CHANNELS, start=1):
        channel = channel_assignment(design, channel_key, plex)
        fields, condition = biological_fields(channel, biospecimens)
        source_name = f"PDC000125-p{plex:02d}_{channel_index}"
        conditions[source_name] = condition
        rows.extend(
            {
                "source name": source_name,
                **fields,
                "assay name": run,
                "technology type": "proteomic profiling by mass spectrometry",
                "comment[data file]": f"{run}.raw",
                "comment[fraction identifier]": str(fraction),
                "comment[label]": label,
                "factor value[condition]": condition,
            }
            for fraction, run in plex_runs
        )
    return rows, conditions


def build_rows(
    runs: dict[int, list[tuple[int, str]]],
    designs: dict[int, dict[str, object]],
    biospecimens: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Build and validate the complete PDC000125 analysis SDRF."""
    if set(runs) != set(designs):
        raise ValueError(
            f"QPX/API plex mismatch: QPX={sorted(runs)}, PDC={sorted(designs)}"
        )

    rows = []
    sample_conditions = {}
    for plex in sorted(runs):
        plex_rows, plex_conditions = build_plex_rows(
            plex, runs[plex], designs[plex], biospecimens
        )
        rows.extend(plex_rows)
        sample_conditions.update(plex_conditions)

    observed = Counter(sample_conditions.values())
    if observed != EXPECTED_CONDITIONS:
        raise ValueError(f"unexpected PDC000125 condition counts: {dict(observed)}")
    if len(rows) != 4_080:
        raise ValueError(f"expected 4,080 SDRF rows, found {len(rows):,}")
    return rows


def parse_args() -> argparse.Namespace:
    """Parse the QPX source and destination SDRF paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_parquet", type=Path)
    parser.add_argument("output_sdrf", type=Path)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    """Fetch PDC metadata and write the validated analysis SDRF."""
    args = parse_args()
    runs = qpx_runs(args.feature_parquet, args.threads)
    designs = design_by_plex(query_pdc(EXPERIMENTAL_DESIGN_QUERY))
    biospecimens = {str(row["aliquot_id"]): row for row in query_pdc(BIOSPECIMEN_QUERY)}
    rows = build_rows(runs, designs, biospecimens)

    args.output_sdrf.parent.mkdir(parents=True, exist_ok=True)
    with args.output_sdrf.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SDRF_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"wrote {len(rows):,} rows to {args.output_sdrf}: "
        "17 plexes, 104 tumors, 49 normals, 17 pooled references"
    )


if __name__ == "__main__":
    main()
