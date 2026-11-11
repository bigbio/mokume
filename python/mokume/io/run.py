"""Read the self-contained sample mapping in a QPX run view."""

import pandas as pd
import pyarrow.parquet as pq


def load_run(path):
    """Return run/sample records in the same table shape as ``load_sdrf``."""
    rows = []
    for run in pq.read_table(path).to_pylist():
        for sample in run["samples"] or []:
            rows.append(
                {
                    "source name": sample["sample_accession"],
                    "comment[data file]": run["run_file_name"],
                    "comment[label]": sample["label"],
                    "comment[fraction identifier]": run.get("fraction") or "1",
                    "characteristics[biological replicate]": sample.get(
                        "biological_replicate"
                    )
                    or 1,
                    "comment[technical replicate]": sample.get("technical_replicate")
                    or 1,
                    "factor value[condition]": sample["sample_accession"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "source name",
            "comment[data file]",
            "comment[label]",
            "comment[fraction identifier]",
            "characteristics[biological replicate]",
            "comment[technical replicate]",
            "factor value[condition]",
        ],
    )
