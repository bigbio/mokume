"""
QPX adapter — thin wrapper over qpx for reading and validation.

All reading, validation, and dataset assembly are delegated to qpx.
Mokume only consumes qpx APIs and applies column-name mapping so
algorithm code receives the names it expects (reference_file_name,
precursor_charge, pg_accessions, etc.).

See docs/plans/qpx-migration-principles.md for division of responsibilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from mokume.model.labeling import QuantificationCategory, IsobaricLabel

logger = logging.getLogger(__name__)

# Column mapping: QPX names -> mokume-internal names used by algorithms
QPX_TO_MOKUME_COLS = {
    "run_file_name": "reference_file_name",
    "charge": "precursor_charge",
    "anchor_protein": "pg_accessions",  # we expose as list of one element for compatibility
}


def _map_qpx_long_to_mokume(df: pd.DataFrame) -> pd.DataFrame:
    """Map QPX long-form columns to names mokume algorithms expect."""
    out = df.copy()
    if "run_file_name" in out.columns:
        out["reference_file_name"] = out["run_file_name"]
        out["run"] = out["run_file_name"]
    if "charge" in out.columns:
        out["precursor_charge"] = out["charge"]
    if "anchor_protein" in out.columns:
        import ast
        def _parse_pg(x):
            if pd.isna(x):
                return []
            s = str(x)
            if s.startswith("["):
                try:
                    return ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    pass
            return [s]
        out["pg_accessions"] = out["anchor_protein"].apply(_parse_pg)
    if "label" in out.columns:
        out["channel"] = out["label"]
    # biological_replicate: from QPX run.samples or default to 1
    if "biological_replicate" not in out.columns:
        out["biological_replicate"] = 1
    else:
        out["biological_replicate"] = out["biological_replicate"].fillna(1).astype(int)
    # fraction: from QPX run table or default to "1"
    if "fraction" not in out.columns:
        out["fraction"] = "1"
    else:
        out["fraction"] = out["fraction"].fillna("1").astype(str)
    return out


def open_qpx(path: str | Path, structures: Optional[list[str]] = None):
    """Open a QPX dataset directory using qpx. All reading/validation done by qpx."""
    try:
        import qpx
    except ImportError as e:
        raise ImportError(
            "QPX directory input requires the 'qpx' package. Install with: pip install qpx"
        ) from e
    return qpx.Dataset(path, structures=structures or ["feature", "sample", "run"])


def open_qpx_feature_single(path: str | Path):
    """Open a single feature.parquet file using qpx. All reading done by qpx."""
    try:
        import qpx
    except ImportError as e:
        raise ImportError(
            "Single-file QPX mode requires the 'qpx' package. Install with: pip install qpx"
        ) from e
    return qpx.read_feature(str(path))


class QpxFeatureAdapter:
    """
    Feature-like facade over qpx.Dataset for quantification.

    Delegates all reading, iteration, and validation to qpx. Only applies
    column-name mapping so mokume algorithms receive expected column names.
    """

    def __init__(self, dataset, filter_builder=None):
        """
        Parameters
        ----------
        dataset : qpx.Dataset
            Opened QPX dataset (must have feature; run/sample optional but recommended).
        filter_builder : object, optional
            SQLFilterBuilder-compatible object; filtering is applied in Python
            after fetching from qpx (qpx does not accept mokume's WHERE strings).
        """
        self._ds = dataset
        self.filter_builder = filter_builder
        self._feature = dataset.feature
        self._run = getattr(dataset, "run", None)
        self._sample = getattr(dataset, "sample", None)
        self._samples: Optional[list[str]] = None
        self._long_form: Optional[pd.DataFrame] = None

    def _get_long_form(self) -> pd.DataFrame:
        """Get feature data in long form from qpx; cache and map column names."""
        if self._long_form is not None:
            return self._long_form
        if self._feature is None:
            raise ValueError("QPX dataset has no feature structure")
        # Use qpx API: for_quantification when run is present, else peptide_intensities
        if self._run is not None:
            result = self._feature.for_quantification(self._ds)
        else:
            result = self._feature.peptide_intensities()
            # peptide_intensities has label but not sample_accession; use label as sample
            df = result.to_df()
            if "sample_accession" not in df.columns and "label" in df.columns:
                df = df.copy()
                df["sample_accession"] = df["label"].astype(str)
            self._long_form = _map_qpx_long_to_mokume(df)
            if "unique" not in self._long_form.columns:
                self._long_form["unique"] = 1
            return self._long_form
        df = result.to_df()
        self._long_form = _map_qpx_long_to_mokume(df)
        if "unique" not in self._long_form.columns:
            self._long_form["unique"] = 1
        return self._long_form

    @property
    def samples(self) -> list[str]:
        """Unique sample accessions from qpx (run/sample or from feature labels)."""
        if self._samples is not None:
            return self._samples
        df = self._get_long_form()
        self._samples = df["sample_accession"].dropna().unique().tolist()
        return self._samples

    def iter_samples(
        self, sample_num: int = 20, columns: Optional[list] = None
    ) -> Iterator[tuple[list[str], pd.DataFrame]]:
        """Iterate over samples in batches. Data and grouping from qpx."""
        df = self._get_long_form()
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available] if available else df
        ref_list = [
            self.samples[i : i + sample_num]
            for i in range(0, len(self.samples), sample_num)
        ]
        for refs in ref_list:
            batch = df[df["sample_accession"].isin(refs)]
            yield refs, batch

    def get_median_map(self) -> dict[str, float]:
        """Sample median intensity map (sample median / global median). Filtering in Python."""
        df = self._get_long_form()
        if self.filter_builder:
            df = _apply_filter_builder(df, self.filter_builder)
        med = df.groupby("sample_accession")["intensity"].median()
        global_med = float(med.median())
        if global_med <= 0:
            return {s: 1.0 for s in med.index}
        return (med / global_med).to_dict()

    def get_median_map_to_condition(self) -> dict[str, dict[str, float]]:
        """Per-condition median map. Uses 'condition' if present, else sample as condition."""
        df = self._get_long_form()
        if self.filter_builder:
            df = _apply_filter_builder(df, self.filter_builder)
        if "condition" not in df.columns:
            df = df.copy()
            df["condition"] = df["sample_accession"]
        grp = df.groupby(["condition", "sample_accession"])["intensity"].median().unstack(level=0)
        med_map = {}
        for cond in grp.columns:
            s = grp[cond].dropna()
            if s.empty:
                continue
            mean_val = s.mean()
            if mean_val <= 0:
                med_map[cond] = {k: 1.0 for k in s.index}
            else:
                med_map[cond] = (s / mean_val).to_dict()
        return med_map

    @property
    def experimental_inference(self) -> tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]:
        """Infer label type and samples from qpx data."""
        df = self._get_long_form()
        labels = df["label"].dropna().unique().tolist() if "label" in df.columns else []
        sample_names = self.samples
        label_enum, choice = QuantificationCategory.classify(labels)
        run_col = df.get("run_file_name", df.get("reference_file_name", None))
        if run_col is not None:
            tech_reps = run_col.nunique()
        else:
            tech_reps = 1
        return tech_reps, label_enum, sample_names, choice

    def get_sample_metadata(self) -> Optional[pd.DataFrame]:
        """Return sample table as DataFrame from qpx (one row per sample).
        Returns None if this dataset has no sample structure.
        Use config condition_column, batch_column, etc. to pick columns."""
        if self._sample is None:
            return None
        return self._sample.to_df()

    def get_run_to_fraction(self) -> Optional[dict[str, str]]:
        """Return run_file_name -> fraction from qpx run table when present.

        Used for ratio quantification so Fraction comes from QPX instead of SDRF.
        Returns None if no run structure or no fraction column; callers use "1" then.
        """
        if self._run is None:
            return None
        try:
            run_df = self._run.to_df()
        except Exception:
            return None
        if run_df is None or run_df.empty:
            return None
        run_col = "run_file_name" if "run_file_name" in run_df.columns else None
        frac_col = "fraction" if "fraction" in run_df.columns else None
        if not run_col or not frac_col:
            return None
        out = {}
        for _, row in run_df.iterrows():
            fname = str(row[run_col]).strip() if pd.notna(row.get(run_col)) else None
            if not fname:
                continue
            frac = row.get(frac_col)
            frac_str = str(frac).strip() if pd.notna(frac) and str(frac).strip() else "1"
            out[fname] = frac_str
            stem = fname.rsplit(".", 1)[0] if "." in fname else fname
            out[stem] = frac_str
        return out if out else None

    def enrich_with_sdrf(self, sdrf_path: str) -> None:
        """No-op when using QPX; sample/run metadata come from qpx."""
        logger.debug("enrich_with_sdrf is a no-op when using QPX directory; metadata from qpx")

    def get_feature_long_form(self) -> pd.DataFrame:
        """Long-form feature table with mokume column names for DirectLFQ path."""
        return self._get_long_form()

    def get_report_from_database(self, samples: list, columns: Optional[list] = None) -> pd.DataFrame:
        """Subset long-form to given samples (and optional columns)."""
        df = self._get_long_form()
        df = df[df["sample_accession"].isin(samples)]
        if columns:
            available = [c for c in columns if c in df.columns]
            if available:
                df = df[available]
        return df

    def get_low_frequency_peptides(self, percentage: float = 0.2) -> tuple:
        """Peptides that appear in less than percentage of samples. Filtering applied in Python."""
        df = self._get_long_form()
        df = _apply_filter_builder(df, self.filter_builder)
        grp = df.groupby(["sequence", "pg_accessions"])["sample_accession"].nunique().reset_index()
        grp = grp[grp["sample_accession"] < percentage * len(self.samples)]
        grp.dropna(subset=["pg_accessions"], inplace=True)
        # Parse protein accession: first element of list, then split by | and take [1] if present
        def _first_acc(x):
            if hasattr(x, "__len__") and len(x):
                v = x[0] if not isinstance(x, str) else x
            else:
                return ""
            return v.split("|")[1] if "|" in str(v) else v
        proteins = grp["pg_accessions"].apply(_first_acc)
        return tuple(zip(proteins.tolist(), grp["sequence"].tolist()))

    def get_irs_scaling_factors(
        self,
        irs_channel: str,
        irs_stat: str = "median",
        irs_scope: str = "global",
    ) -> dict[int, float]:
        """IRS scaling factors from long-form data. Filtering applied in Python."""
        df = self._get_long_form()
        col_label = "label" if "label" in df.columns else "sample_accession"
        if col_label not in df.columns:
            return {}
        df = df[df[col_label].astype(str) == str(irs_channel)]
        df = _apply_filter_builder(df, self.filter_builder)
        run_col = df.get("run_file_name", df.get("reference_file_name", None))
        if run_col is None or run_col.isna().all():
            return {}
        techrep = run_col.astype(str).str.split("_").str.get(-1)
        techrep = pd.to_numeric(techrep, errors="coerce").fillna(0).astype(int)
        df = df.copy()
        df["_techrep"] = techrep
        agg = df.groupby("_techrep")["intensity"].agg(irs_stat.lower())
        agg = agg[agg > 0]
        if agg.empty:
            return {}
        if irs_scope.lower() == "global":
            center = agg.median() if irs_stat.lower() == "median" else agg.mean()
            scale = center / agg
        else:
            center = agg.median() if irs_stat.lower() == "median" else agg.mean()
            scale = center / agg
        return dict(zip(agg.index.tolist(), scale.tolist()))


def _apply_filter_builder(df: pd.DataFrame, filter_builder) -> pd.DataFrame:
    """Apply filter_builder logic in Python (intensity, length, unique, contaminants)."""
    out = df[df["intensity"] > 0].copy()
    if getattr(filter_builder, "min_intensity", 0) > 0:
        out = out[out["intensity"] >= filter_builder.min_intensity]
    if getattr(filter_builder, "min_peptide_length", 0) > 0 and "sequence" in out.columns:
        out = out[out["sequence"].str.len() >= filter_builder.min_peptide_length]
    if getattr(filter_builder, "require_unique", False) and "unique" in out.columns:
        out = out[out["unique"] == 1]
    if getattr(filter_builder, "remove_contaminants", True) and "pg_accessions" in out.columns:
        patterns = getattr(filter_builder, "contaminant_patterns", ["CONTAMINANT", "ENTRAP", "DECOY"])
        for pat in patterns:
            out = out[~out["pg_accessions"].astype(str).str.contains(pat, regex=False, na=False)]
    return out


class _SingleFeatureDataset:
    """Minimal wrapper so QpxFeatureAdapter can consume a single qpx Feature (no run)."""
    def __init__(self, feature):
        self.feature = feature
        self.run = None
        self.sample = None


def create_feature_for_input(
    parquet: Optional[str],
    qpx_dir: Optional[str],
    filter_builder=None,
):
    """
    Create a Feature-like object for pipeline input. All reading is done by qpx.

    - If qpx_dir is set: open qpx.Dataset(qpx_dir) and return QpxFeatureAdapter.
    - If parquet is set: open qpx.read_feature(parquet) and return QpxFeatureAdapter over it.

    At least one of parquet or qpx_dir must be set.
    """
    if qpx_dir:
        ds = open_qpx(qpx_dir)
        return QpxFeatureAdapter(ds, filter_builder=filter_builder)
    if parquet:
        feat = open_qpx_feature_single(parquet)
        return QpxFeatureAdapter(_SingleFeatureDataset(feat), filter_builder=filter_builder)
    raise ValueError("Either parquet or qpx_dir must be set for pipeline input")
