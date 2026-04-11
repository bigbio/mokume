"""
QpxDataset — Hierarchical proteomics data container.

This module provides the core data container for the mokume pipeline.
QpxDataset mirrors the qpx format's hierarchical structure: PSMs,
features, peptides, and proteins are distinct data levels. Each
processing step reads from the appropriate level and writes results back.

Each data level can be backed by either a pandas DataFrame or a DuckDB
``LazyFrame``. Lazy frames defer computation until results are explicitly
requested, enabling mokume to handle datasets with millions of features
without loading them all into memory.

Serialization is native: the dataset saves/loads as a directory of
parquet files with a metadata JSON sidecar.

Example
-------
>>> dataset = QpxDataset.from_parquet("data.parquet")
>>> dataset.validate_level("features")
>>> wide = dataset.to_wide_matrix(level="proteins", value_col="Intensity")
>>> adata = dataset.to_anndata(level="proteins", value_col="Intensity")
>>> dataset.save("output_dir/")
>>>
>>> # Lazy loading for large datasets:
>>> dataset = QpxDataset.from_parquet_lazy("large_data.parquet")
>>> dataset.features  # LazyFrame, no materialization yet
>>> df = dataset.get_level("features")  # Materializes to DataFrame
"""

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mokume.core.constants import PROTEIN_NAME, SAMPLE_ID
from mokume.core.schema import validate_schema

# Type alias for data that can be either eager or lazy
DataLevel = Union[pd.DataFrame, "LazyFrame"]

# Sentinel to avoid circular import at module level
_LazyFrame = None


def _get_lazy_frame_class():
    """Lazy import of LazyFrame to avoid circular imports."""
    global _LazyFrame
    if _LazyFrame is None:
        from mokume.core.duckdb_backend import LazyFrame
        _LazyFrame = LazyFrame
    return _LazyFrame


def _is_lazy(obj) -> bool:
    """Check if an object is a LazyFrame without importing at module level."""
    if obj is None:
        return False
    LazyFrame = _get_lazy_frame_class()
    return isinstance(obj, LazyFrame)


def _ensure_df(obj: Optional[DataLevel]) -> Optional[pd.DataFrame]:
    """Materialize a LazyFrame to DataFrame if needed, or return as-is."""
    if obj is None:
        return None
    if _is_lazy(obj):
        return obj.df()
    return obj


def _extract_scalar(val):
    """Extract a scalar string from a value that may be a list/ndarray.

    QPX sample metadata stores SDRF characteristics as arrays (one element
    per unique value within a sample).  For AnnData obs we need plain
    strings, so we join multi-valued fields with ``"; "``.
    """
    if val is None or (hasattr(val, "__class__") and val.__class__.__name__ == "NAType"):
        return None
    if hasattr(val, "__len__") and not isinstance(val, (str, dict)):
        parts = [str(v) for v in val if v is not None and str(v) != ""]
        return "; ".join(parts) if parts else None
    return str(val) if pd.notna(val) else None


def _flatten_sample_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Prepare QPX sample metadata for AnnData obs.

    1. Flatten ndarray/list columns to scalar strings.
    2. Expand ``additional_properties`` dicts into individual columns.
    3. Drop columns that are entirely NA.
    """
    out = meta.copy()

    # Expand additional_properties dicts into individual columns
    if "additional_properties" in out.columns:
        extra_rows = []
        for idx, val in out["additional_properties"].items():
            row_extra = {}
            if isinstance(val, dict):
                for k, v in val.items():
                    # Normalise the key: "characteristics[biological replicate]" -> "biological_replicate"
                    clean_key = k
                    import re
                    m = re.match(r"characteristics\[(.+)\]", k)
                    if m:
                        clean_key = m.group(1).strip().replace(" ", "_")
                    row_extra[clean_key] = _extract_scalar(v)
            extra_rows.append(row_extra)
        if extra_rows and any(extra_rows):
            extra_df = pd.DataFrame(extra_rows, index=out.index)
            # Only add columns that don't already exist
            for col in extra_df.columns:
                if col not in out.columns:
                    out[col] = extra_df[col]
        out.drop(columns=["additional_properties"], inplace=True)

    # Flatten ndarray/list values to scalars
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(_extract_scalar)

    # Drop columns that are entirely NA
    out.dropna(axis=1, how="all", inplace=True)

    return out


@dataclass
class QpxDataset:
    """Hierarchical proteomics data container backed by qpx format.

    Each data level can hold either a ``pd.DataFrame`` (eager) or a
    ``LazyFrame`` (lazy DuckDB-backed). Use ``get_level()`` to always
    get a materialized DataFrame, or ``get_level_raw()`` to access the
    underlying object (which may be lazy).

    Attributes
    ----------
    psms : DataFrame or LazyFrame, optional
        PSM-level data (used by ratio quantification).
    features : DataFrame or LazyFrame, optional
        Feature-level data (charge states, fractions).
    peptides : DataFrame or LazyFrame, optional
        Peptide-level data (assembled peptidoforms, normalized).
    proteins : DataFrame or LazyFrame, optional
        Protein-level quantification results.
    sample_info : pd.DataFrame, optional
        Per-sample metadata (from SDRF or future metadata format).
    protein_info : pd.DataFrame, optional
        Per-protein metadata (accessions, gene names, MW, etc.).
    uns : dict
        Unstructured metadata (pipeline config, provenance, DE results).
    layers : dict
        Named alternative representations at any level.
        E.g., ``layers["normalized_features"]``, ``layers["batch_corrected"]``.
    """

    psms: Optional[DataLevel] = None
    features: Optional[DataLevel] = None
    peptides: Optional[DataLevel] = None
    proteins: Optional[DataLevel] = None

    sample_info: Optional[pd.DataFrame] = None
    protein_info: Optional[pd.DataFrame] = None

    uns: Dict[str, Any] = field(default_factory=dict)
    layers: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Data level access
    # ------------------------------------------------------------------

    _VALID_LEVELS = ("psms", "features", "peptides", "proteins")

    def get_level(self, level: str) -> Optional[pd.DataFrame]:
        """Get a data level by name, materializing if lazy.

        Parameters
        ----------
        level : str
            One of "psms", "features", "peptides", "proteins".

        Returns
        -------
        pd.DataFrame or None
            Always returns a DataFrame (never a LazyFrame).
        """
        if level not in self._VALID_LEVELS:
            raise ValueError(
                f"Unknown data level: '{level}'. "
                f"Must be one of: {', '.join(self._VALID_LEVELS)}"
            )
        raw = getattr(self, level, None)
        if raw is None:
            return None
        if _is_lazy(raw):
            # Materialize and cache
            df = raw.df()
            setattr(self, level, df)
            return df
        return raw

    def get_level_raw(self, level: str) -> Optional[DataLevel]:
        """Get a data level without materializing.

        Returns the underlying object which may be a LazyFrame
        (DuckDB-backed) or a DataFrame.

        Parameters
        ----------
        level : str
            One of "psms", "features", "peptides", "proteins".

        Returns
        -------
        pd.DataFrame, LazyFrame, or None
        """
        if level not in self._VALID_LEVELS:
            raise ValueError(
                f"Unknown data level: '{level}'. "
                f"Must be one of: {', '.join(self._VALID_LEVELS)}"
            )
        return getattr(self, level, None)

    def set_level(self, level: str, data: DataLevel) -> None:
        """Set a data level by name.

        Parameters
        ----------
        level : str
            One of "psms", "features", "peptides", "proteins".
        data : pd.DataFrame or LazyFrame
            The data to assign.
        """
        if level not in self._VALID_LEVELS:
            raise ValueError(
                f"Unknown data level: '{level}'. "
                f"Must be one of: {', '.join(self._VALID_LEVELS)}"
            )
        setattr(self, level, data)

    def is_lazy(self, level: str) -> bool:
        """Check if a data level is lazy (DuckDB-backed).

        Parameters
        ----------
        level : str
            Data level name.

        Returns
        -------
        bool
        """
        return _is_lazy(getattr(self, level, None))

    @property
    def populated_levels(self) -> List[str]:
        """Return names of data levels that have been populated."""
        levels = []
        for name in self._VALID_LEVELS:
            if getattr(self, name) is not None:
                levels.append(name)
        return levels

    @property
    def lazy_levels(self) -> List[str]:
        """Return names of data levels that are lazy (DuckDB-backed)."""
        return [name for name in self._VALID_LEVELS if self.is_lazy(name)]

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize(self, levels: Optional[List[str]] = None) -> "QpxDataset":
        """Force materialization of lazy levels to DataFrames.

        Parameters
        ----------
        levels : list[str], optional
            Specific levels to materialize. If None, materializes all.

        Returns
        -------
        QpxDataset
            Self (for chaining).
        """
        target_levels = levels or self._VALID_LEVELS
        for level_name in target_levels:
            raw = getattr(self, level_name, None)
            if raw is not None and _is_lazy(raw):
                setattr(self, level_name, raw.df())
        return self

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def validate_level(self, level: str) -> List[str]:
        """Validate that a data level has required columns.

        For lazy levels, validation checks column names without
        materializing the full dataset.

        Parameters
        ----------
        level : str
            Data level to validate.

        Returns
        -------
        list[str]
            List of error messages (empty if valid).
        """
        raw = self.get_level_raw(level)
        if raw is None:
            return [f"Data level '{level}' is not populated"]

        if _is_lazy(raw):
            # Validate column names without materializing
            df_stub = pd.DataFrame(columns=raw.columns)
            return validate_schema(df_stub, level)

        return validate_schema(raw, level)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def to_wide_matrix(
        self,
        level: str = "proteins",
        value_col: str = "Intensity",
        protein_col: str = PROTEIN_NAME,
        sample_col: str = SAMPLE_ID,
    ) -> pd.DataFrame:
        """Pivot a data level to a protein x sample matrix.

        Parameters
        ----------
        level : str
            Data level to pivot.
        value_col : str
            Column containing values for the matrix cells.
        protein_col : str
            Column to use as row index.
        sample_col : str
            Column to use as column headers.

        Returns
        -------
        pd.DataFrame
            Wide-format matrix with protein_col as index.
        """
        df = self.get_level(level)
        if df is None:
            raise ValueError(f"Data level '{level}' is not populated")

        return df.pivot_table(
            index=protein_col,
            columns=sample_col,
            values=value_col,
            aggfunc="first",
        )

    def to_long_format(
        self,
        level: str = "proteins",
    ) -> pd.DataFrame:
        """Return a data level in long format (copy).

        Parameters
        ----------
        level : str
            Data level to return.

        Returns
        -------
        pd.DataFrame
            Copy of the level's DataFrame.
        """
        df = self.get_level(level)
        if df is None:
            raise ValueError(f"Data level '{level}' is not populated")
        return df.copy()

    def peptide_protein_map(self) -> pd.DataFrame:
        """Return a peptide-to-protein mapping table.

        Uses the first available level that has both ProteinName and
        a peptide column.

        Returns
        -------
        pd.DataFrame
            DataFrame with ProteinName and peptide columns.
        """
        from mokume.core.constants import PEPTIDE_CANONICAL, PEPTIDE_SEQUENCE

        for level_name in ("peptides", "features", "psms"):
            raw = self.get_level_raw(level_name)
            if raw is None:
                continue

            # Get column names (works for both DataFrame and LazyFrame)
            cols = raw.columns
            pep_col = None
            if PEPTIDE_CANONICAL in cols:
                pep_col = PEPTIDE_CANONICAL
            elif PEPTIDE_SEQUENCE in cols:
                pep_col = PEPTIDE_SEQUENCE

            if pep_col and PROTEIN_NAME in cols:
                df = self.get_level(level_name)  # materialize if needed
                return df[[PROTEIN_NAME, pep_col]].drop_duplicates()

        raise ValueError("No data level with protein and peptide columns found")

    def sample_metadata(self) -> pd.DataFrame:
        """Return sample metadata.

        Returns
        -------
        pd.DataFrame
            Sample metadata DataFrame.

        Raises
        ------
        ValueError
            If sample_info is not populated.
        """
        if self.sample_info is None:
            raise ValueError("sample_info is not populated")
        return self.sample_info.copy()

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset_samples(self, sample_ids: List[str]) -> "QpxDataset":
        """Subset all populated levels to given samples.

        Creates a new QpxDataset where every data level and sample_info
        are filtered to only include the specified samples. Lazy levels
        are materialized during subsetting.

        Parameters
        ----------
        sample_ids : list[str]
            Sample identifiers to keep.

        Returns
        -------
        QpxDataset
            New dataset with subsetted data.
        """
        sample_set = set(sample_ids)
        new = QpxDataset(
            uns=deepcopy(self.uns),
            layers={},
            protein_info=self.protein_info.copy() if self.protein_info is not None else None,
        )

        for level_name in self._VALID_LEVELS:
            df = self.get_level(level_name)  # materializes if lazy
            if df is not None and SAMPLE_ID in df.columns:
                new.set_level(level_name, df[df[SAMPLE_ID].isin(sample_set)].copy())
            elif df is not None:
                # Wide format — filter columns
                sample_cols = [c for c in df.columns if c in sample_set]
                non_sample_cols = [c for c in df.columns if c not in sample_set and c not in sample_ids]
                if sample_cols:
                    new.set_level(level_name, df[non_sample_cols + sample_cols].copy())

        if self.sample_info is not None:
            # Try common sample ID column names
            for col in [SAMPLE_ID, "sample_accession", "source name"]:
                if col in self.sample_info.columns:
                    new.sample_info = self.sample_info[
                        self.sample_info[col].isin(sample_set)
                    ].copy()
                    break
            else:
                new.sample_info = self.sample_info.copy()

        for key, layer_df in self.layers.items():
            if SAMPLE_ID in layer_df.columns:
                new.layers[key] = layer_df[layer_df[SAMPLE_ID].isin(sample_set)].copy()
            else:
                new.layers[key] = layer_df.copy()

        return new

    def subset_proteins(self, protein_ids: List[str]) -> "QpxDataset":
        """Subset all populated levels to given proteins.

        Parameters
        ----------
        protein_ids : list[str]
            Protein identifiers to keep.

        Returns
        -------
        QpxDataset
            New dataset with subsetted data.
        """
        protein_set = set(protein_ids)
        new = QpxDataset(
            uns=deepcopy(self.uns),
            layers={},
            sample_info=self.sample_info.copy() if self.sample_info is not None else None,
        )

        for level_name in self._VALID_LEVELS:
            df = self.get_level(level_name)  # materializes if lazy
            if df is not None and PROTEIN_NAME in df.columns:
                new.set_level(level_name, df[df[PROTEIN_NAME].isin(protein_set)].copy())

        if self.protein_info is not None and PROTEIN_NAME in self.protein_info.columns:
            new.protein_info = self.protein_info[
                self.protein_info[PROTEIN_NAME].isin(protein_set)
            ].copy()

        for key, layer_df in self.layers.items():
            if PROTEIN_NAME in layer_df.columns:
                new.layers[key] = layer_df[layer_df[PROTEIN_NAME].isin(protein_set)].copy()
            else:
                new.layers[key] = layer_df.copy()

        return new

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save the dataset to a directory of parquet files.

        Lazy levels are materialized during save.

        Directory structure::

            directory/
            ├── psms.parquet        (if populated)
            ├── features.parquet    (if populated)
            ├── peptides.parquet    (if populated)
            ├── proteins.parquet    (if populated)
            ├── sample_info.parquet (if populated)
            ├── protein_info.parquet(if populated)
            ├── layers/
            │   ├── <name>.parquet  (for each layer)
            └── uns.json

        Parameters
        ----------
        directory : str
            Output directory path. Created if it doesn't exist.
        """
        os.makedirs(directory, exist_ok=True)

        # Save data levels (materializing lazy frames)
        for level_name in self._VALID_LEVELS:
            raw = self.get_level_raw(level_name)
            if raw is not None:
                df = _ensure_df(raw)
                df.to_parquet(os.path.join(directory, f"{level_name}.parquet"), index=False)

        # Save metadata
        if self.sample_info is not None:
            self.sample_info.to_parquet(
                os.path.join(directory, "sample_info.parquet"), index=False
            )
        if self.protein_info is not None:
            self.protein_info.to_parquet(
                os.path.join(directory, "protein_info.parquet"), index=False
            )

        # Save layers
        if self.layers:
            layers_dir = os.path.join(directory, "layers")
            os.makedirs(layers_dir, exist_ok=True)
            for name, layer_df in self.layers.items():
                layer_df.to_parquet(
                    os.path.join(layers_dir, f"{name}.parquet"), index=False
                )

        # Save uns as JSON (convert non-serializable values to strings)
        uns_path = os.path.join(directory, "uns.json")
        with open(uns_path, "w") as f:
            json.dump(self.uns, f, indent=2, default=str)

    @classmethod
    def load(cls, directory: str, lazy: bool = False) -> "QpxDataset":
        """Load a dataset from a directory of parquet files.

        Parameters
        ----------
        directory : str
            Directory containing the saved dataset.
        lazy : bool
            If True, load data levels as LazyFrames (DuckDB-backed)
            instead of materializing to DataFrames.

        Returns
        -------
        QpxDataset
        """
        dataset = cls()

        if lazy:
            LazyFrame = _get_lazy_frame_class()

        # Load data levels
        for level_name in cls._VALID_LEVELS:
            path = os.path.join(directory, f"{level_name}.parquet")
            if os.path.exists(path):
                if lazy:
                    dataset.set_level(level_name, LazyFrame.from_parquet(path))
                else:
                    dataset.set_level(level_name, pd.read_parquet(path))

        # Load metadata (always eager — these are small)
        sample_info_path = os.path.join(directory, "sample_info.parquet")
        if os.path.exists(sample_info_path):
            dataset.sample_info = pd.read_parquet(sample_info_path)

        protein_info_path = os.path.join(directory, "protein_info.parquet")
        if os.path.exists(protein_info_path):
            dataset.protein_info = pd.read_parquet(protein_info_path)

        # Load layers (always eager)
        layers_dir = os.path.join(directory, "layers")
        if os.path.isdir(layers_dir):
            for filename in os.listdir(layers_dir):
                if filename.endswith(".parquet"):
                    name = filename.rsplit(".parquet", 1)[0]
                    dataset.layers[name] = pd.read_parquet(
                        os.path.join(layers_dir, filename)
                    )

        # Load uns
        uns_path = os.path.join(directory, "uns.json")
        if os.path.exists(uns_path):
            with open(uns_path, "r") as f:
                dataset.uns = json.load(f)

        return dataset

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_anndata(
        self,
        level: str = "proteins",
        value_col: str = "Intensity",
        protein_col: str = PROTEIN_NAME,
        sample_col: str = SAMPLE_ID,
        layer_names: Optional[List[str]] = None,
    ):
        """Export a data level as an AnnData object.

        Parameters
        ----------
        level : str
            Data level to export.
        value_col : str
            Column for the main X matrix values.
        protein_col : str
            Column for variable (protein) identifiers.
        sample_col : str
            Column for observation (sample) identifiers.
        layer_names : list[str], optional
            Names of layers (from self.layers) to include in AnnData.layers.

        Returns
        -------
        anndata.AnnData
            AnnData object with X matrix, obs, var, and optional layers.
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError(
                "anndata is required for to_anndata(). "
                "Install with: pip install mokume[anndata]"
            )

        df = self.get_level(level)
        if df is None:
            raise ValueError(f"Data level '{level}' is not populated")

        # Detect wide vs long format.
        # Long: has sample_col and value_col as columns (classic long format).
        # Wide: first column is protein IDs, remaining columns are sample names.
        is_long = sample_col in df.columns and value_col in df.columns

        if is_long:
            wide = df.pivot_table(
                index=sample_col,
                columns=protein_col,
                values=value_col,
                aggfunc="first",
            )
        else:
            # Wide format: identify the protein ID column (first column or
            # matching protein_col / common names).
            pid_col = None
            for candidate in [protein_col, "protein", "ProteinName", df.columns[0]]:
                if candidate in df.columns:
                    pid_col = candidate
                    break
            wide = df.set_index(pid_col).T
            wide.index.name = sample_col

        # Build obs (sample metadata)
        obs = pd.DataFrame(index=wide.index)
        obs.index.name = sample_col
        if self.sample_info is not None:
            # Resolve the sample ID column in sample_info; may be
            # ``sample_col`` (SampleID) or ``sample_accession`` (QPX).
            sid_col = None
            for candidate in [sample_col, "sample_accession"]:
                if candidate in self.sample_info.columns:
                    sid_col = candidate
                    break
            if sid_col is not None:
                sample_meta = self.sample_info.set_index(sid_col).copy()
                # Flatten ndarray/list values to scalar strings (QPX stores
                # SDRF characteristics as arrays).
                sample_meta = _flatten_sample_meta(sample_meta)
                obs = obs.join(sample_meta, how="left")

        # Build var (protein metadata)
        var = pd.DataFrame(index=wide.columns)
        var.index.name = protein_col
        if self.protein_info is not None and protein_col in self.protein_info.columns:
            protein_meta = self.protein_info.set_index(protein_col)
            var = var.join(protein_meta, how="left")

        # Build AnnData
        adata = ad.AnnData(
            X=wide.values,
            obs=obs,
            var=var,
        )

        # Add layers if requested
        if layer_names:
            for layer_name in layer_names:
                if layer_name in self.layers:
                    layer_df = self.layers[layer_name]
                    if sample_col in layer_df.columns and protein_col in layer_df.columns:
                        layer_wide = layer_df.pivot_table(
                            index=sample_col,
                            columns=protein_col,
                            values=value_col,
                            aggfunc="first",
                        )
                        # Align to same shape as X
                        layer_wide = layer_wide.reindex(
                            index=wide.index, columns=wide.columns
                        )
                        adata.layers[layer_name] = layer_wide.values

        # Store provenance in uns (serialize complex structures as JSON
        # strings because h5py cannot write deeply-nested dicts/lists).
        if self.uns:
            import json as _json
            safe = {}
            for k, v in self.uns.items():
                if isinstance(v, (str, int, float, bool)):
                    safe[k] = v
                elif isinstance(v, (list, dict)):
                    safe[k] = _json.dumps(v, default=str)
                # skip other types
            adata.uns["mokume"] = safe

        return adata

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def record_step(
        self,
        name: str,
        method: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        rows_in: Optional[int] = None,
        rows_out: Optional[int] = None,
        **extra,
    ) -> None:
        """Record a pipeline step in provenance metadata.

        Parameters
        ----------
        name : str
            Step name (e.g., "loading", "normalization", "quantification").
        method : str, optional
            Method used (e.g., "median", "directlfq").
        duration_seconds : float, optional
            Wall-clock time for the step.
        rows_in : int, optional
            Number of input rows.
        rows_out : int, optional
            Number of output rows.
        **extra
            Additional metadata for the step.
        """
        if "provenance" not in self.uns:
            self.uns["provenance"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "steps": [],
            }

        step = {"name": name}
        if method is not None:
            step["method"] = method
        if duration_seconds is not None:
            step["duration_seconds"] = round(duration_seconds, 3)
        if rows_in is not None:
            step["rows_in"] = rows_in
        if rows_out is not None:
            step["rows_out"] = rows_out
        step.update(extra)

        self.uns["provenance"]["steps"].append(step)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        sdrf_path: Optional[str] = None,
        level: str = "features",
    ) -> "QpxDataset":
        """Create a QpxDataset from a qpx parquet file (eager loading).

        Reads the entire parquet file into a DataFrame. For large
        datasets, use ``from_parquet_lazy()`` instead.

        Parameters
        ----------
        parquet_path : str
            Path to the qpx parquet file.
        sdrf_path : str, optional
            Path to SDRF TSV for sample metadata.
        level : str
            Data level to assign the loaded data to.

        Returns
        -------
        QpxDataset
        """
        dataset = cls()

        # Load parquet
        df = pd.read_parquet(parquet_path)
        dataset.set_level(level, df)

        # Load SDRF if provided
        if sdrf_path is not None:
            from mokume.core.constants import load_sdrf
            dataset.sample_info = load_sdrf(sdrf_path)

        return dataset

    @classmethod
    def from_parquet_lazy(
        cls,
        parquet_path: str,
        sdrf_path: Optional[str] = None,
        level: str = "features",
    ) -> "QpxDataset":
        """Create a QpxDataset with DuckDB lazy backing from a parquet file.

        The parquet file is registered as a DuckDB scan but not loaded
        into memory. Data is only materialized when explicitly requested
        via ``get_level()`` or ``materialize()``.

        Parameters
        ----------
        parquet_path : str
            Path to the qpx parquet file.
        sdrf_path : str, optional
            Path to SDRF TSV for sample metadata.
        level : str
            Data level to assign the lazy scan to.

        Returns
        -------
        QpxDataset
        """
        LazyFrame = _get_lazy_frame_class()
        dataset = cls()

        # Create lazy scan
        lazy = LazyFrame.from_parquet(parquet_path)
        dataset.set_level(level, lazy)

        # Load SDRF if provided (always eager — small file)
        if sdrf_path is not None:
            from mokume.core.constants import load_sdrf
            dataset.sample_info = load_sdrf(sdrf_path)

        return dataset

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = ["QpxDataset("]
        for level_name in self._VALID_LEVELS:
            raw = self.get_level_raw(level_name)
            if raw is None:
                continue
            if _is_lazy(raw):
                ncols = len(raw.columns)
                parts.append(f"  {level_name}: LazyFrame ({ncols} cols, not materialized)")
            else:
                parts.append(f"  {level_name}: {raw.shape[0]} rows x {raw.shape[1]} cols")
        if self.sample_info is not None:
            parts.append(f"  sample_info: {len(self.sample_info)} samples")
        if self.protein_info is not None:
            parts.append(f"  protein_info: {len(self.protein_info)} proteins")
        if self.layers:
            parts.append(f"  layers: {list(self.layers.keys())}")
        if self.uns:
            parts.append(f"  uns: {list(self.uns.keys())}")
        parts.append(")")
        return "\n".join(parts)
