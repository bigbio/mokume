"""
TissueMap pipeline orchestrator — per-dataset processing.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from mokume.tissuemap.batch_correction import batch_correct
from mokume.tissuemap.config import TissueMapConfig
from mokume.tissuemap.embedding import embed
from mokume.tissuemap.loader import TissueLoadOptions, load_dataset
from mokume.tissuemap.plotting.atlas import plot_slide_atlas_dendrogram
from mokume.tissuemap.plotting.embedding import plot_pca_scree
from mokume.tissuemap.plotting.markers import (
    compute_markers,
    plot_dotplot,
    plot_marker_heatmap,
    plot_marker_tsne,
    save_markers_csv,
)
from mokume.tissuemap.plotting.specificity import plot_ts_distribution
from mokume.tissuemap.preprocessing import harmonize_tissues, log2_median_normalize
from mokume.tissuemap.protein_selection import filter_proteins
from mokume.tissuemap.enrichment import build_ts_anndata
from mokume.tissuemap.tissue_specificity import compute_ts_scores

logger = logging.getLogger(__name__)

# Keys in adata.uns that cannot be serialized to h5ad
_UNS_KEYS_TO_REMOVE = {"proteomic_groups", "tissue_colors"}

_RECOVERABLE_ERRORS = (
    ValueError,
    TypeError,
    RuntimeError,
    OSError,
    KeyError,
    IndexError,
    ArithmeticError,
    ImportError,
    AttributeError,
)


def _protein_metadata(matrix):
    """Keep annotation anchors separate from analysis group identities."""
    return pd.DataFrame(
        {"anchor_protein": matrix.index.map(matrix.attrs["anchor_protein"])},
        index=matrix.index,
    )


def _clean_uns_for_h5ad(adata: ad.AnnData) -> None:
    """Remove non-serializable keys from adata.uns before writing h5ad."""
    for key in _UNS_KEYS_TO_REMOVE:
        adata.uns.pop(key, None)


class TissueMapPipeline:
    """Per-dataset tissue proteome analysis pipeline.

    Workflow: load → normalize → harmonize → filter → batch-correct →
    AdaTiSS TS scoring → PCA/t-SNE embedding (optional) → plots → save.
    """

    def __init__(self, config: TissueMapConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Execute the pipeline for each dataset found in scan_dir."""
        scan_dir = self.config.input.scan_dir
        out_dir = self.config.output.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Discover datasets: scan_dir itself may be a PXD directory,
        # or contain PXD* subdirectories.
        ds_dirs = self._discover_datasets(scan_dir)
        if not ds_dirs:
            logger.error("No datasets found in %s", scan_dir)
            return

        n_ds = len(ds_dirs)
        workers = min(self.config.n_jobs, n_ds)

        # Dynamically scale per-dataset embedding threads to avoid
        # oversubscription (workers × embed_jobs ≤ n_jobs).
        self._embed_n_jobs = max(1, self.config.n_jobs // workers)

        tmt_set = frozenset(self.config.input.tmt_datasets)

        if workers > 1:
            logger.info("Processing %d datasets with %d workers", n_ds, workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._run_one, ds_id, ds_dir, out_dir, tmt_set): ds_id
                    for ds_id, ds_dir in ds_dirs.items()
                }
                for future in as_completed(futures):
                    ds_id = futures[future]
                    try:
                        future.result()
                    except _RECOVERABLE_ERRORS as exc:
                        logger.error("Dataset %s failed: %s", ds_id, exc, exc_info=True)
        else:
            for ds_id, ds_dir in ds_dirs.items():
                try:
                    self._run_one(ds_id, ds_dir, out_dir, tmt_set)
                except _RECOVERABLE_ERRORS as exc:
                    logger.error("Dataset %s failed: %s", ds_id, exc, exc_info=True)

    def _discover_datasets(self, scan_dir: Path) -> dict[str, Path]:
        """Return {ds_id: ds_dir} for datasets to process."""
        qpx_dir = scan_dir / "qpx_output"
        if qpx_dir.is_dir():
            # scan_dir itself is a dataset directory
            ds_id = scan_dir.name
            return {ds_id: scan_dir}

        # Look for PXD* subdirectories
        result: dict[str, Path] = {}
        for child in sorted(scan_dir.iterdir()):
            if child.is_dir() and (child / "qpx_output").is_dir():
                result[child.name] = child
        return result

    def _load_and_normalize(
        self,
        ds_id: str,
        ds_dir: Path,
        tmt_set: frozenset[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Load, normalize, harmonize, and filter a dataset.

        Returns (mat_filt, meta) or None if dataset should be skipped.
        """
        is_tmt = ds_id in tmt_set
        logger.info("Loading dataset %s", ds_id)
        mat, meta = load_dataset(
            ds_dir,
            ds_id,
            is_tmt=is_tmt if self.config.input.tmt_datasets else None,
            feature_prefix=self.config.input.feature_prefix,
            options=TissueLoadOptions(
                min_samples=self.config.input.min_tissue_samples,
                warning_threshold=self.config.input.low_sample_warning_threshold,
                n_jobs=self._embed_n_jobs,
            ),
        )

        logger.info("Log2 + median normalization")
        mat_norm = log2_median_normalize(mat)

        logger.info("Tissue harmonization")
        meta = harmonize_tissues(meta, min_samples=self.config.input.min_tissue_samples)
        common = mat_norm.columns.intersection(meta.index)
        mat_norm, meta = mat_norm[common], meta.loc[common]
        logger.info(
            "After harmonization: %d proteins x %d samples, %d tissues",
            mat_norm.shape[0],
            mat_norm.shape[1],
            meta["tissue"].nunique(),
        )

        if mat_norm.shape[1] < 5:
            logger.warning(
                "Too few samples (%d), skipping dataset %s", mat_norm.shape[1], ds_id
            )
            return None

        logger.info("Protein selection")
        mat_filt = filter_proteins(mat_norm, self.config.filtering)
        if mat_filt.shape[0] < 10:
            logger.warning(
                "Too few proteins (%d), skipping dataset %s", mat_filt.shape[0], ds_id
            )
            return None

        return mat_filt, meta

    def _save_outputs(
        self,
        ds_id: str,
        ds_out: Path,
        adata: ad.AnnData,
        ts_adata: ad.AnnData,
        ts_df: pd.DataFrame,
    ) -> None:
        """Write h5ad and CSV outputs to disk."""
        ds_out.mkdir(parents=True, exist_ok=True)
        _clean_uns_for_h5ad(adata)

        corrected_path = ds_out / f"{ds_id}.corrected.h5ad"
        adata.write_h5ad(corrected_path)
        logger.info("Saved: %s", corrected_path)

        ts_path = ds_out / f"{ds_id}.ts_scores.h5ad"
        ts_adata.write_h5ad(ts_path)
        logger.info("Saved: %s", ts_path)

        ts_csv_path = ds_out / "protein_ts_scores.csv"
        ts_df.to_csv(ts_csv_path)
        logger.info("Saved: %s", ts_csv_path)

    def _run_one(
        self,
        ds_id: str,
        ds_dir: Path,
        out_dir: Path,
        tmt_set: frozenset[str],
    ) -> None:
        """Process a single dataset."""
        result = self._load_and_normalize(ds_id, ds_dir, tmt_set)
        if result is None:
            return
        mat_filt, meta = result

        logger.info("Batch correction")
        log2_arr = mat_filt.T.values.astype(np.float32)
        sample_ids = np.array(mat_filt.columns)
        corrected = batch_correct(
            log2_arr, sample_ids, meta.loc[sample_ids, "batch"].values
        )

        adata = ad.AnnData(
            X=corrected.astype(np.float32),
            obs=meta.loc[sample_ids].copy(),
            var=_protein_metadata(mat_filt),
        )
        adata.var.index.name = "protein"
        adata.layers["log2_corrected"] = corrected.astype(np.float32)

        logger.info("Computing tissue specificity scores")
        ts_df = compute_ts_scores(
            corrected,
            adata.obs["tissue"].values,
            np.array(adata.var.index),
            self.config.tissue_specificity,
        )
        unique_tissues = sorted(adata.obs["tissue"].unique())
        ts_df["anchor_protein"] = adata.var["anchor_protein"]
        ts_adata = build_ts_anndata(ts_df, unique_tissues)

        for col in ["enrichment_category", "max_tissue", "max_ts", "mu", "sigma"]:
            if col in ts_df.columns:
                adata.var[col] = ts_df[col].values

        logger.info("PCA + t-SNE embedding")
        adata = embed(adata, self.config.embedding, n_jobs=self._embed_n_jobs)

        logger.info("Generating plots")
        ds_plot_dir = out_dir / ds_id / "plots"
        ds_plot_dir.mkdir(parents=True, exist_ok=True)
        self._generate_plots(adata, ts_df, unique_tissues, ds_plot_dir)

        logger.info("Saving outputs")
        self._save_outputs(ds_id, out_dir / ds_id, adata, ts_adata, ts_df)
        logger.info("Dataset %s completed", ds_id)

    def _plot_embedding_group(
        self,
        adata: ad.AnnData,
        plot_dir: Path,
    ) -> None:
        """Generate embedding-dependent plots (PCA scree, atlas)."""
        cfg = self.config.plotting
        try:
            plot_pca_scree(adata, plot_dir, dpi=cfg.dpi, save_pdf=cfg.save_pdf)
        except _RECOVERABLE_ERRORS as exc:
            logger.error("PCA plotting failed: %s", exc, exc_info=True)
        try:
            plot_slide_atlas_dendrogram(
                adata,
                plot_dir,
                dpi=cfg.dpi,
                save_pdf=cfg.save_pdf,
            )
        except _RECOVERABLE_ERRORS as exc:
            logger.error("Atlas plotting failed: %s", exc, exc_info=True)

    def _plot_marker_group(
        self,
        adata: ad.AnnData,
        unique_tissues: list[str],
        plot_dir: Path,
        has_embedding: bool,
    ) -> None:
        """Generate marker-related plots (heatmap, t-SNE, dotplot, CSV)."""
        cfg = self.config.plotting
        adata = compute_markers(adata)
        plot_marker_heatmap(
            adata,
            unique_tissues,
            plot_dir,
            n_top=5,
            dpi=cfg.dpi,
            save_pdf=cfg.save_pdf,
        )
        if has_embedding:
            plot_marker_tsne(
                adata,
                unique_tissues,
                plot_dir,
                n_showcase=8,
                dpi=cfg.dpi,
                save_pdf=cfg.save_pdf,
            )
        plot_dotplot(adata, unique_tissues, plot_dir, n_top=3, dpi=cfg.dpi)
        save_markers_csv(adata, unique_tissues, plot_dir, n_top=cfg.n_marker_top)

    def _plot_ts_group(
        self,
        ts_df: pd.DataFrame,
        unique_tissues: list[str],
        plot_dir: Path,
    ) -> None:
        """Generate tissue specificity distribution plot."""
        cfg = self.config.plotting
        plot_ts_distribution(
            ts_df,
            unique_tissues,
            plot_dir,
            ts_enriched=ts_df.attrs.get("ts_enriched_threshold", 2.5),
            ts_specific=ts_df.attrs.get("ts_specific_threshold", 4.0),
            ts_housekeeping=ts_df.attrs.get("ts_housekeeping_threshold"),
            dpi=cfg.dpi,
            save_pdf=cfg.save_pdf,
        )

    def _generate_plots(
        self,
        adata: ad.AnnData,
        ts_df: pd.DataFrame,
        unique_tissues: list[str],
        plot_dir: Path,
    ) -> None:
        """Generate all visualization outputs."""
        has_embedding = "X_tsne" in adata.obsm

        if has_embedding:
            self._plot_embedding_group(adata, plot_dir)
        else:
            logger.info("Embedding disabled, skipping PCA/atlas/t-SNE plots")

        try:
            self._plot_marker_group(adata, unique_tissues, plot_dir, has_embedding)
        except _RECOVERABLE_ERRORS as exc:
            logger.error("Marker plotting failed: %s", exc, exc_info=True)

        try:
            self._plot_ts_group(ts_df, unique_tissues, plot_dir)
        except _RECOVERABLE_ERRORS as exc:
            logger.error("TS plotting failed: %s", exc, exc_info=True)
