"""
Quality control report for proteomics quantification workflows.

Generates interactive HTML reports with sample-level and DE-level QC metrics:
- PCA (colored by condition, shaped by plex for up to six plexes)
- Sample correlation heatmap
- Per-condition CV distributions
- Intensity/ratio distribution boxplots
- Missing value patterns
- P-value histogram and DE quality metrics
- Silhouette score (condition clustering quality)
- Variance decomposition (condition vs plex)
"""

import html as html_module
import json
from typing import Optional

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from mokume.plotting.style import (
    CONDITION_COLORS,
    category_colors,
    category_order,
    MISSING_COLOR,
    PLOTLY_TEMPLATE,
    REPORT_CSS,
    SEQUENTIAL_COLORS,
    SIGNIFICANCE_COLORS,
    SINGLE_SERIES_COLOR,
    TEXT_COLOR,
)
from mokume.core.logger import get_logger
from mokume.plotting._common import (
    group_ellipses,
    log2_matrix as prepare_log2,
    read_frame,
)
from mokume.reports._plots import EMBEDDING_LABEL_JS, plot_cards, sample_axis
from mokume.reports._qc_diagnostics import diagnostic_plots

logger = get_logger("mokume.reports.qc")


# ---------------------------------------------------------------------------
# QC metric computation (pure numpy/pandas, no plotting)
# ---------------------------------------------------------------------------


def compute_qc_metrics(
    protein_df: pd.DataFrame,
    sample_to_condition: dict[str, str],
    sample_to_plex: Optional[dict[str, str]] = None,
    de_results: Optional[pd.DataFrame] = None,
    is_log2: bool = False,
) -> dict:
    """
    Compute comprehensive QC metrics from a protein matrix.

    Parameters
    ----------
    protein_df : pd.DataFrame
        Wide-format protein matrix (ProteinName | sample1 | sample2 | ...).
    sample_to_condition : dict
        Sample -> condition mapping.
    sample_to_plex : dict, optional
        Sample -> plex mapping. If None, all samples assigned to "plex1".
    de_results : pd.DataFrame, optional
        DE results with columns: ProteinName, log2FC, pvalue, adj_pvalue, significance.
    is_log2 : bool
        If True, data is already in log2 space (e.g., ratio quantification).

    Returns
    -------
    dict
        Dictionary of QC metrics (all serializable for JSON).
    """
    protein_col = protein_df.columns[0]
    sample_cols = [c for c in protein_df.columns if c != protein_col]
    exp_samples = [s for s in sample_cols if s in sample_to_condition]

    if sample_to_plex is None:
        sample_to_plex = {s: "plex1" for s in exp_samples}

    # Prepare log2 matrix
    intensity = protein_df.set_index(protein_col)[exp_samples].copy()
    if not is_log2:
        intensity = intensity.replace(0, np.nan)
        log2_matrix = np.log2(intensity)
    else:
        log2_matrix = intensity.replace(0, np.nan)

    metrics = {}

    # 1. PCA
    metrics["pca"] = _compute_pca(
        log2_matrix, exp_samples, sample_to_condition, sample_to_plex
    )

    # 3. Sample correlation matrix
    metrics["correlation"] = _compute_sample_correlation(log2_matrix, exp_samples)

    # 4. Per-condition CV
    metrics["condition_cv"] = _compute_condition_cv(
        intensity if not is_log2 else 2**log2_matrix,
        exp_samples,
        sample_to_condition,
    )

    # 5. Intra-condition correlation
    metrics["intra_condition_corr"] = _compute_intra_condition_correlation(
        log2_matrix,
        exp_samples,
        sample_to_condition,
    )

    # 6. Missing value analysis
    metrics["missing"] = _compute_missing_values(
        log2_matrix,
        exp_samples,
        sample_to_condition,
    )

    # 7. Per-sample distribution stats
    metrics["distributions"] = _compute_distribution_stats(
        log2_matrix,
        exp_samples,
        sample_to_condition,
    )

    # 8. Silhouette score (condition clustering)
    metrics["silhouette"] = _compute_silhouette(
        log2_matrix,
        exp_samples,
        sample_to_condition,
    )

    # 9. Variance decomposition (condition vs plex)
    metrics["pvca"] = _compute_variance_decomposition(
        log2_matrix,
        exp_samples,
        sample_to_condition,
        sample_to_plex,
    )

    # 10. DE quality metrics
    if de_results is not None:
        metrics["de_quality"] = _compute_de_quality(de_results)

    # 11. Summary
    metrics["summary"] = {
        "n_proteins": len(protein_df),
        "n_samples": len(exp_samples),
        "n_conditions": len(set(sample_to_condition[s] for s in exp_samples)),
        "n_plexes": len(set(sample_to_plex.get(s, "plex1") for s in exp_samples)),
        "is_log2": is_log2,
    }

    return metrics


def _compute_pca(log2_matrix, samples, sample_to_condition, sample_to_plex):
    """Compute PCA coordinates."""
    data = log2_matrix[samples].T.dropna(axis=1)
    data = data.fillna(data.median())

    n_components = min(3, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(data)

    result = {
        "samples": list(data.index),
        "conditions": [sample_to_condition.get(s, "Unknown") for s in data.index],
        "plexes": [sample_to_plex.get(s, "plex1") for s in data.index],
        "pc1": coords[:, 0].tolist(),
        "pc2": coords[:, 1].tolist() if n_components >= 2 else [0] * len(data),
        "var_explained": (pca.explained_variance_ratio_ * 100).tolist(),
    }
    if n_components >= 3:
        result["pc3"] = coords[:, 2].tolist()
    groups = pd.DataFrame(
        {"pc1": result["pc1"], "pc2": result["pc2"], "condition": result["conditions"]}
    )
    result["ellipses"] = (
        {
            condition: {"x": outline[:, 0].tolist(), "y": outline[:, 1].tolist()}
            for condition, outline in group_ellipses(
                groups, ("pc1", "pc2"), "condition"
            ).items()
        }
        if n_components >= 2
        else {}
    )

    return result


def _compute_sample_correlation(log2_matrix, samples):
    """Compute pairwise Pearson correlation between samples."""
    corr = log2_matrix[samples].corr(method="pearson")
    return {
        "samples": list(corr.columns),
        "matrix": corr.values.tolist(),
    }


def _compute_condition_cv(intensity_linear, samples, sample_to_condition):
    """Compute per-protein CV within each condition, return distributions."""
    result = {}
    for cond in sorted(set(sample_to_condition[s] for s in samples)):
        cond_samples = [s for s in samples if sample_to_condition[s] == cond]
        cond_data = intensity_linear[cond_samples]
        cv = cond_data.std(axis=1) / cond_data.mean(axis=1)
        cv = cv.dropna()
        result[cond] = {
            "median_cv": float(cv.median()) if len(cv) > 0 else float("nan"),
            "mean_cv": float(cv.mean()) if len(cv) > 0 else float("nan"),
            "cv_values": cv.tolist(),
            "q25": float(cv.quantile(0.25)) if len(cv) > 0 else float("nan"),
            "q75": float(cv.quantile(0.75)) if len(cv) > 0 else float("nan"),
            "q99": float(cv.quantile(0.99)),
        }
    return result


def _compute_intra_condition_correlation(log2_matrix, samples, sample_to_condition):
    """Median pairwise Pearson correlation within each condition."""
    result = {}
    for cond in sorted(set(sample_to_condition[s] for s in samples)):
        cond_samples = [s for s in samples if sample_to_condition[s] == cond]
        if len(cond_samples) < 2:
            result[cond] = float("nan")
            continue
        corr = log2_matrix[cond_samples].corr(method="pearson")
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        pairwise = corr.values[mask]
        result[cond] = float(np.nanmedian(pairwise))
    return result


def _compute_missing_values(log2_matrix, samples, sample_to_condition):
    """Missing value rates per sample and per condition."""
    per_sample = {}
    for s in samples:
        n_missing = log2_matrix[s].isna().sum()
        n_total = len(log2_matrix)
        per_sample[s] = {
            "n_missing": int(n_missing),
            "n_total": int(n_total),
            "rate": float(n_missing / n_total) if n_total > 0 else 0,
        }

    per_condition = {}
    for cond in sorted(set(sample_to_condition[s] for s in samples)):
        cond_samples = [s for s in samples if sample_to_condition[s] == cond]
        cond_data = log2_matrix[cond_samples]
        # Protein-level: missing if ALL samples in condition are NaN
        all_missing = cond_data.isna().all(axis=1).sum()
        any_missing = cond_data.isna().any(axis=1).sum()
        per_condition[cond] = {
            "completely_missing": int(all_missing),
            "partially_missing": int(any_missing - all_missing),
            "complete": int(len(cond_data) - any_missing),
        }

    return {"per_sample": per_sample, "per_condition": per_condition}


def _compute_distribution_stats(log2_matrix, samples, sample_to_condition):
    """Per-sample distribution summaries for boxplots."""
    stats = {}
    for s in samples:
        vals = log2_matrix[s].dropna()
        if len(vals) == 0:
            continue
        stats[s] = {
            "condition": sample_to_condition.get(s, "Unknown"),
            "median": float(vals.median()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "q25": float(vals.quantile(0.25)),
            "q75": float(vals.quantile(0.75)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(len(vals)),
        }
    return stats


def _compute_silhouette(log2_matrix, samples, sample_to_condition):
    """Silhouette score measuring how well samples cluster by condition."""
    data = log2_matrix[samples].T.dropna(axis=1)
    data = data.fillna(data.median())
    labels = [sample_to_condition.get(s, "Unknown") for s in data.index]

    if len(set(labels)) < 2 or len(data) < 3:
        return {"score": float("nan"), "interpretation": "Not enough data"}

    try:
        score = float(sklearn_silhouette_score(data.values, labels))
    except Exception:
        score = float("nan")

    if np.isnan(score):
        interpretation = "Could not compute"
    elif score > 0.5:
        interpretation = "Strong condition separation"
    elif score > 0.25:
        interpretation = "Moderate condition separation"
    elif score > 0:
        interpretation = "Weak condition separation"
    else:
        interpretation = "No condition separation (possible batch effect)"

    return {"score": score, "interpretation": interpretation}


def _compute_variance_decomposition(
    log2_matrix,
    samples,
    sample_to_condition,
    sample_to_plex,
):
    """
    PVCA-like variance decomposition: what fraction of variance
    is explained by condition vs plex?

    Uses a simple ANOVA-style decomposition on PCA coordinates.
    """
    data = log2_matrix[samples].T.dropna(axis=1)
    data = data.fillna(data.median())

    conditions = [sample_to_condition.get(s, "Unknown") for s in data.index]
    plexes = [sample_to_plex.get(s, "plex1") for s in data.index]

    # PCA first (reduce to top components explaining 90% variance)
    n_components = min(10, data.shape[0] - 1, data.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    pc_data = pca.fit_transform(data)

    # For each PC, compute R² for condition and plex using simple 1-way ANOVA
    condition_r2_weighted = 0
    plex_r2_weighted = 0
    total_var_explained = sum(pca.explained_variance_ratio_[:n_components])

    for i in range(n_components):
        pc_values = pc_data[:, i]
        weight = pca.explained_variance_ratio_[i] / total_var_explained

        cond_r2 = _anova_r2(pc_values, conditions)
        plex_r2 = _anova_r2(pc_values, plexes)

        condition_r2_weighted += cond_r2 * weight
        plex_r2_weighted += plex_r2 * weight

    residual = max(0, 1 - condition_r2_weighted - plex_r2_weighted)

    return {
        "condition_pct": float(condition_r2_weighted * 100),
        "plex_pct": float(plex_r2_weighted * 100),
        "residual_pct": float(residual * 100),
        "n_pcs_used": n_components,
        "total_var_captured": float(total_var_explained * 100),
    }


def _anova_r2(values, groups):
    """Simple one-way ANOVA R² (SS_between / SS_total)."""
    values = np.array(values, dtype=float)
    groups = np.array(groups)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0.0

    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0

    ss_between = 0
    for g in unique_groups:
        mask = groups == g
        group_mean = np.mean(values[mask])
        ss_between += mask.sum() * (group_mean - grand_mean) ** 2

    return float(ss_between / ss_total)


def _compute_de_quality(de_results):
    """Compute DE quality metrics from results DataFrame."""
    de = de_results.copy()

    n_total = len(de)
    n_up = int((de["significance"] == "UP").sum())
    n_down = int((de["significance"] == "DOWN").sum())
    n_de = n_up + n_down

    # P-value histogram (binned for plotting)
    if "pvalue" in de.columns:
        pvals = de["pvalue"].dropna()
    elif "adj_pvalue" in de.columns:
        pvals = de["adj_pvalue"].dropna()
    else:
        pvals = pd.Series(dtype=float)

    # Raw p-value histogram (20 bins, 0 to 1)
    if len(pvals) > 0 and "pvalue" in de.columns:
        raw_pvals = de["pvalue"].dropna().values
        hist_counts, hist_edges = np.histogram(raw_pvals, bins=20, range=(0, 1))
        pval_histogram = {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        }

        # pi1 estimate (Storey): proportion of true positives
        # pi0 estimated from p-values in (0.5, 1) as flat region
        high_pvals = raw_pvals[(raw_pvals > 0.5) & (raw_pvals <= 1.0)]
        if len(high_pvals) > 0:
            pi0_est = len(high_pvals) / (0.5 * len(raw_pvals))
            pi0_est = min(pi0_est, 1.0)
            pi1 = 1 - pi0_est
        else:
            pi1 = float("nan")
    else:
        pval_histogram = None
        pi1 = float("nan")

    # Effect size distribution for significant genes
    sig_mask = de["significance"].isin(["UP", "DOWN"])
    if sig_mask.any():
        sig_fc = de.loc[sig_mask, "log2FC"].abs()
        effect_size = {
            "median": float(sig_fc.median()),
            "mean": float(sig_fc.mean()),
            "q25": float(sig_fc.quantile(0.25)),
            "q75": float(sig_fc.quantile(0.75)),
        }
    else:
        effect_size = {"median": 0, "mean": 0, "q25": 0, "q75": 0}

    # Direction balance (UP/DOWN ratio)
    if n_de > 0:
        balance = min(n_up, n_down) / max(n_up, n_down)
    else:
        balance = float("nan")

    return {
        "n_proteins_tested": n_total,
        "n_up": n_up,
        "n_down": n_down,
        "n_de": n_de,
        "de_fraction": float(n_de / n_total) if n_total > 0 else 0,
        "pi1_estimate": float(pi1),
        "direction_balance": float(balance),
        "effect_size": effect_size,
        "pvalue_histogram": pval_histogram,
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def generate_qc_report(
    protein_df: pd.DataFrame,
    sample_to_condition: dict[str, str],
    sample_to_plex: Optional[dict[str, str]] = None,
    de_results: Optional[pd.DataFrame] = None,
    output_html: str = "qc_report.html",
    title: str = "QC Report",
    is_log2: bool = False,
    **options,
) -> str:
    """
    Generate interactive HTML QC report.

    Parameters
    ----------
    protein_df : pd.DataFrame
        Wide-format protein matrix.
    sample_to_condition : dict
        Sample -> condition mapping.
    sample_to_plex : dict, optional
        Sample -> plex mapping.
    de_results : pd.DataFrame, optional
        DE results for DE quality analysis.
    output_html : str
        Output HTML file path.
    title : str
        Report title.
    is_log2 : bool
        If True, data is already in log2 space.
    **options
        Optional keyword inputs:

        - ``highlight_genes`` (list): protein accessions to highlight.
        - ``peptides``, ``unnormalized_peptides`` (path or DataFrame): normalized
          and pre-normalization peptide exports for paired distributions.
        - ``unimputed_matrix`` (path or DataFrame): protein matrix before
          imputation, used for missingness and filled-cell diagnostics.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    metrics = compute_qc_metrics(
        protein_df,
        sample_to_condition,
        sample_to_plex,
        de_results,
        is_log2,
    )

    if options.get("unimputed_matrix") is not None:
        options["unimputed_matrix"] = read_frame(options["unimputed_matrix"])
        metrics["missing"] = _compute_missing_values(
            prepare_log2(options["unimputed_matrix"], is_log2),
            metrics["pca"]["samples"],
            sample_to_condition,
        )
    metrics["diagnostics"] = diagnostic_plots(
        protein_df, sample_to_condition, is_log2, **options
    )
    html = _build_qc_html(title, metrics, options.get("highlight_genes"), de_results)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("QC report saved to %s", output_html)
    return output_html


def _safe_json(obj):
    """Serialize to JSON and escape closing script tags for safe HTML embedding."""
    return json.dumps(obj).replace("</", "<\\/")


def _cv_axis_range(condition_cv):
    """Return the CV axis range, or None so plotly autoranges without any CV."""
    # Conditions with a single sample have no CV values.
    cv_q99 = [cv["q99"] for cv in condition_cv.values() if cv["cv_values"]]
    return [0, max(cv_q99) * 1.1] if cv_q99 else None


def _plex_sections(n_plexes, pvca):
    """Build the PVCA bars and plex-related HTML pieces for the QC report."""
    # With a single plex there is no batch to assess, so plex variance is not shown.
    multi_plex = n_plexes > 1
    pvca_sources = [("Condition", pvca["condition_pct"], CONDITION_COLORS[0])]
    if multi_plex:
        pvca_sources.append(("Plex", pvca["plex_pct"], CONDITION_COLORS[1]))
    pvca_sources.append(
        ("Residual", pvca["residual_pct"], SIGNIFICANCE_COLORS["Unchanged"])
    )
    pvca_bars = {
        "x": [name for name, _, _ in pvca_sources],
        "y": [round(value, 1) for _, value, _ in pvca_sources],
        "colors": [color for _, _, color in pvca_sources],
    }
    plex_class = (
        "good"
        if pvca["plex_pct"] < 10
        else "warning"
        if pvca["plex_pct"] < 25
        else "bad"
    )
    plex_label = (
        "Good"
        if pvca["plex_pct"] < 10
        else "Warning"
        if pvca["plex_pct"] < 25
        else "High"
    )
    plex_stat = (
        f"""<div class="header-stat">
            <div class="number {plex_class}">{pvca["plex_pct"]:.1f}%</div>
            <div class="label">Var: Plex</div>
        </div>"""
        if multi_plex
        else ""
    )
    plex_row = (
        f"""<tr><td>Plex (batch)</td>
                                <td><b>{pvca["plex_pct"]:.1f}%</b></td>
                                <td>{plex_label} batch effect</td></tr>"""
        if multi_plex
        else ""
    )
    pvca_intro = (
        "What fraction of total variance is explained by biology (condition) vs technical "
        "batch (plex)? Good normalization maximizes condition and minimizes plex contribution."
        if multi_plex
        else "What fraction of total variance is explained by biology (condition)? All samples "
        "come from one plex, so batch variance is not assessed."
    )
    return {
        "multi_plex": multi_plex,
        "pvca_bars": pvca_bars,
        "plex_stat": plex_stat,
        "plex_row": plex_row,
        "pvca_intro": pvca_intro,
    }


def _build_qc_html(title, metrics, highlight_genes, de_results):
    """Build the complete QC report HTML."""
    _ = highlight_genes, de_results
    cards, diagnostic_js = plot_cards(metrics.get("diagnostics", []))
    summary = metrics["summary"]
    pca = metrics["pca"]
    corr = metrics["correlation"]
    condition_cv = metrics["condition_cv"]
    intra_corr = metrics["intra_condition_corr"]
    missing = metrics["missing"]
    distributions = metrics["distributions"]
    silhouette = metrics["silhouette"]
    pvca = metrics["pvca"]
    de_quality = metrics.get("de_quality")

    plex = _plex_sections(summary["n_plexes"], pvca)

    # Format condition CV summary
    cv_summary_rows = ""
    for cond, cv_data in condition_cv.items():
        cv_summary_rows += (
            f"<tr><td>{cond}</td>"
            f"<td>{cv_data['median_cv']:.3f}</td>"
            f"<td>{cv_data['q25']:.3f} - {cv_data['q75']:.3f}</td>"
            f"<td>{intra_corr.get(cond, float('nan')):.3f}</td></tr>"
        )

    # Format missing value summary
    missing_summary_rows = ""
    for cond, m in missing["per_condition"].items():
        total = m["complete"] + m["partially_missing"] + m["completely_missing"]
        missing_summary_rows += (
            f"<tr><td>{cond}</td>"
            f"<td>{m['complete']}</td>"
            f"<td>{m['partially_missing']}</td>"
            f"<td>{m['completely_missing']}</td>"
            f"<td>{total}</td></tr>"
        )

    # DE quality section
    de_section = ""
    if de_quality:
        de_section = f"""
        <div class="card">
            <h3>DE Quality</h3>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-value">{de_quality["n_proteins_tested"]}</div>
                    <div class="metric-label">Tested</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value sig-up">
                        {de_quality["n_up"]}</div>
                    <div class="metric-label">UP</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value sig-down">
                        {de_quality["n_down"]}</div>
                    <div class="metric-label">DOWN</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{de_quality["pi1_estimate"]:.2f}</div>
                    <div class="metric-label">&pi;1 (true +)</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{de_quality["effect_size"]["median"]:.2f}</div>
                    <div class="metric-label">Median |log2FC|</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{de_quality["direction_balance"]:.2f}</div>
                    <div class="metric-label">Direction balance</div>
                </div>
            </div>
            <div class="row" style="margin-top:16px">
                <div style="flex:1"><div id="pval-hist" style="height:300px"></div></div>
            </div>
        </div>
        """

    condition_names = category_order(pca["conditions"])

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html_module.escape(title)}</title>
    <script>{get_plotlyjs()}</script>
    <style>
        {REPORT_CSS}
        .header-stats {{ display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }}
        .header-stat {{ background: #f5f5f5; padding: 8px 14px;
                        border-radius: 6px; text-align: center; }}
        .header-stat .number {{ font-size: 20px; font-weight: bold; }}
        .header-stat .label {{ font-size: 10px; opacity: 0.8; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .row {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
        .row > div {{ min-width: 300px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                        gap: 12px; }}
        .metric-item {{ text-align: center; padding: 10px; background: #f8f9fa;
                        border-radius: 6px; }}
        .metric-value {{ font-size: 20px; font-weight: bold; color: {TEXT_COLOR}; }}
        .metric-label {{ font-size: 11px; color: #7f8c8d; margin-top: 4px; }}
        .tab-container {{ margin-bottom: 20px; }}
        .tab-buttons {{ display: flex; gap: 4px; margin-bottom: 0; }}
        .tab-btn {{ padding: 10px 20px; cursor: pointer; background: #ecf0f1;
                    border: none; border-radius: 8px 8px 0 0; font-size: 13px;
                    font-weight: 500; }}
        .tab-btn.active {{ background: white; color: {TEXT_COLOR}; font-weight: 600; }}
        .tab-content {{ display: none; background: white; border-radius: 0 8px 8px 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 16px; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>

<div class="header">
    <h1>{html_module.escape(title)}</h1>
    <div class="header-stats">
        <div class="header-stat">
            <div class="number">{summary["n_proteins"]}</div>
            <div class="label">Proteins</div>
        </div>
        <div class="header-stat">
            <div class="number">{summary["n_samples"]}</div>
            <div class="label">Samples</div>
        </div>
        <div class="header-stat">
            <div class="number">{summary["n_conditions"]}</div>
            <div class="label">Conditions</div>
        </div>
        <div class="header-stat">
            <div class="number">{summary["n_plexes"]}</div>
            <div class="label">Plexes</div>
        </div>
        <div class="header-stat">
            <div class="number {
        "good"
        if silhouette["score"] > 0.25
        else "warning"
        if silhouette["score"] > 0
        else "bad"
    }">{silhouette["score"]:.3f}</div>
            <div class="label">Silhouette</div>
        </div>
        <div class="header-stat">
            <div class="number">{pvca["condition_pct"]:.1f}%</div>
            <div class="label">Var: Condition</div>
        </div>
        {plex["plex_stat"]}
    </div>
</div>

<div class="container">

    <!-- Tab navigation -->
    <div class="tab-container">
        <div class="tab-buttons">
            <button class="tab-btn active" onclick="showTab('sample-qc')">Sample QC</button>
            <button class="tab-btn" onclick="showTab('reproducibility')">Reproducibility</button>
            <button class="tab-btn" onclick="showTab('de-quality')">DE Quality</button>
        </div>

        <!-- Tab 1: Sample QC -->
        <div id="tab-sample-qc" class="tab-content active">
            <div class="row">
                <div style="flex:1"><div id="pca-plot" style="height:450px"></div></div>
            </div>
            {
        '<p style="font-size:12px; margin-bottom:16px">Colors identify conditions. '
        "Hover text names each sample's condition and plex.</p>"
        if plex["multi_plex"]
        else ""
    }
            <div class="card">
                <h3>Variance Decomposition (PVCA)</h3>
                <p style="font-size:13px; color:#666; margin-bottom:12px">
                    {plex["pvca_intro"]}
                </p>
                <div class="row">
                    <div style="flex:1"><div id="pvca-plot" style="height:250px"></div></div>
                    <div style="flex:1">
                        <table>
                            <tr><th>Source</th><th>% Variance</th><th>Interpretation</th></tr>
                            <tr><td>Condition (biology)</td>
                                <td><b>{pvca["condition_pct"]:.1f}%</b></td>
                                <td>{
        "Good" if pvca["condition_pct"] > 30 else "Low"
    } biological signal</td></tr>
                            {plex["plex_row"]}
                            <tr><td>Residual</td>
                                <td>{pvca["residual_pct"]:.1f}%</td>
                                <td>Individual variation + noise</td></tr>
                        </table>
                    </div>
                </div>
            </div>
            <div class="row">
                <div style="flex:1"><div id="corr-plot" style="height:450px"></div></div>
                <div style="flex:1"><div id="boxplot" style="height:450px"></div></div>
            </div>
            {cards}
        </div>

        <!-- Tab 2: Reproducibility -->
        <div id="tab-reproducibility" class="tab-content">
            <div class="card">
                <h3>Per-Condition Reproducibility</h3>
                <table>
                    <tr><th>Condition</th><th>Median CV</th><th>CV IQR</th>
                        <th>Median Replicate Correlation</th></tr>
                    {cv_summary_rows}
                </table>
            </div>
            <div class="row">
                <div style="flex:1"><div id="cv-violin" style="height:400px"></div></div>
                <div style="flex:1">
                    <div class="card">
                        <h3>Missing Value Patterns</h3>
                        <table>
                            <tr><th>Condition</th><th>Complete</th><th>Partial Missing</th>
                                <th>Fully Missing</th><th>Total</th></tr>
                            {missing_summary_rows}
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 3: DE Quality -->
        <div id="tab-de-quality" class="tab-content">
            {
        de_section
        if de_section
        else '<div class="card"><h3>No DE results provided</h3></div>'
    }
        </div>
    </div>

</div>

<script>
// Tab switching
function showTab(tabId) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');
    event.target.classList.add('active');
    // Trigger resize for plotly
    window.dispatchEvent(new Event('resize'));
}}

const multiPlex = {_safe_json(plex["multi_plex"])};
const sampleConditions = {_safe_json(dict(zip(pca["samples"], pca["conditions"])))};
const missingColor = {_safe_json(MISSING_COLOR)};
function paddedRange(values) {{
    const low = Math.min(...values), high = Math.max(...values);
    const padding = (high - low || 1) * 0.15;
    return [low - padding, high + padding];
}}
function sampleOrder(a, b) {{
    return categoryOrder(sampleConditions[a], sampleConditions[b]) || a.localeCompare(b);
}}

// Shared colors are computed in Python for every condition panel.
const condColors = {
        _safe_json(dict(zip(condition_names, category_colors(len(condition_names)))))
    };
const singleColor = {_safe_json(SINGLE_SERIES_COLOR)};
const sequentialColors = {_safe_json(SEQUENTIAL_COLORS)};
const plotConfig = {{responsive: true, displaylogo: false, displayModeBar: 'hover'}};
const sampleAxis = {_safe_json(sample_axis(len(distributions)))};
const plotTemplate = {_safe_json(PLOTLY_TEMPLATE)};
const pca = {_safe_json(pca)};
{EMBEDDING_LABEL_JS}
function embeddingTraces(data, xKey, yKey) {{
    const conditions = [...new Set(data.conditions)].sort(categoryOrder);
    const plexes = [...new Set(data.plexes)].sort();
    const traces = [];
    conditions.forEach(condition => {{
        let first = true;
        plexes.forEach(plex => {{
            const indices = data.samples.map((sample, i) => i)
                .filter(i => data.conditions[i] === condition && data.plexes[i] === plex);
            if (!indices.length) return;
            traces.push({{
                type: 'scatter', mode: 'markers', cliponaxis: false,
                meta: {{condition}},
                x: indices.map(i => data[xKey][i]), y: indices.map(i => data[yKey][i]),
                text: indices.map(i => data.samples[i]),
                hovertext: indices.map(i => data.samples[i] + '<br>Condition: ' + condition
                    + (multiPlex ? '<br>Plex: ' + plex : '')), hoverinfo: 'text',
                marker: {{color: condColors[condition], size: 7, opacity: 0.85,
                    symbol: 'circle',
                    line: {{width: 0}}}},
                name: condition, legendgroup: condition, showlegend: first,
            }});
            first = false;
        }});
    }});
    return traces;
}}
function drawEmbedding(target, data, axes, title) {{
    const traces = embeddingTraces(data, axes[0][0], axes[1][0]);
    const outlines = Object.values(data.ellipses);
    Plotly.newPlot(target, [...ellipseTraces(data.ellipses, condColors), ...traces,
            ...labelTraces(traces)], {{
        template: plotTemplate,
        title: {{x: 0, xanchor: 'left', font: {{size: 14}},
            text: title + (traces.length === 1 ? ' (' + traces[0].name + ')' : '')}},
        xaxis: {{title: {{text: axes[0][1]}},
            range: paddedRange(data[axes[0][0]].concat(...outlines.map(outline => outline.x)))}},
        yaxis: {{title: {{text: axes[1][1]}},
            range: paddedRange(data[axes[1][0]].concat(...outlines.map(outline => outline.y)))}},
        showlegend: traces.length > 1, legend: {{font: {{size: 10}}}},
        margin: {{t: 50, b: 50, l: 60, r: 35}},
        ...categoryLayout(target, traces, 450),
    }}, plotConfig).then(labelEmbedding);
}}
drawEmbedding('pca-plot', pca,
    [['pc1', 'PC1 (' + pca.var_explained[0].toFixed(1) + '%)'],
     ['pc2', 'PC2 (' + pca.var_explained[1].toFixed(1) + '%)']], 'PCA');
// PVCA bar chart
(function() {{
const pvcaBars = {_safe_json(plex["pvca_bars"])};
Plotly.newPlot('pvca-plot', [{{
    type: 'bar',
    x: pvcaBars.x,
    y: pvcaBars.y,
    marker: {{ color: pvcaBars.colors }},
    text: pvcaBars.y.map(v => v.toFixed(1) + '%'),
    textposition: 'auto',
}}], {{
    template: plotTemplate,
    showlegend: false,
    title: {{ x: 0, xanchor: 'left', text: 'Variance Decomposition', font: {{ size: 14 }} }},
    yaxis: {{ title: {{ text: '% Variance Explained' }}, range: [0, 100] }},
    bargap: 0.6,
    margin: {{ t: 50, b: 40, l: 60, r: 20 }},
}}, plotConfig);
}})();

// Correlation heatmap
(function() {{
    const corr = {_safe_json(corr)};
    const order = corr.samples.map((sample, index) => index)
        .sort((a, b) => sampleOrder(corr.samples[a], corr.samples[b]));
    const samples = order.map(index => corr.samples[index]);
    const matrix = order.map(y => order.map(x => corr.matrix[y][x]));
    const offDiagonal = matrix.flatMap((row, y) => row.filter((value, x) =>
        x !== y && Number.isFinite(value)));
    // Reduced, as spreading a large cohort's cells into Math.min overflows the stack.
    const lowest = offDiagonal.reduce((low, value) => Math.min(low, value), Infinity);
    Plotly.newPlot('corr-plot', [{{
        type: 'heatmap',
        z: matrix,
        x: samples,
        y: samples,
        colorscale: sequentialColors.map((color, index) =>
            [index / (sequentialColors.length - 1), color]),
        zmin: Math.floor(lowest * 100) / 100,
        zmax: 1.0,
        hovertemplate: '%{{x}} vs %{{y}}: %{{z:.3f}}<extra></extra>',
    }}], {{
        template: plotTemplate,
        title: {{ x: 0, xanchor: 'left', text: 'Sample Correlation (Pearson)',
                  font: {{ size: 14 }} }},
        plot_bgcolor: missingColor, showlegend: false,
        xaxis: {{ ...sampleAxis, tickangle: 45, tickfont: {{ size: 9 }},
                  categoryorder: 'array', categoryarray: samples }},
        yaxis: {{ ...sampleAxis, tickfont: {{ size: 9 }},
                  categoryorder: 'array', categoryarray: samples }},
        margin: {{ t: 50, b: 100, l: 100, r: 20 }},
    }}, plotConfig);
}})();

// Per-sample boxplots (distribution)
(function() {{
    const dist = {json.dumps(distributions)};
    const samples = Object.keys(dist).sort(sampleOrder);
    const boxTraces = [];
    const conds = [...new Set(samples.map(sample => dist[sample].condition))].sort(categoryOrder);
    conds.forEach(cond => {{
        const group = samples.filter(sample => dist[sample].condition === cond);
        boxTraces.push({{
            type: 'box', name: cond, x: group,
            q1: group.map(sample => dist[sample].q25),
            median: group.map(sample => dist[sample].median),
            q3: group.map(sample => dist[sample].q75),
            lowerfence: group.map(sample => dist[sample].min),
            upperfence: group.map(sample => dist[sample].max),
            marker: {{color: condColors[cond]}}, line: {{width: 1}}, boxpoints: false,
        }});
    }});
    Plotly.newPlot('boxplot', boxTraces, {{
        template: plotTemplate, showlegend: boxTraces.length > 1 && boxTraces.length <= 12,
        title: {{ x: 0, xanchor: 'left',
                  text: 'Per-Sample Distribution (quartiles; whiskers=min/max)',
                  font: {{ size: 14 }} }},
        yaxis: {{ title: {{ text: '{
        "log2 ratio" if summary["is_log2"] else "log2 intensity"
    }' }} }},
        xaxis: {{...sampleAxis, tickangle: 45, tickfont: {{size: 9}},
                 categoryorder: 'array', categoryarray: samples}},
        margin: {{ t: 50, b: 100, l: 60, r: 20 }},
    }}, plotConfig);
}})();

// CV violin/box by condition
(function() {{
    const cvData = {json.dumps({k: v["cv_values"] for k, v in condition_cv.items()})};
    const traces = [];
    Object.keys(cvData).sort(categoryOrder).forEach(cond => {{
        traces.push({{
            type: 'violin', points: false,
            name: cond,
            y: cvData[cond],
            box: {{ visible: true }},
            meanline: {{ visible: true }},
            marker: {{ color: condColors[cond] }},
            line: {{ width: 1 }},
        }});
    }});
    Plotly.newPlot('cv-violin', traces, {{
        // Beyond 12 conditions a legend would take the plot's width; hover names them.
        template: plotTemplate, showlegend: traces.length > 1 && traces.length <= 12,
        title: {{ x: 0, xanchor: 'left', text: 'Per-Protein CV Distribution by Condition',
                  font: {{ size: 14 }} }},
        xaxis: {{ showticklabels: traces.length <= 12 }},
        yaxis: {{ title: {{ text: 'Coefficient of Variation' }}, range: {
        _safe_json(_cv_axis_range(condition_cv))
    } }},
        margin: {{ t: 50, b: 40, l: 60, r: 20 }},
    }}, plotConfig);
}})();

{diagnostic_js}

// DE quality plots
{_build_de_quality_js(de_quality) if de_quality else ""}

</script>

<div style="text-align:center; padding:20px; color:#999; font-size:11px;">
    Generated by <a href="https://github.com/bigbio/mokume" style="color:#666;">mokume</a>
</div>
</body>
</html>"""


def _build_de_quality_js(de_quality):
    """Build JavaScript for DE quality plots."""
    js = ""

    # P-value histogram
    if de_quality.get("pvalue_histogram"):
        hist = de_quality["pvalue_histogram"]
        bin_centers = [
            (hist["edges"][i] + hist["edges"][i + 1]) / 2
            for i in range(len(hist["counts"]))
        ]
        js += f"""
(function() {{
    Plotly.newPlot('pval-hist', [{{
        type: 'bar',
        x: {json.dumps(bin_centers)},
        y: {json.dumps(hist["counts"])},
        marker: {{ color: singleColor }},
    }}], {{
        template: plotTemplate,
        showlegend: false,
        title: {{ x: 0, xanchor: 'left', text: 'P-value Distribution (raw)',
                  font: {{ size: 13 }} }},
        xaxis: {{ title: {{ text: 'p-value' }}, range: [0, 1] }},
        yaxis: {{ title: {{ text: 'Count' }} }},
        margin: {{ t: 40, b: 50, l: 50, r: 20 }},
        shapes: [{{
            type: 'line', x0: 0, x1: 1,
            y0: {sum(hist["counts"]) / 20}, y1: {sum(hist["counts"]) / 20},
            line: {{ color: '#777777', width: 1, dash: 'dash' }}
        }}],
        annotations: [{{
            x: 0.75, y: {sum(hist["counts"]) / 20}, yanchor: 'bottom',
            text: 'Expected under null',
            showarrow: false, font: {{ size: 10 }}
        }}],
    }}, plotConfig);
}})();
"""

    return js
