"""
Multi-workflow comparison report for proteomics quantification.

Compares quantification workflows side-by-side using:
- PCA plots per workflow
- Summary metrics table
- DE concordance (overlap, log2FC correlation, concordance-at-top)
- Quality metrics radar chart
- Known marker gene panel

Designed to answer: "Which workflow preserves biology better?"
"""

import html
from typing import Optional

import numpy as np
from plotly.offline import get_plotlyjs

from mokume.reports.qc_report import compute_qc_metrics, _safe_json
from mokume.plotting.style import (
    category_colors,
    category_order,
    PLOTLY_TEMPLATE,
    REPORT_CSS,
    SINGLE_SERIES_COLOR,
    TEXT_COLOR,
)
from mokume.core.logger import get_logger
from mokume.reports._comparison_profiles import marker_profiles
from mokume.reports._plots import EMBEDDING_LABEL_JS, plot_cards

logger = get_logger("mokume.reports.comparison")


def generate_comparison_report(
    workflows: list[dict],
    output_html: str = "workflow_comparison.html",
    title: str = "Workflow Comparison",
    marker_genes: Optional[dict[str, dict]] = None,
) -> str:
    """
    Generate interactive HTML comparing multiple quantification workflows.

    Parameters
    ----------
    workflows : list[dict]
        Each dict has keys:
        - name: str (e.g., "IRS_globalMedian")
        - protein_df: pd.DataFrame (wide format)
        - de_results: pd.DataFrame (optional)
        - sample_to_condition: dict
        - sample_to_plex: dict (optional)
        - is_log2: bool (default False)
    output_html : str
        Output path for the HTML report.
    title : str
        Report title.
    marker_genes : dict, optional
        Expected marker genes. Format: {accession: {"name": "GENE", "expected": "UP/DOWN"}}

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    # Compute metrics for each workflow
    workflow_metrics = []
    for wf in workflows:
        metrics = compute_qc_metrics(
            protein_df=wf["protein_df"],
            sample_to_condition=wf["sample_to_condition"],
            sample_to_plex=wf.get("sample_to_plex"),
            de_results=wf.get("de_results"),
            is_log2=wf.get("is_log2", False),
        )
        workflow_metrics.append(
            {
                "name": wf["name"],
                "metrics": metrics,
                "de_results": wf.get("de_results"),
            }
        )

    # Compute cross-workflow concordance
    concordance = _compute_concordance(workflow_metrics)

    # Marker gene table and per-workflow marker profiles
    marker_html, profiles = "", ("", "")
    marker_results = (
        _compute_marker_results(workflow_metrics, marker_genes) if marker_genes else {}
    )
    if marker_results:
        marker_html = _build_marker_table(
            marker_results, marker_genes, workflow_metrics
        )
        profiles = plot_cards(marker_profiles(workflows, marker_genes))

    # Build HTML
    html_text = _build_comparison_html(
        title, workflow_metrics, concordance, marker_html, profiles
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_text)

    logger.info("Comparison report saved to %s", output_html)
    return output_html


def _compute_concordance(workflow_metrics):
    """Compute pairwise DE concordance between workflows."""
    results = {}

    for i, wf_a in enumerate(workflow_metrics):
        for j, wf_b in enumerate(workflow_metrics):
            if j <= i:
                continue
            de_a = wf_a.get("de_results")
            de_b = wf_b.get("de_results")
            if de_a is None or de_b is None:
                continue

            key = f"{wf_a['name']} vs {wf_b['name']}"

            # DE overlap
            sig_a = set(de_a[de_a["significance"].isin(["UP", "DOWN"])]["ProteinName"])
            sig_b = set(de_b[de_b["significance"].isin(["UP", "DOWN"])]["ProteinName"])
            overlap = sig_a & sig_b
            only_a = sig_a - sig_b
            only_b = sig_b - sig_a

            # Log2FC correlation (all shared proteins)
            shared_proteins = set(de_a["ProteinName"]) & set(de_b["ProteinName"])
            if shared_proteins:
                merged = de_a[["ProteinName", "log2FC"]].merge(
                    de_b[["ProteinName", "log2FC"]],
                    on="ProteinName",
                    suffixes=("_a", "_b"),
                )
                fc_corr = float(merged["log2FC_a"].corr(merged["log2FC_b"]))
                # Rank correlation
                fc_rank_corr = float(
                    merged["log2FC_a"].corr(merged["log2FC_b"], method="spearman")
                )
                fc_data = {
                    "proteins": merged["ProteinName"].tolist(),
                    "fc_a": merged["log2FC_a"].tolist(),
                    "fc_b": merged["log2FC_b"].tolist(),
                }
            else:
                fc_corr = float("nan")
                fc_rank_corr = float("nan")
                fc_data = None

            # Concordance at top (CAT): for top-k DE genes by p-value,
            # what fraction overlap?
            cat_curve = _concordance_at_top(de_a, de_b)

            # Direction concordance: of shared DE genes, how many agree on direction?
            if overlap:
                merged_sig = de_a[de_a["ProteinName"].isin(overlap)][
                    ["ProteinName", "significance"]
                ].merge(
                    de_b[de_b["ProteinName"].isin(overlap)][
                        ["ProteinName", "significance"]
                    ],
                    on="ProteinName",
                    suffixes=("_a", "_b"),
                )
                direction_agree = (
                    merged_sig["significance_a"] == merged_sig["significance_b"]
                ).sum()
                direction_concordance = float(direction_agree / len(merged_sig))
            else:
                direction_concordance = float("nan")

            results[key] = {
                "n_a": len(sig_a),
                "n_b": len(sig_b),
                "overlap": len(overlap),
                "only_a": len(only_a),
                "only_b": len(only_b),
                "jaccard": float(len(overlap) / len(sig_a | sig_b))
                if (sig_a | sig_b)
                else 0,
                "fc_pearson": fc_corr,
                "fc_spearman": fc_rank_corr,
                "direction_concordance": direction_concordance,
                "cat_curve": cat_curve,
                "fc_scatter": fc_data,
                "name_a": wf_a["name"],
                "name_b": wf_b["name"],
            }

    return results


def _concordance_at_top(de_a, de_b, max_k=200):
    """Concordance-At-Top: fraction of top-k DE genes shared."""
    # Sort by p-value
    sorted_a = de_a.sort_values("adj_pvalue")["ProteinName"].tolist()
    sorted_b = de_b.sort_values("adj_pvalue")["ProteinName"].tolist()

    max_k = min(max_k, len(sorted_a), len(sorted_b))
    ks = list(range(10, max_k + 1, 5))
    fractions = []

    for k in ks:
        top_a = set(sorted_a[:k])
        top_b = set(sorted_b[:k])
        frac = len(top_a & top_b) / k if k > 0 else 0
        fractions.append(frac)

    return {"k": ks, "fraction": fractions}


def _compute_marker_results(workflow_metrics, marker_genes):
    """Evaluate known marker genes across workflows."""
    results = {}
    for wf in workflow_metrics:
        de = wf.get("de_results")
        if de is None:
            continue
        wf_results = {}
        for acc, info in marker_genes.items():
            match = de[de["ProteinName"] == acc]
            if not match.empty:
                r = match.iloc[0]
                detected = r["significance"] in ["UP", "DOWN"]
                correct_direction = r["significance"] == info["expected"]
                wf_results[acc] = {
                    "gene": info["name"],
                    "expected": info["expected"],
                    "log2FC": float(r["log2FC"]),
                    "adj_pvalue": float(r["adj_pvalue"]),
                    "significance": str(r["significance"]),
                    "detected": detected,
                    "correct_direction": correct_direction,
                }
            else:
                wf_results[acc] = {
                    "gene": info["name"],
                    "expected": info["expected"],
                    "log2FC": None,
                    "adj_pvalue": None,
                    "significance": "NOT FOUND",
                    "detected": False,
                    "correct_direction": False,
                }
        results[wf["name"]] = wf_results
    return results


def _plex_display(workflow_metrics):
    """Return the plex-dependent header, note and title text for the comparison report."""
    # With a single plex in every workflow there is no batch to compare.
    show_plex = any(wf["metrics"]["summary"]["n_plexes"] > 1 for wf in workflow_metrics)
    criteria = (
        "Higher silhouette, higher condition variance, lower plex variance, and lower CV"
        if show_plex
        else "Higher silhouette, higher condition variance, and lower CV"
    )
    plex_header = (
        '<th title="% variance explained by technical plex (lower=better)">% Var Plex</th>'
        if show_plex
        else ""
    )
    pvca_note = (
        "<b>% Var Condition / Plex</b>: PVCA-style variance decomposition. "
        "Good workflow has high condition %, low plex %."
        if show_plex
        else "<b>% Var Condition</b>: PVCA-style variance decomposition. "
        "Good workflow has high condition %."
    )
    return {
        "show_plex": show_plex,
        "criteria": criteria,
        "plex_header": plex_header,
        "pvca_note": pvca_note,
        "pca_title": "PCA by Condition + Plex" if show_plex else "PCA by Condition",
    }


def _build_comparison_html(title, workflow_metrics, concordance, marker_html, profiles):
    """Build the comparison report HTML."""
    n_workflows = len(workflow_metrics)

    # Build summary table rows
    summary_rows = []
    for wf in workflow_metrics:
        m = wf["metrics"]
        s = m["summary"]
        sil = m["silhouette"]
        pvca = m["pvca"]
        de_q = m.get("de_quality", {})

        # Overall condition CV (median across conditions)
        all_cvs = [v["median_cv"] for v in m["condition_cv"].values()]
        median_cv = float(np.nanmedian(all_cvs)) if all_cvs else float("nan")

        # Overall intra-condition correlation
        all_corrs = list(m["intra_condition_corr"].values())
        median_corr = float(np.nanmedian(all_corrs)) if all_corrs else float("nan")

        summary_rows.append(
            {
                "name": wf["name"],
                "n_proteins": s["n_proteins"],
                "silhouette": sil["score"],
                "var_condition": pvca["condition_pct"],
                "var_plex": pvca["plex_pct"],
                "median_cv": median_cv,
                "replicate_corr": median_corr,
                "n_de": de_q.get("n_de", 0),
                "n_up": de_q.get("n_up", 0),
                "n_down": de_q.get("n_down", 0),
                "pi1": de_q.get("pi1_estimate", float("nan")),
                "effect_size": de_q.get("effect_size", {}).get("median", 0),
                "direction_balance": de_q.get("direction_balance", float("nan")),
            }
        )

    # Build PCA data per workflow
    pca_data = {}
    for wf in workflow_metrics:
        pca_data[wf["name"]] = wf["metrics"]["pca"]

    # Build concordance data
    concordance_json = {}
    for key, data in concordance.items():
        concordance_json[key] = {
            k: v
            for k, v in data.items()
            if k != "fc_scatter"  # exclude large data from main JSON
        }

    # FC scatter data (separate for size)
    fc_scatter_data = {}
    for key, data in concordance.items():
        if data.get("fc_scatter"):
            fc_scatter_data[key] = data["fc_scatter"]

    plex = _plex_display(workflow_metrics)

    condition_names = category_order(
        condition for pca in pca_data.values() for condition in pca["conditions"]
    )

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(title)}</title>
    <script>{get_plotlyjs()}</script>
    <style>
        {REPORT_CSS}
        .header p {{ opacity: 0.8; margin-top: 6px; font-size: 13px; }}
        .container {{ max-width: 1500px; margin: 0 auto; padding: 20px; }}
        .card h4 {{ font-size: 13px; color: #7f8c8d; margin: 12px 0 8px; }}
        .row {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
        .best {{ background: #ecf0f1; font-weight: bold; }}
        .worst {{ background: #f5f5f5; }}
        .neutral {{ color: #7f8c8d; }}
        .section {{ margin-bottom: 30px; }}
        .section-title {{ font-size: 18px; color: {TEXT_COLOR}; margin-bottom: 16px;
                          padding-left: 10px; border-left: 4px solid {
        SINGLE_SERIES_COLOR
    }; }}
        .pca-grid {{ display: grid;
                     grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr));
                     gap: 16px; }}
        .metric-explanation {{ font-size: 12px; color: #7f8c8d; margin-top: 8px;
                               line-height: 1.5; }}
    </style>
</head>
<body>

<div class="header">
    <h1>{html.escape(title)}</h1>
    <p>Comparing {n_workflows} quantification workflows. {
        plex["criteria"]
    } indicate better
       performance.</p>
</div>

<div class="container">

    <!-- Section 1: Summary Table -->
    <div class="section">
        <h2 class="section-title">Summary Metrics</h2>
        <div class="card">
            <table id="summary-table">
                <thead><tr>
                    <th>Workflow</th>
                    <th>Proteins</th>
                    <th title="Silhouette score: how well samples cluster by condition (-1 to 1, higher=better)">Silhouette</th>
                    <th title="% variance explained by biological condition (higher=better)">% Var Condition</th>
                    {plex["plex_header"]}
                    <th title="Median coefficient of variation within conditions (lower=better)">Median CV</th>
                    <th title="Median pairwise Pearson correlation within conditions (higher=better)">Replicate Corr</th>
                    <th>DE (UP/DOWN)</th>
                    <th title="Estimated proportion of true positives (Storey's pi1)">pi1</th>
                    <th title="Median |log2FC| of significant genes">Effect Size</th>
                    <th title="min(UP,DOWN)/max(UP,DOWN) - balance of direction (1=symmetric)">Balance</th>
                </tr></thead>
                <tbody id="summary-tbody"></tbody>
            </table>
            <p class="metric-explanation">
                <b>Silhouette</b>: Measures how well samples cluster by condition (1=perfect, 0=random, negative=wrong clusters).<br>
                {plex["pvca_note"]}<br>
                <b>Replicate Corr</b>: Median pairwise Pearson r between replicates within same condition (higher=more reproducible).<br>
                <b>pi1</b>: Estimated fraction of true positives from p-value distribution (Storey's method).<br>
                <b>Balance</b>: Ratio of smaller to larger DE direction (1=symmetric). Extreme asymmetry may indicate bias.
            </p>
        </div>
    </div>

    <!-- Section 2: PCA Comparison -->
    <div class="section">
        <h2 class="section-title">{plex["pca_title"]}</h2>
        <div class="pca-grid" id="pca-container"></div>
        {
        '<p class="metric-explanation">Colors identify conditions. '
        "Hover text names each sample's condition and plex.</p>"
        if plex["show_plex"]
        else ""
    }
    </div>

    <!-- Section 3: DE Concordance -->
    <div class="section">
        <h2 class="section-title">DE Concordance</h2>
        <div class="row">
            <div style="flex:1">
                <div class="card">
                    <h3>Pairwise Overlap</h3>
                    <table id="concordance-table"></table>
                </div>
            </div>
            <div style="flex:1">
                <div class="card">
                    <h3>Log2FC Correlation (all shared proteins)</h3>
                    <div id="fc-scatter" style="height:400px"></div>
                </div>
            </div>
        </div>
        <div class="card">
            <h3>Concordance at Top</h3>
            <p class="metric-explanation" style="margin-bottom:12px">
                For the top-k most significant proteins (by p-value), what fraction overlap
                between methods? Plateau near 1.0 = strong agreement on most significant hits.
            </p>
            <div id="cat-plot" style="height:350px"></div>
        </div>
    </div>

    <!-- Section 4: Known Markers -->
    {
        f'''
    <div class="section">
        <h2 class="section-title">Known Marker Gene Panel</h2>
        <div class="card">
            <p class="metric-explanation" style="margin-bottom:12px">
                Marker proteins with a known expected direction. A good workflow should detect
                them in that direction with strong significance.
            </p>
            {marker_html}
        </div>
        {profiles[0]}
    </div>
    '''
        if marker_html
        else ""
    }

</div>

<script>
const summaryRows = {_safe_json(summary_rows)};
const pcaData = {_safe_json(pca_data)};
const concordanceData = {_safe_json(concordance_json)};
const fcScatterData = {_safe_json(fc_scatter_data)};
const showPlex = {_safe_json(plex["show_plex"])};

const catColors = {
        _safe_json(dict(zip(concordance, category_colors(len(concordance)))))
    };
const singleColor = {_safe_json(SINGLE_SERIES_COLOR)};
const plotConfig = {{responsive: true, displaylogo: false, displayModeBar: 'hover'}};
const plotTemplate = {_safe_json(PLOTLY_TEMPLATE)};

// ---- Summary Table ----
(function() {{
    const tbody = document.getElementById('summary-tbody');
    // Find best/worst for each metric
    const metrics = ['silhouette', 'var_condition', 'var_plex', 'median_cv', 'replicate_corr', 'pi1', 'effect_size'];
    const higher_is_better = ['silhouette', 'var_condition', 'replicate_corr', 'pi1', 'effect_size'];
    const lower_is_better = ['var_plex', 'median_cv'];

    const best = {{}};
    const worst = {{}};
    metrics.forEach(m => {{
        const vals = summaryRows.map(r => r[m]).filter(v => !isNaN(v) && v !== null);
        if (vals.length === 0) return;
        if (higher_is_better.includes(m)) {{
            best[m] = Math.max(...vals);
            worst[m] = Math.min(...vals);
        }} else if (lower_is_better.includes(m)) {{
            best[m] = Math.min(...vals);
            worst[m] = Math.max(...vals);
        }}
    }});

    function cellClass(metric, value) {{
        if (isNaN(value) || value === null) return '';
        if (value === best[metric]) return 'best';
        if (value === worst[metric] && summaryRows.length > 2) return 'worst';
        return '';
    }}

    function plexCell(row) {{
        if (!showPlex) return '';
        const cls = cellClass('var_plex', row.var_plex);
        return `<td class="${{cls}}">${{row.var_plex.toFixed(1)}}%</td>`;
    }}

    summaryRows.forEach(row => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="text-align:left;font-weight:bold">${{row.name}}</td>
            <td>${{row.n_proteins}}</td>
            <td class="${{cellClass('silhouette', row.silhouette)}}">${{row.silhouette.toFixed(3)}}</td>
            <td class="${{cellClass('var_condition', row.var_condition)}}">${{row.var_condition.toFixed(1)}}%</td>
            ${{plexCell(row)}}
            <td class="${{cellClass('median_cv', row.median_cv)}}">${{row.median_cv.toFixed(3)}}</td>
            <td class="${{cellClass('replicate_corr', row.replicate_corr)}}">${{row.replicate_corr.toFixed(3)}}</td>
            <td>${{row.n_de}} (${{row.n_up}}/${{row.n_down}})</td>
            <td class="${{cellClass('pi1', row.pi1)}}">${{isNaN(row.pi1) ? 'N/A' : row.pi1.toFixed(2)}}</td>
            <td class="${{cellClass('effect_size', row.effect_size)}}">${{row.effect_size.toFixed(2)}}</td>
            <td>${{isNaN(row.direction_balance) ? 'N/A' : row.direction_balance.toFixed(2)}}</td>
        `;
        tbody.appendChild(tr);
    }});
}})();

// ---- PCA Plots ----
const condColors = {
        _safe_json(dict(zip(condition_names, category_colors(len(condition_names)))))
    };
{EMBEDDING_LABEL_JS}
function pcaTraces(pca) {{
    const conditions = [...new Set(pca.conditions)].sort(categoryOrder);
    const plexes = [...new Set(pca.plexes)].sort();
    const multiPlex = plexes.length > 1;
    const traces = [];
    conditions.forEach(cond => {{
        let first = true;
        plexes.forEach(plex => {{
            const indices = pca.samples.map((sample, i) => i)
                .filter(i => pca.conditions[i] === cond && pca.plexes[i] === plex);
            if (!indices.length) return;
            traces.push({{
                type: 'scatter', mode: 'markers', cliponaxis: false,
                meta: {{condition: cond}},
                x: indices.map(i => pca.pc1[i]), y: indices.map(i => pca.pc2[i]),
                text: indices.map(i => pca.samples[i]),
                hovertext: indices.map(i => pca.samples[i] + '<br>Condition: ' + cond
                    + (multiPlex ? '<br>Plex: ' + plex : '')), hoverinfo: 'text',
                marker: {{color: condColors[cond], size: 7, opacity: 0.85, symbol: 'circle',
                    line: {{width: 0}}}},
                name: cond, legendgroup: cond, showlegend: first,
            }});
            first = false;
        }});
    }});
    return traces;
}}
function paddedRange(values) {{
    const low = Math.min(...values), high = Math.max(...values);
    const padding = (high - low || 1) * 0.15;
    return [low - padding, high + padding];
}}
function drawPca(wfName, index) {{
    const pca = pcaData[wfName];
    const traces = pcaTraces(pca);
    const outlines = Object.values(pca.ellipses);
    Plotly.newPlot(`pca-${{index}}`, [...ellipseTraces(pca.ellipses, condColors), ...traces,
            ...labelTraces(traces)], {{
        template: plotTemplate,
        title: {{text: wfName + (traces.length === 1 ? ' (' + traces[0].name + ')' : ''),
                 x: 0, xanchor: 'left', font: {{size: 14}}}},
        xaxis: {{title: {{text: 'PC1 (' + pca.var_explained[0].toFixed(1) + '%)'}},
                 range: paddedRange(pca.pc1.concat(...outlines.map(outline => outline.x)))}},
        yaxis: {{title: {{text: 'PC2 (' + pca.var_explained[1].toFixed(1) + '%)'}},
                 range: paddedRange(pca.pc2.concat(...outlines.map(outline => outline.y)))}},
        showlegend: traces.length > 1,
        legend: {{font: {{size: 9}}, orientation: 'h', y: -70 / 230}},
        margin: {{t: 55, b: 95, l: 60, r: 45}},
        ...categoryLayout(`pca-${{index}}`, traces, 380),
    }}, plotConfig).then(labelEmbedding);
}}
(function() {{
    const container = document.getElementById('pca-container');
    const names = Object.keys(pcaData);
    names.forEach((name, index) => {{
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `<div id="pca-${{index}}" style="height:380px;width:100%"></div>`;
        container.appendChild(card);
    }});
    // Insert every card before measuring any grid item for Plotly.
    requestAnimationFrame(() => names.forEach(drawPca));
}})();

// ---- Concordance Table ----
(function() {{
    const table = document.getElementById('concordance-table');
    let html = '<thead><tr><th>Comparison</th><th>DE A</th><th>DE B</th><th>Overlap</th>' +
               '<th>Jaccard</th><th>FC Pearson</th><th>FC Spearman</th><th>Direction</th></tr></thead><tbody>';

    Object.keys(concordanceData).forEach(key => {{
        const d = concordanceData[key];
        html += `<tr>
            <td style="text-align:left">${{key}}</td>
            <td>${{d.n_a}}</td><td>${{d.n_b}}</td>
            <td><b>${{d.overlap}}</b></td>
            <td>${{d.jaccard.toFixed(3)}}</td>
            <td>${{d.fc_pearson.toFixed(3)}}</td>
            <td>${{d.fc_spearman.toFixed(3)}}</td>
            <td>${{isNaN(d.direction_concordance) ? 'N/A' : (d.direction_concordance * 100).toFixed(1) + '%'}}</td>
        </tr>`;
    }});
    html += '</tbody>';
    table.innerHTML = html;
}})();

// ---- FC Scatter ----
(function() {{
    const keys = Object.keys(fcScatterData);
    if (keys.length === 0) return;
    // Show first pair
    const key = keys[0];
    const d = fcScatterData[key];
    const cd = concordanceData[key];

    const limit = Math.max(...[...d.fc_a, ...d.fc_b].filter(Number.isFinite).map(Math.abs)) * 1.05;
    Plotly.newPlot('fc-scatter', [{{
        type: 'scatter', mode: 'markers',
        x: d.fc_a, y: d.fc_b,
        text: d.proteins,
        marker: {{ size: 6, color: singleColor, opacity: 0.6, line: {{width: 0}} }},
        hovertemplate: '<b>%{{text}}</b><br>%{{xaxis.title.text}}: %{{x:.2f}}<br>%{{yaxis.title.text}}: %{{y:.2f}}<extra></extra>',
    }}, {{
        type: 'scatter', mode: 'lines',
        x: [-limit, limit], y: [-limit, limit],
        line: {{ color: plotTemplate.layout.xaxis.linecolor, width: 1 }},
        showlegend: false,
    }}], {{
        template: plotTemplate,
        title: {{ x: 0, xanchor: 'left',
                  text: `log2FC: ${{cd.name_a}} vs ${{cd.name_b}} ` +
                      `(r=${{cd.fc_pearson.toFixed(3)}})`,
                  font: {{ size: 13 }} }},
        xaxis: {{ title: {{ text: cd.name_a + ' log2FC' }},
                  range: [-limit, limit], constrain: 'domain' }},
        yaxis: {{ title: {{ text: cd.name_b + ' log2FC' }}, range: [-limit, limit],
                  constrain: 'domain', scaleanchor: 'x', scaleratio: 1 }},
        showlegend: false,
        margin: {{ t: 50, b: 50, l: 60, r: 20 }},
    }}, plotConfig);
}})();

// ---- CAT Curve ----
(function() {{
    const traces = [];
    Object.keys(concordanceData).forEach(key => {{
        const cat = concordanceData[key].cat_curve;
        if (!cat) return;
        const color = Object.keys(concordanceData).length === 1
            ? singleColor : catColors[key];
        traces.push({{
            type: 'scatter', mode: 'lines+markers',
            x: cat.k, y: cat.fraction,
            name: key,
            line: {{width: 2, color}},
            marker: {{size: 7, color, symbol: 'circle', opacity: 0.85, line: {{width: 0}}}},
        }});
    }});
    if (traces.length === 0) return;
    Plotly.newPlot('cat-plot', traces, {{
        template: plotTemplate, showlegend: traces.length > 1,
        title: {{ x: 0, xanchor: 'left', text: 'Concordance at Top-k', font: {{ size: 14 }} }},
        xaxis: {{ title: {{ text: 'Top k proteins (by p-value)' }} }},
        yaxis: {{ title: {{ text: 'Fraction overlap' }}, range: [0, 1] }},
        margin: {{ t: 50, b: 50, l: 60, r: 20 }},
    }}, plotConfig);
}})();
{profiles[1]}
</script>

<div style="text-align:center; padding:20px; color:#999; font-size:11px;">
    Generated by <a href="https://github.com/bigbio/mokume" style="color:#666;">mokume</a>
</div>
</body>
</html>"""


def _direction_mark(result):
    """Check mark for the expected direction, cross for significance the other way."""
    if result["correct_direction"]:
        return " &#10003;"
    return " &#10007;" if result["detected"] else ""


def _build_marker_table(marker_results, marker_genes, workflow_metrics):
    """Build HTML table for marker gene results."""
    wf_names = [wf["name"] for wf in workflow_metrics if wf["name"] in marker_results]
    if not wf_names:
        return ""

    # Header row
    header = "".join(
        [
            "<tr><th>Gene</th><th>Accession</th><th>Expected</th>",
            *(f"<th colspan='2'>{wf}</th>" for wf in wf_names),
            "</tr>",
        ]
    )

    # Sub-header
    subheader = "".join(
        [
            "<tr><th></th><th></th><th></th>",
            *("<th>log2FC</th><th>Significance</th>" for _ in wf_names),
            "</tr>",
        ]
    )

    rows = []
    for acc, info in marker_genes.items():
        row = f"<tr><td><b>{info['name']}</b></td><td>{acc}</td>"
        row += f"<td class='sig-{info['expected'].lower()}'>{info['expected']}</td>"

        for wf in wf_names:
            r = marker_results[wf].get(acc, {})
            if r.get("log2FC") is not None:
                fc = f"{r['log2FC']:.2f}"
                sig = r["significance"]
                css = f"sig-{sig.lower()}"
                row += f"<td>{fc}</td><td class='{css}'>{sig}{_direction_mark(r)}</td>"
            else:
                row += "<td>-</td><td class='neutral'>NOT FOUND</td>"

        row += "</tr>"
        rows.append(row)

    return f"""
    <table>
        <thead>{header}{subheader}</thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """
