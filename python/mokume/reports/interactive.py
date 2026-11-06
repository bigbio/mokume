"""Standalone DE report with a volcano plot, grouped expression points and results.

Volcano heights use raw p-values. Colors use adjusted p-values and fold-change
cutoffs. The caller supplies the condition mapping for the current contrast.
"""

from string import Template

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from mokume.reports._plots import VOLCANO_LABEL_JS, compact_json as dump
from mokume.core.logger import get_logger
from mokume.plotting._common import (
    ordered_samples,
    short_protein_name,
    volcano_label_rows,
)
from mokume.reports._de_diagnostics import PROFILE_CARD, ma_payload, profile_script
from mokume.plotting.style import (
    PLOTLY_TEMPLATE,
    REPORT_CSS,
    SAMPLE_POINT_COLOR,
    SIGNIFICANCE_COLORS,
    TEXT_COLOR,
)

logger = get_logger("mokume.reports.interactive")


def _experimental_samples(protein_df: pd.DataFrame, sample_to_condition: dict) -> list:
    """Return matrix samples included in the caller's condition mapping."""
    return [
        sample
        for sample in protein_df.columns[1:]
        if sample_to_condition.get(sample) is not None
    ]


def _prepare_de_results(de_results: pd.DataFrame) -> pd.DataFrame:
    """Copy DE results and add raw-p volcano heights."""
    de = de_results.copy()
    de["neg_log10_p"] = -np.log10(de["pvalue"].clip(1e-300))
    return de


def _summary_counts(de: pd.DataFrame) -> dict:
    """Return DE significance summary counts."""
    return {
        "n_total": len(de),
        **{
            key: int((de["significance"] == label).sum())
            for key, label in [
                ("n_up", "UP"),
                ("n_down", "DOWN"),
                ("n_unchanged", "Unchanged"),
                ("n_not_tested", "NotTested"),
            ]
        },
    }


def _intensity_payload(protein_df: pd.DataFrame, exp_samples: list) -> dict:
    """Build per-protein intensities for the selected contrast samples."""
    protein_col = protein_df.columns[0]
    intensity_data = {}
    for _, row in protein_df.iterrows():
        intensities = {
            sample: round(float(np.log2(row[sample])), 2)
            for sample in exp_samples
            if pd.notna(row[sample]) and row[sample] > 0
        }
        if intensities:
            intensity_data[str(row[protein_col])] = intensities
    return intensity_data


def _table_rows(de: pd.DataFrame, highlight_genes: list) -> list[dict]:
    """Build sortable result rows with both raw and adjusted p-values."""
    return [
        {
            "protein": str(row["ProteinName"]),
            "log2FC": round(float(row["log2FC"]), 4),
            "pvalue": f"{float(row['pvalue']):.2e}",
            "pvalue_raw": float(row["pvalue"]),
            "adj_pvalue": f"{float(row['adj_pvalue']):.2e}",
            "adj_pvalue_raw": float(row["adj_pvalue"]),
            "sig": str(row["significance"]),
            "highlighted": str(row["ProteinName"]) in highlight_genes,
        }
        for _, row in de.iterrows()
    ]


def _volcano_traces(de: pd.DataFrame, highlight_genes: list) -> list[dict]:
    """Build significance-colored traces with outlined highlighted proteins."""
    traces = []
    for significance in ("Unchanged", "NotTested", "UP", "DOWN"):
        subset = de[de["significance"] == significance]
        if subset.empty:
            continue
        # Plotly draws rows in order; highlighted proteins go last to stay on top.
        subset = subset.iloc[
            np.argsort(subset["ProteinName"].isin(highlight_genes), kind="stable")
        ]
        highlighted = subset["ProteinName"].isin(highlight_genes)
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": subset["log2FC"].tolist(),
                "y": subset["neg_log10_p"].tolist(),
                "customdata": subset[
                    ["ProteinName", "log2FC", "pvalue", "adj_pvalue", "significance"]
                ].values.tolist(),
                "marker": {
                    "color": SIGNIFICANCE_COLORS[significance],
                    # Proteins read as density; highlighted ones stay opaque and outlined.
                    "size": np.where(highlighted, 12, 6).tolist(),
                    "opacity": np.where(highlighted, 1, 0.6).tolist(),
                    "line": {
                        "color": TEXT_COLOR,
                        "width": np.where(highlighted, 1, 0).tolist(),
                    },
                },
                "name": f"{significance} ({len(subset)})",
                "hovertemplate": "<b>%{customdata[0]}</b><br>log2FC: %{customdata[1]:.3f}<br>"
                "p-value: %{customdata[2]:.2e}<br>adj p-value: %{customdata[3]:.2e}<br>"
                "Significance: %{customdata[4]}<extra></extra>",
            }
        )
    return traces


def _highlight_annotations(de: pd.DataFrame, highlight_genes: list) -> list[dict]:
    """Label highlighted proteins at their raw-p volcano coordinates."""
    return [
        {
            "x": float(row["log2FC"]),
            "y": float(row["neg_log10_p"]),
            "text": str(row["ProteinName"]),
            "showarrow": True,
            "arrowhead": 2,
            "xanchor": "center",
            "yanchor": "middle",
            "arrowwidth": 0.7,
            "arrowcolor": SAMPLE_POINT_COLOR,
            "ax": 20,
            "ay": -25,
            "font": {"size": 10, "color": "black"},
        }
        for _, row in de[de["ProteinName"].isin(highlight_genes)].iterrows()
    ]


def _top_annotations(de, highlight_genes):
    """Name the ten lowest-p changed proteins; the page hides labels that do not fit."""
    return [
        {
            "x": float(row["log2FC"]),
            "y": float(row["neg_log10_p"]),
            "text": short_protein_name(row["ProteinName"]),
            "name": "optional",
            "showarrow": True,
            "arrowhead": 0,
            "xanchor": "center",
            "yanchor": "middle",
            "arrowwidth": 0.7,
            "arrowcolor": SAMPLE_POINT_COLOR,
            "ax": 0,
            "ay": -14,
            "font": {"size": 9, "color": TEXT_COLOR},
        }
        for _, row in volcano_label_rows(de, highlight_genes).iterrows()
    ]


def _volcano_layout(de, highlight_genes, log2fc_threshold, fdr_threshold, conditions):
    """Place the horizontal guide at the raw-p cutoff accepted by adjusted p."""
    shapes = [
        {
            "type": "line",
            "x0": value,
            "x1": value,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"dash": "dash", "color": "grey", "width": 1},
        }
        for value in (log2fc_threshold, -log2fc_threshold)
    ]
    cutoff = de.loc[de["adj_pvalue"] < fdr_threshold, "pvalue"].max()
    if pd.notna(cutoff):
        height = float(-np.log10(max(cutoff, 1e-300)))
        shapes.append(
            {
                "type": "line",
                "x0": 0,
                "x1": 1,
                "xref": "paper",
                "y0": height,
                "y1": height,
                "line": {"dash": "dash", "color": "grey", "width": 1},
            }
        )
    return {
        "template": PLOTLY_TEMPLATE,
        "xaxis": {
            "title": {
                "text": (
                    f"log2 fold change ({conditions[0]} vs {conditions[1]})"
                    if conditions
                    else "log2 Fold Change"
                )
            }
        },
        "yaxis": {"title": {"text": "-log10(p-value)"}},
        "hovermode": "closest",
        "showlegend": True,
        "shapes": shapes,
        "annotations": _highlight_annotations(de, highlight_genes)
        + _top_annotations(de, highlight_genes),
        "margin": {"t": 35, "b": 55, "l": 65, "r": 20},
    }


def generate_de_report(
    de_results: pd.DataFrame,
    protein_df: pd.DataFrame,
    sample_to_condition: dict,
    output_html: str,
    title: str = "Differential Expression Report",
    log2fc_threshold: float = 0.5,
    fdr_threshold: float = 0.05,
    **options,
) -> str:
    """Write a standalone report from DE results and contrast-specific samples.

    DE results require ProteinName, log2FC, pvalue, adj_pvalue and significance.
    Intensities are linear and converted to log2 for per-protein expression.
    Optional keyword inputs are ``highlight_genes`` (proteins to label),
    ``peptides`` (four- or six-column exports; SDRF supplies conditions) and
    ``conditions`` (the contrast's two conditions, named on the volcano x axis).
    """
    highlight_genes, peptides, conditions = _report_options(**options)
    samples = ordered_samples(
        _experimental_samples(protein_df, sample_to_condition), sample_to_condition
    )
    de = _prepare_de_results(de_results)
    traces = _volcano_traces(de, highlight_genes)
    payload = {
        **_summary_counts(de),
        "log2fc_threshold": log2fc_threshold,
        "fdr_threshold": fdr_threshold,
        "intensity_data": dump(_intensity_payload(protein_df, samples)),
        **ma_payload(de, protein_df, sample_to_condition, traces, log2fc_threshold),
        "label_layout": VOLCANO_LABEL_JS,
        "diagnostic_script": profile_script(peptides, sample_to_condition),
        "profile_html": PROFILE_CARD if peptides is not None else "",
        "table_rows": dump(_table_rows(de, highlight_genes)),
        "sample_to_condition": dump(
            {sample: sample_to_condition[sample] for sample in samples}
        ),
        "exp_samples": dump(samples),
        "plot_template": dump(PLOTLY_TEMPLATE),
        "point_color": dump(SAMPLE_POINT_COLOR),
        "text_color": dump(TEXT_COLOR),
        "volcano_traces": dump(traces),
        "volcano_layout": dump(
            _volcano_layout(
                de, highlight_genes, log2fc_threshold, fdr_threshold, conditions
            )
        ),
    }
    with open(output_html, "w", encoding="utf-8") as stream:
        stream.write(_build_html(title, payload))
    logger.info("Interactive report saved to %s", output_html)
    return output_html


def _report_options(highlight_genes=None, peptides=None, conditions=None):
    """Unpack optional report inputs; unknown keywords raise TypeError."""
    return highlight_genes or [], peptides, conditions


def _build_html(title, payload):
    """Assemble the shared styling, report sections and chart interactions."""
    return Template(
        _REPORT_HEAD + _REPORT_BODY + _REPORT_SCRIPT + _REPORT_END
    ).substitute(
        title=title, plotly_js=get_plotlyjs(), report_css=REPORT_CSS, **payload
    )


_REPORT_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>$title</title>
<script>$plotly_js</script><style>
$report_css
.stats { display: flex; gap: 20px; margin-top: 10px; flex-wrap: wrap; }
.stat-box { background: #f5f5f5; padding: 8px 16px;
            border-radius: 6px; text-align: center; }
.stat-box .number { font-size: 22px; font-weight: bold; }
.stat-box .label { font-size: 11px; opacity: 0.8; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
.row { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
.col-8 { flex: 2; min-width: 300px; }
.col-4 { flex: 1; min-width: 300px; }
#volcano-plot { width: 100%; height: 500px; }
#expression-plot { width: 100%; height: 350px; }
.table-controls { margin-bottom: 10px; display: flex; gap: 10px; align-items: center; }
.table-controls input, .table-controls select { padding: 6px 8px; border: 1px solid #ddd;
                                               border-radius: 4px; font-size: 13px; }
#de-table th { cursor: pointer; position: sticky; top: 0; }
#de-table tr { cursor: pointer; }
#de-table tr.selected { background: #d4edda; }
.table-wrapper { max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; }
</style></head><body>
<div class="header"><h1>$title</h1><div class="stats">
<div class="stat-box"><div class="number">$n_total</div>
<div class="label">Proteins Tested</div></div>
<div class="stat-box stat-up"><div class="number">$n_up</div>
<div class="label">Upregulated</div></div>
<div class="stat-box stat-down"><div class="number">$n_down</div>
<div class="label">Downregulated</div></div>
<div class="stat-box"><div class="number">$n_unchanged</div><div class="label">Unchanged</div></div>
<div class="stat-box"><div class="number">$n_not_tested</div>
<div class="label">Not tested</div></div>
<div class="stat-box"><div class="number">$log2fc_threshold</div>
<div class="label">|log2FC| cutoff</div></div>
<div class="stat-box"><div class="number">$fdr_threshold</div>
<div class="label">FDR cutoff</div></div>
</div></div>
"""

_REPORT_BODY = """
<div class="container"><div class="row">
<div class="col-8"><div class="card"><h3>Interactive Volcano Plot</h3>
<div id="volcano-plot"></div><p class="info-text">Colors use adjusted p-value and |log2FC| cutoffs.
Click a point to view per-sample expression.</p></div></div>
<div class="col-8 card"><h3>MA Plot</h3><div id="ma-plot" style="height:500px"></div></div>
</div><div class="row">
<div class="col-4"><div class="card"><h3>Protein Expression</h3><div id="expression-plot"></div>
<p class="info-text" id="expression-info">Select a protein from the volcano plot or table.</p>
</div></div>
$profile_html
</div><div class="card"><h3>Differential Expression Results</h3>
<div class="table-controls">
<input type="text" id="search-input" placeholder="Search protein accession...">
<select id="sig-filter"><option value="all">All</option><option value="UP">UP only</option>
<option value="DOWN">DOWN only</option><option value="NotTested">Not tested</option>
<option value="significant">Significant (UP + DOWN)</option></select>
<span id="table-count"></span></div>
<div class="table-wrapper"><table id="de-table"><thead><tr>
<th onclick="sortTable(0)">Protein</th>
<th onclick="sortTable(1)" style="text-align:right">log2FC</th>
<th onclick="sortTable(2)" style="text-align:right">p-value</th>
<th onclick="sortTable(3)" style="text-align:right">adj p-value</th>
<th onclick="sortTable(4)">Significance</th>
</tr></thead><tbody id="de-tbody"></tbody></table></div></div></div>
"""

_REPORT_SCRIPT = """
<script>
const intensityData = $intensity_data;
const tableRows = $table_rows;
const sampleToCondition = $sample_to_condition;
const expSamples = $exp_samples;
const plotTemplate = $plot_template;
const pointColor = $point_color;
const textColor = $text_color;
const plotConfig = {responsive: true, displaylogo: false, displayModeBar: 'hover'};
$label_layout
Plotly.newPlot('volcano-plot', $volcano_traces, $volcano_layout, plotConfig).then(labelVolcano);
Plotly.newPlot('ma-plot', $ma_traces, $ma_layout, plotConfig);
$diagnostic_script
['volcano-plot', 'ma-plot'].forEach(target => document.getElementById(target).on('plotly_click', function(data) {
    const protein = data.points[0].customdata[0];
    showExpression(protein);
    highlightTableRow(protein);
}));
function expressionTraces(data) {
    const samples = expSamples.filter(sample => sample in data);
    const conditions = [...new Set(samples.map(sample => sampleToCondition[sample]))].sort(categoryOrder);
    const x = [], y = [], names = [], shapes = [];
    conditions.forEach((condition, index) => {
        const group = samples.filter(sample => sampleToCondition[sample] === condition);
        const values = group.map(sample => data[sample]);
        x.push(...group.map((sample, i) => index +
            (group.length > 1 ? (i / (group.length - 1) - 0.5) * 0.18 : 0)));
        y.push(...values);
        names.push(...group);
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
        shapes.push({type: 'line', x0: index - 0.18, x1: index + 0.18,
            y0: mean, y1: mean, line: {color: textColor, width: 3}});
    });
    const traces = [{type: 'scatter', mode: 'markers', x, y, customdata: names,
        marker: {color: pointColor, size: 9, line: {width: 0}},
        hovertemplate: '<b>%{customdata}</b><br>log2 intensity: %{y:.2f}<extra></extra>'}];
    return {traces, shapes, conditions, samples};
}
function showExpression(protein) {
    showProfile(protein);
    const data = intensityData[protein];
    if (!data) {
        Plotly.purge('expression-plot');
        document.getElementById('expression-info').textContent = 'No intensity data for ' + protein;
        return;
    }
    const deRow = tableRows.find(row => row.protein === protein);
    const fcText = deRow ? ' | log2FC ' + deRow.log2FC.toFixed(2) + ', ' + deRow.sig : '';
    const {traces, shapes, conditions, samples} = expressionTraces(data);
    // The protein is named in the caption below, so no plot title meets the modebar.
    Plotly.newPlot('expression-plot', traces, {
        template: plotTemplate, shapes,
        // Plotly drops a name that runs past the plot edge, so long names wrap.
        xaxis: {title: {text: 'Condition'}, automargin: true, tickangle: 0,
                tickvals: conditions.map((condition, i) => i),
                ticktext: conditions.map(name => name.replace(/(.{1,18})( |$$)/g, '$$1<br>'))},
        yaxis: {title: {text: 'log2 Intensity'}, autorange: true},
        margin: {t: 30, b: 65, l: 55, r: 20}, showlegend: false,
    }, plotConfig);
    document.getElementById('expression-info').textContent =
        protein + fcText + ' | ' + samples.length + ' observed samples';
}
function populateTable(filter, search) {
    const tbody = document.getElementById('de-tbody');
    tbody.innerHTML = '';
    let count = 0;
    tableRows.forEach(row => {
        if (filter === 'UP' && row.sig !== 'UP') return;
        if (filter === 'DOWN' && row.sig !== 'DOWN') return;
        if (filter === 'NotTested' && row.sig !== 'NotTested') return;
        if (filter === 'significant' && row.sig !== 'UP' && row.sig !== 'DOWN') return;
        if (search && !row.protein.toLowerCase().includes(search.toLowerCase())) return;
        count++;
        const tr = document.createElement('tr');
        tr.id = 'row-' + row.protein;
        tr.onclick = function() { showExpression(row.protein); highlightTableRow(row.protein); };
        tr.innerHTML = '<td>' + (row.highlighted ? '<b>' + row.protein + '</b>' : row.protein) + '</td>'
            + '<td style="text-align:right">' + row.log2FC.toFixed(3) + '</td>'
            + '<td style="text-align:right">' + row.pvalue + '</td>'
            + '<td style="text-align:right">' + row.adj_pvalue + '</td>'
            + '<td class="sig-' + row.sig.toLowerCase() + '">' + row.sig + '</td>';
        tbody.appendChild(tr);
    });
    document.getElementById('table-count').textContent = count + ' / ' + tableRows.length + ' proteins';
}
function highlightTableRow(protein) {
    document.querySelectorAll('#de-table tr.selected').forEach(tr => tr.classList.remove('selected'));
    const row = document.getElementById('row-' + protein);
    if (row) {
        row.classList.add('selected');
        row.scrollIntoView({behavior: 'smooth', block: 'center'});
    }
}
const sortDir = [1, 1, 1, 1, 1];
function sortTable(col) {
    sortDir[col] *= -1;
    const keys = ['protein', 'log2FC', 'pvalue_raw', 'adj_pvalue_raw', 'sig'];
    tableRows.sort((a, b) => typeof a[keys[col]] === 'string'
        ? a[keys[col]].localeCompare(b[keys[col]]) * sortDir[col]
        : (a[keys[col]] - b[keys[col]]) * sortDir[col]);
    populateTable(document.getElementById('sig-filter').value,
                  document.getElementById('search-input').value);
}
document.getElementById('search-input').addEventListener('input', function() {
    populateTable(document.getElementById('sig-filter').value, this.value);
});
document.getElementById('sig-filter').addEventListener('change', function() {
    populateTable(this.value, document.getElementById('search-input').value);
});
populateTable('all', '');
const firstHighlight = tableRows.find(row => row.highlighted);
if (firstHighlight) showExpression(firstHighlight.protein);
</script>
"""

_REPORT_END = """
<div style="text-align:center; padding:20px; color:#999; font-size:11px;">
Generated by <a href="https://github.com/bigbio/mokume" style="color:#666;">mokume</a>
</div></body></html>
"""
