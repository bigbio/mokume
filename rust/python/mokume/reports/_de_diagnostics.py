"""MA and peptide-profile payloads for the standalone DE report."""

from string import Template

from mokume.plotting._common import finite_list
from mokume.plotting.ma import ma_values
from mokume.plotting.profile import MAX_CONNECTED_SAMPLES, peptide_payload
from mokume.plotting.style import (
    AXIS_COLOR,
    PEPTIDE_COLOR,
    PLOTLY_TEMPLATE,
    SINGLE_SERIES_COLOR,
)
from mokume.reports._plots import compact_json, condition_guides


PROFILE_CARD = (
    '<div class="col-8 card"><h3>Protein and peptide profile</h3>'
    '<div id="profile-plot" style="height:400px"></div></div>'
)


def ma_payload(de, protein_df, mapping, volcano_traces, threshold):
    """Reuse volcano hover data and colors at mean-log2/FC coordinates."""
    abundance = ma_values(de, protein_df, sorted(set(mapping.values())), mapping)
    points = de.assign(A=abundance).set_index("ProteinName")
    traces = []
    for trace in volcano_traces:
        rows = points.loc[[row[0] for row in trace["customdata"]]]
        traces.append(
            {**trace, "x": finite_list(rows["A"]), "y": finite_list(rows["log2FC"])}
        )
    shapes = [
        {
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "y0": value,
            "y1": value,
            "line": {
                "color": "grey" if value else AXIS_COLOR,
                "width": 1,
                "dash": "dash" if value else "solid",
            },
        }
        for value in (-threshold, 0, threshold)
    ]
    layout = {
        "template": PLOTLY_TEMPLATE,
        "shapes": shapes,
        "xaxis": {"title": {"text": "Mean log2 intensity (A)"}},
        "yaxis": {"title": {"text": "log2 fold change (M)"}},
        "showlegend": len(traces) > 1,
        "margin": {"t": 35, "b": 55, "l": 65, "r": 20},
    }
    return {"ma_traces": compact_json(traces), "ma_layout": compact_json(layout)}


def profile_script(peptides, mapping):
    """Create an optional, compact peptide-profile payload and renderer."""
    profiles, samples = (
        peptide_payload(peptides, mapping) if peptides is not None else ({}, [])
    )
    return Template(_PROFILE_SCRIPT).substitute(
        profiles=compact_json(profiles),
        samples=compact_json(samples),
        guides=compact_json(condition_guides(samples, mapping)),
        peptide_color=compact_json(PEPTIDE_COLOR),
        protein_color=compact_json(SINGLE_SERIES_COLOR),
        max_connected=MAX_CONNECTED_SAMPLES,
    )


_PROFILE_SCRIPT = """
const peptideProfiles = $profiles;
const profileSamples = $samples;
const peptideColor = $peptide_color;
const proteinColor = $protein_color;
function showProfile(protein) {
    const target = document.getElementById('profile-plot');
    if (!target) return;
    const peptides = peptideProfiles[protein] || [];
    const data = intensityData[protein] || {};
    // As in the static profile, many samples get dots without connecting lines.
    const dense = profileSamples.length > $max_connected;
    const traces = peptides.map((values, index) => ({
        type: 'scatter', mode: dense ? 'markers' : 'lines', x: profileSamples, y: values,
        line: {color: peptideColor, width: 1},
        marker: {color: peptideColor, size: 2, opacity: 0.5}, name: 'Peptides',
        legendgroup: 'Peptides', showlegend: index === 0,
        hovertemplate: '%{x}<br>Peptide: %{y:.2f}<extra></extra>',
    }));
    traces.push({type: 'scatter', mode: dense ? 'markers' : 'lines+markers', x: profileSamples,
        y: profileSamples.map(sample => data[sample] ?? null), name: 'Protein',
        line: {color: proteinColor, width: 2},
        marker: {color: proteinColor, size: dense ? 4 : 7, opacity: dense ? 0.85 : 1},
        hovertemplate: '%{x}<br>Protein: %{y:.2f}<extra></extra>'});
    // Like the static profile, hide sample names (hover keeps them) when
    // 12 px labels at 60 degrees would overlap, and otherwise grow the plot
    // below for the longest name so the data area keeps its height. The plot
    // width excludes the margins and the legend.
    const step = (target.clientWidth - 85 - (traces.length > 1 ? 100 : 0))
        / profileSamples.length;
    const labels = step >= 1.2 * 12 / Math.sin(Math.PI / 3);
    const context = document.createElement('canvas').getContext('2d');
    context.font = '12px "Open Sans", verdana, arial, sans-serif';
    const longest = labels
        ? Math.max(...profileSamples.map(name => context.measureText(name).width))
        : 0;
    const bottom = 55 + longest * Math.sin(Math.PI / 3);
    target.style.height = `$${345 + bottom}px`;
    Plotly.newPlot(target, traces, {template: plotTemplate, ...$guides,
        showlegend: traces.length > 1, legend: {itemsizing: 'constant'},
        xaxis: {title: {text: 'Sample'}, tickangle: 60, automargin: true,
            showticklabels: labels,
            categoryorder: 'array', categoryarray: profileSamples},
        yaxis: {title: {text: 'log2 intensity'}},
        margin: {t: 60, b: bottom, l: 65, r: 20}}, plotConfig);
}
"""
