"""QC diagnostic data: normalization, missingness, imputation and abundance.

Missingness always uses the unimputed matrix when supplied. Distribution curves
are density histograms with shared bins, drawn as lines with translucent fill.
"""

import numpy as np

from mokume.plotting._common import (
    finite_list,
    log2_matrix,
    ordered_samples,
    read_frame,
)
from mokume.plotting.profile import peptide_matrix
from mokume.plotting.style import (
    CONDITION_COLORS,
    category_colors,
    category_order,
    MISSING_COLOR,
    SINGLE_SERIES_COLOR,
    TEXT_COLOR,
)
from mokume.reports._plots import sample_axis


def imputation_values(before, after):
    """Identify originally observed and newly filled cells on aligned matrices."""
    before = before.reindex(index=after.index, columns=after.columns)
    filled = before.isna() & after.notna()
    return before.to_numpy()[before.notna()], after.to_numpy()[filled]


def completeness_counts(values):
    """Count proteins quantified in exactly k samples, including k=0."""
    return (
        values.notna()
        .sum(axis=1)
        .value_counts()
        .reindex(range(values.shape[1] + 1), fill_value=0)
        .sort_index()
    )


def _missing_subset(values):
    """Sample at most 500 incomplete proteins, stratified by missing count."""
    counts = values.isna().sum(axis=1)
    counts = counts[counts > 0].sort_values(kind="stable")
    sampled = len(counts) > 500
    if sampled:
        sizes = counts.value_counts().sort_index()
        quotas = sizes * 500 / sizes.sum()
        allocation = np.floor(quotas).astype(int)
        extra = (quotas - allocation).sort_values(ascending=False, kind="stable")
        allocation.loc[extra.index[: 500 - allocation.sum()]] += 1
        selected = []
        for count, size in allocation.items():
            group = counts[counts == count]
            positions = np.linspace(0, len(group) - 1, size, dtype=int)
            selected.extend(group.index[positions])
        counts = counts.loc[selected]
    return values.loc[counts.index], sampled


def _overview_plots(values, conditions):
    """Build sample counts, completeness and a binary missingness heatmap."""
    samples = ordered_samples(values.columns, conditions)
    values = values[samples]
    counts = values.notna().sum()
    histogram = completeness_counts(values)
    plots = [
        {
            "id": "quantified-plot",
            "title": "Quantified proteins per sample",
            "data": [
                {
                    "type": "bar",
                    "x": samples,
                    "y": counts.tolist(),
                    "customdata": finite_list(values.isna().mean() * 100),
                    "marker": {"color": SINGLE_SERIES_COLOR},
                    "hovertemplate": "%{x}<br>Proteins: %{y}<br>Missing: "
                    "%{customdata:.1f}%<extra></extra>",
                }
            ],
            "layout": {
                "showlegend": False,
                "xaxis": sample_axis(len(samples), title={"text": "Sample"}),
                "yaxis": {"title": {"text": "Quantified proteins"}},
            },
        },
        {
            "id": "completeness-plot",
            "title": "Protein completeness",
            "data": [
                {
                    "type": "bar",
                    "x": histogram.index.tolist(),
                    "y": histogram.tolist(),
                    "marker": {"color": SINGLE_SERIES_COLOR},
                }
            ],
            "layout": {
                "showlegend": False,
                "xaxis": {"title": {"text": "Quantified in k samples"}},
                "yaxis": {"title": {"text": "Proteins"}},
            },
        },
    ]
    subset, sampled = _missing_subset(values)
    if not subset.empty:
        plots.append(_missing_heatmap(subset, sampled))
    return plots


def _missing_heatmap(values, sampled):
    """Use a binary color key with observed and missing entries in the legend."""
    colors = {"Missing": MISSING_COLOR, "Observed": SINGLE_SERIES_COLOR}
    data = [
        {
            "type": "heatmap",
            "z": values.notna().astype(int).values.tolist(),
            "x": list(values.columns),
            "y": list(values.index),
            "colorscale": [[0, MISSING_COLOR], [1, SINGLE_SERIES_COLOR]],
            "zmin": 0,
            "zmax": 1,
            "showscale": False,
            "hovertemplate": "%{y}<br>%{x}<br>Observed: %{z}<extra></extra>",
        }
    ]
    data.extend(
        {
            "type": "scatter",
            "x": [None],
            "y": [None],
            "mode": "markers",
            "name": name,
            "marker": {"color": color, "size": 7},
        }
        for name, color in colors.items()
    )
    return {
        "id": "missingness-heatmap",
        "title": "Missingness by protein and sample",
        "data": data,
        "height": 550,
        "wide": True,
        "caption": ("500 proteins sampled by missing-count strata. " if sampled else "")
        + "Incomplete proteins ordered by missing count; samples by condition.",
        "layout": {
            "showlegend": True,
            "xaxis": sample_axis(values.shape[1], title={"text": "Sample"}),
            "yaxis": {
                "title": {"text": "Protein"},
                "showticklabels": False,
                "ticks": "",
            },
        },
    }


def _density_plot(groups, identifier, title):
    """Draw density curves with shared bins and ten-percent fill."""
    groups = {
        name: np.asarray(values)[np.isfinite(values)] for name, values in groups.items()
    }
    groups = {name: values for name, values in groups.items() if len(values)}
    if not groups:
        return []
    bins = np.histogram_bin_edges(np.concatenate(list(groups.values())), bins=40)
    traces = []
    for index, (name, values) in enumerate(groups.items()):
        color = CONDITION_COLORS[index]
        rgb = tuple(int(color[offset : offset + 2], 16) for offset in (1, 3, 5))
        density, edges = np.histogram(values, bins=bins, density=True)
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": name,
                "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
                "y": density.tolist(),
                "line": {"color": color, "width": 2},
                "fill": "tozeroy",
                "fillcolor": f"rgba{(*rgb, 0.1)}",
            }
        )
    return [
        {
            "id": identifier,
            "title": title,
            "data": traces,
            "layout": {
                "showlegend": len(traces) > 1,
                "xaxis": {"title": {"text": "log2 intensity"}},
                "yaxis": {"title": {"text": "Density"}},
            },
        }
    ]


def _normalization_plot(before, after, conditions):
    """Compare peptide distributions on a shared intensity axis."""
    samples = ordered_samples(after.columns, conditions)
    names = category_order(conditions[sample] for sample in samples)
    traces = []
    for panel, values in enumerate((before, after)):
        traces.extend(_peptide_boxes(values, samples, conditions, panel))
    limits = [
        float(min(before.min().min(), after.min().min())),
        float(max(before.max().max(), after.max().max())),
    ]
    return {
        "id": "normalization-plot",
        "title": "Peptide normalization: before / after",
        "data": traces,
        "wide": True,
        "caption": "Quartiles; whiskers show min/max. Shared log2 scale.",
        "layout": {
            "showlegend": 1 < len(names) <= 12,
            "xaxis": sample_axis(
                len(samples),
                domain=[0, 0.46],
                categoryorder="array",
                categoryarray=samples,
                title={"text": "Before"},
            ),
            "xaxis2": sample_axis(
                len(samples),
                domain=[0.54, 1],
                categoryorder="array",
                categoryarray=samples,
                title={"text": "After"},
            ),
            "yaxis": {
                "title": {"text": "log2 peptide intensity"},
                "range": [limits[0] - 0.5, limits[1] + 0.5],
            },
        },
    }


def _rank_plot(values, highlights):
    """Build an abundance curve and label selected proteins."""
    abundance = values.mean(axis=1).dropna().sort_values(ascending=False)
    ranks = {protein: index + 1 for index, protein in enumerate(abundance.index)}
    selected = sorted(
        (protein for protein in highlights or [] if protein in ranks), key=ranks.get
    )
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": list(ranks.values()),
            "y": finite_list(abundance),
            "customdata": list(abundance.index),
            "line": {"color": SINGLE_SERIES_COLOR, "width": 2},
            "hovertemplate": "%{customdata}<br>Rank %{x}: %{y:.2f}<extra></extra>",
        }
    ]
    if selected:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers+text",
                "text": selected,
                # Alternate sides so labels of neighboring ranks do not collide.
                "textposition": [
                    "top right" if index % 2 == 0 else "bottom left"
                    for index in range(len(selected))
                ],
                "cliponaxis": False,
                "x": [ranks[protein] for protein in selected],
                "y": finite_list(abundance.loc[selected]),
                "marker": {
                    "color": SINGLE_SERIES_COLOR,
                    "size": 9,
                    "line": {"color": TEXT_COLOR, "width": 1},
                },
                "textfont": {"color": TEXT_COLOR},
            }
        )
    return {
        "id": "abundance-rank",
        "title": "Abundance rank",
        "data": traces,
        "wide": True,
        "layout": {
            "showlegend": False,
            "xaxis": {"title": {"text": "Protein rank"}},
            "yaxis": {"title": {"text": "Mean log2 intensity"}},
        },
    }


def _diagnostic_inputs(
    highlight_genes=None,
    peptides=None,
    unnormalized_peptides=None,
    unimputed_matrix=None,
):
    """Unpack optional QC inputs; unknown keywords raise TypeError."""
    return highlight_genes, peptides, unnormalized_peptides, unimputed_matrix


def diagnostic_plots(protein_df, conditions, is_log2, **options):
    """Prepare only diagnostics supported by the supplied intermediate tables."""
    highlight_genes, peptides, unnormalized_peptides, unimputed_matrix = (
        _diagnostic_inputs(**options)
    )
    final = log2_matrix(protein_df, is_log2)
    source = (
        log2_matrix(read_frame(unimputed_matrix), is_log2)
        if unimputed_matrix is not None
        else final
    )
    samples = [sample for sample in source if sample in conditions]
    source = source[samples]
    plots = _overview_plots(source, conditions)
    plots.append(_rank_plot(final[samples], highlight_genes))
    if source.isna().any().any():
        complete = source.notna().all(axis=1)
        plots.extend(
            _density_plot(
                {
                    "Complete proteins": source[complete].mean(axis=1).values,
                    "Incomplete proteins": source[~complete].mean(axis=1).values,
                },
                "mnar-plot",
                "Missingness and observed mean abundance",
            )
        )
    if unimputed_matrix is not None:
        observed, filled = imputation_values(source, final[samples])
        plots.extend(
            _density_plot(
                {"Observed": observed, "Imputed": filled},
                "imputation-plot",
                "Observed and imputed intensities",
            )
        )
    if peptides is not None and unnormalized_peptides is not None:
        plots.append(
            _normalization_plot(
                peptide_matrix(unnormalized_peptides, conditions),
                peptide_matrix(peptides, conditions),
                conditions,
            )
        )
    return plots


def _peptide_boxes(values, samples, conditions, panel):
    """Summarize sample quartiles for one normalization panel."""
    names = category_order(conditions[sample] for sample in samples)
    traces = []
    for condition, color in zip(names, category_colors(len(names))):
        group = [sample for sample in samples if conditions[sample] == condition]
        quantiles = values[group].quantile([0, 0.25, 0.5, 0.75, 1])
        traces.append(
            {
                "type": "box",
                "x": group,
                "name": condition,
                "legendgroup": condition,
                "showlegend": panel == 0,
                "xaxis": "x" if panel == 0 else "x2",
                "boxpoints": False,
                "q1": finite_list(quantiles.loc[0.25]),
                "median": finite_list(quantiles.loc[0.5]),
                "q3": finite_list(quantiles.loc[0.75]),
                "lowerfence": finite_list(quantiles.loc[0]),
                "upperfence": finite_list(quantiles.loc[1]),
                "marker": {"color": color},
                "line": {"width": 2},
            }
        )
    return traces
