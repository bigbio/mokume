"""Centered marker-protein profiles across quantification workflows."""

from mokume.plotting._common import finite_list, log2_matrix, ordered_samples
from mokume.plotting.style import category_colors
from mokume.reports._plots import condition_guides


def marker_profiles(workflows, marker_genes):
    """Center each workflow's marker at its own observed log2 median."""
    conditions = {
        sample: condition
        for workflow in workflows
        for sample, condition in workflow["sample_to_condition"].items()
    }
    plots = []
    for index, (protein, info) in enumerate(marker_genes.items()):
        traces, missing = _marker_traces(workflows, protein, conditions)
        order = ordered_samples(
            {sample for trace in traces for sample in trace["x"]}, conditions
        )
        plots.append(
            {
                "id": f"marker-profile-{index}",
                "title": f"{info['name']} ({protein})",
                "data": traces,
                "caption": "Each workflow centered at its protein median."
                + (" Not present in: " + ", ".join(missing) if missing else ""),
                "layout": {
                    # Room above the plot keeps condition labels below the modebar.
                    "margin": {"t": 60, "b": 100, "l": 70, "r": 30},
                    "showlegend": len(traces) > 1,
                    **condition_guides(order, conditions),
                    "xaxis": {
                        "title": {"text": "Sample"},
                        "categoryorder": "array",
                        "categoryarray": order,
                    },
                    "yaxis": {"title": {"text": "log2 intensity − row median"}},
                },
            }
        )
    return plots


def _marker_traces(workflows, protein, conditions):
    """Prepare each workflow curve, retaining absent-workflow names for the caption."""
    traces, missing = [], []
    for workflow, color in zip(workflows, category_colors(len(workflows))):
        values = log2_matrix(workflow["protein_df"], workflow.get("is_log2", False))
        if protein not in values.index:
            missing.append(workflow["name"])
            continue
        samples = ordered_samples(
            [sample for sample in values if sample in conditions], conditions
        )
        row = values.loc[protein, samples]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": workflow["name"],
                "x": samples,
                "y": finite_list(row - row.median()),
                "line": {"color": color, "width": 2},
                "marker": {
                    "size": 7,
                    "symbol": "circle",
                    "opacity": 0.85,
                    "line": {"width": 0},
                },
            }
        )
    return traces, missing
