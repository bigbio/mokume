"""Wire MA and multi-contrast plots to existing DE result tables."""

import pandas as pd

from mokume.plotting.ma import plot_ma, read_protein_groups
from mokume.plotting.diagnostics import plot_fold_changes, plot_upset


def expected_ratios(args, groups):
    """Validate explicit contrast/group references before drawing any MA plots."""
    keys = {contrast[0] for contrast in args.contrast}
    expected = {key: {} for key in keys}
    for key, group, value in args.expected_log2fc:
        if key not in keys:
            raise ValueError(f"--expected-log2fc references unknown contrast: {key}")
        if groups is None or group not in set(groups):
            raise ValueError(f"--expected-log2fc references unknown group: {group}")
        expected[key][group] = float(value)
    return expected


def generate_diagnostics(args, protein_df, mapping, plot_dir):
    """Generate optional MA plots and automatic multi-contrast summaries."""
    groups = read_protein_groups(args.protein_groups) if args.protein_groups else None
    expected = expected_ratios(args, groups)
    tables = {
        key: pd.read_csv(path, float_precision="round_trip")
        for key, _a, _b, path in args.contrast
    }
    if args.ma:
        for key, condition_a, condition_b, _path in args.contrast:
            table = tables[key]
            columns = [f"mean_{condition}" for condition in (condition_a, condition_b)]
            if not all(column in table for column in columns) and not mapping:
                raise ValueError(
                    "MA plots require --sdrf when DE group means are absent"
                )
            output = plot_dir / f"ma_{key}.png"
            plot_ma(
                table,
                protein_df,
                mapping,
                output,
                {
                    "conditions": (condition_a, condition_b),
                    "groups": groups,
                    "expected": expected[key],
                    "log2fc": args.log2fc,
                    "title": f"MA Plot: {key}",
                },
            )
            print(f"MA plot saved to {output}")
    if len(tables) > 1:
        sets = {
            key: set(
                table.loc[table["significance"].isin(["UP", "DOWN"]), "ProteinName"]
            )
            for key, table in tables.items()
        }
        plot_upset(sets, plot_dir / "upset_significant.png")
        if args.highlight_protein:
            plot_fold_changes(
                tables, args.highlight_protein, plot_dir / "log2fc_across_contrasts.png"
            )
