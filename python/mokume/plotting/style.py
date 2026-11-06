"""Shared palettes and presentation constants for static plots and reports."""

import numpy as np

CONDITION_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
)


def _linear_rgb(lab):
    """Convert OKLab coordinates to linear sRGB without clipping the gamut."""
    lms = (
        lab
        @ np.array(
            [
                [1, 0.3963377774, 0.2158037573],
                [1, -0.1055613458, -0.0638541728],
                [1, -0.0894841775, -1.2914855480],
            ]
        ).T
    )
    return (
        lms**3
        @ np.array(
            [
                [4.0767416621, -3.3077115913, 0.2309699292],
                [-1.2684380046, 2.6097574011, -0.3413193965],
                [-0.0041960863, -0.7034186147, 1.7076147010],
            ]
        ).T
    )


def _oklab(colors):
    """Convert hexadecimal sRGB seeds to OKLab for perceptual distances."""
    rgb = (
        np.array([[int(color[i : i + 2], 16) for i in (1, 3, 5)] for color in colors])
        / 255
    )
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = (
        rgb
        @ np.array(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        ).T
    )
    return (
        np.cbrt(lms)
        @ np.array(
            [
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660],
            ]
        ).T
    )


def _palette_candidates():
    """Enumerate an ordered OKLCH grid of in-gamut sRGB colors."""
    # Chroma stays within the seeds' range so added hues sit with Okabe-Ito.
    light, chroma, hue = np.meshgrid(
        np.linspace(0.43, 0.77, 35),
        np.linspace(0.10, 0.17, 8),
        np.deg2rad(np.arange(0, 360, 5)),
        indexing="ij",
    )
    lab = np.column_stack(
        (light.ravel(), (chroma * np.cos(hue)).ravel(), (chroma * np.sin(hue)).ravel())
    )
    rgb = _linear_rgb(lab)
    valid = ((rgb >= 0) & (rgb <= 1)).all(axis=1)
    rgb = rgb[valid]
    rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    colors = [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in np.rint(rgb * 255).astype(int)]
    return lab[valid], colors


def category_order(names):
    """Unique category names in case-insensitive order, the palette's assignment order."""
    return sorted(set(names), key=lambda name: (str(name).casefold(), str(name)))


def category_colors(n):
    """Extend six seeds greedily in OKLab, with deterministic stable prefixes."""
    colors = list(CONDITION_COLORS)
    lab, candidates = _palette_candidates()
    distances = np.sum((lab[:, None, :] - _oklab(colors)) ** 2, axis=2).min(axis=1)
    for _ in range(n - len(colors)):
        index = int(np.argmax(distances))
        colors.append(candidates[index])
        distances = np.minimum(distances, np.sum((lab - lab[index]) ** 2, axis=1))
        distances[index] = -np.inf
    return tuple(colors[:n])


SIGNIFICANCE_COLORS = {
    "UP": "#D55E00",
    "DOWN": "#0072B2",
    "Unchanged": "#BBBBBB",
    "NotTested": "#777777",
}
MISSING_COLOR = "#BDBDBD"
# Single-hue blue ramp, light (low) to dark (high), for magnitude heatmaps.
SEQUENTIAL_COLORS = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)
SINGLE_SERIES_COLOR = "#0072B2"
SAMPLE_POINT_COLOR = "#52514e"
TEXT_COLOR = "#333333"
GRID_COLOR = "#E5E5E5"
AXIS_COLOR = "#D9D9D9"
WHITE = "#ffffff"
PEPTIDE_COLOR = "#D0D0D0"
INACTIVE_COLOR = "#E5E5E5"
MATPLOTLIB_FONTS = ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans")
MATPLOTLIB_STYLE = {
    "font.size": 10,
    "font.weight": "normal",
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "figure.titlesize": 11,
    "figure.titleweight": "normal",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": TEXT_COLOR,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
}
PLOTLY_AXIS_STYLE = {
    "showgrid": True,
    "gridcolor": GRID_COLOR,
    "gridwidth": 1,
    "griddash": "solid",
    "showline": True,
    "linecolor": AXIS_COLOR,
    "linewidth": 1,
    "zeroline": False,
}
PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"color": TEXT_COLOR},
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": AXIS_COLOR,
            "font": {"color": TEXT_COLOR},
        },
        "xaxis": PLOTLY_AXIS_STYLE,
        "yaxis": PLOTLY_AXIS_STYLE,
        "bargap": 0.3,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
    },
    "data": {
        "scatter": [
            {
                "marker": {"size": 8, "line": {"width": 0}},
                "textfont": {"color": TEXT_COLOR},
            }
        ],
        "bar": [{"marker": {"line": {"width": 0}}, "textfont": {"color": TEXT_COLOR}}],
    },
}

REPORT_CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f5f5; color: #333; }}
.header {{ background: #2c3e50; color: #ffffff; padding: 20px 30px; }}
.header h1 {{ font-size: 24px; margin-bottom: 8px; }}
.stat-box, .header-stat {{ background: #f5f5f5; color: {TEXT_COLOR}; }}
.card {{ background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
         padding: 16px; margin-bottom: 20px; min-width: 0; }}
.card h3 {{ margin-bottom: 12px; font-size: 15px; color: {TEXT_COLOR};
           border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #ecf0f1; color: {TEXT_COLOR}; padding: 8px 10px; text-align: left; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f8f9fa; }}
.sig-up, .sig-down {{ font-weight: bold; }}
.sig-up::before, .sig-down::before, .sig-unchanged::before, .sig-nottested::before {{
    content: ''; display: inline-block; width: 0.7em; height: 0.7em; margin-right: 0.4em;
}}
.sig-up::before {{ background: {SIGNIFICANCE_COLORS["UP"]}; }}
.sig-down::before {{ background: {SIGNIFICANCE_COLORS["DOWN"]}; }}
.sig-unchanged::before {{ background: {SIGNIFICANCE_COLORS["Unchanged"]}; }}
.sig-nottested::before {{ background: {SIGNIFICANCE_COLORS["NotTested"]}; }}
.stat-up {{ border-left: 3px solid {SIGNIFICANCE_COLORS["UP"]}; }}
.stat-down {{ border-left: 3px solid {SIGNIFICANCE_COLORS["DOWN"]}; }}
.good::before, .warning::before, .bad::before {{
    content: ''; display: inline-block; width: 0.35em; height: 0.7em; margin-right: 0.2em;
}}
.good::before {{ background: {CONDITION_COLORS[2]}; }}
.warning::before {{ background: {CONDITION_COLORS[1]}; }}
.bad::before {{ background: {SIGNIFICANCE_COLORS["UP"]}; }}
.info-text {{ font-size: 12px; color: #666; margin-top: 8px; }}
.plot-grid {{ display: grid; gap: 0 20px;
              grid-template-columns: repeat(auto-fit, minmax(min(100%, 560px), 1fr)); }}
.plot-grid .wide {{ grid-column: 1 / -1; }}
"""
