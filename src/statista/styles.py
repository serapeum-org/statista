"""Shared style constants for statista plots.

Centralizes colors, figure sizes, marker properties, and font sizes used
across distribution, time-series, sensitivity, and EVA plots.  Every
plotting module in the package imports from here instead of hard-coding
hex strings, ensuring a consistent visual identity.

Module Attributes:
    COLORS (dict[str, str]):
        Named color palette.  Keys correspond to semantic roles
        (``"fitted_curve"``, ``"histogram_fill"``, etc.); values are
        matplotlib-compatible color strings.
    FIG_SIZES (dict[str, tuple[int, int]]):
        Named figure dimensions in inches, keyed by layout intent
        (``"single"``, ``"square"``, ``"wide"``, ``"ts_default"``).
    BOX_MEAN_PROP (dict[str, str]):
        Marker properties passed to ``matplotlib.axes.Axes.boxplot``
        via the *meanprops* argument to style the mean marker on box
        plots.
    VIOLIN_PROP (dict[str, str | float]):
        Default face/edge color and alpha for violin plot bodies.
    DEFAULT_LINEWIDTH (int):
        Standard line width for fitted curves and reference lines.
    DEFAULT_FONTSIZE (int):
        Standard font size for axis labels and legends.

Examples:
    - Access the fitted-curve color and use it in a manual plot:
        ```python
        >>> from statista.styles import COLORS
        >>> COLORS["fitted_curve"]
        '#27408B'

        ```
    - Look up a preset figure size by layout name:
        ```python
        >>> from statista.styles import FIG_SIZES
        >>> FIG_SIZES["wide"]
        (10, 5)

        ```
    - Iterate over the full palette:
        ```python
        >>> from statista.styles import COLORS
        >>> sorted(COLORS.keys())
        ['ci_bounds', 'empirical_scatter', 'fitted_curve', 'histogram_fill', 'negative_bar', 'reference_line', 'ts_line']

        ```
    - Use box-plot mean-marker properties:
        ```python
        >>> from statista.styles import BOX_MEAN_PROP
        >>> BOX_MEAN_PROP["marker"]
        'x'

        ```
"""

from __future__ import annotations

# -- Colors ------------------------------------------------------------------
COLORS: dict[str, str] = {
    "fitted_curve": "#27408B",
    "histogram_fill": "#DC143C",
    "reference_line": "#3D59AB",
    "empirical_scatter": "orangered",
    "ts_line": "steelblue",
    "negative_bar": "firebrick",
    "ci_bounds": "grey",
}
"""Named color palette for statista plots.

Each key is a semantic role describing where the color is used:

- ``"fitted_curve"`` — PDF/CDF fitted lines, violin/box faces, theoretical
  reference curves.
- ``"histogram_fill"`` — histogram bars, empirical-CDF scatter, violin/box
  edge color.
- ``"reference_line"`` — 1:1 diagonal on Q-Q and confidence-interval plots.
- ``"empirical_scatter"`` — CDF empirical scatter points.
- ``"ts_line"`` — time-series lines, CUSUM curves, positive anomaly bars,
  double-mass-curve lines.
- ``"negative_bar"`` — negative anomaly bars, box-plot mean-marker face.
- ``"ci_bounds"`` — confidence-interval upper/lower bound markers.
"""

# -- Figure sizes ------------------------------------------------------------
FIG_SIZES: dict[str, tuple[int, int]] = {
    "single": (6, 5),
    "square": (6, 6),
    "wide": (10, 5),
    "ts_default": (8, 6),
}
"""Named figure sizes in ``(width, height)`` inches.

- ``"single"`` — single-panel PDF or CDF plot.
- ``"square"`` — confidence-interval / Q-Q plot.
- ``"wide"`` — side-by-side detail plot (PDF + CDF).
- ``"ts_default"`` — time-series and sensitivity plots.
"""

# -- Marker / line properties -----------------------------------------------
BOX_MEAN_PROP: dict[str, str] = {
    "marker": "x",
    "markeredgecolor": "w",
    "markerfacecolor": "firebrick",
}
"""Marker properties for box-plot mean indicators.

Passed as the *meanprops* keyword to ``matplotlib.axes.Axes.boxplot``
when *showmeans=True*.
"""

VIOLIN_PROP: dict[str, str | float] = {
    "face": "#27408B",
    "edge": "black",
    "alpha": 0.7,
}
"""Default appearance for violin-plot bodies.

- ``"face"`` — fill color of the violin body.
- ``"edge"`` — edge color of the violin body.
- ``"alpha"`` — opacity (0 = fully transparent, 1 = fully opaque).
"""

DEFAULT_LINEWIDTH: int = 2
"""Standard line width used for fitted curves and reference lines."""

DEFAULT_FONTSIZE: int = 11
"""Standard font size for axis labels and legend text."""
