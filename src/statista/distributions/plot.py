"""Plotting functions for statistical distribution visualization.

Provides the :class:`DistributionPlot` class with static methods for
PDF, CDF, detail, and confidence-interval plots.

All methods return ``(Figure, Axes)`` tuples (or ``(Figure, (Axes, Axes))``
for :meth:`DistributionPlot.details`) and never call ``plt.show()``,
leaving display control to the caller.

Module Attributes:
    XLABEL (str):
        Default x-axis label shared across all plotting methods.

Examples:
    - Quick PDF plot with synthetic data:
        ```python
        >>> import numpy as np
        >>> from statista.distributions.plot import DistributionPlot
        >>> np.random.seed(0)
        >>> data = np.random.normal(loc=5, scale=1, size=200)
        >>> x = np.linspace(2, 8, 500)
        >>> pdf_vals = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 5) ** 2)
        >>> fig, ax = DistributionPlot.pdf(x, pdf_vals, data)
        >>> ax.get_ylabel()
        'pdf'

        ```
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statista.styles import COLORS, DEFAULT_FONTSIZE, DEFAULT_LINEWIDTH, FIG_SIZES

XLABEL = "Actual data"


class DistributionPlot:
    """Visualization utilities for statistical distributions.

    All methods are static.  Each accepts optional *fig* / *ax* parameters
    for composition into existing figures and returns ``(Figure, Axes)``
    (or ``(Figure, (Axes, Axes))`` for :meth:`details`).

    Colors are drawn from :mod:`statista.styles` so every distribution
    plot shares a consistent visual identity.  No method calls
    ``plt.show()``; rendering is left to the caller or to the Jupyter
    ``%matplotlib inline`` backend.

    Examples:
        - Create a standalone PDF plot with synthetic data:
            ```python
            >>> import numpy as np
            >>> from statista.distributions.plot import DistributionPlot
            >>> np.random.seed(0)
            >>> data = np.random.normal(loc=5, scale=1, size=200)
            >>> x = np.linspace(2, 8, 500)
            >>> pdf_vals = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 5) ** 2)
            >>> fig, ax = DistributionPlot.pdf(x, pdf_vals, data)
            >>> ax.get_ylabel()
            'pdf'

            ```
        - Compose into pre-existing axes:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from statista.distributions.plot import DistributionPlot
            >>> import numpy as np
            >>> fig, ax = plt.subplots()
            >>> x = np.linspace(0, 10, 100)
            >>> pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 5) ** 2)
            >>> fig2, ax2 = DistributionPlot.pdf(x, pdf, np.random.normal(5, 1, 50), ax=ax)
            >>> ax2 is ax
            True

            ```

    See Also:
        statista.styles: Color palette and size constants used by all methods.
    """

    @staticmethod
    def _get_or_create_ax(
        fig: Figure | None,
        ax: Axes | None,
        fig_size: tuple[float, float],
    ) -> tuple[Figure, Axes]:
        """Return the supplied fig/ax or create new ones.

        Args:
            fig: Pre-existing figure, or None.
            ax: Pre-existing axes, or None.
            fig_size: Size ``(width, height)`` in inches, used only when
                both *fig* and *ax* are None.

        Returns:
            tuple[Figure, Axes]: The figure and axes to draw on.
        """
        if ax is not None:
            result = (fig if fig is not None else ax.get_figure(), ax)
        else:
            new_fig = plt.figure(figsize=fig_size)
            result = (new_fig, new_fig.add_subplot())
        return result

    @staticmethod
    def pdf(
        qx: np.ndarray,
        pdf_fitted: np.ndarray | list,
        data_sorted: np.ndarray,
        fig_size: tuple = (6, 5),
        xlabel: str = XLABEL,
        ylabel: str = "pdf",
        fontsize: int = DEFAULT_FONTSIZE,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Create a probability density function (PDF) plot.

        Plots the fitted PDF curve (solid line) overlaid on a normalized
        histogram of the observed data so the two can be visually compared.

        Args:
            qx: X-values for the fitted PDF curve.  Typically a
                ``np.linspace`` spanning the data range.
            pdf_fitted: PDF values corresponding to each point in *qx*.
                Obtained from a distribution's ``pdf`` method.
            data_sorted: Observed data plotted as a histogram.
            fig_size: Figure size ``(width, height)`` in inches.
                Defaults to ``(6, 5)``.
            xlabel: Label for the x-axis.  Defaults to ``"Actual data"``.
            ylabel: Label for the y-axis.  Defaults to ``"pdf"``.
            fontsize: Font size for axis labels.  Defaults to 11.
            fig: Pre-existing matplotlib Figure.  When supplied together
                with *ax*, the plot is drawn on those axes instead of
                creating a new figure.
            ax: Pre-existing matplotlib Axes to draw on.

        Returns:
            tuple[Figure, Axes]: The figure and axes containing the plot.

        Examples:
            - Plot a synthetic normal PDF and inspect the axes:
                ```python
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> np.random.seed(1)
                >>> data = np.random.normal(loc=10, scale=2, size=100)
                >>> x = np.linspace(4, 16, 500)
                >>> pdf_vals = (1 / (2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 10) / 2) ** 2)
                >>> fig, ax = DistributionPlot.pdf(x, pdf_vals, data)
                >>> ax.get_xlabel()
                'Actual data'
                >>> ax.get_ylabel()
                'pdf'
                >>> len(ax.lines)
                1

                ```
            - Use custom axis labels and figure size:
                ```python
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> data = np.random.normal(0, 1, 80)
                >>> x = np.linspace(-4, 4, 300)
                >>> pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
                >>> fig, ax = DistributionPlot.pdf(
                ...     x, pdf, data, fig_size=(8, 4), xlabel="Z-score", ylabel="density"
                ... )
                >>> ax.get_xlabel()
                'Z-score'
                >>> round(fig.get_size_inches()[0])
                8

                ```
            - Compose into a pre-existing axes:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import matplotlib.pyplot as plt
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> fig, ax = plt.subplots()
                >>> x = np.linspace(0, 10, 100)
                >>> pdf = np.exp(-x)
                >>> _, returned_ax = DistributionPlot.pdf(x, pdf, np.random.exponential(1, 50), ax=ax)
                >>> returned_ax is ax
                True

                ```

        See Also:
            - DistributionPlot.cdf: For plotting cumulative distribution functions.
            - DistributionPlot.details: For plotting both PDF and CDF together.
        """
        fig, ax = DistributionPlot._get_or_create_ax(fig, ax, fig_size)
        ax.plot(
            qx, pdf_fitted, "-",
            color=COLORS["fitted_curve"], linewidth=DEFAULT_LINEWIDTH,
        )
        ax.hist(
            data_sorted, density=True, histtype="stepfilled",
            color=COLORS["histogram_fill"],
        )
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        return fig, ax

    @staticmethod
    def cdf(
        qx: np.ndarray,
        cdf_fitted: np.ndarray,
        data_sorted: np.ndarray,
        cdf_weibul: np.ndarray,
        fig_size: tuple[float, float] = (6, 5),
        xlabel: str = XLABEL,
        ylabel: str = "cdf",
        fontsize: int = DEFAULT_FONTSIZE,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Create a cumulative distribution function (CDF) plot.

        Plots the fitted CDF as a solid curve alongside the empirical CDF
        shown as hollow scatter points.  A legend distinguishes the two
        series.

        Args:
            qx: X-values for the fitted CDF curve.
            cdf_fitted: CDF values corresponding to each point in *qx*.
            data_sorted: Sorted observed data points (x-positions of the
                empirical CDF scatter).
            cdf_weibul: Empirical CDF values, typically calculated using the
                Weibull plotting-position formula ``i / (n + 1)``.
            fig_size: Figure size ``(width, height)`` in inches.
                Defaults to ``(6, 5)``.
            xlabel: Label for the x-axis.  Defaults to ``"Actual data"``.
            ylabel: Label for the y-axis.  Defaults to ``"cdf"``.
            fontsize: Font size for labels and legend.  Defaults to 11.
            fig: Pre-existing matplotlib Figure.
            ax: Pre-existing matplotlib Axes to draw on.

        Returns:
            tuple[Figure, Axes]: The figure and axes containing the plot.

        Examples:
            - Plot a synthetic normal CDF and check the legend:
                ```python
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> np.random.seed(42)
                >>> data = np.sort(np.random.normal(10, 2, 100))
                >>> cdf_emp = np.arange(1, 101) / 101
                >>> x = np.linspace(4, 16, 300)
                >>> cdf_fit = 0.5 * (1 + np.tanh(0.798 * (x - 10) / 2))
                >>> fig, ax = DistributionPlot.cdf(x, cdf_fit, data, cdf_emp)
                >>> [t.get_text() for t in ax.get_legend().get_texts()]
                ['Estimated CDF', 'Empirical CDF']

                ```
            - Use custom labels:
                ```python
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> data = np.sort(np.random.exponential(2, 60))
                >>> cdf_emp = np.arange(1, 61) / 61
                >>> x = np.linspace(0, 12, 200)
                >>> cdf_fit = 1 - np.exp(-x / 2)
                >>> fig, ax = DistributionPlot.cdf(
                ...     x, cdf_fit, data, cdf_emp,
                ...     xlabel="Flow [m³/s]", ylabel="Probability",
                ... )
                >>> ax.get_xlabel()
                'Flow [m³/s]'

                ```

        See Also:
            - DistributionPlot.pdf: For plotting probability density functions.
            - DistributionPlot.details: For plotting both PDF and CDF together.
        """
        fig, ax = DistributionPlot._get_or_create_ax(fig, ax, fig_size)
        ax.plot(
            qx, cdf_fitted, "-", label="Estimated CDF",
            color=COLORS["fitted_curve"], linewidth=DEFAULT_LINEWIDTH,
        )
        ax.scatter(
            data_sorted, cdf_weibul, label="Empirical CDF",
            color=COLORS["empirical_scatter"], facecolors="none",
        )
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.legend(fontsize=fontsize, framealpha=1)
        return fig, ax

    @staticmethod
    def details(
        qx: np.ndarray | list,
        q_act: np.ndarray | list,
        pdf: np.ndarray | list,
        cdf_fitted: np.ndarray | list,
        cdf: np.ndarray | list,
        fig_size: tuple[float, float] = (10, 5),
        xlabel: str = XLABEL,
        ylabel: str = "cdf",
        fontsize: int = DEFAULT_FONTSIZE,
    ) -> tuple[Figure, tuple[Axes, Axes]]:
        """Create a side-by-side PDF and CDF detail plot.

        Generates a two-panel figure using ``matplotlib.gridspec.GridSpec``:
        the left panel shows the fitted PDF curve overlaid on a histogram,
        and the right panel shows the fitted CDF curve alongside the
        empirical CDF scatter.  This provides a comprehensive view of
        distribution-fit quality.

        The *q_act* array is sorted internally for the CDF scatter; the
        original array is **not** mutated.

        Args:
            qx: X-values for both fitted curves.
            q_act: Observed data points.  Used as-is for the PDF histogram
                and sorted internally for the CDF scatter.
            pdf: PDF values corresponding to *qx*.
            cdf_fitted: CDF values corresponding to *qx*.
            cdf: Empirical CDF values for the sorted data.
            fig_size: Figure size ``(width, height)`` in inches.
                Defaults to ``(10, 5)``.
            xlabel: Label for both x-axes.  Defaults to ``"Actual data"``.
            ylabel: Label for the CDF y-axis.  Defaults to ``"cdf"``.
            fontsize: Font size for all labels.  Defaults to 11.

        Returns:
            tuple[Figure, tuple[Axes, Axes]]:
                The figure and a pair of axes ``(ax_pdf, ax_cdf)``.

        Examples:
            - Create a detail plot and inspect each subplot:
                ```python
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> np.random.seed(7)
                >>> data = np.random.normal(0, 1, 120)
                >>> x = np.linspace(-4, 4, 300)
                >>> pdf_v = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
                >>> cdf_v = 0.5 * (1 + np.tanh(0.798 * x))
                >>> cdf_emp = np.arange(1, 121) / 121
                >>> fig, (ax1, ax2) = DistributionPlot.details(x, data, pdf_v, cdf_v, cdf_emp)
                >>> ax1.get_ylabel()
                'pdf'
                >>> ax2.get_ylabel()
                'cdf'
                >>> len(fig.axes)
                2

                ```
            - Custom figure size and labels:
                ```python
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> data = np.random.exponential(3, 80)
                >>> x = np.linspace(0, 15, 200)
                >>> pdf_v = (1 / 3) * np.exp(-x / 3)
                >>> cdf_v = 1 - np.exp(-x / 3)
                >>> cdf_emp = np.arange(1, 81) / 81
                >>> fig, (ax1, ax2) = DistributionPlot.details(
                ...     x, data, pdf_v, cdf_v, cdf_emp,
                ...     fig_size=(12, 4), xlabel="Rainfall [mm]",
                ... )
                >>> round(fig.get_size_inches()[0])
                12
                >>> ax1.get_xlabel()
                'Rainfall [mm]'

                ```

        See Also:
            - DistributionPlot.pdf: For plotting only the PDF.
            - DistributionPlot.cdf: For plotting only the CDF.
        """
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(
            qx, pdf, "-",
            color=COLORS["fitted_curve"], linewidth=DEFAULT_LINEWIDTH,
        )
        ax1.hist(
            q_act, density=True, histtype="stepfilled",
            color=COLORS["histogram_fill"],
        )
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel("pdf", fontsize=fontsize)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(
            qx, cdf_fitted, "-",
            color=COLORS["fitted_curve"], linewidth=DEFAULT_LINEWIDTH,
        )
        q_act = np.sort(q_act)
        ax2.scatter(
            q_act, cdf,
            color=COLORS["histogram_fill"], facecolors="none",
        )
        ax2.set_xlabel(xlabel, fontsize=fontsize)
        ax2.set_ylabel(ylabel, fontsize=fontsize)
        return fig, (ax1, ax2)

    @staticmethod
    def confidence_level(
        qth: np.ndarray | list,
        q_act: np.ndarray | list,
        q_lower: np.ndarray | list,
        q_upper: np.ndarray | list,
        fig_size: tuple[float, float] = (6, 6),
        fontsize: int = DEFAULT_FONTSIZE,
        alpha: float = 0.05,
        marker_size: int = 10,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Create a confidence interval plot for distribution quantiles.

        Generates a Q-Q-style plot with four visual elements:

        1. A **1:1 reference line** (dash-dot) representing perfect
           agreement between theoretical and actual quantiles.
        2. **Lower CI bound** (star-dashed markers).
        3. **Upper CI bound** (star-dashed markers).
        4. **Actual data** scatter (hollow circles).

        Points falling along the 1:1 line indicate a good fit; points
        outside the confidence bands suggest the distribution may not
        adequately describe the data at those quantiles.

        The *q_act* array is sorted internally; the original is **not**
        mutated.

        Args:
            qth: Theoretical quantiles, typically obtained via
                ``inverse_cdf``.
            q_act: Observed data points.  Sorted internally before
                plotting.
            q_lower: Lower CI bound for each theoretical quantile.
            q_upper: Upper CI bound for each theoretical quantile.
            fig_size: Figure size ``(width, height)`` in inches.
                Defaults to ``(6, 6)``.
            fontsize: Font size for labels and legend.  Defaults to 11.
            alpha: Significance level for legend labels (e.g. 0.05
                produces ``"95 % CI"``).  The intervals themselves
                must be pre-computed.  Defaults to 0.05.
            marker_size: Size of CI-bound star markers.  Defaults to 10.
            fig: Pre-existing matplotlib Figure.
            ax: Pre-existing matplotlib Axes to draw on.

        Returns:
            tuple[Figure, Axes]: The figure and axes containing the plot.

        Examples:
            - Plot a confidence-interval band and inspect legend entries:
                ```python
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> np.random.seed(99)
                >>> qth = np.linspace(5, 15, 40)
                >>> q_act = np.random.normal(10, 2, 40)
                >>> q_lower = qth - 1.5
                >>> q_upper = qth + 1.5
                >>> fig, ax = DistributionPlot.confidence_level(qth, q_act, q_lower, q_upper)
                >>> labels = [t.get_text() for t in ax.get_legend().get_texts()]
                >>> labels[0]
                'Theoretical Data'
                >>> '95 % CI' in labels[1]
                True

                ```
            - Change alpha to 90 % CI and verify the label updates:
                ```python
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> qth = np.linspace(0, 20, 30)
                >>> q_act = np.sort(np.random.normal(10, 3, 30))
                >>> fig, ax = DistributionPlot.confidence_level(
                ...     qth, q_act, qth - 2, qth + 2, alpha=0.1,
                ... )
                >>> '90 % CI' in ax.get_legend().get_texts()[1].get_text()
                True
                >>> ax.get_xlabel()
                'Theoretical Values'

                ```
            - Compose into an existing axes:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import matplotlib.pyplot as plt
                >>> import numpy as np
                >>> from statista.distributions.plot import DistributionPlot
                >>> fig, ax = plt.subplots(figsize=(7, 7))
                >>> qth = np.linspace(1, 10, 20)
                >>> _, returned_ax = DistributionPlot.confidence_level(
                ...     qth, np.random.normal(5, 1, 20), qth - 1, qth + 1, ax=ax,
                ... )
                >>> returned_ax is ax
                True

                ```

        Notes:
            The 1:1 line represents perfect agreement between theoretical
            and actual values.  Points outside the confidence bands suggest
            potential issues with the distribution fit at those quantiles.

        See Also:
            - DistributionPlot.details: For plotting PDF and CDF together.
        """
        q_act = np.sort(q_act)
        ci_pct = int((1 - alpha) * 100)

        fig, ax = DistributionPlot._get_or_create_ax(fig, ax, fig_size)
        ax.plot(
            qth, qth, "-.",
            color=COLORS["reference_line"], linewidth=DEFAULT_LINEWIDTH,
            label="Theoretical Data",
        )
        ax.plot(
            qth, q_lower, "*--",
            color=COLORS["ci_bounds"], markersize=marker_size,
            label=f"Lower limit ({ci_pct} % CI)",
        )
        ax.plot(
            qth, q_upper, "*--",
            color=COLORS["ci_bounds"], markersize=marker_size,
            label=f"Upper limit ({ci_pct} % CI)",
        )
        ax.scatter(
            qth, q_act,
            color=COLORS["histogram_fill"], facecolors="none",
            label="Actual Data", zorder=10,
        )
        ax.legend(fontsize=fontsize, framealpha=1)
        ax.set_xlabel("Theoretical Values", fontsize=fontsize)
        ax.set_ylabel("Actual Values", fontsize=fontsize)
        return fig, ax
