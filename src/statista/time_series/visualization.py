"""Visualization mixin for TimeSeries."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from pandas import DataFrame

from statista.time_series.base import BOX_MEAN_PROP, VIOLIN_PROP

if TYPE_CHECKING:
    from pandas import Index


class Visualization:
    """Visualization methods for TimeSeries.

    This mixin is designed to be composed with ``TimeSeriesBase`` (a ``pandas.DataFrame`` subclass).
    All attribute access (``self.columns``, ``self.values``, indexing) is provided by DataFrame
    at runtime.
    """

    if TYPE_CHECKING:
        columns: Index
        values: np.ndarray
        index: Index
        ndim: int

        @staticmethod
        def _get_ax_fig(  # noqa: E704
            n_subplots: int = 1, **kwargs: object
        ) -> Tuple[Figure, Axes]: ...

        @staticmethod
        def _adjust_axes_labels(  # noqa: E704
            ax: Axes, tick_labels: List[str] | None = None, **kwargs: object
        ) -> Axes: ...

        def __getitem__(self, key: str) -> DataFrame:  # noqa: E704
            ...

    def box_plot(
        self, mean: bool = False, notch: bool = False, **kwargs
    ) -> Tuple[Figure, Axes]:
        """a box and whisker plot.

        The box extends from the first quartile (Q1) to the third quartile (Q3) of the data, with a line at the median.
        The whiskers extend from the box to the farthest data point lying within 1.5x the inter-quartile range (IQR)
        from the box.
        Flier points are those past the end of the whiskers. See https://en.wikipedia.org/wiki/Box_plot for reference.

        The box plot can give the following insights:
            - Summary of Distribution: A box plot provides a graphical summary of the distribution of data based on five
                summary statistics: the minimum, first quartile (Q1), median, third quartile (Q3), and maximum.
            - Outliers: It highlights outliers, which are data points that fall significantly above or below the rest of
                the data. Outliers are typically shown as individual points beyond the "whiskers" of the box plot.
            - Central Tendency: The line inside the box indicates the median (50th percentile), giving insight into the
                central tendency of the data.
            - Spread and Skewness: The length of the box (interquartile range, IQR) shows the spread of the middle 50% of
                the data, while the position of the median line within the box can suggest skewness.

                      Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
                                   |-----:-----|
                   o      |--------|     :     |--------|    o  o
                                   |-----:-----|
                 flier             <----------->            fliers
                                        IQR

        Use Case:
            - Useful for quickly comparing the distribution of the time series data and identifying any anomalies or
                outliers.

        Args:
            mean: bool, optional, default is False.
                Whether to show the mean value in the box plot.
            notch: bool, optional, default is False.
                    Whether to draw a notched boxplot (`True`), or a rectangular
                    boxplot (`False`).  The notches represent the confidence interval
                    (CI) around the median.  The documentation for *bootstrap*
                    describes how the locations of the notches are computed by
                    default, but their locations may also be overridden by setting the
                    *conf_intervals* parameter.
            **kwargs: dict, optional
                fig: matplotlib.figure.Figure, optional
                    Existing figure to plot on. If None, a new figure is created.
                ax: matplotlib.axes.Axes, optional
                    Existing axes to plot on. If None, a new axes is created.
                grid: bool, optional, Default is False.
                    Whether to show grid lines.
                color: dict, optional, default is None.
                    Colors to use for the plot elements. Default is None.
                    ```color = {"boxes", "#27408B"}```
                title: str, optional
                    Title of the plot.
                xlabel: str, optional
                    Label for the x-axis.
                ylabel: str, optional
                    Label for the y-axis.

        Returns:
            fig: matplotlib.figure.Figure
                The figure object containing the plot.
            ax: matplotlib.axes.Axes
                The axes object containing the plot.

        Examples:
            - Plot the box plot for a 1D time series:
                ```python
                >>> ts = TimeSeries(np.random.randn(100))  # doctest: +SKIP
                >>> fig, ax = ts.box_plot()  # doctest: +SKIP

                ```
                ![box_plot_1d](./../_images/time_series/box_plot_1d.png)

            - Plot the box plot for a multiple time series:
                ```python
                >>> data_2d = np.random.randn(100, 4)  # doctest: +SKIP
                >>> ts_2d = TimeSeries(data_2d, columns=['A', 'B', 'C', 'D'])  # doctest: +SKIP
                >>> fig, ax = ts_2d.box_plot(mean=True, grid=True)  # doctest: +SKIP

                ```
                ![box_plot_2d](./../_images/time_series/box_plot_2d.png)
                ```python
                >>> fig, ax = ts_2d.box_plot(grid=True, mean=True, color={"boxes": "#DC143C"})  # doctest: +SKIP

                ```
                ![box_plot_color](./../_images/time_series/box_plot_color.png)
                ```python
                >>> fig, ax = ts_2d.box_plot(xlabel='Custom X', ylabel='Custom Y', title='Custom Box Plot')  # doctest: +SKIP

                ```
                ![box_plot_axes-label](./../_images/time_series/box_plot_axes-label.png)
                ```python
                >>> fig, ax = ts_2d.box_plot(notch=True)  # doctest: +SKIP

                ```
                ![box_plot_notch](./../_images/time_series/box_plot_notch.png)
        """
        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)
        color = kwargs.get("color", None)
        data = [self[col].dropna() for col in self.columns]
        ax.boxplot(
            data,
            notch=notch,
            patch_artist=True,
            showmeans=mean,
            meanprops=BOX_MEAN_PROP,
            boxprops={
                "facecolor": (
                    "#27408B" if color is None else color.get("boxes", "#27408B")
                )
            },
        )
        ax = self._adjust_axes_labels(
            ax,
            self.columns,
            **kwargs,
        )

        plt.show()
        return fig, ax

    @staticmethod
    def calculate_whiskers(data: Union[np.ndarray, list], q1: float, q3: float):
        """Calculate the upper and lower whiskers for a box plot.

        Args:
            data: np.ndarray
                Input array of data.
            q1: float
                first quartile
            q3: float
                third quartile

        Returns:
            lower_wisker: float
                Lower whisker value.
            upper_wisker: float
                Upper whisker value.
        """
        inter_quartile = q3 - q1
        upper_whisker = q3 + inter_quartile * 1.5
        upper_whisker = np.clip(upper_whisker, q3, data[-1])

        lower_whisker = q1 - inter_quartile * 1.5
        lower_whisker = np.clip(lower_whisker, data[0], q1)
        return lower_whisker, upper_whisker

    def violin(
        self,
        mean: bool = True,
        median: bool = False,
        extrema: bool = True,
        side: Literal["both", "low", "high"] = "both",
        spacing: int = 0,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plots a violin plot of the time series data.

        Args:
            mean: bool, optional, default is True.
                Whether to show the means in the violin plot.
            median: bool, optional, default is False.
                Whether to show the median in the violin plot.
            extrema: bool, optional, default is False.
                Whether to show the minima and maxima in the violin plot.
            side: {'both', 'low', 'high'}, default: 'both'
                'both' plots standard violins. 'low'/'high' only
                plots the side below/above the position value.
            spacing: int, optional, default is 0.
                The spacing (number of ticks) between the violins.
            **kwargs: dict, optional
                fig: matplotlib.figure.Figure, optional
                    Existing figure to plot on. If None, a new figure is created.
                ax: matplotlib.axes.Axes, optional
                    Existing axes to plot on. If None, a new axes is created.
                grid: bool, optional
                    Whether to show grid lines. Default is True.
                color: dict, optional, default is None.
                    Colors to use for the plot elements. Default is None.
                    ```color = {"face", "#27408B", "edge", "#DC143C", "alpha", 0.7}```
                title: str, optional
                    Title of the plot. Default is 'Box Plot'.
                xlabel: str, optional
                    Label for the x-axis. Default is 'Index'.
                ylabel: str, optional
                    Label for the y-axis. Default is 'Value'.

        Returns:
            fig: matplotlib.figure.Figure
                The figure object containing the plot.
            ax: matplotlib.axes.Axes
                The axes object containing the plot.

        Examples:
            - Plot the box plot for a 1D time series:
                ```python
                >>> ts = TimeSeries(np.random.randn(100))  # doctest: +SKIP
                >>> fig, ax = ts.violin()  # doctest: +SKIP

                ```
                ![violin_1d](./../_images/time_series/violin_1d.png)

            - Plot the box plot for a multiple time series:
                ```python
                >>> data_2d = np.random.randn(100, 4)  # doctest: +SKIP
                >>> ts_2d = TimeSeries(data_2d, columns=['A', 'B', 'C', 'D'])  # doctest: +SKIP
                >>> fig, ax = ts_2d.violin()  # doctest: +SKIP

                ```
                ![violin_2d](./../_images/time_series/violin_2d.png)

            - you can control the spacing between the violins using the `spacing` parameter:
                ```python
                >>> fig, ax = ts_2d.violin(spacing=2)  # doctest: +SKIP

                ```
                ![violin_2d_spacing](./../_images/time_series/violin_2d_spacing.png)

            - You can change the title, xlabel, and ylabel using the respective parameters:
                ```python
                >>> fig, ax = ts_2d.violin(xlabel='Random Data', ylabel='Custom Y', title='Custom Box Plot')  # doctest: +SKIP

                ```
                ![violin_labels_titles](./../_images/time_series/violin_labels_titles.png)

            - You can display the means, medians, and extrema using the respective parameters:
                ```python
                >>> fig, ax = ts_2d.violin(mean=True, median=True, extrema=True)  # doctest: +SKIP

                ```
                ![violin_means_medians_extrema](./../_images/time_series/violin_means_medians_extrema.png)

            - You can display the violins on the low side only using the `side` parameter:
                ```python
                >>> fig, ax = ts_2d.violin(side='low')  # doctest: +SKIP

                ```
                ![violin_low_side](./../_images/time_series/violin_low_side.png)
        """
        fig, ax = self._get_ax_fig(**kwargs)
        # kwargs.pop("fig", None)

        # positions where violins are plotted (1, 3, 5, ...)ing labels
        positions = np.arange(1, len(self.columns) * (spacing + 1) + 1, spacing + 1)

        violin_data = [self[col].dropna().values for col in self.columns]
        violin_parts = ax.violinplot(
            violin_data,
            showmeans=mean,
            showmedians=median,
            showextrema=extrema,
            side=side,
            positions=positions,
        )
        color = kwargs.get("color") if "color" in kwargs else VIOLIN_PROP

        for pc in violin_parts["bodies"]:  # type: ignore[attr-defined]
            pc.set_facecolor(color.get("face"))  # type: ignore[union-attr]
            pc.set_edgecolor(color.get("edge"))  # type: ignore[union-attr]
            pc.set_alpha(color.get("alpha"))  # type: ignore[union-attr]

        ax.xaxis.set_ticks(positions)
        # remove the ax from the kwargs to avoid passing it to the adjust_axes_labels method twice
        kwargs.pop("ax", None)
        ax = self._adjust_axes_labels(
            ax,
            self.columns,
            **kwargs,
        )

        plt.show()
        return fig, ax

    def raincloud(
        self,
        overlay: bool = True,
        violin_width: float = 0.4,
        scatter_offset: float = 0.15,
        boxplot_width: float = 0.1,
        order: List[str] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """RainCloud plot.

        Args:
            overlay: bool, optional, default is True.
                Whether to overlay the plots or display them side-by-side.
            violin_width: float, optional, default is 0.4.
                Width of the violins.
            scatter_offset: float, optional, default is 0.15.
                Offset for the scatter plot.
            boxplot_width: float, optional, default is
                Width of the box plot.
            order: list, optional, default is None.
                Order of the plots. Default is ['violin', 'scatter', 'box'].
            **kwargs: dict, optional
                fig: matplotlib.figure.Figure, optional
                    Existing figure to plot on. If None, a new figure is created.
                ax: matplotlib.axes.Axes, optional
                    Existing axes to plot on. If None, a new axes is created.
                grid: bool, optional
                    Whether to show grid lines. Default is True.
                color: dict, optional, default is None.
                    Colors to use for the plot elements. Default is None.
                    ```color = {"boxes", "#27408B"}```
                title: str, optional
                    Title of the plot. Default is 'Box Plot'.
                xlabel: str, optional
                    Label for the x-axis. Default is 'Index'.
                ylabel: str, optional
                    Label for the y-axis. Default is 'Value'.

        Returns:
            fig: matplotlib.figure.Figure
                The figure object containing the plot.
            ax: matplotlib.axes.Axes
                The axes object containing the plot.

        Examples:
            - Plot the raincloud plot for a 1D time series, and use the `overlay` parameter to overlay the plots:
                ```python
                >>> ts = TimeSeries(np.random.randn(100))  # doctest: +SKIP
                >>> fig, ax = ts.raincloud()  # doctest: +SKIP

                ```
                ![raincloud_1d](./../_images/time_series/raincloud_1d.png)

                ```python
                >>> fig, ax = ts.raincloud(overlay=False)  # doctest: +SKIP

                ```
                ![raincloud-overlay-false](./../_images/time_series/raincloud-overlay-false.png)

            - Plot the box plot for a multiple time series:
                ```python
                >>> data_2d = np.random.randn(100, 4)  # doctest: +SKIP
                >>> ts_2d = TimeSeries(data_2d, columns=['A', 'B', 'C', 'D'])  # doctest: +SKIP
                >>> fig, ax = ts_2d.raincloud(mean=True, grid=True)  # doctest: +SKIP

                ```
        """
        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)
        if order is None:
            order = ["violin", "scatter", "box"]

        n_groups = len(self.columns)
        positions = np.arange(1, n_groups + 1)

        # Dictionary to map plot types to the functions
        plot_funcs = {
            "violin": lambda pos, d: ax.violinplot(
                [d],
                positions=[pos],
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=violin_width,
            ),
            "scatter": lambda pos, d: ax.scatter(
                np.random.normal(pos, 0.04, size=len(d)),
                d,
                alpha=0.6,
                color="black",
                s=10,
                edgecolor="white",
                linewidth=0.5,
            ),
            "box": lambda pos, d: ax.boxplot(
                [d],
                positions=[pos],
                widths=boxplot_width,
                vert=True,
                patch_artist=True,
                boxprops={"facecolor": "lightblue", "color": "blue"},
                medianprops={"color": "red"},
            ),
        }

        # Plot elements according to the specified order and selected plots
        # for i, d in enumerate(data):
        for i in range(len(self.columns)):
            if self.ndim == 1:
                d = self.values
            else:
                d = self.values[:, i]
            base_pos = positions[i]
            if overlay:
                for plot_type in order:
                    plot_funcs[plot_type](base_pos, d)
            else:
                for j, plot_type in enumerate(order):
                    offset = (j - 1) * scatter_offset
                    plot_funcs[plot_type](base_pos + offset, d)

        # Customize the appearance of violins if they are included
        if "violin" in order:
            for (
                pc
            ) in (
                ax.collections
            ):  # all polygons created by violinplot are in ax.collections
                if isinstance(pc, PolyCollection):
                    pc.set_facecolor("skyblue")
                    pc.set_edgecolor("blue")
                    pc.set_alpha(0.3)
                    pc.set_linewidth(1)
                    pc.set_linestyle("-")

        # Set x-tick labels
        ax.set_xticks(positions)
        ax = self._adjust_axes_labels(
            ax,
            self.columns,
            **kwargs,
        )

        # Add grid lines for better readability
        # ax.yaxis.grid(True)

        # Display the plot
        plt.show()
        return fig, ax

    def histogram(
        self, bins=10, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Figure, Axes]:
        """
        Plots a histogram of the time series data.

        Args:
            bins : int, optional, default is 10.
                Number of histogram bins.
            **kwargs: dict, optional
                fig: matplotlib.figure.Figure, optional
                    Existing figure to plot on. If None, a new figure is created.
                ax: matplotlib.axes.Axes, optional
                    Existing axes to plot on. If None, a new axes is created.
                grid: bool, optional
                    Whether to show grid lines. Default is True.
                color: str, optional, default is None.
                    Colors to use for the plot elements.
                title: str, optional
                    Title of the plot. Default is 'Box Plot'.
                xlabel: str, optional
                    Label for the x-axis. Default is 'Index'.
                ylabel: str, optional
                    Label for the y-axis. Default is 'Value'.
                title_fontsize: int, optional
                    Font size of the title.
                label_fontsize: int, optional
                    Font size of the title and labels.
                tick_fontsize: int, optional
                    Font size of the tick labels.
                xtick_labels: List[str], optional
                    Labels for the x-axis ticks.
                legend: List[str], optional
                    Legend to display in the plot.
                legend_fontsize: int, optional
                    Font size of the legend.

        Returns:
            fig : matplotlib.figure.Figure
                The figure object containing the plot.
            ax : matplotlib.axes.Axes
                The axes object containing the plot.
            n_values : np.ndarray
                The number of values in each histogram bin.
            bin_edges : np.ndarray
                The edges of the bins. Length nbins + 1 (nbins left edges and right
                edge of last bin).  Always a single array even when multiple data
                sets are passed in.

        Examples:
            - Plot the box plot for a 1D time series:
                ```python
                >>> ts = TimeSeries(np.random.randn(100))  # doctest: +SKIP
                >>> n_values, bin_edges, fig, ax = ts.histogram()  # doctest: +SKIP
                >>> print(n_values) #doctest: +SKIP
                [ 5.  8. 11. 12. 14. 17. 15.  9.  4.  5.]
                >>> print(bin_edges) #doctest: +SKIP
                [-2.41934673 -1.9628219  -1.50629707 -1.04977224 -0.5932474  -0.13672257
                  0.31980226  0.77632709  1.23285192  1.68937676  2.14590159]
                ```
                ![histogram](./../_images/time_series/histogram.png)

            - Plot the box plot for a multiple time series:
                ```python
                >>> data_2d = np.random.randn(100, 4)  # doctest: +SKIP
                >>> ts_2d = TimeSeries(data_2d, columns=['A', 'B', 'C', 'D'])  # doctest: +SKIP
                >>> n_values, bin_edges, fig, ax = ts_2d.histogram(legend=['A', 'B', 'C', 'D'])  # doctest: +SKIP
                >>> print(n_values) #doctest: +SKIP
                [[ 0.  7.  9. 12. 20. 20. 19.  7.  5.  1.]
                 [ 1.  1.  9. 12. 20. 25. 13. 14.  5.  0.]
                 [ 5.  4. 11. 10. 18. 23. 13.  9.  4.  3.]
                 [ 1.  2. 11. 18. 16. 20. 13. 11.  6.  2.]]
                >>> print(bin_edges) #doctest: +SKIP
                [-2.76976813 -2.22944508 -1.68912202 -1.14879896 -0.6084759  -0.06815285
                  0.47217021  1.01249327  1.55281633  2.09313939  2.63346244]
                ```
                ![histogram-2d](./../_images/time_series/histogram-2d.png)
        """
        # plt.style.use('ggplot')

        fig, ax = self._get_ax_fig(**kwargs)

        color = kwargs.get("color") if "color" in kwargs else VIOLIN_PROP
        if len(self.columns) > 1 and not isinstance(color.get("face"), list):  # type: ignore[union-attr]
            color = None
            warnings.warn(
                "Multiple columns detected. Please provide a list of colors for each column, Otherwise the given"
                "color will be ignored."
            )
        n_values, bin_edges, _ = ax.hist(
            self.values,
            bins=bins,
            color=color.get("face") if color else None,  # type: ignore[union-attr,arg-type]
            edgecolor=color.get("edge") if color else None,  # type: ignore[union-attr]
            alpha=color.get("alpha") if color else None,  # type: ignore[union-attr]
        )

        kwargs.pop("ax", None)
        kwargs["legend"] = (
            kwargs.get("legend")
            if kwargs.get("legend") is not None
            else self.columns.to_list()
        )

        ax = self._adjust_axes_labels(
            ax,
            kwargs.get("xtick_labels"),
            **kwargs,
        )

        plt.show()
        return n_values, bin_edges, fig, ax  # type: ignore[return-value]

    def density(self, **kwargs) -> Tuple[Figure, Axes]:
        """
        Plots a density (KDE) plot of the time series data.

        - KDE is a non-parametric method for estimating the probability density function of a random variable.
        - It provides a smoothed estimate of the underlying probability distribution based on observed data points
        - This function uses Gaussian kernels and includes automatic bandwidth determination

        Args:
            **kwargs: dict, optional
                color (str, optional):
                    Color of the density line. Default is 'blue'.
                fig (matplotlib.figure.Figure, optional):
                    Existing figure to plot on. If None, a new figure is created.
                ax (matplotlib.axes.Axes, optional):
                    Existing axes to plot on. If None, a new axes is created.
                grid (bool, optional):
                    Whether to show grid lines. Default is False.
                color (dict, optional):
                    Colors to use for the plot elements. Default is None.
                    ```color = {"boxes", "#27408B"}```
                title (str, optional):
                    Title of the plot.
                xlabel (str, optional):
                    Label for the x-axis.
                ylabel (str, optional):
                    Label for the y-axis.

        Returns:
            fig (matplotlib.figure.Figure):
                The figure object containing the plot.
            ax (matplotlib.axes.Axes):
                The axes object containing the plot.

        Examples:
            - Plot the KDE density plot for a 1D time series:
                ```python
                >>> ts = TimeSeries(np.random.randn(100))  # doctest: +SKIP
                >>> fig, ax = ts.density(title='Density Plot', xlabel='Random Values', ylabel='KDE density')  # doctest: +SKIP

                ```
                ![density-1d](./../_images/time_series/density-1d.png)

            - Plot the KDE density plot for a 2D time series:
                ```python
                >>> ts = TimeSeries(np.random.randn(100, 4))  # doctest: +SKIP
                >>> fig, ax = ts.density(title='Density Plot', xlabel='Random Values', ylabel='KDE density')  # doctest: +SKIP

                ```
                ![density-2d](./../_images/time_series/density-2d.png)
        """
        fig, ax = self._get_ax_fig(**kwargs)
        color = kwargs.get("color", None)
        self[self.columns.to_list()].plot(kind="density", ax=ax, color=color)
        kwargs.pop("ax", None)
        ax = self._adjust_axes_labels(
            ax,
            kwargs.get("xtick_labels"),
            **kwargs,
        )

        plt.show()

        return fig, ax

    def rolling_statistics(self, window=10, **kwargs) -> Tuple[Figure, Axes]:
        """
        Plots the rolling mean and standard deviation of the time series data.

        Args:
            window : int, optional, default is 10.
                The window size for the rolling statistics.
            **kwargs: dict, optional
                fig: matplotlib.figure.Figure, optional
                    Existing figure to plot on. If None, a new figure is created.
                ax: matplotlib.axes.Axes, optional
                    Existing axes to plot on. If None, a new axes is created.
                grid: bool, optional
                    Whether to show grid lines. Default is True.
                color: str, optional, default is None.
                    Colors to use for the plot elements.
                title: str, optional
                    Title of the plot. Default is 'Rolling Mean & Standard Deviation'.
                xlabel: str, optional
                    Label for the x-axis. Default is 'Index'.
                ylabel: str, optional
                    Label for the y-axis. Default is 'Value'.
                title_fontsize: int, optional
                    Font size of the title.
                label_fontsize: int, optional
                    Font size of the title and labels.
                tick_fontsize: int, optional
                    Font size of the tick labels.
                xtick_labels: List[str], optional
                    Labels for the x-axis ticks.
                legend: List[str], optional
                    Legend to display in the plot.
                legend_fontsize: int, optional
                    Font size of the legend.

        Returns:
            fig : matplotlib.figure.Figure
                The figure object containing the plot.
            ax : matplotlib.axes.Axes
                The axes object containing the plot.

        Examples:
            - Plot the rolling average and standard deviation for a 1D time series:
                ```python
                >>> ts = TimeSeries(np.random.randn(100))  # doctest: +SKIP
                >>> fig, ax = ts.rolling_statistics(  # doctest: +SKIP
                ...    window=20, title='Rolling Statistics', xlabel='Random Values', ylabel='Random Y',
                ...    legend=['Rolling Mean', 'Rolling Std']
                ... )

                ```
                ![rolling-statistics](./../_images/time_series/rolling-statistics.png)

            - Plot the rolling average and standard deviation for a 2D time series:
                ```python
                >>> ts = TimeSeries(np.random.randn(100, 3))  # doctest: +SKIP
                >>> fig, ax = ts.rolling_statistics(  # doctest: +SKIP
                ...    window=10, title='Rolling Statistics', xlabel='Random Values', ylabel='Random Y',
                ... )

                ```
                ![rolling-statistics-2d](./../_images/time_series/rolling-statistics-2d.png)
        """
        fig, ax = self._get_ax_fig(**kwargs)

        rolling_mean = self[self.columns].rolling(window=window).mean()
        rolling_std = self[self.columns].rolling(window=window).std()

        ax.plot(self.index, rolling_mean, label="Rolling Mean", color="blue")
        ax.plot(self.index, rolling_std, label="Rolling Std", color="red")
        kwargs.pop("ax", None)
        ax = self._adjust_axes_labels(
            ax,
            kwargs.get("xtick_labels"),
            **kwargs,
        )
        plt.show()
        return fig, ax
