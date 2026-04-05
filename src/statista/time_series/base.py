"""Base TimeSeries class — core DataFrame subclass with shared utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

BOX_MEAN_PROP = {"marker": "x", "markeredgecolor": "w", "markerfacecolor": "firebrick"}
VIOLIN_PROP = {"face": "#27408B", "edge": "black", "alpha": 0.7}


class TimeSeriesBase(DataFrame):
    """A class to represent and analyze time series data using pandas DataFrame.

    Inherits from `pandas.DataFrame` and adds additional methods for statistical-analysis and visualization specific
    to time series data.

    Args:
        data: array-like (1D or 2D)
            The data to be converted into a time series. If 2D, each column is treated as a separate series.
        index: array-like, optional
            The index for the time series data. If None, a default RangeIndex is used.
        name : str, optional
            The name of the column in the DataFrame. Default is 'TimeSeriesData'.
        *args : tuple
            Additional positional arguments to pass to the DataFrame constructor.
        **kwargs : dict
            Additional keyword arguments to pass to the DataFrame constructor.

    Examples:
        - Create a time series from a 1D array:
            ```python
            >>> data = np.random.randn(100)  # doctest: +SKIP
            >>> ts = TimeSeries(data)  # doctest: +SKIP
            >>> print(ts.stats)  # doctest: +SKIP
                      Series1
            count  100.000000
            mean     0.061816
            std      1.016592
            min     -2.622123
            25%     -0.539548
            50%     -0.010321
            75%      0.751756
            max      2.344767

            ```
        - Create a time series from a 2D array:
            ```python
            >>> data_2d = np.random.randn(100, 3)  # doctest: +SKIP
            >>> ts_2d = TimeSeries(data_2d, columns=['A', 'B', 'C'])  # doctest: +SKIP
            >>> print(ts_2d.stats)  # doctest: +SKIP
                      Series1     Series2     Series3
            count  100.000000  100.000000  100.000000
            mean     0.239437    0.058122   -0.063077
            std      1.002170    0.980495    1.000381
            min     -2.254215   -2.500011   -2.081786
            25%     -0.405632   -0.574242   -0.799128
            50%      0.308706    0.022795   -0.245399
            75%      0.879848    0.606253    0.607085
            max      2.628358    2.822292    2.538793

            ```
    """

    def __init__(
        self,
        data: DataFrame | list[float] | np.ndarray,
        index=None,
        columns=None,
        *args,
        **kwargs,
    ):

        # Normalize list input to numpy array
        if isinstance(data, list):
            data = np.array(data)

        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(-1, 1)  # Convert 1D array to 2D with one column

        if columns is None:
            if isinstance(data, dict):
                # the _constructor method overrides the original constructor of the dataframe and gives an error if the
                # data is a dictionary
                columns = list(data.keys())
            elif isinstance(data, DataFrame):
                # Preserve existing DataFrame column names
                columns = list(data.columns)
            else:
                columns = [
                    f"Series{i + 1}" for i in range(data.shape[1])  # type: ignore[union-attr]
                ]

        if not isinstance(data, DataFrame):
            # Convert input data to a pandas DataFrame
            data = DataFrame(data, index=index, columns=columns)

        super().__init__(data, *args, **kwargs)
        self.columns = columns

    @staticmethod
    def _get_ax_fig(n_subplots: int = 1, **kwargs: Any) -> tuple[Figure, Axes]:
        fig: Figure | None = kwargs.get("fig")
        ax: Axes | None = kwargs.get("ax")
        if ax is None and fig is None:
            fig, ax = plt.subplots(n_subplots, figsize=(8, 6))
        elif ax is None and fig is not None:
            ax = fig.add_subplot(111)
        elif fig is None and ax is not None:
            fig = ax.figure  # type: ignore[assignment]
        return fig, ax  # type: ignore[return-value]

    @staticmethod
    def _adjust_axes_labels(
        ax: Axes, tick_labels: list[str] | None = None, **kwargs: Any
    ) -> Axes:
        """Adjust the labels of the axes."""
        if tick_labels is not None:
            ax.set_xticklabels(tick_labels)

        ax.set_title(
            kwargs.get("title", ""),
            fontsize=kwargs.get("title_fontsize", 18),
            fontweight="bold",
        )
        ax.set_xlabel(
            kwargs.get("xlabel", ""),
            fontsize=kwargs.get("xlabel_fontsize", 14),
        )
        ax.set_ylabel(
            kwargs.get("ylabel", ""),
            fontsize=kwargs.get("ylabel_fontsize", 14),
        )

        ax.grid(
            kwargs.get("grid", True),
            axis=kwargs.get("grid_axis", "both"),
            linestyle=kwargs.get("grid_line_style", "-."),
            linewidth=kwargs.get("grid_line_width", 0.3),
        )

        # Customize ticks and their labels
        ax.tick_params(
            axis="both", which="major", labelsize=kwargs.get("tick_fontsize", 12)
        )

        # Add a legend if needed
        if "legend" in kwargs:
            labels: list[str] = kwargs["legend"]
            ax.legend(labels, fontsize=kwargs.get("legend_fontsize", 12))

        # Adjust layout for better spacing
        plt.tight_layout()

        return ax
