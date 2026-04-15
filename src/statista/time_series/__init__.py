"""
Time Series Analysis and Visualization
======================================
This module provides a class to represent and analyze time series data using pandas DataFrame. It inherits from
`pandas.DataFrame` and adds additional methods for statistical-analysis and visualization specific to time series data.

Time Series Analysis
--------------------
- `stats`: Returns a detailed statistical summary of the time series.
- `extended_stats`: Returns a comprehensive statistical summary with CV, skewness, kurtosis, IQR, MAD.
- `box_plot`: Plots a box plot of the time series data.
- `violin`: Plots a violin plot of the time series data.
- `raincloud`: Plots a raincloud plot of the time series data.
- `histogram`: Plots a histogram of the time series data.

Statistical Testing
-------------------
Most hypothesis testing methods (e.g., ``mann_kendall``, ``adf_test``, ``pettitt_test``) default to
``alpha=0.05`` (5% significance level). For safety-critical or high-consequence applications, consider using
more conservative thresholds like ``alpha=0.01`` or ``alpha=0.10`` depending on your domain requirements.

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
"""

from statista.time_series.base import BOX_MEAN_PROP, VIOLIN_PROP, TimeSeriesBase
from statista.time_series.constants import DEFAULT_ALPHA
from statista.time_series.changepoint import ChangePoint
from statista.time_series.comparison import Comparison
from statista.time_series.correlation import Correlation
from statista.time_series.decomposition import Decomposition
from statista.time_series.descriptive import Descriptive
from statista.time_series.distribution import Distribution
from statista.time_series.hydrological import Hydrological
from statista.time_series.missing import MissingData
from statista.time_series.seasonal import Seasonal
from statista.time_series.stationarity import Stationarity
from statista.time_series.trend import Trend
from statista.time_series.visualization import Visualization


class TimeSeries(
    Descriptive,
    Visualization,
    MissingData,
    Correlation,
    Stationarity,
    Trend,
    Distribution,
    ChangePoint,
    Decomposition,
    Seasonal,
    Hydrological,
    Comparison,
    TimeSeriesBase,
):
    """A pandas DataFrame subclass with 53 statistical analysis methods.

    ``TimeSeries`` extends ``pandas.DataFrame`` with methods for descriptive
    statistics, autocorrelation, stationarity testing, trend detection, change
    point analysis, distribution fitting, decomposition, seasonal analysis,
    hydrological signatures, and comparison/anomaly detection.

    Accepts 1D arrays, 2D arrays, lists, dicts, or DataFrames as input.
    Column names are auto-generated as Series1, Series2, ... unless provided.

    Args:
        data: Input data — numpy array (1D or 2D), list, dict, or DataFrame.
        index: Index for the time series. If None, uses default RangeIndex.
        columns: Column names. If None, auto-generated or preserved from DataFrame.

    Examples:
        - Create from a 1D array and inspect statistics:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> data = np.loadtxt("examples/data/time_series1.txt")
            >>> ts = TimeSeries(data)
            >>> ts.shape
            (27, 1)
            >>> round(float(ts.extended_stats.loc["mean", "Series1"]), 2)
            16.93

            ```
        - Create from a 2D array with named columns:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(
            ...     np.column_stack([np.random.randn(100), np.random.randn(100) * 2]),
            ...     columns=["Flow", "Temp"],
            ... )
            >>> ts.columns.tolist()
            ['Flow', 'Temp']
            >>> ts.shape
            (100, 2)

            ```
        - Create from a Python list:
            ```python
            >>> from statista.time_series import TimeSeries
            >>> ts = TimeSeries([10.0, 20.0, 30.0, 40.0, 50.0])
            >>> ts.shape
            (5, 1)
            >>> float(ts.extended_stats.loc["mean", "Series1"])
            30.0

            ```
    """

    @property
    def _constructor(self):
        """Returns the constructor of the class."""
        return TimeSeries


__all__ = ["TimeSeries", "BOX_MEAN_PROP", "VIOLIN_PROP", "DEFAULT_ALPHA"]
