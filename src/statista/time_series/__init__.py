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

# Default significance level for hypothesis tests
DEFAULT_ALPHA = 0.05  # Consider 0.01 for conservative analyses in safety-critical applications

from statista.time_series.base import BOX_MEAN_PROP, VIOLIN_PROP, TimeSeriesBase
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
    """A class to represent and analyze time series data using pandas DataFrame.

    Inherits from `pandas.DataFrame` and adds additional methods for statistical-analysis and visualization specific
    to time series data. See ``TimeSeriesBase`` for constructor documentation.
    """

    @property
    def _constructor(self):
        """Returns the constructor of the class."""
        return TimeSeries


__all__ = ["TimeSeries", "BOX_MEAN_PROP", "VIOLIN_PROP", "DEFAULT_ALPHA"]
