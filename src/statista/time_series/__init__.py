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

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
"""

from statista.time_series._base import BOX_MEAN_PROP, VIOLIN_PROP, TimeSeriesBase
from statista.time_series._changepoint import ChangePointMixin
from statista.time_series._correlation import CorrelationMixin
from statista.time_series._decomposition import DecompositionMixin
from statista.time_series._descriptive import DescriptiveMixin
from statista.time_series._distribution import DistributionMixin
from statista.time_series._hydrological import HydrologicalMixin
from statista.time_series._missing import MissingDataMixin
from statista.time_series._seasonal import SeasonalMixin
from statista.time_series._stationarity import StationarityMixin
from statista.time_series._trend import TrendMixin
from statista.time_series._visualization import VisualizationMixin


class TimeSeries(
    DescriptiveMixin,
    VisualizationMixin,
    MissingDataMixin,
    CorrelationMixin,
    StationarityMixin,
    TrendMixin,
    DistributionMixin,
    ChangePointMixin,
    DecompositionMixin,
    SeasonalMixin,
    HydrologicalMixin,
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


__all__ = ["TimeSeries", "BOX_MEAN_PROP", "VIOLIN_PROP"]
