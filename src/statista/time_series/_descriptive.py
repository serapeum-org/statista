"""Descriptive statistics mixin for TimeSeries."""

import numpy as np
from pandas import DataFrame
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import median_abs_deviation, skew


class DescriptiveMixin:
    """Mixin providing descriptive statistical methods for TimeSeries."""

    @property
    def stats(self) -> DataFrame:
        """
        Returns a detailed statistical summary of the time series.

        Returns:
            pandas.DataFrame
                Statistical summary including count, mean, std, min, 25%, 50%, 75%, max.

        Examples:
            ```python
            >>> ts = TimeSeries(np.random.randn(100))
            >>> ts.stats # doctest: +SKIP
                      Series1
            count  100.000000
            mean     0.130961
            std      0.912412
            min     -2.850323
            25%     -0.442838
            50%      0.157995
            75%      0.787275
            max      1.972673
            ```
        """
        return self.describe()

    @property
    def extended_stats(self) -> DataFrame:
        """Comprehensive statistical summary of the time series.

        Extends the basic ``stats`` property with additional measures commonly needed in research: coefficient of
        variation, skewness, kurtosis, interquartile range, and median absolute deviation.

        Returns:
            pandas.DataFrame
                Statistical summary with one column per series. Rows: count, mean, std, cv, skewness, kurtosis, min,
                5%, 10%, 25%, 50%, 75%, 90%, 95%, max, iqr, mad.

        Examples:
            - Compute extended statistics for a 1D time series:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> np.random.seed(42)
                >>> ts = TimeSeries(np.random.randn(200))
                >>> result = ts.extended_stats
                >>> print(result.index.tolist())
                ['count', 'mean', 'std', 'cv', 'skewness', 'kurtosis', 'min', '5%', '10%', '25%', '50%', '75%', '90%', '95%', 'max', 'iqr', 'mad']

                ```
            - Compute extended statistics for a 2D time series:
                ```python
                >>> ts_2d = TimeSeries(np.random.randn(200, 3), columns=['A', 'B', 'C'])
                >>> result = ts_2d.extended_stats
                >>> result.columns.tolist()
                ['A', 'B', 'C']

                ```

        See Also:
            - stats: Basic descriptive statistics (count, mean, std, min, quartiles, max).
        """
        stat_names = [
            "count", "mean", "std", "cv", "skewness", "kurtosis",
            "min", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "max",
            "iqr", "mad",
        ]
        result = DataFrame(index=stat_names, columns=self.columns, dtype=float)

        for col in self.columns:
            data = self[col].dropna().values
            n = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            quantiles = np.percentile(data, [5, 10, 25, 50, 75, 90, 95])
            q25, q75 = quantiles[2], quantiles[4]

            result.loc["count", col] = n
            result.loc["mean", col] = mean
            result.loc["std", col] = std
            result.loc["cv", col] = std / mean if not np.isclose(mean, 0.0) else np.nan
            result.loc["skewness", col] = skew(data, bias=False)
            result.loc["kurtosis", col] = scipy_kurtosis(data, bias=False)
            result.loc["min", col] = np.min(data)
            result.loc["5%", col] = quantiles[0]
            result.loc["10%", col] = quantiles[1]
            result.loc["25%", col] = q25
            result.loc["50%", col] = quantiles[3]
            result.loc["75%", col] = q75
            result.loc["90%", col] = quantiles[5]
            result.loc["95%", col] = quantiles[6]
            result.loc["max", col] = np.max(data)
            result.loc["iqr", col] = q75 - q25
            result.loc["mad", col] = median_abs_deviation(data)

        return result
