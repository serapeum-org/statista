"""Descriptive statistics for TimeSeries.

This module provides the ``Descriptive`` class which adds statistical summary
properties and methods to ``TimeSeries``. It includes conventional moments
(mean, std, skewness, kurtosis), robust measures (MAD, IQR), and L-moments
for distribution identification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import median_abs_deviation, skew

from statista.parameters import Lmoments

if TYPE_CHECKING:
    from statista.time_series.stubs import _TimeSeriesStub
else:
    _TimeSeriesStub = object


class Descriptive(_TimeSeriesStub):
    """Descriptive statistical methods for TimeSeries."""

    @property
    def stats(self) -> DataFrame:
        """Basic statistical summary of the time series.

        Delegates to ``pandas.DataFrame.describe()``, returning count, mean,
        standard deviation, min, quartiles, and max for each column.

        Returns:
            pandas.DataFrame:
                One column per series. Rows: count, mean, std, min, 25%, 50%, 75%, max.

        Examples:
            - Load real hydrological data and inspect basic statistics:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> data = np.loadtxt("examples/data/time_series1.txt")
                >>> ts = TimeSeries(data)
                >>> s = ts.stats
                >>> s.index.tolist()
                ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                >>> round(float(s.loc["mean", "Series1"]), 2)
                16.93

                ```
            - Multi-column data returns one column per series:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> np.random.seed(42)
                >>> ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
                >>> s = ts.stats
                >>> s.columns.tolist()
                ['A', 'B']
                >>> int(s.loc["count", "A"])
                100

                ```

        See Also:
            extended_stats: Adds CV, skewness, kurtosis, IQR, MAD, and extra percentiles.
            summary: Paper-ready table combining extended_stats and L-moment ratios.
        """
        return self.describe()

    @property
    def extended_stats(self) -> DataFrame:
        """Comprehensive statistical summary with 17 measures per column.

        Extends ``stats`` with coefficient of variation (CV), skewness (bias-corrected),
        excess kurtosis (bias-corrected), additional percentiles (5%, 10%, 90%, 95%),
        interquartile range (IQR), and median absolute deviation (MAD). NaN values are
        dropped per column before computation.

        Returns:
            pandas.DataFrame:
                One column per series. Rows: count, mean, std, cv, skewness, kurtosis,
                min, 5%, 10%, 25%, 50%, 75%, 90%, 95%, max, iqr, mad.

        Examples:
            - Compute extended statistics for real hydrological data:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> data = np.loadtxt("examples/data/time_series1.txt")
                >>> ts = TimeSeries(data)
                >>> e = ts.extended_stats
                >>> e.index.tolist()
                ['count', 'mean', 'std', 'cv', 'skewness', 'kurtosis', 'min', '5%', '10%', '25%', '50%', '75%', '90%', '95%', 'max', 'iqr', 'mad']
                >>> round(float(e.loc["cv", "Series1"]), 4)
                0.0615
                >>> round(float(e.loc["skewness", "Series1"]), 4)
                0.9279

                ```
            - Multi-column with named columns:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> np.random.seed(42)
                >>> data = np.column_stack([np.random.randn(100) * 10 + 50,
                ...                         np.random.randn(100) * 5 + 20])
                >>> ts = TimeSeries(data, columns=["Flow", "Temp"])
                >>> e = ts.extended_stats
                >>> round(float(e.loc["mean", "Flow"]), 2)
                48.96
                >>> round(float(e.loc["mean", "Temp"]), 2)
                20.11

                ```
            - NaN values are excluded from calculations:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                >>> ts = TimeSeries(data)
                >>> e = ts.extended_stats
                >>> int(e.loc["count", "Series1"])
                9
                >>> round(float(e.loc["mean", "Series1"]), 4)
                5.7778

                ```

        See Also:
            stats: Basic 8-row summary via pandas describe().
            l_moments: L-moment ratios for robust distribution identification.
            summary: Combined table of extended_stats + L-moment ratios.
        """
        stat_names = [
            "count",
            "mean",
            "std",
            "cv",
            "skewness",
            "kurtosis",
            "min",
            "5%",
            "10%",
            "25%",
            "50%",
            "75%",
            "90%",
            "95%",
            "max",
            "iqr",
            "mad",
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

    def l_moments(self, nmom: int = 5) -> DataFrame:
        """Compute sample L-moments and L-moment ratios for each column.

        L-moments (Hosking, 1990) are summary statistics computed from linear
        combinations of order statistics. They are more robust to outliers than
        conventional moments and better suited for small samples and heavy-tailed
        data. They are the standard tool for distribution identification in
        regional frequency analysis.

        Args:
            nmom: Number of L-moments to compute. Must be >= 2. Default is 5.

        Returns:
            pandas.DataFrame:
                Rows: L1 (L-location = mean), L2 (L-scale), t (L-CV = L2/L1),
                t3 (L-skewness = L3/L2), t4 (L-kurtosis = L4/L2), and
                optionally t5 if nmom >= 5. One column per series.

        Raises:
            ValueError: If nmom < 2.

        Examples:
            - Compute L-moments for real hydrological data:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> data = np.loadtxt("examples/data/time_series1.txt")
                >>> ts = TimeSeries(data)
                >>> lm = ts.l_moments(nmom=4)
                >>> lm.index.tolist()
                ['L1', 'L2', 't', 't3', 't4']
                >>> round(float(lm.loc["L1", "Series1"]), 4)
                16.9292
                >>> round(float(lm.loc["t3", "Series1"]), 4)
                0.4815

                ```
            - L1 equals the sample mean:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                >>> ts = TimeSeries(data)
                >>> lm = ts.l_moments(nmom=2)
                >>> lm.index.tolist()
                ['L1', 'L2', 't']
                >>> float(lm.loc["L1", "Series1"])
                5.5

                ```
            - Symmetric data has L-skewness near zero:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> data = np.array([-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.])
                >>> ts = TimeSeries(data)
                >>> lm = ts.l_moments(nmom=4)
                >>> round(float(lm.loc["t3", "Series1"]), 4)
                0.0

                ```

        References:
            Hosking, J.R.M. (1990). L-moments: Analysis and estimation of
            distributions using linear combinations of order statistics. Journal
            of the Royal Statistical Society, Series B, 52(1), 105-124.

        See Also:
            extended_stats: Conventional moments (skewness, kurtosis) and percentiles.
            summary: Combined table of extended_stats + L-moment ratios.
        """
        if nmom < 2:
            raise ValueError("nmom must be >= 2")

        row_names = ["L1", "L2"]
        if nmom >= 2:
            row_names.append("t")
        if nmom >= 3:
            row_names.append("t3")
        if nmom >= 4:
            row_names.append("t4")
        if nmom >= 5:
            row_names.append("t5")

        result = DataFrame(index=row_names, columns=self.columns, dtype=float)

        for col in self.columns:
            data = self[col].dropna().values
            lmom_calc = Lmoments(data)
            lmoms = lmom_calc.calculate(nmom=nmom)

            l1 = lmoms[0]
            l2 = lmoms[1]
            result.loc["L1", col] = l1
            result.loc["L2", col] = l2
            result.loc["t", col] = l2 / l1 if not np.isclose(l1, 0.0) else np.nan
            if nmom >= 3:
                result.loc["t3", col] = (
                    lmoms[2] / l2 if not np.isclose(l2, 0.0) else np.nan
                )
            if nmom >= 4:
                result.loc["t4", col] = (
                    lmoms[3] / l2 if not np.isclose(l2, 0.0) else np.nan
                )
            if nmom >= 5:
                result.loc["t5", col] = (
                    lmoms[4] / l2 if not np.isclose(l2, 0.0) else np.nan
                )

        return result

    def summary(self) -> DataFrame:
        """Comprehensive summary table suitable for a research paper.

        Combines ``extended_stats`` and ``l_moments`` into a single table with
        13 rows per column. This produces the kind of "Table 1" that goes into
        a paper's methods or study-area section.

        Returns:
            pandas.DataFrame:
                One column per series. Rows: count, mean, std, cv, skewness,
                kurtosis, min, max, iqr, mad, L-CV, L-skewness, L-kurtosis.

        Examples:
            - Generate a paper-ready summary from real data:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> data = np.loadtxt("examples/data/time_series1.txt")
                >>> ts = TimeSeries(data)
                >>> sm = ts.summary()
                >>> sm.index.tolist()
                ['count', 'mean', 'std', 'cv', 'skewness', 'kurtosis', 'min', 'max', 'iqr', 'mad', 'L-CV', 'L-skewness', 'L-kurtosis']
                >>> len(sm)
                13
                >>> round(float(sm.loc["L-skewness", "Series1"]), 4)
                0.4815

                ```
            - Multi-column summary for comparing stations:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> np.random.seed(42)
                >>> data = np.column_stack([np.random.randn(100) * 10 + 50,
                ...                         np.random.randn(100) * 5 + 20])
                >>> ts = TimeSeries(data, columns=["Flow", "Temp"])
                >>> sm = ts.summary()
                >>> sm.columns.tolist()
                ['Flow', 'Temp']
                >>> round(float(sm.loc["mean", "Flow"]), 2)
                48.96

                ```

        See Also:
            extended_stats: The 17-row conventional statistics table.
            l_moments: L-moment ratios independently.
        """
        estats = self.extended_stats
        lmom = self.l_moments(nmom=4)

        summary_rows = [
            "count",
            "mean",
            "std",
            "cv",
            "skewness",
            "kurtosis",
            "min",
            "max",
            "iqr",
            "mad",
        ]
        result = estats.loc[summary_rows, :].copy()

        result.loc["L-CV"] = lmom.loc["t"]
        result.loc["L-skewness"] = lmom.loc["t3"]
        result.loc["L-kurtosis"] = lmom.loc["t4"]

        return result
