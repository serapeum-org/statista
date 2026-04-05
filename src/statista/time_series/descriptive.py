"""Descriptive statistics mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING  # noqa: F401

import numpy as np
from pandas import DataFrame
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import median_abs_deviation, skew

from statista.parameters import Lmoments


class DescriptiveMixin:
    """Mixin providing descriptive statistical methods for TimeSeries.

    This mixin is designed to be composed with ``TimeSeriesBase`` (a ``pandas.DataFrame`` subclass).
    All attribute access (``self.columns``, ``self.describe()``, indexing) is provided by DataFrame
    at runtime.
    """

    if TYPE_CHECKING:
        # Allow mypy to see DataFrame attributes on the mixin.
        columns: DataFrame.columns  # type: ignore[assignment]

        def describe(self) -> DataFrame:  # noqa: E704
            ...

        def __getitem__(self, key: str) -> DataFrame:  # noqa: E704
            ...

    @property
    def stats(self) -> DataFrame:
        """
        Returns a detailed statistical summary of the time series.

        Returns:
            pandas.DataFrame
                Statistical summary including count, mean, std, min, 25%, 50%, 75%, max.

        Examples:
            ```python
            >>> from statista.time_series import TimeSeries
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> ts = TimeSeries(np.random.randn(100))
            >>> ts.stats.index.tolist()
            ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

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

        L-moments (Hosking, 1990) are summary statistics computed from linear combinations of order statistics.
        They are more robust to outliers than conventional moments and better suited for small samples and
        heavy-tailed data. They are the standard tool for distribution identification in regional frequency
        analysis.

        Args:
            nmom: Number of L-moments to compute. Must be >= 2. Default is 5.

        Returns:
            pandas.DataFrame
                Rows: L1 (L-location = mean), L2 (L-scale), t (L-CV = L2/L1), t3 (L-skewness = L3/L2),
                t4 (L-kurtosis = L4/L2), and optionally t5 if nmom >= 5. One column per series.

        Raises:
            ValueError: If nmom < 2.

        Examples:
            - Compute L-moments for a 1D time series:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> np.random.seed(42)
                >>> ts = TimeSeries(np.random.randn(200))
                >>> result = ts.l_moments(nmom=4)
                >>> "L1" in result.index
                True
                >>> "t3" in result.index
                True

                ```
            - Compute L-moments for a 2D time series:
                ```python
                >>> ts_2d = TimeSeries(np.random.randn(200, 3), columns=['A', 'B', 'C'])
                >>> result = ts_2d.l_moments()
                >>> result.columns.tolist()
                ['A', 'B', 'C']

                ```

        References:
            Hosking, J.R.M. (1990). L-moments: Analysis and estimation of distributions using linear
            combinations of order statistics. Journal of the Royal Statistical Society, Series B, 52(1), 105-124.
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

        Combines ``extended_stats`` and ``l_moments`` into a single table. This produces the kind of
        "Table 1" that goes into a paper's methods or study-area section.

        Returns:
            pandas.DataFrame
                Rows: count, mean, std, cv, skewness, kurtosis, min, max, iqr, mad,
                L-CV (t), L-skewness (t3), L-kurtosis (t4). One column per series.

        Examples:
            - Compute summary for a 1D time series:
                ```python
                >>> import numpy as np
                >>> from statista.time_series import TimeSeries
                >>> np.random.seed(42)
                >>> ts = TimeSeries(np.random.randn(200))
                >>> result = ts.summary()
                >>> "L-CV" in result.index
                True
                >>> "L-skewness" in result.index
                True

                ```
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
