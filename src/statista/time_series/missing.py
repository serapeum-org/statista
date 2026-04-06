"""Missing data and quality control mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.stats import median_abs_deviation

if TYPE_CHECKING:
    from statista.time_series.stubs import _TimeSeriesStub
else:
    _TimeSeriesStub = object


class MissingData(_TimeSeriesStub):
    """Missing data diagnostics and outlier detection for TimeSeries."""

    def missing_summary(self) -> DataFrame:
        """Per-column summary of missing data.

        Returns a DataFrame with one row per column containing: total count, missing count,
        missing percentage, valid count, longest gap (consecutive NaN run), number of gaps,
        mean gap length, first valid index, and last valid index.

        Returns:
            pandas.DataFrame
                Rows are series column names. Columns: total_count, missing_count, missing_pct,
                valid_count, longest_gap, n_gaps, mean_gap_length, first_valid, last_valid.

        Examples:
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries

            Basic usage with a small array containing two gaps:

            >>> data = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])
            >>> ts = TimeSeries(data)
            >>> result = ts.missing_summary()
            >>> int(result.loc["Series1", "missing_count"])
            3
            >>> round(float(result.loc["Series1", "missing_pct"]), 4)
            42.8571
            >>> int(result.loc["Series1", "longest_gap"])
            2
            >>> int(result.loc["Series1", "n_gaps"])
            2
            >>> round(float(result.loc["Series1", "mean_gap_length"]), 1)
            1.5

            Real data with no missing values:

            >>> data = np.loadtxt("examples/data/time_series1.txt")
            >>> ts = TimeSeries(data)
            >>> result = ts.missing_summary()
            >>> int(result.loc["Series1", "missing_count"])
            0
            >>> int(result.loc["Series1", "total_count"])
            27

            Random data with inserted gaps:

            >>> np.random.seed(42)
            >>> data = np.random.randn(50)
            >>> data[5:10] = np.nan
            >>> data[20:23] = np.nan
            >>> data[40] = np.nan
            >>> ts = TimeSeries(data)
            >>> result = ts.missing_summary()
            >>> int(result.loc["Series1", "missing_count"])
            9
            >>> int(result.loc["Series1", "longest_gap"])
            5
            >>> int(result.loc["Series1", "n_gaps"])
            3
        """
        rows = {}
        for col in self.columns:
            series = self[col]
            total = len(series)
            missing = int(series.isna().sum())
            valid = total - missing
            mask = series.isna().values

            gap_lengths = _run_lengths(mask, value=True)
            n_gaps = len(gap_lengths)
            longest_gap = max(gap_lengths) if gap_lengths else 0
            mean_gap = float(np.mean(gap_lengths)) if gap_lengths else 0.0

            rows[col] = {
                "total_count": total,
                "missing_count": missing,
                "missing_pct": 100.0 * missing / total if total > 0 else 0.0,
                "valid_count": valid,
                "longest_gap": longest_gap,
                "n_gaps": n_gaps,
                "mean_gap_length": mean_gap,
                "first_valid": series.first_valid_index(),
                "last_valid": series.last_valid_index(),
            }

        result = DataFrame.from_dict(rows, orient="index")
        return result

    def gap_analysis(self, column: str = None) -> DataFrame:
        """Identify all contiguous gaps (runs of NaN) in the data.

        Args:
            column: Column name to analyze. If None, analyzes all columns.

        Returns:
            pandas.DataFrame
                Each row is one gap. Columns: column, gap_start, gap_end, gap_length.
                Sorted by gap_length descending.

        Examples:
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries

            Two gaps of different lengths (sorted longest first):

            >>> data = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
            >>> ts = TimeSeries(data)
            >>> gaps = ts.gap_analysis()
            >>> len(gaps)
            2
            >>> int(gaps.iloc[0]["gap_length"])
            2
            >>> int(gaps.iloc[1]["gap_length"])
            1

            Multiple gaps including one at the end of the series:

            >>> data = np.array([1.0, np.nan, np.nan, np.nan, 5.0, np.nan, 7.0, 8.0, np.nan, np.nan])
            >>> ts = TimeSeries(data)
            >>> gaps = ts.gap_analysis()
            >>> len(gaps)
            3
            >>> [int(gaps.iloc[i]["gap_length"]) for i in range(3)]
            [3, 2, 1]

            Data with no gaps returns an empty DataFrame:

            >>> data = np.loadtxt("examples/data/time_series1.txt")
            >>> ts = TimeSeries(data)
            >>> gaps = ts.gap_analysis()
            >>> len(gaps)
            0
        """
        cols = [column] if column is not None else list(self.columns)
        records = []

        for col in cols:
            series = self[col]
            mask = series.isna().values
            idx = series.index

            in_gap = False
            gap_start_pos = 0
            for i, is_nan in enumerate(mask):
                if is_nan and not in_gap:
                    in_gap = True
                    gap_start_pos = i
                elif not is_nan and in_gap:
                    in_gap = False
                    records.append(
                        {
                            "column": col,
                            "gap_start": idx[gap_start_pos],
                            "gap_end": idx[i - 1],
                            "gap_length": i - gap_start_pos,
                        }
                    )
            if in_gap:
                records.append(
                    {
                        "column": col,
                        "gap_start": idx[gap_start_pos],
                        "gap_end": idx[len(mask) - 1],
                        "gap_length": len(mask) - gap_start_pos,
                    }
                )

        result = (
            DataFrame(records)
            if records
            else DataFrame(columns=["column", "gap_start", "gap_end", "gap_length"])
        )
        result = result.sort_values("gap_length", ascending=False).reset_index(
            drop=True
        )
        return result

    def completeness_report(self, freq: str = "YE") -> DataFrame:
        """Data completeness percentage per time period.

        Groups data by the specified frequency and computes the percentage of non-missing
        values in each period for each column.

        Args:
            freq: Pandas offset alias for grouping (e.g., "YE" for yearly, "ME" for monthly).
                Default is "YE".

        Returns:
            pandas.DataFrame
                Index is the period end date. Columns are the series names.
                Values are completeness percentages (0-100).

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from statista.time_series import TimeSeries

            Yearly completeness with 10 missing days:

            >>> np.random.seed(42)
            >>> idx = pd.date_range("2000-01-01", periods=365, freq="D")
            >>> data = np.random.randn(365)
            >>> data[10:20] = np.nan
            >>> ts = TimeSeries(data, index=idx)
            >>> report = ts.completeness_report(freq="YE")
            >>> report.shape[0]
            1
            >>> round(float(report.iloc[0, 0]), 4)
            97.2603

            Monthly completeness for a 60-day window with a gap in January:

            >>> np.random.seed(42)
            >>> idx = pd.date_range("2020-01-01", periods=60, freq="D")
            >>> data = np.random.randn(60)
            >>> data[5:15] = np.nan
            >>> ts = TimeSeries(data, index=idx)
            >>> report = ts.completeness_report(freq="ME")
            >>> round(float(report.iloc[0, 0]), 4)
            67.7419
            >>> round(float(report.iloc[1, 0]), 4)
            100.0
        """
        result = self.resample(freq).apply(lambda x: x.notna().mean() * 100)
        return DataFrame(result)

    def detect_outliers(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        column: str = None,
    ) -> DataFrame:
        """Detect outliers in the data.

        Args:
            method: Detection method. One of:
                - "iqr": Values outside [Q1 - threshold*IQR, Q3 + threshold*IQR]. Default threshold=1.5.
                - "zscore": |z| > threshold. Default threshold=3.0.
                - "modified_zscore": MAD-based z-score > threshold. Default threshold=3.5.
                  Uses median and MAD instead of mean and std (Iglewicz & Hoaglin, 1993).
            threshold: Threshold for the chosen method. Interpretation depends on method.
            column: Column name to analyze. If None, analyzes all columns.

        Returns:
            pandas.DataFrame
                Boolean DataFrame with same shape as input (or single column if column specified).
                True indicates an outlier.

        Examples:
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries

            Z-score method detects the extreme value at index 3:

            >>> data = np.array([1.0, 2.0, 3.0, 100.0, 2.5, 1.5])
            >>> ts = TimeSeries(data)
            >>> outliers = ts.detect_outliers(method="zscore", threshold=2.0)
            >>> bool(outliers.loc[3, "Series1"])
            True
            >>> int(outliers.sum().iloc[0])
            1

            IQR method on the same data:

            >>> outliers_iqr = ts.detect_outliers(method="iqr", threshold=1.5)
            >>> bool(outliers_iqr.loc[3, "Series1"])
            True
            >>> int(outliers_iqr.sum().iloc[0])
            1

            Random data with injected extreme values:

            >>> np.random.seed(42)
            >>> data = np.concatenate([np.random.randn(100), [10.0, -10.0]])
            >>> ts = TimeSeries(data)
            >>> outliers = ts.detect_outliers(method="zscore", threshold=2.0)
            >>> int(outliers.sum().iloc[0])
            2
        """
        cols = [column] if column is not None else list(self.columns)
        result = DataFrame(index=self.index, columns=cols, dtype=bool)
        result[:] = False

        for col in cols:
            data = self[col]
            valid = data.dropna()

            if method == "iqr":
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                result[col] = (data < lower) | (data > upper)
            elif method == "zscore":
                mean = valid.mean()
                std = valid.std()
                if std > 0:
                    z = np.abs((data - mean) / std)
                    result[col] = z > threshold
            elif method == "modified_zscore":
                median = valid.median()
                mad = median_abs_deviation(valid.values)
                if mad > 0:
                    modified_z = 0.6745 * np.abs(data - median) / mad
                    result[col] = modified_z > threshold
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Choose from 'iqr', 'zscore', 'modified_zscore'."
                )

        return result

    def outlier_plot(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        column: str = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Time series plot with outliers highlighted.

        Plots the time series with detected outlier points shown in a different color and marker.

        Args:
            method: Detection method (see ``detect_outliers``).
            threshold: Threshold for the chosen method.
            column: Column to plot. If None and single-column, uses that column.
            **kwargs: Passed to ``_adjust_axes_labels`` (title, xlabel, ylabel, etc.).

        Returns:
            tuple: (Figure, Axes)

        Examples:
            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> np.random.seed(42)  # doctest: +SKIP
            >>> data = np.concatenate([np.random.randn(100), [10.0, -10.0]])  # doctest: +SKIP
            >>> ts = TimeSeries(data)  # doctest: +SKIP
            >>> fig, ax = ts.outlier_plot(method="zscore", threshold=2.5)  # doctest: +SKIP

            Using IQR method:

            >>> data = np.loadtxt("examples/data/time_series1.txt")  # doctest: +SKIP
            >>> ts = TimeSeries(data)  # doctest: +SKIP
            >>> fig, ax = ts.outlier_plot(method="iqr", threshold=1.5)  # doctest: +SKIP
        """
        if column is None and len(self.columns) == 1:
            column = self.columns[0]
        elif column is None:
            column = self.columns[0]

        outlier_mask = self.detect_outliers(
            method=method, threshold=threshold, column=column
        )

        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)

        series = self[column]
        mask = outlier_mask[column]

        ax.plot(
            series.index, series.values, color="steelblue", linewidth=0.8, label="Data"
        )
        ax.scatter(
            series.index[mask],
            series.values[mask],
            color="red",
            s=30,
            zorder=5,
            label="Outliers",
        )

        if method == "iqr":
            valid = series.dropna()
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            ax.axhline(
                q1 - threshold * iqr,
                color="orange",
                linestyle="--",
                linewidth=0.7,
                label="Lower bound",
            )
            ax.axhline(
                q3 + threshold * iqr,
                color="orange",
                linestyle="--",
                linewidth=0.7,
                label="Upper bound",
            )

        ax = self._adjust_axes_labels(ax, **kwargs)

        plt.show()
        return fig, ax


def _run_lengths(mask: np.ndarray, value: bool = True) -> list[int]:
    """Compute lengths of consecutive runs of a given value in a boolean array.

    Args:
        mask: 1D boolean array.
        value: The value to find runs of. Default True (NaN runs).

    Returns:
        List of run lengths.
    """
    if len(mask) == 0:
        return []

    lengths = []
    count = 0
    for m in mask:
        if m == value:
            count += 1
        else:
            if count > 0:
                lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)

    return lengths
