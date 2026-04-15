"""Comparison and anomaly mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.stats import mannwhitneyu

if TYPE_CHECKING:
    from statista.time_series.stubs import _TimeSeriesStub
else:
    _TimeSeriesStub = object


class Comparison(_TimeSeriesStub):
    """Comparison, anomaly, and regime analysis methods."""

    def anomaly(
        self,
        reference: str = "mean",
        column: str = None,
        plot: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, tuple[Figure, Axes] | None]:
        """Compute anomaly (deviation from reference) and optionally plot.

        Args:
            reference: Reference to compute deviation from.
                - "mean": long-term mean. Default.
                - "median": long-term median.
            column: Column to analyze. If None, uses first column.
            plot: Whether to produce a bar/filled plot colored by sign. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (anomaly_ts, (fig, ax)) or (anomaly_ts, None).
                anomaly_ts is a TimeSeries of deviations.

        Examples:
            Compute anomaly (deviation from mean) without plotting:

            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(100))
            >>> anom, _ = ts.anomaly(plot=False)
            >>> round(float(anom.values.mean()), 4)
            -0.0
            >>> [round(float(v), 4) for v in anom.values.flatten()[:3]]
            [0.6006, -0.0344, 0.7515]

            Anomaly relative to the median:

            >>> anom2, _ = ts.anomaly(reference="median", plot=False)
            >>> [round(float(v), 4) for v in anom2.values.flatten()[:3]]
            [0.6237, -0.0113, 0.7746]
        """
        from statista.time_series import TimeSeries

        if column is None:
            column = self.columns[0]

        series = self[column].dropna()

        if reference == "mean":
            ref_val = series.mean()
        elif reference == "median":
            ref_val = series.median()
        else:
            raise ValueError(
                f"reference must be 'mean' or 'median', got '{reference}'."
            )

        anom_values = series - ref_val
        anom_ts = TimeSeries(
            anom_values.values.reshape(-1, 1),
            index=series.index,
            columns=[column],
        )

        fig_ax: tuple[Figure, Axes] | None = None
        if plot:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            colors = [
                "steelblue" if v >= 0 else "firebrick" for v in anom_values.values
            ]
            ax.bar(
                series.index,
                anom_values.values,
                color=colors,
                width=1.0,
                edgecolor="none",
            )
            ax.axhline(0, color="black", linewidth=0.5)

            if "title" not in kwargs:
                kwargs["title"] = f"Anomaly — {column} (ref={reference})"
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Index"
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = "Anomaly"

            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return anom_ts, fig_ax

    def standardized_anomaly(self, column: str = None) -> Any:
        """Compute standardized anomaly per month.

        Removes the seasonal cycle: (x - monthly_mean) / monthly_std.
        Result is dimensionless (in units of standard deviation).

        Requires a DatetimeIndex.

        Args:
            column: Column to analyze. If None, uses first column.

        Returns:
            TimeSeries: Standardized anomaly series.

        Raises:
            TypeError: If the index is not a DatetimeIndex.

        Examples:
            Remove the seasonal cycle from two years of daily data:

            >>> import numpy as np
            >>> import pandas as pd
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> idx = pd.date_range("2000-01-01", periods=730, freq="D")
            >>> ts = TimeSeries(np.random.randn(730), index=idx)
            >>> sa = ts.standardized_anomaly()
            >>> sa.shape
            (730, 1)
            >>> round(float(sa.values.mean()), 4)
            -0.0
            >>> round(float(sa.values.std()), 4)
            0.9917
        """
        import pandas as pd

        from statista.time_series import TimeSeries

        if column is None:
            column = self.columns[0]

        series = self[column].dropna()
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("standardized_anomaly requires a DatetimeIndex.")

        monthly_mean = series.groupby(series.index.month).transform("mean")
        monthly_std = series.groupby(series.index.month).transform("std")

        # Replace zero std with NaN to avoid inf, then fill all NaN/inf with 0.0
        monthly_std = monthly_std.replace(0.0, np.nan)
        std_anom = (series - monthly_mean) / monthly_std
        std_anom = std_anom.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        result = TimeSeries(
            std_anom.values.reshape(-1, 1),
            index=series.index,
            columns=[column],
        )
        return result

    def double_mass_curve(
        self,
        col_x: str,
        col_y: str,
        plot: bool = True,
        **kwargs: Any,
    ) -> tuple[DataFrame, tuple[Figure, Axes] | None]:
        """Double mass curve — cumulative X vs cumulative Y.

        Used to detect inconsistencies in the relationship between two correlated
        time series (e.g., precipitation at two stations). A slope change indicates
        a shift in the relationship.

        Args:
            col_x: First column (x-axis cumulative).
            col_y: Second column (y-axis cumulative).
            plot: Whether to produce a plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (dmc_df, (fig, ax)) or (dmc_df, None).
                dmc_df has columns: cumsum_x, cumsum_y.

        Examples:
            Check consistency between two correlated series:

            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.column_stack([np.random.randn(100), np.random.randn(100) * 2])
            >>> ts = TimeSeries(data, columns=["A", "B"])
            >>> dmc, _ = ts.double_mass_curve("A", "B", plot=False)
            >>> list(dmc.columns)
            ['cumsum_A', 'cumsum_B']
            >>> dmc.shape
            (100, 2)
            >>> round(float(dmc["cumsum_A"].iloc[0]), 4)
            0.4967
            >>> round(float(dmc["cumsum_A"].iloc[-1]), 4)
            -10.3847
        """
        x = self[col_x].dropna().values
        y = self[col_y].dropna().values
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        cumsum_x = np.cumsum(x)
        cumsum_y = np.cumsum(y)

        dmc_df = DataFrame(
            {
                f"cumsum_{col_x}": cumsum_x,
                f"cumsum_{col_y}": cumsum_y,
            }
        )

        fig_ax: tuple[Figure, Axes] | None = None
        if plot:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            ax.plot(
                cumsum_x, cumsum_y, "o-", markersize=2, color="steelblue", linewidth=1
            )

            if "title" not in kwargs:
                kwargs["title"] = f"Double Mass Curve — {col_x} vs {col_y}"
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = f"Cumulative {col_x}"
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = f"Cumulative {col_y}"

            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return dmc_df, fig_ax

    def regime_comparison(
        self,
        split_at: int,
        column: str = None,
    ) -> DataFrame:
        """Compare statistics before and after a split point (e.g., change point).

        Splits the series and computes mean, std, cv, median, min, max, skewness
        for each segment. Also runs a Mann-Whitney U test for significance.

        Args:
            split_at: Index position to split the series.
            column: Column to analyze. If None, uses first column.

        Returns:
            pandas.DataFrame: Columns for 'before' and 'after' statistics, plus
                relative_change_pct and mann_whitney_p.

        Examples:
            Compare regimes before and after a shift at index 50:

            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.concatenate([np.random.randn(50), np.random.randn(50) + 3])
            >>> ts = TimeSeries(data)
            >>> result = ts.regime_comparison(split_at=50)
            >>> round(float(result.loc["mean", "before"]), 4)
            -0.2255
            >>> round(float(result.loc["mean", "after"]), 4)
            3.0178
            >>> round(float(result.loc["std", "before"]), 4)
            0.9337

            The Mann-Whitney U test detects a significant difference:

            >>> float(result.loc["mann_whitney_U", "after"]) < 0.001
            True

        References:
            Mann, H.B. and Whitney, D.R. (1947). On a test of whether one of two random
            variables is stochastically larger than the other. Annals of Mathematical
            Statistics, 18(1), 50-60.
        """
        from scipy.stats import skew

        if column is None:
            column = self.columns[0]

        data = self[column].dropna().values
        if split_at <= 0 or split_at >= len(data):
            raise ValueError(
                f"split_at must be between 1 and {len(data) - 1}, got {split_at}."
            )
        before = data[:split_at]
        after = data[split_at:]

        stats_names = ["mean", "std", "cv", "median", "min", "max", "skewness"]
        result = DataFrame(
            index=stats_names, columns=["before", "after", "relative_change_pct"]
        )

        for label, segment in [("before", before), ("after", after)]:
            mean = float(np.mean(segment))
            std = float(np.std(segment, ddof=1))
            result.loc["mean", label] = mean
            result.loc["std", label] = std
            result.loc["cv", label] = (
                std / mean if not np.isclose(mean, 0.0) else np.nan
            )
            result.loc["median", label] = float(np.median(segment))
            result.loc["min", label] = float(np.min(segment))
            result.loc["max", label] = float(np.max(segment))
            result.loc["skewness", label] = (
                float(skew(segment, bias=False)) if len(segment) >= 3 else np.nan
            )

        # Relative change (%)
        for stat in stats_names:
            b = (
                float(result.loc[stat, "before"])
                if result.loc[stat, "before"] is not None
                else 0
            )
            a = (
                float(result.loc[stat, "after"])
                if result.loc[stat, "after"] is not None
                else 0
            )
            if b != 0 and not np.isnan(b) and not np.isnan(a):
                result.loc[stat, "relative_change_pct"] = ((a - b) / abs(b)) * 100
            else:
                result.loc[stat, "relative_change_pct"] = np.nan

        # Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(before, after, alternative="two-sided")
        result.loc["mann_whitney_U", "before"] = float(u_stat)
        result.loc["mann_whitney_U", "after"] = float(p_value)
        result.loc["mann_whitney_U", "relative_change_pct"] = np.nan

        return result
