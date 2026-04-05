"""Seasonal and periodic analysis mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.signal import periodogram as scipy_periodogram
from scipy.signal import welch as scipy_welch

if TYPE_CHECKING:
    from pandas import Index


class SeasonalMixin:
    """Mixin providing seasonal and periodic analysis methods for TimeSeries.

    This mixin is designed to be composed with ``TimeSeriesBase`` (a ``pandas.DataFrame`` subclass).
    """

    if TYPE_CHECKING:
        columns: Index
        index: Index

        @staticmethod
        def _get_ax_fig(  # noqa: E704
            n_subplots: int = 1, **kwargs: object
        ) -> Tuple[Figure, Axes]: ...

        @staticmethod
        def _adjust_axes_labels(  # noqa: E704
            ax: Axes, tick_labels: list[str] | None = None, **kwargs: object
        ) -> Axes: ...

        def __getitem__(self, key: str) -> DataFrame:  # noqa: E704
            ...

    def monthly_stats(self, column: str = None) -> DataFrame:
        """Compute statistics grouped by month.

        Requires a DatetimeIndex. Computes mean, std, cv, min, max, median,
        and skewness for each month (1-12) and each column.

        Args:
            column: Column to analyze. If None, uses first column.

        Returns:
            pandas.DataFrame: Index is month (1-12), columns are statistic names.

        Raises:
            TypeError: If the index is not a DatetimeIndex.

        Examples:
            ```python
            >>> import numpy as np
            >>> import pandas as pd
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> idx = pd.date_range("2000-01-01", periods=730, freq="D")
            >>> ts = TimeSeries(np.random.randn(730), index=idx)
            >>> result = ts.monthly_stats()
            >>> result.shape[0] == 12
            True

            ```
        """
        import pandas as pd
        from scipy.stats import skew

        if column is None:
            column = self.columns[0]

        series = self[column].dropna()
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("monthly_stats requires a DatetimeIndex.")

        grouped = series.groupby(series.index.month)

        rows = []
        for month, group in grouped:
            vals = group.values
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            rows.append(
                {
                    "month": int(month),
                    "mean": mean,
                    "std": std,
                    "cv": std / mean if not np.isclose(mean, 0.0) else np.nan,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "median": float(np.median(vals)),
                    "skewness": (
                        float(skew(vals, bias=False)) if len(vals) >= 3 else np.nan
                    ),
                }
            )

        result = DataFrame(rows).set_index("month")
        return result

    def seasonal_subseries(
        self,
        period: int = 12,
        column: str = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """Seasonal subseries plot.

        Plots each season (e.g., each month) as a separate mini time series,
        with horizontal lines at the season mean. Reveals seasonal patterns
        and trends within individual seasons.

        Args:
            period: Number of seasons per cycle. Default 12 (monthly).
            column: Column to plot. If None, uses first column.
            **kwargs: Passed to figure layout.

        Returns:
            tuple: (Figure, Axes)

        Examples:
            ```python
            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> ts = TimeSeries(np.sin(np.arange(120) * 2 * np.pi / 12))  # doctest: +SKIP
            >>> fig, ax = ts.seasonal_subseries(period=12)  # doctest: +SKIP

            ```
        """
        if column is None:
            column = self.columns[0]

        data = self[column].dropna().values

        fig, axes = plt.subplots(
            1, period, figsize=(max(period * 1.5, 10), 4), sharey=True
        )
        if period == 1:
            axes = [axes]

        for s in range(period):
            season_data = data[s::period]
            ax = axes[s]
            ax.plot(
                range(len(season_data)),
                season_data,
                "o-",
                markersize=3,
                linewidth=0.8,
                color="steelblue",
            )
            ax.axhline(np.mean(season_data), color="red", linewidth=1, linestyle="--")
            ax.set_title(f"S{s + 1}", fontsize=9)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"Seasonal Subseries — {column}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()
        return fig, axes[-1]

    def annual_cycle(
        self,
        column: str = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """Overlay all years on a single Jan-Dec axis.

        Requires a DatetimeIndex. Plots each year as a gray line with the
        long-term monthly mean as a bold line.

        Args:
            column: Column to plot. If None, uses first column.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (Figure, Axes)

        Raises:
            TypeError: If the index is not a DatetimeIndex.

        Examples:
            ```python
            >>> import numpy as np  # doctest: +SKIP
            >>> import pandas as pd  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> idx = pd.date_range("2000-01-01", periods=730, freq="D")  # doctest: +SKIP
            >>> ts = TimeSeries(np.random.randn(730), index=idx)  # doctest: +SKIP
            >>> fig, ax = ts.annual_cycle()  # doctest: +SKIP

            ```
        """
        import pandas as pd

        if column is None:
            column = self.columns[0]

        series = self[column].dropna()
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("annual_cycle requires a DatetimeIndex.")

        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)

        # Plot each year as a gray line
        for year, group in series.groupby(series.index.year):
            month_means = group.groupby(group.index.month).mean()
            ax.plot(
                month_means.index,
                month_means.values,
                color="gray",
                alpha=0.3,
                linewidth=0.8,
            )

        # Bold mean line
        overall_monthly = series.groupby(series.index.month).mean()
        ax.plot(
            overall_monthly.index,
            overall_monthly.values,
            color="darkblue",
            linewidth=2.5,
            label="Mean",
        )

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

        if "title" not in kwargs:
            kwargs["title"] = f"Annual Cycle — {column}"
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = "Month"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "Value"

        ax = self._adjust_axes_labels(ax, **kwargs)
        plt.show()
        return fig, ax

    def periodogram(
        self,
        column: str = None,
        method: str = "welch",
        fs: float = 1.0,
        plot: bool = True,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[Figure, Axes]]]:
        """Compute and optionally plot the power spectral density.

        Identifies dominant periodicities/frequencies in the time series.

        Args:
            column: Column to analyze. If None, uses first column.
            method: Spectral estimation method.
                - "periodogram": Raw periodogram (``scipy.signal.periodogram``).
                - "welch": Smoothed estimate (``scipy.signal.welch``). Default.
            fs: Sampling frequency. Default 1.0 (one sample per time unit).
            plot: Whether to produce a plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (frequencies, power, (fig, ax)) or (frequencies, power, None).

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> t = np.arange(500)
            >>> data = np.sin(2 * np.pi * t / 50) + np.random.randn(500) * 0.5
            >>> ts = TimeSeries(data)
            >>> freqs, power, _ = ts.periodogram(plot=False)
            >>> len(freqs) > 0
            True

            ```
        """
        if column is None:
            column = self.columns[0]

        data = self[column].dropna().values

        if method == "welch":
            freqs, power = scipy_welch(data, fs=fs)
        elif method == "periodogram":
            freqs, power = scipy_periodogram(data, fs=fs)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'welch', 'periodogram'."
            )

        fig_ax: Optional[Tuple[Figure, Axes]] = None
        if plot:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            ax.semilogy(freqs, power, color="steelblue", linewidth=0.8)

            # Annotate dominant peak
            if len(power) > 1:
                peak_idx = np.argmax(power[1:]) + 1  # skip DC component
                peak_freq = freqs[peak_idx]
                peak_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
                ax.axvline(peak_freq, color="red", linestyle="--", linewidth=0.7)
                ax.annotate(
                    f"Peak: T={peak_period:.1f}",
                    xy=(peak_freq, power[peak_idx]),
                    fontsize=9,
                    color="red",
                )

            if "title" not in kwargs:
                kwargs["title"] = f"Power Spectral Density — {column}"
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Frequency"
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = "Power"

            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return freqs, power, fig_ax

    def seasonal_mann_kendall(
        self,
        period: int = 12,
        alpha: float = 0.05,
        column: str = None,
    ) -> DataFrame:
        """Seasonal Mann-Kendall trend test.

        Applies the Mann-Kendall test season-by-season and combines the results.
        The combined S statistic and its variance are summed across seasons
        (Hirsch et al., 1982).

        Args:
            period: Number of seasons per cycle (e.g., 12 for monthly). Default 12.
            alpha: Significance level. Default 0.05.
            column: Column to test. If None, tests all columns.

        Returns:
            pandas.DataFrame: One row per column with: trend, h, p_value, z,
                combined_s, combined_var_s, per_season_s (list).

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> t = np.arange(120)
            >>> data = 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.randn(120)
            >>> ts = TimeSeries(data)
            >>> result = ts.seasonal_mann_kendall(period=12)
            >>> result.loc["Series1", "trend"]
            'increasing'

            ```

        References:
            Hirsch, R.M., Slack, J.R. and Smith, R.A. (1982). Techniques of trend
            analysis for monthly water quality data. Water Resources Research, 18(1), 107-121.
        """
        from scipy.stats import norm as scipy_norm

        from statista.time_series.trend import _mk_score, _mk_variance

        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values

            combined_s = 0.0
            combined_var = 0.0
            per_season_s = []

            for s in range(period):
                season_data = data[s::period]
                if len(season_data) < 3:
                    per_season_s.append(0.0)
                    continue
                ns = len(season_data)
                s_val = _mk_score(season_data, ns)
                var_val = _mk_variance(season_data, ns)
                combined_s += s_val
                combined_var += var_val
                per_season_s.append(float(s_val))

            if combined_var > 0:
                if combined_s > 0:
                    z = (combined_s - 1) / np.sqrt(combined_var)
                elif combined_s < 0:
                    z = (combined_s + 1) / np.sqrt(combined_var)
                else:
                    z = 0.0
            else:
                z = 0.0

            p_value = 2.0 * (1.0 - scipy_norm.cdf(abs(z)))
            h = abs(z) > scipy_norm.ppf(1 - alpha / 2)

            if z > 0 and h:
                trend = "increasing"
            elif z < 0 and h:
                trend = "decreasing"
            else:
                trend = "no trend"

            rows.append(
                {
                    "column": col,
                    "trend": trend,
                    "h": bool(h),
                    "p_value": float(p_value),
                    "z": float(z),
                    "combined_s": float(combined_s),
                    "combined_var_s": float(combined_var),
                    "per_season_s": per_season_s,
                }
            )

        result_df = DataFrame(rows).set_index("column")
        return result_df
