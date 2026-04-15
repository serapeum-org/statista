"""Decomposition and smoothing mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    from statista.time_series.stubs import _TimeSeriesStub
else:
    _TimeSeriesStub = object


class Decomposition(_TimeSeriesStub):
    """Time series decomposition and smoothing methods."""

    def classical_decompose(
        self,
        period: int = None,
        model: str = "additive",
        column: str = None,
        plot: bool = True,
        **kwargs: Any,
    ) -> tuple[DataFrame, tuple[Figure, Axes] | None]:
        """Classical seasonal decomposition using moving averages.

        Decomposes the time series into trend, seasonal, and residual components.

        - Additive: Y(t) = Trend(t) + Seasonal(t) + Residual(t)
        - Multiplicative: Y(t) = Trend(t) * Seasonal(t) * Residual(t)

        Implemented from scratch (no statsmodels dependency).

        Args:
            period: Length of the seasonal cycle (e.g., 12 for monthly data with
                annual seasonality, 7 for daily data with weekly seasonality).
                **Required** — there is no auto-detection.
            model: "additive" or "multiplicative". Default "additive".
            column: Column to decompose. If None, uses first column.
            plot: Whether to produce a 4-panel decomposition plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (decomposition_df, (fig, axes)) or (decomposition_df, None).
                decomposition_df has columns: observed, trend, seasonal, residual.

        Raises:
            ValueError: If period is None or data length < 2 * period.

        Examples:
            Decompose a synthetic monthly series with trend and seasonality:

            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> t = np.arange(120)
            >>> seasonal = 5 * np.sin(2 * np.pi * t / 12)
            >>> trend = 0.1 * t
            >>> data = trend + seasonal + np.random.randn(120) * 0.5
            >>> ts = TimeSeries(data)
            >>> result, _ = ts.classical_decompose(period=12, plot=False)
            >>> list(result.columns)
            ['observed', 'trend', 'seasonal', 'residual']
            >>> result.shape
            (120, 4)
            >>> round(float(result["trend"].dropna().mean()), 4)
            5.8866

            Verify seasonal component captures the pattern:

            >>> round(float(result["seasonal"].std()), 4)
            3.495

            Decompose a shorter series with stronger trend:

            >>> np.random.seed(42)
            >>> t2 = np.arange(48)
            >>> data2 = 10 + 0.5 * t2 + 3 * np.sin(2 * np.pi * t2 / 12) + np.random.randn(48) * 0.3
            >>> ts2 = TimeSeries(data2)
            >>> result2, _ = ts2.classical_decompose(period=12, plot=False)
            >>> round(float(result2["trend"].iloc[24]), 4)
            21.3941

        References:
            Persons, W.M. (1919). Indices of business conditions.
            Review of Economics and Statistics, 1(1), 5-107.
        """
        if period is None:
            raise ValueError(
                "period must be specified (e.g., 12 for monthly, 7 for daily)."
            )

        if column is None:
            column = self.columns[0]

        data = self[column].dropna().values.astype(float)
        n = len(data)
        idx = self[column].dropna().index

        if n < 2 * period:
            raise ValueError(f"Data length ({n}) must be >= 2 * period ({2 * period}).")

        # Step 1: Trend via centered moving average
        trend = _centered_moving_average(data, period)

        # Step 2: Detrended series
        if model == "additive":
            detrended = data - trend
        elif model == "multiplicative":
            detrended = data / np.where(trend != 0, trend, np.nan)
        else:
            raise ValueError(
                f"model must be 'additive' or 'multiplicative', got '{model}'."
            )

        # Step 3: Seasonal component (average of detrended values at each position in the cycle)
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            valid = detrended[indices]
            valid = valid[~np.isnan(valid)]
            season_mean = np.mean(valid) if len(valid) > 0 else 0.0
            seasonal[indices] = season_mean

        # Center the seasonal component (should sum to ~0 for additive)
        if model == "additive":
            seasonal -= np.mean(seasonal[:period])

        # Step 4: Residual
        if model == "additive":
            residual = data - trend - seasonal
        else:
            residual = data / (
                np.where(trend != 0, trend, np.nan)
                * np.where(seasonal != 0, seasonal, np.nan)
            )

        result_df = DataFrame(
            {
                "observed": data,
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual,
            },
            index=idx,
        )

        fig_ax: tuple[Figure, Axes] | None = None
        if plot:
            fig_ax = _plot_decomposition(result_df, column, model, **kwargs)

        return result_df, fig_ax

    def smooth(
        self,
        method: str = "moving_average",
        window: int = 10,
        **params: Any,
    ) -> Any:
        """Apply smoothing to the time series.

        Args:
            method: Smoothing method.
                - "moving_average": Centered moving average via pandas rolling.
                - "exponential": Exponential weighted moving average via pandas ewm.
                - "savgol": Savitzky-Golay filter (preserves peaks better than MA).
                  Extra param: polyorder (default 2).
            window: Window size. For savgol, must be odd. Default 10.
            **params: Method-specific parameters (e.g., polyorder for savgol).

        Returns:
            TimeSeries: New TimeSeries with smoothed values. Same index as original.

        Examples:
            Moving average smoothing (NaN at edges where window is incomplete):

            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(100))
            >>> smoothed = ts.smooth(method="moving_average", window=10)
            >>> smoothed.shape
            (100, 1)
            >>> round(float(smoothed.dropna().iloc[0, 0]), 4)
            0.4481

            Exponential weighted moving average (no NaN values):

            >>> np.random.seed(42)
            >>> ts2 = TimeSeries(np.random.randn(100))
            >>> smoothed2 = ts2.smooth(method="exponential", window=10)
            >>> round(float(smoothed2.iloc[0, 0]), 4)
            0.4967
            >>> round(float(smoothed2.iloc[2, 0]), 4)
            0.3486

            Savitzky-Golay filter preserves peaks better (reduces std less aggressively):

            >>> np.random.seed(42)
            >>> ts3 = TimeSeries(np.random.randn(100))
            >>> smoothed3 = ts3.smooth(method="savgol", window=11, polyorder=2)
            >>> round(float(smoothed3.iloc[0, 0]), 4)
            0.2353
            >>> round(float(smoothed3.values.std() / ts3.values.std()), 4)
            0.4354
        """
        from statista.time_series import TimeSeries

        if method == "moving_average":
            smoothed_df = self.rolling(window, center=True).mean()
            result = TimeSeries(
                smoothed_df.values,
                index=self.index,
                columns=list(self.columns),
            )
        elif method == "exponential":
            smoothed_df = self.ewm(span=window).mean()
            result = TimeSeries(
                smoothed_df.values,
                index=self.index,
                columns=list(self.columns),
            )
        elif method == "savgol":
            polyorder = params.get("polyorder", 2)
            # savgol requires odd window
            win = window if window % 2 == 1 else window + 1
            result_data = np.empty_like(self.values, dtype=float)
            for i, col in enumerate(self.columns):
                data = self[col].values.astype(float)
                result_data[:, i] = savgol_filter(
                    data, window_length=win, polyorder=polyorder
                )
            result = TimeSeries(
                result_data,
                index=self.index,
                columns=list(self.columns),
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'moving_average', 'exponential', 'savgol'."
            )

        return result

    def envelope(
        self,
        window: int = 30,
        lower_pct: float = 5,
        upper_pct: float = 95,
        column: str = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the time series with rolling percentile envelope bands.

        Shows the natural variability range of the data over a rolling window.

        Args:
            window: Rolling window size. Default 30.
            lower_pct: Lower percentile for the band (0-100). Default 5.
            upper_pct: Upper percentile for the band (0-100). Default 95.
            column: Column to plot. If None, uses first column.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (Figure, Axes)

        Examples:
            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> ts = TimeSeries(np.random.randn(200))  # doctest: +SKIP
            >>> fig, ax = ts.envelope(window=20)  # doctest: +SKIP
        """
        if column is None:
            column = self.columns[0]

        series = self[column]
        rolling_lower = series.rolling(window).quantile(lower_pct / 100.0)
        rolling_upper = series.rolling(window).quantile(upper_pct / 100.0)
        rolling_median = series.rolling(window).median()

        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)

        ax.plot(
            series.index,
            series.values,
            color="steelblue",
            alpha=0.4,
            linewidth=0.5,
            label="Data",
        )
        ax.plot(
            series.index,
            rolling_median.values,
            color="darkblue",
            linewidth=1.2,
            label="Rolling median",
        )
        ax.fill_between(
            series.index,
            rolling_lower.values,
            rolling_upper.values,
            color="lightblue",
            alpha=0.4,
            label=f"{lower_pct}%-{upper_pct}% envelope",
        )

        if "title" not in kwargs:
            kwargs["title"] = f"Envelope — {column}"
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = "Index"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "Value"

        ax = self._adjust_axes_labels(ax, **kwargs)
        plt.show()
        return fig, ax


def _centered_moving_average(data: np.ndarray, period: int) -> np.ndarray:
    """Compute centered moving average for trend extraction.

    For even periods, uses a 2xperiod moving average (convolution approach)
    to produce a properly centered result.

    Args:
        data: 1D array.
        period: Window size.

    Returns:
        Array of same length as data, with NaN at boundaries.
    """
    n = len(data)
    trend = np.full(n, np.nan)

    if period % 2 == 1:
        # Odd period: simple centered MA
        half = period // 2
        for i in range(half, n - half):
            trend[i] = np.mean(data[i - half : i + half + 1])
    else:
        # Even period: 2x moving average to center
        half = period // 2
        # First pass: period-wide MA
        ma1 = np.full(n, np.nan)
        for i in range(half, n - half + 1):
            if i - half >= 0 and i + half <= n:
                ma1[i] = np.mean(data[i - half : i + half])
        # Second pass: 2-wide MA to center
        for i in range(1, n):
            if not np.isnan(ma1[i]) and not np.isnan(ma1[i - 1]):
                trend[i] = (ma1[i] + ma1[i - 1]) / 2.0

    return trend


def _plot_decomposition(
    result_df: DataFrame,
    column: str,
    model: str,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot 4-panel decomposition (observed, trend, seasonal, residual)."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    components = ["observed", "trend", "seasonal", "residual"]
    colors = ["steelblue", "darkred", "green", "gray"]

    for ax, comp, color in zip(axes, components, colors):
        ax.plot(result_df.index, result_df[comp].values, color=color, linewidth=1)
        ax.set_ylabel(comp.capitalize(), fontsize=10)
        ax.grid(True, linestyle="-.", linewidth=0.3)
        if comp == "residual":
            ax.axhline(0, color="black", linewidth=0.5)

    axes[0].set_title(
        f"Classical Decomposition ({model}) — {column}", fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    plt.show()
    return fig, axes[-1]
