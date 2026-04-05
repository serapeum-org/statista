"""Trend detection mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.signal import detrend as scipy_detrend
from scipy.stats import norm, theilslopes

from statista.time_series._correlation import _compute_acf

if TYPE_CHECKING:
    from pandas import Index


class TrendMixin:
    """Mixin providing trend detection methods for TimeSeries.

    This mixin is designed to be composed with ``TimeSeriesBase`` (a ``pandas.DataFrame`` subclass).
    """

    if TYPE_CHECKING:
        columns: Index
        index: Index
        values: np.ndarray

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

    def mann_kendall(
        self,
        alpha: float = 0.05,
        method: str = "original",
        lag: int = None,
        column: str = None,
    ) -> DataFrame:
        """Mann-Kendall trend test.

        Tests the null hypothesis of no monotonic trend. Supports multiple variants
        to handle autocorrelated data.

        Args:
            alpha: Significance level. Default 0.05.
            method: Test variant.
                - "original": Standard MK (assumes serial independence).
                - "hamed_rao": Variance correction for autocorrelation (Hamed & Rao, 1998).
                  **Recommended for environmental data** where autocorrelation inflates significance.
                - "yue_wang": Alternative autocorrelation correction (Yue & Wang, 2004).
                - "pre_whitening": Remove lag-1 autocorrelation before testing.
                - "trend_free_pre_whitening": Remove trend, pre-whiten, re-add trend, then test.
            lag: Maximum lag for autocorrelation correction (Hamed-Rao / Yue-Wang).
                If None, uses n//2 - 1.
            column: Column to test. If None, tests all columns.

        Returns:
            pandas.DataFrame: One row per column with: trend, h, p_value, z, tau, s, var_s,
                slope, intercept.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.arange(50, dtype=float) + np.random.randn(50) * 3
            >>> ts = TimeSeries(data)
            >>> result = ts.mann_kendall()
            >>> result.loc["Series1", "trend"]
            'increasing'

            ```

        References:
            Mann, H.B. (1945). Nonparametric tests against trend. Econometrica, 13(3), 245-259.

            Hamed, K.H. and Rao, A.R. (1998). A modified Mann-Kendall trend test for autocorrelated
            data. Journal of Hydrology, 204(1-4), 182-196.

            Yue, S. and Wang, C. (2004). The Mann-Kendall test modified by effective sample size to
            detect trend in serially correlated hydrological series. Water Resources Management, 18, 201-218.
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values
            result = _mann_kendall_single(data, alpha=alpha, method=method, lag=lag)
            rows.append({"column": col, **result})

        result_df = DataFrame(rows).set_index("column")
        return result_df

    def sens_slope(
        self,
        alpha: float = 0.05,
        column: str = None,
    ) -> DataFrame:
        """Sen's slope estimator — robust non-parametric trend magnitude.

        Computes the median of all pairwise slopes. More robust to outliers than OLS.
        Always pair with ``mann_kendall()`` — Sen's slope gives the magnitude, MK gives
        the significance.

        Uses ``scipy.stats.theilslopes`` internally.

        Args:
            alpha: Significance level for confidence interval. Default 0.05.
            column: Column to analyze. If None, analyzes all columns.

        Returns:
            pandas.DataFrame: One row per column with: slope, intercept, slope_lower_ci,
                slope_upper_ci.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.arange(50, dtype=float) * 2 + np.random.randn(50) * 3
            >>> ts = TimeSeries(data)
            >>> result = ts.sens_slope()
            >>> abs(result.loc["Series1", "slope"] - 2.0) < 0.5
            True

            ```

        References:
            Sen, P.K. (1968). Estimates of the regression coefficient based on Kendall's tau.
            JASA, 63(324), 1379-1389.
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values
            x = np.arange(len(data))
            slope, intercept, low_slope, high_slope = theilslopes(data, x, alpha=alpha)
            rows.append(
                {
                    "column": col,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "slope_lower_ci": float(low_slope),
                    "slope_upper_ci": float(high_slope),
                }
            )

        result_df = DataFrame(rows).set_index("column")
        return result_df

    def detrend(self, method: str = "linear", order: int = 1) -> Any:
        """Remove trend from the time series.

        Args:
            method: Detrending method.
                - "linear": Remove linear trend via scipy.signal.detrend.
                - "constant": Subtract the mean.
                - "polynomial": Remove polynomial trend of given order.
                - "sens": Remove trend using Sen's slope (robust to outliers).
            order: Polynomial order (only used when method="polynomial"). Default 1.

        Returns:
            TimeSeries: New TimeSeries with the trend removed. Same index as original.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.arange(100, dtype=float) + np.random.randn(100) * 5
            >>> ts = TimeSeries(data)
            >>> detrended = ts.detrend(method="linear")
            >>> abs(detrended.values.mean()) < 5.0
            True

            ```
        """
        from statista.time_series import TimeSeries

        result_data = np.empty_like(self.values, dtype=float)

        for i, col in enumerate(self.columns):
            data = self[col].values.astype(float)

            if method == "linear":
                result_data[:, i] = scipy_detrend(data, type="linear")
            elif method == "constant":
                result_data[:, i] = scipy_detrend(data, type="constant")
            elif method == "polynomial":
                x = np.arange(len(data), dtype=float)
                coeffs = np.polyfit(x, data, order)
                trend = np.polyval(coeffs, x)
                result_data[:, i] = data - trend
            elif method == "sens":
                x = np.arange(len(data), dtype=float)
                slope, intercept, _, _ = theilslopes(data, x)
                trend = intercept + slope * x
                result_data[:, i] = data - trend
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Choose from 'linear', 'constant', 'polynomial', 'sens'."
                )

        result = TimeSeries(result_data, index=self.index, columns=list(self.columns))
        return result

    def innovative_trend_analysis(
        self,
        column: str = None,
        **kwargs: Any,
    ) -> Tuple[DataFrame, Tuple[Figure, Axes]]:
        """Innovative Trend Analysis (ITA) — Sen (2012) method.

        Splits the sorted data into two halves and plots the first half (x-axis) against
        the second half (y-axis). Points above the 1:1 line indicate an increasing trend,
        points below indicate a decreasing trend.

        Args:
            column: Column to analyze. If None, uses first column.
            **kwargs: Passed to ``_adjust_axes_labels`` (title, xlabel, ylabel, etc.).

        Returns:
            tuple: (results_df, (fig, ax)).
                results_df has columns: column, trend_indicator (positive = increasing).

        Examples:
            ```python
            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> ts = TimeSeries(np.arange(100, dtype=float))  # doctest: +SKIP
            >>> result_df, (fig, ax) = ts.innovative_trend_analysis()  # doctest: +SKIP

            ```

        References:
            Sen, Z. (2012). Innovative Trend Analysis Methodology. Journal of Hydrologic
            Engineering, 17(9), 1042-1046.
        """
        if column is None:
            column = self.columns[0]

        data = np.sort(self[column].dropna().values)
        n = len(data)
        mid = n // 2

        first_half = data[:mid]
        second_half = data[mid : 2 * mid]

        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)

        ax.scatter(
            first_half,
            second_half,
            alpha=0.5,
            s=15,
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )

        # 1:1 line
        min_val = min(first_half.min(), second_half.min())
        max_val = max(first_half.max(), second_half.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "k-", linewidth=1, label="1:1 line"
        )

        # +/- 10% envelope
        range_val = max_val - min_val
        offset = 0.10 * range_val
        ax.plot(
            [min_val, max_val],
            [min_val + offset, max_val + offset],
            "r--",
            linewidth=0.7,
            label="+10%",
        )
        ax.plot(
            [min_val, max_val],
            [min_val - offset, max_val - offset],
            "b--",
            linewidth=0.7,
            label="-10%",
        )

        # Trend indicator: mean deviation from 1:1 line
        trend_indicator = float(np.mean(second_half - first_half))

        ax.annotate(
            f"Trend indicator: {trend_indicator:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            va="top",
        )

        if "title" not in kwargs:
            kwargs["title"] = f"Innovative Trend Analysis — {column}"
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = "First half (sorted)"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "Second half (sorted)"

        ax = self._adjust_axes_labels(ax, **kwargs)
        plt.show()

        result_df = DataFrame(
            [
                {
                    "column": column,
                    "trend_indicator": trend_indicator,
                }
            ]
        ).set_index("column")

        return result_df, (fig, ax)


# ---------------------------------------------------------------------------
# Mann-Kendall core implementation
# ---------------------------------------------------------------------------


def _mann_kendall_single(
    data: np.ndarray,
    alpha: float = 0.05,
    method: str = "original",
    lag: int = None,
) -> dict:
    """Run Mann-Kendall trend test on a single series."""
    n = len(data)
    if n < 3:
        raise ValueError(
            f"Mann-Kendall test requires at least 3 observations, got {n}."
        )

    # S statistic
    s = _mk_score(data, n)

    # Tie correction for variance
    var_s = _mk_variance(data, n)

    # Autocorrelation correction
    if method == "hamed_rao":
        var_s = _hamed_rao_correction(data, n, s, var_s, lag)
    elif method == "yue_wang":
        var_s = _yue_wang_correction(data, n, s, var_s, lag)
    elif method == "pre_whitening":
        data = _pre_whiten(data)
        n = len(data)
        s = _mk_score(data, n)
        var_s = _mk_variance(data, n)
    elif method == "trend_free_pre_whitening":
        data = _trend_free_pre_whiten(data)
        n = len(data)
        s = _mk_score(data, n)
        var_s = _mk_variance(data, n)
    elif method != "original":
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'original', 'hamed_rao', "
            f"'yue_wang', 'pre_whitening', 'trend_free_pre_whitening'."
        )

    # Z statistic with continuity correction
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))

    # Trend direction
    h = abs(z) > norm.ppf(1 - alpha / 2)
    if z > 0 and h:
        trend = "increasing"
    elif z < 0 and h:
        trend = "decreasing"
    else:
        trend = "no trend"

    # Kendall's tau
    tau = s / (n * (n - 1) / 2)

    # Sen's slope
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((data[j] - data[i]) / (j - i))
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(data - slope * np.arange(n)))

    result = {
        "trend": trend,
        "h": bool(h),
        "p_value": float(p_value),
        "z": float(z),
        "tau": float(tau),
        "s": float(s),
        "var_s": float(var_s),
        "slope": slope,
        "intercept": intercept,
    }
    return result


def _mk_score(data: np.ndarray, n: int) -> float:
    """Compute Mann-Kendall S statistic."""
    s = 0.0
    for k in range(n - 1):
        diffs = data[k + 1 : n] - data[k]
        s += np.sum(np.sign(diffs))
    return float(s)


def _mk_variance(data: np.ndarray, n: int) -> float:
    """Compute variance of S with tie correction."""
    unique, counts = np.unique(data, return_counts=True)
    tie_groups = counts[counts > 1]

    var_s_int = n * (n - 1) * (2 * n + 5)
    for tp in tie_groups:
        var_s_int -= tp * (tp - 1) * (2 * tp + 5)

    return float(var_s_int) / 18.0


def _hamed_rao_correction(
    data: np.ndarray, n: int, s: float, var_s: float, lag: int = None
) -> float:
    """Hamed & Rao (1998) variance correction for autocorrelation.

    Detrends with Sen slope, computes ACF on ranked residuals, applies
    variance correction using only significant autocorrelation coefficients.
    """
    if lag is None:
        lag = n // 2 - 1

    # Detrend with Sen slope
    x = np.arange(n, dtype=float)
    slope, intercept, _, _ = theilslopes(data, x)
    residuals = data - (intercept + slope * x)

    # Rank the residuals
    from scipy.stats import rankdata

    ranked = rankdata(residuals)

    # ACF of ranked residuals
    acf_vals = _compute_acf(ranked, nlags=lag, fft=True)

    # Significance threshold for autocorrelation
    ci = 1.96 / np.sqrt(n)

    # Correction factor: n/s*
    correction = 0.0
    for i in range(1, lag + 1):
        if abs(acf_vals[i]) > ci:
            correction += (n - i) * (n - i - 1) * (n - i - 2) * acf_vals[i]

    ns_ratio = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * correction
    corrected_var = var_s * ns_ratio

    return max(float(corrected_var), 1.0)


def _yue_wang_correction(
    data: np.ndarray, n: int, s: float, var_s: float, lag: int = None
) -> float:
    """Yue & Wang (2004) variance correction for autocorrelation.

    Unlike Hamed-Rao, uses ACF on raw detrended values (not ranked) and
    includes ALL lags without significance thresholding, per the original paper.
    """
    if lag is None:
        lag = n // 2 - 1

    # Detrend with Sen slope
    x = np.arange(n, dtype=float)
    slope, intercept, _, _ = theilslopes(data, x)
    residuals = data - (intercept + slope * x)

    # ACF of raw residuals
    acf_vals = _compute_acf(residuals, nlags=lag, fft=True)

    # Correction factor — no significance thresholding per Yue & Wang (2004)
    correction = 0.0
    for i in range(1, lag + 1):
        correction += (n - i) * acf_vals[i]

    ns_ratio = 1.0 + (2.0 / n) * correction
    corrected_var = var_s * ns_ratio

    return max(float(corrected_var), 1.0)


def _pre_whiten(data: np.ndarray) -> np.ndarray:
    """Remove lag-1 autocorrelation: x_new[t] = x[t+1] - r1 * x[t]."""
    acf_vals = _compute_acf(data, nlags=1, fft=True)
    r1 = acf_vals[1]
    result = data[1:] - r1 * data[:-1]
    return result


def _trend_free_pre_whiten(data: np.ndarray) -> np.ndarray:
    """Trend-free pre-whitening: remove trend, pre-whiten, re-add trend."""
    n = len(data)
    x = np.arange(n, dtype=float)
    slope, intercept, _, _ = theilslopes(data, x)
    trend = intercept + slope * x

    # Remove trend
    detrended = data - trend

    # Pre-whiten the detrended series
    acf_vals = _compute_acf(detrended, nlags=1, fft=True)
    r1 = acf_vals[1]
    pw = detrended[1:] - r1 * detrended[:-1]

    # Re-add trend (shortened by 1 due to differencing)
    result = pw + trend[1:]
    return result
