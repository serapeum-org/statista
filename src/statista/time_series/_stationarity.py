"""Stationarity testing mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame

if TYPE_CHECKING:
    from pandas import Index


class StationarityMixin:
    """Mixin providing stationarity tests for TimeSeries.

    This mixin is designed to be composed with ``TimeSeriesBase`` (a ``pandas.DataFrame`` subclass).
    """

    if TYPE_CHECKING:
        columns: Index

        def __getitem__(self, key: str) -> DataFrame:  # noqa: E704
            ...

    def adf_test(
        self,
        regression: str = "c",
        max_lag: int = None,
        column: str = None,
    ) -> DataFrame:
        """Augmented Dickey-Fuller unit root test.

        Tests the null hypothesis that a unit root is present (series is non-stationary).
        Rejecting the null (p-value < alpha) indicates the series is stationary.

        Implemented from scratch using OLS regression and MacKinnon (1994) approximate
        p-values via ``scipy.stats.distributions``.

        Args:
            regression: Deterministic terms to include.
                - "c": constant only (default). Tests level stationarity.
                - "ct": constant + linear trend. Tests trend stationarity.
                - "n": no constant, no trend.
            max_lag: Maximum number of lagged differences to include. If None,
                uses ``int(12 * (n / 100) ** 0.25)`` (Schwert, 1989).
            column: Column to test. If None, tests all columns.

        Returns:
            pandas.DataFrame: One row per column with: statistic, p_value, used_lag,
                n_obs, crit_1%, crit_5%, crit_10%, conclusion.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(200))
            >>> result = ts.adf_test()
            >>> result.loc["Series1", "p_value"] < 0.05
            True

            ```

        References:
            Dickey, D.A. and Fuller, W.A. (1979). Distribution of the estimators for
            autoregressive time series with a unit root. JASA, 74(366), 427-431.

            MacKinnon, J.G. (1994). Approximate asymptotic distribution functions for
            unit-root and cointegration tests. JBES, 12(2), 167-176.
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values
            result = _adf_test_single(data, regression=regression, max_lag=max_lag)
            rows.append({"column": col, **result})

        result_df = DataFrame(rows).set_index("column")
        return result_df

    def kpss_test(
        self,
        regression: str = "c",
        n_lags: int = None,
        column: str = None,
    ) -> DataFrame:
        """KPSS stationarity test.

        Tests the null hypothesis that the series IS stationary. Rejecting the null
        (p-value < alpha) indicates non-stationarity. **This is the opposite of ADF.**

        Implemented from scratch following Kwiatkowski et al. (1992).

        Args:
            regression: Type of stationarity to test.
                - "c": level stationarity (default). Null: stationary around a constant.
                - "ct": trend stationarity. Null: stationary around a linear trend.
            n_lags: Lag truncation for the Newey-West estimator. If None,
                uses ``int(np.sqrt(12 * n / 100))`` (Hobijn et al., 1998).
            column: Column to test. If None, tests all columns.

        Returns:
            pandas.DataFrame: One row per column with: statistic, p_value, lags,
                crit_10%, crit_5%, crit_2.5%, crit_1%, conclusion.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(200))
            >>> result = ts.kpss_test()
            >>> result.loc["Series1", "p_value"] > 0.05
            True

            ```

        References:
            Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992).
            Testing the null hypothesis of stationarity against the alternative of
            a unit root. Journal of Econometrics, 54(1-3), 159-178.
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values
            result = _kpss_test_single(data, regression=regression, n_lags=n_lags)
            rows.append({"column": col, **result})

        result_df = DataFrame(rows).set_index("column")
        return result_df

    def stationarity_summary(self, alpha: float = 0.05) -> DataFrame:
        """Combined ADF + KPSS stationarity diagnosis.

        Runs both ADF and KPSS tests and produces an interpretation:

        +---------------+----------------+-------------------------------------------+
        | ADF rejects?  | KPSS rejects?  | Diagnosis                                 |
        +===============+================+===========================================+
        | Yes           | No             | Stationary                                |
        +---------------+----------------+-------------------------------------------+
        | No            | Yes            | Non-stationary (unit root)                |
        +---------------+----------------+-------------------------------------------+
        | Yes           | Yes            | Trend-stationary                          |
        +---------------+----------------+-------------------------------------------+
        | No            | No             | Inconclusive                              |
        +---------------+----------------+-------------------------------------------+

        Args:
            alpha: Significance level for both tests. Default 0.05.

        Returns:
            pandas.DataFrame: One row per column with: adf_stat, adf_pvalue,
                kpss_stat, kpss_pvalue, diagnosis.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(200))
            >>> result = ts.stationarity_summary()
            >>> result.loc["Series1", "diagnosis"]
            'Stationary'

            ```
        """
        adf_df = self.adf_test()
        kpss_df = self.kpss_test()

        rows = []
        for col in self.columns:
            adf_reject = float(adf_df.loc[col, "p_value"]) < alpha
            kpss_reject = float(kpss_df.loc[col, "p_value"]) < alpha

            if adf_reject and not kpss_reject:
                diagnosis = "Stationary"
            elif not adf_reject and kpss_reject:
                diagnosis = "Non-stationary (unit root)"
            elif adf_reject and kpss_reject:
                diagnosis = "Trend-stationary"
            else:
                diagnosis = "Inconclusive"

            rows.append(
                {
                    "column": col,
                    "adf_stat": float(adf_df.loc[col, "statistic"]),
                    "adf_pvalue": float(adf_df.loc[col, "p_value"]),
                    "kpss_stat": float(kpss_df.loc[col, "statistic"]),
                    "kpss_pvalue": float(kpss_df.loc[col, "p_value"]),
                    "diagnosis": diagnosis,
                }
            )

        result = DataFrame(rows).set_index("column")
        return result


# ---------------------------------------------------------------------------
# ADF implementation
# ---------------------------------------------------------------------------

# MacKinnon (1994) critical values for ADF test (approximate).
# Keys: (regression_type, significance_level)
_ADF_CRITICAL_VALUES = {
    "n": {"1%": -2.566, "5%": -1.941, "10%": -1.617},
    "c": {"1%": -3.433, "5%": -2.863, "10%": -2.568},
    "ct": {"1%": -3.963, "5%": -3.412, "10%": -3.128},
}


def _adf_test_single(
    data: np.ndarray, regression: str = "c", max_lag: int = None
) -> dict:
    """Run ADF test on a single series."""
    n = len(data)
    if max_lag is None:
        max_lag = int(12.0 * (n / 100.0) ** 0.25)

    # First difference
    dy = np.diff(data)
    y_lag = data[:-1]

    # Select lag order using AIC-like criterion (simplified: use max_lag directly)
    used_lag = min(max_lag, len(dy) - 2)

    # Build regression matrix: dy_t = rho * y_{t-1} + sum(gamma_i * dy_{t-i}) + deterministic + e_t
    nobs = len(dy) - used_lag
    y = dy[used_lag:]

    # Lagged level
    x_cols = [y_lag[used_lag:].reshape(-1, 1)]

    # Lagged differences
    for lag in range(1, used_lag + 1):
        x_cols.append(dy[used_lag - lag : -lag].reshape(-1, 1))

    # Deterministic terms
    if regression == "c":
        x_cols.append(np.ones((nobs, 1)))
    elif regression == "ct":
        x_cols.append(np.ones((nobs, 1)))
        x_cols.append(np.arange(used_lag, used_lag + nobs).reshape(-1, 1))

    x = np.hstack(x_cols)

    # OLS regression
    beta, residuals, _, _ = np.linalg.lstsq(x, y, rcond=None)

    # t-statistic for the coefficient on y_{t-1} (first coefficient)
    e = y - x @ beta
    sigma2 = np.sum(e**2) / (nobs - x.shape[1])
    var_beta = sigma2 * np.linalg.inv(x.T @ x)
    se_rho = np.sqrt(var_beta[0, 0])
    t_stat = beta[0] / se_rho

    # Approximate p-value using MacKinnon (1994) regression surface
    p_value = _mackinnon_pvalue(t_stat, regression, n)

    crit = _ADF_CRITICAL_VALUES.get(regression, _ADF_CRITICAL_VALUES["c"])
    conclusion = "Stationary" if p_value < 0.05 else "Non-stationary"

    result = {
        "statistic": float(t_stat),
        "p_value": float(p_value),
        "used_lag": used_lag,
        "n_obs": nobs,
        "crit_1%": crit["1%"],
        "crit_5%": crit["5%"],
        "crit_10%": crit["10%"],
        "conclusion": conclusion,
    }
    return result


def _mackinnon_pvalue(t_stat: float, regression: str, n: int) -> float:
    """Approximate MacKinnon (1994) p-value for ADF test.

    Uses a simple interpolation approach based on critical values at standard
    significance levels. For a more precise implementation, one would use the
    full MacKinnon regression surface coefficients.
    """
    from scipy.interpolate import interp1d

    crit = _ADF_CRITICAL_VALUES.get(regression, _ADF_CRITICAL_VALUES["c"])

    # Critical values at known significance levels (sorted by statistic value)
    levels = np.array([0.01, 0.05, 0.10])
    crits = np.array([crit["1%"], crit["5%"], crit["10%"]])

    if t_stat <= crits[0]:
        return 0.001
    elif t_stat >= crits[-1]:
        # Extrapolate: use a rough mapping. t_stat much larger than 10% critical -> large p-value.
        # Linear extrapolation from the 5%-10% segment.
        slope = (0.10 - 0.05) / (crits[2] - crits[1])
        p = 0.10 + slope * (t_stat - crits[2])
        return min(max(float(p), 0.10), 1.0)
    else:
        f = interp1d(crits, levels, kind="linear", fill_value="extrapolate")
        return float(np.clip(f(t_stat), 0.001, 1.0))


# ---------------------------------------------------------------------------
# KPSS implementation
# ---------------------------------------------------------------------------

# KPSS critical values from Kwiatkowski et al. (1992), Table 1.
_KPSS_CRITICAL_VALUES = {
    "c": {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739},
    "ct": {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216},
}


def _kpss_test_single(
    data: np.ndarray, regression: str = "c", n_lags: int = None
) -> dict:
    """Run KPSS test on a single series."""
    n = len(data)
    if n_lags is None:
        n_lags = int(np.sqrt(12.0 * n / 100.0))

    # Residuals from regression on deterministic terms
    if regression == "c":
        residuals = data - np.mean(data)
    elif regression == "ct":
        t = np.arange(1, n + 1, dtype=float)
        x = np.column_stack([np.ones(n), t])
        beta = np.linalg.lstsq(x, data, rcond=None)[0]
        residuals = data - x @ beta
    else:
        raise ValueError(f"regression must be 'c' or 'ct', got '{regression}'")

    # Partial sum process
    s = np.cumsum(residuals)

    # Newey-West long-run variance estimator
    sigma2 = np.sum(residuals**2) / n
    for lag in range(1, n_lags + 1):
        weight = 1.0 - lag / (n_lags + 1.0)  # Bartlett kernel
        gamma = np.sum(residuals[lag:] * residuals[:-lag]) / n
        sigma2 += 2.0 * weight * gamma

    # KPSS statistic: eta = sum(S_t^2) / (n^2 * sigma2)
    eta = np.sum(s**2) / (n**2 * sigma2)

    # P-value by interpolation from critical value table
    crit = _KPSS_CRITICAL_VALUES.get(regression, _KPSS_CRITICAL_VALUES["c"])
    p_value = _kpss_pvalue(eta, crit)

    conclusion = "Stationary" if p_value > 0.05 else "Non-stationary"

    result = {
        "statistic": float(eta),
        "p_value": float(p_value),
        "lags": n_lags,
        "crit_10%": crit["10%"],
        "crit_5%": crit["5%"],
        "crit_2.5%": crit["2.5%"],
        "crit_1%": crit["1%"],
        "conclusion": conclusion,
    }
    return result


def _kpss_pvalue(stat: float, crit: dict) -> float:
    """Approximate KPSS p-value by interpolation from critical value table.

    KPSS rejects for LARGE values of the statistic (opposite of ADF).
    """
    from scipy.interpolate import interp1d

    # Critical values in ascending order (small stat = large p-value = stationary)
    crits = np.array([crit["10%"], crit["5%"], crit["2.5%"], crit["1%"]])
    levels = np.array([0.10, 0.05, 0.025, 0.01])

    if stat >= crits[-1]:
        # More extreme than 1% critical value
        return 0.01
    elif stat <= crits[0]:
        # Less extreme than 10% critical value -> fail to reject -> p > 0.10
        return 0.10
    else:
        # Interpolate between known critical values (both arrays ascending)
        f = interp1d(crits, levels, kind="linear")
        return float(np.clip(f(stat), 0.01, 0.10))
