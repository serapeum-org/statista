"""Autocorrelation and dependence mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.stats import chi2, kendalltau, norm, pearsonr, spearmanr

if TYPE_CHECKING:
    from pandas import Index


class CorrelationMixin:
    """Mixin providing autocorrelation and dependence analysis for TimeSeries.

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

        def corr(
            self, method: str = "pearson", **kwargs: object
        ) -> DataFrame:  # noqa: E704
            ...

    def acf(
        self,
        nlags: int = 40,
        alpha: float = 0.05,
        fft: bool = True,
        column: str = None,
        plot: bool = True,
        **kwargs: Any,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Optional[Tuple[Figure, Axes]]]:
        """Compute and optionally plot the autocorrelation function.

        Args:
            nlags: Number of lags to compute. Default 40.
            alpha: Significance level for confidence bands. Default 0.05.
            fft: Use FFT for computation (faster for long series). Default True.
            column: Column name. If None and single-column, uses that column.
                For multi-column without column specified, computes per column.
            plot: Whether to produce a plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels`` (title, xlabel, ylabel, etc.).

        Returns:
            tuple: (acf_values, (fig, ax)) or (acf_values, None) if plot=False.
                For multi-column: acf_values is a dict mapping column names to arrays.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(200))
            >>> acf_vals, _ = ts.acf(nlags=10, plot=False)
            >>> abs(acf_vals[0] - 1.0) < 1e-10
            True

            ```
        """
        cols = _resolve_columns(self.columns, column)

        acf_results: Dict[str, np.ndarray] = {}
        for col in cols:
            data = self[col].dropna().values
            acf_results[col] = _compute_acf(data, nlags=nlags, fft=fft)

        fig_ax: Optional[Tuple[Figure, Axes]] = None
        if plot:
            fig_ax = _plot_acf_pacf(
                acf_results,
                len(self[cols[0]].dropna()),
                alpha,
                "ACF",
                self._get_ax_fig,
                **kwargs,
            )

        single_result: Union[np.ndarray, Dict[str, np.ndarray]] = (
            acf_results[cols[0]] if len(cols) == 1 else acf_results
        )
        return single_result, fig_ax

    def pacf(
        self,
        nlags: int = 40,
        alpha: float = 0.05,
        column: str = None,
        plot: bool = True,
        **kwargs: Any,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Optional[Tuple[Figure, Axes]]]:
        """Compute and optionally plot the partial autocorrelation function.

        Uses the Levinson-Durbin recursion to compute PACF from ACF values.

        Args:
            nlags: Number of lags to compute. Default 40.
            alpha: Significance level for confidence bands. Default 0.05.
            column: Column name. If None and single-column, uses that column.
            plot: Whether to produce a plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (pacf_values, (fig, ax)) or (pacf_values, None) if plot=False.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(200))
            >>> pacf_vals, _ = ts.pacf(nlags=10, plot=False)
            >>> abs(pacf_vals[0] - 1.0) < 1e-10
            True

            ```
        """
        cols = _resolve_columns(self.columns, column)

        pacf_results: Dict[str, np.ndarray] = {}
        for col in cols:
            data = self[col].dropna().values
            effective_nlags = min(nlags, len(data) // 2 - 1)
            acf_vals = _compute_acf(data, nlags=effective_nlags, fft=True)
            pacf_results[col] = _levinson_durbin_pacf(acf_vals)

        fig_ax: Optional[Tuple[Figure, Axes]] = None
        if plot:
            fig_ax = _plot_acf_pacf(
                pacf_results,
                len(self[cols[0]].dropna()),
                alpha,
                "PACF",
                self._get_ax_fig,
                **kwargs,
            )

        single_result: Union[np.ndarray, Dict[str, np.ndarray]] = (
            pacf_results[cols[0]] if len(cols) == 1 else pacf_results
        )
        return single_result, fig_ax

    def cross_correlation(
        self,
        col_x: str,
        col_y: str,
        nlags: int = 40,
        plot: bool = True,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[Tuple[Figure, Axes]]]:
        """Compute the cross-correlation function between two columns.

        Args:
            col_x: First column name.
            col_y: Second column name.
            nlags: Number of lags. Default 40.
            plot: Whether to produce a plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (ccf_values, (fig, ax)) or (ccf_values, None) if plot=False.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.column_stack([np.random.randn(100), np.random.randn(100)])
            >>> ts = TimeSeries(data, columns=["A", "B"])
            >>> ccf_vals, _ = ts.cross_correlation("A", "B", nlags=10, plot=False)
            >>> len(ccf_vals) > 0
            True

            ```
        """
        x = self[col_x].dropna().values
        y = self[col_y].dropna().values
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        ccf_vals = _compute_ccf(x, y, nlags=nlags)

        fig_ax: Optional[Tuple[Figure, Axes]] = None
        if plot:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            lags = np.arange(len(ccf_vals))
            ax.vlines(lags, 0, ccf_vals, colors="steelblue", linewidth=1.5)
            ax.scatter(lags, ccf_vals, color="steelblue", s=15, zorder=5)
            ax.axhline(0, color="black", linewidth=0.5)

            ci = 1.96 / np.sqrt(min_len)
            ax.axhline(ci, color="red", linestyle="--", linewidth=0.7, label="95% CI")
            ax.axhline(-ci, color="red", linestyle="--", linewidth=0.7)

            peak_lag = int(np.argmax(np.abs(ccf_vals)))
            ax.annotate(
                f"max |r| at lag {peak_lag}",
                xy=(peak_lag, ccf_vals[peak_lag]),
                fontsize=9,
                color="red",
            )

            if "title" not in kwargs:
                kwargs["title"] = f"Cross-Correlation: {col_x} vs {col_y}"
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Lag"
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = "CCF"

            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return ccf_vals, fig_ax

    def lag_plot(
        self,
        lag: int = 1,
        column: str = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """Scatter plot of x(t) vs x(t-lag) for visual serial dependence check.

        Args:
            lag: Lag to use. Default 1.
            column: Column name. If None, uses first column.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (Figure, Axes)

        Examples:
            ```python
            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> ts = TimeSeries(np.random.randn(100))  # doctest: +SKIP
            >>> fig, ax = ts.lag_plot(lag=1)  # doctest: +SKIP

            ```
        """
        if column is None:
            column = self.columns[0]

        data = self[column].dropna().values
        x = data[:-lag]
        y = data[lag:]

        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)

        ax.scatter(
            x, y, alpha=0.5, s=10, color="steelblue", edgecolor="white", linewidth=0.3
        )

        r = np.corrcoef(x, y)[0, 1]
        ax.annotate(
            f"r = {r:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=11,
            va="top",
        )

        if "title" not in kwargs:
            kwargs["title"] = f"Lag Plot (lag={lag})"
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = "x(t)"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = f"x(t+{lag})"

        ax = self._adjust_axes_labels(ax, **kwargs)
        plt.show()
        return fig, ax

    def correlation_matrix(
        self,
        method: str = "pearson",
        plot: bool = True,
        **kwargs: Any,
    ) -> Tuple[DataFrame, DataFrame, Optional[Tuple[Figure, Axes]]]:
        """Compute pairwise correlation matrix WITH p-values.

        Pandas ``.corr()`` provides no p-values. This method computes both the correlation
        coefficients and their corresponding p-values using ``scipy.stats``.

        Args:
            method: Correlation method — "pearson", "spearman", or "kendall". Default "pearson".
            plot: Whether to produce a heatmap. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (corr_df, pvalue_df, (fig, ax)) or (corr_df, pvalue_df, None) if plot=False.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(100, 3), columns=["A", "B", "C"])
            >>> corr, pvals, _ = ts.correlation_matrix(plot=False)
            >>> corr.loc["A", "A"]
            1.0

            ```
        """
        corr_funcs = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}
        if method not in corr_funcs:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'pearson', 'spearman', 'kendall'."
            )

        cols = list(self.columns)
        corr_df = DataFrame(np.nan, index=cols, columns=cols)
        pval_df = DataFrame(np.nan, index=cols, columns=cols)
        func = corr_funcs[method]

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i == j:
                    corr_df.loc[c1, c2] = 1.0
                    pval_df.loc[c1, c2] = 0.0
                elif i < j:
                    d1 = self[c1].dropna()
                    d2 = self[c2].dropna()
                    common = d1.index.intersection(d2.index)
                    r, p = func(d1.loc[common], d2.loc[common])
                    corr_df.loc[c1, c2] = r
                    corr_df.loc[c2, c1] = r
                    pval_df.loc[c1, c2] = p
                    pval_df.loc[c2, c1] = p

        fig_ax: Optional[Tuple[Figure, Axes]] = None
        if plot and len(cols) >= 2:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            corr_arr = corr_df.values.astype(float)
            im = ax.imshow(corr_arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            plt.colorbar(im, ax=ax, shrink=0.8)

            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_yticklabels(cols)

            for ii in range(len(cols)):
                for jj in range(len(cols)):
                    r_val = corr_arr[ii, jj]
                    p_val = float(pval_df.iloc[ii, jj])
                    sig = (
                        "***"
                        if p_val < 0.001
                        else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    )
                    ax.text(
                        jj,
                        ii,
                        f"{r_val:.2f}{sig}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

            if "title" not in kwargs:
                kwargs["title"] = f"Correlation Matrix ({method})"
            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return corr_df, pval_df, fig_ax

    def ljung_box(
        self,
        lags: int = 10,
        column: str = None,
    ) -> DataFrame:
        """Ljung-Box test for autocorrelation (white noise test).

        Tests whether autocorrelations of a series are significantly different from zero.
        Implemented from scratch using numpy/scipy (no statsmodels dependency).

        Args:
            lags: Number of lags to test. Default 10.
            column: Column to test. If None, tests all columns and stacks results.

        Returns:
            pandas.DataFrame: With columns ``lb_stat`` and ``lb_pvalue`` for each lag.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(200))
            >>> result = ts.ljung_box(lags=5)
            >>> "lb_stat" in result.columns
            True

            ```

        References:
            Ljung, G. M. and Box, G. E. P. (1978). On a measure of lack of fit in time series models.
            Biometrika, 65(2), 297-303.
        """
        cols = [column] if column is not None else list(self.columns)

        frames = []
        for col in cols:
            data = self[col].dropna().values
            n = len(data)
            acf_vals = _compute_acf(data, nlags=lags, fft=True)
            # acf_vals[0] = 1.0 (lag 0), we need lags 1..lags
            rho = acf_vals[1:]

            lb_stats = np.zeros(lags)
            lb_pvalues = np.zeros(lags)
            for k in range(lags):
                q = n * (n + 2) * np.sum(rho[: k + 1] ** 2 / (n - np.arange(1, k + 2)))
                lb_stats[k] = q
                lb_pvalues[k] = 1.0 - chi2.cdf(q, df=k + 1)

            result = DataFrame(
                {"lb_stat": lb_stats, "lb_pvalue": lb_pvalues},
                index=np.arange(1, lags + 1),
            )
            if len(cols) > 1:
                result.insert(0, "column", col)
            frames.append(result)

        import pandas as pd

        combined = (
            frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        )
        return combined


def _resolve_columns(columns: Any, column: Optional[str]) -> list:
    """Resolve which columns to operate on."""
    if column is not None:
        return [column]
    if len(columns) == 1:
        return [columns[0]]
    return list(columns)


def _compute_acf(data: np.ndarray, nlags: int, fft: bool = True) -> np.ndarray:
    """Compute autocorrelation function values.

    Args:
        data: 1D array of time series values.
        nlags: Number of lags.
        fft: Use FFT convolution (faster for long series).

    Returns:
        Array of length nlags+1, starting with lag 0 (= 1.0).
    """
    n = len(data)
    nlags = min(nlags, n - 1)
    x = data - np.mean(data)

    if fft:
        fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        xf = np.fft.rfft(x, n=fft_size)
        acov = np.fft.irfft(xf * np.conj(xf), n=fft_size)[:n]
    else:
        acov = np.correlate(x, x, mode="full")[n - 1 :]

    # Guard for constant data (zero variance)
    if acov[0] == 0.0:
        result = np.zeros(nlags + 1)
        result[0] = 1.0
        return result

    result = acov[: nlags + 1] / acov[0]
    return result


def _compute_ccf(x: np.ndarray, y: np.ndarray, nlags: int) -> np.ndarray:
    """Compute cross-correlation function.

    CCF(k) = corr(x_t, y_{t+k}) for k = 0, 1, ..., nlags.
    """
    n = len(x)
    nlags = min(nlags, n - 1)
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))

    if denom == 0.0:
        return np.zeros(nlags + 1)

    ccf_vals = np.zeros(nlags + 1)
    for k in range(nlags + 1):
        ccf_vals[k] = np.sum(xm[: n - k] * ym[k:]) / denom

    return ccf_vals


def _levinson_durbin_pacf(acf_vals: np.ndarray) -> np.ndarray:
    """Compute PACF from ACF using Levinson-Durbin recursion.

    Args:
        acf_vals: ACF values starting from lag 0.

    Returns:
        PACF values of same length as acf_vals. pacf[0] = 1.0.
    """
    nlags = len(acf_vals) - 1
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0

    if nlags == 0:
        return pacf

    # Levinson-Durbin recursion
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = acf_vals[1]
    pacf[1] = acf_vals[1]

    for k in range(2, nlags + 1):
        num = acf_vals[k] - np.sum(phi[k - 1, 1:k] * acf_vals[k - 1 : 0 : -1])
        den = 1.0 - np.sum(phi[k - 1, 1:k] * acf_vals[1:k])
        if np.isclose(den, 0.0):
            break
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf[k] = phi[k, k]

    return pacf


def _plot_acf_pacf(
    results: Dict[str, np.ndarray],
    n: int,
    alpha: float,
    title_prefix: str,
    get_ax_fig_fn: Any,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Shared stem-plot renderer for ACF and PACF."""
    n_cols = len(results)
    ci = norm.ppf(1 - alpha / 2) / np.sqrt(n)

    if n_cols == 1:
        fig, ax = get_ax_fig_fn(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)
        col = list(results.keys())[0]
        _draw_acf_stem(ax, results[col], ci, title_prefix)
    else:
        fig, axes = plt.subplots(n_cols, 1, figsize=(8, 3 * n_cols), sharex=True)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)
        for ax_i, col in zip(axes, results.keys()):
            _draw_acf_stem(ax_i, results[col], ci, f"{title_prefix} — {col}")
        ax = axes[-1]

    plt.tight_layout()
    plt.show()
    return fig, ax


def _draw_acf_stem(ax: Axes, values: np.ndarray, ci: float, title: str) -> None:
    """Draw a single ACF/PACF stem plot with confidence bands."""
    lags = np.arange(len(values))
    ax.vlines(lags, 0, values, colors="steelblue", linewidth=1.5)
    ax.scatter(lags, values, color="steelblue", s=15, zorder=5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(ci, color="red", linestyle="--", linewidth=0.7)
    ax.axhline(-ci, color="red", linestyle="--", linewidth=0.7)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
