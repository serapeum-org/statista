"""Change point detection mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.stats import rankdata

if TYPE_CHECKING:
    from pandas import Index


class ChangePoint:
    """Change point detection methods for TimeSeries.

    Implements Pettitt, SNHT, and Buishand range tests from scratch following
    the algorithms in pyhomogeneity (Moges et al., 2020). All tests detect a
    single change point in the mean of a time series.
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

    def pettitt_test(
        self,
        alpha: float = 0.05,
        column: str = None,
    ) -> DataFrame:
        """Pettitt non-parametric change point test.

        Tests the null hypothesis that the series is homogeneous (no change point).
        Uses rank-based U statistic. P-value computed analytically.

        Args:
            alpha: Significance level. Default 0.05.
            column: Column to test. If None, tests all columns.

        Returns:
            pandas.DataFrame: One row per column with: h, change_point_index,
                statistic, p_value, mean_before, mean_after, conclusion.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.concatenate([np.random.randn(50), np.random.randn(50) + 3])
            >>> ts = TimeSeries(data)
            >>> result = ts.pettitt_test()
            >>> 40 <= result.loc["Series1", "change_point_index"] <= 60
            True

            ```

        References:
            Pettitt, A.N. (1979). A non-parametric approach to the change-point problem.
            Applied Statistics, 28(2), 126-135.
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values
            n = len(data)
            if n < 5:
                raise ValueError(
                    f"Change point tests require at least 5 observations, got {n} for column '{col}'."
                )

            r = rankdata(data)
            s = np.cumsum(r)
            k_range = np.arange(1, n)
            u_values = 2 * s[:-1] - k_range * (n + 1)
            u_abs = np.abs(u_values)

            cp_pos = int(np.argmax(u_abs))
            k_stat = float(u_abs[cp_pos])

            # Analytical p-value approximation
            p_value = 2.0 * np.exp(-6.0 * k_stat**2 / (n**3 + n**2))
            p_value = min(float(p_value), 1.0)

            h = p_value < alpha
            mean_before = float(np.mean(data[: cp_pos + 1]))
            mean_after = float(np.mean(data[cp_pos + 1 :]))

            rows.append(
                {
                    "column": col,
                    "h": bool(h),
                    "change_point_index": cp_pos,
                    "statistic": k_stat,
                    "p_value": p_value,
                    "mean_before": mean_before,
                    "mean_after": mean_after,
                    "conclusion": "Inhomogeneous" if h else "Homogeneous",
                }
            )

        result_df = DataFrame(rows).set_index("column")
        return result_df

    def snht_test(
        self,
        alpha: float = 0.05,
        column: str = None,
    ) -> DataFrame:
        """Standard Normal Homogeneity Test (SNHT).

        Detects a shift in the mean by comparing standardized sub-period means.

        Args:
            alpha: Significance level. Default 0.05.
            column: Column to test. If None, tests all columns.

        Returns:
            pandas.DataFrame: One row per column with: h, change_point_index,
                statistic, p_value, mean_before, mean_after, conclusion.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.concatenate([np.random.randn(50), np.random.randn(50) + 3])
            >>> ts = TimeSeries(data)
            >>> result = ts.snht_test()
            >>> result.loc["Series1", "conclusion"]
            'Inhomogeneous'

            ```

        References:
            Alexandersson, H. (1986). A homogeneity test applied to precipitation data.
            Journal of Climatology, 6(6), 661-675.
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values
            n = len(data)
            if n < 5:
                raise ValueError(
                    f"Change point tests require at least 5 observations, got {n} for column '{col}'."
                )

            mean = np.mean(data)
            std = np.std(data, ddof=1)
            if std == 0:
                rows.append(_homogeneous_row(col, data))
                continue

            z = (data - mean) / std

            t_values = np.zeros(n - 1)
            cumsum_z = np.cumsum(z)
            total_z = cumsum_z[-1]
            for t in range(1, n):
                z1_mean = cumsum_z[t - 1] / t
                z2_mean = (total_z - cumsum_z[t - 1]) / (n - t)
                t_values[t - 1] = t * z1_mean**2 + (n - t) * z2_mean**2

            cp_pos = int(np.argmax(t_values))
            t0 = float(t_values[cp_pos])

            p_value = _snht_approx_pvalue(t0, n)

            h = p_value < alpha
            mean_before = float(np.mean(data[: cp_pos + 1]))
            mean_after = float(np.mean(data[cp_pos + 1 :]))

            rows.append(
                {
                    "column": col,
                    "h": bool(h),
                    "change_point_index": cp_pos,
                    "statistic": t0,
                    "p_value": p_value,
                    "mean_before": mean_before,
                    "mean_after": mean_after,
                    "conclusion": "Inhomogeneous" if h else "Homogeneous",
                }
            )

        result_df = DataFrame(rows).set_index("column")
        return result_df

    def buishand_range_test(
        self,
        alpha: float = 0.05,
        column: str = None,
    ) -> DataFrame:
        """Buishand range test for change point detection.

        Uses adjusted partial sums to detect a shift in the mean.

        Args:
            alpha: Significance level. Default 0.05.
            column: Column to test. If None, tests all columns.

        Returns:
            pandas.DataFrame: One row per column with: h, change_point_index,
                statistic, p_value, mean_before, mean_after, conclusion.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.concatenate([np.random.randn(50), np.random.randn(50) + 3])
            >>> ts = TimeSeries(data)
            >>> result = ts.buishand_range_test()
            >>> result.loc["Series1", "conclusion"]
            'Inhomogeneous'

            ```

        References:
            Buishand, T.A. (1982). Some methods for testing the homogeneity of rainfall
            records. Journal of Hydrology, 58(1-2), 11-27.
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values
            n = len(data)
            if n < 5:
                raise ValueError(
                    f"Change point tests require at least 5 observations, got {n} for column '{col}'."
                )

            std = np.std(data, ddof=1)
            if std == 0:
                rows.append(_homogeneous_row(col, data))
                continue

            # Adjusted partial sums
            s = np.cumsum(data - np.mean(data))
            s_std = s / std

            # Range statistic
            r_stat = (np.max(s_std) - np.min(s_std)) / np.sqrt(n)

            # Change point at max |S*|
            cp_pos = int(np.argmax(np.abs(s_std)))

            # Approximate p-value using critical values from Buishand (1982)
            # Table values for R/sqrt(n): 1% ~ 1.70, 5% ~ 1.42, 10% ~ 1.27
            p_value = _buishand_approx_pvalue(r_stat)

            h = p_value < alpha
            mean_before = float(np.mean(data[: cp_pos + 1]))
            mean_after = (
                float(np.mean(data[cp_pos + 1 :])) if cp_pos < n - 1 else mean_before
            )

            rows.append(
                {
                    "column": col,
                    "h": bool(h),
                    "change_point_index": cp_pos,
                    "statistic": float(r_stat),
                    "p_value": p_value,
                    "mean_before": mean_before,
                    "mean_after": mean_after,
                    "conclusion": "Inhomogeneous" if h else "Homogeneous",
                }
            )

        result_df = DataFrame(rows).set_index("column")
        return result_df

    def cusum(
        self,
        column: str = None,
        plot: bool = True,
        **kwargs: Any,
    ) -> Tuple[DataFrame, Optional[Tuple[Figure, Axes]]]:
        """Cumulative sum (CUSUM) of deviations from the mean.

        Visual method for detecting shifts. A sustained upward/downward drift
        indicates a change in the mean.

        Args:
            column: Column to analyze. If None, uses first column.
            plot: Whether to produce a CUSUM plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (cusum_df, (fig, ax)) or (cusum_df, None) if plot=False.
                cusum_df has the cumulative sums with same index as input.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(100))
            >>> cusum_df, _ = ts.cusum(plot=False)
            >>> cusum_df.shape[0] == 100
            True

            ```
        """
        if column is None:
            column = self.columns[0]

        data = self[column].dropna().values
        n = len(data)
        mean = np.mean(data)
        cusum_vals = np.cumsum(data - mean)

        cusum_df = DataFrame(
            {column: cusum_vals},
            index=self[column].dropna().index,
        )

        fig_ax: Optional[Tuple[Figure, Axes]] = None
        if plot:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            ax.plot(cusum_df.index, cusum_vals, color="steelblue", linewidth=1.2)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="-")

            # Confidence bounds
            std = np.std(data, ddof=1)
            ci = 1.96 * std * np.sqrt(np.arange(1, n + 1))
            ax.fill_between(
                cusum_df.index,
                -ci,
                ci,
                color="lightblue",
                alpha=0.3,
                label="95% CI",
            )

            # Mark the point of max absolute CUSUM
            cp_idx = int(np.argmax(np.abs(cusum_vals)))
            ax.axvline(
                cusum_df.index[cp_idx],
                color="red",
                linestyle="--",
                linewidth=0.8,
                label=f"Max |CUSUM| at {cusum_df.index[cp_idx]}",
            )

            if "title" not in kwargs:
                kwargs["title"] = f"CUSUM — {column}"
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Index"
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = "Cumulative deviation"

            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return cusum_df, fig_ax

    def homogeneity_summary(
        self,
        alpha: float = 0.05,
    ) -> DataFrame:
        """Run Pettitt + SNHT + Buishand and combine into a diagnosis.

        If 2 or more tests agree on a change point location (within +/-2 indices),
        the result is marked as "confirmed".

        Args:
            alpha: Significance level for all tests. Default 0.05.

        Returns:
            pandas.DataFrame: One row per column with test results and confirmation status.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> data = np.concatenate([np.random.randn(50), np.random.randn(50) + 3])
            >>> ts = TimeSeries(data)
            >>> result = ts.homogeneity_summary()
            >>> "confirmed" in result.columns
            True

            ```
        """
        pettitt = self.pettitt_test(alpha=alpha)
        snht = self.snht_test(alpha=alpha)
        buishand = self.buishand_range_test(alpha=alpha)

        rows = []
        for col in self.columns:
            p_cp = int(pettitt.loc[col, "change_point_index"])
            s_cp = int(snht.loc[col, "change_point_index"])
            b_cp = int(buishand.loc[col, "change_point_index"])

            p_h = bool(pettitt.loc[col, "h"])
            s_h = bool(snht.loc[col, "h"])
            b_h = bool(buishand.loc[col, "h"])

            # Count how many tests reject AND agree on location
            cps = []
            if p_h:
                cps.append(p_cp)
            if s_h:
                cps.append(s_cp)
            if b_h:
                cps.append(b_cp)

            confirmed = False
            if len(cps) >= 2:
                # Check if at least 2 change points are within +/-2 of each other
                for i in range(len(cps)):
                    for j in range(i + 1, len(cps)):
                        if abs(cps[i] - cps[j]) <= 2:
                            confirmed = True

            rows.append(
                {
                    "column": col,
                    "pettitt_cp": p_cp,
                    "pettitt_p": float(pettitt.loc[col, "p_value"]),
                    "snht_cp": s_cp,
                    "snht_p": float(snht.loc[col, "p_value"]),
                    "buishand_cp": b_cp,
                    "buishand_p": float(buishand.loc[col, "p_value"]),
                    "confirmed": confirmed,
                }
            )

        result_df = DataFrame(rows).set_index("column")
        return result_df


def _homogeneous_row(col: str, data: np.ndarray) -> dict:
    """Return a row for constant (zero-variance) data — always homogeneous."""
    return {
        "column": col,
        "h": False,
        "change_point_index": 0,
        "statistic": 0.0,
        "p_value": 1.0,
        "mean_before": float(np.mean(data)),
        "mean_after": float(np.mean(data)),
        "conclusion": "Homogeneous",
    }


def _snht_approx_pvalue(t0: float, n: int) -> float:
    """Approximate p-value for SNHT statistic.

    Uses a rough interpolation from published critical values.
    For n >= 50: 1% ~ 10.45+, 5% ~ 7.65+, 10% ~ 6.25+
    """
    from scipy.interpolate import interp1d

    crits = np.array([6.25, 7.65, 10.45])
    levels = np.array([0.10, 0.05, 0.01])

    if t0 >= crits[-1]:
        return 0.01
    elif t0 <= crits[0]:
        return 0.10
    else:
        f = interp1d(crits, levels, kind="linear")
        return float(np.clip(f(t0), 0.01, 0.10))


def _buishand_approx_pvalue(r_stat: float) -> float:
    """Approximate p-value for Buishand range statistic R/sqrt(n).

    Critical values from Buishand (1982): 1% ~ 1.70, 5% ~ 1.42, 10% ~ 1.27
    """
    from scipy.interpolate import interp1d

    crits = np.array([1.27, 1.42, 1.70])
    levels = np.array([0.10, 0.05, 0.01])

    if r_stat >= crits[-1]:
        return 0.01
    elif r_stat <= crits[0]:
        return 0.10
    else:
        f = interp1d(crits, levels, kind="linear")
        return float(np.clip(f(r_stat), 0.01, 0.10))
