"""Distribution-aware methods mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy import stats as scipy_stats

from statista.time_series.constants import DEFAULT_ALPHA

if TYPE_CHECKING:
    from statista.time_series.stubs import _TimeSeriesStub
else:
    _TimeSeriesStub = object


class Distribution(_TimeSeriesStub):
    """Distribution fitting, normality tests, and diagnostic plots."""

    def qq_plot(
        self,
        distribution: str = "norm",
        column: str = None,
        confidence: float = 0.95,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Quantile-Quantile plot against a theoretical distribution.

        The single most useful diagnostic plot for assessing distributional assumptions.
        Points near the 1:1 line indicate a good fit; deviations in the tails indicate
        heavy or light tails.

        Args:
            distribution: Name of a ``scipy.stats`` distribution (e.g., "norm", "expon",
                "gumbel_r"). Default "norm".
            column: Column to plot. If None, uses first column.
            confidence: Confidence level for the envelope (0-1). Default 0.95.
            **kwargs: Passed to ``_adjust_axes_labels`` (title, xlabel, ylabel, etc.).

        Returns:
            tuple: (Figure, Axes)

        Examples:
            Basic QQ plot against a normal distribution:

            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> np.random.seed(42)  # doctest: +SKIP
            >>> ts = TimeSeries(np.random.randn(200))  # doctest: +SKIP
            >>> fig, ax = ts.qq_plot()  # doctest: +SKIP

            QQ plot against an exponential distribution:

            >>> ts2 = TimeSeries(np.random.exponential(2, 200))  # doctest: +SKIP
            >>> fig, ax = ts2.qq_plot(distribution="expon")  # doctest: +SKIP

        References:
            Wilk, M.B. and Gnanadesikan, R. (1968). Probability plotting methods for the
            analysis of data. Biometrika, 55(1), 1-17.
        """
        if column is None:
            column = self.columns[0]

        data = np.sort(self[column].dropna().values)
        n = len(data)

        dist = getattr(scipy_stats, distribution)
        params = dist.fit(data)

        theoretical_quantiles = dist.ppf((np.arange(1, n + 1) - 0.5) / n, *params)

        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)

        ax.scatter(
            theoretical_quantiles,
            data,
            alpha=0.6,
            s=15,
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )

        # 1:1 reference line
        lims = [
            min(theoretical_quantiles.min(), data.min()),
            max(theoretical_quantiles.max(), data.max()),
        ]
        ax.plot(lims, lims, "r-", linewidth=1, label="1:1 line")

        if "title" not in kwargs:
            kwargs["title"] = f"QQ Plot — {column} vs {distribution}"
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = f"Theoretical ({distribution})"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "Sample quantiles"

        ax = self._adjust_axes_labels(ax, **kwargs)
        plt.show()
        return fig, ax

    def pp_plot(
        self,
        distribution: str = "norm",
        column: str = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Probability-Probability plot.

        Plots the empirical CDF against the theoretical CDF at each data point.
        Complementary to QQ plot -- PP emphasizes the center of the distribution,
        QQ emphasizes the tails.

        Args:
            distribution: Name of a ``scipy.stats`` distribution. Default "norm".
            column: Column to plot. If None, uses first column.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (Figure, Axes)

        Examples:
            Basic PP plot against a normal distribution:

            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> np.random.seed(42)  # doctest: +SKIP
            >>> ts = TimeSeries(np.random.randn(200))  # doctest: +SKIP
            >>> fig, ax = ts.pp_plot()  # doctest: +SKIP

            PP plot against a Gumbel distribution:

            >>> ts2 = TimeSeries(np.random.gumbel(0, 1, 200))  # doctest: +SKIP
            >>> fig, ax = ts2.pp_plot(distribution="gumbel_r")  # doctest: +SKIP
        """
        if column is None:
            column = self.columns[0]

        data = np.sort(self[column].dropna().values)
        n = len(data)

        dist = getattr(scipy_stats, distribution)
        params = dist.fit(data)

        # Empirical CDF (Weibull plotting position)
        empirical_cdf = (np.arange(1, n + 1)) / (n + 1)
        # Theoretical CDF
        theoretical_cdf = dist.cdf(data, *params)

        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)

        ax.scatter(
            theoretical_cdf,
            empirical_cdf,
            alpha=0.6,
            s=15,
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )

        ax.plot([0, 1], [0, 1], "r-", linewidth=1, label="1:1 line")

        if "title" not in kwargs:
            kwargs["title"] = f"PP Plot — {column} vs {distribution}"
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = f"Theoretical CDF ({distribution})"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "Empirical CDF"

        ax = self._adjust_axes_labels(ax, **kwargs)
        plt.show()
        return fig, ax

    def normality_test(
        self,
        method: str = "auto",
        alpha: float = DEFAULT_ALPHA,
    ) -> DataFrame:
        """Test each column for normality.

        Args:
            method: Test to use.
                - "auto": Shapiro-Wilk if n < 5000, D'Agostino-Pearson if n >= 5000.
                - "shapiro": Shapiro-Wilk test (``scipy.stats.shapiro``). Best for n < 5000.
                - "dagostino": D'Agostino-Pearson test (``scipy.stats.normaltest``). For large n.
                - "anderson": Anderson-Darling test (``scipy.stats.anderson``).
                - "jarque_bera": Jarque-Bera test (``scipy.stats.jarque_bera``).
            alpha: Significance level. Default 0.05.

        Returns:
            pandas.DataFrame: One row per column with: test_name, statistic, p_value,
                is_normal, conclusion.

        Examples:
            Test normally distributed data (Shapiro-Wilk auto-selected for n < 5000):

            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.random.randn(200))
            >>> result = ts.normality_test()
            >>> result.loc["Series1", "test_name"]
            'Shapiro-Wilk'
            >>> round(float(result.loc["Series1", "statistic"]), 4)
            0.9956
            >>> round(float(result.loc["Series1", "p_value"]), 4)
            0.829
            >>> bool(result.loc["Series1", "is_normal"])
            True

            Test non-normal data (exponential distribution fails normality):

            >>> np.random.seed(42)
            >>> ts2 = TimeSeries(np.random.exponential(2, 200))
            >>> result2 = ts2.normality_test()
            >>> result2.loc["Series1", "conclusion"]
            'Non-normal'
            >>> round(float(result2.loc["Series1", "statistic"]), 4)
            0.8522

            Use Jarque-Bera test explicitly:

            >>> np.random.seed(42)
            >>> ts3 = TimeSeries(np.random.randn(200))
            >>> result3 = ts3.normality_test(method="jarque_bera")
            >>> round(float(result3.loc["Series1", "p_value"]), 4)
            0.7464
        """
        rows = []

        for col in self.columns:
            data = self[col].dropna().values
            n = len(data)

            if method == "auto":
                chosen = "shapiro" if n < 5000 else "dagostino"
            else:
                chosen = method

            if chosen == "shapiro":
                stat, p = scipy_stats.shapiro(data)
                test_name = "Shapiro-Wilk"
            elif chosen == "dagostino":
                stat, p = scipy_stats.normaltest(data)
                test_name = "D'Agostino-Pearson"
            elif chosen == "anderson":
                result = scipy_stats.anderson(data, dist="norm")
                stat = result.statistic
                # Interpolate p from Anderson-Darling critical values
                # significance_level: [15%, 10%, 5%, 2.5%, 1%]
                sig_levels = result.significance_level / 100.0
                crits = result.critical_values
                if stat >= crits[-1]:
                    p = sig_levels[-1]  # beyond 1% critical
                elif stat <= crits[0]:
                    p = sig_levels[0]  # below 15% critical
                else:
                    from scipy.interpolate import interp1d

                    f = interp1d(crits, sig_levels, kind="linear")
                    p = float(f(stat))
                test_name = "Anderson-Darling"
            elif chosen == "jarque_bera":
                stat, p = scipy_stats.jarque_bera(data)
                test_name = "Jarque-Bera"
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Choose from "
                    "'auto', 'shapiro', 'dagostino', 'anderson', 'jarque_bera'."
                )

            is_normal = bool(p > alpha)
            conclusion = "Normal" if is_normal else "Non-normal"

            rows.append(
                {
                    "column": col,
                    "test_name": test_name,
                    "statistic": float(stat),
                    "p_value": float(p),
                    "is_normal": is_normal,
                    "conclusion": conclusion,
                }
            )

        result_df = DataFrame(rows).set_index("column")
        return result_df

    def empirical_cdf(
        self,
        column: str = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Step-function plot of the empirical CDF.

        Simpler than KDE -- no bandwidth choice needed. Shows the actual data
        distribution as a monotonically increasing step function.

        Args:
            column: Column to plot. If None, plots all columns overlaid.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (Figure, Axes)

        Examples:
            Plot the empirical CDF of normally distributed data:

            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> np.random.seed(42)  # doctest: +SKIP
            >>> ts = TimeSeries(np.random.randn(100))  # doctest: +SKIP
            >>> fig, ax = ts.empirical_cdf()  # doctest: +SKIP

            Overlay multiple columns on one plot:

            >>> data = np.column_stack([np.random.randn(100), np.random.randn(100) + 2])  # doctest: +SKIP
            >>> ts2 = TimeSeries(data, columns=["A", "B"])  # doctest: +SKIP
            >>> fig, ax = ts2.empirical_cdf()  # doctest: +SKIP
        """
        cols = [column] if column is not None else list(self.columns)

        fig, ax = self._get_ax_fig(**kwargs)
        kwargs.pop("fig", None)
        kwargs.pop("ax", None)

        for col in cols:
            data = np.sort(self[col].dropna().values)
            n = len(data)
            ecdf = np.arange(1, n + 1) / n
            ax.step(data, ecdf, where="post", linewidth=1.2, label=col)

        if "title" not in kwargs:
            kwargs["title"] = "Empirical CDF"
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = "Value"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "Cumulative probability"
        if "legend" not in kwargs and len(cols) > 1:
            kwargs["legend"] = cols

        ax = self._adjust_axes_labels(ax, **kwargs)
        plt.show()
        return fig, ax

    def fit_distributions(self, method: str = "lmoments") -> DataFrame:
        """Fit distributions to each column and select the best fit.

        Uses the statista ``Distributions`` facade to fit all available distributions
        (GEV, Gumbel, Exponential, Normal) and selects the best by KS test.

        Args:
            method: Parameter estimation method -- "lmoments", "mle", or "mm".
                Default "lmoments".

        Returns:
            pandas.DataFrame: One row per column with: best_distribution, loc, scale,
                shape, ks_statistic, ks_p_value.

        Examples:
            Fit distributions using MLE and inspect the best fit:

            >>> import numpy as np  # doctest: +SKIP
            >>> from statista.time_series import TimeSeries  # doctest: +SKIP
            >>> np.random.seed(42)  # doctest: +SKIP
            >>> ts = TimeSeries(np.random.randn(200))  # doctest: +SKIP
            >>> result = ts.fit_distributions(method="mle")  # doctest: +SKIP
            >>> result.loc["Series1", "best_distribution"]  # doctest: +SKIP
            'Normal'
            >>> round(float(result.loc["Series1", "ks_p_value"]), 4)  # doctest: +SKIP
            0.9997

            Check that the result DataFrame has the expected columns:

            >>> sorted(result.columns.tolist())  # doctest: +SKIP
            ['best_distribution', 'ks_p_value', 'ks_statistic', 'loc', 'scale', 'shape']
        """
        from statista.distributions import Distributions

        rows = []

        for col in self.columns:
            data = self[col].dropna().values

            try:
                dist = Distributions(data=data)
                best_name, best_info = dist.best_fit(method=method)

                params = best_info["parameters"]
                ks_stat, ks_p = best_info["ks"]

                rows.append(
                    {
                        "column": col,
                        "best_distribution": best_name,
                        "loc": params.get("loc", np.nan),
                        "scale": params.get("scale", np.nan),
                        "shape": params.get("shape", np.nan),
                        "ks_statistic": float(ks_stat),
                        "ks_p_value": float(ks_p),
                    }
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                rows.append(
                    {
                        "column": col,
                        "best_distribution": f"Error: {e}",
                        "loc": np.nan,
                        "scale": np.nan,
                        "shape": np.nan,
                        "ks_statistic": np.nan,
                        "ks_p_value": np.nan,
                    }
                )

        result_df = DataFrame(rows).set_index("column")
        return result_df
