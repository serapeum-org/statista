"""Hydrological methods mixin for TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

if TYPE_CHECKING:
    from statista.time_series.stubs import _TimeSeriesStub
else:
    _TimeSeriesStub = object


class Hydrological(_TimeSeriesStub):
    """Hydrology-specific analysis methods for TimeSeries.

    Implements flow duration curves, baseflow separation, annual extremes, and
    hydrological indices commonly used in water resources engineering.
    """

    def flow_duration_curve(
        self,
        log_scale: bool = True,
        method: str = "weibull",
        column: str = None,
        plot: bool = True,
        **kwargs: Any,
    ) -> tuple[DataFrame, tuple[Figure, Axes] | None]:
        """Compute and plot the flow duration curve (FDC).

        The FDC is the most widely used plot in hydrology. It shows the percentage
        of time a given flow value is equalled or exceeded.

        Args:
            log_scale: Use log scale for the y-axis. Default True.
            method: Plotting position formula.
                - "weibull": i / (n+1). Default.
                - "gringorten": (i - 0.44) / (n + 0.12). Recommended for Gumbel/GEV.
                - "cunnane": (i - 0.4) / (n + 0.2).
            column: Column to analyze. If None, overlays all columns.
            plot: Whether to produce a plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (fdc_df, (fig, ax)) or (fdc_df, None).
                fdc_df has columns: value, exceedance_pct for single column,
                or one value column per series.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.abs(np.random.randn(365)) * 100)
            >>> fdc, _ = ts.flow_duration_curve(plot=False)
            >>> "exceedance_pct" in fdc.columns
            True

            ```

        References:
            Vogel, R.M. and Fennessey, N.M. (1994). Flow-Duration Curves. I: New Interpretation
            and Confidence Intervals. Journal of Water Resources Planning and Management, 120(4).
        """
        cols = [column] if column is not None else list(self.columns)

        all_results = []
        for col in cols:
            data = np.sort(self[col].dropna().values)[::-1]  # Sort descending
            n = len(data)

            # Validate non-empty data after removing NaN values
            if n == 0:
                raise ValueError(
                    f"Column '{col}' contains no valid data after removing NaN values"
                )

            ranks = np.arange(1, n + 1)

            if method == "weibull":
                exceedance = ranks / (n + 1) * 100
            elif method == "gringorten":
                exceedance = (ranks - 0.44) / (n + 0.12) * 100  # type: ignore[assignment]
            elif method == "cunnane":
                exceedance = (ranks - 0.4) / (n + 0.2) * 100  # type: ignore[assignment]
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Choose from 'weibull', 'gringorten', 'cunnane'."
                )

            all_results.append((col, data, exceedance))

        # Build result DataFrame
        if len(cols) == 1:
            col, data, exceedance = all_results[0]
            fdc_df = DataFrame({"value": data, "exceedance_pct": exceedance})
        else:
            # Each column may have different lengths after dropna, so build per-column
            frames = []
            for col, col_data, exc in all_results:
                frames.append(DataFrame({col: col_data, f"{col}_exceedance_pct": exc}))
            fdc_df = pd.concat(frames, axis=1)

        fig_ax: tuple[Figure, Axes] | None = None
        if plot:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            for col, data, exceedance in all_results:
                ax.plot(exceedance, data, linewidth=1.2, label=col)

            if log_scale:
                ax.set_yscale("log")

            if "title" not in kwargs:
                kwargs["title"] = "Flow Duration Curve"
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Exceedance Probability (%)"
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = "Value"
            if len(cols) > 1 and "legend" not in kwargs:
                kwargs["legend"] = cols

            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return fdc_df, fig_ax

    def annual_extremes(
        self,
        kind: str = "max",
        water_year_start: str = "YE-OCT",
        column: str = None,
    ) -> Any:
        """Extract annual maxima or minima series.

        Args:
            kind: "max" for annual maxima, "min" for annual minima. Default "max".
            water_year_start: Pandas offset alias for resampling rule defining the
                water year. Default "YE-OCT" (Oct-Sep water year).
            column: Column to extract. If None, extracts from all columns.

        Returns:
            TimeSeries: New TimeSeries with one value per year.

        Raises:
            ValueError: If kind is not "max" or "min".

        Examples:
            ```python
            >>> import numpy as np
            >>> import pandas as pd
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> idx = pd.date_range("2000-01-01", periods=730, freq="D")
            >>> ts = TimeSeries(np.random.randn(730), index=idx)
            >>> ams = ts.annual_extremes(kind="max")
            >>> ams.shape[0] >= 1
            True

            ```
        """
        from statista.time_series import TimeSeries

        if column is not None:
            data = self[column]
        else:
            data = self

        if kind == "max":
            result = data.resample(water_year_start).max()
        elif kind == "min":
            result = data.resample(water_year_start).min()
        else:
            raise ValueError(f"kind must be 'max' or 'min', got '{kind}'.")

        result_vals = (
            result.values if result.values.ndim == 2 else result.values.reshape(-1, 1)
        )
        cols = (
            list(result.columns)
            if hasattr(result, "columns")
            else [column or self.columns[0]]
        )
        ts_result = TimeSeries(result_vals, index=result.index, columns=cols)
        return ts_result

    def exceedance_probability(
        self,
        method: str = "weibull",
        column: str = None,
    ) -> DataFrame:
        """Compute empirical exceedance probability for each value.

        Args:
            method: Plotting position formula — "weibull", "gringorten", or "cunnane".
                Default "weibull".
            column: Column to analyze. If None, analyzes all columns.

        Returns:
            pandas.DataFrame: Sorted by value (descending) with columns:
                value, exceedance_probability, return_period.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> ts = TimeSeries(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
            >>> result = ts.exceedance_probability()
            >>> "return_period" in result.columns
            True

            ```
        """
        cols = [column] if column is not None else list(self.columns)
        frames = []

        for col in cols:
            data = np.sort(self[col].dropna().values)[::-1]
            n = len(data)
            ranks = np.arange(1, n + 1)

            if method == "weibull":
                exc = ranks / (n + 1)
            elif method == "gringorten":
                exc = (ranks - 0.44) / (n + 0.12)  # type: ignore[assignment]
            elif method == "cunnane":
                exc = (ranks - 0.4) / (n + 0.2)  # type: ignore[assignment]
            else:
                raise ValueError(f"Unknown method '{method}'.")

            rp = 1.0 / exc

            frame = DataFrame(
                {
                    "column": col,
                    "value": data,
                    "exceedance_probability": exc,
                    "return_period": rp,
                }
            )
            frames.append(frame)

        result = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        return result

    def baseflow_separation(
        self,
        method: str = "lyne_hollick",
        alpha: float = 0.925,
        column: str = None,
        plot: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, tuple[Figure, Axes] | None]:
        """Separate streamflow into baseflow and quickflow.

        Args:
            method: Separation method.
                - "lyne_hollick": Digital filter (Lyne & Hollick, 1979).
                  ``b_t = alpha * b_{t-1} + (1-alpha)/2 * (q_t + q_{t-1})``
                - "eckhardt": Two-parameter filter (Eckhardt, 2005).
                  Extra param via kwargs: bfi_max (default 0.80).
                - "chapman_maxwell": One-parameter filter (Chapman & Maxwell, 1996).
                  ``b_t = k/(2-k) * b_{t-1} + (1-k)/(2-k) * q_t``
            alpha: Filter coefficient. Default 0.925.
            column: Column to analyze. If None, uses first column.
            plot: Whether to produce a hydrograph with baseflow shading. Default True.
            **kwargs: bfi_max for Eckhardt, or passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (separation_df, (fig, ax)) or (separation_df, None).
                separation_df has columns: total_flow, baseflow, quickflow.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.abs(np.random.randn(200)) * 10 + 5)
            >>> result, _ = ts.baseflow_separation(plot=False)
            >>> "baseflow" in result.columns
            True

            ```

        References:
            Lyne, V. and Hollick, M. (1979). Stochastic time-variable rainfall-runoff
            modelling. Inst. Eng. Aust. Natl. Conf.

            Eckhardt, K. (2005). How to construct recursive digital filters for baseflow
            separation. Hydrological Processes, 19(2), 507-515.
        """
        if column is None:
            column = self.columns[0]

        q = self[column].dropna().values.astype(float)
        idx = self[column].dropna().index

        if method == "lyne_hollick":
            baseflow = _lyne_hollick(q, alpha)
        elif method == "eckhardt":
            bfi_max = kwargs.pop("bfi_max", 0.80)
            baseflow = _eckhardt(q, alpha, bfi_max)
        elif method == "chapman_maxwell":
            baseflow = _chapman_maxwell(q, alpha)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'lyne_hollick', 'eckhardt', 'chapman_maxwell'."
            )

        quickflow = q - baseflow

        result_df = DataFrame(
            {"total_flow": q, "baseflow": baseflow, "quickflow": quickflow},
            index=idx,
        )

        fig_ax: tuple[Figure, Axes] | None = None
        if plot:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            ax.plot(idx, q, color="steelblue", linewidth=0.8, label="Total flow")
            ax.fill_between(
                idx, 0, baseflow, color="lightblue", alpha=0.6, label="Baseflow"
            )

            if "title" not in kwargs:
                kwargs["title"] = f"Baseflow Separation ({method}) — {column}"
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Time"
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = "Flow"

            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return result_df, fig_ax

    def baseflow_index(
        self,
        method: str = "lyne_hollick",
        alpha: float = 0.925,
        column: str = None,
    ) -> DataFrame:
        """Compute the Baseflow Index (BFI) — ratio of baseflow to total flow.

        BFI = sum(baseflow) / sum(total_flow). Values near 1 indicate
        groundwater-dominated systems; near 0 indicate flashy systems.

        Args:
            method: Separation method (see ``baseflow_separation``). Default "lyne_hollick".
            alpha: Filter coefficient. Default 0.925.
            column: Column to analyze. If None, analyzes all columns.

        Returns:
            pandas.DataFrame: One row per column with: bfi value.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> ts = TimeSeries(np.abs(np.random.randn(200)) * 10 + 5)
            >>> result = ts.baseflow_index()
            >>> 0.0 <= result.loc["Series1", "bfi"] <= 1.0
            True

            ```
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            sep_df, _ = self.baseflow_separation(
                method=method, alpha=alpha, column=col, plot=False
            )
            bfi = float(sep_df["baseflow"].sum() / sep_df["total_flow"].sum())
            rows.append({"column": col, "bfi": bfi})

        result = DataFrame(rows).set_index("column")
        return result

    def flashiness_index(self, column: str = None) -> DataFrame:
        """Richards-Baker Flashiness Index.

        Measures the oscillations in flow relative to total flow:
        FI = sum(|Q_t - Q_{t-1}|) / sum(Q_t)

        Higher values indicate flashier (more variable) flow regimes.

        Args:
            column: Column to analyze. If None, analyzes all columns.

        Returns:
            pandas.DataFrame: One row per column with: flashiness value.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> ts = TimeSeries(np.array([10.0, 50.0, 10.0, 50.0, 10.0]))
            >>> result = ts.flashiness_index()
            >>> result.loc["Series1", "flashiness"] > 0
            True

            ```

        References:
            Baker, D.B. et al. (2004). A new flashiness index: characteristics and
            applications to midwestern rivers and streams. JAWRA, 40(2), 503-522.
        """
        cols = [column] if column is not None else list(self.columns)
        rows = []

        for col in cols:
            data = self[col].dropna().values
            fi = (
                float(np.sum(np.abs(np.diff(data))) / np.sum(data))
                if np.sum(data) > 0
                else 0.0
            )
            rows.append({"column": col, "flashiness": fi})

        result = DataFrame(rows).set_index("column")
        return result

    def recession_analysis(
        self,
        min_length: int = 5,
        column: str = None,
        plot: bool = True,
        **kwargs: Any,
    ) -> tuple[DataFrame, tuple[Figure, Axes] | None]:
        """Extract recession segments and fit a master recession curve.

        Identifies periods of monotonically decreasing flow (recession limbs)
        and fits an exponential recession model: Q(t) = Q0 * exp(-t / k),
        where k is the recession constant.

        Args:
            min_length: Minimum number of consecutive decreasing steps to
                qualify as a recession segment. Default 5.
            column: Column to analyze. If None, uses first column.
            plot: Whether to produce a log(Q) vs t recession plot. Default True.
            **kwargs: Passed to ``_adjust_axes_labels``.

        Returns:
            tuple: (recession_df, (fig, ax)) or (recession_df, None).
                recession_df has columns: recession_id, start_index, end_index, length,
                recession_constant_k, r_squared.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.time_series import TimeSeries
            >>> np.random.seed(42)
            >>> q = 100 * np.exp(-np.arange(50) / 15.0) + np.random.randn(50) * 0.5
            >>> ts = TimeSeries(np.abs(q))
            >>> result, _ = ts.recession_analysis(min_length=3, plot=False)
            >>> len(result) >= 1
            True

            ```

        References:
            Tallaksen, L.M. (1995). A review of baseflow recession analysis.
            Journal of Hydrology, 165(1-4), 349-370.
        """
        if column is None:
            column = self.columns[0]

        data = self[column].dropna().values.astype(float)
        n = len(data)

        # Identify recession segments (monotonically decreasing runs)
        segments = []
        start = None
        for i in range(1, n):
            if data[i] < data[i - 1]:
                if start is None:
                    start = i - 1
            else:
                if start is not None:
                    length = i - start
                    if length >= min_length:
                        segments.append((start, i - 1))
                    start = None
        if start is not None and (n - start) >= min_length:
            segments.append((start, n - 1))

        # Fit exponential decay to each segment
        rows = []
        for seg_id, (s, e) in enumerate(segments):
            seg = data[s : e + 1]
            seg_positive = seg[seg > 0]
            if len(seg_positive) < 3:
                continue

            t = np.arange(len(seg_positive), dtype=float)
            log_q = np.log(seg_positive)

            # Linear fit: log(Q) = log(Q0) - t/k  =>  slope = -1/k
            coeffs = np.polyfit(t, log_q, 1)
            slope = coeffs[0]
            k = -1.0 / slope if slope != 0 else np.inf

            # R-squared
            predicted = np.polyval(coeffs, t)
            ss_res = np.sum((log_q - predicted) ** 2)
            ss_tot = np.sum((log_q - np.mean(log_q)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            rows.append(
                {
                    "recession_id": seg_id,
                    "start_index": s,
                    "end_index": e,
                    "length": e - s + 1,
                    "recession_constant_k": float(k),
                    "r_squared": float(r_squared),
                }
            )

        recession_df = (
            DataFrame(rows)
            if rows
            else DataFrame(
                columns=[
                    "recession_id",
                    "start_index",
                    "end_index",
                    "length",
                    "recession_constant_k",
                    "r_squared",
                ]
            )
        )

        fig_ax: tuple[Figure, Axes] | None = None
        if plot and len(segments) > 0:
            fig, ax = self._get_ax_fig(**kwargs)
            kwargs.pop("fig", None)
            kwargs.pop("ax", None)

            for s, e in segments:
                seg = data[s : e + 1]
                seg_positive = seg[seg > 0]
                if len(seg_positive) >= 3:
                    t = np.arange(len(seg_positive))
                    ax.plot(
                        t, seg_positive, "o-", markersize=3, linewidth=0.8, alpha=0.6
                    )

            ax.set_yscale("log")

            if "title" not in kwargs:
                kwargs["title"] = f"Recession Analysis — {column}"
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Time steps from recession start"
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = "Flow (log scale)"

            ax = self._adjust_axes_labels(ax, **kwargs)
            plt.show()
            fig_ax = (fig, ax)

        return recession_df, fig_ax


# ---------------------------------------------------------------------------
# Baseflow separation algorithms
# ---------------------------------------------------------------------------


def _lyne_hollick(q: np.ndarray, alpha: float) -> np.ndarray:
    """Lyne & Hollick (1979) digital filter for baseflow separation.

    b_t = alpha * b_{t-1} + (1 - alpha) / 2 * (q_t + q_{t-1})
    with constraints: 0 <= b_t <= q_t
    """
    n = len(q)
    b = np.zeros(n)
    b[0] = q[0]

    for t in range(1, n):
        b[t] = alpha * b[t - 1] + (1 - alpha) / 2 * (q[t] + q[t - 1])
        b[t] = min(b[t], q[t])
        b[t] = max(b[t], 0.0)

    return b


def _eckhardt(q: np.ndarray, k: float, bfi_max: float) -> np.ndarray:
    """Eckhardt (2005) two-parameter digital filter.

    b_t = ((1 - BFImax) * k * b_{t-1} + (1 - k) * BFImax * q_t) / (1 - k * BFImax)
    """
    n = len(q)
    b = np.zeros(n)
    b[0] = q[0]

    denom = 1 - k * bfi_max
    for t in range(1, n):
        b[t] = ((1 - bfi_max) * k * b[t - 1] + (1 - k) * bfi_max * q[t]) / denom
        b[t] = min(b[t], q[t])
        b[t] = max(b[t], 0.0)

    return b


def _chapman_maxwell(q: np.ndarray, k: float) -> np.ndarray:
    """Chapman & Maxwell (1996) one-parameter filter.

    b_t = k / (2 - k) * b_{t-1} + (1 - k) / (2 - k) * q_t
    """
    n = len(q)
    b = np.zeros(n)
    b[0] = q[0]

    c1 = k / (2 - k)
    c2 = (1 - k) / (2 - k)
    for t in range(1, n):
        b[t] = c1 * b[t - 1] + c2 * q[t]
        b[t] = min(b[t], q[t])
        b[t] = max(b[t], 0.0)

    return b
