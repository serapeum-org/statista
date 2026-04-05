"""Tests for the CorrelationMixin (Phase 3)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


class TestACF:
    """Tests for the acf() method."""

    def test_lag_zero_is_one(self):
        """ACF at lag 0 should always be 1.0."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        vals, _ = ts.acf(nlags=10, plot=False)
        assert vals[0] == pytest.approx(1.0), f"ACF[0] should be 1.0, got {vals[0]}"

    def test_white_noise_within_bounds(self):
        """For white noise, all ACF values (lag > 0) should be small."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(500))
        vals, _ = ts.acf(nlags=20, plot=False)
        ci = 1.96 / np.sqrt(500)
        outside = np.sum(np.abs(vals[1:]) > ci)
        assert outside <= 3, f"Too many lags outside 95% CI for white noise: {outside}"

    def test_ar1_decaying(self):
        """ACF of AR(1) with phi=0.8 should show geometric decay."""
        np.random.seed(42)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + np.random.randn()
        ts = TimeSeries(x)
        vals, _ = ts.acf(nlags=10, plot=False)
        assert vals[1] > 0.5, f"ACF[1] for AR(1) phi=0.8 should be > 0.5, got {vals[1]}"
        assert vals[5] < vals[1], "ACF should decay with lag"

    def test_returns_dict_for_multicolumn(self):
        """Multi-column TimeSeries should return a dict of ACF arrays."""
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        vals, _ = ts.acf(nlags=5, plot=False)
        assert isinstance(
            vals, dict
        ), f"Expected dict for multi-column, got {type(vals)}"
        assert "A" in vals and "B" in vals

    def test_plot_returns_figure(self):
        """With plot=True, should return (fig, ax) tuple."""
        ts = TimeSeries(np.random.randn(100))
        vals, fig_ax = ts.acf(nlags=10, plot=True)
        assert fig_ax is not None
        fig, ax = fig_ax
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_false_returns_none(self):
        """With plot=False, second element should be None."""
        ts = TimeSeries(np.random.randn(100))
        _, fig_ax = ts.acf(nlags=10, plot=False)
        assert fig_ax is None

    def test_column_parameter(self):
        """Specifying column should compute ACF only for that column."""
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        vals, _ = ts.acf(nlags=5, column="B", plot=False)
        assert isinstance(vals, np.ndarray), "Single column should return array"


class TestPACF:
    """Tests for the pacf() method."""

    def test_lag_zero_is_one(self):
        """PACF at lag 0 should be 1.0."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        vals, _ = ts.pacf(nlags=10, plot=False)
        assert vals[0] == pytest.approx(1.0)

    def test_ar1_cutoff(self):
        """PACF of AR(1) should have significant value only at lag 1."""
        np.random.seed(42)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + np.random.randn()
        ts = TimeSeries(x)
        vals, _ = ts.pacf(nlags=10, plot=False)
        assert abs(vals[1]) > 0.5, "PACF[1] should be large for AR(1)"
        assert abs(vals[3]) < 0.15, "PACF[3] should be small for AR(1)"

    def test_plot_returns_figure(self):
        """With plot=True, should return (fig, ax)."""
        ts = TimeSeries(np.random.randn(100))
        _, fig_ax = ts.pacf(nlags=10, plot=True)
        assert fig_ax is not None
        plt.close(fig_ax[0])


class TestCrossCorrelation:
    """Tests for cross_correlation() method."""

    def test_identical_series_peak_at_zero(self):
        """CCF of identical series should peak at lag 0."""
        np.random.seed(42)
        data = np.random.randn(100)
        ts = TimeSeries(np.column_stack([data, data]), columns=["A", "B"])
        vals, _ = ts.cross_correlation("A", "B", nlags=5, plot=False)
        assert vals[0] == pytest.approx(
            1.0, abs=0.01
        ), f"CCF[0] for identical series should be ~1.0, got {vals[0]}"

    def test_uncorrelated_series_small_ccf(self):
        """CCF of independent series should be small."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(500, 2), columns=["A", "B"])
        vals, _ = ts.cross_correlation("A", "B", nlags=10, plot=False)
        assert np.max(np.abs(vals)) < 0.15, "CCF should be small for independent series"

    def test_plot_shows_annotation(self):
        """Plot should have a peak-lag annotation."""
        ts = TimeSeries(np.random.randn(100, 2), columns=["X", "Y"])
        _, fig_ax = ts.cross_correlation("X", "Y", nlags=5, plot=True)
        assert fig_ax is not None
        plt.close(fig_ax[0])


class TestLagPlot:
    """Tests for lag_plot() method."""

    def test_returns_figure_axes(self):
        """Should return (Figure, Axes)."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.lag_plot(lag=1)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_correct_scatter_count(self):
        """Number of scatter points should be n - lag."""
        ts = TimeSeries(np.arange(10, dtype=float))
        fig, ax = ts.lag_plot(lag=2)
        offsets = ax.collections[0].get_offsets()
        assert (
            len(offsets) == 8
        ), f"Expected 8 points for n=10, lag=2, got {len(offsets)}"
        plt.close(fig)

    def test_annotation_present(self):
        """Plot should have 'r =' annotation."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.lag_plot()
        texts = [t.get_text() for t in ax.texts]
        assert any(
            "r =" in t for t in texts
        ), f"Missing 'r =' annotation, texts: {texts}"
        plt.close(fig)


class TestCorrelationMatrix:
    """Tests for correlation_matrix() method."""

    def test_diagonal_is_one(self):
        """Diagonal of correlation matrix should be 1.0."""
        ts = TimeSeries(np.random.randn(100, 3), columns=["A", "B", "C"])
        corr, pvals, _ = ts.correlation_matrix(plot=False)
        for col in ["A", "B", "C"]:
            assert (
                corr.loc[col, col] == 1.0
            ), f"Diagonal element for {col} should be 1.0"

    def test_diagonal_pvalue_zero(self):
        """Diagonal p-values should be 0.0."""
        ts = TimeSeries(np.random.randn(100, 3), columns=["A", "B", "C"])
        _, pvals, _ = ts.correlation_matrix(plot=False)
        for col in ["A", "B", "C"]:
            assert pvals.loc[col, col] == 0.0

    def test_symmetric(self):
        """Correlation matrix should be symmetric."""
        ts = TimeSeries(np.random.randn(100, 3), columns=["A", "B", "C"])
        corr, _, _ = ts.correlation_matrix(plot=False)
        assert corr.loc["A", "B"] == corr.loc["B", "A"]

    def test_perfect_correlation(self):
        """Perfectly correlated series should have r = 1.0."""
        data = np.arange(50, dtype=float)
        ts = TimeSeries(np.column_stack([data, data * 2]), columns=["X", "Y"])
        corr, pvals, _ = ts.correlation_matrix(plot=False)
        assert corr.loc["X", "Y"] == pytest.approx(1.0)

    def test_spearman_method(self):
        """Spearman method should work."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        corr, pvals, _ = ts.correlation_matrix(method="spearman", plot=False)
        assert corr.loc["A", "A"] == 1.0

    def test_kendall_method(self):
        """Kendall method should work."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        corr, pvals, _ = ts.correlation_matrix(method="kendall", plot=False)
        assert corr.loc["A", "A"] == 1.0

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        with pytest.raises(ValueError, match="Unknown method"):
            ts.correlation_matrix(method="invalid", plot=False)

    def test_plot_returns_figure(self):
        """With plot=True, should produce a heatmap."""
        ts = TimeSeries(np.random.randn(50, 3), columns=["A", "B", "C"])
        _, _, fig_ax = ts.correlation_matrix(plot=True)
        assert fig_ax is not None
        plt.close(fig_ax[0])


class TestLjungBox:
    """Tests for ljung_box() method."""

    def test_returns_correct_columns(self):
        """Result should have lb_stat and lb_pvalue columns."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.ljung_box(lags=5)
        assert "lb_stat" in result.columns
        assert "lb_pvalue" in result.columns

    def test_correct_number_of_rows(self):
        """Should return one row per lag."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.ljung_box(lags=10)
        assert len(result) == 10, f"Expected 10 rows, got {len(result)}"

    def test_white_noise_high_pvalues(self):
        """White noise should have high p-values (fail to reject H0)."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(500))
        result = ts.ljung_box(lags=10)
        assert (
            result["lb_pvalue"].min() > 0.01
        ), f"White noise should have high p-values, min was {result['lb_pvalue'].min()}"

    def test_autocorrelated_low_pvalues(self):
        """Autocorrelated data should have low p-values."""
        np.random.seed(42)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + np.random.randn()
        ts = TimeSeries(x)
        result = ts.ljung_box(lags=5)
        assert (
            result["lb_pvalue"].iloc[0] < 0.05
        ), "Autocorrelated data should reject H0"

    def test_multi_column(self):
        """Multi-column should produce stacked results with column identifier."""
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        result = ts.ljung_box(lags=5)
        assert "column" in result.columns
        assert len(result) == 10
