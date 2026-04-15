"""Tests for the DecompositionMixin (Phase 8)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


@pytest.fixture
def ts_seasonal():
    """TimeSeries with known trend + seasonality."""
    np.random.seed(42)
    t = np.arange(120)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    trend = 0.1 * t
    data = trend + seasonal + np.random.randn(120) * 0.3
    return TimeSeries(data)


class TestClassicalDecompose:
    """Tests for classical_decompose() method."""

    def test_returns_expected_columns(self, ts_seasonal):
        """Result should have observed, trend, seasonal, residual columns."""
        result, _ = ts_seasonal.classical_decompose(period=12, plot=False)
        expected = ["observed", "trend", "seasonal", "residual"]
        assert result.columns.tolist() == expected

    def test_returns_dataframe(self, ts_seasonal):
        """Should return a DataFrame."""
        result, _ = ts_seasonal.classical_decompose(period=12, plot=False)
        assert isinstance(result, DataFrame)

    def test_additive_reconstruction(self, ts_seasonal):
        """For additive model: observed ~ trend + seasonal + residual (where trend is valid)."""
        result, _ = ts_seasonal.classical_decompose(
            period=12, model="additive", plot=False
        )
        mask = ~np.isnan(result["trend"].values)
        reconstructed = (
            result["trend"].values[mask]
            + result["seasonal"].values[mask]
            + result["residual"].values[mask]
        )
        original = result["observed"].values[mask]
        assert np.allclose(
            reconstructed, original, atol=1e-10
        ), "Additive reconstruction should match observed"

    def test_seasonal_amplitude(self, ts_seasonal):
        """Seasonal component should have amplitude near 5 (the input)."""
        result, _ = ts_seasonal.classical_decompose(period=12, plot=False)
        seasonal = result["seasonal"].values
        amp = (np.nanmax(seasonal) - np.nanmin(seasonal)) / 2
        assert 3.0 < amp < 7.0, f"Expected seasonal amplitude near 5, got {amp:.2f}"

    def test_trend_slope(self, ts_seasonal):
        """Trend should have a positive slope near 0.1."""
        result, _ = ts_seasonal.classical_decompose(period=12, plot=False)
        trend = result["trend"].dropna().values
        x = np.arange(len(trend))
        slope = np.polyfit(x, trend, 1)[0]
        assert 0.05 < slope < 0.15, f"Expected trend slope ~0.1, got {slope:.3f}"

    def test_multiplicative_model(self):
        """Multiplicative decomposition should work without error."""
        np.random.seed(42)
        t = np.arange(120)
        data = (10 + 0.1 * t) * (
            1 + 0.3 * np.sin(2 * np.pi * t / 12)
        ) + np.random.randn(120) * 0.1
        ts = TimeSeries(data)
        result, _ = ts.classical_decompose(
            period=12, model="multiplicative", plot=False
        )
        assert "trend" in result.columns

    def test_period_none_raises(self):
        """period=None should raise ValueError."""
        ts = TimeSeries(np.random.randn(100))
        with pytest.raises(ValueError, match="period must be specified"):
            ts.classical_decompose(period=None, plot=False)

    def test_short_data_raises(self):
        """Data shorter than 2*period should raise ValueError."""
        ts = TimeSeries(np.random.randn(20))
        with pytest.raises(ValueError, match="must be >= 2"):
            ts.classical_decompose(period=12, plot=False)

    def test_invalid_model_raises(self):
        """Invalid model should raise ValueError."""
        ts = TimeSeries(np.random.randn(50))
        with pytest.raises(ValueError, match="model must be"):
            ts.classical_decompose(period=5, model="invalid", plot=False)

    def test_plot_returns_figure(self, ts_seasonal):
        """With plot=True, should return (fig, ax)."""
        _, fig_ax = ts_seasonal.classical_decompose(period=12, plot=True)
        assert fig_ax is not None
        fig, ax = fig_ax
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_false_returns_none(self, ts_seasonal):
        """With plot=False, figure should be None."""
        _, fig_ax = ts_seasonal.classical_decompose(period=12, plot=False)
        assert fig_ax is None

    def test_column_parameter(self):
        """Should work with specified column."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(60, 2), columns=["A", "B"])
        result, _ = ts.classical_decompose(period=6, column="B", plot=False)
        assert isinstance(result, DataFrame)

    def test_odd_period(self):
        """Should work with odd period."""
        np.random.seed(42)
        t = np.arange(100)
        data = np.sin(2 * np.pi * t / 7) + np.random.randn(100) * 0.1
        ts = TimeSeries(data)
        result, _ = ts.classical_decompose(period=7, plot=False)
        assert "seasonal" in result.columns


class TestSmooth:
    """Tests for smooth() method."""

    def test_moving_average(self):
        """Moving average should produce smoother output."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        smoothed = ts.smooth(method="moving_average", window=10)
        assert isinstance(smoothed, TimeSeries)
        assert smoothed.shape == ts.shape

    def test_exponential(self):
        """Exponential smoothing should work."""
        ts = TimeSeries(np.random.randn(100))
        smoothed = ts.smooth(method="exponential", window=10)
        assert isinstance(smoothed, TimeSeries)
        assert smoothed.shape == ts.shape

    def test_savgol(self):
        """Savitzky-Golay filter should work."""
        ts = TimeSeries(np.random.randn(100))
        smoothed = ts.smooth(method="savgol", window=11, polyorder=2)
        assert isinstance(smoothed, TimeSeries)
        assert smoothed.shape == ts.shape

    def test_savgol_even_window(self):
        """Savgol with even window should auto-adjust to odd."""
        ts = TimeSeries(np.random.randn(100))
        smoothed = ts.smooth(method="savgol", window=10)
        assert isinstance(smoothed, TimeSeries)

    def test_preserves_index(self):
        """Smoothed series should have same index."""
        ts = TimeSeries(np.random.randn(50))
        smoothed = ts.smooth(method="moving_average", window=5)
        assert (smoothed.index == ts.index).all()

    def test_preserves_columns(self):
        """Smoothed series should have same column names."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        smoothed = ts.smooth(method="exponential", window=5)
        assert smoothed.columns.tolist() == ["A", "B"]

    def test_reduces_variance(self):
        """Smoothing should reduce the variance of noisy data."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        smoothed = ts.smooth(method="savgol", window=21, polyorder=2)
        original_std = np.nanstd(ts.values)
        smoothed_std = np.nanstd(smoothed.values)
        assert (
            smoothed_std < original_std
        ), f"Smoothing should reduce std: original={original_std:.3f}, smoothed={smoothed_std:.3f}"

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        ts = TimeSeries(np.random.randn(50))
        with pytest.raises(ValueError, match="Unknown method"):
            ts.smooth(method="invalid")


class TestEnvelope:
    """Tests for envelope() method."""

    def test_returns_figure_axes(self):
        """Should return (Figure, Axes)."""
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.envelope(window=10)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_percentiles(self):
        """Custom percentiles should work."""
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.envelope(window=10, lower_pct=10, upper_pct=90)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_column_parameter(self):
        """Should use specified column."""
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        fig, ax = ts.envelope(window=10, column="B")
        assert "B" in ax.get_title()
        plt.close(fig)

    def test_custom_labels(self):
        """Custom title should override default."""
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.envelope(window=10, title="My Envelope")
        assert ax.get_title() == "My Envelope"
        plt.close(fig)
