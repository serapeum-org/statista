"""Tests for the SeasonalMixin (Phase 9)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


@pytest.fixture
def ts_daily_2yr():
    """TimeSeries with DatetimeIndex, 2 years of daily data."""
    np.random.seed(42)
    idx = pd.date_range("2000-01-01", periods=730, freq="D")
    data = np.sin(np.arange(730) * 2 * np.pi / 365) + np.random.randn(730) * 0.3
    return TimeSeries(data, index=idx)


class TestMonthlyStats:
    """Tests for monthly_stats() method."""

    def test_returns_12_rows(self, ts_daily_2yr):
        """Should return 12 rows (one per month)."""
        result = ts_daily_2yr.monthly_stats()
        assert result.shape[0] == 12, f"Expected 12 rows, got {result.shape[0]}"

    def test_returns_expected_columns(self, ts_daily_2yr):
        """Should have mean, std, cv, min, max, median, skewness."""
        result = ts_daily_2yr.monthly_stats()
        expected = ["mean", "std", "cv", "min", "max", "median", "skewness"]
        assert set(expected).issubset(set(result.columns))

    def test_returns_dataframe(self, ts_daily_2yr):
        """Return type should be DataFrame."""
        result = ts_daily_2yr.monthly_stats()
        assert isinstance(result, DataFrame)

    def test_index_is_month(self, ts_daily_2yr):
        """Index should be month numbers 1-12."""
        result = ts_daily_2yr.monthly_stats()
        assert result.index.tolist() == list(range(1, 13))

    def test_raises_without_datetime_index(self):
        """Should raise TypeError if index is not DatetimeIndex."""
        ts = TimeSeries(np.random.randn(100))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            ts.monthly_stats()

    def test_column_parameter(self, ts_daily_2yr):
        """Should work with column parameter on multi-column data."""
        data = np.random.randn(730, 2)
        idx = pd.date_range("2000-01-01", periods=730, freq="D")
        ts = TimeSeries(data, index=idx, columns=["A", "B"])
        result = ts.monthly_stats(column="B")
        assert result.shape[0] == 12


class TestSeasonalSubseries:
    """Tests for seasonal_subseries() method."""

    def test_returns_figure_axes(self):
        """Should return (Figure, Axes)."""
        ts = TimeSeries(np.sin(np.arange(120) * 2 * np.pi / 12))
        fig, ax = ts.seasonal_subseries(period=12)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_period(self):
        """Should work with non-12 periods."""
        ts = TimeSeries(np.random.randn(70))
        fig, ax = ts.seasonal_subseries(period=7)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_column_parameter(self):
        """Should use specified column."""
        ts = TimeSeries(np.random.randn(60, 2), columns=["A", "B"])
        fig, ax = ts.seasonal_subseries(period=6, column="B")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestAnnualCycle:
    """Tests for annual_cycle() method."""

    def test_returns_figure_axes(self, ts_daily_2yr):
        """Should return (Figure, Axes)."""
        fig, ax = ts_daily_2yr.annual_cycle()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_raises_without_datetime_index(self):
        """Should raise TypeError if index is not DatetimeIndex."""
        ts = TimeSeries(np.random.randn(100))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            ts.annual_cycle()

    def test_custom_labels(self, ts_daily_2yr):
        """Custom title should override default."""
        fig, ax = ts_daily_2yr.annual_cycle(title="My Cycle")
        assert ax.get_title() == "My Cycle"
        plt.close(fig)


class TestPeriodogram:
    """Tests for periodogram() method."""

    def test_returns_frequencies_and_power(self):
        """Should return non-empty frequency and power arrays."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        freqs, power, _ = ts.periodogram(plot=False)
        assert len(freqs) > 0, "Frequencies should not be empty"
        assert len(power) > 0, "Power should not be empty"
        assert len(freqs) == len(power)

    def test_detects_known_frequency(self):
        """Should detect a known periodic signal."""
        t = np.arange(500)
        data = np.sin(2 * np.pi * t / 50) + np.random.randn(500) * 0.1
        ts = TimeSeries(data)
        freqs, power, _ = ts.periodogram(plot=False, method="periodogram")
        peak_idx = np.argmax(power[1:]) + 1
        peak_freq = freqs[peak_idx]
        expected_freq = 1.0 / 50
        assert (
            abs(peak_freq - expected_freq) < 0.005
        ), f"Expected peak at {expected_freq:.4f}, got {peak_freq:.4f}"

    def test_welch_method(self):
        """Welch method should work."""
        ts = TimeSeries(np.random.randn(200))
        freqs, power, _ = ts.periodogram(method="welch", plot=False)
        assert len(freqs) > 0

    def test_periodogram_method(self):
        """Periodogram method should work."""
        ts = TimeSeries(np.random.randn(200))
        freqs, power, _ = ts.periodogram(method="periodogram", plot=False)
        assert len(freqs) > 0

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        ts = TimeSeries(np.random.randn(100))
        with pytest.raises(ValueError, match="Unknown method"):
            ts.periodogram(method="invalid", plot=False)

    def test_plot_returns_figure(self):
        """With plot=True, should return (fig, ax)."""
        ts = TimeSeries(np.random.randn(200))
        freqs, power, fig_ax = ts.periodogram(plot=True)
        assert fig_ax is not None
        plt.close(fig_ax[0])

    def test_plot_false_returns_none(self):
        """With plot=False, figure should be None."""
        ts = TimeSeries(np.random.randn(200))
        _, _, fig_ax = ts.periodogram(plot=False)
        assert fig_ax is None

    def test_column_parameter(self):
        """Should use specified column."""
        ts = TimeSeries(np.random.randn(200, 2), columns=["A", "B"])
        freqs, power, _ = ts.periodogram(column="B", plot=False)
        assert len(freqs) > 0

    def test_custom_fs(self):
        """Custom sampling frequency should shift frequencies."""
        ts = TimeSeries(np.random.randn(200))
        freqs1, _, _ = ts.periodogram(fs=1.0, plot=False)
        freqs2, _, _ = ts.periodogram(fs=12.0, plot=False)
        assert (
            freqs2.max() > freqs1.max()
        ), "Higher fs should produce higher max frequency"


class TestSeasonalMannKendall:
    """Tests for seasonal_mann_kendall() method."""

    def test_increasing_trend_detected(self):
        """Data with seasonal pattern + positive trend should detect increasing."""
        np.random.seed(42)
        t = np.arange(120)
        data = 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.randn(120)
        ts = TimeSeries(data)
        result = ts.seasonal_mann_kendall(period=12)
        assert result.loc["Series1", "trend"] == "increasing"

    def test_no_trend(self):
        """Pure seasonal data without trend should show no trend."""
        np.random.seed(42)
        t = np.arange(120)
        data = 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.5
        ts = TimeSeries(data)
        result = ts.seasonal_mann_kendall(period=12)
        assert result.loc["Series1", "trend"] == "no trend"

    def test_returns_expected_columns(self):
        """Result should have all expected columns."""
        ts = TimeSeries(np.random.randn(60))
        result = ts.seasonal_mann_kendall(period=6)
        expected = [
            "trend",
            "h",
            "p_value",
            "z",
            "combined_s",
            "combined_var_s",
            "per_season_s",
        ]
        assert set(expected).issubset(set(result.columns))

    def test_per_season_s_length(self):
        """per_season_s should have length equal to period."""
        ts = TimeSeries(np.random.randn(120))
        result = ts.seasonal_mann_kendall(period=12)
        assert len(result.loc["Series1", "per_season_s"]) == 12

    def test_multi_column(self):
        """Should return one row per column."""
        ts = TimeSeries(np.random.randn(60, 2), columns=["A", "B"])
        result = ts.seasonal_mann_kendall(period=6)
        assert result.shape[0] == 2

    def test_column_parameter(self):
        """Should only test specified column."""
        ts = TimeSeries(np.random.randn(60, 2), columns=["X", "Y"])
        result = ts.seasonal_mann_kendall(period=6, column="Y")
        assert result.shape[0] == 1
        assert result.index[0] == "Y"
