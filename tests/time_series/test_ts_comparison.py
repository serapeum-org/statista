"""Tests for the ComparisonMixin (Phase 11)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


class TestAnomaly:
    """Tests for anomaly() method."""

    def test_mean_reference(self):
        """Anomaly from mean should sum to approximately zero."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        anom, _ = ts.anomaly(reference="mean", plot=False)
        assert abs(anom.values.sum()) < 1e-10, "Anomaly from mean should sum to ~0"

    def test_median_reference(self):
        """Anomaly from median should work without error."""
        ts = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        anom, _ = ts.anomaly(reference="median", plot=False)
        assert isinstance(anom, TimeSeries)

    def test_invalid_reference_raises(self):
        """Unknown reference should raise ValueError."""
        ts = TimeSeries(np.random.randn(50))
        with pytest.raises(ValueError, match="reference must be"):
            ts.anomaly(reference="invalid", plot=False)

    def test_returns_timeseries(self):
        """Anomaly should return a TimeSeries."""
        ts = TimeSeries(np.random.randn(50))
        anom, _ = ts.anomaly(plot=False)
        assert isinstance(anom, TimeSeries)

    def test_plot_returns_figure(self):
        """With plot=True, should return (fig, ax)."""
        ts = TimeSeries(np.random.randn(50))
        _, fig_ax = ts.anomaly(plot=True)
        assert fig_ax is not None
        plt.close(fig_ax[0])

    def test_plot_false_returns_none(self):
        """With plot=False, figure should be None."""
        ts = TimeSeries(np.random.randn(50))
        _, fig_ax = ts.anomaly(plot=False)
        assert fig_ax is None


class TestStandardizedAnomaly:
    """Tests for standardized_anomaly() method."""

    def test_returns_timeseries(self):
        """Should return a TimeSeries."""
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        ts = TimeSeries(np.random.randn(365), index=idx)
        result = ts.standardized_anomaly()
        assert isinstance(result, TimeSeries)

    def test_near_zero_mean(self):
        """Standardized anomaly should have mean near zero."""
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=730, freq="D")
        ts = TimeSeries(np.random.randn(730), index=idx)
        result = ts.standardized_anomaly()
        assert (
            abs(result.values.mean()) < 0.5
        ), f"Standardized anomaly mean should be near 0, got {result.values.mean():.3f}"

    def test_near_unit_std(self):
        """Standardized anomaly should have std near 1."""
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=730, freq="D")
        ts = TimeSeries(np.random.randn(730), index=idx)
        result = ts.standardized_anomaly()
        assert 0.5 < np.nanstd(result.values) < 1.5

    def test_raises_without_datetime_index(self):
        """Should raise TypeError without DatetimeIndex."""
        ts = TimeSeries(np.random.randn(100))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            ts.standardized_anomaly()

    def test_preserves_index(self):
        """Should preserve the DatetimeIndex."""
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        ts = TimeSeries(np.random.randn(365), index=idx)
        result = ts.standardized_anomaly()
        assert (result.index == ts[ts.columns[0]].dropna().index).all()


class TestDoubleMassCurve:
    """Tests for double_mass_curve() method."""

    def test_returns_dataframe(self):
        """Should return a DataFrame with cumulative sums."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        dmc, _ = ts.double_mass_curve("A", "B", plot=False)
        assert isinstance(dmc, DataFrame)
        assert "cumsum_A" in dmc.columns
        assert "cumsum_B" in dmc.columns

    def test_cumulative_sums_increase(self):
        """Cumulative sums of positive data should be monotonically increasing."""
        data = np.abs(np.random.randn(50, 2)) + 0.1
        ts = TimeSeries(data, columns=["X", "Y"])
        dmc, _ = ts.double_mass_curve("X", "Y", plot=False)
        assert (np.diff(dmc["cumsum_X"].values) >= 0).all()

    def test_plot_returns_figure(self):
        """With plot=True, should return (fig, ax)."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        _, fig_ax = ts.double_mass_curve("A", "B", plot=True)
        assert fig_ax is not None
        plt.close(fig_ax[0])

    def test_plot_false_returns_none(self):
        """With plot=False, figure should be None."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        _, fig_ax = ts.double_mass_curve("A", "B", plot=False)
        assert fig_ax is None


class TestRegimeComparison:
    """Tests for regime_comparison() method."""

    def test_detects_mean_shift(self):
        """After segment should have higher mean for upward shift."""
        np.random.seed(42)
        data = np.concatenate([np.random.randn(50), np.random.randn(50) + 5])
        ts = TimeSeries(data)
        result = ts.regime_comparison(split_at=50)
        assert float(result.loc["mean", "after"]) > float(
            result.loc["mean", "before"]
        ), "After-mean should be > before-mean for upward shift"

    def test_returns_expected_rows(self):
        """Should have stat rows plus mann_whitney_U."""
        data = np.random.randn(100)
        ts = TimeSeries(data)
        result = ts.regime_comparison(split_at=50)
        expected_rows = [
            "mean",
            "std",
            "cv",
            "median",
            "min",
            "max",
            "skewness",
            "mann_whitney_U",
        ]
        assert set(expected_rows).issubset(set(result.index))

    def test_returns_expected_columns(self):
        """Should have before, after, relative_change_pct columns."""
        data = np.random.randn(100)
        ts = TimeSeries(data)
        result = ts.regime_comparison(split_at=50)
        expected_cols = ["before", "after", "relative_change_pct"]
        assert result.columns.tolist() == expected_cols

    def test_mann_whitney_significant_for_shift(self):
        """Mann-Whitney p-value should be low for clear mean shift."""
        np.random.seed(42)
        data = np.concatenate([np.random.randn(100), np.random.randn(100) + 5])
        ts = TimeSeries(data)
        result = ts.regime_comparison(split_at=100)
        mw_p = float(result.loc["mann_whitney_U", "after"])
        assert mw_p < 0.05, f"Expected p < 0.05 for clear shift, got {mw_p}"

    def test_mann_whitney_not_significant_for_homogeneous(self):
        """Mann-Whitney p-value should be high for homogeneous data."""
        np.random.seed(42)
        data = np.random.randn(200)
        ts = TimeSeries(data)
        result = ts.regime_comparison(split_at=100)
        mw_p = float(result.loc["mann_whitney_U", "after"])
        assert mw_p > 0.05, f"Expected p > 0.05 for homogeneous data, got {mw_p}"

    def test_relative_change_calculated(self):
        """Relative change should be computed for non-zero before values."""
        data = np.concatenate([np.ones(50) * 10, np.ones(50) * 20])
        ts = TimeSeries(data)
        result = ts.regime_comparison(split_at=50)
        change = float(result.loc["mean", "relative_change_pct"])
        assert change == pytest.approx(100.0), f"Expected 100% change, got {change}"

    def test_column_parameter(self):
        """Should work with column parameter."""
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        result = ts.regime_comparison(split_at=50, column="B")
        assert "mean" in result.index
