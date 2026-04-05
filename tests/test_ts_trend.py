"""Tests for the TrendMixin (Phase 5)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


class TestMannKendall:
    """Tests for mann_kendall() method."""

    def test_increasing_trend_detected(self):
        """Linearly increasing data should be detected as increasing."""
        np.random.seed(42)
        data = np.arange(100, dtype=float) + np.random.randn(100) * 5
        ts = TimeSeries(data)
        result = ts.mann_kendall()
        assert (
            result.loc["Series1", "trend"] == "increasing"
        ), f"Expected increasing, got {result.loc['Series1', 'trend']}"
        assert result.loc["Series1", "h"] == True  # noqa: E712

    def test_decreasing_trend_detected(self):
        """Linearly decreasing data should be detected as decreasing."""
        np.random.seed(42)
        data = -np.arange(100, dtype=float) + np.random.randn(100) * 5
        ts = TimeSeries(data)
        result = ts.mann_kendall()
        assert result.loc["Series1", "trend"] == "decreasing"
        assert result.loc["Series1", "z"] < 0

    def test_no_trend(self):
        """Random data should show no trend."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.mann_kendall()
        assert result.loc["Series1", "trend"] == "no trend"
        assert result.loc["Series1", "h"] == False  # noqa: E712

    def test_positive_slope_for_increasing(self):
        """Slope should be positive for increasing data."""
        np.random.seed(42)
        data = np.arange(50, dtype=float) * 2 + np.random.randn(50)
        ts = TimeSeries(data)
        result = ts.mann_kendall()
        assert result.loc["Series1", "slope"] > 0

    def test_returns_expected_columns(self):
        """Result should have all 9 expected columns."""
        ts = TimeSeries(np.random.randn(50))
        result = ts.mann_kendall()
        expected = [
            "trend",
            "h",
            "p_value",
            "z",
            "tau",
            "s",
            "var_s",
            "slope",
            "intercept",
        ]
        assert set(expected).issubset(set(result.columns))

    def test_hamed_rao_method(self):
        """Hamed-Rao method should run without error and may give different p-value."""
        np.random.seed(42)
        data = np.arange(50, dtype=float) + np.random.randn(50) * 3
        ts = TimeSeries(data)
        result = ts.mann_kendall(method="hamed_rao")
        assert "trend" in result.columns

    def test_yue_wang_method(self):
        """Yue-Wang method should run without error."""
        np.random.seed(42)
        data = np.arange(50, dtype=float) + np.random.randn(50) * 3
        ts = TimeSeries(data)
        result = ts.mann_kendall(method="yue_wang")
        assert "trend" in result.columns

    def test_pre_whitening_method(self):
        """Pre-whitening method should run without error."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.mann_kendall(method="pre_whitening")
        assert "trend" in result.columns

    def test_trend_free_pre_whitening_method(self):
        """Trend-free pre-whitening should run without error."""
        np.random.seed(42)
        data = np.arange(100, dtype=float) + np.random.randn(100) * 10
        ts = TimeSeries(data)
        result = ts.mann_kendall(method="trend_free_pre_whitening")
        assert "trend" in result.columns

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        ts = TimeSeries(np.random.randn(50))
        with pytest.raises(ValueError, match="Unknown method"):
            ts.mann_kendall(method="invalid")

    def test_multi_column(self):
        """Should return one row per column."""
        ts = TimeSeries(np.random.randn(50, 3), columns=["A", "B", "C"])
        result = ts.mann_kendall()
        assert result.shape[0] == 3

    def test_column_parameter(self):
        """Specifying column should only test that column."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["X", "Y"])
        result = ts.mann_kendall(column="Y")
        assert result.shape[0] == 1
        assert result.index[0] == "Y"

    def test_tau_between_minus1_and_1(self):
        """Kendall's tau should be in [-1, 1]."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(50))
        result = ts.mann_kendall()
        tau = result.loc["Series1", "tau"]
        assert -1.0 <= tau <= 1.0, f"Tau should be in [-1, 1], got {tau}"


class TestSensSlope:
    """Tests for sens_slope() method."""

    def test_known_slope(self):
        """Data with slope ~2 should produce sens slope ~2."""
        np.random.seed(42)
        data = np.arange(50, dtype=float) * 2 + np.random.randn(50) * 0.5
        ts = TimeSeries(data)
        result = ts.sens_slope()
        assert result.loc["Series1", "slope"] == pytest.approx(
            2.0, abs=0.3
        ), f"Expected slope ~2.0, got {result.loc['Series1', 'slope']}"

    def test_confidence_interval_contains_true_slope(self):
        """CI should contain the true slope for noiseless data."""
        data = np.arange(30, dtype=float) * 3
        ts = TimeSeries(data)
        result = ts.sens_slope()
        assert (
            result.loc["Series1", "slope_lower_ci"]
            <= 3.0
            <= result.loc["Series1", "slope_upper_ci"]
        )

    def test_returns_expected_columns(self):
        """Result should have slope, intercept, and CI columns."""
        ts = TimeSeries(np.random.randn(50))
        result = ts.sens_slope()
        expected = ["slope", "intercept", "slope_lower_ci", "slope_upper_ci"]
        assert set(expected).issubset(set(result.columns))

    def test_multi_column(self):
        """Should return one row per column."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        result = ts.sens_slope()
        assert result.shape[0] == 2

    def test_flat_data_near_zero_slope(self):
        """Constant data should produce slope ~0."""
        data = np.ones(30) * 5.0
        ts = TimeSeries(data)
        result = ts.sens_slope()
        assert result.loc["Series1", "slope"] == pytest.approx(0.0, abs=1e-10)


class TestDetrend:
    """Tests for detrend() method."""

    def test_linear_detrend(self):
        """Detrended linear data should have near-zero mean."""
        data = np.arange(100, dtype=float)
        ts = TimeSeries(data)
        dt = ts.detrend(method="linear")
        assert (
            abs(dt.values.mean()) < 1e-10
        ), f"Detrended mean should be ~0, got {dt.values.mean()}"

    def test_constant_detrend(self):
        """Constant detrend should subtract the mean."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ts = TimeSeries(data)
        dt = ts.detrend(method="constant")
        assert dt.values.mean() == pytest.approx(0.0, abs=1e-10)

    def test_polynomial_detrend(self):
        """Polynomial detrend should remove quadratic trend."""
        x = np.arange(100, dtype=float)
        data = x**2 + np.random.randn(100) * 10
        ts = TimeSeries(data)
        dt = ts.detrend(method="polynomial", order=2)
        assert abs(dt.values.mean()) < 50, "Quadratic trend should be mostly removed"

    def test_sens_detrend(self):
        """Sens detrend should remove trend using robust estimator."""
        np.random.seed(42)
        data = np.arange(50, dtype=float) * 3 + np.random.randn(50) * 2
        ts = TimeSeries(data)
        dt = ts.detrend(method="sens")
        assert isinstance(dt, TimeSeries)
        # Detrended slope should be much smaller
        from scipy.stats import theilslopes

        original_slope = theilslopes(data, np.arange(50))[0]
        detrended_slope = theilslopes(dt.values.ravel(), np.arange(50))[0]
        assert abs(detrended_slope) < abs(original_slope) * 0.1

    def test_returns_timeseries(self):
        """detrend should return a TimeSeries instance."""
        ts = TimeSeries(np.random.randn(50))
        dt = ts.detrend()
        assert isinstance(dt, TimeSeries), f"Expected TimeSeries, got {type(dt)}"

    def test_preserves_index(self):
        """Detrended series should have the same index."""
        ts = TimeSeries(np.random.randn(50))
        dt = ts.detrend()
        assert (dt.index == ts.index).all()

    def test_preserves_columns(self):
        """Detrended series should have the same column names."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        dt = ts.detrend()
        assert dt.columns.tolist() == ["A", "B"]

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        ts = TimeSeries(np.random.randn(50))
        with pytest.raises(ValueError, match="Unknown method"):
            ts.detrend(method="invalid")


class TestInnovativeTrendAnalysis:
    """Tests for innovative_trend_analysis() method."""

    def test_increasing_data_positive_indicator(self):
        """Increasing data should produce positive trend indicator."""
        data = np.arange(100, dtype=float)
        ts = TimeSeries(data)
        result_df, (fig, ax) = ts.innovative_trend_analysis()
        assert result_df.loc["Series1", "trend_indicator"] > 0
        plt.close(fig)

    def test_returns_dataframe_and_plot(self):
        """Should return (DataFrame, (Figure, Axes))."""
        ts = TimeSeries(np.random.randn(50))
        result_df, (fig, ax) = ts.innovative_trend_analysis()
        assert isinstance(result_df, DataFrame)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_constant_data_zero_indicator(self):
        """Constant data should have trend indicator near zero."""
        data = np.ones(100) * 5.0
        ts = TimeSeries(data)
        result_df, (fig, ax) = ts.innovative_trend_analysis()
        assert result_df.loc["Series1", "trend_indicator"] == pytest.approx(
            0.0, abs=1e-10
        )
        plt.close(fig)

    def test_column_parameter(self):
        """Should work with column parameter."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        result_df, _ = ts.innovative_trend_analysis(column="B")
        assert result_df.index[0] == "B"
        plt.close("all")
