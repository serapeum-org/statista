"""Tests for the ChangePointMixin (Phase 7)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


@pytest.fixture
def ts_with_shift():
    """TimeSeries with a mean shift at index 50."""
    np.random.seed(42)
    data = np.concatenate([np.random.randn(50), np.random.randn(50) + 3])
    return TimeSeries(data)


@pytest.fixture
def ts_homogeneous():
    """TimeSeries with no change point (white noise)."""
    np.random.seed(42)
    return TimeSeries(np.random.randn(100))


class TestPettittTest:
    """Tests for pettitt_test() method."""

    def test_detects_shift(self, ts_with_shift):
        """Should detect the change point near index 50."""
        result = ts_with_shift.pettitt_test()
        cp = int(result.loc["Series1", "change_point_index"])
        assert 40 <= cp <= 60, f"Expected CP near 50, got {cp}"
        assert result.loc["Series1", "conclusion"] == "Inhomogeneous"

    def test_homogeneous_data(self, ts_homogeneous):
        """Homogeneous data should not reject."""
        result = ts_homogeneous.pettitt_test()
        assert result.loc["Series1", "conclusion"] == "Homogeneous"

    def test_returns_expected_columns(self, ts_homogeneous):
        """Result should have all expected columns."""
        result = ts_homogeneous.pettitt_test()
        expected = [
            "h",
            "change_point_index",
            "statistic",
            "p_value",
            "mean_before",
            "mean_after",
            "conclusion",
        ]
        assert set(expected).issubset(set(result.columns))

    def test_mean_before_after(self, ts_with_shift):
        """mean_before should be < mean_after for upward shift."""
        result = ts_with_shift.pettitt_test()
        assert (
            result.loc["Series1", "mean_before"] < result.loc["Series1", "mean_after"]
        )

    def test_multi_column(self):
        """Should return one row per column."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        result = ts.pettitt_test()
        assert result.shape[0] == 2

    def test_column_parameter(self, ts_with_shift):
        """Specifying column should only test that column."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["X", "Y"])
        result = ts.pettitt_test(column="X")
        assert result.shape[0] == 1
        assert result.index[0] == "X"


class TestSNHTTest:
    """Tests for snht_test() method."""

    def test_detects_shift(self, ts_with_shift):
        """Should detect the change point near index 50."""
        result = ts_with_shift.snht_test()
        cp = int(result.loc["Series1", "change_point_index"])
        assert 40 <= cp <= 60, f"Expected CP near 50, got {cp}"

    def test_homogeneous_data(self, ts_homogeneous):
        """Homogeneous data should not reject."""
        result = ts_homogeneous.snht_test()
        assert result.loc["Series1", "conclusion"] == "Homogeneous"

    def test_returns_expected_columns(self, ts_homogeneous):
        """Result should have expected columns."""
        result = ts_homogeneous.snht_test()
        expected = [
            "h",
            "change_point_index",
            "statistic",
            "p_value",
            "mean_before",
            "mean_after",
            "conclusion",
        ]
        assert set(expected).issubset(set(result.columns))

    def test_constant_data(self):
        """Constant data (zero std) should be homogeneous."""
        ts = TimeSeries(np.ones(50) * 5.0)
        result = ts.snht_test()
        assert result.loc["Series1", "conclusion"] == "Homogeneous"


class TestBuishandRangeTest:
    """Tests for buishand_range_test() method."""

    def test_detects_shift(self, ts_with_shift):
        """Should detect the change point near index 50."""
        result = ts_with_shift.buishand_range_test()
        cp = int(result.loc["Series1", "change_point_index"])
        assert 40 <= cp <= 60, f"Expected CP near 50, got {cp}"

    def test_homogeneous_data(self, ts_homogeneous):
        """Homogeneous data should not reject."""
        result = ts_homogeneous.buishand_range_test()
        assert result.loc["Series1", "conclusion"] == "Homogeneous"

    def test_returns_expected_columns(self, ts_homogeneous):
        """Result should have expected columns."""
        result = ts_homogeneous.buishand_range_test()
        expected = [
            "h",
            "change_point_index",
            "statistic",
            "p_value",
            "mean_before",
            "mean_after",
            "conclusion",
        ]
        assert set(expected).issubset(set(result.columns))

    def test_constant_data(self):
        """Constant data should be homogeneous."""
        ts = TimeSeries(np.ones(50) * 5.0)
        result = ts.buishand_range_test()
        assert result.loc["Series1", "conclusion"] == "Homogeneous"


class TestCUSUM:
    """Tests for cusum() method."""

    def test_returns_dataframe(self, ts_homogeneous):
        """cusum() should return a DataFrame of cumulative sums."""
        cusum_df, _ = ts_homogeneous.cusum(plot=False)
        assert isinstance(cusum_df, DataFrame)
        assert cusum_df.shape[0] == 100

    def test_cusum_starts_near_zero(self, ts_homogeneous):
        """First CUSUM value should be close to zero (one deviation from mean)."""
        cusum_df, _ = ts_homogeneous.cusum(plot=False)
        assert abs(cusum_df.iloc[0, 0]) < 5.0

    def test_plot_returns_figure(self, ts_homogeneous):
        """With plot=True, should return (fig, ax)."""
        cusum_df, fig_ax = ts_homogeneous.cusum(plot=True)
        assert fig_ax is not None
        fig, ax = fig_ax
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_false_returns_none(self, ts_homogeneous):
        """With plot=False, figure should be None."""
        _, fig_ax = ts_homogeneous.cusum(plot=False)
        assert fig_ax is None

    def test_shift_shows_drift(self, ts_with_shift):
        """CUSUM of shifted data should show a clear drift."""
        cusum_df, _ = ts_with_shift.cusum(plot=False)
        # The cusum should deviate significantly from 0
        max_abs = np.max(np.abs(cusum_df.values))
        assert (
            max_abs > 10
        ), f"Expected large CUSUM deviation for shifted data, got {max_abs}"


class TestHomogeneitySummary:
    """Tests for homogeneity_summary() method."""

    def test_confirmed_for_clear_shift(self, ts_with_shift):
        """Clear mean shift should be confirmed by multiple tests."""
        result = ts_with_shift.homogeneity_summary()
        assert result.loc["Series1", "confirmed"] == True  # noqa: E712

    def test_not_confirmed_for_homogeneous(self, ts_homogeneous):
        """Homogeneous data should not be confirmed as having a change point."""
        result = ts_homogeneous.homogeneity_summary()
        assert result.loc["Series1", "confirmed"] == False  # noqa: E712

    def test_returns_expected_columns(self, ts_homogeneous):
        """Result should have all test results and confirmation."""
        result = ts_homogeneous.homogeneity_summary()
        expected = [
            "pettitt_cp",
            "pettitt_p",
            "snht_cp",
            "snht_p",
            "buishand_cp",
            "buishand_p",
            "confirmed",
        ]
        assert set(expected).issubset(set(result.columns))

    def test_multi_column(self):
        """Should return one row per column."""
        ts = TimeSeries(np.random.randn(100, 3), columns=["A", "B", "C"])
        result = ts.homogeneity_summary()
        assert result.shape[0] == 3

    def test_change_points_near_each_other(self, ts_with_shift):
        """For clear shift, all three tests should detect similar change points."""
        result = ts_with_shift.homogeneity_summary()
        cps = [
            int(result.loc["Series1", "pettitt_cp"]),
            int(result.loc["Series1", "snht_cp"]),
            int(result.loc["Series1", "buishand_cp"]),
        ]
        spread = max(cps) - min(cps)
        assert spread <= 5, f"Change points should be close, spread was {spread}"
