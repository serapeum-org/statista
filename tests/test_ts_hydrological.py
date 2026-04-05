"""Tests for the HydrologicalMixin (Phase 10)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


@pytest.fixture
def ts_flow():
    """Simulated streamflow (positive values)."""
    np.random.seed(42)
    return TimeSeries(np.abs(np.random.randn(365)) * 50 + 10)


class TestFlowDurationCurve:
    """Tests for flow_duration_curve() method."""

    def test_returns_dataframe(self, ts_flow):
        """Should return a DataFrame with value and exceedance_pct."""
        fdc, _ = ts_flow.flow_duration_curve(plot=False)
        assert isinstance(fdc, DataFrame)
        assert "exceedance_pct" in fdc.columns

    def test_values_sorted_descending(self, ts_flow):
        """Values in FDC should be sorted descending."""
        fdc, _ = ts_flow.flow_duration_curve(plot=False)
        values = fdc["value"].values
        assert np.all(values[:-1] >= values[1:]), "FDC values should be descending"

    def test_exceedance_range(self, ts_flow):
        """Exceedance probabilities should be between 0 and 100."""
        fdc, _ = ts_flow.flow_duration_curve(plot=False)
        assert fdc["exceedance_pct"].min() > 0
        assert fdc["exceedance_pct"].max() < 100

    def test_weibull_method(self, ts_flow):
        """Weibull plotting position should work."""
        fdc, _ = ts_flow.flow_duration_curve(method="weibull", plot=False)
        assert len(fdc) > 0

    def test_gringorten_method(self, ts_flow):
        """Gringorten plotting position should work."""
        fdc, _ = ts_flow.flow_duration_curve(method="gringorten", plot=False)
        assert len(fdc) > 0

    def test_cunnane_method(self, ts_flow):
        """Cunnane plotting position should work."""
        fdc, _ = ts_flow.flow_duration_curve(method="cunnane", plot=False)
        assert len(fdc) > 0

    def test_invalid_method_raises(self, ts_flow):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            ts_flow.flow_duration_curve(method="invalid", plot=False)

    def test_plot_returns_figure(self, ts_flow):
        """With plot=True, should return (fig, ax)."""
        _, fig_ax = ts_flow.flow_duration_curve(plot=True)
        assert fig_ax is not None
        plt.close(fig_ax[0])

    def test_multi_column(self):
        """Multi-column should overlay all series."""
        ts = TimeSeries(np.abs(np.random.randn(100, 2)) * 10, columns=["A", "B"])
        fdc, _ = ts.flow_duration_curve(plot=False)
        assert "A" in fdc.columns and "B" in fdc.columns


class TestAnnualExtremes:
    """Tests for annual_extremes() method."""

    def test_returns_timeseries(self):
        """Should return a TimeSeries."""
        idx = pd.date_range("2000-01-01", periods=730, freq="D")
        ts = TimeSeries(np.random.randn(730), index=idx)
        result = ts.annual_extremes(kind="max")
        assert isinstance(result, TimeSeries)

    def test_max_extraction(self):
        """Annual max values should all appear in the original data."""
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=730, freq="D")
        data = np.random.randn(730) * 10
        ts = TimeSeries(data, index=idx)
        ams = ts.annual_extremes(kind="max")
        for val in ams.values.ravel():
            assert val in data, f"AMS value {val} should be in original data"

    def test_min_extraction(self):
        """Annual min values should all appear in the original data."""
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=730, freq="D")
        data = np.random.randn(730) * 10
        ts = TimeSeries(data, index=idx)
        ams = ts.annual_extremes(kind="min")
        for val in ams.values.ravel():
            assert val in data, f"Min value {val} should be in original data"

    def test_invalid_kind_raises(self):
        """Invalid kind should raise ValueError."""
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        ts = TimeSeries(np.random.randn(365), index=idx)
        with pytest.raises(ValueError, match="kind must be"):
            ts.annual_extremes(kind="median")


class TestExceedanceProbability:
    """Tests for exceedance_probability() method."""

    def test_returns_expected_columns(self, ts_flow):
        """Result should have value, exceedance_probability, return_period."""
        result = ts_flow.exceedance_probability()
        expected = ["value", "exceedance_probability", "return_period"]
        assert set(expected).issubset(set(result.columns))

    def test_exceedance_between_0_and_1(self, ts_flow):
        """Exceedance probabilities should be in (0, 1)."""
        result = ts_flow.exceedance_probability()
        assert result["exceedance_probability"].min() > 0
        assert result["exceedance_probability"].max() < 1

    def test_return_period_positive(self, ts_flow):
        """Return periods should all be positive."""
        result = ts_flow.exceedance_probability()
        assert (result["return_period"] > 0).all()

    def test_multi_column(self):
        """Multi-column should produce stacked results."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        result = ts.exceedance_probability()
        assert "column" in result.columns


class TestBaseflowSeparation:
    """Tests for baseflow_separation() method."""

    def test_lyne_hollick(self, ts_flow):
        """Lyne-Hollick method should produce baseflow <= total flow."""
        result, _ = ts_flow.baseflow_separation(method="lyne_hollick", plot=False)
        assert (result["baseflow"] <= result["total_flow"] + 1e-10).all()
        assert (result["baseflow"] >= 0).all()

    def test_eckhardt(self, ts_flow):
        """Eckhardt method should produce valid baseflow."""
        result, _ = ts_flow.baseflow_separation(method="eckhardt", plot=False)
        assert (result["baseflow"] <= result["total_flow"] + 1e-10).all()

    def test_chapman_maxwell(self, ts_flow):
        """Chapman-Maxwell method should produce valid baseflow."""
        result, _ = ts_flow.baseflow_separation(method="chapman_maxwell", plot=False)
        assert (result["baseflow"] <= result["total_flow"] + 1e-10).all()

    def test_returns_expected_columns(self, ts_flow):
        """Result should have total_flow, baseflow, quickflow."""
        result, _ = ts_flow.baseflow_separation(plot=False)
        expected = ["total_flow", "baseflow", "quickflow"]
        assert result.columns.tolist() == expected

    def test_quickflow_is_difference(self, ts_flow):
        """quickflow should equal total_flow - baseflow."""
        result, _ = ts_flow.baseflow_separation(plot=False)
        diff = result["total_flow"] - result["baseflow"] - result["quickflow"]
        assert np.allclose(diff.values, 0, atol=1e-10)

    def test_invalid_method_raises(self, ts_flow):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            ts_flow.baseflow_separation(method="invalid", plot=False)

    def test_plot_returns_figure(self, ts_flow):
        """With plot=True, should return (fig, ax)."""
        _, fig_ax = ts_flow.baseflow_separation(plot=True)
        assert fig_ax is not None
        plt.close(fig_ax[0])


class TestBaseflowIndex:
    """Tests for baseflow_index() method."""

    def test_bfi_between_0_and_1(self, ts_flow):
        """BFI should be between 0 and 1."""
        result = ts_flow.baseflow_index()
        bfi = result.loc["Series1", "bfi"]
        assert 0.0 <= bfi <= 1.0, f"BFI should be in [0, 1], got {bfi}"

    def test_multi_column(self):
        """Should return one BFI per column."""
        ts = TimeSeries(np.abs(np.random.randn(100, 2)) * 10 + 1, columns=["A", "B"])
        result = ts.baseflow_index()
        assert result.shape[0] == 2


class TestFlashinessIndex:
    """Tests for flashiness_index() method."""

    def test_constant_flow_zero_flashiness(self):
        """Constant flow should have flashiness = 0."""
        ts = TimeSeries(np.ones(100) * 10.0)
        result = ts.flashiness_index()
        assert result.loc["Series1", "flashiness"] == pytest.approx(0.0)

    def test_oscillating_flow_high_flashiness(self):
        """Rapidly oscillating flow should have high flashiness."""
        data = np.array([10.0, 100.0, 10.0, 100.0, 10.0])
        ts = TimeSeries(data)
        result = ts.flashiness_index()
        assert result.loc["Series1", "flashiness"] > 1.0

    def test_positive_values(self, ts_flow):
        """Flashiness should be non-negative."""
        result = ts_flow.flashiness_index()
        assert result.loc["Series1", "flashiness"] >= 0

    def test_multi_column(self):
        """Should return one row per column."""
        ts = TimeSeries(np.abs(np.random.randn(50, 2)) * 10, columns=["A", "B"])
        result = ts.flashiness_index()
        assert result.shape[0] == 2
