"""Tests for the StationarityMixin (Phase 4)."""

import numpy as np
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


class TestADFTest:
    """Tests for adf_test() method."""

    def test_stationary_data_rejects(self):
        """White noise should reject the null (non-stationary) at 5%."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(300))
        result = ts.adf_test()
        assert (
            result.loc["Series1", "p_value"] < 0.10
        ), f"ADF should reject for white noise, p={result.loc['Series1', 'p_value']}"

    def test_random_walk_fails_to_reject(self):
        """Random walk should NOT reject the null (non-stationary)."""
        np.random.seed(42)
        rw = np.cumsum(np.random.randn(300))
        ts = TimeSeries(rw)
        result = ts.adf_test()
        assert (
            result.loc["Series1", "p_value"] > 0.10
        ), f"ADF should not reject for random walk, p={result.loc['Series1', 'p_value']}"

    def test_returns_expected_columns(self):
        """Result DataFrame should have all expected columns."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.adf_test()
        expected = [
            "statistic",
            "p_value",
            "used_lag",
            "n_obs",
            "crit_1%",
            "crit_5%",
            "crit_10%",
            "conclusion",
        ]
        assert set(expected).issubset(
            set(result.columns)
        ), f"Missing columns: {set(expected) - set(result.columns)}"

    def test_conclusion_string(self):
        """Conclusion should be either 'Stationary' or 'Non-stationary'."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.adf_test()
        assert result.loc["Series1", "conclusion"] in ["Stationary", "Non-stationary"]

    def test_regression_ct(self):
        """regression='ct' should use constant + trend critical values."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        result = ts.adf_test(regression="ct")
        assert (
            result.loc["Series1", "crit_5%"] < -3.0
        ), "ct critical values should be more negative than c"

    def test_regression_n(self):
        """regression='n' should use no-constant critical values."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        result = ts.adf_test(regression="n")
        assert (
            result.loc["Series1", "crit_5%"] > -2.5
        ), "n critical values should be less negative than c"

    def test_multi_column(self):
        """Multi-column TimeSeries should return one row per column."""
        ts = TimeSeries(np.random.randn(100, 3), columns=["A", "B", "C"])
        result = ts.adf_test()
        assert result.shape[0] == 3, f"Expected 3 rows, got {result.shape[0]}"

    def test_column_parameter(self):
        """Specifying column should only test that column."""
        ts = TimeSeries(np.random.randn(100, 2), columns=["X", "Y"])
        result = ts.adf_test(column="X")
        assert result.shape[0] == 1
        assert result.index[0] == "X"

    def test_max_lag_parameter(self):
        """Custom max_lag should be used."""
        ts = TimeSeries(np.random.randn(200))
        result = ts.adf_test(max_lag=5)
        assert result.loc["Series1", "used_lag"] <= 5

    def test_returns_dataframe(self):
        """Return type should be DataFrame."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.adf_test()
        assert isinstance(result, DataFrame)


class TestKPSSTest:
    """Tests for kpss_test() method."""

    def test_stationary_data_does_not_reject(self):
        """White noise should NOT reject KPSS null (stationary)."""
        np.random.seed(99)
        ts = TimeSeries(np.random.randn(300))
        result = ts.kpss_test()
        # KPSS null is stationarity, so large p-value = stationary
        assert (
            result.loc["Series1", "p_value"] > 0.01
        ), f"KPSS should not reject for white noise, p={result.loc['Series1', 'p_value']}"

    def test_random_walk_rejects(self):
        """Random walk should reject KPSS null (stationary)."""
        np.random.seed(42)
        rw = np.cumsum(np.random.randn(300))
        ts = TimeSeries(rw)
        result = ts.kpss_test()
        assert (
            result.loc["Series1", "p_value"] <= 0.05
        ), f"KPSS should reject for random walk, p={result.loc['Series1', 'p_value']}"

    def test_returns_expected_columns(self):
        """Result should have expected columns."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.kpss_test()
        expected = [
            "statistic",
            "p_value",
            "lags",
            "crit_10%",
            "crit_5%",
            "crit_2.5%",
            "crit_1%",
            "conclusion",
        ]
        assert set(expected).issubset(set(result.columns))

    def test_regression_ct(self):
        """regression='ct' should use trend-stationarity critical values."""
        ts = TimeSeries(np.random.randn(200))
        result = ts.kpss_test(regression="ct")
        assert (
            result.loc["Series1", "crit_5%"] < 0.2
        ), "ct critical values should be smaller than c"

    def test_multi_column(self):
        """Should return one row per column."""
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        result = ts.kpss_test()
        assert result.shape[0] == 2

    def test_returns_dataframe(self):
        """Return type should be DataFrame."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.kpss_test()
        assert isinstance(result, DataFrame)


class TestStationaritySummary:
    """Tests for stationarity_summary() method."""

    def test_stationary_diagnosis(self):
        """White noise should be diagnosed as Stationary or Trend-stationary."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(300))
        result = ts.stationarity_summary()
        assert result.loc["Series1", "diagnosis"] in [
            "Stationary",
            "Trend-stationary",
        ], f"Expected Stationary/Trend-stationary, got {result.loc['Series1', 'diagnosis']}"

    def test_nonstationary_diagnosis(self):
        """Random walk should be diagnosed as Non-stationary (unit root)."""
        np.random.seed(42)
        rw = np.cumsum(np.random.randn(300))
        ts = TimeSeries(rw)
        result = ts.stationarity_summary()
        assert (
            "Non-stationary" in result.loc["Series1", "diagnosis"]
        ), f"Expected Non-stationary, got {result.loc['Series1', 'diagnosis']}"

    def test_returns_expected_columns(self):
        """Result should have diagnosis and both test stats."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.stationarity_summary()
        expected = ["adf_stat", "adf_pvalue", "kpss_stat", "kpss_pvalue", "diagnosis"]
        assert set(expected).issubset(set(result.columns))

    def test_multi_column(self):
        """Should return one row per column."""
        ts = TimeSeries(np.random.randn(100, 3), columns=["A", "B", "C"])
        result = ts.stationarity_summary()
        assert result.shape[0] == 3

    def test_diagnosis_values(self):
        """All diagnoses should be one of the four valid options."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.stationarity_summary()
        valid = {
            "Stationary",
            "Non-stationary (unit root)",
            "Trend-stationary",
            "Inconclusive",
        }
        assert result.loc["Series1", "diagnosis"] in valid
