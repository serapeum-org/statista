"""Tests for statista.stat_result dataclasses."""

import pytest
from dataclasses import FrozenInstanceError

from statista.stat_result import ChangePointResult, StatTestResult, TrendTestResult


class TestStatTestResult:
    """Tests for the StatTestResult frozen dataclass."""

    @pytest.fixture
    def result(self):
        """Create a StatTestResult instance with typical values."""
        return StatTestResult(
            test_name="Augmented Dickey-Fuller",
            statistic=-3.45,
            p_value=0.009,
            conclusion="Stationary at 5% significance level",
            alpha=0.05,
            details={"used_lag": 2, "n_obs": 100},
        )

    def test_field_access(self, result):
        """All fields should be accessible via attribute access."""
        assert result.test_name == "Augmented Dickey-Fuller", f"Expected 'Augmented Dickey-Fuller', got {result.test_name}"
        assert result.statistic == -3.45, f"Expected -3.45, got {result.statistic}"
        assert result.p_value == 0.009, f"Expected 0.009, got {result.p_value}"
        assert result.conclusion == "Stationary at 5% significance level"
        assert result.alpha == 0.05, f"Expected 0.05, got {result.alpha}"
        assert result.details == {"used_lag": 2, "n_obs": 100}

    def test_defaults(self):
        """Alpha should default to 0.05 and details to empty dict."""
        result = StatTestResult(
            test_name="KS", statistic=0.1, p_value=0.5, conclusion="Accept"
        )
        assert result.alpha == 0.05, f"Expected default alpha 0.05, got {result.alpha}"
        assert result.details == {}, f"Expected empty dict, got {result.details}"

    def test_frozen_immutability(self, result):
        """Frozen dataclass should reject attribute assignment."""
        with pytest.raises(FrozenInstanceError):
            result.p_value = 0.5

    def test_equality(self):
        """Two instances with identical values should be equal."""
        a = StatTestResult(test_name="T", statistic=1.0, p_value=0.05, conclusion="C")
        b = StatTestResult(test_name="T", statistic=1.0, p_value=0.05, conclusion="C")
        assert a == b, "Identical StatTestResult instances should be equal"

    def test_inequality(self):
        """Instances with different values should not be equal."""
        a = StatTestResult(test_name="T", statistic=1.0, p_value=0.05, conclusion="C")
        b = StatTestResult(test_name="T", statistic=2.0, p_value=0.05, conclusion="C")
        assert a != b, "Different StatTestResult instances should not be equal"

    def test_repr_contains_fields(self, result):
        """Repr should include all field names and values."""
        r = repr(result)
        assert "Augmented Dickey-Fuller" in r, f"test_name missing from repr: {r}"
        assert "-3.45" in r, f"statistic missing from repr: {r}"


class TestTrendTestResult:
    """Tests for the TrendTestResult frozen dataclass."""

    @pytest.fixture
    def result(self):
        """Create a TrendTestResult with typical increasing-trend values."""
        return TrendTestResult(
            trend="increasing",
            h=True,
            p_value=0.001,
            z=3.29,
            tau=0.45,
            s=156.0,
            var_s=2200.0,
            slope=0.12,
            intercept=1.5,
        )

    def test_field_access(self, result):
        """All 9 fields should be accessible and hold correct values."""
        assert result.trend == "increasing", f"Expected 'increasing', got {result.trend}"
        assert result.h is True, f"Expected True, got {result.h}"
        assert result.p_value == 0.001, f"Expected 0.001, got {result.p_value}"
        assert result.z == 3.29, f"Expected 3.29, got {result.z}"
        assert result.tau == 0.45, f"Expected 0.45, got {result.tau}"
        assert result.s == 156.0, f"Expected 156.0, got {result.s}"
        assert result.var_s == 2200.0, f"Expected 2200.0, got {result.var_s}"
        assert result.slope == 0.12, f"Expected 0.12, got {result.slope}"
        assert result.intercept == 1.5, f"Expected 1.5, got {result.intercept}"

    def test_no_trend_result(self):
        """Should represent 'no trend' with h=False."""
        result = TrendTestResult(
            trend="no trend", h=False, p_value=0.45, z=0.75,
            tau=0.05, s=10.0, var_s=500.0, slope=0.001, intercept=5.0,
        )
        assert result.trend == "no trend"
        assert result.h is False

    def test_decreasing_trend(self):
        """Should represent decreasing trend with negative z and slope."""
        result = TrendTestResult(
            trend="decreasing", h=True, p_value=0.01, z=-2.58,
            tau=-0.35, s=-120.0, var_s=1800.0, slope=-0.08, intercept=10.0,
        )
        assert result.z < 0, f"Z should be negative for decreasing trend, got {result.z}"
        assert result.slope < 0, f"Slope should be negative for decreasing trend, got {result.slope}"

    def test_frozen_immutability(self, result):
        """Frozen dataclass should reject attribute assignment."""
        with pytest.raises(FrozenInstanceError):
            result.trend = "decreasing"

    def test_equality(self):
        """Two identical TrendTestResult instances should be equal."""
        kwargs = dict(
            trend="increasing", h=True, p_value=0.01, z=2.5,
            tau=0.3, s=100.0, var_s=1000.0, slope=0.1, intercept=0.0,
        )
        assert TrendTestResult(**kwargs) == TrendTestResult(**kwargs)


class TestChangePointResult:
    """Tests for the ChangePointResult frozen dataclass."""

    @pytest.fixture
    def result(self):
        """Create a ChangePointResult with a detected change point."""
        return ChangePointResult(
            test_name="Pettitt",
            h=True,
            change_point=50,
            change_point_date=None,
            statistic=1234.0,
            p_value=0.003,
            mean_before=10.5,
            mean_after=15.2,
        )

    def test_field_access(self, result):
        """All 8 fields should be accessible and hold correct values."""
        assert result.test_name == "Pettitt", f"Expected 'Pettitt', got {result.test_name}"
        assert result.h is True
        assert result.change_point == 50
        assert result.change_point_date is None
        assert result.statistic == 1234.0
        assert result.p_value == 0.003
        assert result.mean_before == 10.5
        assert result.mean_after == 15.2

    def test_no_change_detected(self):
        """Should represent homogeneous series with h=False."""
        result = ChangePointResult(
            test_name="SNHT", h=False, change_point=0,
            change_point_date=None, statistic=2.5,
            p_value=0.65, mean_before=10.0, mean_after=10.1,
        )
        assert result.h is False
        assert result.p_value > 0.05

    def test_with_datetime_change_point(self):
        """change_point_date should accept datetime objects."""
        from datetime import datetime
        dt = datetime(2000, 6, 15)
        result = ChangePointResult(
            test_name="Buishand", h=True, change_point=100,
            change_point_date=dt, statistic=500.0,
            p_value=0.01, mean_before=5.0, mean_after=8.0,
        )
        assert result.change_point_date == dt, f"Expected {dt}, got {result.change_point_date}"

    def test_frozen_immutability(self, result):
        """Frozen dataclass should reject attribute assignment."""
        with pytest.raises(FrozenInstanceError):
            result.change_point = 99

    def test_repr_contains_test_name(self, result):
        """Repr should include the test name."""
        r = repr(result)
        assert "Pettitt" in r, f"test_name missing from repr: {r}"
