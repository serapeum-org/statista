"""Tests for statista.distributions.goodness_of_fit.GoodnessOfFitResult."""

from dataclasses import FrozenInstanceError

import pytest

from statista.distributions import GoodnessOfFitResult


class TestGoodnessOfFitResult:
    """Tests for the GoodnessOfFitResult frozen dataclass."""

    @pytest.fixture
    def result(self):
        """Create a GoodnessOfFitResult instance with typical KS-test values."""
        return GoodnessOfFitResult(
            test_name="Kolmogorov-Smirnov",
            statistic=0.019,
            p_value=0.9937,
            conclusion="Accept Hypothesis",
            alpha=0.05,
            details={"kstable": 0.136},
        )

    def test_field_access(self, result):
        """All fields should be accessible via attribute access."""
        assert result.test_name == "Kolmogorov-Smirnov"
        assert result.statistic == 0.019
        assert result.p_value == 0.9937
        assert result.conclusion == "Accept Hypothesis"
        assert result.alpha == 0.05
        assert result.details == {"kstable": 0.136}

    def test_defaults(self):
        """alpha should default to 0.05, conclusion to empty, details to {}."""
        result = GoodnessOfFitResult(test_name="KS", statistic=0.1, p_value=0.5)
        assert result.alpha == 0.05
        assert result.conclusion == ""
        assert result.details == {}

    def test_frozen_immutability(self, result):
        """Frozen dataclass should reject attribute assignment."""
        with pytest.raises(FrozenInstanceError):
            result.p_value = 0.5

    def test_equality(self):
        """Two instances with identical values should be equal."""
        a = GoodnessOfFitResult(test_name="T", statistic=1.0, p_value=0.05)
        b = GoodnessOfFitResult(test_name="T", statistic=1.0, p_value=0.05)
        assert a == b

    def test_tuple_unpacking(self, result):
        """Supports backward-compatible (statistic, p_value) unpacking."""
        stat, p = result
        assert stat == 0.019
        assert p == 0.9937

    def test_iter_yields_statistic_then_pvalue(self, result):
        """__iter__ should yield exactly statistic then p_value."""
        values = list(result)
        assert values == [0.019, 0.9937]

    def test_repr_contains_fields(self, result):
        """Repr should include the test name and statistic."""
        r = repr(result)
        assert "Kolmogorov-Smirnov" in r
        assert "0.019" in r


class TestKSReturnsGoodnessOfFitResult:
    """ks() and chisquare() return GoodnessOfFitResult; tuple unpacking still works."""

    def test_ks_returns_stat_test_result(self, capsys):
        """Gumbel.ks() should return a GoodnessOfFitResult instance."""
        import numpy as np

        from statista.distributions import Gumbel

        np.random.seed(42)
        data = np.random.gumbel(loc=0.0, scale=1.0, size=200)
        dist = Gumbel(data)
        dist.fit_model()
        result = dist.ks()
        capsys.readouterr()  # discard printed output

        assert isinstance(result, GoodnessOfFitResult)
        assert result.test_name == "Kolmogorov-Smirnov"
        assert 0.0 <= result.statistic <= 1.0
        assert 0.0 <= result.p_value <= 1.0

    def test_ks_tuple_unpacking_still_works(self, capsys):
        """Legacy `stat, p = dist.ks()` must continue to work."""
        import numpy as np

        from statista.distributions import Gumbel

        np.random.seed(42)
        data = np.random.gumbel(loc=0.0, scale=1.0, size=200)
        dist = Gumbel(data)
        dist.fit_model()
        stat, p = dist.ks()
        capsys.readouterr()

        assert isinstance(stat, float)
        assert isinstance(p, float)

    def test_chisquare_returns_stat_test_result(self, capsys):
        """Gumbel.chisquare() should return a GoodnessOfFitResult."""
        import numpy as np

        from statista.distributions import Gumbel

        np.random.seed(42)
        data = np.random.gumbel(loc=0.0, scale=1.0, size=200)
        dist = Gumbel(data)
        dist.fit_model()
        result = dist.chisquare()
        capsys.readouterr()

        assert isinstance(result, GoodnessOfFitResult)
        assert result.test_name == "Chi-square"
        assert "ddof" in result.details
