import numpy as np
import pytest

from statista.distributions import (
    GEV,
    Distributions,
    Exponential,
    Gumbel,
    Normal,
)


class TestDistributionsClass:
    """Tests for the Distributions class."""

    def test_invalid_distribution(self):
        """Test that an error is raised when an invalid distribution is provided."""
        with pytest.raises(ValueError, match="InvalidDist not supported"):
            Distributions("InvalidDist", data=[1, 2, 3, 4, 5])

    def test_invalid_attribute(self):
        """Test that an error is raised when accessing a non-existent attribute."""
        dist = Distributions("Gumbel", data=[1, 2, 3, 4, 5])
        with pytest.raises(
            AttributeError,
            match="'Distributions' object has no attribute 'invalid_method'",
        ):
            dist.invalid_method()


class TestGumbelClass:
    """Tests for uncovered lines in the Gumbel class."""

    def test_scale_parameter_error_in_pdf(self):
        """Test that an error is raised when scale parameter is <= 0 in pdf."""
        gumbel = Gumbel(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Scale parameter is negative"):
            gumbel.pdf(parameters={"loc": 0, "scale": 0})

    def test_scale_parameter_error_in_cdf(self):
        """Test that an error is raised when scale parameter is <= 0 in cdf."""
        gumbel = Gumbel(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Scale parameter is negative"):
            gumbel.cdf(parameters={"loc": 0, "scale": -1})

    def test_cdf_out_of_range_raises(self):
        """Test that out-of-range CDF values raise ValueError."""
        gumbel = Gumbel(parameters={"loc": 0, "scale": 1})
        with pytest.raises(ValueError):
            gumbel.inverse_cdf(cdf=[0.5, 2])
        with pytest.raises(ValueError):
            gumbel.inverse_cdf(cdf=[-0.1, 0.5])

    def test_cdf_boundary_values(self):
        """Test that boundary CDF values (0 and 1) are accepted."""
        gumbel = Gumbel(parameters={"loc": 0, "scale": 1})
        result = gumbel.inverse_cdf(cdf=[0, 0.5, 1])
        assert np.isneginf(result[0])
        assert np.isposinf(result[2])


class TestGEVClass:
    """Tests for uncovered lines in the GEV class."""

    def test_cdf_out_of_range_raises(self):
        """Test that out-of-range CDF values raise ValueError."""
        gev = GEV(parameters={"loc": 0, "scale": 1, "shape": 0.1})
        with pytest.raises(ValueError):
            gev.inverse_cdf(cdf=[0.5, 2])
        with pytest.raises(ValueError):
            gev.inverse_cdf(cdf=[-0.1, 0.5])

    def test_cdf_boundary_values(self):
        """Test that boundary CDF values (0 and 1) are accepted."""
        gev = GEV(parameters={"loc": 0, "scale": 1, "shape": 0.1})
        result = gev.inverse_cdf(cdf=[0, 0.5, 1])
        assert np.isneginf(result[0])

    def test_invalid_method_in_fit_model(self):
        """Test that an error is raised when an invalid method is provided to fit_model."""
        gev = GEV(data=[1, 2, 3, 4, 5])
        with pytest.raises(
            ValueError,
            match="invalid_method value should be 'mle', 'mm', 'lmoments' or 'optimization'",
        ):
            gev.fit_model(method="invalid_method")


class TestExponentialClass:
    """Tests for uncovered lines in the Exponential class."""

    def test_scale_parameter_error_in_pdf(self):
        """Test that an error is raised when scale parameter is <= 0 in pdf."""
        exp = Exponential(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Scale parameter is negative"):
            exp.pdf(parameters={"loc": 0, "scale": 0})

    def test_scale_parameter_error_in_cdf(self):
        """Test that an error is raised when scale parameter is <= 0 in cdf."""
        exp = Exponential(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Scale parameter is negative"):
            exp.cdf(parameters={"loc": 0, "scale": -1})

    def test_cdf_out_of_range_raises(self):
        """Test that out-of-range CDF values raise ValueError."""
        exp = Exponential(parameters={"loc": 0, "scale": 1})
        with pytest.raises(ValueError):
            exp.inverse_cdf(cdf=[0.5, 2])
        with pytest.raises(ValueError):
            exp.inverse_cdf(cdf=[-0.1, 0.5])

    def test_cdf_boundary_values(self):
        """Test that boundary CDF values (0 and 1) are accepted."""
        exp = Exponential(parameters={"loc": 0, "scale": 1})
        result = exp.inverse_cdf(cdf=[0, 0.5, 1])
        assert result[0] == 0.0


class TestNormalClass:
    """Tests for uncovered lines in the Normal class."""

    def test_scale_parameter_error_in_pdf(self):
        """Test that an error is raised when scale parameter is <= 0 in pdf."""
        norm = Normal(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Scale parameter is negative"):
            norm.pdf(parameters={"loc": 0, "scale": 0})

    def test_scale_parameter_error_in_cdf(self):
        """Test that an error is raised when scale parameter is <= 0 in cdf."""
        norm = Normal(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Scale parameter is negative"):
            norm.cdf(parameters={"loc": 0, "scale": -1})

    def test_cdf_out_of_range_raises(self):
        """Test that out-of-range CDF values raise ValueError."""
        norm = Normal(parameters={"loc": 1, "scale": 1})
        with pytest.raises(ValueError):
            norm.inverse_cdf(cdf=[0.5, 2])
        with pytest.raises(ValueError):
            norm.inverse_cdf(cdf=[-0.1, 0.5])

    def test_cdf_boundary_values(self):
        """Test that boundary CDF values (0 and 1) are accepted."""
        norm = Normal(parameters={"loc": 1, "scale": 1})
        result = norm.inverse_cdf(cdf=[0, 0.5, 1])
        assert np.isneginf(result[0])
        assert np.isposinf(result[2])

    def test_invalid_method_in_fit_model(self):
        """Test that an error is raised when an invalid method is provided to fit_model."""
        norm = Normal(data=[1, 2, 3, 4, 5])
        with pytest.raises(
            ValueError,
            match="invalid_method value should be 'mle', 'mm', 'lmoments' or 'optimization'",
        ):
            norm.fit_model(method="invalid_method")

    def test_optimization_without_obj_func(self):
        """Test that an error is raised when optimization method is used without obj_func."""
        norm = Normal(data=[1, 2, 3, 4, 5])
        with pytest.raises(TypeError, match="threshold should be numeric value"):
            norm.fit_model(method="optimization")
