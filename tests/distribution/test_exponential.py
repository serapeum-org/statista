import numpy as np
import pytest
from matplotlib.figure import Figure

from statista.distributions import Exponential, Parameters


class TestExponential:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        expo_dist = Exponential(time_series1)
        assert isinstance(expo_dist.data, np.ndarray)
        assert isinstance(expo_dist.data_sorted, np.ndarray)

    def test_fit_model(
        self,
        time_series2: list,
        dist_estimation_parameters: list[str],
        exp_dist_parameters: dict[str, float],
    ):
        expo_dist = Exponential(time_series2)
        for method in dist_estimation_parameters:
            param = expo_dist.fit_model(method=method, test=False)
            assert isinstance(param, Parameters)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert expo_dist.parameters.get("loc") is not None
            assert expo_dist.parameters.get("scale") is not None
            assert param == exp_dist_parameters[method]

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        exp_dist_parameters: dict[str, dict[str, float]],
        exp_pdf: np.ndarray,
    ):
        param = exp_dist_parameters[dist_estimation_parameters_ks]
        expo_dist = Exponential(time_series2, param)
        pdf, fig, _ = expo_dist.pdf(plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(exp_pdf, pdf)
        assert isinstance(fig, Figure)
        # test if you provide the pdf method with the data parameter
        pdf, fig, _ = expo_dist.pdf(data=time_series2, plot_figure=True)
        assert isinstance(pdf, np.ndarray)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        exp_dist_parameters: dict[str, dict[str, float]],
        exp_cdf: np.ndarray,
    ):
        param = exp_dist_parameters[dist_estimation_parameters_ks]
        expo_dist = Exponential(time_series2, param)
        cdf, fig, _ = expo_dist.cdf(plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        np.testing.assert_almost_equal(exp_cdf, cdf)
        assert isinstance(fig, Figure)
        # test if you provide the cdf method with the data parameter
        cdf, fig, _ = expo_dist.cdf(data=time_series2, plot_figure=True)
        assert isinstance(cdf, np.ndarray)

    def test_inverse_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        exp_dist_parameters: dict[str, dict[str, float]],
        generated_cdf: list[float],
        exp_inverse_cdf: np.ndarray,
    ):
        param = exp_dist_parameters[dist_estimation_parameters_ks]
        expo_dist = Exponential(time_series2, param)
        qth = expo_dist.inverse_cdf(generated_cdf)
        assert isinstance(qth, np.ndarray)
        np.testing.assert_almost_equal(exp_inverse_cdf, qth)


class TestExponentialClassInvalid:
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
