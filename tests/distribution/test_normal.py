import numpy as np
import pytest
from matplotlib.figure import Figure

from statista.distributions import Normal, Parameters


class TestNormal:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        norm_dist = Normal(time_series1)
        assert isinstance(norm_dist.data, np.ndarray)
        assert isinstance(norm_dist.data_sorted, np.ndarray)

    def test_fit_model(
        self,
        time_series2: list,
        dist_estimation_parameters: list[str],
        normal_dist_parameters: dict[str, dict[str, float]],
    ):
        norm_dist = Normal(time_series2)
        for method in dist_estimation_parameters:
            param = norm_dist.fit_model(method=method, test=False)
            assert isinstance(param, Parameters)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert norm_dist.parameters.get("loc") is not None
            assert norm_dist.parameters.get("scale") is not None
            assert param == normal_dist_parameters[method]

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        normal_dist_parameters: dict[str, dict[str, float]],
        normal_pdf: np.ndarray,
    ):
        param = normal_dist_parameters[dist_estimation_parameters_ks]
        norm_dist = Normal(time_series2, param)
        pdf, fig, _ = norm_dist.pdf(plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(normal_pdf, pdf)
        assert isinstance(fig, Figure)
        # test if you provide the pdf method with the data parameter
        pdf, fig, _ = norm_dist.pdf(data=time_series2, plot_figure=True)
        assert isinstance(pdf, np.ndarray)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        normal_dist_parameters: dict[str, dict[str, float]],
        normal_cdf: np.ndarray,
    ):
        param = normal_dist_parameters[dist_estimation_parameters_ks]
        norm_dist = Normal(time_series2, param)
        cdf, fig, _ = norm_dist.cdf(plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        np.testing.assert_almost_equal(normal_cdf, cdf)
        assert isinstance(fig, Figure)
        # test if you provide the cdf method with the data parameter
        cdf, fig, _ = norm_dist.cdf(data=time_series2, plot_figure=True)
        assert isinstance(cdf, np.ndarray)

    def test_inverse_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        normal_dist_parameters: dict[str, dict[str, float]],
        generated_cdf: list[float],
        normal_inverse_cdf: np.ndarray,
    ):
        param = normal_dist_parameters[dist_estimation_parameters_ks]
        norm_dist = Normal(time_series2, param)
        qth = norm_dist.inverse_cdf(generated_cdf)
        assert isinstance(qth, np.ndarray)
        np.testing.assert_almost_equal(normal_inverse_cdf, qth)


class TestNormalClassInvalid:
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
