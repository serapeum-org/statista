import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statista.distributions import Gumbel, Parameters, PlottingPosition


class TestGumbel:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        dist = Gumbel(time_series1)
        assert isinstance(dist.data, np.ndarray)
        assert isinstance(dist.data_sorted, np.ndarray)
        assert dist.parameters is None

    def test_create_instance_with_wrong_data_type(self):
        data = {"key": "value"}
        with pytest.raises(TypeError):
            Gumbel(data=data)

    def test_create_instance_with_wrong_parameter_type(self):
        parameters = [1, 2, 3]
        with pytest.raises(TypeError):
            Gumbel(parameters=parameters)

    def test_random(
        self,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: dict[str, dict[str, float]],
    ):
        param = {"loc": 0, "scale": 1}
        dist = Gumbel(parameters=param)
        rv = dist.random(100)
        assert isinstance(rv, np.ndarray)
        assert rv.shape == (100,)

    def test_fit_model(
        self,
        time_series2: list,
        dist_estimation_parameters: list[str],
        gum_dist_parameters: dict[str, float],
    ):
        dist = Gumbel(time_series2)
        for method in dist_estimation_parameters:
            param = dist.fit_model(method=method, test=False)
            assert isinstance(param, Parameters)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert dist.parameters.loc is not None
            assert dist.parameters.scale is not None
            assert param == gum_dist_parameters[method]

    def test_parameter_estimation_optimization(
        self,
        time_series2: list,
        dist_estimation_parameters: list[str],
        parameter_estimation_optimization_threshold: int,
    ):
        dist = Gumbel(time_series2)
        param = dist.fit_model(
            method="optimization",
            obj_func=Gumbel.truncated_distribution,
            threshold=parameter_estimation_optimization_threshold,
        )
        assert isinstance(param, Parameters)
        assert all(i in param.keys() for i in ["loc", "scale"])
        assert dist.parameters.loc is not None
        assert dist.parameters.scale is not None

    def test_ks(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: dict[str, dict[str, float]],
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        dstatic, pvalue = dist.ks()
        assert dstatic == pytest.approx(0.07407407407407407)
        assert pvalue == pytest.approx(0.9987375782247235)

    def test_chisquare(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: dict[str, dict[str, float]],
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        dstatic, p_value = dist.chisquare()
        assert dstatic == pytest.approx(0.5768408126308443)
        assert p_value == pytest.approx(0.7494464539783021)

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: dict[str, dict[str, float]],
        gum_pdf: np.ndarray,
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        pdf, fig, _ = dist.pdf(plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(gum_pdf, pdf)
        assert isinstance(fig, Figure)
        # test if you provide the pdf method with the data parameter
        pdf, fig, _ = dist.pdf(data=time_series2, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(gum_pdf, pdf)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: dict[str, dict[str, float]],
        gum_cdf: np.ndarray,
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        cdf, fig, _ = dist.cdf(plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        np.testing.assert_almost_equal(gum_cdf, cdf)
        assert isinstance(fig, Figure)
        # test if you provide the cdf method with the data parameter
        cdf, fig, _ = dist.cdf(data=time_series2, plot_figure=True)
        assert isinstance(cdf, np.ndarray)

    def test_inverse_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: dict[str, dict[str, float]],
        gev_inverse_cdf: np.ndarray,
        generated_cdf: list[float],
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        qth = dist.inverse_cdf(generated_cdf)
        assert isinstance(qth, np.ndarray)
        np.testing.assert_almost_equal(gev_inverse_cdf, qth)

    def test_confidence_interval(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
        gum_dist_parameters: dict[str, dict[str, float]],
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        cdf_weibul = PlottingPosition.weibul(time_series2)
        # test by providing the cdf function
        upper, lower = dist.confidence_interval(
            prob_non_exceed=cdf_weibul, alpha=confidence_interval_alpha
        )
        assert isinstance(upper, np.ndarray)
        assert isinstance(lower, np.ndarray)
        # test the default parameters
        upper, lower = dist.confidence_interval()
        assert isinstance(upper, np.ndarray)
        assert isinstance(lower, np.ndarray)

        # test with plot_figure
        upper, lower, fig, ax = dist.confidence_interval(plot_figure=True)
        assert isinstance(upper, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
        gum_dist_parameters: dict[str, dict[str, float]],
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        # test default parameters.
        fig, ax = dist.plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax[0], Axes)
        assert isinstance(ax[1], Axes)
        # test with the cdf parameter
        cdf_weibul = PlottingPosition.weibul(time_series2)
        fig, ax = dist.plot(cdf=cdf_weibul)
        assert isinstance(fig, Figure)
        assert isinstance(ax[0], Axes)
        assert isinstance(ax[1], Axes)


class TestGumbelClassInvalid:
    """Tests for uncovered lines in the Gumbel class."""

    def test_scale_parameter_error_in_pdf(self):
        """Test that an error is raised when scale parameter is <= 0 in pdf."""
        gumbel = Gumbel(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="scale must be positive"):
            gumbel.pdf(parameters={"loc": 0, "scale": 0})

    def test_scale_parameter_error_in_cdf(self):
        """Test that an error is raised when scale parameter is <= 0 in cdf."""
        gumbel = Gumbel(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="scale must be positive"):
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
