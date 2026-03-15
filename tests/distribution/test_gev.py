import pytest
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statista.distributions import GEV, PlottingPosition


class TestGEV:
    def test_create_gev_instance(
        self,
        time_series1: list,
    ):
        dist = GEV(time_series1)
        assert isinstance(dist.data, np.ndarray)
        assert isinstance(dist.data_sorted, np.ndarray)

    def test_gev_fit_model(
        self,
        time_series1: list,
        dist_estimation_parameters: list[str],
        gev_dist_parameters: dict[str, str],
    ):
        dist = GEV(time_series1)
        for method in dist_estimation_parameters:
            param = dist.fit_model(method=method, test=False)

            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale", "shape"])
            assert dist.parameters.get("loc") is not None
            assert dist.parameters.get("scale") is not None
            assert dist.parameters.get("shape") is not None
            assert param == gev_dist_parameters[method]

    def test_gev_ks(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: dict[str, dict[str, float]],
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        dstatic, pvalue = dist.ks()
        assert dstatic == pytest.approx(0.14814814814814814)
        assert pvalue == pytest.approx(0.9356622290518453)

    def test_gev_chisquare(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: dict[str, dict[str, float]],
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        dstatic, p_value = dist.chisquare()
        assert dstatic == pytest.approx(1.745019092902356)

    def test_gev_pdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: dict[str, dict[str, float]],
        gev_pdf: np.ndarray,
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)

        pdf, fig, _ = dist.pdf(plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(gev_pdf, pdf)
        assert isinstance(fig, Figure)
        # test if you provide the pdf method with the data parameter
        pdf, fig, _ = dist.pdf(data=time_series1, plot_figure=True)
        assert isinstance(pdf, np.ndarray)

    def test_gev_cdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: dict[str, dict[str, float]],
        gev_cdf: np.ndarray,
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        cdf, fig, _ = dist.cdf(plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        np.testing.assert_almost_equal(gev_cdf, cdf)
        assert isinstance(fig, Figure)
        # test if you provide the cdf method with the data parameter
        cdf, fig, _ = dist.cdf(data=time_series1, plot_figure=True)
        assert isinstance(cdf, np.ndarray)

    def test_random(
        self,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: dict[str, dict[str, float]],
    ):
        param = {"loc": 0, "scale": 1, "shape": 0.1}
        dist = GEV(parameters=param)
        rv = dist.random(100)
        assert isinstance(rv, np.ndarray)
        assert rv.shape == (100,)

    def test_gev_inverse_cdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: dict[str, dict[str, float]],
        generated_cdf: list[float],
        gum_inverse_cdf: np.ndarray,
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        qth = dist.inverse_cdf(generated_cdf)
        assert isinstance(qth, np.ndarray)
        np.testing.assert_almost_equal(gum_inverse_cdf, qth)

    def test_gev_confidence_interval(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
        gev_dist_parameters: dict[str, dict[str, float]],
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        cdf_weibul = PlottingPosition.weibul(time_series1)

        upper, lower = dist.confidence_interval(
            prob_non_exceed=cdf_weibul,
            alpha=confidence_interval_alpha,
            n_samples=100,
        )
        assert isinstance(upper, np.ndarray)
        assert isinstance(lower, np.ndarray)
        # test with plot_figure
        upper, lower, fig, ax = dist.confidence_interval(
            prob_non_exceed=cdf_weibul,
            alpha=confidence_interval_alpha,
            plot_figure=True,
        )
        assert isinstance(upper, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_gev_plot(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
        gev_dist_parameters: dict[str, dict[str, float]],
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        # test default parameters.
        fig, ax = dist.plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax[0], Axes)
        assert isinstance(ax[1], Axes)
        # test with the cdf parameter
        cdf_weibul = PlottingPosition.weibul(time_series1)
        fig, ax = dist.plot(cdf=cdf_weibul)
        assert isinstance(fig, Figure)
        assert isinstance(ax[0], Axes)
        assert isinstance(ax[1], Axes)


class TestGEVClassInvalid:
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