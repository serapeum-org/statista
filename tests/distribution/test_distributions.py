"""Test distributions module."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from statista.distributions import (
    Distributions,
    Gumbel,
    PlottingPosition,
)


class TestPlottingPosition:
    def test_plotting_position_weibul(
        self,
        time_series1: list,
    ):
        cdf = PlottingPosition.weibul(time_series1)
        assert isinstance(cdf, np.ndarray)
        rp = PlottingPosition.weibul(time_series1, return_period=True)
        assert isinstance(rp, np.ndarray)

    def test_plotting_position_rp(
        self,
        time_series1: list,
    ):
        cdf = PlottingPosition.weibul(time_series1)
        rp = PlottingPosition.return_period(cdf)
        assert isinstance(rp, np.ndarray)


class TestAbstractDistribution:
    def test_abstract_distribution(self, time_series1: list, gev_dist_parameters):
        text_1 = "\n                    Dataset of 27 value\n                    min: 15.790480003140171\n                    max: 19.39645340792385\n                    mean: 16.929171461473548\n                    median: 16.626465201654593\n                    mode: 15.999737471905252\n                    std: 1.0211514099144634\n                    Distribution : Gumbel\n                    parameters: None\n                    "
        parameters = gev_dist_parameters["lmoments"]
        dist = Gumbel(time_series1)
        assert str(dist) == text_1

        text_2 = (
            "\n                Distribution : Gumbel\n                parameters: {'loc': 16.392889171307772, "
            "'scale': 0.7005442761744839, 'shape': -0.1614793298009645}\n                "
        )
        dist = Gumbel(parameters=parameters)
        assert str(dist) == text_2
        dist = Gumbel(data=time_series1, parameters=parameters)
        text_3 = "\n                    Dataset of 27 value\n                    min: 15.790480003140171\n                    max: 19.39645340792385\n                    mean: 16.929171461473548\n                    median: 16.626465201654593\n                    mode: 15.999737471905252\n                    std: 1.0211514099144634\n                    Distribution : Gumbel\n                    parameters: {'loc': 16.392889171307772, 'scale': 0.7005442761744839, 'shape': -0.1614793298009645}\n                    \n                Distribution : Gumbel\n                parameters: {'loc': 16.392889171307772, 'scale': 0.7005442761744839, 'shape': -0.1614793298009645}\n                "
        assert str(dist) == text_3


class TestDistribution:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        dist = Distributions("Gumbel", data=time_series1)
        assert isinstance(dist.data, np.ndarray)
        assert isinstance(dist.data_sorted, np.ndarray)

    def test_getter_method(
        self,
        time_series2: list,
        dist_estimation_parameters: list[str],
    ):
        dist = Distributions("Gumbel", data=time_series2)
        for i in range(len(dist_estimation_parameters)):
            param = dist.fit_model(method=dist_estimation_parameters[i], test=False)
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert dist.parameters.get("loc") is not None
            assert dist.parameters.get("scale") is not None

    def test_create_without_distribution(self, time_series2: list):
        """Test creating Distributions with data only (no distribution name)."""
        dist = Distributions(data=time_series2)
        assert dist._data is not None
        assert dist.distribution is None

    def test_create_without_distribution_or_data(self):
        """Test that creating Distributions without distribution or data raises."""
        with pytest.raises(
            ValueError, match="Either distribution or data must be provided"
        ):
            Distributions()

    def test_no_delegation_without_distribution(self, time_series2: list):
        """Test that attribute delegation raises when no distribution is set."""
        dist = Distributions(data=time_series2)
        with pytest.raises(
            AttributeError,
            match="'Distributions' object has no attribute 'fit_model'",
        ):
            dist.fit_model()


class TestDistributionsClassInvalid:
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


class TestFitAll:
    """Tests for the Distributions.fit_all method."""

    def test_fit_all_default(self, time_series2: list):
        """Test fit_all with default parameters fits all distributions."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        assert set(results.keys()) == {"GEV", "Gumbel", "Exponential", "Normal"}
        for name, info in results.items():
            assert "distribution" in info
            assert "parameters" in info
            assert "ks" in info
            assert "chisquare" in info
            assert isinstance(info["parameters"], dict)
            assert "loc" in info["parameters"]
            assert "scale" in info["parameters"]
            assert len(info["ks"]) == 2
            assert len(info["chisquare"]) == 2

    def test_fit_all_selected_distributions(self, time_series2: list):
        """Test fit_all with a subset of distributions."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all(distributions=["Gumbel", "GEV"])
        assert set(results.keys()) == {"Gumbel", "GEV"}

    def test_fit_all_mle_method(self, time_series2: list):
        """Test fit_all with MLE method."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all(method="mle")
        assert len(results) == 4
        for info in results.values():
            assert isinstance(info["parameters"], dict)

    def test_fit_all_invalid_distribution(self, time_series2: list):
        """Test fit_all raises ValueError for invalid distribution name."""
        dist = Distributions(data=time_series2)
        with pytest.raises(ValueError, match="InvalidDist not supported"):
            dist.fit_all(distributions=["InvalidDist"])

    def test_fit_all_handles_nan(self):
        """Test fit_all removes NaN values from data."""
        data = [100, 200, 300, np.nan, 400, 500, 600, 700, 800, 900]
        dist = Distributions(data=data)
        results = dist.fit_all()
        assert len(results) == 4

    def test_fit_all_distribution_instances(self, time_series2: list):
        """Test that returned distribution instances can be used for further analysis."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        for info in results.values():
            d = info["distribution"]
            params = info["parameters"]
            cdf_values = d.cdf(parameters=params)
            assert isinstance(cdf_values, np.ndarray)


class TestBestFit:
    """Tests for the Distributions.best_fit method."""

    def test_best_fit_from_data(self, time_series2: list):
        """Test best_fit directly from data without calling fit_all first."""
        dist = Distributions(data=time_series2)
        best_name, best_info = dist.best_fit()
        assert best_name in Distributions.available_distributions
        assert "distribution" in best_info
        assert "parameters" in best_info
        assert "ks" in best_info
        assert "chisquare" in best_info

    def test_best_fit_ks(self, time_series2: list):
        """Test best_fit selects the distribution with the highest KS p-value."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        best_name, best_info = dist.best_fit(
            fit_results=results, criterion="ks"
        )
        assert best_name in results
        best_pvalue = best_info["ks"][1]
        for info in results.values():
            assert info["ks"][1] <= best_pvalue

    def test_best_fit_chisquare(self, time_series2: list):
        """Test best_fit selects by Chi-square p-value when specified."""
        dist = Distributions(data=time_series2)
        best_name, best_info = dist.best_fit(criterion="chisquare")
        assert best_name in Distributions.available_distributions

    def test_best_fit_default_criterion(self, time_series2: list):
        """Test that default criterion is 'ks'."""
        dist = Distributions(data=time_series2)
        default_name, _ = dist.best_fit()
        ks_name, _ = dist.best_fit(criterion="ks")
        assert default_name == ks_name

    def test_best_fit_with_precomputed_results(self, time_series2: list):
        """Test best_fit reuses pre-computed fit_all results."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        best_name, best_info = dist.best_fit(fit_results=results)
        assert best_name in results
        assert best_info is results[best_name]

    def test_best_fit_selected_distributions(self, time_series2: list):
        """Test best_fit with a subset of distributions."""
        dist = Distributions(data=time_series2)
        best_name, _ = dist.best_fit(distributions=["Gumbel", "GEV"])
        assert best_name in ("Gumbel", "GEV")

    def test_best_fit_invalid_criterion(self, time_series2: list):
        """Test best_fit raises ValueError for invalid criterion."""
        dist = Distributions(data=time_series2)
        with pytest.raises(ValueError, match="criterion must be"):
            dist.best_fit(criterion="invalid")
