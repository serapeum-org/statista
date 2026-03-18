"""Test distributions module."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from statista.distributions import (
    Distributions,
    GEV,
    Gumbel,
    Normal,
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


class TestDistributionsInit:
    """Tests for Distributions.__init__."""

    def test_create_instance(self, time_series1: list):
        """Test single-distribution mode with data delegates correctly."""
        dist = Distributions("Gumbel", data=time_series1)
        assert isinstance(dist.data, np.ndarray), "data should be numpy array"
        assert isinstance(dist.data_sorted, np.ndarray), "data_sorted should be numpy array"

    def test_create_with_parameters_only(self):
        """Test single-distribution mode with parameters only (no data).

        Test scenario:
            Wrapping a distribution with only parameters should store them
            and allow access through delegation.
        """
        params = {"loc": 500, "scale": 200}
        dist = Distributions("Normal", parameters=params)
        assert dist.parameters == params, f"Expected {params}, got {dist.parameters}"
        assert dist.distribution is not None, "distribution instance should be set"

    def test_create_with_data_and_parameters(self, time_series2: list):
        """Test single-distribution mode with both data and parameters.

        Test scenario:
            Providing both data and parameters should pass them through to
            the underlying distribution.
        """
        params = {"loc": 500, "scale": 200}
        dist = Distributions("Normal", data=time_series2, parameters=params)
        assert dist.parameters == params, "Parameters should match provided values"
        assert len(dist.data) == len(time_series2), "Data length should match"

    def test_create_with_list_data(self):
        """Test that list data is accepted and stored as numpy array.

        Test scenario:
            A plain Python list should be converted to ndarray on _data.
        """
        data = [100, 200, 300, 400, 500]
        dist = Distributions(data=data)
        assert isinstance(dist._data, np.ndarray), f"Expected ndarray, got {type(dist._data)}"
        assert len(dist._data) == 5, f"Expected length 5, got {len(dist._data)}"

    def test_create_with_numpy_data(self):
        """Test that numpy array data is accepted and stored.

        Test scenario:
            A numpy array should be stored directly as _data.
        """
        data = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        dist = Distributions(data=data)
        assert isinstance(dist._data, np.ndarray), f"Expected ndarray, got {type(dist._data)}"
        np.testing.assert_array_equal(dist._data, data)

    def test_create_without_distribution(self, time_series2: list):
        """Test multi-distribution mode stores data and sets distribution to None."""
        dist = Distributions(data=time_series2)
        assert dist._data is not None, "_data should be set"
        assert dist.distribution is None, "distribution should be None in multi mode"

    def test_create_without_distribution_or_data(self):
        """Test that providing neither distribution nor data raises ValueError."""
        with pytest.raises(
            ValueError, match="Either distribution or data must be provided"
        ):
            Distributions()

    def test_create_with_only_parameters_no_distribution(self):
        """Test that parameters alone without distribution name raises ValueError.

        Test scenario:
            Parameters without a distribution name have no meaning since we
            don't know which distribution to create.
        """
        with pytest.raises(
            ValueError, match="Either distribution or data must be provided"
        ):
            Distributions(parameters={"loc": 0, "scale": 1})

    def test_invalid_distribution_name(self):
        """Test that an unsupported distribution name raises ValueError."""
        with pytest.raises(ValueError, match="InvalidDist not supported"):
            Distributions("InvalidDist", data=[1, 2, 3, 4, 5])

    @pytest.mark.parametrize("name", ["GEV", "Gumbel", "Exponential", "Normal"])
    def test_all_distribution_names_accepted(self, name: str, time_series2: list):
        """Test that all registered distribution names are accepted.

        Args:
            name: Distribution name from available_distributions.

        Test scenario:
            Each name in the registry should create an instance without error.
        """
        dist = Distributions(name, data=time_series2)
        assert dist.distribution is not None, f"{name} should create a distribution instance"

    def test_data_stored_in_single_mode(self, time_series2: list):
        """Test that _data is stored even in single-distribution mode.

        Test scenario:
            When both distribution and data are provided, _data should
            still be populated for use by fit_all/best_fit.
        """
        dist = Distributions("Gumbel", data=time_series2)
        assert dist._data is not None, "_data should be set in single mode"
        assert len(dist._data) == len(time_series2), "Data length should match"


class TestDistributionsGetattr:
    """Tests for Distributions.__getattr__ delegation."""

    def test_delegates_fit_model(self, time_series2: list):
        """Test that fit_model is delegated to the underlying distribution.

        Test scenario:
            Calling fit_model on Distributions should delegate to the
            wrapped Gumbel and return parameters.
        """
        dist = Distributions("Gumbel", data=time_series2)
        param = dist.fit_model(method="lmoments", test=False)
        assert isinstance(param, dict), f"Expected dict, got {type(param)}"
        assert "loc" in param, "Parameters should contain 'loc'"
        assert "scale" in param, "Parameters should contain 'scale'"

    def test_delegates_parameters_property(self):
        """Test that the parameters property is accessible through delegation."""
        params = {"loc": 500, "scale": 200}
        dist = Distributions("Normal", parameters=params)
        assert dist.parameters == params, f"Expected {params}, got {dist.parameters}"

    def test_delegates_data_property(self, time_series2: list):
        """Test that the data property is accessible through delegation."""
        dist = Distributions("Gumbel", data=time_series2)
        assert isinstance(dist.data, np.ndarray), "data property should return ndarray"

    def test_raises_for_nonexistent_attribute(self):
        """Test that accessing a nonexistent attribute raises AttributeError."""
        dist = Distributions("Gumbel", data=[1, 2, 3, 4, 5])
        with pytest.raises(
            AttributeError,
            match="'Distributions' object has no attribute 'invalid_method'",
        ):
            dist.invalid_method()

    def test_raises_when_no_distribution_set(self, time_series2: list):
        """Test that delegation raises when distribution is None (multi mode)."""
        dist = Distributions(data=time_series2)
        with pytest.raises(
            AttributeError,
            match="'Distributions' object has no attribute 'fit_model'",
        ):
            dist.fit_model()

    def test_raises_for_nonexistent_attr_multi_mode(self, time_series2: list):
        """Test AttributeError message in multi-distribution mode.

        Test scenario:
            Even for attributes that don't exist on any distribution, the
            error message should reference the Distributions class.
        """
        dist = Distributions(data=time_series2)
        with pytest.raises(
            AttributeError,
            match="'Distributions' object has no attribute 'totally_fake'",
        ):
            dist.totally_fake


class TestFitAll:
    """Tests for the Distributions.fit_all method."""

    def test_fit_all_default(self, time_series2: list):
        """Test fit_all with default parameters fits all distributions."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        assert set(results.keys()) == {"GEV", "Gumbel", "Exponential", "Normal"}, (
            f"Expected all 4 distributions, got {set(results.keys())}"
        )
        for name, info in results.items():
            assert "distribution" in info, f"{name}: missing 'distribution' key"
            assert "parameters" in info, f"{name}: missing 'parameters' key"
            assert "ks" in info, f"{name}: missing 'ks' key"
            assert "chisquare" in info, f"{name}: missing 'chisquare' key"
            assert isinstance(info["parameters"], dict), f"{name}: parameters should be dict"
            assert "loc" in info["parameters"], f"{name}: parameters should contain 'loc'"
            assert "scale" in info["parameters"], f"{name}: parameters should contain 'scale'"
            assert len(info["ks"]) == 2, f"{name}: KS result should be 2-tuple"
            assert len(info["chisquare"]) == 2, f"{name}: chisquare result should be 2-tuple"

    def test_fit_all_selected_distributions(self, time_series2: list):
        """Test fit_all with a subset of distributions."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all(distributions=["Gumbel", "GEV"])
        assert set(results.keys()) == {"Gumbel", "GEV"}, (
            f"Expected only Gumbel and GEV, got {set(results.keys())}"
        )

    def test_fit_all_single_distribution(self, time_series2: list):
        """Test fit_all with a single distribution name.

        Test scenario:
            Providing a list with one distribution should return a dict
            with exactly one entry.
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all(distributions=["Normal"])
        assert list(results.keys()) == ["Normal"], (
            f"Expected ['Normal'], got {list(results.keys())}"
        )

    def test_fit_all_mle_method(self, time_series2: list):
        """Test fit_all with MLE method."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all(method="mle")
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        for info in results.values():
            assert isinstance(info["parameters"], dict)

    def test_fit_all_invalid_distribution(self, time_series2: list):
        """Test fit_all raises ValueError for invalid distribution name."""
        dist = Distributions(data=time_series2)
        with pytest.raises(ValueError, match="InvalidDist not supported"):
            dist.fit_all(distributions=["InvalidDist"])

    def test_fit_all_invalid_method(self, time_series2: list):
        """Test fit_all raises ValueError for invalid fitting method.

        Test scenario:
            An unsupported method name should be rejected at the facade
            level with a clear error message listing valid options.
        """
        dist = Distributions(data=time_series2)
        with pytest.raises(ValueError, match="method must be one of"):
            dist.fit_all(method="invalid")

    def test_fit_all_empty_distributions_list(self, time_series2: list):
        """Test fit_all raises ValueError for empty distributions list.

        Test scenario:
            An empty list means no distributions to fit, which should
            fail early rather than return an empty dict.
        """
        dist = Distributions(data=time_series2)
        with pytest.raises(ValueError, match="distributions list must not be empty"):
            dist.fit_all(distributions=[])

    def test_fit_all_handles_nan(self):
        """Test fit_all removes NaN values before fitting."""
        data = [100, 200, 300, np.nan, 400, 500, 600, 700, 800, 900]
        dist = Distributions(data=data)
        results = dist.fit_all()
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

    def test_fit_all_distribution_instances_usable(self, time_series2: list):
        """Test that returned distribution instances can compute CDF values."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        for name, info in results.items():
            d = info["distribution"]
            params = info["parameters"]
            cdf_values = d.cdf(parameters=params)
            assert isinstance(cdf_values, np.ndarray), (
                f"{name}: CDF should return ndarray"
            )

    def test_fit_all_ks_pvalues_in_valid_range(self, time_series2: list):
        """Test that KS p-values are between 0 and 1.

        Test scenario:
            Goodness-of-fit p-values must always be in [0, 1].
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        for name, info in results.items():
            ks_stat, ks_pval = info["ks"]
            assert 0 <= ks_stat, f"{name}: KS statistic should be >= 0, got {ks_stat}"
            assert 0 <= ks_pval <= 1, f"{name}: KS p-value should be in [0,1], got {ks_pval}"

    def test_fit_all_chisquare_pvalues_in_valid_range(self, time_series2: list):
        """Test that Chi-square p-values are between 0 and 1.

        Test scenario:
            Goodness-of-fit p-values must always be in [0, 1].
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        for name, info in results.items():
            chi_stat, chi_pval = info["chisquare"]
            assert 0 <= chi_stat, f"{name}: chi-square statistic should be >= 0, got {chi_stat}"
            assert 0 <= chi_pval <= 1, (
                f"{name}: chi-square p-value should be in [0,1], got {chi_pval}"
            )

    def test_fit_all_returns_independent_instances(self, time_series2: list):
        """Test that each returned distribution is an independent instance.

        Test scenario:
            The distribution instances in the result dict should not be
            the same object — fitting one should not affect another.
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        instances = [info["distribution"] for info in results.values()]
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                assert instances[i] is not instances[j], (
                    "Distribution instances should be independent objects"
                )

    def test_fit_all_called_twice_gives_consistent_results(self, time_series2: list):
        """Test that calling fit_all twice produces the same parameters.

        Test scenario:
            The method should be deterministic — same data, same method,
            same results.
        """
        dist = Distributions(data=time_series2)
        results1 = dist.fit_all(distributions=["Gumbel"])
        results2 = dist.fit_all(distributions=["Gumbel"])
        params1 = results1["Gumbel"]["parameters"]
        params2 = results2["Gumbel"]["parameters"]
        assert params1 == params2, (
            f"Parameters should be identical across calls: {params1} != {params2}"
        )


class TestBestFit:
    """Tests for the Distributions.best_fit method."""

    def test_best_fit_from_data(self, time_series2: list):
        """Test best_fit directly from data without calling fit_all first."""
        dist = Distributions(data=time_series2)
        best_name, best_info = dist.best_fit()
        assert best_name in Distributions.available_distributions, (
            f"best_name '{best_name}' not in available distributions"
        )
        assert "distribution" in best_info, "Result should contain 'distribution'"
        assert "parameters" in best_info, "Result should contain 'parameters'"
        assert "ks" in best_info, "Result should contain 'ks'"
        assert "chisquare" in best_info, "Result should contain 'chisquare'"

    def test_best_fit_ks_selects_highest_pvalue(self, time_series2: list):
        """Test best_fit with KS criterion selects the highest KS p-value.

        Test scenario:
            The selected distribution's KS p-value should be >= all others.
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        best_name, best_info = dist.best_fit(
            fit_results=results, criterion="ks"
        )
        assert best_name in results, f"'{best_name}' not in results"
        best_pvalue = best_info["ks"][1]
        for name, info in results.items():
            assert info["ks"][1] <= best_pvalue, (
                f"{name} has higher KS p-value ({info['ks'][1]}) "
                f"than selected '{best_name}' ({best_pvalue})"
            )

    def test_best_fit_chisquare_selects_highest_pvalue(self, time_series2: list):
        """Test best_fit with chisquare criterion selects highest Chi-square p-value.

        Test scenario:
            The selected distribution's chisquare p-value should be >= all others.
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        best_name, best_info = dist.best_fit(
            fit_results=results, criterion="chisquare"
        )
        best_pvalue = best_info["chisquare"][1]
        for name, info in results.items():
            assert info["chisquare"][1] <= best_pvalue, (
                f"{name} has higher chisquare p-value ({info['chisquare'][1]}) "
                f"than selected '{best_name}' ({best_pvalue})"
            )

    def test_best_fit_default_criterion_is_ks(self, time_series2: list):
        """Test that default criterion is 'ks'."""
        dist = Distributions(data=time_series2)
        default_name, _ = dist.best_fit()
        ks_name, _ = dist.best_fit(criterion="ks")
        assert default_name == ks_name, (
            f"Default ({default_name}) should equal KS ({ks_name})"
        )

    def test_best_fit_with_precomputed_results(self, time_series2: list):
        """Test best_fit reuses pre-computed fit_all results without refitting."""
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        best_name, best_info = dist.best_fit(fit_results=results)
        assert best_name in results, f"'{best_name}' not in results"
        assert best_info is results[best_name], (
            "best_info should be the exact same object from results, not a copy"
        )

    def test_best_fit_selected_distributions(self, time_series2: list):
        """Test best_fit with a subset of distributions."""
        dist = Distributions(data=time_series2)
        best_name, _ = dist.best_fit(distributions=["Gumbel", "GEV"])
        assert best_name in ("Gumbel", "GEV"), (
            f"Expected Gumbel or GEV, got '{best_name}'"
        )

    def test_best_fit_single_distribution(self, time_series2: list):
        """Test best_fit with only one distribution returns that distribution.

        Test scenario:
            When only one distribution is available, it must be selected.
        """
        dist = Distributions(data=time_series2)
        best_name, best_info = dist.best_fit(distributions=["Normal"])
        assert best_name == "Normal", f"Expected 'Normal', got '{best_name}'"
        assert "loc" in best_info["parameters"], "Normal should have 'loc' parameter"

    def test_best_fit_invalid_criterion(self, time_series2: list):
        """Test best_fit raises ValueError for invalid criterion."""
        dist = Distributions(data=time_series2)
        with pytest.raises(ValueError, match="criterion must be"):
            dist.best_fit(criterion="invalid")

    def test_best_fit_fit_results_ignores_method_and_distributions(
        self, time_series2: list
    ):
        """Test that method and distributions params are ignored when fit_results is provided.

        Test scenario:
            Passing fit_results with only Gumbel, but distributions=["Normal"]
            should still return Gumbel since fit_results takes precedence.
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all(distributions=["Gumbel"])
        best_name, _ = dist.best_fit(
            distributions=["Normal"],
            method="mle",
            fit_results=results,
        )
        assert best_name == "Gumbel", (
            f"fit_results should take precedence, expected 'Gumbel', got '{best_name}'"
        )

    def test_best_fit_single_entry_fit_results(self, time_series2: list):
        """Test best_fit with a single-entry fit_results dict.

        Test scenario:
            A pre-computed result with exactly one distribution should return it.
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all(distributions=["Exponential"])
        best_name, best_info = dist.best_fit(fit_results=results)
        assert best_name == "Exponential", (
            f"Expected 'Exponential', got '{best_name}'"
        )
        assert best_info is results["Exponential"], "Should return the same object"


class TestDistributionsIntegration:
    """Integration tests for the full Distributions workflow."""

    def test_single_mode_fit_then_multi_mode(self, time_series2: list):
        """Test that a single-mode instance can also use fit_all/best_fit.

        Test scenario:
            An instance created with a distribution name still has _data,
            so fit_all/best_fit should work on it.
        """
        dist = Distributions("Gumbel", data=time_series2)
        params = dist.fit_model(method="lmoments", test=False)
        assert "loc" in params, "Single-mode fit_model should return params"

        results = dist.fit_all(distributions=["Gumbel", "Normal"])
        assert set(results.keys()) == {"Gumbel", "Normal"}, (
            "fit_all should work on a single-mode instance"
        )

    def test_fit_all_then_best_fit_pipeline(self, time_series2: list):
        """Test the full pipeline: create → fit_all → best_fit → use distribution.

        Test scenario:
            The complete workflow should produce a usable distribution
            instance that can compute CDF and inverse CDF.
        """
        dist = Distributions(data=time_series2)
        results = dist.fit_all()
        best_name, best_info = dist.best_fit(fit_results=results)

        fitted_dist = best_info["distribution"]
        params = best_info["parameters"]
        cdf_values = fitted_dist.cdf(parameters=params)
        assert len(cdf_values) > 0, "CDF should return values"

        prob = np.array([0.5, 0.9, 0.99])
        quantiles = fitted_dist.inverse_cdf(prob, params)
        assert len(quantiles) == 3, f"Expected 3 quantiles, got {len(quantiles)}"
        assert quantiles[0] < quantiles[1] < quantiles[2], (
            "Quantiles should increase with probability"
        )

    def test_available_distributions_registry(self):
        """Test that available_distributions contains exactly the expected entries.

        Test scenario:
            The registry should map 4 names to their concrete classes.
        """
        expected = {"GEV", "Gumbel", "Exponential", "Normal"}
        actual = set(Distributions.available_distributions.keys())
        assert actual == expected, f"Expected {expected}, got {actual}"

        assert Distributions.available_distributions["GEV"] is GEV
        assert Distributions.available_distributions["Gumbel"] is Gumbel
        assert Distributions.available_distributions["Normal"] is Normal
