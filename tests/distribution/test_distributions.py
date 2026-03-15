"""Test distributions module."""

import matplotlib

matplotlib.use("Agg")
from typing import Dict, List

import numpy as np
import pytest

from statista.distributions import (
    GEV,
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
        dist_estimation_parameters: List[str],
    ):
        dist = Distributions("Gumbel", data=time_series2)
        for i in range(len(dist_estimation_parameters)):
            param = dist.fit_model(method=dist_estimation_parameters[i], test=False)
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert dist.parameters.get("loc") is not None
            assert dist.parameters.get("scale") is not None
