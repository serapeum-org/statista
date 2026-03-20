"""Tests for Lmoments.generalized_pareto method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def generalized_pareto_lmoments():
    """Valid L-moments for testing Generalized Pareto distribution."""
    return [10.0, 2.0, 0.1, 0.0, 0.0]


@pytest.fixture(scope="module")
def generalized_pareto_parameters():
    """Expected parameters for Generalized Pareto distribution."""
    return [4.7272727272727275, 8.628099173553718, 0.6363636363636362]


class TestGeneralizedParetoMethod:
    """Test generalized_pareto method."""

    def test_generalized_pareto_valid_inputs(
        self,
        generalized_pareto_lmoments,
        generalized_pareto_parameters,
    ):
        """Test generalized_pareto method with valid inputs."""
        result = Lmoments.generalized_pareto(generalized_pareto_lmoments)
        np.testing.assert_almost_equal(
            result, generalized_pareto_parameters, decimal=4
        )

    def test_generalized_pareto_invalid_inputs(self):
        """Test generalized_pareto method with invalid inputs returns None."""
        result = Lmoments.generalized_pareto([1.0, -0.5, 0.1])
        assert result is None

        result = Lmoments.generalized_pareto([1.0, 0.5, 1.0])
        assert result is None

        result = Lmoments.generalized_pareto([1.0, 0.5, -1.0])
        assert result is None
