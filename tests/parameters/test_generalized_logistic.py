"""Tests for Lmoments.generalized_logistic method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def generalized_logistic_lmoments():
    """Valid L-moments for testing Generalized Logistic distribution."""
    return [10.0, 2.0, -0.1, 0.0, 0.0]


@pytest.fixture(scope="module")
def generalized_logistic_parameters():
    """Expected parameters for Generalized Logistic distribution."""
    return [10.327367138330683, 1.967263286166932, 0.1]


class TestGeneralizedLogisticMethod:
    """Test generalized_logistic method."""

    def test_generalized_logistic_valid_inputs(
        self,
        generalized_logistic_lmoments,
        generalized_logistic_parameters,
    ):
        """Test generalized_logistic method with valid inputs."""
        result = Lmoments.generalized_logistic(generalized_logistic_lmoments)
        np.testing.assert_almost_equal(
            result, generalized_logistic_parameters, decimal=4
        )

    def test_generalized_logistic_invalid_inputs(self):
        """Test generalized_logistic method with invalid inputs returns None."""
        result = Lmoments.generalized_logistic([1.0, -0.5, 0.1])
        assert result is None

        result = Lmoments.generalized_logistic([1.0, 0.5, 1.0])
        assert result is None

        result = Lmoments.generalized_logistic([1.0, 0.5, -1.0])
        assert result is None

    def test_generalized_logistic_small_third_moment(self):
        """Test generalized_logistic with small third moment gives shape=0."""
        result = Lmoments.generalized_logistic([10.0, 2.0, 0.0000001])
        assert result[2] == 0
