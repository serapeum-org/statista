"""Tests for Lmoments.generalized_normal method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def generalized_normal_lmoments():
    """Valid L-moments for testing Generalized Normal distribution."""
    return [10.0, 2.0, 0.1, 0.0, 0.0]


@pytest.fixture(scope="module")
def generalized_normal_parameters():
    """Expected parameters for Generalized Normal distribution."""
    return [9.638928100246755, 3.4832722896983213, -0.2051440978274827]


class TestGeneralizedNormalMethod:
    """Test generalized_normal method."""

    def test_generalized_normal_valid_inputs(
        self,
        generalized_normal_lmoments,
        generalized_normal_parameters,
    ):
        """Test generalized_normal method with valid inputs."""
        result = Lmoments.generalized_normal(generalized_normal_lmoments)
        np.testing.assert_almost_equal(
            result, generalized_normal_parameters, decimal=4
        )

    def test_generalized_normal_invalid_inputs(self):
        """Test generalized_normal method with invalid inputs returns None."""
        result = Lmoments.generalized_normal([1.0, -0.5, 0.1])
        assert result is None

        result = Lmoments.generalized_normal([1.0, 0.5, 1.0])
        assert result is None

        result = Lmoments.generalized_normal([1.0, 0.5, -1.0])
        assert result is None

    def test_generalized_normal_large_third_moment(self):
        """Test generalized_normal with large third moment returns [0, -1, 0]."""
        result = Lmoments.generalized_normal([10.0, 2.0, 0.95])
        assert result == [0, -1, 0]
