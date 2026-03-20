"""Tests for Lmoments.gamma method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def gamma_lmoments():
    """Valid L-moments for testing Gamma distribution."""
    return [10.0, 3.0, 0.2, 0.0, 0.0]


@pytest.fixture(scope="module")
def gamma_parameters():
    """Expected parameters for Gamma distribution."""
    return [3.278019029280183, 3.0506229252109893]


class TestGammaMethod:
    """Test gamma method."""

    def test_gamma_valid_inputs(self, gamma_lmoments, gamma_parameters):
        """Test gamma method with valid inputs."""
        result = Lmoments.gamma(gamma_lmoments)
        np.testing.assert_almost_equal(result, gamma_parameters, decimal=5)

    def test_gamma_invalid_inputs(self):
        """Test gamma method with invalid inputs returns None."""
        result = Lmoments.gamma([1.0, -0.5])
        assert result is None

        result = Lmoments.gamma([1.0, 1.0])
        assert result is None
