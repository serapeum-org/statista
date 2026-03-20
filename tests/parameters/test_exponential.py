"""Tests for Lmoments.exponential method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def exponential_lmoments():
    """Valid L-moments for testing Exponential distribution."""
    return [10.0, 5.0, 0.33, 0.0, 0.0]


@pytest.fixture(scope="module")
def exponential_parameters():
    """Expected parameters for Exponential distribution."""
    return [0.0, 10.0]


class TestExponentialMethod:
    """Test exponential method."""

    def test_exponential_valid_inputs(
        self, exponential_lmoments, exponential_parameters
    ):
        """Test exponential method with valid inputs."""
        result = Lmoments.exponential(exponential_lmoments)
        np.testing.assert_almost_equal(
            result, exponential_parameters, decimal=5
        )

    def test_exponential_invalid_inputs(self):
        """Test exponential method with invalid inputs returns None."""
        result = Lmoments.exponential([1.0, -0.5])
        assert result is None
