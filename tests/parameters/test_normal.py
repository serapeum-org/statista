"""Tests for Lmoments.normal method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def normal_lmoments():
    """Valid L-moments for testing Normal distribution."""
    return [10.0, 2.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def normal_parameters():
    """Expected parameters for Normal distribution."""
    return [10.0, 3.5449077018110318]


class TestNormalMethod:
    """Test normal method."""

    def test_normal_valid_inputs(self, normal_lmoments, normal_parameters):
        """Test normal method with valid inputs."""
        result = Lmoments.normal(normal_lmoments)
        np.testing.assert_almost_equal(
            result, normal_parameters, decimal=5
        )

    def test_normal_invalid_inputs(self):
        """Test normal method with invalid inputs returns None."""
        result = Lmoments.normal([1.0, -0.5])
        assert result is None
