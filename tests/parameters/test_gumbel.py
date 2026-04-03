"""Tests for Lmoments.gumbel method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def gumbel_lmoments():
    """Valid L-moments for testing Gumbel distribution."""
    return [10.0, 2.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def gumbel_parameters():
    """Expected parameters for Gumbel distribution."""
    return [8.334507645446266, 2.8853900817779268]


class TestGumbelMethod:
    """Test gumbel method."""

    def test_gumbel_valid_inputs(self, gumbel_lmoments, gumbel_parameters):
        """Test gumbel method with valid inputs."""
        result = Lmoments.gumbel(gumbel_lmoments)
        np.testing.assert_almost_equal(result, gumbel_parameters, decimal=5)

    def test_gumbel_invalid_inputs(self):
        """Test gumbel method with invalid inputs raises ValueError."""
        with pytest.raises(ValueError):
            Lmoments.gumbel([1.0, -0.5])
