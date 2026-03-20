"""Tests for Lmoments.gev method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def gev_lmoments():
    """Valid L-moments for testing GEV distribution."""
    return [10.0, 2.0, 0.1, 0.05, 0.02]


@pytest.fixture(scope="module")
def gev_parameters():
    """Expected parameters for GEV distribution."""
    return [0.11189502871959642, 8.490058310239982, 3.1676863588272224]


class TestGEVMethod:
    """Test gev method."""

    def test_gev_valid_inputs(self, gev_lmoments, gev_parameters):
        """Test gev method with valid inputs."""
        result = Lmoments.gev(gev_lmoments)
        np.testing.assert_almost_equal(result, gev_parameters, decimal=5)

    def test_gev_invalid_inputs(self):
        """Test gev method with invalid inputs raises ValueError."""
        with pytest.raises(ValueError):
            Lmoments.gev([1.0, -0.5, 0.1])

        with pytest.raises(ValueError):
            Lmoments.gev([1.0, 0.5, 1.0])

        with pytest.raises(ValueError):
            Lmoments.gev([1.0, 0.5, -1.0])
