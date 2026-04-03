"""Tests for Lmoments.pearson_3 method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def pearson_3_lmoments():
    """Valid L-moments for testing Pearson Type III distribution."""
    return [10.0, 2.0, 0.2, 0.0, 0.0]


@pytest.fixture(scope="module")
def pearson_3_parameters():
    """Expected parameters for Pearson Type III distribution."""
    return [10.0, 3.70994578417498, 1.2099737178678576]


class TestPearson3Method:
    """Test pearson_3 method."""

    def test_pearson_3_valid_inputs(self, pearson_3_lmoments, pearson_3_parameters):
        """Test pearson_3 method with valid inputs."""
        result = Lmoments.pearson_3(pearson_3_lmoments)
        np.testing.assert_almost_equal(result, pearson_3_parameters, decimal=4)

    def test_pearson_3_invalid_inputs(self):
        """Test pearson_3 method with invalid inputs returns [0, 0, 0]."""
        result = Lmoments.pearson_3([1.0, -0.5, 0.1])
        assert result == [0, 0, 0]

        result = Lmoments.pearson_3([1.0, 0.5, 1.0])
        assert result == [0, 0, 0]

    def test_pearson_3_small_third_moment(self):
        """Test pearson_3 with small third moment gives skew=0."""
        result = Lmoments.pearson_3([10.0, 2.0, 0.0000001])
        assert result[2] == 0
