"""Tests for Lmoments.wakeby method."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def wakeby_lmoments():
    """Valid L-moments for testing Wakeby distribution."""
    return [10.0, 2.0, 0.1, 0.05, 0.02]


@pytest.fixture(scope="module")
def wakeby_parameters():
    """Expected parameters for Wakeby distribution."""
    return [
        4.51860465116279,
        4.00999858552907,
        3.296933739370589,
        6.793895411225928,
        -0.49376393504801414,
    ]


class TestWakebyMethod:
    """Test wakeby method."""

    def test_wakeby_valid_inputs(self, wakeby_lmoments, wakeby_parameters):
        """Test wakeby method with valid inputs."""
        result = Lmoments.wakeby(wakeby_lmoments)
        np.testing.assert_almost_equal(result, wakeby_parameters, decimal=4)

    def test_wakeby_invalid_inputs(self):
        """Test wakeby method with invalid inputs returns None."""
        result = Lmoments.wakeby([1.0, -0.5, 0.1, 0.05, 0.02])
        assert result is None

        result = Lmoments.wakeby([1.0, 0.5, 1.0, 0.05, 0.02])
        assert result is None

        result = Lmoments.wakeby([1.0, 0.5, 0.1, 1.0, 0.02])
        assert result is None

        result = Lmoments.wakeby([1.0, 0.5, 0.1, 0.05, 1.0])
        assert result is None
