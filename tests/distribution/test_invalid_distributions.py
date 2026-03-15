import pytest

from statista.distributions import Distributions


class TestDistributionsClassInvalid:
    """Tests for the Distributions class."""

    def test_invalid_distribution(self):
        """Test that an error is raised when an invalid distribution is provided."""
        with pytest.raises(ValueError, match="InvalidDist not supported"):
            Distributions("InvalidDist", data=[1, 2, 3, 4, 5])

    def test_invalid_attribute(self):
        """Test that an error is raised when accessing a non-existent attribute."""
        dist = Distributions("Gumbel", data=[1, 2, 3, 4, 5])
        with pytest.raises(
            AttributeError,
            match="'Distributions' object has no attribute 'invalid_method'",
        ):
            dist.invalid_method()



