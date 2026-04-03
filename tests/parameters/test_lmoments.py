"""Tests for Lmoments class core functionality."""

import numpy as np
import pytest

from statista.parameters import Lmoments


class TestLmomentsInitialization:
    """Test Lmoments class initialization and calculate method."""

    def test_initialization(self, sample_data):
        """Test initialization of Lmoments class with valid data."""
        lmom = Lmoments(sample_data)
        assert lmom.data == sample_data

    def test_calculate_default(self, sample_data, expected_lmoments):
        """Test calculate method with default nmom=5."""
        lmom = Lmoments(sample_data)
        result = lmom.calculate()
        np.testing.assert_almost_equal(result, expected_lmoments)

    def test_calculate_custom_nmom(self, sample_data):
        """Test calculate method with custom nmom value."""
        lmom = Lmoments(sample_data)
        result = lmom.calculate(nmom=3)
        assert len(result) == 3

    def test_calculate_empty_data(self):
        """Test calculate method with empty data raises ValueError."""
        lmom = Lmoments([])
        with pytest.raises(ValueError):
            lmom.calculate()

    def test_calculate_single_value(self):
        """Test calculate method with a single value."""
        lmom = Lmoments([5.0])
        with pytest.raises(ValueError):
            lmom.calculate(nmom=2)

        result = lmom.calculate(nmom=1)
        assert result == [5.0]


class TestLmomentsComb:
    """Test _comb static method."""

    def test_comb_valid_inputs(self):
        """Test _comb method with valid inputs."""
        assert Lmoments._comb(5, 2) == 10
        assert Lmoments._comb(10, 3) == 120
        assert Lmoments._comb(7, 0) == 1
        assert Lmoments._comb(6, 6) == 1

    def test_comb_invalid_inputs(self):
        """Test _comb method with invalid inputs returns 0."""
        assert Lmoments._comb(3, 5) == 0
        assert Lmoments._comb(-1, 2) == 0
        assert Lmoments._comb(5, -1) == 0


class TestLmomentsHelperMethods:
    """Test private helper methods _samlmularge and _samlmusmall."""

    def test_samlmularge_valid_inputs(self, sample_data, expected_lmoments):
        """Test _samlmularge method with valid inputs."""
        lmom = Lmoments(sample_data)
        result = lmom._samlmularge(nmom=5)
        np.testing.assert_almost_equal(result, expected_lmoments)

    def test_samlmusmall_valid_inputs(self, sample_data, expected_lmoments):
        """Test _samlmusmall method with valid inputs."""
        lmom = Lmoments(sample_data)
        result = lmom._samlmusmall(nmom=5)
        np.testing.assert_almost_equal(result, expected_lmoments)

    def test_samlmularge_error_cases(self):
        """Test _samlmularge method with invalid inputs."""
        lmom = Lmoments([])
        with pytest.raises(ValueError):
            lmom._samlmularge()

        lmom = Lmoments([1.0, 2.0])
        with pytest.raises(ValueError):
            lmom._samlmularge(nmom=3)

        lmom = Lmoments([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            lmom._samlmularge(nmom=0)

    def test_samlmusmall_error_cases(self):
        """Test _samlmusmall method with invalid inputs."""
        lmom = Lmoments([])
        with pytest.raises(ValueError):
            lmom._samlmusmall()

        lmom = Lmoments([1.0, 2.0])
        with pytest.raises(ValueError):
            lmom._samlmusmall(nmom=3)

        lmom = Lmoments([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            lmom._samlmusmall(nmom=0)
