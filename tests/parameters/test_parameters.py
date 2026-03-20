"""Test parameters module."""

import numpy as np
import pytest

from statista.parameters import Lmoments


@pytest.fixture(scope="module")
def sample_data():
    """Sample data for testing Lmoments calculation."""
    return [1.2, 3.4, 5.6, 7.8, 9.0, 2.3, 4.5, 6.7, 8.9]


@pytest.fixture(scope="module")
def expected_lmoments():
    """Expected L-moments for sample_data."""
    return [
        5.488888888888889,
        1.722222222222222,
        -0.06451612903225806,
        -0.0645161290322581,
        -0.0645161290322581,
    ]


@pytest.fixture(scope="module")
def invalid_lmoments():
    """Invalid L-moments for testing error cases."""
    return [1.0, -0.5, 1.2, 0.8, 0.6]  # Second L-moment is negative


@pytest.fixture(scope="module")
def valid_lmoments():
    """Valid L-moments for testing distribution methods."""
    return [10.0, 2.0, 0.1, 0.05, 0.02]


@pytest.fixture(scope="module")
def gev_lmoments():
    """Valid L-moments for testing GEV distribution."""
    return [10.0, 2.0, 0.1, 0.05, 0.02]


@pytest.fixture(scope="module")
def gev_parameters():
    """Expected parameters for GEV distribution."""
    return [0.11189502871959642, 8.490058310239982, 3.1676863588272224]


@pytest.fixture(scope="module")
def gumbel_lmoments():
    """Valid L-moments for testing Gumbel distribution."""
    return [10.0, 2.0, 0.0, 0.0, 0.0]  # Third moment is 0 for Gumbel


@pytest.fixture(scope="module")
def gumbel_parameters():
    """Expected parameters for Gumbel distribution."""
    return [8.334507645446266, 2.8853900817779268]


@pytest.fixture(scope="module")
def exponential_lmoments():
    """Valid L-moments for testing Exponential distribution."""
    return [10.0, 5.0, 0.33, 0.0, 0.0]


@pytest.fixture(scope="module")
def exponential_parameters():
    """Expected parameters for Exponential distribution."""
    return [0.0, 10.0]


@pytest.fixture(scope="module")
def gamma_lmoments():
    """Valid L-moments for testing Gamma distribution."""
    return [10.0, 3.0, 0.2, 0.0, 0.0]


@pytest.fixture(scope="module")
def gamma_parameters():
    """Expected parameters for Gamma distribution."""
    return [3.278019029280183, 3.0506229252109893]


@pytest.fixture(scope="module")
def generalized_logistic_lmoments():
    """Valid L-moments for testing Generalized Logistic distribution."""
    return [10.0, 2.0, -0.1, 0.0, 0.0]


@pytest.fixture(scope="module")
def generalized_logistic_parameters():
    """Expected parameters for Generalized Logistic distribution."""
    return [10.327367138330683, 1.967263286166932, 0.1]


@pytest.fixture(scope="module")
def generalized_normal_lmoments():
    """Valid L-moments for testing Generalized Normal distribution."""
    return [10.0, 2.0, 0.1, 0.0, 0.0]


@pytest.fixture(scope="module")
def generalized_normal_parameters():
    """Expected parameters for Generalized Normal distribution."""
    return [9.638928100246755, 3.4832722896983213, -0.2051440978274827]


@pytest.fixture(scope="module")
def generalized_pareto_lmoments():
    """Valid L-moments for testing Generalized Pareto distribution."""
    return [10.0, 2.0, 0.1, 0.0, 0.0]


@pytest.fixture(scope="module")
def generalized_pareto_parameters():
    """Expected parameters for Generalized Pareto distribution."""
    return [4.7272727272727275, 8.628099173553718, 0.6363636363636362]


@pytest.fixture(scope="module")
def normal_lmoments():
    """Valid L-moments for testing Normal distribution."""
    return [10.0, 2.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def normal_parameters():
    """Expected parameters for Normal distribution."""
    return [10.0, 3.5449077018110318]


@pytest.fixture(scope="module")
def pearson_3_lmoments():
    """Valid L-moments for testing Pearson Type III distribution."""
    return [10.0, 2.0, 0.2, 0.0, 0.0]


@pytest.fixture(scope="module")
def pearson_3_parameters():
    """Expected parameters for Pearson Type III distribution."""
    return [10.0, 3.70994578417498, 1.2099737178678576]


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


class TestLmomentsInitialization:
    """Test Lmoments class initialization and calculate method."""

    def test_initialization(self, sample_data):
        """
        Test initialization of Lmoments class with valid data.

        Inputs:
            sample_data: A list of numerical values

        Expected:
            Lmoments object is created with data attribute set to sample_data
        """
        lmom = Lmoments(sample_data)
        assert lmom.data == sample_data

    def test_calculate_default(self, sample_data, expected_lmoments):
        """
        Test calculate method with default nmom=5.

        Inputs:
            sample_data: A list of numerical values
            expected_lmoments: Expected L-moments for sample_data

        Expected:
            calculate method returns the correct L-moments
        """
        lmom = Lmoments(sample_data)
        result = lmom.calculate()
        np.testing.assert_almost_equal(result, expected_lmoments)

    def test_calculate_custom_nmom(self, sample_data):
        """
        Test calculate method with custom nmom value.

        Inputs:
            sample_data: A list of numerical values

        Expected:
            calculate method returns the correct number of L-moments
        """
        lmom = Lmoments(sample_data)
        result = lmom.calculate(nmom=3)
        assert len(result) == 3

    def test_calculate_empty_data(self):
        """
        Test calculate method with empty data.

        Expected:
            ValueError is raised
        """
        lmom = Lmoments([])
        with pytest.raises(ValueError):
            lmom.calculate()

    def test_calculate_single_value(self):
        """
        Test calculate method with a single value.

        Expected:
            ValueError is raised when nmom > 1
        """
        lmom = Lmoments([5.0])
        with pytest.raises(ValueError):
            lmom.calculate(nmom=2)

        # But it should work for nmom=1
        result = lmom.calculate(nmom=1)
        assert result == [5.0]


class TestLmomentsComb:
    """Test _comb static method."""

    def test_comb_valid_inputs(self):
        """
        Test _comb method with valid inputs.

        Expected:
            _comb returns correct combination values
        """
        assert Lmoments._comb(5, 2) == 10
        assert Lmoments._comb(10, 3) == 120
        assert Lmoments._comb(7, 0) == 1
        assert Lmoments._comb(6, 6) == 1

    def test_comb_invalid_inputs(self):
        """
        Test _comb method with invalid inputs.

        Expected:
            _comb returns 0 for invalid combinations
        """
        assert Lmoments._comb(3, 5) == 0  # k > n
        assert Lmoments._comb(-1, 2) == 0  # n < 0
        assert Lmoments._comb(5, -1) == 0  # k < 0


class TestLmomentsHelperMethods:
    """Test private helper methods _samlmularge and _samlmusmall."""

    def test_samlmularge_valid_inputs(self, sample_data, expected_lmoments):
        """
        Test _samlmularge method with valid inputs.

        Inputs:
            sample_data: A list of numerical values
            expected_lmoments: Expected L-moments for sample_data

        Expected:
            _samlmularge returns the correct L-moments
        """
        lmom = Lmoments(sample_data)
        result = lmom._samlmularge(nmom=5)
        np.testing.assert_almost_equal(result, expected_lmoments)

    def test_samlmusmall_valid_inputs(self, sample_data, expected_lmoments):
        """
        Test _samlmusmall method with valid inputs.

        Inputs:
            sample_data: A list of numerical values
            expected_lmoments: Expected L-moments for sample_data

        Expected:
            _samlmusmall returns the correct L-moments
        """
        lmom = Lmoments(sample_data)
        result = lmom._samlmusmall(nmom=5)
        np.testing.assert_almost_equal(result, expected_lmoments)

    def test_samlmularge_error_cases(self):
        """
        Test _samlmularge method with invalid inputs.

        Expected:
            ValueError is raised for invalid inputs
        """
        # Test with empty data
        lmom = Lmoments([])
        with pytest.raises(ValueError):
            lmom._samlmularge()

        # Test with insufficient data length
        lmom = Lmoments([1.0, 2.0])
        with pytest.raises(ValueError):
            lmom._samlmularge(nmom=3)

        # Test with invalid nmom
        lmom = Lmoments([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            lmom._samlmularge(nmom=0)

    def test_samlmusmall_error_cases(self):
        """
        Test _samlmusmall method with invalid inputs.

        Expected:
            ValueError is raised for invalid inputs
        """
        # Test with empty data
        lmom = Lmoments([])
        with pytest.raises(ValueError):
            lmom._samlmusmall()

        # Test with insufficient data length
        lmom = Lmoments([1.0, 2.0])
        with pytest.raises(ValueError):
            lmom._samlmusmall(nmom=3)

        # Test with invalid nmom
        lmom = Lmoments([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            lmom._samlmusmall(nmom=0)


class TestGEVMethod:
    """Test gev method."""

    def test_gev_valid_inputs(self, gev_lmoments, gev_parameters):
        """
        Test gev method with valid inputs.

        Inputs:
            gev_lmoments: Valid L-moments for GEV distribution
            gev_parameters: Expected parameters for GEV distribution

        Expected:
            gev method returns the correct parameters
        """
        result = Lmoments.gev(gev_lmoments)
        np.testing.assert_almost_equal(result, gev_parameters, decimal=5)

    def test_gev_invalid_inputs(self):
        """
        Test gev method with invalid inputs.

        Expected:
            ValueError is raised for invalid L-moments
        """
        # Test with negative second L-moment
        with pytest.raises(ValueError):
            Lmoments.gev([1.0, -0.5, 0.1])

        # Test with third L-moment >= 1
        with pytest.raises(ValueError):
            Lmoments.gev([1.0, 0.5, 1.0])

        # Test with third L-moment <= -1
        with pytest.raises(ValueError):
            Lmoments.gev([1.0, 0.5, -1.0])


class TestGumbelMethod:
    """Test gumbel method."""

    def test_gumbel_valid_inputs(self, gumbel_lmoments, gumbel_parameters):
        """
        Test gumbel method with valid inputs.

        Inputs:
            gumbel_lmoments: Valid L-moments for Gumbel distribution
            gumbel_parameters: Expected parameters for Gumbel distribution

        Expected:
            gumbel method returns the correct parameters
        """
        result = Lmoments.gumbel(gumbel_lmoments)
        np.testing.assert_almost_equal(result, gumbel_parameters, decimal=5)

    def test_gumbel_invalid_inputs(self):
        """
        Test gumbel method with invalid inputs.

        Expected:
            ValueError is raised for invalid L-moments
        """
        # Test with negative second L-moment
        with pytest.raises(ValueError):
            Lmoments.gumbel([1.0, -0.5])


class TestExponentialMethod:
    """Test exponential method."""

    def test_exponential_valid_inputs(
        self, exponential_lmoments, exponential_parameters
    ):
        """
        Test exponential method with valid inputs.

        Inputs:
            exponential_lmoments: Valid L-moments for Exponential distribution
            exponential_parameters: Expected parameters for Exponential distribution

        Expected:
            exponential method returns the correct parameters
        """
        result = Lmoments.exponential(exponential_lmoments)
        np.testing.assert_almost_equal(result, exponential_parameters, decimal=5)

    def test_exponential_invalid_inputs(self):
        """
        Test exponential method with invalid inputs.

        Expected:
            None is returned for invalid L-moments
        """
        # Test with negative second L-moment
        result = Lmoments.exponential([1.0, -0.5])
        assert result is None


class TestGammaMethod:
    """Test gamma method."""

    def test_gamma_valid_inputs(self, gamma_lmoments, gamma_parameters):
        """
        Test gamma method with valid inputs.

        Inputs:
            gamma_lmoments: Valid L-moments for Gamma distribution
            gamma_parameters: Expected parameters for Gamma distribution

        Expected:
            gamma method returns the correct parameters
        """
        result = Lmoments.gamma(gamma_lmoments)
        np.testing.assert_almost_equal(result, gamma_parameters, decimal=5)

    def test_gamma_invalid_inputs(self):
        """
        Test gamma method with invalid inputs.

        Expected:
            None is returned for invalid L-moments
        """
        # Test with negative second L-moment
        result = Lmoments.gamma([1.0, -0.5])
        assert result is None

        # Test with first moment <= second moment
        result = Lmoments.gamma([1.0, 1.0])
        assert result is None


class TestGeneralizedLogisticMethod:
    """Test generalized_logistic method."""

    def test_generalized_logistic_valid_inputs(
        self, generalized_logistic_lmoments, generalized_logistic_parameters
    ):
        """
        Test generalized_logistic method with valid inputs.

        Inputs:
            generalized_logistic_lmoments: Valid L-moments for Generalized Logistic distribution
            generalized_logistic_parameters: Expected parameters for Generalized Logistic distribution

        Expected:
            generalized_logistic method returns the correct parameters
        """
        result = Lmoments.generalized_logistic(generalized_logistic_lmoments)
        np.testing.assert_almost_equal(
            result, generalized_logistic_parameters, decimal=4
        )

    def test_generalized_logistic_invalid_inputs(self):
        """
        Test generalized_logistic method with invalid inputs.

        Expected:
            None is returned for invalid L-moments
        """
        # Test with negative second L-moment
        result = Lmoments.generalized_logistic([1.0, -0.5, 0.1])
        assert result is None

        # Test with third L-moment >= 1
        result = Lmoments.generalized_logistic([1.0, 0.5, 1.0])
        assert result is None

        # Test with third L-moment <= -1
        result = Lmoments.generalized_logistic([1.0, 0.5, -1.0])
        assert result is None

    def test_generalized_logistic_small_third_moment(self):
        """
        Test generalized_logistic method with small third moment.

        Expected:
            generalized_logistic method returns parameters with shape=0
        """
        # Test with very small third moment
        result = Lmoments.generalized_logistic([10.0, 2.0, 0.0000001])
        assert result[2] == 0


class TestGeneralizedNormalMethod:
    """Test generalized_normal method."""

    def test_generalized_normal_valid_inputs(
        self, generalized_normal_lmoments, generalized_normal_parameters
    ):
        """
        Test generalized_normal method with valid inputs.

        Inputs:
            generalized_normal_lmoments: Valid L-moments for Generalized Normal distribution
            generalized_normal_parameters: Expected parameters for Generalized Normal distribution

        Expected:
            generalized_normal method returns the correct parameters
        """
        result = Lmoments.generalized_normal(generalized_normal_lmoments)
        np.testing.assert_almost_equal(result, generalized_normal_parameters, decimal=4)

    def test_generalized_normal_invalid_inputs(self):
        """
        Test generalized_normal method with invalid inputs.

        Expected:
            None is returned for invalid L-moments
        """
        # Test with negative second L-moment
        result = Lmoments.generalized_normal([1.0, -0.5, 0.1])
        assert result is None

        # Test with third L-moment >= 1
        result = Lmoments.generalized_normal([1.0, 0.5, 1.0])
        assert result is None

        # Test with third L-moment <= -1
        result = Lmoments.generalized_normal([1.0, 0.5, -1.0])
        assert result is None

    def test_generalized_normal_large_third_moment(self):
        """
        Test generalized_normal method with large third moment.

        Expected:
            generalized_normal method returns [0, -1, 0] for large third moment
        """
        # Test with third moment close to 0.95
        result = Lmoments.generalized_normal([10.0, 2.0, 0.95])
        assert result == [0, -1, 0]


class TestGeneralizedParetoMethod:
    """Test generalized_pareto method."""

    def test_generalized_pareto_valid_inputs(
        self, generalized_pareto_lmoments, generalized_pareto_parameters
    ):
        """
        Test generalized_pareto method with valid inputs.

        Inputs:
            generalized_pareto_lmoments: Valid L-moments for Generalized Pareto distribution
            generalized_pareto_parameters: Expected parameters for Generalized Pareto distribution

        Expected:
            generalized_pareto method returns the correct parameters
        """
        result = Lmoments.generalized_pareto(generalized_pareto_lmoments)
        np.testing.assert_almost_equal(result, generalized_pareto_parameters, decimal=4)

    def test_generalized_pareto_invalid_inputs(self):
        """
        Test generalized_pareto method with invalid inputs.

        Expected:
            None is returned for invalid L-moments
        """
        # Test with negative second L-moment
        result = Lmoments.generalized_pareto([1.0, -0.5, 0.1])
        assert result is None

        # Test with third L-moment >= 1
        result = Lmoments.generalized_pareto([1.0, 0.5, 1.0])
        assert result is None

        # Test with third L-moment <= -1
        result = Lmoments.generalized_pareto([1.0, 0.5, -1.0])
        assert result is None


class TestNormalMethod:
    """Test normal method."""

    def test_normal_valid_inputs(self, normal_lmoments, normal_parameters):
        """
        Test normal method with valid inputs.

        Inputs:
            normal_lmoments: Valid L-moments for Normal distribution
            normal_parameters: Expected parameters for Normal distribution

        Expected:
            normal method returns the correct parameters
        """
        result = Lmoments.normal(normal_lmoments)
        np.testing.assert_almost_equal(result, normal_parameters, decimal=5)

    def test_normal_invalid_inputs(self):
        """
        Test normal method with invalid inputs.

        Expected:
            None is returned for invalid L-moments
        """
        # Test with negative second L-moment
        result = Lmoments.normal([1.0, -0.5])
        assert result is None


class TestPearson3Method:
    """Test pearson_3 method."""

    def test_pearson_3_valid_inputs(self, pearson_3_lmoments, pearson_3_parameters):
        """
        Test pearson_3 method with valid inputs.

        Inputs:
            pearson_3_lmoments: Valid L-moments for Pearson Type III distribution
            pearson_3_parameters: Expected parameters for Pearson Type III distribution

        Expected:
            pearson_3 method returns the correct parameters
        """
        result = Lmoments.pearson_3(pearson_3_lmoments)
        np.testing.assert_almost_equal(result, pearson_3_parameters, decimal=4)

    def test_pearson_3_invalid_inputs(self):
        """
        Test pearson_3 method with invalid inputs.

        Expected:
            [0, 0, 0] is returned for invalid L-moments
        """
        # Test with negative second L-moment
        result = Lmoments.pearson_3([1.0, -0.5, 0.1])
        assert result == [0, 0, 0]

        # Test with third L-moment >= 1
        result = Lmoments.pearson_3([1.0, 0.5, 1.0])
        assert result == [0, 0, 0]

    def test_pearson_3_small_third_moment(self):
        """
        Test pearson_3 method with small third moment.

        Expected:
            pearson_3 method returns parameters with skew=0
        """
        # Test with very small third moment
        result = Lmoments.pearson_3([10.0, 2.0, 0.0000001])
        assert result[2] == 0


class TestWakebyMethod:
    """Test wakeby method."""

    def test_wakeby_valid_inputs(self, wakeby_lmoments, wakeby_parameters):
        """
        Test wakeby method with valid inputs.

        Inputs:
            wakeby_lmoments: Valid L-moments for Wakeby distribution
            wakeby_parameters: Expected parameters for Wakeby distribution

        Expected:
            wakeby method returns the correct parameters
        """
        result = Lmoments.wakeby(wakeby_lmoments)
        np.testing.assert_almost_equal(result, wakeby_parameters, decimal=4)

    def test_wakeby_invalid_inputs(self):
        """
        Test wakeby method with invalid inputs.

        Expected:
            None is returned for invalid L-moments
        """
        # Test with negative second L-moment
        result = Lmoments.wakeby([1.0, -0.5, 0.1, 0.05, 0.02])
        assert result is None

        # Test with third L-moment >= 1
        result = Lmoments.wakeby([1.0, 0.5, 1.0, 0.05, 0.02])
        assert result is None

        # Test with fourth L-moment >= 1
        result = Lmoments.wakeby([1.0, 0.5, 0.1, 1.0, 0.02])
        assert result is None

        # Test with fifth L-moment >= 1
        result = Lmoments.wakeby([1.0, 0.5, 0.1, 0.05, 1.0])
        assert result is None
