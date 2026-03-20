"""Tests for the Parameters dataclass."""

import pytest

from statista.distributions import Parameters


@pytest.fixture(scope="module")
def two_param():
    """2-parameter Parameters instance (Gumbel/Normal/Exponential)."""
    return Parameters(loc=500.0, scale=200.0)


@pytest.fixture(scope="module")
def three_param():
    """3-parameter Parameters instance (GEV)."""
    return Parameters(loc=463.8, scale=220.1, shape=-0.16)


class TestParametersInit:
    """Tests for Parameters.__init__ and __post_init__."""

    def test_two_param_creation(self):
        """Test creating a 2-parameter instance with loc and scale.

        Test scenario:
            Shape should default to None for 2-param distributions.
        """
        p = Parameters(loc=100.0, scale=50.0)
        assert p.loc == 100.0, f"Expected loc=100.0, got {p.loc}"
        assert p.scale == 50.0, f"Expected scale=50.0, got {p.scale}"
        assert p.shape is None, f"Expected shape=None, got {p.shape}"

    def test_three_param_creation(self):
        """Test creating a 3-parameter instance with loc, scale, and shape.

        Test scenario:
            All three fields should be stored as provided.
        """
        p = Parameters(loc=100.0, scale=50.0, shape=-0.2)
        assert p.loc == 100.0, f"Expected loc=100.0, got {p.loc}"
        assert p.scale == 50.0, f"Expected scale=50.0, got {p.scale}"
        assert p.shape == -0.2, f"Expected shape=-0.2, got {p.shape}"

    def test_zero_scale_raises(self):
        """Test that scale=0 raises ValueError.

        Test scenario:
            Scale must be strictly positive; zero is not valid.
        """
        with pytest.raises(ValueError, match="scale must be positive"):
            Parameters(loc=0.0, scale=0.0)

    def test_negative_scale_raises(self):
        """Test that negative scale raises ValueError.

        Test scenario:
            Scale must be strictly positive.
        """
        with pytest.raises(ValueError, match="scale must be positive, got -5.0"):
            Parameters(loc=0.0, scale=-5.0)

    def test_negative_loc_accepted(self):
        """Test that negative loc values are accepted.

        Test scenario:
            Location parameter has no sign constraint.
        """
        p = Parameters(loc=-100.0, scale=1.0)
        assert p.loc == -100.0, f"Expected loc=-100.0, got {p.loc}"

    def test_shape_zero_accepted(self):
        """Test that shape=0.0 is accepted and treated as 3-param.

        Test scenario:
            shape=0.0 is a valid value (e.g., Gumbel as a special case
            of GEV), distinct from shape=None.
        """
        p = Parameters(loc=0.0, scale=1.0, shape=0.0)
        assert p.shape == 0.0, f"Expected shape=0.0, got {p.shape}"
        assert len(p) == 3, "shape=0.0 should count as a 3-param instance"

    def test_very_small_scale_accepted(self):
        """Test that a very small positive scale is accepted.

        Test scenario:
            Any positive value, no matter how small, should be valid.
        """
        p = Parameters(loc=0.0, scale=1e-10)
        assert p.scale == 1e-10, f"Expected scale=1e-10, got {p.scale}"


class TestParametersRepr:
    """Tests for Parameters.__repr__."""

    def test_repr_two_param(self, two_param):
        """Test repr omits shape when it is None.

        Test scenario:
            2-param instance should show only loc and scale.
        """
        result = repr(two_param)
        assert result == "Parameters(loc=500.0, scale=200.0)", (
            f"Unexpected repr: {result}"
        )

    def test_repr_three_param(self, three_param):
        """Test repr includes shape when set.

        Test scenario:
            3-param instance should show loc, scale, and shape.
        """
        result = repr(three_param)
        assert result == "Parameters(loc=463.8, scale=220.1, shape=-0.16)", (
            f"Unexpected repr: {result}"
        )

    def test_repr_shape_zero(self):
        """Test repr shows shape=0.0 explicitly.

        Test scenario:
            shape=0.0 is a real value and must appear in repr.
        """
        p = Parameters(loc=0.0, scale=1.0, shape=0.0)
        assert "shape=0.0" in repr(p), f"shape=0.0 missing from repr: {repr(p)}"


class TestParametersGetitem:
    """Tests for Parameters.__getitem__."""

    @pytest.mark.parametrize(
        "key, expected",
        [("loc", 500.0), ("scale", 200.0)],
    )
    def test_getitem_valid_keys(self, two_param, key, expected):
        """Test bracket access for valid keys.

        Args:
            key: Parameter name.
            expected: Expected value.

        Test scenario:
            params["loc"] and params["scale"] should return the stored values.
        """
        assert two_param[key] == expected, (
            f"Expected params['{key}'] = {expected}, got {two_param[key]}"
        )

    def test_getitem_shape_none(self, two_param):
        """Test bracket access for shape returns None on 2-param.

        Test scenario:
            params["shape"] should return None, not raise KeyError.
        """
        assert two_param["shape"] is None, (
            f"Expected None for shape, got {two_param['shape']}"
        )

    def test_getitem_shape_set(self, three_param):
        """Test bracket access for shape returns value on 3-param."""
        assert three_param["shape"] == -0.16, (
            f"Expected -0.16, got {three_param['shape']}"
        )

    def test_getitem_invalid_key(self, two_param):
        """Test bracket access with invalid key raises KeyError.

        Test scenario:
            Keys other than loc/scale/shape should raise KeyError.
        """
        with pytest.raises(KeyError, match="invalid"):
            two_param["invalid"]


class TestParametersGet:
    """Tests for Parameters.get."""

    def test_get_existing_key(self, two_param):
        """Test get returns value for existing key."""
        assert two_param.get("loc") == 500.0, "get('loc') should return 500.0"

    def test_get_shape_none_returns_default(self, two_param):
        """Test get returns default when shape is None.

        Test scenario:
            For a 2-param instance, get("shape", 0.0) should return
            the default since shape is None.
        """
        assert two_param.get("shape", 0.0) == 0.0, (
            "get('shape', 0.0) should return default 0.0 when shape is None"
        )

    def test_get_shape_none_default_none(self, two_param):
        """Test get returns None when shape is None and no default given."""
        assert two_param.get("shape") is None, (
            "get('shape') with no default should return None"
        )

    def test_get_shape_set(self, three_param):
        """Test get returns actual value when shape is set."""
        assert three_param.get("shape", 0.0) == -0.16, (
            "get('shape') should return actual value, not default"
        )

    def test_get_unknown_key_returns_default(self, two_param):
        """Test get with unknown key returns default."""
        assert two_param.get("unknown", 42) == 42, (
            "get('unknown', 42) should return default 42"
        )

    def test_get_unknown_key_default_none(self, two_param):
        """Test get with unknown key and no default returns None."""
        assert two_param.get("unknown") is None, (
            "get('unknown') should return None"
        )


class TestParametersKeys:
    """Tests for Parameters.keys."""

    def test_keys_two_param(self, two_param):
        """Test keys returns ['loc', 'scale'] for 2-param."""
        assert two_param.keys() == ["loc", "scale"], (
            f"Expected ['loc', 'scale'], got {two_param.keys()}"
        )

    def test_keys_three_param(self, three_param):
        """Test keys returns ['loc', 'scale', 'shape'] for 3-param."""
        assert three_param.keys() == ["loc", "scale", "shape"], (
            f"Expected ['loc', 'scale', 'shape'], got {three_param.keys()}"
        )


class TestParametersValues:
    """Tests for Parameters.values."""

    def test_values_two_param(self, two_param):
        """Test values returns [loc, scale] for 2-param."""
        assert two_param.values() == [500.0, 200.0], (
            f"Expected [500.0, 200.0], got {two_param.values()}"
        )

    def test_values_three_param(self, three_param):
        """Test values returns [loc, scale, shape] for 3-param."""
        assert three_param.values() == [463.8, 220.1, -0.16], (
            f"Expected [463.8, 220.1, -0.16], got {three_param.values()}"
        )


class TestParametersItems:
    """Tests for Parameters.items."""

    def test_items_two_param(self, two_param):
        """Test items returns (key, value) pairs for 2-param."""
        expected = [("loc", 500.0), ("scale", 200.0)]
        assert two_param.items() == expected, (
            f"Expected {expected}, got {two_param.items()}"
        )

    def test_items_three_param(self, three_param):
        """Test items returns (key, value) pairs for 3-param."""
        expected = [("loc", 463.8), ("scale", 220.1), ("shape", -0.16)]
        assert three_param.items() == expected, (
            f"Expected {expected}, got {three_param.items()}"
        )


class TestParametersContains:
    """Tests for Parameters.__contains__."""

    @pytest.mark.parametrize("key", ["loc", "scale"])
    def test_contains_always_present(self, two_param, key):
        """Test that loc and scale are always 'in' Parameters.

        Args:
            key: Parameter name to check.
        """
        assert key in two_param, f"'{key}' should always be in Parameters"

    def test_contains_shape_when_none(self, two_param):
        """Test that 'shape' is not 'in' 2-param Parameters."""
        assert "shape" not in two_param, (
            "'shape' should not be 'in' a 2-param Parameters"
        )

    def test_contains_shape_when_set(self, three_param):
        """Test that 'shape' is 'in' 3-param Parameters."""
        assert "shape" in three_param, (
            "'shape' should be 'in' a 3-param Parameters"
        )

    def test_contains_unknown_key(self, two_param):
        """Test that unknown keys are not 'in' Parameters."""
        assert "unknown" not in two_param, (
            "'unknown' should not be 'in' Parameters"
        )


class TestParametersLen:
    """Tests for Parameters.__len__."""

    def test_len_two_param(self, two_param):
        """Test len is 2 for 2-param instance."""
        assert len(two_param) == 2, f"Expected 2, got {len(two_param)}"

    def test_len_three_param(self, three_param):
        """Test len is 3 for 3-param instance."""
        assert len(three_param) == 3, f"Expected 3, got {len(three_param)}"

    def test_len_shape_zero(self):
        """Test len is 3 when shape=0.0 (not None)."""
        p = Parameters(loc=0.0, scale=1.0, shape=0.0)
        assert len(p) == 3, f"Expected 3, got {len(p)}"


class TestParametersIter:
    """Tests for Parameters.__iter__."""

    def test_iter_two_param(self, two_param):
        """Test iteration yields key names for 2-param."""
        assert list(two_param) == ["loc", "scale"], (
            f"Expected ['loc', 'scale'], got {list(two_param)}"
        )

    def test_iter_three_param(self, three_param):
        """Test iteration yields key names for 3-param."""
        assert list(three_param) == ["loc", "scale", "shape"], (
            f"Expected ['loc', 'scale', 'shape'], got {list(three_param)}"
        )

    def test_dict_constructor_from_iter(self, three_param):
        """Test that dict(params) works via __iter__ + __getitem__.

        Test scenario:
            The combination of __iter__ and __getitem__ should allow
            constructing a plain dict from a Parameters instance.
        """
        d = {k: three_param[k] for k in three_param}
        assert d == {"loc": 463.8, "scale": 220.1, "shape": -0.16}, (
            f"Dict construction failed: {d}"
        )


class TestParametersEq:
    """Tests for Parameters.__eq__."""

    def test_eq_same_values(self):
        """Test equality between two Parameters with same values."""
        p1 = Parameters(loc=1.0, scale=2.0)
        p2 = Parameters(loc=1.0, scale=2.0)
        assert p1 == p2, "Parameters with same values should be equal"

    def test_eq_different_values(self):
        """Test inequality between Parameters with different values."""
        p1 = Parameters(loc=1.0, scale=2.0)
        p2 = Parameters(loc=1.0, scale=3.0)
        assert p1 != p2, "Parameters with different scale should not be equal"

    def test_eq_shape_none_vs_set(self):
        """Test inequality between 2-param and 3-param.

        Test scenario:
            shape=None and shape=-0.1 should be unequal even if
            loc and scale match.
        """
        p1 = Parameters(loc=1.0, scale=2.0)
        p2 = Parameters(loc=1.0, scale=2.0, shape=-0.1)
        assert p1 != p2, "2-param and 3-param should not be equal"

    def test_eq_three_param(self):
        """Test equality between two 3-param Parameters."""
        p1 = Parameters(loc=1.0, scale=2.0, shape=-0.1)
        p2 = Parameters(loc=1.0, scale=2.0, shape=-0.1)
        assert p1 == p2, "3-param Parameters with same values should be equal"

    def test_eq_with_dict_two_param(self):
        """Test equality between Parameters and a matching dict.

        Test scenario:
            Backward compatibility — Parameters should equal a dict
            with the same keys and values.
        """
        p = Parameters(loc=100.0, scale=50.0)
        d = {"loc": 100.0, "scale": 50.0}
        assert p == d, f"Parameters should equal matching dict, got {p} != {d}"
        assert d == p, "Dict should also equal matching Parameters"

    def test_eq_with_dict_three_param(self):
        """Test equality between 3-param Parameters and matching dict."""
        p = Parameters(loc=1.0, scale=2.0, shape=-0.1)
        d = {"loc": 1.0, "scale": 2.0, "shape": -0.1}
        assert p == d, "3-param Parameters should equal matching dict"

    def test_eq_with_dict_mismatch(self):
        """Test inequality between Parameters and a non-matching dict."""
        p = Parameters(loc=1.0, scale=2.0)
        d = {"loc": 1.0, "scale": 999.0}
        assert p != d, "Parameters should not equal dict with different values"

    def test_eq_with_unrelated_type(self):
        """Test that comparison with unrelated type returns NotImplemented.

        Test scenario:
            Comparing with a string or int should not raise, and should
            return False via NotImplemented.
        """
        p = Parameters(loc=1.0, scale=2.0)
        assert p != "not a parameters", "Should not equal a string"
        assert p != 42, "Should not equal an int"


class TestParametersIntegration:
    """Integration tests: Parameters used in distribution workflows."""

    def test_round_trip_from_fit_model(self):
        """Test that fit_model returns Parameters and it works in cdf.

        Test scenario:
            The full workflow — fit parameters from data, use them
            to compute CDF — should work with the Parameters dataclass.
        """
        import numpy as np
        from statista.distributions import Gumbel

        data = np.loadtxt("examples/data/time_series2.txt")
        dist = Gumbel(data=data)
        params = dist.fit_model(method="lmoments", test=False)
        assert isinstance(params, Parameters), (
            f"fit_model should return Parameters, got {type(params)}"
        )
        cdf_values = dist.cdf(parameters=params)
        assert len(cdf_values) == len(data), (
            f"CDF should return same length as data"
        )

    def test_parameters_accepted_by_constructor(self):
        """Test that Parameters can be passed to distribution constructor.

        Test scenario:
            A distribution should accept Parameters directly, not just dicts.
        """
        from statista.distributions import Normal

        params = Parameters(loc=500.0, scale=200.0)
        dist = Normal(parameters=params)
        assert dist.parameters.loc == 500.0, "loc should be accessible"
        assert dist.parameters.scale == 200.0, "scale should be accessible"

    def test_dict_still_accepted_by_constructor(self):
        """Test that plain dicts are still accepted and auto-converted.

        Test scenario:
            Backward compatibility — passing a dict should still work
            and produce a Parameters instance internally.
        """
        from statista.distributions import Normal

        dist = Normal(parameters={"loc": 500.0, "scale": 200.0})
        assert isinstance(dist.parameters, Parameters), (
            f"Dict should be auto-converted to Parameters, got {type(dist.parameters)}"
        )
        assert dist.parameters.loc == 500.0, "loc should be accessible"
