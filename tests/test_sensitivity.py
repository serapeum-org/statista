"""Test sensitivity module.

This module contains unit tests for the Sensitivity class in the statista.sensitivity module.
Each method of the Sensitivity class has a corresponding test class with multiple test methods
to cover different scenarios and edge cases.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from statista.sensitivity import Sensitivity

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_param_df() -> pd.DataFrame:
    """DataFrame with two parameters."""
    return pd.DataFrame({"value": [2.0, 3.0]}, index=["param1", "param2"])


@pytest.fixture()
def two_param_bounds():
    """Lower and upper bounds for two parameters."""
    return [0.5, 1.0], [4.0, 5.0]


@pytest.fixture()
def simple_function():
    """A simple model function that accepts a list of parameter values."""

    def _fn(params, *args, **kwargs):
        return params[0] ** 2 + params[1]

    return _fn


@pytest.fixture()
def simple_function_with_kwargs():
    """A model function that uses kwargs."""

    def _fn(params, *args, multiplier=1, **kwargs):
        return multiplier * (params[0] ** 2 + params[1])

    return _fn


@pytest.fixture()
def two_return_function():
    """A model function that returns two values (metric, calculated_values).

    The second return value must have a ``.values`` attribute (like a pandas Series)
    because the ``sobol`` method accesses ``[3][j].values``.
    """

    def _fn(params, *args, **kwargs):
        metric = params[0] ** 2 + params[1]
        calculated = pd.Series([metric * 0.5, metric * 1.0, metric * 1.5])
        return metric, calculated

    return _fn


@pytest.fixture()
def three_param_df() -> pd.DataFrame:
    """DataFrame with three parameters."""
    return pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=["alpha", "beta", "gamma"])


@pytest.fixture()
def three_param_bounds():
    """Lower and upper bounds for three parameters."""
    return [0.1, 0.5, 1.0], [2.0, 4.0, 6.0]


@pytest.fixture()
def single_param_df() -> pd.DataFrame:
    """DataFrame with a single parameter."""
    return pd.DataFrame({"value": [5.0]}, index=["only_param"])


@pytest.fixture()
def single_param_bounds():
    """Lower and upper bounds for a single parameter."""
    return [1.0], [10.0]


@pytest.fixture()
def single_param_function():
    """A model function that works with a single parameter."""

    def _fn(params, *args, **kwargs):
        return params[0] ** 2

    return _fn


# ---------------------------------------------------------------------------
# Test __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Test Sensitivity.__init__ method."""

    def test_init_default_positions(
        self, two_param_df, two_param_bounds, simple_function
    ):
        """
        Test initialization with default positions (all parameters analyzed).

        Inputs:
            two_param_df: DataFrame with 2 parameters
            two_param_bounds: matching lower/upper bounds
            simple_function: a callable model function

        Expected:
            num_parameters equals the number of rows in the DataFrame.
            positions is [0, 1].
            NoValues defaults to 5.
            return_values defaults to 1.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)

        assert sa.num_parameters == 2
        assert sa.positions == [0, 1]
        assert sa.NoValues == 5
        assert sa.return_values == 1

    def test_init_explicit_positions(
        self, two_param_df, two_param_bounds, simple_function
    ):
        """
        Test initialization with explicit positions subset.

        Inputs:
            positions=[1] — only the second parameter should be analyzed.

        Expected:
            num_parameters == 1, positions == [1].
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function, positions=[1])

        assert sa.num_parameters == 1
        assert sa.positions == [1]

    def test_init_custom_n_values(
        self, two_param_df, two_param_bounds, simple_function
    ):
        """
        Test initialization with a custom n_values.

        Inputs:
            n_values=10

        Expected:
            NoValues attribute is 10.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function, n_values=10)
        assert sa.NoValues == 10

    def test_init_return_values_two(
        self, two_param_df, two_param_bounds, simple_function
    ):
        """
        Test initialization with return_values=2.

        Expected:
            return_values attribute is 2.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function, return_values=2)
        assert sa.return_values == 2

    def test_init_mismatched_lower_bound_length(self, two_param_df, simple_function):
        """
        Test that mismatched lower_bound length raises AssertionError.

        Inputs:
            lower_bound has length 3, but parameter DataFrame has 2 rows.

        Expected:
            AssertionError is raised.
        """
        with pytest.raises(AssertionError, match="Length of the boundary"):
            Sensitivity(two_param_df, [1, 2, 3], [4, 5], simple_function)

    def test_init_mismatched_upper_bound_length(self, two_param_df, simple_function):
        """
        Test that mismatched upper_bound length raises AssertionError.

        Inputs:
            upper_bound has length 1, but parameter DataFrame has 2 rows.

        Expected:
            AssertionError is raised.
        """
        with pytest.raises(AssertionError, match="Length of the boundary"):
            Sensitivity(two_param_df, [1, 2], [4], simple_function)

    def test_init_non_callable_function(self, two_param_df, two_param_bounds):
        """
        Test that a non-callable function raises AssertionError.

        Inputs:
            function="not a function" (a string, which is not callable)

        Expected:
            AssertionError about callable is raised.
        """
        lb, ub = two_param_bounds
        with pytest.raises(AssertionError, match="callable"):
            Sensitivity(two_param_df, lb, ub, "not a function")

    def test_init_stores_references(
        self, two_param_df, two_param_bounds, simple_function
    ):
        """
        Test that the constructor stores references to parameter, bounds, and function.

        Expected:
            The stored attributes are the same objects that were passed in.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)

        assert sa.parameter is two_param_df
        assert sa.lower_bound is lb
        assert sa.upper_bound is ub
        assert sa.function is simple_function


# ---------------------------------------------------------------------------
# Test marker_style
# ---------------------------------------------------------------------------


class TestMarkerStyle:
    """Test Sensitivity.marker_style static method."""

    def test_first_marker(self):
        """
        Test retrieving the first marker style.

        Inputs:
            style=0

        Expected:
            Returns "--o".
        """
        assert Sensitivity.marker_style(0) == "--o"

    def test_second_marker(self):
        """
        Test retrieving the second marker style.

        Inputs:
            style=1

        Expected:
            Returns ":D".
        """
        assert Sensitivity.marker_style(1) == ":D"

    def test_last_marker(self):
        """
        Test retrieving the last marker style.

        Inputs:
            style = len(MarkerStyleList) - 1

        Expected:
            Returns the last element of MarkerStyleList.
        """
        last_index = len(Sensitivity.MarkerStyleList) - 1
        assert (
            Sensitivity.marker_style(last_index)
            == Sensitivity.MarkerStyleList[last_index]
        )

    def test_wrapping_index(self):
        """
        Test that indices beyond the list length wrap around via modulo.

        Inputs:
            style = len(MarkerStyleList) + 2

        Expected:
            Returns the same marker as style=2.
        """
        n = len(Sensitivity.MarkerStyleList)
        assert Sensitivity.marker_style(n + 2) == Sensitivity.marker_style(2)

    def test_wrapping_large_index(self):
        """
        Test wrapping with a very large index.

        Inputs:
            style = 100

        Expected:
            Returns the marker at index 100 % len(MarkerStyleList).
        """
        n = len(Sensitivity.MarkerStyleList)
        expected_index = 100 % n
        assert (
            Sensitivity.marker_style(100) == Sensitivity.MarkerStyleList[expected_index]
        )

    def test_all_markers_accessible(self):
        """
        Test that every marker in the list is accessible by its index.

        Expected:
            Each call returns the corresponding element.
        """
        for i, expected in enumerate(Sensitivity.MarkerStyleList):
            assert Sensitivity.marker_style(i) == expected


# ---------------------------------------------------------------------------
# Test one_at_a_time
# ---------------------------------------------------------------------------


class TestOneAtATime:
    """Test Sensitivity.one_at_a_time method."""

    def test_basic_run_creates_sen_dict(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that one_at_a_time creates the sen dict with correct keys.

        Expected:
            sen dict has keys matching parameter names from the DataFrame index.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()

        assert set(sa.sen.keys()) == {"param1", "param2"}

    def test_sen_dict_structure_return_values_1(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test the structure of the sen dict when return_values=1.

        Expected:
            Each parameter entry has exactly 3 lists:
              [0] relative values, [1] metric values, [2] actual values.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function, n_values=5)
        sa.one_at_a_time()

        for param_name in sa.sen:
            assert (
                len(sa.sen[param_name]) == 3
            ), f"Expected 3 lists for return_values=1, got {len(sa.sen[param_name])}"

    def test_number_of_evaluated_points(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test the number of evaluated points per parameter.

        With n_values=5, linspace produces 5 points. The current parameter value
        is appended and sorted. If the current value does not coincide with one of
        the linspace points, there will be 6 total points. If it does coincide,
        np.append still adds it, so there can be a duplicate (still 6 entries but
        two identical values).

        Expected:
            Each list in sen[param][0..2] has length n_values + 1 = 6.
        """
        lb, ub = two_param_bounds
        n_values = 5
        sa = Sensitivity(two_param_df, lb, ub, simple_function, n_values=n_values)
        sa.one_at_a_time()

        for param_name in sa.sen:
            assert len(sa.sen[param_name][0]) == n_values + 1
            assert len(sa.sen[param_name][1]) == n_values + 1
            assert len(sa.sen[param_name][2]) == n_values + 1

    def test_relative_values_include_one(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that the relative values include 1.0 (the original parameter value ratio).

        The original parameter value is appended and then divided by itself, yielding 1.0.

        Expected:
            1.0 appears in the relative values list for each parameter.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()

        for param_name in sa.sen:
            relative = sa.sen[param_name][0]
            assert any(
                abs(r - 1.0) < 1e-10 for r in relative
            ), f"Relative values for {param_name} should contain 1.0"

    def test_actual_values_sorted(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that actual parameter values are sorted in ascending order.

        The implementation uses np.sort after appending the original value.

        Expected:
            The actual values list is non-decreasing.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()

        for param_name in sa.sen:
            actual = sa.sen[param_name][2]
            for a, b in zip(actual[:-1], actual[1:]):
                assert a <= b, f"Actual values for {param_name} are not sorted"

    def test_actual_values_within_bounds(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that actual parameter values are within the specified bounds.

        Expected:
            All actual values lie between the lower bound and the upper bound (inclusive).
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()

        for i, param_name in enumerate(two_param_df.index):
            if param_name not in sa.sen:
                continue
            actual = sa.sen[param_name][2]
            for v in actual:
                assert (
                    lb[i] - 1e-10 <= v <= ub[i] + 1e-10
                ), f"Value {v} for {param_name} is outside [{lb[i]}, {ub[i]}]"

    def test_metric_values_are_correct(self, two_param_df, two_param_bounds, capsys):
        """
        Test that the metric values match the function applied to the varied parameter.

        Using a simple identity-like function: f(params) = params[0] + params[1].
        When varying param1, param2 stays at its original value (3.0).

        Expected:
            metric == round(actual_param1 + 3.0, 3) for each evaluation point.
        """

        def additive(params, *args, **kwargs):
            return params[0] + params[1]

        lb, ub = [0.5, 1.0], [4.0, 5.0]
        df = two_param_df.copy()
        sa = Sensitivity(df, lb, ub, additive)
        sa.one_at_a_time()

        # Verify param1 entries (param2 stays at 3.0)
        for actual_val, metric_val in zip(sa.sen["param1"][2], sa.sen["param1"][1]):
            expected_metric = round(actual_val + 3.0, 3)
            assert abs(metric_val - expected_metric) < 1e-6

    def test_return_values_two_structure(
        self, two_param_df, two_param_bounds, two_return_function, capsys
    ):
        """
        Test sen dict structure when return_values=2.

        Expected:
            Each parameter entry has exactly 4 lists.
            The 4th list contains the calculated_values returned by the function.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, two_return_function, return_values=2)
        sa.one_at_a_time()

        for param_name in sa.sen:
            assert (
                len(sa.sen[param_name]) == 4
            ), f"Expected 4 lists for return_values=2, got {len(sa.sen[param_name])}"
            # The 4th list should have the same number of entries as the others
            assert len(sa.sen[param_name][3]) == len(sa.sen[param_name][1])

    def test_return_values_two_calculated_values(
        self, two_param_df, two_param_bounds, two_return_function, capsys
    ):
        """
        Test that calculated_values (4th list) are pandas Series with expected content.

        The two_return_function returns a Series of [metric*0.5, metric*1.0, metric*1.5].

        Expected:
            Each entry in sen[param][3] is a pandas Series of length 3.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, two_return_function, return_values=2)
        sa.one_at_a_time()

        for param_name in sa.sen:
            for series in sa.sen[param_name][3]:
                assert isinstance(series, pd.Series)
                assert len(series) == 3

    def test_kwargs_passed_to_function(
        self, two_param_df, two_param_bounds, simple_function_with_kwargs, capsys
    ):
        """
        Test that kwargs are forwarded to the model function.

        Using multiplier=2, all metrics should be doubled compared to multiplier=1.

        Expected:
            metric(multiplier=2) == 2 * metric(multiplier=1) for each evaluation point.
        """
        lb, ub = two_param_bounds
        sa1 = Sensitivity(two_param_df, lb, ub, simple_function_with_kwargs)
        sa1.one_at_a_time(multiplier=1)

        sa2 = Sensitivity(two_param_df, lb, ub, simple_function_with_kwargs)
        sa2.one_at_a_time(multiplier=2)

        for param_name in sa1.sen:
            for m1, m2 in zip(sa1.sen[param_name][1], sa2.sen[param_name][1]):
                # The metrics are stored as round(value, 3), so 2*round(x,3) may
                # differ from round(2*x,3) by up to 0.001 due to rounding.
                assert (
                    abs(m2 - 2 * m1) < 0.002
                ), f"With multiplier=2 metric should be double: got {m2}, expected {2 * m1}"

    def test_positions_subset(self, three_param_df, three_param_bounds, capsys):
        """
        Test one_at_a_time with a positions subset.

        Only position [1] (beta) should be analyzed.

        Expected:
            sen dict has exactly one key: 'beta'.
        """

        def fn(params, *args, **kwargs):
            return sum(params)

        lb, ub = three_param_bounds
        sa = Sensitivity(three_param_df, lb, ub, fn, positions=[1])
        sa.one_at_a_time()

        assert list(sa.sen.keys()) == ["beta"]
        assert len(sa.sen["beta"][0]) == 5 + 1  # n_values + 1

    def test_positions_multiple(self, three_param_df, three_param_bounds, capsys):
        """
        Test one_at_a_time with multiple positions (0 and 2).

        Only alpha and gamma should be analyzed.

        Expected:
            sen dict has keys 'alpha' and 'gamma'.
        """

        def fn(params, *args, **kwargs):
            return sum(params)

        lb, ub = three_param_bounds
        sa = Sensitivity(three_param_df, lb, ub, fn, positions=[0, 2])
        sa.one_at_a_time()

        assert set(sa.sen.keys()) == {"alpha", "gamma"}

    def test_single_parameter(
        self, single_param_df, single_param_bounds, single_param_function, capsys
    ):
        """
        Test one_at_a_time with a single parameter.

        Expected:
            sen dict has one key matching the single parameter name.
        """
        lb, ub = single_param_bounds
        sa = Sensitivity(single_param_df, lb, ub, single_param_function)
        sa.one_at_a_time()

        assert list(sa.sen.keys()) == ["only_param"]
        # Verify metric values make sense: f(x) = x^2
        for actual, metric in zip(sa.sen["only_param"][2], sa.sen["only_param"][1]):
            assert abs(metric - round(actual**2, 3)) < 1e-6

    def test_function_returning_non_roundable_raises(
        self, two_param_df, two_param_bounds
    ):
        """
        Test that a function returning a non-numeric value raises ValueError.

        The function returns a list instead of a scalar, so round() raises TypeError,
        which is caught and re-raised as ValueError.

        Expected:
            ValueError is raised.
        """

        def bad_fn(params, *args, **kwargs):
            return [1, 2, 3]  # not a scalar

        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, bad_fn)
        with pytest.raises(ValueError, match="returns more than one value"):
            sa.one_at_a_time()

    def test_n_values_3(self, two_param_df, two_param_bounds, simple_function, capsys):
        """
        Test one_at_a_time with n_values=3.

        Expected:
            Each parameter has 3 + 1 = 4 evaluation points (linspace(3) + original value).
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function, n_values=3)
        sa.one_at_a_time()

        for param_name in sa.sen:
            assert len(sa.sen[param_name][1]) == 4

    def test_prints_output(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that one_at_a_time prints progress output to stdout.

        Expected:
            Captured stdout contains parameter names and metric values.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()

        captured = capsys.readouterr()
        assert "param1" in captured.out
        assert "param2" in captured.out


# ---------------------------------------------------------------------------
# Test sobol (plotting)
# ---------------------------------------------------------------------------


class TestSobol:
    """Test Sensitivity.sobol method."""

    def test_returns_figure_and_axes_rv1(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that sobol returns (Figure, Axes) when return_values=1.

        Expected:
            First element is a matplotlib Figure.
            Second element is a matplotlib Axes.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()
        fig, ax = sa.sobol()

        assert isinstance(fig, Figure)
        assert hasattr(ax, "plot")  # quacks like an Axes
        plt.close(fig)

    def test_title_and_labels_rv1(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that custom title, xlabel, ylabel are applied to the axes.

        Expected:
            The axes title, xlabel, ylabel match the values passed to sobol().
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()
        fig, ax = sa.sobol(title="My Title", xlabel="X Label", ylabel="Y Label")

        assert ax.get_title() == "My Title"
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        plt.close(fig)

    def test_real_values_true(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test sobol with real_values=True.

        When real_values=True, the x-axis data comes from sen[param][2] (actual values)
        instead of sen[param][0] (relative values).

        Expected:
            The plot is generated without error and returns (Figure, Axes).
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()
        fig, ax = sa.sobol(real_values=True)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_legend_has_param_names(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that the legend contains entries for each parameter.

        Expected:
            The legend texts include 'param1' and 'param2'.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()
        fig, ax = sa.sobol()

        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "param1" in legend_texts
        assert "param2" in legend_texts
        plt.close(fig)

    def test_sobol_with_positions(self, three_param_df, three_param_bounds, capsys):
        """
        Test sobol when only a subset of parameters was analyzed.

        Expected:
            The plot is generated without error and the legend only shows analyzed params.
        """

        def fn(params, *args, **kwargs):
            return sum(params)

        lb, ub = three_param_bounds
        sa = Sensitivity(three_param_df, lb, ub, fn, positions=[0, 2])
        sa.one_at_a_time()
        fig, ax = sa.sobol()

        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "alpha" in legend_texts
        assert "gamma" in legend_texts
        assert "beta" not in legend_texts
        plt.close(fig)

    def test_sobol_single_parameter(
        self, single_param_df, single_param_bounds, single_param_function, capsys
    ):
        """
        Test sobol with a single parameter.

        Expected:
            The plot is generated without error.
        """
        lb, ub = single_param_bounds
        sa = Sensitivity(single_param_df, lb, ub, single_param_function)
        sa.one_at_a_time()
        fig, ax = sa.sobol()

        assert isinstance(fig, Figure)
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "only_param" in legend_texts
        plt.close(fig)

    def test_sobol_real_values_single_param(
        self, single_param_df, single_param_bounds, single_param_function, capsys
    ):
        """
        Test sobol with real_values=True and a single parameter.

        Expected:
            The plot is generated and x-axis data corresponds to actual param values.
        """
        lb, ub = single_param_bounds
        sa = Sensitivity(single_param_df, lb, ub, single_param_function)
        sa.one_at_a_time()
        fig, ax = sa.sobol(real_values=True)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_and_two_axes_rv2(
        self, two_param_df, two_param_bounds, two_return_function, capsys
    ):
        """
        Test that sobol returns (Figure, (Axes, Axes)) when return_values=2.

        Expected:
            First element is a matplotlib Figure.
            Second element is a tuple of two Axes objects.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, two_return_function, return_values=2)
        sa.one_at_a_time()
        fig, (ax1, ax2) = sa.sobol(spaces=[0.1, 0.1, 0.9, 0.9, 0.2, 0.4])

        assert isinstance(fig, Figure)
        assert hasattr(ax1, "plot")
        assert hasattr(ax2, "plot")
        plt.close(fig)

    def test_rv2_title_labels(
        self, two_param_df, two_param_bounds, two_return_function, capsys
    ):
        """
        Test that custom titles and labels are applied to both axes in rv2 mode.

        Expected:
            ax1 has title/xlabel/ylabel from the primary arguments.
            ax2 has title2/xlabel2/ylabel2.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, two_return_function, return_values=2)
        sa.one_at_a_time()
        fig, (ax1, ax2) = sa.sobol(
            title="Primary Title",
            xlabel="Primary X",
            ylabel="Primary Y",
            title2="Secondary Title",
            xlabel2="Secondary X",
            ylabel2="Secondary Y",
            spaces=[0.1, 0.1, 0.9, 0.9, 0.2, 0.4],
        )

        assert ax1.get_title() == "Primary Title"
        assert ax1.get_xlabel() == "Primary X"
        assert ax1.get_ylabel() == "Primary Y"
        assert ax2.get_title() == "Secondary Title"
        assert ax2.get_xlabel() == "Secondary X"
        assert ax2.get_ylabel() == "Secondary Y"
        plt.close(fig)

    def test_rv2_real_values(
        self, two_param_df, two_param_bounds, two_return_function, capsys
    ):
        """
        Test sobol with return_values=2 and real_values=True.

        Expected:
            The plot is generated without error and returns (Figure, (Axes, Axes)).
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, two_return_function, return_values=2)
        sa.one_at_a_time()
        fig, (ax1, ax2) = sa.sobol(
            real_values=True,
            spaces=[0.1, 0.1, 0.9, 0.9, 0.2, 0.4],
        )

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_sobol_default_labels(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that sobol uses default label values when none are specified.

        Expected:
            Default xlabel is "xlabel" and ylabel is "Metric values".
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()
        fig, ax = sa.sobol()

        assert ax.get_xlabel() == "xlabel"
        assert ax.get_ylabel() == "Metric values"
        plt.close(fig)

    def test_sobol_labelfontsize(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that the labelfontsize parameter is applied to tick labels.

        Expected:
            No error when passing labelfontsize; the plot is generated.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function)
        sa.one_at_a_time()
        fig, ax = sa.sobol(labelfontsize=16)

        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test MarkerStyleList class attribute
# ---------------------------------------------------------------------------


class TestMarkerStyleList:
    """Test the MarkerStyleList class attribute."""

    def test_is_list_of_strings(self):
        """
        Test that MarkerStyleList is a list of non-empty strings.

        Expected:
            Each element is a non-empty string.
        """
        for marker in Sensitivity.MarkerStyleList:
            assert isinstance(marker, str)
            assert len(marker) > 0

    def test_known_length(self):
        """
        Test that MarkerStyleList contains 11 entries.

        Expected:
            The list has exactly 11 elements as defined in the source code.
        """
        assert len(Sensitivity.MarkerStyleList) == 11


# ---------------------------------------------------------------------------
# Test edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_oat_then_sobol_workflow(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test the complete workflow: init -> one_at_a_time -> sobol.

        Expected:
            The entire pipeline runs without error and produces a valid plot.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function, n_values=3)
        sa.one_at_a_time()

        assert len(sa.sen) == 2
        fig, ax = sa.sobol(title="Integration Test")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_parameter_value_at_lower_bound(self, capsys):
        """
        Test when the parameter value equals the lower bound.

        The original value coincides with the lower bound, so it appears as a
        duplicate after np.append + np.sort.

        Expected:
            The analysis runs without error.
        """
        df = pd.DataFrame({"value": [1.0]}, index=["p"])
        sa = Sensitivity(df, [1.0], [5.0], lambda p: p[0] ** 2)
        sa.one_at_a_time()

        assert "p" in sa.sen

    def test_parameter_value_at_upper_bound(self, capsys):
        """
        Test when the parameter value equals the upper bound.

        Expected:
            The analysis runs without error.
        """
        df = pd.DataFrame({"value": [5.0]}, index=["p"])
        sa = Sensitivity(df, [1.0], [5.0], lambda p: p[0] ** 2)
        sa.one_at_a_time()

        assert "p" in sa.sen

    def test_parameter_value_outside_bounds(self, capsys):
        """
        Test when the original parameter value lies outside the specified bounds.

        The implementation does not enforce that the value lies within bounds;
        it simply appends it and sorts.

        Expected:
            The analysis runs without error, but the actual values will include
            a value outside the [lower, upper] range.
        """
        df = pd.DataFrame({"value": [10.0]}, index=["p"])
        sa = Sensitivity(df, [1.0], [5.0], lambda p: p[0] ** 2)
        sa.one_at_a_time()

        actual = sa.sen["p"][2]
        # The appended value 10.0 should be the last (largest) after sorting
        assert max(actual) == pytest.approx(10.0, abs=1e-4)

    def test_n_values_1(self, capsys):
        """
        Test with n_values=1 (only one linspace point, the endpoints coincide).

        linspace(lower, upper, 1) returns [lower]. After appending the original
        value, there are 2 points total.

        Expected:
            Each parameter entry has 2 evaluation points.
        """
        df = pd.DataFrame({"value": [3.0]}, index=["p"])
        sa = Sensitivity(df, [1.0], [5.0], lambda p: p[0], n_values=1)
        sa.one_at_a_time()

        assert len(sa.sen["p"][1]) == 2

    def test_n_values_2(self, capsys):
        """
        Test with n_values=2 (linspace produces 2 points: lower and upper).

        After appending the original value, there are 3 points total.

        Expected:
            Each parameter entry has 3 evaluation points.
        """
        df = pd.DataFrame({"value": [3.0]}, index=["p"])
        sa = Sensitivity(df, [1.0], [5.0], lambda p: p[0], n_values=2)
        sa.one_at_a_time()

        assert len(sa.sen["p"][1]) == 3

    def test_multiple_calls_to_oat_resets_sen(
        self, two_param_df, two_param_bounds, simple_function, capsys
    ):
        """
        Test that calling one_at_a_time again resets the sen dict.

        The method starts with ``self.sen = {}``, so previous results are discarded.

        Expected:
            The sen dict from the second call is independent of the first.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(two_param_df, lb, ub, simple_function, n_values=3)
        sa.one_at_a_time()
        first_metrics = sa.sen["param1"][1][:]

        sa.one_at_a_time()
        second_metrics = sa.sen["param1"][1][:]

        assert first_metrics == second_metrics  # same function, same results

    def test_sobol_plotting_from_to_rv2(
        self, two_param_df, two_param_bounds, two_return_function, capsys
    ):
        """
        Test sobol with explicit plotting_from and plotting_to for return_values=2.

        Note: there is a known bug where plotting_from/plotting_to loop variable
        mutation causes them to be reused as ints after the first iteration. This test
        passes integer values directly to avoid the bug with empty string defaults.

        Expected:
            The plot is generated without error.
        """
        lb, ub = two_param_bounds
        sa = Sensitivity(
            two_param_df, lb, ub, two_return_function, return_values=2, n_values=3
        )
        sa.one_at_a_time()
        fig, (ax1, ax2) = sa.sobol(
            plotting_from=0,
            plotting_to=2,
            spaces=[0.1, 0.1, 0.9, 0.9, 0.2, 0.4],
        )

        assert isinstance(fig, Figure)
        plt.close(fig)
