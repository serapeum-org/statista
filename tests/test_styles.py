"""Comprehensive tests for statista.styles module constants."""

import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")

import pytest

from statista.styles import (
    BOX_MEAN_PROP,
    COLORS,
    DEFAULT_FONTSIZE,
    DEFAULT_LINEWIDTH,
    FIG_SIZES,
    VIOLIN_PROP,
)


class TestColors:
    """Tests for the COLORS palette dictionary."""

    EXPECTED_KEYS = {
        "fitted_curve",
        "histogram_fill",
        "reference_line",
        "empirical_scatter",
        "ts_line",
        "negative_bar",
        "ci_bounds",
    }

    def test_has_all_expected_keys(self):
        """COLORS dict contains exactly the seven semantic role keys."""
        assert set(COLORS.keys()) == self.EXPECTED_KEYS, (
            f"COLORS keys mismatch: got {set(COLORS.keys())}"
        )

    def test_key_count(self):
        """COLORS contains exactly 7 entries."""
        assert len(COLORS) == 7, f"Expected 7 entries, got {len(COLORS)}"

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_values_are_strings(self, key):
        """Each COLORS value is a non-empty string.

        Args:
            key: Color role name to check.

        Test scenario:
            Every color entry must be a non-empty string that matplotlib
            can resolve.
        """
        val = COLORS[key]
        assert isinstance(val, str), f"COLORS['{key}'] should be str, got {type(val)}"
        assert len(val) > 0, f"COLORS['{key}'] must not be empty"

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_values_are_valid_matplotlib_colors(self, key):
        """Each COLORS value is a color string that matplotlib can parse.

        Args:
            key: Color role name to check.

        Test scenario:
            ``matplotlib.colors.to_rgba`` should not raise for any value.
        """
        try:
            mcolors.to_rgba(COLORS[key])
        except ValueError:
            pytest.fail(
                f"COLORS['{key}'] = {COLORS[key]!r} is not a valid matplotlib color"
            )

    def test_fitted_curve_value(self):
        """fitted_curve is the expected dark-blue hex."""
        assert COLORS["fitted_curve"] == "#27408B", (
            f"Expected '#27408B', got {COLORS['fitted_curve']!r}"
        )

    def test_histogram_fill_value(self):
        """histogram_fill is the expected crimson hex."""
        assert COLORS["histogram_fill"] == "#DC143C", (
            f"Expected '#DC143C', got {COLORS['histogram_fill']!r}"
        )

    def test_reference_line_value(self):
        """reference_line is the expected royal-blue hex."""
        assert COLORS["reference_line"] == "#3D59AB", (
            f"Expected '#3D59AB', got {COLORS['reference_line']!r}"
        )


class TestFigSizes:
    """Tests for the FIG_SIZES dimension dictionary."""

    EXPECTED_KEYS = {"single", "square", "wide", "ts_default"}

    def test_has_all_expected_keys(self):
        """FIG_SIZES dict contains exactly the four layout keys."""
        assert set(FIG_SIZES.keys()) == self.EXPECTED_KEYS, (
            f"FIG_SIZES keys mismatch: got {set(FIG_SIZES.keys())}"
        )

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_values_are_two_int_tuples(self, key):
        """Each FIG_SIZES value is a tuple of two positive integers.

        Args:
            key: Layout name to check.

        Test scenario:
            Width and height must both be positive integers.
        """
        val = FIG_SIZES[key]
        assert isinstance(val, tuple), (
            f"FIG_SIZES['{key}'] should be tuple, got {type(val)}"
        )
        assert len(val) == 2, (
            f"FIG_SIZES['{key}'] should have 2 elements, got {len(val)}"
        )
        w, h = val
        assert isinstance(w, int) and isinstance(h, int), (
            f"FIG_SIZES['{key}'] elements should be int, got ({type(w)}, {type(h)})"
        )
        assert w > 0 and h > 0, (
            f"FIG_SIZES['{key}'] dimensions must be positive, got ({w}, {h})"
        )

    @pytest.mark.parametrize(
        "key, expected",
        [
            ("single", (6, 5)),
            ("square", (6, 6)),
            ("wide", (10, 5)),
            ("ts_default", (8, 6)),
        ],
    )
    def test_specific_values(self, key, expected):
        """FIG_SIZES entries match the documented default dimensions.

        Args:
            key: Layout name.
            expected: Expected (width, height) tuple.

        Test scenario:
            Hard-coded sizes must not drift from documented values.
        """
        assert FIG_SIZES[key] == expected, (
            f"FIG_SIZES['{key}'] expected {expected}, got {FIG_SIZES[key]}"
        )


class TestBoxMeanProp:
    """Tests for the BOX_MEAN_PROP marker property dictionary."""

    def test_has_required_keys(self):
        """BOX_MEAN_PROP contains marker, markeredgecolor, markerfacecolor."""
        expected = {"marker", "markeredgecolor", "markerfacecolor"}
        assert set(BOX_MEAN_PROP.keys()) == expected, (
            f"BOX_MEAN_PROP keys mismatch: got {set(BOX_MEAN_PROP.keys())}"
        )

    def test_marker_value(self):
        """Default marker is 'x'."""
        assert BOX_MEAN_PROP["marker"] == "x", (
            f"Expected 'x', got {BOX_MEAN_PROP['marker']!r}"
        )

    def test_markeredgecolor_value(self):
        """Default markeredgecolor is 'w' (white)."""
        assert BOX_MEAN_PROP["markeredgecolor"] == "w", (
            f"Expected 'w', got {BOX_MEAN_PROP['markeredgecolor']!r}"
        )

    def test_markerfacecolor_is_valid(self):
        """markerfacecolor is a valid matplotlib color."""
        try:
            mcolors.to_rgba(BOX_MEAN_PROP["markerfacecolor"])
        except ValueError:
            pytest.fail(
                f"markerfacecolor {BOX_MEAN_PROP['markerfacecolor']!r} "
                "is not a valid matplotlib color"
            )


class TestViolinProp:
    """Tests for the VIOLIN_PROP appearance dictionary."""

    def test_has_required_keys(self):
        """VIOLIN_PROP contains face, edge, alpha."""
        expected = {"face", "edge", "alpha"}
        assert set(VIOLIN_PROP.keys()) == expected, (
            f"VIOLIN_PROP keys mismatch: got {set(VIOLIN_PROP.keys())}"
        )

    def test_face_is_valid_color(self):
        """face value is a valid matplotlib color."""
        try:
            mcolors.to_rgba(VIOLIN_PROP["face"])
        except ValueError:
            pytest.fail(
                f"VIOLIN_PROP['face'] = {VIOLIN_PROP['face']!r} "
                "is not a valid matplotlib color"
            )

    def test_edge_is_valid_color(self):
        """edge value is a valid matplotlib color."""
        try:
            mcolors.to_rgba(VIOLIN_PROP["edge"])
        except ValueError:
            pytest.fail(
                f"VIOLIN_PROP['edge'] = {VIOLIN_PROP['edge']!r} "
                "is not a valid matplotlib color"
            )

    def test_alpha_in_valid_range(self):
        """alpha is a float between 0 and 1."""
        alpha = VIOLIN_PROP["alpha"]
        assert isinstance(alpha, (int, float)), (
            f"alpha should be numeric, got {type(alpha)}"
        )
        assert 0 <= alpha <= 1, (
            f"alpha must be in [0, 1], got {alpha}"
        )


class TestDefaultLinewidth:
    """Tests for DEFAULT_LINEWIDTH constant."""

    def test_value(self):
        """DEFAULT_LINEWIDTH is 2."""
        assert DEFAULT_LINEWIDTH == 2, (
            f"Expected 2, got {DEFAULT_LINEWIDTH}"
        )

    def test_type(self):
        """DEFAULT_LINEWIDTH is an integer."""
        assert isinstance(DEFAULT_LINEWIDTH, int), (
            f"Expected int, got {type(DEFAULT_LINEWIDTH)}"
        )


class TestDefaultFontsize:
    """Tests for DEFAULT_FONTSIZE constant."""

    def test_value(self):
        """DEFAULT_FONTSIZE is 11."""
        assert DEFAULT_FONTSIZE == 11, (
            f"Expected 11, got {DEFAULT_FONTSIZE}"
        )

    def test_type(self):
        """DEFAULT_FONTSIZE is an integer."""
        assert isinstance(DEFAULT_FONTSIZE, int), (
            f"Expected int, got {type(DEFAULT_FONTSIZE)}"
        )
