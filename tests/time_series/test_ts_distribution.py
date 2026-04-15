"""Tests for the DistributionMixin (Phase 6)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


class TestQQPlot:
    """Tests for qq_plot() method."""

    def test_returns_figure_axes(self):
        """Should return (Figure, Axes)."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.qq_plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_normal_distribution(self):
        """Normal data against normal QQ should show points near 1:1 line."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        fig, ax = ts.qq_plot(distribution="norm")
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_other_distribution(self):
        """Should work with other scipy distributions (e.g., expon)."""
        np.random.seed(42)
        ts = TimeSeries(np.random.exponential(2.0, 100))
        fig, ax = ts.qq_plot(distribution="expon")
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_column_parameter(self):
        """Should use specified column."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        fig, ax = ts.qq_plot(column="B")
        assert "B" in ax.get_title()
        plt.close(fig)

    def test_custom_labels(self):
        """Custom title should override default."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.qq_plot(title="My QQ")
        assert ax.get_title() == "My QQ"
        plt.close(fig)


class TestPPPlot:
    """Tests for pp_plot() method."""

    def test_returns_figure_axes(self):
        """Should return (Figure, Axes)."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.pp_plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_has_reference_line(self):
        """Plot should include a 1:1 reference line."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.pp_plot()
        lines = ax.get_lines()
        assert len(lines) >= 1, "Should have at least the 1:1 reference line"
        plt.close(fig)

    def test_column_parameter(self):
        """Should use specified column."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["X", "Y"])
        fig, ax = ts.pp_plot(column="Y")
        assert "Y" in ax.get_title()
        plt.close(fig)


class TestNormalityTest:
    """Tests for normality_test() method."""

    def test_normal_data_is_normal(self):
        """Normal data should pass the normality test."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        result = ts.normality_test()
        assert (
            result.loc["Series1", "is_normal"] == True
        ), f"Normal data should be classified as normal, p={result.loc['Series1', 'p_value']}"  # noqa: E712

    def test_uniform_data_not_normal(self):
        """Uniform data should fail the normality test."""
        np.random.seed(42)
        ts = TimeSeries(np.random.uniform(0, 1, 200))
        result = ts.normality_test()
        assert (
            result.loc["Series1", "is_normal"] == False
        ), "Uniform data should not be classified as normal"  # noqa: E712

    def test_returns_expected_columns(self):
        """Result should have test_name, statistic, p_value, is_normal, conclusion."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.normality_test()
        expected = ["test_name", "statistic", "p_value", "is_normal", "conclusion"]
        assert set(expected).issubset(set(result.columns))

    def test_auto_uses_shapiro_for_small_n(self):
        """Auto method should use Shapiro-Wilk for n < 5000."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.normality_test(method="auto")
        assert result.loc["Series1", "test_name"] == "Shapiro-Wilk"

    def test_dagostino_method(self):
        """D'Agostino-Pearson method should work."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        result = ts.normality_test(method="dagostino")
        assert result.loc["Series1", "test_name"] == "D'Agostino-Pearson"

    def test_anderson_method(self):
        """Anderson-Darling method should work."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        result = ts.normality_test(method="anderson")
        assert result.loc["Series1", "test_name"] == "Anderson-Darling"

    def test_jarque_bera_method(self):
        """Jarque-Bera method should work."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        result = ts.normality_test(method="jarque_bera")
        assert result.loc["Series1", "test_name"] == "Jarque-Bera"

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        ts = TimeSeries(np.random.randn(50))
        with pytest.raises(ValueError, match="Unknown method"):
            ts.normality_test(method="invalid")

    def test_multi_column(self):
        """Should return one row per column."""
        ts = TimeSeries(np.random.randn(100, 3), columns=["A", "B", "C"])
        result = ts.normality_test()
        assert result.shape[0] == 3

    def test_conclusion_values(self):
        """Conclusion should be 'Normal' or 'Non-normal'."""
        ts = TimeSeries(np.random.randn(100))
        result = ts.normality_test()
        assert result.loc["Series1", "conclusion"] in ["Normal", "Non-normal"]


class TestEmpiricalCDF:
    """Tests for empirical_cdf() method."""

    def test_returns_figure_axes(self):
        """Should return (Figure, Axes)."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.empirical_cdf()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_single_column(self):
        """Single-column data should produce one step line."""
        ts = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        fig, ax = ts.empirical_cdf()
        assert len(ax.get_lines()) >= 1
        plt.close(fig)

    def test_multi_column_overlaid(self):
        """Multi-column should overlay all columns."""
        ts = TimeSeries(np.random.randn(50, 3), columns=["A", "B", "C"])
        fig, ax = ts.empirical_cdf()
        assert len(ax.get_lines()) >= 3, "Should have at least 3 step lines"
        plt.close(fig)

    def test_column_parameter(self):
        """Should plot only the specified column."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["X", "Y"])
        fig, ax = ts.empirical_cdf(column="X")
        assert len(ax.get_lines()) == 1
        plt.close(fig)


class TestFitDistributions:
    """Tests for fit_distributions() method."""

    def test_returns_expected_columns(self):
        """Result should have distribution name, parameters, and KS test results."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.fit_distributions(method="mle")
        expected = ["best_distribution", "loc", "scale", "ks_statistic", "ks_p_value"]
        assert set(expected).issubset(
            set(result.columns)
        ), f"Missing columns: {set(expected) - set(result.columns)}"

    def test_normal_data_fits_normal(self):
        """Normal data should be best fit by Normal distribution."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        result = ts.fit_distributions(method="mle")
        assert (
            result.loc["Series1", "best_distribution"] == "Normal"
        ), f"Expected Normal, got {result.loc['Series1', 'best_distribution']}"

    def test_ks_pvalue_reasonable(self):
        """KS p-value for well-fitted data should be > 0.05."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(200))
        result = ts.fit_distributions(method="mle")
        assert result.loc["Series1", "ks_p_value"] > 0.05

    def test_multi_column(self):
        """Should return one row per column."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100, 2), columns=["A", "B"])
        result = ts.fit_distributions(method="mle")
        assert result.shape[0] == 2

    def test_returns_dataframe(self):
        """Return type should be DataFrame."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.fit_distributions(method="mle")
        assert isinstance(result, DataFrame)
