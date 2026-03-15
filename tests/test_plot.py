"""Comprehensive tests for statista.plot.Plot static methods."""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statista.plot import Plot


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to prevent memory accumulation."""
    yield
    plt.close("all")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_data():
    """Return a reproducible normal-ish dataset (100 points)."""
    np.random.seed(42)
    return np.random.normal(loc=10, scale=2, size=100)


@pytest.fixture()
def pdf_inputs(seeded_data):
    """Pre-compute arrays needed by Plot.pdf."""
    data_sorted = np.sort(seeded_data)
    qx = np.linspace(data_sorted.min(), data_sorted.max(), 200)
    # simple gaussian pdf for plotting purposes
    pdf_fitted = (1 / (2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((qx - 10) / 2) ** 2)
    return qx, pdf_fitted, data_sorted


@pytest.fixture()
def cdf_inputs(seeded_data):
    """Pre-compute arrays needed by Plot.cdf."""
    data_sorted = np.sort(seeded_data)
    n = len(data_sorted)
    cdf_weibul = np.arange(1, n + 1) / (n + 1)
    qx = np.linspace(data_sorted.min(), data_sorted.max(), 200)
    # monotonic curve between 0 and 1 for a plausible fitted CDF
    cdf_fitted = (qx - qx.min()) / (qx.max() - qx.min())
    return qx, cdf_fitted, data_sorted, cdf_weibul


@pytest.fixture()
def details_inputs(seeded_data):
    """Pre-compute arrays needed by Plot.details."""
    data_sorted = np.sort(seeded_data)
    n = len(data_sorted)
    cdf_empirical = np.arange(1, n + 1) / (n + 1)
    qx = np.linspace(data_sorted.min(), data_sorted.max(), 200)
    pdf_vals = (1 / (2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((qx - 10) / 2) ** 2)
    cdf_fitted = (qx - qx.min()) / (qx.max() - qx.min())
    return qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical


@pytest.fixture()
def confidence_inputs(seeded_data):
    """Pre-compute arrays needed by Plot.confidence_level."""
    data_sorted = np.sort(seeded_data)
    n = len(data_sorted)
    qth = np.linspace(data_sorted.min(), data_sorted.max(), n)
    q_lower = qth - 1.0
    q_upper = qth + 1.0
    return qth, data_sorted, q_lower, q_upper


# ===================================================================
# Plot.pdf
# ===================================================================


class TestPlotPdf:
    """Tests for Plot.pdf static method."""

    def test_returns_figure_and_axes(self, pdf_inputs):
        """Plot.pdf must return (Figure, Axes)."""
        qx, pdf_fitted, data_sorted = pdf_inputs
        result = Plot.pdf(qx, pdf_fitted, data_sorted)
        fig, ax = result
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_default_labels(self, pdf_inputs):
        """Default xlabel='Actual data', ylabel='pdf'."""
        qx, pdf_fitted, data_sorted = pdf_inputs
        _, ax = Plot.pdf(qx, pdf_fitted, data_sorted)
        assert ax.get_xlabel() == "Actual data"
        assert ax.get_ylabel() == "pdf"

    def test_custom_labels(self, pdf_inputs):
        """Custom xlabel / ylabel are applied."""
        qx, pdf_fitted, data_sorted = pdf_inputs
        _, ax = Plot.pdf(
            qx, pdf_fitted, data_sorted, xlabel="Discharge", ylabel="density"
        )
        assert ax.get_xlabel() == "Discharge"
        assert ax.get_ylabel() == "density"

    def test_custom_fig_size(self, pdf_inputs):
        """Custom fig_size is reflected in the Figure."""
        qx, pdf_fitted, data_sorted = pdf_inputs
        fig, _ = Plot.pdf(qx, pdf_fitted, data_sorted, fig_size=(12, 8))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12)
        assert h == pytest.approx(8)

    def test_custom_fontsize(self, pdf_inputs):
        """Fontsize parameter is applied to axis labels."""
        qx, pdf_fitted, data_sorted = pdf_inputs
        _, ax = Plot.pdf(qx, pdf_fitted, data_sorted, fontsize=18)
        assert ax.xaxis.label.get_fontsize() == 18
        assert ax.yaxis.label.get_fontsize() == 18

    def test_contains_line_and_histogram(self, pdf_inputs):
        """The axes should contain at least one Line2D (PDF curve) and histogram patches."""
        qx, pdf_fitted, data_sorted = pdf_inputs
        _, ax = Plot.pdf(qx, pdf_fitted, data_sorted)
        # Line2D from ax.plot
        assert len(ax.lines) >= 1
        # Histogram produces patches
        assert len(ax.patches) >= 1

    def test_small_data(self):
        """Plot.pdf handles a very small array without error."""
        np.random.seed(42)
        small = np.array([1.0, 2.0, 3.0])
        qx = np.linspace(0.5, 3.5, 20)
        pdf_vals = np.ones_like(qx) * 0.1
        fig, ax = Plot.pdf(qx, pdf_vals, small)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_pdf_fitted_as_list(self, pdf_inputs):
        """pdf_fitted may be a plain list (Union[np.ndarray, list])."""
        qx, pdf_fitted, data_sorted = pdf_inputs
        fig, ax = Plot.pdf(qx, pdf_fitted.tolist(), data_sorted)
        assert isinstance(fig, Figure)


# ===================================================================
# Plot.cdf
# ===================================================================


class TestPlotCdf:
    """Tests for Plot.cdf static method."""

    def test_returns_figure_and_axes(self, cdf_inputs):
        """Plot.cdf must return (Figure, Axes)."""
        qx, cdf_fitted, data_sorted, cdf_weibul = cdf_inputs
        fig, ax = Plot.cdf(qx, cdf_fitted, data_sorted, cdf_weibul)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_default_labels(self, cdf_inputs):
        """Default xlabel='Actual data', ylabel='cdf'."""
        qx, cdf_fitted, data_sorted, cdf_weibul = cdf_inputs
        _, ax = Plot.cdf(qx, cdf_fitted, data_sorted, cdf_weibul)
        assert ax.get_xlabel() == "Actual data"
        assert ax.get_ylabel() == "cdf"

    def test_custom_labels(self, cdf_inputs):
        """Custom xlabel / ylabel are applied."""
        qx, cdf_fitted, data_sorted, cdf_weibul = cdf_inputs
        _, ax = Plot.cdf(
            qx,
            cdf_fitted,
            data_sorted,
            cdf_weibul,
            xlabel="Flow",
            ylabel="probability",
        )
        assert ax.get_xlabel() == "Flow"
        assert ax.get_ylabel() == "probability"

    def test_custom_fig_size(self, cdf_inputs):
        """Custom fig_size is reflected in the Figure."""
        qx, cdf_fitted, data_sorted, cdf_weibul = cdf_inputs
        fig, _ = Plot.cdf(qx, cdf_fitted, data_sorted, cdf_weibul, fig_size=(14, 7))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(14)
        assert h == pytest.approx(7)

    def test_custom_fontsize(self, cdf_inputs):
        """Fontsize parameter is applied to axis labels."""
        qx, cdf_fitted, data_sorted, cdf_weibul = cdf_inputs
        _, ax = Plot.cdf(qx, cdf_fitted, data_sorted, cdf_weibul, fontsize=20)
        assert ax.xaxis.label.get_fontsize() == 20
        assert ax.yaxis.label.get_fontsize() == 20

    def test_contains_line_and_scatter(self, cdf_inputs):
        """Axes should have a line (fitted CDF) and a scatter collection (empirical)."""
        qx, cdf_fitted, data_sorted, cdf_weibul = cdf_inputs
        _, ax = Plot.cdf(qx, cdf_fitted, data_sorted, cdf_weibul)
        assert len(ax.lines) >= 1
        # scatter produces a PathCollection
        assert len(ax.collections) >= 1

    def test_legend_present(self, cdf_inputs):
        """A legend with 'Estimated CDF' and 'Empirical CDF' labels must exist."""
        qx, cdf_fitted, data_sorted, cdf_weibul = cdf_inputs
        _, ax = Plot.cdf(qx, cdf_fitted, data_sorted, cdf_weibul)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Estimated CDF" in legend_texts
        assert "Empirical CDF" in legend_texts

    def test_small_data(self):
        """Plot.cdf handles a very small array without error."""
        np.random.seed(42)
        small = np.array([1.0, 2.0, 3.0])
        cdf_w = np.array([0.25, 0.5, 0.75])
        qx = np.linspace(0.5, 3.5, 20)
        cdf_fitted = np.linspace(0, 1, 20)
        fig, ax = Plot.cdf(qx, cdf_fitted, small, cdf_w)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


# ===================================================================
# Plot.details
# ===================================================================


class TestPlotDetails:
    """Tests for Plot.details static method."""

    def test_returns_figure_and_two_axes(self, details_inputs):
        """Plot.details must return (Figure, (Axes, Axes))."""
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        fig, (ax1, ax2) = Plot.details(
            qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax1, Axes)
        assert isinstance(ax2, Axes)

    def test_two_subplots_exist(self, details_inputs):
        """The figure should contain exactly two axes (side-by-side)."""
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        fig, _ = Plot.details(qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical)
        assert len(fig.axes) == 2

    def test_default_labels(self, details_inputs):
        """ax1 ylabel='pdf', ax2 ylabel='cdf', both xlabels='Actual data'."""
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        _, (ax1, ax2) = Plot.details(
            qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical
        )
        assert ax1.get_xlabel() == "Actual data"
        assert ax1.get_ylabel() == "pdf"
        assert ax2.get_xlabel() == "Actual data"
        assert ax2.get_ylabel() == "cdf"

    def test_custom_labels(self, details_inputs):
        """Custom xlabel / ylabel are applied to both subplots."""
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        _, (ax1, ax2) = Plot.details(
            qx,
            data_sorted,
            pdf_vals,
            cdf_fitted,
            cdf_empirical,
            xlabel="Q (m3/s)",
            ylabel="cumulative",
        )
        assert ax1.get_xlabel() == "Q (m3/s)"
        assert ax2.get_xlabel() == "Q (m3/s)"
        # ylabel param only controls the CDF subplot; PDF subplot is hardcoded to "pdf"
        assert ax1.get_ylabel() == "pdf"
        assert ax2.get_ylabel() == "cumulative"

    def test_custom_fig_size(self, details_inputs):
        """Custom fig_size is reflected in the Figure."""
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        fig, _ = Plot.details(
            qx,
            data_sorted,
            pdf_vals,
            cdf_fitted,
            cdf_empirical,
            fig_size=(16, 9),
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(16)
        assert h == pytest.approx(9)

    def test_custom_fontsize_on_ax1(self, details_inputs):
        """Fontsize is applied to ax1 labels."""
        fontsize = 14
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        _, (ax1, _) = Plot.details(
            qx,
            data_sorted,
            pdf_vals,
            cdf_fitted,
            cdf_empirical,
            fontsize=fontsize,
        )
        assert ax1.xaxis.label.get_fontsize() == fontsize
        assert ax1.yaxis.label.get_fontsize() == fontsize

    def test_custom_fontsize_on_ax2_xlabel(self, details_inputs):
        """Fontsize is applied to ax2 xlabel (ax2 ylabel uses hardcoded 15)."""
        fontsize = 14
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        _, (_, ax2) = Plot.details(
            qx,
            data_sorted,
            pdf_vals,
            cdf_fitted,
            cdf_empirical,
            fontsize=fontsize,
        )
        assert ax2.xaxis.label.get_fontsize() == fontsize
        assert ax2.yaxis.label.get_fontsize() == fontsize

    def test_pdf_subplot_has_line_and_histogram(self, details_inputs):
        """ax1 (PDF side) must contain a Line2D and histogram patches."""
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        _, (ax1, _) = Plot.details(qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical)
        assert len(ax1.lines) >= 1
        assert len(ax1.patches) >= 1

    def test_cdf_subplot_has_line_and_scatter(self, details_inputs):
        """ax2 (CDF side) must contain a Line2D and scatter collection."""
        qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical = details_inputs
        _, (_, ax2) = Plot.details(qx, data_sorted, pdf_vals, cdf_fitted, cdf_empirical)
        assert len(ax2.lines) >= 1
        assert len(ax2.collections) >= 1

    def test_input_not_mutated(self):
        """Plot.details should not mutate the input q_act array."""
        np.random.seed(42)
        q_act = np.random.normal(loc=10, scale=2, size=50)
        original_order = q_act.copy()

        n = len(q_act)
        cdf_empirical = np.arange(1, n + 1) / (n + 1)
        qx = np.linspace(q_act.min(), q_act.max(), 200)
        pdf_vals = np.ones_like(qx) * 0.05
        cdf_fitted = np.linspace(0, 1, 200)

        Plot.details(qx, q_act, pdf_vals, cdf_fitted, cdf_empirical)

        np.testing.assert_array_equal(q_act, original_order)

    def test_small_data(self):
        """Plot.details handles a very small array without error."""
        np.random.seed(42)
        small = np.array([1.0, 2.0, 3.0])
        cdf_emp = np.array([0.25, 0.5, 0.75])
        qx = np.linspace(0.5, 3.5, 20)
        pdf_vals = np.ones(20) * 0.1
        cdf_fitted = np.linspace(0, 1, 20)
        fig, (ax1, ax2) = Plot.details(qx, small, pdf_vals, cdf_fitted, cdf_emp)
        assert isinstance(fig, Figure)
        assert isinstance(ax1, Axes)
        assert isinstance(ax2, Axes)


# ===================================================================
# Plot.confidence_level
# ===================================================================


class TestPlotConfidenceLevel:
    """Tests for Plot.confidence_level static method."""

    def test_returns_figure_and_axes(self, confidence_inputs):
        """Plot.confidence_level must return (Figure, Axes)."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        fig, ax = Plot.confidence_level(qth, q_act.copy(), q_lower, q_upper)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_default_labels(self, confidence_inputs):
        """Default xlabel='Theoretical Values', ylabel='Actual Values'."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        _, ax = Plot.confidence_level(qth, q_act.copy(), q_lower, q_upper)
        assert ax.get_xlabel() == "Theoretical Values"
        assert ax.get_ylabel() == "Actual Values"

    def test_default_fig_size(self, confidence_inputs):
        """Default fig_size is (6, 6)."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        fig, _ = Plot.confidence_level(qth, q_act.copy(), q_lower, q_upper)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(6)
        assert h == pytest.approx(6)

    def test_custom_fig_size(self, confidence_inputs):
        """Custom fig_size is applied."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        fig, _ = Plot.confidence_level(
            qth, q_act.copy(), q_lower, q_upper, fig_size=(10, 8)
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(10)
        assert h == pytest.approx(8)

    def test_custom_fontsize(self, confidence_inputs):
        """Fontsize parameter is applied to axis labels."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        _, ax = Plot.confidence_level(qth, q_act.copy(), q_lower, q_upper, fontsize=16)
        assert ax.xaxis.label.get_fontsize() == 16
        assert ax.yaxis.label.get_fontsize() == 16

    def test_default_alpha_legend_text(self, confidence_inputs):
        """When alpha is None, default 0.05 is used -> '95 % CI' in legend."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        _, ax = Plot.confidence_level(qth, q_act.copy(), q_lower, q_upper)
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert any("95 % CI" in t for t in texts)

    def test_custom_alpha_legend_text(self, confidence_inputs):
        """alpha=0.1 should produce '90 % CI' in the legend labels."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        _, ax = Plot.confidence_level(qth, q_act.copy(), q_lower, q_upper, alpha=0.1)
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        assert any("90 % CI" in t for t in texts)

    def test_legend_labels_content(self, confidence_inputs):
        """Legend must contain Theoretical Data, Lower limit, Upper limit, Actual Data."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        _, ax = Plot.confidence_level(qth, q_act.copy(), q_lower, q_upper, alpha=0.05)
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        assert "Theoretical Data" in texts
        assert "Actual Data" in texts
        assert any("Lower limit" in t for t in texts)
        assert any("Upper limit" in t for t in texts)

    def test_plot_elements_count(self, confidence_inputs):
        """Should contain 3 lines (theoretical, lower, upper) and 1 scatter collection."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        _, ax = Plot.confidence_level(qth, q_act.copy(), q_lower, q_upper)
        # 3 ax.plot calls -> 3 Line2D objects
        assert len(ax.lines) == 3
        # 1 ax.scatter call -> 1 PathCollection
        assert len(ax.collections) == 1

    def test_custom_marker_size(self, confidence_inputs):
        """marker_size parameter should affect the marker size of bound lines."""
        qth, q_act, q_lower, q_upper = confidence_inputs
        _, ax = Plot.confidence_level(
            qth, q_act.copy(), q_lower, q_upper, marker_size=20
        )
        # Lines at index 1 and 2 are the lower/upper bound lines with markers
        lower_line = ax.lines[1]
        upper_line = ax.lines[2]
        assert lower_line.get_markersize() == 20
        assert upper_line.get_markersize() == 20

    def test_input_not_mutated(self):
        """Plot.confidence_level should not mutate the input q_act array."""
        np.random.seed(42)
        q_act = np.random.normal(loc=10, scale=2, size=30)
        original_order = q_act.copy()
        n = len(q_act)
        qth = np.linspace(q_act.min(), q_act.max(), n)
        q_lower = qth - 0.5
        q_upper = qth + 0.5

        Plot.confidence_level(qth, q_act, q_lower, q_upper)

        np.testing.assert_array_equal(q_act, original_order)

    def test_small_data(self):
        """Plot.confidence_level handles a very small array without error."""
        np.random.seed(42)
        small = np.array([3.0, 1.0, 2.0])
        qth = np.array([1.0, 2.0, 3.0])
        q_lower = qth - 0.5
        q_upper = qth + 0.5
        fig, ax = Plot.confidence_level(qth, small, q_lower, q_upper)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_list_inputs(self):
        """confidence_level should accept plain lists (Union[np.ndarray, list])."""
        qth = [1.0, 2.0, 3.0, 4.0, 5.0]
        q_act = [1.1, 2.3, 2.8, 4.2, 5.1]
        q_lower = [0.5, 1.5, 2.5, 3.5, 4.5]
        q_upper = [1.5, 2.5, 3.5, 4.5, 5.5]
        fig, ax = Plot.confidence_level(qth, q_act, q_lower, q_upper)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
