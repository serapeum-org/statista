import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statista.time_series import TimeSeries


@pytest.fixture
def sample_data_1d():
    return np.random.randn(100)


@pytest.fixture
def sample_data_2d():
    return np.random.randn(100, 3)


@pytest.fixture
def ts_1d(sample_data_1d) -> TimeSeries:
    return TimeSeries(sample_data_1d)


@pytest.fixture
def ts_2d(sample_data_2d) -> TimeSeries:
    return TimeSeries(sample_data_2d, columns=["A", "B", "C"])


@pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
def test_stats(ts: TimeSeries, request):
    """Test the stats method."""
    ts = request.getfixturevalue(ts)
    stats = ts.stats
    assert stats.index.to_list() == [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]


EXTENDED_STATS_INDEX = [
    "count", "mean", "std", "cv", "skewness", "kurtosis",
    "min", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "max",
    "iqr", "mad",
]


class TestExtendedStats:
    """Tests for the extended_stats property."""

    @pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
    def test_extended_stats_rows(self, ts: str, request):
        """extended_stats should return all expected statistic rows."""
        ts = request.getfixturevalue(ts)
        result = ts.extended_stats
        assert result.index.tolist() == EXTENDED_STATS_INDEX

    @pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
    def test_extended_stats_columns_match(self, ts: str, request):
        """Column names in extended_stats should match the TimeSeries columns."""
        ts = request.getfixturevalue(ts)
        result = ts.extended_stats
        assert result.columns.tolist() == ts.columns.tolist()

    def test_extended_stats_known_values(self):
        """Verify computed statistics against known values for a fixed dataset."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ts = TimeSeries(data)
        result = ts.extended_stats

        assert result.loc["count", "Series1"] == 10.0
        assert result.loc["mean", "Series1"] == pytest.approx(5.5)
        assert result.loc["min", "Series1"] == 1.0
        assert result.loc["max", "Series1"] == 10.0
        assert result.loc["50%", "Series1"] == pytest.approx(5.5)
        # std with ddof=1
        assert result.loc["std", "Series1"] == pytest.approx(np.std(data, ddof=1))
        # CV = std / mean
        expected_cv = np.std(data, ddof=1) / np.mean(data)
        assert result.loc["cv", "Series1"] == pytest.approx(expected_cv)
        # IQR = Q3 - Q1
        q25, q75 = np.percentile(data, [25, 75])
        assert result.loc["iqr", "Series1"] == pytest.approx(q75 - q25)
        # MAD
        from scipy.stats import median_abs_deviation
        assert result.loc["mad", "Series1"] == pytest.approx(median_abs_deviation(data))

    def test_extended_stats_cv_near_zero_mean(self):
        """CV should be NaN when the mean is close to zero."""
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        ts = TimeSeries(data)
        result = ts.extended_stats
        assert np.isnan(result.loc["cv", "Series1"])

    def test_extended_stats_skewness_symmetric(self):
        """Skewness should be close to zero for symmetric data."""
        data = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        ts = TimeSeries(data)
        result = ts.extended_stats
        assert result.loc["skewness", "Series1"] == pytest.approx(0.0, abs=1e-10)

    def test_extended_stats_handles_nan(self):
        """NaN values in the data should be excluded from calculations."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ts = TimeSeries(data)
        result = ts.extended_stats
        assert result.loc["count", "Series1"] == 4.0
        assert result.loc["mean", "Series1"] == pytest.approx(3.0)


class TestBoxPlot:

    @pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
    def test_plot_box(self, ts: TimeSeries, request):
        """Test the plot_box method."""
        ts = request.getfixturevalue(ts)
        fig, ax = ts.box_plot()
        assert isinstance(
            fig, plt.Figure
        ), "plot_box should return a matplotlib Figure."
        assert isinstance(ax, plt.Axes), "plot_box should return a matplotlib Axes."

        fig, ax = plt.subplots()
        fig2, ax2 = ts.box_plot(fig=fig, ax=ax)
        assert fig2 is fig, "If fig is provided, plot_box should use it."
        assert ax2 is ax, "If ax is provided, plot_box should use it."
        if ts.shape[1] > 1:
            assert len(ax.get_xticklabels()) == 3

    def test_calculate_wiskers(self):
        data = list(range(100))
        # ts = TimeSeries(data)
        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
        whiskers = TimeSeries.calculate_whiskers(data, quartile1, quartile3)
        assert isinstance(whiskers, tuple)
        assert whiskers[0] == 0
        assert whiskers[1] == 99


class TestViolin:

    @pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
    def test_violin(self, ts: TimeSeries, request):
        """Test the plot_box method."""
        ts = request.getfixturevalue(ts)
        fig, ax = ts.violin()
        assert isinstance(
            fig, plt.Figure
        ), "plot_box should return a matplotlib Figure."
        assert isinstance(ax, plt.Axes), "plot_box should return a matplotlib Axes."

        fig, ax = plt.subplots()
        fig2, ax2 = ts.violin(fig=fig, ax=ax)
        assert fig2 is fig, "If fig is provided, plot_box should use it."
        assert ax2 is ax, "If ax is provided, plot_box should use it."
        if ts.shape[1] > 1:
            assert len(ax.get_xticklabels()) == 3


class TestRainCloud:

    @pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
    def test_raincloud(self, ts: TimeSeries, request):
        """Test the plot_box method."""
        ts = request.getfixturevalue(ts)
        fig, ax = ts.raincloud()
        assert isinstance(
            fig, plt.Figure
        ), "plot_box should return a matplotlib Figure."
        assert isinstance(ax, plt.Axes), "plot_box should return a matplotlib Axes."

        fig, ax = plt.subplots()
        fig2, ax2 = ts.raincloud(fig=fig, ax=ax)
        assert fig2 is fig, "If fig is provided, plot_box should use it."
        assert ax2 is ax, "If ax is provided, plot_box should use it."
        if ts.shape[1] > 1:
            assert len(ax.get_xticklabels()) == 3


class TestHistogram:

    def test_default(self):
        # Test with default parameters
        ts = TimeSeries(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
        n_values, bin_edges, fig, ax = ts.histogram()
        assert ax.get_title() == ""
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        plt.close()

    def test_default_2d(self):
        # Test with default parameters
        arr = np.random.randn(100, 4)
        ts = TimeSeries(arr)
        n_values, bin_edges, fig, ax = ts.histogram()
        assert ax.get_title() == ""
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        plt.close()

    def test_custom_labels(self):
        # Test with custom title and labels
        ts = TimeSeries(np.array([1, 2, 3, 4, 5]))
        n_values, bin_edges, fig, ax = ts.histogram(
            title="Custom Title", xlabel="Custom X", ylabel="Custom Y"
        )

        assert ax.get_title() == "Custom Title"
        assert ax.get_xlabel() == "Custom X"
        assert ax.get_ylabel() == "Custom Y"
        plt.close()

    def test_custom_colors(self):
        # Test with custom colors
        ts = TimeSeries(np.array([1, 2, 3, 4, 5]))
        n_values, bin_edges, fig, ax = ts.histogram(
            color=dict(face="green", edge="red", alpha=0.5)
        )

        patches = ax.patches
        assert patches[0].get_facecolor() == (
            0.0,
            0.5019607843137255,
            0.0,
            0.5,
        )  # RGBA for green with alpha 0.5
        assert patches[0].get_edgecolor() == (1.0, 0.0, 0.0, 0.5)  # RGBA for red
        plt.close()

    def test_legend(self):
        # Test with a legend
        ts = TimeSeries(np.array([1, 2, 3, 4, 5]))
        # default legend
        n_values, bin_edges, fig, ax = ts.histogram()
        legend = ax.get_legend()
        assert legend.get_texts()[0].get_text() == "Series1"
        # custom legend
        n_values, bin_edges, fig, ax = ts.histogram(legend=["Sample Legend"])

        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_texts()[0].get_text() == "Sample Legend"
        plt.close()
        # 2D data
        data_2d = np.random.randn(100, 4)
        cols = ["A", "B", "C", "D"]
        ts_2d = TimeSeries(data_2d, columns=cols)
        n_values, bin_edges, fig, ax = ts_2d.histogram(legend=cols)
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [legend.get_texts()[i].get_text() for i in range(len(cols))]
        assert legend_labels == cols

        plt.close()

    def test_bins(self):
        # Test with different number of bins
        ts = TimeSeries(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        n_values, bin_edges, fig, ax = ts.histogram(bins=5)

        # Number of bars should match the number of bins
        assert len(ax.patches) == 5
        plt.close()

    def test_grid_and_ticks(self):
        # Test grid and tick customization
        ts = TimeSeries(np.array([1, 2, 3, 4, 5]))
        n_values, bin_edges, fig, ax = ts.histogram(tick_fontsize=16)

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            assert tick.get_fontsize() == 16  # Check the fontsize of the ticks
        plt.close()


class TestRollingStatistics:

    @pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
    def test_plot_rolling_statistics(self, ts: TimeSeries, request):
        """Test the plot_rolling_statistics method."""
        ts = request.getfixturevalue(ts)
        fig, ax = ts.rolling_statistics()
        assert isinstance(
            fig, plt.Figure
        ), "plot_rolling_statistics should return a matplotlib Figure."
        assert isinstance(
            ax, plt.Axes
        ), "plot_rolling_statistics should return a matplotlib Axes."

        fig, ax = plt.subplots()
        fig2, ax2 = ts.rolling_statistics(fig=fig, ax=ax)
        assert fig2 is fig, "If fig is provided, plot_rolling_statistics should use it."
        assert ax2 is ax, "If ax is provided, plot_rolling_statistics should use it."


class TestDensity:

    @pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
    def test_plot_density(self, ts: TimeSeries, request):
        """Test the plot_density method."""
        ts = request.getfixturevalue(ts)
        fig, ax = ts.density()
        assert isinstance(
            fig, plt.Figure
        ), "plot_density should return a matplotlib Figure."
        assert isinstance(ax, plt.Axes), "plot_density should return a matplotlib Axes."

        fig, ax = plt.subplots()
        fig2, ax2 = ts.density(fig=fig, ax=ax)
        assert fig2 is fig, "If fig is provided, plot_density should use it."
        assert ax2 is ax, "If ax is provided, plot_density should use it."
