import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas import DataFrame

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


class TestTimeSeriesInit:
    """Tests for TimeSeries constructor (__init__) and _constructor property."""

    def test_1d_array_creates_single_column(self):
        """A 1D numpy array should be reshaped into a single-column DataFrame."""
        data = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries(data)
        assert ts.shape == (3, 1), f"Expected shape (3, 1), got {ts.shape}"
        assert ts.columns.tolist() == [
            "Series1"
        ], f"Expected ['Series1'], got {ts.columns.tolist()}"

    def test_2d_array_preserves_shape(self):
        """A 2D numpy array should keep its shape with auto-generated column names."""
        data = np.random.randn(10, 3)
        ts = TimeSeries(data)
        assert ts.shape == (10, 3), f"Expected shape (10, 3), got {ts.shape}"
        assert ts.columns.tolist() == ["Series1", "Series2", "Series3"]

    def test_custom_columns(self):
        """User-provided column names should override auto-generated ones."""
        data = np.random.randn(5, 2)
        ts = TimeSeries(data, columns=["X", "Y"])
        assert ts.columns.tolist() == [
            "X",
            "Y",
        ], f"Expected ['X', 'Y'], got {ts.columns.tolist()}"

    def test_from_list(self):
        """A plain Python list should be accepted and converted to single-column."""
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0])
        assert ts.shape == (4, 1), f"Expected shape (4, 1), got {ts.shape}"
        assert ts.columns.tolist() == ["Series1"]

    def test_from_dataframe_preserves_columns(self):
        """DataFrame column names should be preserved when columns=None."""
        df = DataFrame({"flow": [1, 2, 3], "temp": [4, 5, 6]})
        ts = TimeSeries(df)
        assert ts.columns.tolist() == [
            "flow",
            "temp",
        ], f"Expected ['flow', 'temp'], got {ts.columns.tolist()}"

    def test_custom_index(self):
        """User-provided index should be preserved."""
        data = np.array([10.0, 20.0, 30.0])
        idx = [100, 200, 300]
        ts = TimeSeries(data, index=idx)
        assert ts.index.tolist() == idx, f"Expected {idx}, got {ts.index.tolist()}"

    def test_from_dataframe(self):
        """Passing a pandas DataFrame should work without re-wrapping."""
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        ts = TimeSeries(df, columns=["A", "B"])
        assert ts.shape == (2, 2), f"Expected (2, 2), got {ts.shape}"
        assert ts["A"].tolist() == [1, 2]

    def test_from_dict(self):
        """Passing a dict should create a TimeSeries with dict keys as columns."""
        data = {"col_a": [1, 2, 3], "col_b": [4, 5, 6]}
        ts = TimeSeries(data)
        assert (
            "col_a" in ts.columns
        ), f"Expected 'col_a' in columns, got {ts.columns.tolist()}"
        assert ts["col_b"].tolist() == [4, 5, 6]

    def test_constructor_returns_timeseries(self, ts_1d):
        """The _constructor property should return the TimeSeries class for pandas method chaining."""
        assert (
            ts_1d._constructor is TimeSeries
        ), f"Expected TimeSeries, got {ts_1d._constructor}"

    def test_pandas_operation_returns_timeseries(self):
        """Pandas operations (e.g., copy) should return TimeSeries, not plain DataFrame."""
        ts = TimeSeries(np.array([1.0, 2.0, 3.0]))
        copied = ts.copy()
        assert isinstance(
            copied, TimeSeries
        ), f"copy() should return TimeSeries, got {type(copied)}"

    def test_empty_array_raises_valueerror(self):
        """Creating TimeSeries from empty array should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot create TimeSeries from empty array"):
            TimeSeries(np.array([]))

    def test_empty_2d_array_raises_valueerror(self):
        """Creating TimeSeries from empty 2D array should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot create TimeSeries from empty array"):
            TimeSeries(np.array([]).reshape(0, 3))

    def test_empty_list_raises_valueerror(self):
        """Creating TimeSeries from empty list should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot create TimeSeries from empty array"):
            TimeSeries([])

    def test_empty_dataframe_raises_valueerror(self):
        """Creating TimeSeries from empty DataFrame should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot create TimeSeries from empty DataFrame"):
            TimeSeries(DataFrame())


class TestGetAxFig:
    """Isolated tests for the _get_ax_fig static method."""

    def test_creates_new_figure_and_axes(self):
        """When no fig/ax provided, should create both from scratch."""
        fig, ax = TimeSeries._get_ax_fig()
        assert isinstance(fig, plt.Figure), f"Expected Figure, got {type(fig)}"
        assert isinstance(ax, plt.Axes), f"Expected Axes, got {type(ax)}"
        plt.close(fig)

    def test_reuses_provided_fig_and_ax(self):
        """When both fig and ax are provided, should return them unchanged."""
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = TimeSeries._get_ax_fig(fig=fig_in, ax=ax_in)
        assert fig_out is fig_in, "Should reuse provided Figure"
        assert ax_out is ax_in, "Should reuse provided Axes"
        plt.close(fig_in)

    def test_reuses_fig_creates_ax(self):
        """When only fig is provided (no ax), should add a subplot to it."""
        fig_in = plt.figure()
        fig_out, ax_out = TimeSeries._get_ax_fig(fig=fig_in)
        assert fig_out is fig_in, "Should reuse provided Figure"
        assert isinstance(ax_out, plt.Axes), f"Should create Axes, got {type(ax_out)}"
        plt.close(fig_in)

    def test_reuses_ax_gets_fig(self):
        """When only ax is provided (no fig), should extract fig from ax.figure."""
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = TimeSeries._get_ax_fig(ax=ax_in)
        assert ax_out is ax_in, "Should reuse provided Axes"
        assert fig_out is fig_in, "Should extract Figure from Axes"
        plt.close(fig_in)

    def test_n_subplots(self):
        """n_subplots > 1 should create multiple axes."""
        fig, axes = TimeSeries._get_ax_fig(n_subplots=3)
        assert len(axes) == 3, f"Expected 3 axes, got {len(axes)}"
        plt.close(fig)


class TestAdjustAxesLabels:
    """Isolated tests for _adjust_axes_labels static method."""

    def test_sets_title_and_labels(self):
        """Title, xlabel, ylabel should be applied to the axes."""
        fig, ax = plt.subplots()
        TimeSeries._adjust_axes_labels(ax, title="T", xlabel="X", ylabel="Y")
        assert ax.get_title() == "T", f"Expected title 'T', got '{ax.get_title()}'"
        assert ax.get_xlabel() == "X", f"Expected xlabel 'X', got '{ax.get_xlabel()}'"
        assert ax.get_ylabel() == "Y", f"Expected ylabel 'Y', got '{ax.get_ylabel()}'"
        plt.close(fig)

    def test_sets_tick_labels(self):
        """Tick labels should be applied when provided."""
        fig, ax = plt.subplots()
        ax.bar([1, 2, 3], [4, 5, 6])
        TimeSeries._adjust_axes_labels(ax, tick_labels=["a", "b", "c"])
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "a" in labels, f"Expected 'a' in tick labels, got {labels}"
        plt.close(fig)

    def test_sets_legend(self):
        """Legend should be created when legend kwarg is provided."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        TimeSeries._adjust_axes_labels(ax, legend=["MyLine"])
        legend = ax.get_legend()
        assert legend is not None, "Legend should be created"
        assert legend.get_texts()[0].get_text() == "MyLine"
        plt.close(fig)

    def test_font_sizes(self):
        """Custom font sizes should be applied."""
        fig, ax = plt.subplots()
        TimeSeries._adjust_axes_labels(
            ax,
            title="T",
            title_fontsize=20,
            xlabel="X",
            xlabel_fontsize=16,
            ylabel="Y",
            ylabel_fontsize=14,
            tick_fontsize=10,
        )
        assert (
            ax.title.get_fontsize() == 20
        ), f"Expected title fontsize 20, got {ax.title.get_fontsize()}"
        plt.close(fig)

    def test_no_kwargs_does_not_crash(self):
        """Calling with no kwargs should not raise."""
        fig, ax = plt.subplots()
        result = TimeSeries._adjust_axes_labels(ax)
        assert result is ax, "Should return the same axes object"
        plt.close(fig)


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
    "count",
    "mean",
    "std",
    "cv",
    "skewness",
    "kurtosis",
    "min",
    "5%",
    "10%",
    "25%",
    "50%",
    "75%",
    "90%",
    "95%",
    "max",
    "iqr",
    "mad",
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


class TestExtendedStatsEdgeCases:
    """Additional edge-case tests for the extended_stats property."""

    def test_negative_mean_cv(self):
        """CV should be computed correctly for data with negative mean."""
        data = np.array([-10.0, -20.0, -30.0, -40.0, -50.0])
        ts = TimeSeries(data)
        result = ts.extended_stats
        expected_cv = np.std(data, ddof=1) / np.mean(data)
        assert result.loc["cv", "Series1"] == pytest.approx(
            expected_cv
        ), f"Expected CV {expected_cv}, got {result.loc['cv', 'Series1']}"

    def test_single_value(self):
        """A single-value series should produce std=0, NaN for CV, and consistent min/max."""
        data = np.array([42.0])
        ts = TimeSeries(data)
        result = ts.extended_stats
        assert result.loc["count", "Series1"] == 1.0
        assert result.loc["mean", "Series1"] == 42.0
        assert result.loc["min", "Series1"] == 42.0
        assert result.loc["max", "Series1"] == 42.0

    def test_kurtosis_heavy_tails(self):
        """Kurtosis should be positive for heavy-tailed data (leptokurtic)."""
        np.random.seed(99)
        data = np.concatenate(
            [np.random.randn(100), np.array([10.0, -10.0, 15.0, -15.0])]
        )
        ts = TimeSeries(data)
        result = ts.extended_stats
        assert (
            result.loc["kurtosis", "Series1"] > 0
        ), f"Expected positive kurtosis for heavy tails, got {result.loc['kurtosis', 'Series1']}"

    def test_stats_values_match_pandas_describe(self):
        """The stats property values should match pandas describe() output."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data)
        stats = ts.stats
        desc = ts.describe()
        assert (
            stats.loc["mean", "Series1"] == desc.loc["mean", "Series1"]
        ), "stats and describe() should produce identical mean"
        assert (
            stats.loc["std", "Series1"] == desc.loc["std", "Series1"]
        ), "stats and describe() should produce identical std"


class TestBoxPlotParameters:
    """Tests for box_plot parameter coverage."""

    def test_mean_marker(self):
        """box_plot(mean=True) should not raise and should return valid axes."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.box_plot(mean=True)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_notch(self):
        """box_plot(notch=True) should not raise."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.box_plot(notch=True)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_color(self):
        """box_plot with color dict should apply the color."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.box_plot(color={"boxes": "#DC143C"})
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_title_xlabel_ylabel(self):
        """box_plot should apply title and axis labels."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.box_plot(title="My Title", xlabel="X", ylabel="Y")
        assert ax.get_title() == "My Title"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        plt.close(fig)


class TestCalculateWhiskersEdgeCases:
    """Additional tests for calculate_whiskers static method."""

    def test_float_data(self):
        """Should work with floating point data."""
        data = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        q1, q3 = np.percentile(data, [25, 75])
        lower, upper = TimeSeries.calculate_whiskers(data, q1, q3)
        assert lower >= data[0], f"Lower whisker {lower} should be >= min {data[0]}"
        assert upper <= data[-1], f"Upper whisker {upper} should be <= max {data[-1]}"

    def test_zero_iqr(self):
        """When Q1 == Q3 (IQR=0), whiskers should equal Q1 and Q3."""
        data = [5, 5, 5, 5, 5]
        lower, upper = TimeSeries.calculate_whiskers(data, 5.0, 5.0)
        assert lower == 5.0, f"Expected lower whisker 5.0, got {lower}"
        assert upper == 5.0, f"Expected upper whisker 5.0, got {upper}"

    def test_negative_data(self):
        """Should handle negative values correctly."""
        data = list(range(-50, 50))
        q1, q3 = np.percentile(data, [25, 75])
        lower, upper = TimeSeries.calculate_whiskers(data, q1, q3)
        assert lower >= data[0], f"Lower whisker {lower} should be >= min {data[0]}"
        assert upper <= data[-1], f"Upper whisker {upper} should be <= max {data[-1]}"


class TestViolinParameters:
    """Tests for violin plot parameter coverage."""

    def test_median_shown(self):
        """violin(median=True) should not raise."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.violin(median=True)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_no_mean_no_extrema(self):
        """violin(mean=False, extrema=False) should not raise."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.violin(mean=False, extrema=False)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    @pytest.mark.parametrize("side", ["low", "high", "both"])
    def test_side_parameter(self, side):
        """All side options ('low', 'high', 'both') should produce valid plots."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.violin(side=side)
        assert isinstance(ax, plt.Axes), f"side='{side}' should produce valid Axes"
        plt.close(fig)

    def test_spacing(self):
        """Violin with spacing > 0 should work for multi-column data."""
        ts = TimeSeries(np.random.randn(50, 3), columns=["A", "B", "C"])
        fig, ax = ts.violin(spacing=2)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_color(self):
        """Custom color dict should be applied to violin bodies."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.violin(color={"face": "red", "edge": "blue", "alpha": 0.5})
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestRaincloudParameters:
    """Tests for raincloud plot parameter coverage."""

    def test_overlay_false(self):
        """raincloud(overlay=False) should separate violin, scatter, box."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.raincloud(overlay=False)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_widths(self):
        """Custom violin_width, scatter_offset, boxplot_width should not raise."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.raincloud(violin_width=0.6, scatter_offset=0.2, boxplot_width=0.15)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_order(self):
        """Custom order should control which elements appear."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.raincloud(order=["box", "scatter"])
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_2d_overlay_false(self):
        """Overlay=False with multi-column data should not raise."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["X", "Y"])
        fig, ax = ts.raincloud(overlay=False)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestRollingStatisticsParameters:
    """Tests for rolling_statistics parameter coverage."""

    def test_custom_window(self):
        """Different window sizes should produce valid plots."""
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.rolling_statistics(window=5)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_labels(self):
        """Title and axis labels should be applied."""
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.rolling_statistics(
            window=10, title="Rolling", xlabel="Time", ylabel="Value"
        )
        assert ax.get_title() == "Rolling"
        assert ax.get_xlabel() == "Time"
        plt.close(fig)


class TestDensityParameters:
    """Tests for density plot parameter coverage."""

    def test_with_labels(self):
        """Title and axis labels should be applied to density plot."""
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.density(title="KDE", xlabel="Val", ylabel="Density")
        assert ax.get_title() == "KDE"
        assert ax.get_xlabel() == "Val"
        plt.close(fig)

    def test_custom_color(self):
        """Custom color kwarg should not raise."""
        ts = TimeSeries(np.random.randn(100))
        fig, ax = ts.density(color="red")
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestHistogramParameters:
    """Additional tests for histogram parameter coverage."""

    def test_custom_color_list_2d(self):
        """2D data with a list of face colors should not trigger the 'Multiple columns' warning."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plt.subplots()
            n, edges, fig2, ax2 = ts.histogram(
                fig=fig,
                ax=ax,
                color={"face": ["red", "blue"], "edge": "black", "alpha": 0.7},
            )
        multi_col_warnings = [x for x in w if "Multiple columns" in str(x.message)]
        assert (
            len(multi_col_warnings) == 0
        ), "Should not warn about multiple columns when a color list is provided"
        assert isinstance(ax2, plt.Axes)
        plt.close(fig)


class TestLMoments:
    """Tests for the l_moments() method."""

    def test_returns_expected_rows_default(self):
        """Default nmom=5 should return rows L1, L2, t, t3, t4, t5."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.l_moments()
        expected_rows = ["L1", "L2", "t", "t3", "t4", "t5"]
        assert (
            result.index.tolist() == expected_rows
        ), f"Expected rows {expected_rows}, got {result.index.tolist()}"

    def test_returns_expected_rows_nmom4(self):
        """nmom=4 should return rows L1, L2, t, t3, t4 (no t5)."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.l_moments(nmom=4)
        expected_rows = ["L1", "L2", "t", "t3", "t4"]
        assert result.index.tolist() == expected_rows

    def test_returns_expected_rows_nmom2(self):
        """nmom=2 should return rows L1, L2, t only."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.l_moments(nmom=2)
        expected_rows = ["L1", "L2", "t"]
        assert result.index.tolist() == expected_rows

    def test_columns_match(self):
        """Columns in result should match the TimeSeries columns."""
        ts = TimeSeries(np.random.randn(50, 3), columns=["A", "B", "C"])
        result = ts.l_moments()
        assert result.columns.tolist() == ["A", "B", "C"]

    def test_l1_equals_mean(self):
        """L1 (first L-moment) should equal the sample mean."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ts = TimeSeries(data)
        result = ts.l_moments(nmom=4)
        assert result.loc["L1", "Series1"] == pytest.approx(
            5.5
        ), f"L1 should equal mean 5.5, got {result.loc['L1', 'Series1']}"

    def test_symmetric_data_t3_near_zero(self):
        """L-skewness (t3) should be near zero for symmetric data."""
        data = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data)
        result = ts.l_moments(nmom=4)
        assert result.loc["t3", "Series1"] == pytest.approx(
            0.0, abs=0.05
        ), f"L-skewness should be near 0 for symmetric data, got {result.loc['t3', 'Series1']}"

    def test_l2_positive(self):
        """L2 (L-scale) should be positive for non-constant data."""
        np.random.seed(99)
        ts = TimeSeries(np.random.randn(100))
        result = ts.l_moments(nmom=4)
        assert (
            result.loc["L2", "Series1"] > 0
        ), f"L2 should be positive, got {result.loc['L2', 'Series1']}"

    def test_handles_nan(self):
        """NaN values should be dropped before computing L-moments."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        ts = TimeSeries(data)
        result = ts.l_moments(nmom=4)
        assert not np.isnan(result.loc["L1", "Series1"]), "L1 should not be NaN"

    def test_nmom_less_than_2_raises(self):
        """nmom < 2 should raise ValueError."""
        ts = TimeSeries(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="nmom must be >= 2"):
            ts.l_moments(nmom=1)


class TestSummary:
    """Tests for the summary() method."""

    def test_returns_dataframe(self):
        """summary() should return a pandas DataFrame."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.summary()
        assert isinstance(result, DataFrame), f"Expected DataFrame, got {type(result)}"

    def test_contains_all_expected_rows(self):
        """summary() should contain both extended_stats rows and L-moment ratios."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        result = ts.summary()
        expected_rows = [
            "count",
            "mean",
            "std",
            "cv",
            "skewness",
            "kurtosis",
            "min",
            "max",
            "iqr",
            "mad",
            "L-CV",
            "L-skewness",
            "L-kurtosis",
        ]
        assert (
            result.index.tolist() == expected_rows
        ), f"Expected rows {expected_rows}, got {result.index.tolist()}"

    def test_columns_match(self):
        """Columns in summary should match the TimeSeries columns."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["X", "Y"])
        result = ts.summary()
        assert result.columns.tolist() == ["X", "Y"]

    def test_values_consistent_with_extended_stats(self):
        """Values for shared rows should match extended_stats output."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        summary = ts.summary()
        estats = ts.extended_stats
        for row in ["mean", "std", "cv", "skewness", "kurtosis"]:
            assert summary.loc[row, "Series1"] == pytest.approx(
                float(estats.loc[row, "Series1"])
            ), f"Row '{row}' mismatch between summary and extended_stats"

    def test_values_consistent_with_l_moments(self):
        """L-moment ratio rows should match l_moments() output."""
        np.random.seed(42)
        ts = TimeSeries(np.random.randn(100))
        summary = ts.summary()
        lmom = ts.l_moments(nmom=4)
        assert summary.loc["L-CV", "Series1"] == pytest.approx(
            float(lmom.loc["t", "Series1"])
        ), "L-CV should match l_moments().loc['t']"
        assert summary.loc["L-skewness", "Series1"] == pytest.approx(
            float(lmom.loc["t3", "Series1"])
        ), "L-skewness should match l_moments().loc['t3']"
        assert summary.loc["L-kurtosis", "Series1"] == pytest.approx(
            float(lmom.loc["t4", "Series1"])
        ), "L-kurtosis should match l_moments().loc['t4']"
