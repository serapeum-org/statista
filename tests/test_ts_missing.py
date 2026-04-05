"""Tests for the MissingDataMixin (Phase 2)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from statista.time_series import TimeSeries


class TestMissingSummary:
    """Tests for missing_summary() method."""

    def test_no_missing_data(self):
        """All-valid data should show zero missing counts."""
        ts = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = ts.missing_summary()
        assert result.loc["Series1", "missing_count"] == 0
        assert result.loc["Series1", "missing_pct"] == 0.0
        assert result.loc["Series1", "longest_gap"] == 0
        assert result.loc["Series1", "n_gaps"] == 0

    def test_all_missing(self):
        """All-NaN data should show 100% missing."""
        ts = TimeSeries(np.array([np.nan, np.nan, np.nan]))
        result = ts.missing_summary()
        assert result.loc["Series1", "missing_count"] == 3
        assert result.loc["Series1", "missing_pct"] == pytest.approx(100.0)
        assert result.loc["Series1", "longest_gap"] == 3
        assert result.loc["Series1", "n_gaps"] == 1

    def test_gaps_counted_correctly(self):
        """Two separate gaps should be counted as n_gaps=2."""
        data = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
        ts = TimeSeries(data)
        result = ts.missing_summary()
        assert (
            result.loc["Series1", "n_gaps"] == 2
        ), f"Expected 2 gaps, got {result.loc['Series1', 'n_gaps']}"
        assert result.loc["Series1", "longest_gap"] == 2

    def test_mean_gap_length(self):
        """Mean gap length for gaps of length 2 and 1 should be 1.5."""
        data = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
        ts = TimeSeries(data)
        result = ts.missing_summary()
        assert result.loc["Series1", "mean_gap_length"] == pytest.approx(1.5)

    def test_first_last_valid(self):
        """first_valid and last_valid should reflect the correct indices."""
        data = np.array([np.nan, 2.0, 3.0, np.nan])
        ts = TimeSeries(data)
        result = ts.missing_summary()
        assert result.loc["Series1", "first_valid"] == 1
        assert result.loc["Series1", "last_valid"] == 2

    def test_multi_column(self):
        """Should produce one row per column."""
        data = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 3.0]])
        ts = TimeSeries(data, columns=["A", "B"])
        result = ts.missing_summary()
        assert result.shape[0] == 2, f"Expected 2 rows, got {result.shape[0]}"
        assert result.loc["A", "missing_count"] == 1
        assert result.loc["B", "missing_count"] == 1

    def test_returns_dataframe(self):
        """Return type should be DataFrame."""
        ts = TimeSeries(np.array([1.0, 2.0]))
        result = ts.missing_summary()
        assert isinstance(result, DataFrame)


class TestGapAnalysis:
    """Tests for gap_analysis() method."""

    def test_no_gaps(self):
        """All-valid data should return empty DataFrame."""
        ts = TimeSeries(np.array([1.0, 2.0, 3.0]))
        result = ts.gap_analysis()
        assert len(result) == 0

    def test_single_gap(self):
        """Single gap should return one row with correct length."""
        data = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        ts = TimeSeries(data)
        result = ts.gap_analysis()
        assert len(result) == 1
        assert int(result.iloc[0]["gap_length"]) == 3

    def test_multiple_gaps_sorted(self):
        """Multiple gaps should be sorted by length descending."""
        data = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
        ts = TimeSeries(data)
        result = ts.gap_analysis()
        assert len(result) == 2
        assert int(result.iloc[0]["gap_length"]) >= int(result.iloc[1]["gap_length"])

    def test_gap_at_end(self):
        """Gap at the end of the series should be detected."""
        data = np.array([1.0, 2.0, np.nan, np.nan])
        ts = TimeSeries(data)
        result = ts.gap_analysis()
        assert len(result) == 1
        assert int(result.iloc[0]["gap_length"]) == 2

    def test_gap_at_start(self):
        """Gap at the beginning of the series should be detected."""
        data = np.array([np.nan, np.nan, 3.0, 4.0])
        ts = TimeSeries(data)
        result = ts.gap_analysis()
        assert len(result) == 1
        assert int(result.iloc[0]["gap_start"]) == 0

    def test_column_filter(self):
        """Specifying column should only analyze that column."""
        data = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan]])
        ts = TimeSeries(data, columns=["A", "B"])
        result = ts.gap_analysis(column="A")
        assert all(result["column"] == "A")

    def test_returns_correct_columns(self):
        """Result should have columns: column, gap_start, gap_end, gap_length."""
        data = np.array([1.0, np.nan, 3.0])
        ts = TimeSeries(data)
        result = ts.gap_analysis()
        assert set(result.columns) == {"column", "gap_start", "gap_end", "gap_length"}


class TestCompletenessReport:
    """Tests for completeness_report() method."""

    def test_full_data(self):
        """100% complete data should return 100.0 for all periods."""
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        ts = TimeSeries(np.random.randn(365), index=idx)
        report = ts.completeness_report(freq="YE")
        assert report.values[0, 0] == pytest.approx(100.0)

    def test_partial_data(self):
        """Data with NaN should show less than 100% completeness."""
        idx = pd.date_range("2000-01-01", periods=100, freq="D")
        data = np.ones(100)
        data[0:10] = np.nan
        ts = TimeSeries(data, index=idx)
        report = ts.completeness_report(freq="YE")
        assert report.values[0, 0] < 100.0

    def test_monthly_freq(self):
        """Monthly frequency should produce ~12 rows for a year of data."""
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        ts = TimeSeries(np.random.randn(365), index=idx)
        report = ts.completeness_report(freq="ME")
        assert report.shape[0] >= 11


class TestDetectOutliers:
    """Tests for detect_outliers() method."""

    def test_iqr_method(self):
        """IQR method should detect extreme values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        ts = TimeSeries(data)
        result = ts.detect_outliers(method="iqr", threshold=1.5)
        assert (
            result.loc[5, "Series1"] is np.True_
        ), "Value 100.0 should be flagged as outlier"

    def test_zscore_method(self):
        """Z-score method should detect values far from the mean."""
        data = np.array([1.0, 2.0, 3.0, 100.0, 2.5, 1.5])
        ts = TimeSeries(data)
        result = ts.detect_outliers(method="zscore", threshold=2.0)
        assert result.loc[3, "Series1"] is np.True_

    def test_modified_zscore_method(self):
        """Modified Z-score (MAD-based) should detect outliers robustly."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 50.0])
        ts = TimeSeries(data)
        result = ts.detect_outliers(method="modified_zscore", threshold=3.5)
        assert result.loc[5, "Series1"] is np.True_

    def test_no_outliers(self):
        """Uniform data should have no outliers."""
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ts = TimeSeries(data)
        result = ts.detect_outliers(method="iqr")
        assert not result.any().any(), "Constant data should have no outliers"

    def test_returns_boolean_dataframe(self):
        """Return type should be a boolean DataFrame."""
        ts = TimeSeries(np.array([1.0, 2.0, 3.0]))
        result = ts.detect_outliers()
        assert result.dtypes.iloc[0] == bool

    def test_column_filter(self):
        """Specifying column should only return that column."""
        data = np.random.randn(50, 3)
        ts = TimeSeries(data, columns=["A", "B", "C"])
        result = ts.detect_outliers(column="B")
        assert result.columns.tolist() == ["B"]

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        ts = TimeSeries(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Unknown method"):
            ts.detect_outliers(method="invalid")

    def test_handles_nan_in_data(self):
        """NaN values should not be flagged as outliers."""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        ts = TimeSeries(data)
        result = ts.detect_outliers(method="iqr")
        assert not result.loc[2, "Series1"], "NaN should not be flagged as outlier"

    def test_zscore_zero_std(self):
        """Z-score with zero std (constant data) should flag nothing."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        ts = TimeSeries(data)
        result = ts.detect_outliers(method="zscore", threshold=2.0)
        assert not result.any().any()

    def test_modified_zscore_zero_mad(self):
        """Modified Z-score with zero MAD should flag nothing."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        ts = TimeSeries(data)
        result = ts.detect_outliers(method="modified_zscore", threshold=3.5)
        assert not result.any().any()


class TestOutlierPlot:
    """Tests for outlier_plot() method."""

    def test_returns_figure_axes(self):
        """Should return (Figure, Axes) tuple."""
        np.random.seed(42)
        data = np.concatenate([np.random.randn(50), [10.0]])
        ts = TimeSeries(data)
        fig, ax = ts.outlier_plot(method="zscore", threshold=2.5)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_iqr_shows_bounds(self):
        """IQR method should draw threshold lines."""
        data = np.concatenate([np.random.randn(50), [10.0]])
        ts = TimeSeries(data)
        fig, ax = ts.outlier_plot(method="iqr", threshold=1.5)
        lines = ax.get_lines()
        assert len(lines) >= 3, "Should have data line + 2 threshold lines"
        plt.close(fig)

    def test_custom_labels(self):
        """Title and axis labels should be applied."""
        ts = TimeSeries(np.random.randn(50))
        fig, ax = ts.outlier_plot(title="Outliers", xlabel="Time", ylabel="Value")
        assert ax.get_title() == "Outliers"
        plt.close(fig)

    def test_multi_column_uses_first(self):
        """Multi-column without column arg should default to first column."""
        ts = TimeSeries(np.random.randn(50, 2), columns=["A", "B"])
        fig, ax = ts.outlier_plot()
        assert isinstance(ax, plt.Axes)
        plt.close(fig)
