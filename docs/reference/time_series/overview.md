# Time Series Subpackage — Developer Documentation

## Overview

The `statista.time_series` subpackage provides the `TimeSeries` class — a pandas `DataFrame` subclass
extended with 53 statistical analysis and visualization methods designed for researchers in hydrology,
climate science, and environmental engineering.

The `statista.stat_result` module provides structured return types (frozen dataclasses) for
statistical hypothesis tests.

## Architecture

`TimeSeries` is composed via **mixin inheritance**. Each mixin lives in its own file and adds a
specific category of functionality:

```
pandas.DataFrame
    └── TimeSeriesBase          (_base.py)
            └── TimeSeries      (__init__.py)
                 ├── DescriptiveMixin      (_descriptive.py)
                 ├── VisualizationMixin    (_visualization.py)
                 ├── MissingDataMixin      (_missing.py)
                 ├── CorrelationMixin      (_correlation.py)
                 ├── StationarityMixin     (_stationarity.py)
                 ├── TrendMixin            (_trend.py)
                 ├── DistributionMixin     (_distribution.py)
                 ├── ChangePointMixin      (_changepoint.py)
                 ├── DecompositionMixin    (_decomposition.py)
                 ├── SeasonalMixin         (_seasonal.py)
                 ├── HydrologicalMixin     (_hydrological.py)
                 └── ComparisonMixin       (_comparison.py)
```

## File Structure

```
src/statista/
├── stat_result.py                  # Frozen dataclasses for test results
├── time_series/
│   ├── __init__.py                 # TimeSeries composition + re-exports
│   ├── _base.py                    # TimeSeriesBase (DataFrame subclass + helpers)
│   ├── _descriptive.py             # stats, extended_stats, l_moments, summary
│   ├── _visualization.py           # box_plot, violin, raincloud, histogram, density, rolling_statistics
│   ├── _missing.py                 # missing_summary, gap_analysis, completeness_report, detect_outliers, outlier_plot
│   ├── _correlation.py             # acf, pacf, cross_correlation, lag_plot, correlation_matrix, ljung_box
│   ├── _stationarity.py            # adf_test, kpss_test, stationarity_summary
│   ├── _trend.py                   # mann_kendall (5 variants), sens_slope, detrend (4 methods), innovative_trend_analysis
│   ├── _distribution.py            # qq_plot, pp_plot, normality_test (5 methods), empirical_cdf, fit_distributions
│   ├── _changepoint.py             # pettitt_test, snht_test, buishand_range_test, cusum, homogeneity_summary
│   ├── _decomposition.py           # classical_decompose, smooth (3 methods), envelope
│   ├── _seasonal.py                # monthly_stats, seasonal_subseries, annual_cycle, periodogram, seasonal_mann_kendall
│   ├── _hydrological.py            # flow_duration_curve, annual_extremes, exceedance_probability,
│   │                               # baseflow_separation (3 algorithms), baseflow_index, flashiness_index, recession_analysis
│   └── _comparison.py              # anomaly, standardized_anomaly, double_mass_curve, regime_comparison
tests/
├── test_stat_result.py             # 16 tests
├── test_time_series.py             # 103 tests (base + descriptive + visualization)
├── test_ts_missing.py              # 31 tests
├── test_ts_correlation.py          # 29 tests
├── test_ts_stationarity.py         # 21 tests
├── test_ts_trend.py                # 30 tests
├── test_ts_distribution.py         # 27 tests
├── test_ts_changepoint.py          # 24 tests
├── test_ts_decomposition.py        # 25 tests
├── test_ts_seasonal.py             # 27 tests
├── test_ts_hydrological.py         # 37 tests
└── test_ts_comparison.py           # 22 tests
```

**Total: 376 tests, 0 external dependencies added.**

---

## Complete API Reference

### Result Dataclasses (`statista.stat_result`)

| Class | Fields | Used By |
|---|---|---|
| `StatTestResult` | test_name, statistic, p_value, conclusion, alpha, details | ADF, KPSS, normality tests |
| `TrendTestResult` | trend, h, p_value, z, tau, s, var_s, slope, intercept | Mann-Kendall |
| `ChangePointResult` | test_name, h, change_point, change_point_date, statistic, p_value, mean_before, mean_after | Pettitt, SNHT, Buishand |

### Descriptive Statistics (`_descriptive.py`)

| Method | Returns | Description |
|---|---|---|
| `stats` (property) | DataFrame | Basic summary via pandas describe() |
| `extended_stats` (property) | DataFrame | 17 statistics: count, mean, std, cv, skewness, kurtosis, percentiles, iqr, mad |
| `l_moments(nmom=5)` | DataFrame | L1, L2, t (L-CV), t3 (L-skewness), t4 (L-kurtosis) per column |
| `summary()` | DataFrame | Paper-ready table combining extended_stats + L-moment ratios |

### Visualization (`_visualization.py`)

| Method | Returns | Description |
|---|---|---|
| `box_plot(mean, notch)` | (Fig, Ax) | Box-and-whisker with optional mean markers and notches |
| `violin(mean, median, extrema, side, spacing)` | (Fig, Ax) | Violin plot with half-violin and spacing control |
| `raincloud(overlay, order)` | (Fig, Ax) | Combined violin + scatter + box |
| `histogram(bins)` | (n, edges, Fig, Ax) | Histogram with bin counts |
| `density()` | (Fig, Ax) | KDE density plot |
| `rolling_statistics(window)` | (Fig, Ax) | Rolling mean + std |

### Missing Data & Quality Control (`_missing.py`)

| Method | Returns | Description |
|---|---|---|
| `missing_summary()` | DataFrame | Per-column: count, pct, longest gap, n_gaps, mean gap length |
| `gap_analysis(column)` | DataFrame | All gaps with start/end/length, sorted by length |
| `completeness_report(freq)` | DataFrame | Data completeness % per time period |
| `detect_outliers(method, threshold)` | DataFrame (bool) | IQR, Z-score, or modified Z-score outlier detection |
| `outlier_plot(method, threshold)` | (Fig, Ax) | Time series with outliers highlighted |

### Autocorrelation & Dependence (`_correlation.py`)

| Method | Returns | Description |
|---|---|---|
| `acf(nlags, alpha, fft, plot)` | (array, (Fig,Ax)) | Autocorrelation function with confidence bands |
| `pacf(nlags, alpha, plot)` | (array, (Fig,Ax)) | Partial ACF via Levinson-Durbin recursion |
| `cross_correlation(col_x, col_y, nlags, plot)` | (array, (Fig,Ax)) | Cross-correlation with peak-lag annotation |
| `lag_plot(lag, column)` | (Fig, Ax) | x(t) vs x(t-lag) scatter with Pearson r |
| `correlation_matrix(method, plot)` | (corr_df, pval_df, (Fig,Ax)) | Pairwise correlation WITH p-values + significance heatmap |
| `ljung_box(lags, column)` | DataFrame | Ljung-Box white noise test per lag |

### Stationarity Testing (`_stationarity.py`)

| Method | Returns | Description |
|---|---|---|
| `adf_test(regression, max_lag)` | DataFrame | Augmented Dickey-Fuller unit root test |
| `kpss_test(regression, n_lags)` | DataFrame | KPSS stationarity test (opposite null of ADF) |
| `stationarity_summary(alpha)` | DataFrame | Combined ADF+KPSS diagnosis: Stationary / Non-stationary / Trend-stationary / Inconclusive |

### Trend Detection (`_trend.py`)

| Method | Returns | Description |
|---|---|---|
| `mann_kendall(alpha, method, lag)` | DataFrame | MK with 5 variants: original, hamed_rao, yue_wang, pre_whitening, trend_free_pre_whitening |
| `sens_slope(alpha)` | DataFrame | Robust non-parametric slope + CI via theilslopes |
| `detrend(method, order)` | TimeSeries | Remove trend: linear, constant, polynomial, sens |
| `innovative_trend_analysis(column)` | (DataFrame, (Fig,Ax)) | Sen (2012) ITA scatter with 1:1 line |

### Distribution-Aware (`_distribution.py`)

| Method | Returns | Description |
|---|---|---|
| `qq_plot(distribution, confidence)` | (Fig, Ax) | QQ plot against any scipy.stats distribution |
| `pp_plot(distribution)` | (Fig, Ax) | PP plot (empirical vs theoretical CDF) |
| `normality_test(method, alpha)` | DataFrame | 5 methods: auto, shapiro, dagostino, anderson, jarque_bera |
| `empirical_cdf(column)` | (Fig, Ax) | Step-function empirical CDF, multi-column overlay |
| `fit_distributions(method)` | DataFrame | Fit all statista distributions, return best fit + KS test |

### Change Point Detection (`_changepoint.py`)

| Method | Returns | Description |
|---|---|---|
| `pettitt_test(alpha)` | DataFrame | Rank-based U statistic, analytical p-value |
| `snht_test(alpha)` | DataFrame | Standard Normal Homogeneity Test |
| `buishand_range_test(alpha)` | DataFrame | Adjusted partial sums range statistic |
| `cusum(column, plot)` | (DataFrame, (Fig,Ax)) | CUSUM with confidence bands |
| `homogeneity_summary(alpha)` | DataFrame | Combined 3-test diagnosis with confirmation |

### Decomposition & Smoothing (`_decomposition.py`)

| Method | Returns | Description |
|---|---|---|
| `classical_decompose(period, model, plot)` | (DataFrame, (Fig,Ax)) | Additive/multiplicative decomposition with 4-panel plot |
| `smooth(method, window)` | TimeSeries | moving_average, exponential, savgol (Savitzky-Golay) |
| `envelope(window, lower_pct, upper_pct)` | (Fig, Ax) | Rolling percentile band plot |

### Seasonal & Periodic Analysis (`_seasonal.py`)

| Method | Returns | Description |
|---|---|---|
| `monthly_stats(column)` | DataFrame | Per-month mean, std, cv, min, max, median, skewness |
| `seasonal_subseries(period, column)` | (Fig, Ax) | Mini time series per season with mean lines |
| `annual_cycle(column)` | (Fig, Ax) | All years overlaid on Jan-Dec axis |
| `periodogram(column, method, fs, plot)` | (freqs, power, (Fig,Ax)) | PSD via welch or periodogram with peak annotation |
| `seasonal_mann_kendall(period, alpha)` | DataFrame | Per-season MK combined (Hirsch et al., 1982) |

### Hydrological Methods (`_hydrological.py`)

| Method | Returns | Description |
|---|---|---|
| `flow_duration_curve(log_scale, method, plot)` | (DataFrame, (Fig,Ax)) | FDC with 3 plotting positions, multi-column overlay |
| `annual_extremes(kind, water_year_start)` | TimeSeries | Annual max/min series with water year config |
| `exceedance_probability(method)` | DataFrame | Empirical exceedance + return period per value |
| `baseflow_separation(method, alpha, plot)` | (DataFrame, (Fig,Ax)) | 3 algorithms: Lyne-Hollick, Eckhardt, Chapman-Maxwell |
| `baseflow_index(method, alpha)` | DataFrame | BFI = sum(baseflow) / sum(total_flow) |
| `flashiness_index(column)` | DataFrame | Richards-Baker flashiness index |
| `recession_analysis(min_length, column, plot)` | (DataFrame, (Fig,Ax)) | Extract recessions, fit Q(t)=Q0*exp(-t/k) |

### Comparison & Anomaly (`_comparison.py`)

| Method | Returns | Description |
|---|---|---|
| `anomaly(reference, column, plot)` | (TimeSeries, (Fig,Ax)) | Deviation from mean/median, colored bar plot |
| `standardized_anomaly(column)` | TimeSeries | (x - monthly_mean) / monthly_std |
| `double_mass_curve(col_x, col_y, plot)` | (DataFrame, (Fig,Ax)) | Cumulative X vs Y for consistency detection |
| `regime_comparison(split_at, column)` | DataFrame | Before/after stats + Mann-Whitney U + relative change % |

---

## Dependencies

All methods implemented from scratch using only:
- **numpy** — core computation
- **scipy.stats** — statistical distributions, tests, correlation functions
- **scipy.signal** — detrend, periodogram, Savitzky-Golay filter
- **matplotlib** — all plotting
- **pandas** — DataFrame operations

**No statsmodels, no pymannkendall, no pyhomogeneity** — zero new dependencies.

---

## Usage Examples

```python
import numpy as np
import pandas as pd
from statista.time_series import TimeSeries

# Create from data
ts = TimeSeries(np.random.randn(365), index=pd.date_range("2020", periods=365, freq="D"))

# Descriptive
ts.extended_stats
ts.summary()
ts.l_moments()

# Missing data
ts.missing_summary()
ts.detect_outliers(method="iqr")

# Autocorrelation
ts.acf(nlags=20)
ts.correlation_matrix(method="spearman")

# Stationarity
ts.stationarity_summary()

# Trend
ts.mann_kendall(method="hamed_rao")
ts.sens_slope()

# Change points
ts.homogeneity_summary()

# Decomposition
ts.classical_decompose(period=12)
ts.smooth(method="savgol", window=15)

# Distribution
ts.normality_test()
ts.qq_plot()

# Seasonal
ts.monthly_stats()
ts.periodogram()

# Hydrological
ts.flow_duration_curve()
ts.baseflow_separation(method="lyne_hollick")
ts.flashiness_index()

# Comparison
ts.regime_comparison(split_at=180)
```
