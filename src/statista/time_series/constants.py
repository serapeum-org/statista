"""Shared constants for the time_series subpackage."""

# Default significance level for hypothesis tests across all methods that accept
# an ``alpha`` parameter (mann_kendall, adf_test, kpss_test, pettitt_test,
# snht_test, buishand_range_test, homogeneity_summary, stationarity_summary,
# normality_test, sens_slope, seasonal_mann_kendall, ljung_box, acf, pacf,
# correlation_matrix). Considered conservative for safety-critical analyses
# might warrant 0.01; exploratory work might warrant 0.10.
DEFAULT_ALPHA = 0.05
