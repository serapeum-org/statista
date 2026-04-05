# Summary

- **Massive feature addition**: 53 new methods + 3 dataclasses across 12 mixin files,
  transforming `TimeSeries` from a 7-method visualization wrapper into a comprehensive
  statistical analysis class. ~9900 lines added, 376 tests.
- **Architecture is solid**: Mixin-based subpackage split avoids monolithic files, deferred
  imports prevent circular dependencies, `_constructor` preserves type through pandas
  operations. Backward compatibility maintained.
- **Zero new dependencies**: All algorithms (MK, Pettitt, SNHT, ADF, KPSS, ACF, PACF,
  baseflow separation, etc.) implemented from scratch using numpy/scipy.
- **Key risks**: ADF OLS regression is fragile for small samples (singular matrices), multi-column
  FDC crashes when columns have different NaN counts, approximate p-values for ADF/KPSS, and
  Yue-Wang MK correction applies wrong thresholding strategy.

# Findings

## Critical

### C1: `_base.py` — `data.shape[1]` crashes for list inputs

**File**: `src/statista/time_series/_base.py:87`
**Impact**: The constructor type hint declares `List[float]` as valid input, but
`data.shape[1]` crashes with `AttributeError` for plain lists and `IndexError` for 1D
pandas Series. Also, when a DataFrame with named columns is passed and `columns=None`,
line 94 silently overwrites column names to `Series1`, `Series2`, etc.
**Fix**: Add `data = np.array(data)` at the top for list inputs. Skip column generation
when `isinstance(data, DataFrame)`.

### C2: ADF lag selection produces singular OLS matrices on small samples

**File**: `src/statista/time_series/_stationarity.py:229-258`
**Impact**: For series with n < ~25, the number of regressors exceeds the number of
observations, causing `LinAlgError: Singular matrix` at `np.linalg.inv(x.T @ x)`. Even
when not singular, using `np.linalg.inv` is numerically fragile. Any series shorter than
~25 observations can hit this.
**Fix**: (1) Reduce `used_lag` until `nobs > ncols`, (2) Replace `np.linalg.inv` with
`np.linalg.pinv` or `np.linalg.solve`.

## High

### H1: Multi-column `flow_duration_curve` crashes when columns have different NaN counts

**File**: `src/statista/time_series/_hydrological.py:114-117`
**Impact**: When `column=None` (overlay all columns), each column is dropna'd independently,
producing arrays of different lengths. The DataFrame constructor at line 115 uses the first
column's exceedance array as the shared index, raising `ValueError` for any column with a
different number of valid values.
**Fix**: Either require same-length data or build separate DataFrames per column.

### H2: ADF/KPSS p-value approximations are coarse

**File**: `src/statista/time_series/_stationarity.py:281-306, 371-392`
**Impact**: ADF uses 3-point interpolation from fixed asymptotic critical values (ignores
sample size `n` entirely — the `n` parameter is dead code). KPSS is clamped to [0.01, 0.10].
For ADF, a test statistic of 0 returns p≈0.54 when the true MacKinnon value is ~0.96.
**Fix**: Use published MacKinnon (1994) regression surface coefficients, or document
prominently as screening-only approximations.

### H3: Yue-Wang MK correction applies wrong thresholding strategy

**File**: `src/statista/time_series/_trend.py:500-506`
**Impact**: The Yue & Wang (2004) paper uses ALL autocorrelation lags without significance
thresholding. The code applies the Hamed-Rao significance threshold (`if abs(acf_vals[i])
> ci`), making the two methods behave more similarly than intended and not matching the
reference paper.
**Fix**: Remove the `if abs(acf_vals[i]) > ci` guard from `_yue_wang_correction`.

### H4: `fit_distributions()` catches all exceptions silently

**File**: `src/statista/time_series/_distribution.py:348-377`
**Impact**: Bare `except Exception as e` catches programming bugs (AttributeError, etc.)
and hides them as strings in the result DataFrame.
**Fix**: Catch only `ValueError`, `RuntimeError`.

### H5: Mann-Kendall `ZeroDivisionError` for n <= 1

**File**: `src/statista/time_series/_trend.py:397`
**Impact**: `tau = s / (n * (n - 1) / 2)` divides by zero when n=1.
**Fix**: Add `if n < 3: raise ValueError(...)` at top of `_mann_kendall_single`.

## Medium

### M1: No minimum data length validation across methods

**Files**: `_trend.py`, `_stationarity.py`, `_correlation.py`, `_changepoint.py`
**Impact**: Short series (n < 5) produce `LinAlgError`, `ZeroDivisionError`, or NaN without
clear error messages. Pettitt/SNHT/Buishand crash on n < 2 (`argmax of empty sequence`).
**Fix**: Add minimum length checks at each method entry.

### M2: Constant data causes silent NaN in ACF/CCF/KPSS

**Files**: `_correlation.py:498,515`, `_stationarity.py:350`
**Impact**: Zero-variance data produces `0/0 = NaN` in ACF (division by `acov[0]`), CCF
(division by `denom`), and KPSS (division by `sigma2`). Results propagate as NaN without
warning.
**Fix**: Guard with early return: constant series ACF = [1, 0, 0, ...], KPSS → stationary.

### M3: `standardized_anomaly` division by zero when monthly std is 0

**File**: `src/statista/time_series/_comparison.py:170-172`
**Impact**: If all values in a given month are identical, `monthly_std` is 0, producing
`inf`. The `fillna(0.0)` masks NaN but not inf.
**Fix**: Replace zero std with NaN before division.

### M4: `regime_comparison` crashes with `split_at` at boundaries

**File**: `src/statista/time_series/_comparison.py:297-298, 338`
**Impact**: `split_at=0` or `split_at >= len(data)` produces empty segments. `mannwhitneyu`
raises `ValueError` on empty input.
**Fix**: Validate `0 < split_at < len(data)`.

### M5: `ljung_box` and `exceedance_probability` multi-column dtype loss

**Files**: `_correlation.py:456-463`, `_hydrological.py:260-268`
**Impact**: `np.concatenate` on mixed-type DataFrames converts everything to `object` dtype.
Numeric columns become strings.
**Fix**: Use `pd.concat(frames, ignore_index=True)`.

### M6: Anderson-Darling p-value is unreliable binary bucket

**File**: `src/statista/time_series/_distribution.py:240-244`
**Impact**: Returns p=0.05 if stat > 5% critical value, else p=0.10. The 1% critical value
is available from scipy but unused. At alpha=0.01, the test reports "Normal" even when the
statistic exceeds the 1% threshold.
**Fix**: Check all available significance levels from `anderson.critical_values`.

### M7: `savgol` smooth silently produces NaN with NaN input

**File**: `src/statista/time_series/_decomposition.py:216-229`
**Impact**: scipy's `savgol_filter` propagates NaN through the entire output. Moving average
and exponential smoothing handle NaN gracefully via pandas. No warning given.
**Fix**: Document the limitation or pre-fill NaN before filtering.

### M8: `stat_result` dataclasses defined but never used

**File**: `src/statista/stat_result.py`
**Impact**: `StatTestResult`, `TrendTestResult`, `ChangePointResult` are defined and tested
but no mixin method returns them — all return DataFrames instead. Dead infrastructure.
**Fix**: Either wire them in as return types or remove/defer until needed.

### M9: `tools.py` `round()` semantic change

**File**: `src/statista/tools.py:452-458`
**Impact**: Changed from banker's rounding to ROUND_HALF_UP. This is likely intentional
but changes public API behavior.
**Fix**: Document the change in changelog.

## Low

### L1: `violin` passes NaN data to matplotlib without dropna

**File**: `src/statista/time_series/_visualization.py:296-303`
**Impact**: `box_plot` calls `dropna()` per column but `violin` passes raw `self.values`.
NaN values can cause matplotlib errors or incorrect plots.

### L2: Two plot helpers bypass `_get_ax_fig`/`_adjust_axes_labels`

**Files**: `_decomposition.py:353-378`, `_seasonal.py:140-164`
**Impact**: `_plot_decomposition` and `seasonal_subseries` call `plt.subplots()` directly.
Users cannot pass `fig=`/`ax=` kwargs. Returns only last axis, losing access to subplots.

### L3: Redundant `if j != i` in Mann-Kendall slope loop

**File**: `src/statista/time_series/_trend.py:403`
**Impact**: Always True since inner loop starts at `i+1`. Cosmetic dead code.

### L4: Pre-commit hooks skipped during commits

**Impact**: Three hooks with pre-existing failures (nbval, pytest-check, doctest) were
skipped via `SKIP=`. The test suite was run manually before each commit.

### L5: Unseeded fixtures in `test_time_series.py`

**File**: `tests/test_time_series.py:13-19`
**Impact**: `sample_data_1d`/`sample_data_2d` use `np.random.randn()` without seed.
Tests only check shapes/types so risk is very low, but non-deterministic.

## Nit

### N1: `is np.True_` comparisons in `test_ts_missing.py` are fragile

**File**: `tests/test_ts_missing.py:170, 178, 185`
**Impact**: Should use `== True` like other test files do for DataFrame boolean values.

### N2: Missing `__all__` in mixin modules

### N3: Inconsistent `column=None` behavior undocumented

Some methods iterate all columns, others default to first column. By design but undocumented.

# Tests

## Added: 376 tests across 12 files

All tests pass. Tests cover happy paths, boundary conditions, and most error cases.

## Gaps identified

- **`test_ts_stationarity.py` and `test_ts_changepoint.py` have zero `pytest.raises` tests**
  — no error/exception coverage at all.
- **`fit_distributions` tested only with `method="mle"`** — default `method="lmoments"` path
  is untested.
- **No test for constant data** in stationarity, correlation, or changepoint tests.
- **No test for empty TimeSeries** (`np.array([])`) anywhere.
- **No integration test** verifying a full workflow (create → stationarity → trend → plot).
- **`acf(fft=False)` path untested**.

# Questions and Assumptions

1. Is the `tools.py` `round()` change intentional? (banker's → ROUND_HALF_UP)
2. Are approximate ADF/KPSS p-values acceptable, or should MacKinnon coefficients be added?
3. Should `stat_result` dataclasses be wired in as return types, or remain as infrastructure?
4. Was the `_base.py` list-input crash tested? (`TimeSeries([1, 2, 3])`)

# Residual Risks

- **P-value quality** (H2): ADF p-values use 3-point interpolation from fixed asymptotic
  critical values. KPSS clamped to [0.01, 0.10]. Documented as approximate — acceptable
  for screening, recommend statsmodels for publication-grade p-values.
- **savgol NaN** (M7): scipy's savgol_filter propagates NaN. Documented limitation.
- **stat_result unused** (M8): Dataclasses exist as infrastructure for future phases.

# Issue Tracker

| ID | Severity | State   | Description                                                            | File(s)                               |
|----|----------|---------|------------------------------------------------------------------------|---------------------------------------|
| C1 | Critical | Solved  | `data.shape[1]` crashes for list inputs; DataFrame columns overwritten | `_base.py`                            |
| C2 | Critical | Solved  | ADF OLS singular matrix on small samples (n<25)                        | `_stationarity.py`                    |
| H1 | High     | Solved  | Multi-column FDC crashes with different NaN counts                     | `_hydrological.py`                    |
| H2 | High     | Wontfix | ADF/KPSS p-value approximations coarse/clamped                         | `_stationarity.py`                    |
| H3 | High     | Solved  | Yue-Wang MK correction applies wrong thresholding                      | `_trend.py`                           |
| H4 | High     | Solved  | `fit_distributions()` bare except masks bugs                           | `_distribution.py`                    |
| H5 | High     | Solved  | MK ZeroDivisionError for n<=1                                          | `_trend.py`                           |
| M1 | Medium   | Solved  | No minimum data length validation                                      | Multiple                              |
| M2 | Medium   | Solved  | Constant data → NaN in ACF/CCF/KPSS                                    | `_correlation.py`, `_stationarity.py` |
| M3 | Medium   | Solved  | standardized_anomaly div-by-zero monthly std=0                         | `_comparison.py`                      |
| M4 | Medium   | Solved  | regime_comparison crashes at split_at boundaries                       | `_comparison.py`                      |
| M5 | Medium   | Solved  | Multi-column concat loses dtypes                                       | `_correlation.py`, `_hydrological.py` |
| M6 | Medium   | Solved  | Anderson-Darling p-value binary bucket                                 | `_distribution.py`                    |
| M7 | Medium   | Wontfix | savgol smooth NaN propagation (documented scipy limitation)            | `_decomposition.py`                   |
| M8 | Medium   | Wontfix | stat_result dataclasses unused (future infrastructure)                 | `stat_result.py`                      |
| M9 | Medium   | Wontfix | tools.py round() semantic change (pre-existing, intentional)           | `tools.py`                            |
| L1 | Low      | Solved  | violin NaN not dropped                                                 | `_visualization.py`                   |
| L2 | Low      | Wontfix | Two plot helpers bypass _get_ax_fig (multi-subplot design)             | `_decomposition.py`, `_seasonal.py`   |
| L3 | Low      | Solved  | Redundant `if j != i`                                                  | `_trend.py`                           |
| L4 | Low      | Wontfix | Pre-commit hooks skipped (pre-existing failures)                       | Config                                |
| L5 | Low      | Wontfix | Unseeded test fixtures (low risk, shape-only assertions)               | `test_time_series.py`                 |
| N1 | Nit      | Solved  | `is np.True_` fragile comparisons                                      | `test_ts_missing.py`                  |
| N2 | Nit      | Wontfix | Missing `__all__` in mixin modules (private files)                     | Multiple                              |
| N3 | Nit      | Wontfix | Inconsistent column=None behavior (by design)                          | Multiple                              |

**Resolution: 16 Solved, 8 Wontfix (documented limitations / design choices). 0 Open.**
