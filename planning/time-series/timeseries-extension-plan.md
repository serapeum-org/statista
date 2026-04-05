# TimeSeries Extension Plan — Comprehensive Statistical Capabilities

## Current State

`TimeSeries` inherits from `pandas.DataFrame` and adds 7 methods — almost all visualization:

| Method | Type | What it does |
|---|---|---|
| `stats` (property) | Descriptive | Delegates to `.describe()` |
| `extended_stats` (property) | Descriptive | count, mean, std, cv, skewness, kurtosis, percentiles, IQR, MAD |
| `box_plot()` | Visualization | Box-and-whisker with optional mean/notch |
| `violin()` | Visualization | Violin plot with side control, spacing |
| `raincloud()` | Visualization | Violin + scatter + box overlay |
| `histogram()` | Visualization | Histogram with multi-series support |
| `density()` | Visualization | KDE via pandas `.plot(kind="density")` |
| `rolling_statistics()` | Visualization | Rolling mean + std plot |
| `calculate_whiskers()` | Utility | IQR-based whisker bounds (static) |

**The problem**: The class is a plotting wrapper with almost no statistical computation. A researcher who creates a `TimeSeries` object still has to leave the class to do any real analysis.

**The goal**: Extend `TimeSeries` into a statistical analysis object that competes with hydrostats, pymannkendall, pyhomogeneity, pingouin, and hydrosignatures — but integrated into a single DataFrame-based API.

---

## Research: What Other Packages Do

### Packages Audited

| Package | Focus | API Style | Return Types |
|---|---|---|---|
| **hydrostats / HydroErr** | 75 error metrics | Functional, `treat_values()` preprocessing | `np.floating` scalar |
| **hydroeval** | 7 metrics + evaluator | `evaluator(fn, sim, obs, transform=)` dispatcher | numpy arrays (KGE: 4-row decomposed) |
| **pyhomogeneity** | 6 homogeneity tests | Functional, uniform `(x, alpha, sim)` signature | Named tuples: `(h, cp, p, stat, avg)` |
| **pymannkendall** | 11 MK variants + slopes | Functional, uniform signature | Named tuples: 9 fields |
| **pingouin** | Full statistical testing | Functional, everything returns DataFrame | DataFrames with effect size, BF10, power |
| **ruptures** | Change point detection | OOP: `fit(signal).predict(pen=)` | `list[int]` of breakpoint indices |
| **statsmodels.tsa** | ACF/PACF/ADF/KPSS/STL | Functional, raw tuples | Tuples, DecomposeResult |
| **tsfresh** | 70+ time series features | Functional, parameterized extractors | Scalars or dicts |
| **hydrosignatures** | Hydrological signatures | Class facade + module functions | Series/DataFrame |
| **pastas** | Groundwater TFN modeling | OOP with `.stats`/`.plots` namespaces | DataFrame, (Fig, Ax) |
| **baseflow** | 12 baseflow separation methods | Single dispatcher function | DataFrame |
| **hydrotoolbox** | CLI + Python hydro tools | Functional | DataFrame |
| **darts.utils.statistics** | Clean facade for TS stats | Functional, combined tests | bool / Tuple |
| **scipy.signal** | Detrend, spectral, filtering | Functional | ndarray |

### Key Design Patterns to Adopt

1. **Named tuples / dataclasses for test results** (from pyhomogeneity, pymannkendall). NOT raw tuples like statsmodels.
2. **Consistent preprocessing pipeline** (from HydroErr's `treat_values()`). Every method handles NaN/Inf uniformly.
3. **KGE always returns decomposed components** (from hydroeval). Don't hide r, alpha, beta behind a flag.
4. **Monte Carlo p-values with seed parameter** (from pyhomogeneity). Add `seed` for reproducibility.
5. **`transform` parameter for metrics** (from hydroeval): `"log"`, `"sqrt"`, `"inv"` with automatic epsilon.
6. **DataFrame returns for test batteries** (from pingouin). Every test that runs per-column returns a summary DataFrame.
7. **Combined stationarity diagnosis** (from darts): ADF + KPSS together with interpretation.
8. **`plot` flag pattern** (from hydrobox, hydrosignatures): toggle visualization on/off.
9. **Hydrological signatures as a batch** (from hydrosignatures): `HydroSignatures` class computing all at once.

---

## Architecture Decisions

### A. Test Result Dataclass

All statistical tests return structured results, NOT raw tuples.

```python
# src/statista/stat_result.py

from dataclasses import dataclass, field

@dataclass(frozen=True)
class StatTestResult:
    """Result of a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    conclusion: str          # e.g., "Stationary at 5% significance level"
    alpha: float = 0.05
    details: dict = field(default_factory=dict)

@dataclass(frozen=True)
class TrendTestResult:
    """Result of a trend test (Mann-Kendall family)."""
    trend: str               # "increasing", "decreasing", "no trend"
    h: bool                  # reject null hypothesis
    p_value: float
    z: float                 # standardized test statistic
    tau: float               # Kendall's tau
    s: float                 # Mann-Kendall S statistic
    var_s: float             # variance of S
    slope: float             # Sen's slope
    intercept: float         # intercept of Sen's line

@dataclass(frozen=True)
class ChangePointResult:
    """Result of a change point test."""
    test_name: str
    h: bool                  # reject null (True = inhomogeneous)
    change_point: int        # index of change point
    change_point_date: object  # datetime if DatetimeIndex, else None
    statistic: float
    p_value: float
    mean_before: float
    mean_after: float
```

**Design rationale**: pymannkendall and pyhomogeneity use named tuples. We use frozen dataclasses for the same immutability but with better IDE support (type hints, docstrings). The field names match pymannkendall's conventions (`trend`, `h`, `p`, `z`, `Tau`, `s`, `var_s`, `slope`, `intercept`) for familiarity.

### B. File Organization — Mixin Subpackage

```
src/statista/time_series/
    __init__.py              # re-export TimeSeries (backward compat: `from statista.time_series import TimeSeries`)
    _base.py                 # TimeSeries class, __init__, _constructor, _get_ax_fig, _adjust_axes_labels
    _descriptive.py          # mixin: extended_stats, l_moments, summary
    _visualization.py        # mixin: box_plot, violin, raincloud, histogram, density, rolling_statistics
    _correlation.py          # mixin: acf, pacf, ccf, lag_plot, correlation_matrix, ljung_box
    _stationarity.py         # mixin: adf_test, kpss_test, stationarity_summary
    _trend.py                # mixin: mann_kendall, sens_slope, detrend, innovative_trend_analysis
    _changepoint.py          # mixin: pettitt_test, snht_test, buishand_range_test, cusum, homogeneity_summary
    _decomposition.py        # mixin: stl, classical_decompose, smooth, envelope
    _seasonal.py             # mixin: seasonal_subseries, annual_cycle, monthly_stats, periodogram
    _missing.py              # mixin: missing_summary, gap_analysis, completeness_report, detect_outliers
    _distribution.py         # mixin: qq_plot, pp_plot, normality_test, empirical_cdf, fit_distributions
    _comparison.py           # mixin: anomaly, standardized_anomaly, double_mass_curve, regime_comparison
    _hydrological.py         # mixin: flow_duration_curve, annual_extremes, exceedance_probability,
                             #         baseflow_separation, baseflow_index, flashiness_index, recession_analysis
```

**Implementation**: Each `_*.py` defines a mixin class. `TimeSeries` in `_base.py` inherits from all mixins + DataFrame. `__init__.py` re-exports `TimeSeries` so `from statista.time_series import TimeSeries` continues to work.

### C. Column-Wise Operation Convention

Every analytical method operates per-column by default. When returning test results for multi-column data, return a **summary DataFrame** with one row per column:

```python
# Example: ts.adf_test() on a 3-column TimeSeries returns:
#          statistic   p_value  used_lag  n_obs  crit_1%  crit_5%  crit_10%  conclusion
# col_A     -3.452    0.0092        2    100   -3.497   -2.891    -2.583  Stationary
# col_B     -1.234    0.6612        3    100   -3.497   -2.891    -2.583  Non-stationary
# col_C     -4.102    0.0009        1    100   -3.497   -2.891    -2.583  Stationary
```

For methods that only make sense on a single column (ACF, lag plot), accept a `column: str = None` parameter. If None and multi-column, either iterate with subplots or raise a clear error.

### D. Plotting Convention

All existing visualization methods use `_get_ax_fig()` + `_adjust_axes_labels()` + `plt.show()`. New methods follow the same pattern. Methods that are primarily analytical (tests, statistics) do NOT plot by default but accept `plot: bool = False` to optionally show a visualization.

### E. Dependency Strategy

| Dependency | Status | Used For |
|---|---|---|
| scipy.stats | Already in project | skew, kurtosis, MAD, normality tests, distributions |
| scipy.signal | Already in project (scipy) | detrend, periodogram, welch, savgol_filter |
| statsmodels | Available (used in examples/) | adfuller, kpss, acf, pacf, ccf, STL, seasonal_decompose, lowess, ljung_box |
| numpy | Already in project | Core computation |
| pandas | Already in project | DataFrame operations |
| matplotlib | Already in project | All plotting |

**No new package installations.** Mann-Kendall, Pettitt, SNHT, Buishand, Sen's slope, baseflow separation are implemented from scratch following pymannkendall and pyhomogeneity algorithms. This avoids adding pymannkendall/pyhomogeneity as dependencies while giving us full control.

---

## Phase 0 — Structural Refactoring

**Goal**: Split `time_series.py` (900+ lines) into a subpackage before adding 50+ methods.

### Task 0.1: Create `stat_result.py`

**File**: `src/statista/stat_result.py`

Create `StatTestResult`, `TrendTestResult`, `ChangePointResult` dataclasses as defined above. These are shared across multiple phases.

### Task 0.2: Create `time_series/` subpackage

1. Create `src/statista/time_series/` directory.
2. Move current `time_series.py` content into `_base.py` + `_visualization.py` + `_descriptive.py`.
3. Create `__init__.py` that re-exports `TimeSeries`.
4. Verify `from statista.time_series import TimeSeries` still works.
5. Verify all existing tests pass unchanged.
### Task 0.3: Set up mixin pattern

Define empty mixin classes in each `_*.py` file. Wire `TimeSeries` to inherit from all mixins. Verify nothing breaks.

---

## Phase 1 — Descriptive Statistics

### Task 1.1: `extended_stats` property **[DONE]**

Already implemented. Returns DataFrame with 17 statistics: count, mean, std, cv, skewness, kurtosis, min, 5%, 10%, 25%, 50%, 75%, 90%, 95%, max, iqr, mad.

### Task 1.2: `l_moments()` method

**Signature**:
```python
def l_moments(self, nmom: int = 5) -> DataFrame:
```

**What it does**: Expose L-moment computation from `parameters/lmoments.py` as a TimeSeries method. L-moments are more robust than conventional moments for small samples and heavy-tailed data (Hosking, 1990).

**Returns**: DataFrame with rows `["L1", "L2", "L3", "L4", "t2", "t3", "t4"]` and one column per series.
- L1 = L-location (= mean)
- L2 = L-scale
- L3, L4 = third and fourth L-moments
- t2 = L-CV = L2/L1
- t3 = L-skewness = L3/L2
- t4 = L-kurtosis = L4/L2

**Implementation**: Read `src/statista/parameters/lmoments.py` to find the existing `Lmoments` class. Call its `samlmu()` or equivalent method per column. Do NOT duplicate the computation — import and reuse.

**Tests**: Verify against known L-moments for simple datasets (e.g., uniform, exponential). Verify t3 ~ 0 for symmetric data.

### Task 1.3: `summary()` method

**Signature**:
```python
def summary(self) -> DataFrame:
```

**What it does**: The "Table 1 for a research paper" method. Combines everything a reviewer wants in one call.

**Returns**: DataFrame with rows:
- count, mean, std, cv, skewness, kurtosis, min, max, iqr, mad (from extended_stats)
- L-CV (t2), L-skewness (t3), L-kurtosis (t4) (from l_moments)
- normality_pvalue (from normality_test — Phase 8)
- adf_pvalue (from adf_test — Phase 3)

**Implementation**: Call `self.extended_stats`, `self.l_moments()`. The normality and ADF rows depend on later phases — add them when those phases are complete. Use sentinel `np.nan` until then.

**Tests**: Verify output shape, row names. Verify values match individual method outputs.

---

## Phase 2 — Autocorrelation & Dependence

**Reference implementations**: statsmodels `acf`/`pacf`/`ccf` (wrap these), `examples/auto-correlation.py` (existing prototype).

### Task 2.1: `acf()` method

**Signature**:
```python
def acf(
    self,
    nlags: int = 40,
    alpha: float = 0.05,
    fft: bool = True,
    column: str = None,
    plot: bool = True,
    **kwargs,
) -> Tuple[Union[np.ndarray, dict], Optional[Tuple[Figure, Axes]]]:
```

**What it does**: Compute and optionally plot the autocorrelation function.

**Implementation**:
- Wrap `statsmodels.tsa.stattools.acf(data, adjusted=False, nlags=nlags, fft=fft, alpha=alpha)`.
- statsmodels `acf` returns: `(acf_values,)` or `(acf_values, confint)` when `alpha` is set. `confint` shape is `(nlags+1, 2)`.
- **Single column** (or `column` specified): Return `(acf_array, (fig, ax))` or `(acf_array, None)` if `plot=False`.
- **Multi-column, no column specified**: Iterate over columns. Return `(dict_of_acf_arrays, (fig, axes))` with subplot grid.
- **Plot**: Stem plot (`ax.stem` or `ax.vlines` + `ax.hlines(0)`). Shaded confidence bands from the confint array. Follow existing `_get_ax_fig` + `_adjust_axes_labels` pattern.
- **Confidence bands**: statsmodels returns Bartlett bands by default (appropriate for MA(q) null). For simple white noise null, the band is `+/- z_{alpha/2} / sqrt(n)`.

**Tests**:
- White noise input: all lags (except 0) should be within confidence bands.
- AR(1) input with known phi: ACF should decay geometrically, verify `acf[1] ~ phi`.
- 1D and 2D inputs.
- `plot=False` returns None for figure.

### Task 2.2: `pacf()` method

**Signature**:
```python
def pacf(
    self,
    nlags: int = 40,
    alpha: float = 0.05,
    method: str = "ywadjusted",
    column: str = None,
    plot: bool = True,
    **kwargs,
) -> Tuple[Union[np.ndarray, dict], Optional[Tuple[Figure, Axes]]]:
```

**What it does**: Compute and optionally plot the partial autocorrelation function.

**Implementation**:
- Wrap `statsmodels.tsa.stattools.pacf(data, nlags=nlags, method=method, alpha=alpha)`.
- `method` options (from statsmodels): `"ywadjusted"` (default, Yule-Walker adjusted), `"ywm"`, `"ols"`, `"ols-adjusted"`, `"ld"`, `"ldb"`, `"burg"`.
- Same return pattern and plotting as `acf()`.
- **Confidence bands**: `+/- z_{alpha/2} / sqrt(n)` (standard for PACF under null of no partial correlation).

**Tests**:
- AR(1) input: PACF should cut off after lag 1.
- AR(2) input: PACF should cut off after lag 2.
- Verify method parameter is passed through correctly.

### Task 2.3: `cross_correlation()` method

**Signature**:
```python
def cross_correlation(
    self,
    col_x: str,
    col_y: str,
    nlags: int = 40,
    alpha: float = 0.05,
    plot: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, Optional[Tuple[Figure, Axes]]]:
```

**What it does**: Compute the cross-correlation function between two columns at lags -nlags to +nlags.

**Implementation**:
- Wrap `statsmodels.tsa.stattools.ccf(x, y, adjusted=True, fft=True, nlags=nlags, alpha=alpha)`.
- statsmodels `ccf` returns `(ccf_values,)` or `(ccf_values, confint)` when `alpha` is set.
- **Plot**: Stem plot centered at lag 0. Annotate lag of maximum absolute correlation with a marker.
- **Practical use**: Identify travel time (upstream-downstream), rainfall-runoff delay.

**Tests**:
- Two identical series: peak at lag 0.
- Shifted copy: peak at the correct lag.
- Verify annotations on plot.

### Task 2.4: `lag_plot()` method

**Signature**:
```python
def lag_plot(
    self,
    lag: int = 1,
    column: str = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
```

**What it does**: Scatter plot of x(t) vs x(t-lag). Quick visual diagnostic for serial dependence.

**Implementation**:
- Extract column data. Plot `data[lag:]` (y-axis) vs `data[:-lag]` (x-axis).
- Annotate with Pearson r value in the corner.
- If no serial dependence, points fill uniformly. Clustering = structure.

**Tests**: Verify scatter has correct number of points (n - lag). Verify annotation text contains "r =".

### Task 2.5: `correlation_matrix()` method

**Signature**:
```python
def correlation_matrix(
    self,
    method: str = "pearson",
    plot: bool = True,
    **kwargs,
) -> Tuple[DataFrame, DataFrame, Optional[Tuple[Figure, Axes]]]:
```

**What it does**: Compute pairwise correlation matrix WITH p-values (pandas `.corr()` has NO p-values).

**Implementation**:
- Compute correlation matrix via `self.corr(method=method)`.
- Compute p-value matrix using pairwise `scipy.stats.pearsonr`, `spearmanr`, or `kendalltau` (depending on `method`). Each returns `(r, pvalue)`.
- **Plot**: Annotated heatmap using `ax.imshow()`. Colormap diverging around 0. Annotate cells with correlation coefficient. Add significance markers: `*` for p < 0.05, `**` for p < 0.01, `***` for p < 0.001 (pingouin pattern).
- Return `(corr_df, pvalue_df, (fig, ax))` or `(corr_df, pvalue_df, None)` if `plot=False`.

**Reference**: pingouin's `rcorr()` method does exactly this — correlation matrix with p-values rendered as upper/lower triangle. Adopt the asterisk notation.

**Tests**: Verify diagonal of corr_df is 1.0. Verify diagonal of pvalue_df is 0.0 (or NaN). Verify symmetry. Verify 2-column case.

### Task 2.6: `ljung_box()` method

**Signature**:
```python
def ljung_box(
    self,
    lags: int = 10,
    column: str = None,
) -> DataFrame:
```

**What it does**: Test whether autocorrelations are significantly different from zero (white noise test).

**Implementation**:
- Wrap `statsmodels.stats.diagnostic.acorr_ljungbox(data, lags=lags, return_df=True)`.
- Returns DataFrame with columns: `lb_stat`, `lb_pvalue` (per lag).
- For multi-column: stack results with a column-name level.

**Tests**: White noise: p-values should be > 0.05. AR(1) with phi=0.9: low p-values.

---

## Phase 3 — Stationarity Testing

**Reference implementations**: statsmodels `adfuller` / `kpss`, darts `stationarity_tests()`.

### Task 3.1: `adf_test()` method

**Signature**:
```python
def adf_test(
    self,
    regression: str = "c",
    autolag: str = "AIC",
    column: str = None,
) -> DataFrame:
```

**What it does**: Augmented Dickey-Fuller test for unit root. Null = non-stationary.

**Implementation**:
- Wrap `statsmodels.tsa.stattools.adfuller(data, maxlag=None, regression=regression, autolag=autolag)`.
- adfuller returns: `(adf_stat, pvalue, usedlag, nobs, critical_values_dict, icbest)`.
- `regression` options: `"c"` (constant), `"ct"` (constant + trend), `"ctt"` (constant + trend + trend^2), `"n"` (none).
- **Per-column output** DataFrame with columns: `statistic`, `p_value`, `used_lag`, `n_obs`, `crit_1%`, `crit_5%`, `crit_10%`, `conclusion`.
- `conclusion`: "Stationary" if p_value < 0.05 (or user's alpha), "Non-stationary" otherwise.

**Tests**:
- White noise: should reject null (p < 0.05) — stationary.
- Random walk `cumsum(randn)`: should NOT reject null — non-stationary.
- Verify critical values are populated.

### Task 3.2: `kpss_test()` method

**Signature**:
```python
def kpss_test(
    self,
    regression: str = "c",
    nlags: str = "auto",
    column: str = None,
) -> DataFrame:
```

**What it does**: KPSS test for stationarity. Null = stationary. **Opposite of ADF.**

**Implementation**:
- Wrap `statsmodels.tsa.stattools.kpss(data, regression=regression, nlags=nlags)`.
- kpss returns: `(kpss_stat, p_value, lags, crit_dict)`.
- `regression`: `"c"` (level stationarity) or `"ct"` (trend stationarity).
- **Per-column output** DataFrame: `statistic`, `p_value`, `lags`, `crit_10%`, `crit_5%`, `crit_2.5%`, `crit_1%`, `conclusion`.
- `conclusion`: "Stationary" if p_value > 0.05, "Non-stationary" if p_value < 0.05.
- **Note**: statsmodels issues a warning for p-values outside the lookup table bounds. Suppress and report boundary values.

**Tests**: Same data as ADF tests, verify opposite interpretation logic.

### Task 3.3: `stationarity_summary()` method

**Signature**:
```python
def stationarity_summary(self, alpha: float = 0.05) -> DataFrame:
```

**What it does**: Run BOTH ADF and KPSS, combine into a diagnosis. This is what researchers actually want (darts pattern).

**Implementation**:
- Call `self.adf_test()` and `self.kpss_test()`.
- Per column, produce diagnosis:

| ADF rejects? | KPSS rejects? | Diagnosis |
|---|---|---|
| Yes | No | **Stationary** |
| No | Yes | **Non-stationary (unit root)** |
| Yes | Yes | **Trend-stationary** (stationary around a deterministic trend) |
| No | No | **Inconclusive** (possibly difference-stationary) |

- Return DataFrame: `adf_stat`, `adf_pvalue`, `kpss_stat`, `kpss_pvalue`, `diagnosis`.

**Tests**: Verify all four diagnosis cases with synthetic data.

---

## Phase 4 — Trend Detection

**Reference implementations**: pymannkendall (algorithms + return structure), scipy.stats.theilslopes.

### Task 4.1: `mann_kendall()` method

**Signature**:
```python
def mann_kendall(
    self,
    alpha: float = 0.05,
    method: str = "hamed_rao",
    lag: int = None,
    column: str = None,
) -> DataFrame:
```

**What it does**: Mann-Kendall trend test with autocorrelation correction.

**Available methods** (matching pymannkendall names for familiarity):
- `"original"`: Standard MK. Assumes serial independence.
- `"hamed_rao"`: Hamed & Rao (1998) variance correction for autocorrelation. **Default** because naively ignoring autocorrelation inflates significance in hydrological data.
- `"yue_wang"`: Yue & Wang (2004) alternative correction.
- `"pre_whitening"`: Remove lag-1 autocorrelation before testing.
- `"trend_free_pre_whitening"`: Remove trend via Sen slope, pre-whiten residuals, re-add trend, then test.

**Implementation from scratch** (following pymannkendall algorithms):

1. **S statistic**: `S = sum(sgn(x_j - x_i))` for all j > i. Use vectorized inner comparison with loop over k (pymannkendall pattern). O(n^2).

2. **Variance with tie correction**:
   ```
   Var(S) = [n(n-1)(2n+5) - sum(tp*(tp-1)*(2*tp+5))] / 18
   ```
   where `tp` are tie group sizes.

3. **Z statistic with continuity correction**:
   ```
   Z = (S - 1) / sqrt(Var(S))    if S > 0
   Z = (S + 1) / sqrt(Var(S))    if S < 0
   Z = 0                          if S == 0
   ```

4. **Hamed-Rao correction**: Detrend with Sen slope. Compute ACF on ranked residuals. Compute variance correction factor:
   ```
   n/s = 1 + (2 / (n(n-1)(n-2))) * sum((n-i)(n-i-1)(n-i-2) * rho(i))
   ```
   Only use significant autocorrelation coefficients.

5. **Yue-Wang correction**: Like Hamed-Rao but ACF on raw detrended values (not ranked), and:
   ```
   n/s = 1 + (2/n) * sum((n-i) * rho(i))
   ```

6. **Pre-whitening**: `x_new[t] = x[t+1] - acf1 * x[t]`, then apply original test.

7. **Trend-free pre-whitening**: (a) Remove trend via Sen slope, (b) pre-whiten residuals, (c) re-add trend, (d) apply original test.

8. **p-value**: Two-tailed from `scipy.stats.norm.cdf`.

9. **Tau**: `Tau = S / (n*(n-1)/2)` adjusted for ties.

10. **Slope and intercept**: Compute via Sen's slope (Task 4.2).

**Return** DataFrame per column: `trend`, `h`, `p_value`, `z`, `tau`, `s`, `var_s`, `slope`, `intercept`.

**Tests**:
- Linearly increasing data: trend = "increasing", h = True.
- Constant data: trend = "no trend".
- Random data: h should be False (mostly).
- Compare output against pymannkendall for the same input.
- Verify Hamed-Rao gives different (usually larger) p-value than original for autocorrelated data.

### Task 4.2: `sens_slope()` method

**Signature**:
```python
def sens_slope(
    self,
    alpha: float = 0.05,
    column: str = None,
) -> DataFrame:
```

**What it does**: Sen's slope estimator — median of all pairwise slopes.

**Implementation**:
- Use `scipy.stats.theilslopes(data, alpha=alpha)` which returns `(slope, intercept, low_slope, high_slope)`.
- Return DataFrame per column: `slope`, `intercept`, `slope_lower_ci`, `slope_upper_ci`.
- CI is based on the method in Gilbert (1987) / Conover (1999).

**Tests**: Verify slope for `y = 2x + noise` is approximately 2. Verify CI contains true slope.

### Task 4.3: `detrend()` method

**Signature**:
```python
def detrend(self, method: str = "linear", order: int = 1) -> "TimeSeries":
```

**What it does**: Remove trend and return detrended series.

**Implementation**:
- `"linear"`: `scipy.signal.detrend(data, type='linear')`. Supports `bp` parameter for piecewise linear.
- `"constant"`: `scipy.signal.detrend(data, type='constant')` (subtract mean).
- `"polynomial"`: Fit polynomial of given `order`, subtract.
- `"sens"`: Remove trend using `self.sens_slope()` — robust to outliers.
- Return new `TimeSeries` with same index.

**Tests**: Linear trend data: detrended series should have near-zero slope. Verify index is preserved.

### Task 4.4: `innovative_trend_analysis()` method

**Signature**:
```python
def innovative_trend_analysis(
    self,
    column: str = None,
    **kwargs,
) -> Tuple[DataFrame, Tuple[Figure, Axes]]:
```

**What it does**: Sen (2012) ITA method. Split sorted data into two halves, scatter plot first half vs second half.

**Implementation**:
- Sort data ascending. Split at midpoint.
- Plot first half (x-axis) vs second half (y-axis).
- Add 1:1 line and +/-10% envelope lines.
- Points above 1:1 = increasing trend. Below = decreasing.
- Compute trend indicator: slope of best-fit line through the scatter minus 1.

**Tests**: Increasing data: points above 1:1 line. Verify trend indicator sign.

---

## Phase 5 — Change Point Detection

**Reference implementations**: pyhomogeneity (algorithms, named tuple returns, Monte Carlo p-values).

### Task 5.1: `pettitt_test()` method

**Signature**:
```python
def pettitt_test(
    self,
    alpha: float = 0.05,
    sim: int = 20000,
    seed: int = None,
    column: str = None,
) -> DataFrame:
```

**What it does**: Non-parametric test for single change point in the mean.

**Implementation** (following pyhomogeneity):
1. Compute ranks: `r = scipy.stats.rankdata(x)`.
2. Cumulative rank sum: `S_k = cumsum(r)[:-1]`.
3. Test statistic: `U_k = 2 * S_k - (k+1) * (n+1)` for k = 1..n-1.
4. Change point at `k*` where `|U_k|` is maximized. `K = max|U_k|`.
5. **P-value** (two options):
   - Approximate: `p = 2 * exp(-6K^2 / (n^3 + n^2))`.
   - Monte Carlo (if `sim > 0`): Generate `sim` random normal series of length n, compute K for each, `p = count(K_random > K_observed) / sim`.
6. `seed` parameter for reproducible Monte Carlo (use `np.random.default_rng(seed)`).
7. Compute `mean_before`, `mean_after` at change point.

**Return**: DataFrame per column: `h`, `change_point_index`, `change_point_date`, `statistic`, `p_value`, `mean_before`, `mean_after`.

**Tests**:
- Concatenate two normals with different means: should detect change point at the junction.
- Homogeneous data: h = False.
- Verify change point index is correct.

### Task 5.2: `snht_test()` method

**Signature**:
```python
def snht_test(
    self,
    alpha: float = 0.05,
    sim: int = 20000,
    seed: int = None,
    column: str = None,
) -> DataFrame:
```

**What it does**: Standard Normal Homogeneity Test (Alexandersson 1986).

**Implementation** (following pyhomogeneity):
1. Standardize: `z = (x - mean) / std`.
2. For each t = 1..n-1:
   - `z1 = mean(z[:t])`, `z2 = mean(z[t:])`.
   - `T_t = t * z1^2 + (n-t) * z2^2`.
3. Change point at `t*` where `T_t` is maximized. `T0 = max(T_t)`.
4. Monte Carlo p-value same as Pettitt.

**Return**: Same structure as pettitt_test.

### Task 5.3: `buishand_range_test()` method

**Signature**:
```python
def buishand_range_test(
    self,
    alpha: float = 0.05,
    sim: int = 20000,
    seed: int = None,
    column: str = None,
) -> DataFrame:
```

**Implementation** (following pyhomogeneity):
1. Adjusted partial sums: `S_k = sum(x_i - mean) for i=1..k`.
2. Standardize: `S_k* = S_k / std`.
3. Range statistic: `R = (max(S_k*) - min(S_k*)) / sqrt(n)`.
4. Change point at k* where `|S_k*|` is maximized.
5. Monte Carlo p-value.

### Task 5.4: `cusum()` method

**Signature**:
```python
def cusum(
    self,
    column: str = None,
    plot: bool = True,
    **kwargs,
) -> Tuple[DataFrame, Optional[Tuple[Figure, Axes]]]:
```

**What it does**: Cumulative sum of deviations from mean. Visual + analytical.

**Implementation**:
- `S_t = cumsum(x - mean(x))` for t = 1..n.
- **Plot**: CUSUM line with confidence bounds (`+/- 1.96 * std * sqrt(t)` under null of no change). Mark detected change point with vertical line.
- Return CUSUM values as DataFrame.

### Task 5.5: `homogeneity_summary()` method

**Signature**:
```python
def homogeneity_summary(self, alpha: float = 0.05, sim: int = 20000, seed: int = None) -> DataFrame:
```

**What it does**: Run Pettitt + SNHT + Buishand on each column. Combined diagnosis.

**Return**: DataFrame per column: `pettitt_cp`, `pettitt_p`, `snht_cp`, `snht_p`, `buishand_cp`, `buishand_p`, `confirmed` (True if 2+ tests agree on change point location within +/-2 indices).

---

## Phase 6 — Decomposition & Smoothing

**Reference implementations**: statsmodels `STL`/`seasonal_decompose`, scipy.signal filters, statsmodels `lowess`.

### Task 6.1: `stl()` method

**Signature**:
```python
def stl(
    self,
    period: int = None,
    seasonal: int = 7,
    robust: bool = False,
    column: str = None,
    plot: bool = True,
    **kwargs,
) -> Tuple[DataFrame, Optional[Tuple[Figure, Axes]]]:
```

**Implementation**:
- Wrap `statsmodels.tsa.seasonal.STL(data, period=period, seasonal=seasonal, robust=robust).fit()`.
- `period`: Auto-detect from DatetimeIndex frequency if possible. Raise error if None and cannot detect.
- `seasonal`: Smoother length, must be odd >= 7 (statsmodels requirement).
- `robust`: If True, use outlier-resistant fitting (LOESS with bi-square weights).
- Result has `.trend`, `.seasonal`, `.resid`, `.weights` attributes.
- **Plot**: 4-panel standard STL layout (observed, trend, seasonal, residual).
- **Return**: DataFrame with columns `observed`, `trend`, `seasonal`, `residual`.

### Task 6.2: `classical_decompose()` method

**Signature**:
```python
def classical_decompose(
    self,
    period: int = None,
    model: str = "additive",
    column: str = None,
    plot: bool = True,
    **kwargs,
) -> Tuple[DataFrame, Optional[Tuple[Figure, Axes]]]:
```

**Implementation**:
- Wrap `statsmodels.tsa.seasonal.seasonal_decompose(data, model=model, period=period)`.
- `model`: `"additive"` (Y = T + S + e) or `"multiplicative"` (Y = T * S * e).
- Same return pattern as `stl()`.

### Task 6.3: `smooth()` method

**Signature**:
```python
def smooth(
    self,
    method: str = "moving_average",
    window: int = 10,
    **params,
) -> "TimeSeries":
```

**Methods and their backends**:

| Method | Backend | Key params |
|---|---|---|
| `"moving_average"` | `self.rolling(window, center=True).mean()` | — |
| `"exponential"` | `self.ewm(span=window).mean()` | — |
| `"savgol"` | `scipy.signal.savgol_filter(data, window_length=window, polyorder=params.get("polyorder", 2))` | `polyorder` (default 2) |
| `"lowess"` | `statsmodels.nonparametric.lowess(data, index, frac=params.get("frac", 0.1))` | `frac` (default 0.1) |

- Returns new `TimeSeries` with smoothed values. Same index as original.
- Savitzky-Golay preserves peaks better than moving average — recommended for hydrograph smoothing.
- `window` must be odd for savgol; auto-adjust if even.

### Task 6.4: `envelope()` method

**Signature**:
```python
def envelope(
    self,
    window: int = 30,
    lower_pct: float = 5,
    upper_pct: float = 95,
    column: str = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
```

**Implementation**:
- Compute rolling percentile bands: `self.rolling(window).quantile(lower_pct/100)` and `self.rolling(window).quantile(upper_pct/100)`.
- Plot: original series line + shaded band between lower and upper.
- Practical use: show natural variability range, identify excursions beyond normal.

---

## Phase 7 — Missing Data & Quality Control

### Task 7.1: `missing_summary()` method

**Signature**:
```python
def missing_summary(self) -> DataFrame:
```

**Return** per column: `total_count`, `missing_count`, `missing_pct`, `valid_count`, `longest_gap`, `n_gaps`, `mean_gap_length`, `first_valid`, `last_valid`.

**Implementation**:
- `missing_count`: `self[col].isna().sum()`.
- `longest_gap` and `n_gaps`: Use run-length encoding on the boolean NaN mask. Group consecutive NaNs.
- `first_valid` / `last_valid`: `self[col].first_valid_index()` / `last_valid_index()`.

### Task 7.2: `gap_analysis()` method

**Signature**:
```python
def gap_analysis(self, column: str = None) -> DataFrame:
```

**Return**: DataFrame with rows = individual gaps. Columns: `column`, `gap_start`, `gap_end`, `gap_length`, `gap_duration` (timedelta if DatetimeIndex). Sorted by gap_length descending.

**Implementation**: Iterate over NaN mask, find contiguous runs of True. Record start/end indices and lengths.

### Task 7.3: `completeness_report()` method

**Signature**:
```python
def completeness_report(self, freq: str = "YE") -> DataFrame:
```

**Return**: DataFrame with index = period, columns = series names, values = completeness percentage (0-100).

**Implementation**: `self.resample(freq).apply(lambda x: x.notna().mean() * 100)`.

### Task 7.4: `detect_outliers()` method

**Signature**:
```python
def detect_outliers(
    self,
    method: str = "iqr",
    threshold: float = 1.5,
    column: str = None,
) -> DataFrame:
```

**Methods**:

| Method | Logic | Default threshold |
|---|---|---|
| `"iqr"` | Outside [Q1 - threshold*IQR, Q3 + threshold*IQR] | 1.5 (standard), 3.0 (far) |
| `"zscore"` | `abs((x - mean) / std) > threshold` | 3.0 |
| `"modified_zscore"` | `abs(0.6745 * (x - median) / MAD) > threshold` | 3.5 |

- Return boolean DataFrame (same shape as input, `True` = outlier).
- Modified Z-score (Iglewicz & Hoaglin, 1993) is more robust — uses median and MAD instead of mean and std.

### Task 7.5: `outlier_plot()` method

**Signature**:
```python
def outlier_plot(
    self,
    method: str = "iqr",
    threshold: float = 1.5,
    column: str = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
```

**Implementation**: Time series line plot with outlier points highlighted in red. Show threshold lines if applicable (for IQR: horizontal lines at Q1-1.5*IQR and Q3+1.5*IQR).

---

## Phase 8 — Distribution-Aware Methods

### Task 8.1: `qq_plot()` method

**Signature**:
```python
def qq_plot(
    self,
    distribution: str = "norm",
    column: str = None,
    confidence: float = 0.95,
    **kwargs,
) -> Tuple[Figure, Axes]:
```

**Implementation**:
- Use `scipy.stats.probplot(data, dist=distribution, plot=None)` to get theoretical quantiles and ordered values.
- Plot with 1:1 reference line. Add confidence envelope using order statistic medians.
- Accept any `scipy.stats` distribution name.
- For multi-column with no column specified: subplot grid.
- **Reference**: pingouin's `qqplot()` has a clean `confidence` parameter and `sparams` for distribution shape.

### Task 8.2: `pp_plot()` method

**Signature**:
```python
def pp_plot(
    self,
    distribution: str = "norm",
    column: str = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
```

**Implementation**: Plot empirical CDF vs theoretical CDF at each data point. Points near the 1:1 line = good fit.

### Task 8.3: `normality_test()` method

**Signature**:
```python
def normality_test(
    self,
    method: str = "auto",
    alpha: float = 0.05,
) -> DataFrame:
```

**Methods**:

| Method | Backend | Best for |
|---|---|---|
| `"auto"` | Shapiro-Wilk if n < 5000, D'Agostino if n >= 5000 | General use |
| `"shapiro"` | `scipy.stats.shapiro` | Small-moderate samples (n < 5000) |
| `"dagostino"` | `scipy.stats.normaltest` | Large samples |
| `"anderson"` | `scipy.stats.anderson` | Multiple significance levels |
| `"lilliefors"` | `statsmodels.stats.diagnostic.lilliefors` | KS with estimated params |
| `"jarque_bera"` | `scipy.stats.jarque_bera` | Based on skewness + kurtosis |

**Return** per column: `test_name`, `statistic`, `p_value`, `is_normal` (bool), `conclusion`.

### Task 8.4: `empirical_cdf()` method

**Signature**:
```python
def empirical_cdf(
    self,
    column: str = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
```

**Implementation**: Step-function plot using `np.sort(data)` and `np.arange(1, n+1) / n`. No bandwidth choice needed (simpler than KDE).

### Task 8.5: `fit_distributions()` method

**Signature**:
```python
def fit_distributions(self, method: str = "lmoments") -> DataFrame:
```

**Implementation**: Apply `statista.distributions.Distributions(data=col_data).best_fit()` per column. Return: `column`, `best_distribution`, `loc`, `scale`, `shape`, `ks_statistic`, `ks_p_value`.

---

## Phase 9 — Seasonal & Periodic Analysis

### Task 9.1: `seasonal_subseries()` — Plot each season as mini time series
### Task 9.2: `annual_cycle()` — Overlay all years on Jan-Dec axis with envelope
### Task 9.3: `monthly_stats()` — Grouped stats per month, returned as DataFrame
### Task 9.4: `periodogram()` — Wrap `scipy.signal.periodogram` or `welch`, annotate dominant peaks
### Task 9.5: `seasonal_mann_kendall()` — Per-season MK combined via Hirsch et al. (1982)

---

## Phase 10 — Comparison & Anomaly

### Task 10.1: `anomaly()` — Deviation from mean/median/climatology, colored bar plot
### Task 10.2: `standardized_anomaly()` — (x - seasonal_mean) / seasonal_std per month
### Task 10.3: `double_mass_curve()` — Cumulative X vs cumulative Y, detect slope breaks
### Task 10.4: `regime_comparison()` — Split at change point, compare before/after stats

---

## Phase 11 — Hydrological Methods

**Reference implementations**: hydrosignatures, baseflow package, hydrobox, hydrotoolbox.

### Task 11.1: `flow_duration_curve()` method

**Signature**:
```python
def flow_duration_curve(
    self,
    log_scale: bool = True,
    method: str = "weibull",
    column: str = None,
    **kwargs,
) -> Tuple[DataFrame, Tuple[Figure, Axes]]:
```

**Implementation**:
- Sort values descending. Compute exceedance probability using plotting position formula.
- `method`: `"weibull"` (i/(n+1)), `"gringorten"` ((i-0.44)/(n+0.12)), `"cunnane"` ((i-0.4)/(n+0.2)).
- Plot: y-axis = flow (log-scale by default), x-axis = exceedance probability (0-100%).
- Annotate Q10, Q50, Q90, Q95 with horizontal dashed lines and labels.
- Multi-column: overlay all series on same plot.
- Return DataFrame: `value`, `exceedance_pct`, `return_period`.
- **This is the single most used plot in hydrology** (hydrobox, hydrosignatures both implement it).

### Task 11.2: `annual_extremes()` — Extract annual max/min with water year config
### Task 11.3: `exceedance_probability()` — Empirical exceedance with multiple plotting positions

### Task 11.4: `baseflow_separation()` method

**Signature**:
```python
def baseflow_separation(
    self,
    method: str = "lyne_hollick",
    alpha: float = 0.925,
    column: str = None,
    plot: bool = True,
    **kwargs,
) -> Tuple["TimeSeries", Optional[Tuple[Figure, Axes]]]:
```

**Methods** (from baseflow package research):

| Method | Formula | Params |
|---|---|---|
| `"lyne_hollick"` | `b_t = alpha * b_{t-1} + (1-alpha)/2 * (q_t + q_{t-1})` | `alpha` (0.925) |
| `"eckhardt"` | `b_t = ((1-BFI_max)*k*b_{t-1} + (1-k)*BFI_max*q_t) / (1-k*BFI_max)` | `alpha` (k), `bfi_max` (0.80) |
| `"chapman_maxwell"` | `b_t = k/(2-k)*b_{t-1} + (1-k)/(2-k)*q_t` | `alpha` (k) |

- Enforce `b_t <= q_t` and `b_t >= 0` at each step.
- Return `TimeSeries` with columns: `total_flow`, `baseflow`, `quickflow`.
- Plot: hydrograph with baseflow area shaded.

### Task 11.5: `baseflow_index()` — BFI = sum(baseflow) / sum(total_flow), scalar per column
### Task 11.6: `flashiness_index()` — Richards-Baker flashiness = sum(|Q_t - Q_{t-1}|) / sum(Q_t)
### Task 11.7: `recession_analysis()` — Extract recessions, fit dQ/dt = -aQ^b

---

## Implementation Order

| Order | Phase           | Tasks                                                                     | Dependencies           | Estimated Methods  |
|-------|-----------------|---------------------------------------------------------------------------|------------------------|--------------------|
| 0     | Structural      | 0.1-0.3: stat_result.py, subpackage split, mixin wiring                   | None                   | 0 (infrastructure) |
| 1     | Descriptive     | 1.2-1.3: l_moments, summary                                               | Phase 0                | 2                  |
| 2     | Missing Data    | 7.1-7.5: missing_summary, gaps, outliers                                  | Phase 0                | 5                  |
| 3     | Autocorrelation | 2.1-2.6: acf, pacf, ccf, lag_plot, corr_matrix, ljung_box                 | Phase 0                | 6                  |
| 4     | Stationarity    | 3.1-3.3: adf, kpss, stationarity_summary                                  | Phase 0                | 3                  |
| 5     | Trend           | 4.1-4.4: mann_kendall, sens_slope, detrend, ITA                           | Phase 0 + stat_result  | 4                  |
| 6     | Distribution    | 8.1-8.5: qq_plot, pp_plot, normality, ecdf, fit_distributions             | Phase 0                | 5                  |
| 7     | Change Point    | 5.1-5.5: pettitt, snht, buishand, cusum, homogeneity_summary              | Phase 0 + stat_result  | 5                  |
| 8     | Decomposition   | 6.1-6.4: stl, classical_decompose, smooth, envelope                       | Phase 0                | 4                  |
| 9     | Seasonal        | 9.1-9.5: subseries, annual_cycle, monthly_stats, periodogram, seasonal_mk | Phase 5 (MK)           | 5                  |
| 10    | Hydrological    | 11.1-11.7: FDC, annual_extremes, baseflow, BFI, flashiness, recession     | Phase 0                | 7                  |
| 11    | Comparison      | 10.1-10.4: anomaly, standardized_anomaly, double_mass, regime             | Phase 7 (change point) | 4                  |
| —     | Summary update  | 1.3 revisit: add normality + ADF p-values to summary()                    | Phases 4 + 6           | 0 (update)         |

**Total new methods: 50** (from 8 to 58).

---

## Testing Strategy

- Each phase gets its own test file: `tests/test_ts_descriptive.py`, `tests/test_ts_correlation.py`, etc.
- Use data from `examples/data/` for realistic tests: `temp.csv`, `rhine-full-time-series.csv`, `rees.csv`, `time_series1.txt`, `time_series2.txt`.
- Synthetic data for known-answer tests: white noise, AR(1), linear trend, step change, sinusoidal.
- **Cross-validate against reference packages**: Compare MK output against pymannkendall, homogeneity output against pyhomogeneity, ADF/KPSS against statsmodels raw calls.
- Mark fast unit tests with `@pytest.mark.fast`, slow Monte Carlo tests with `@pytest.mark.slow`.
- Test both 1D and 2D inputs for every method.
- Test NaN handling explicitly.
