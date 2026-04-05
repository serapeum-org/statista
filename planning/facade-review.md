# Architectural Review — `Distributions.fit_all` / `best_fit`

## 1. Replace inner dict with a `FitResult` dataclass

**Problem**: `fit_all` returns `dict[str, dict[str, Any]]`. Callers access results via string keys and positional tuple indices:

```python
results["Gumbel"]["ks"][1]  # what is [1]? statistic? p-value?
results["Gumbel"]["parameters"]  # no IDE autocomplete, no type safety
```

**Fix**: Introduce a `FitResult` dataclass with a nested `GoodnessOfFit` dataclass:

```python
@dataclass
class GoodnessOfFit:
    statistic: float
    p_value: float

@dataclass
class FitResult:
    distribution: AbstractDistribution
    parameters: dict[str, float]
    ks: GoodnessOfFit
    chisquare: GoodnessOfFit
```

After:

```python
results["Gumbel"].ks.p_value       # self-documenting
results["Gumbel"].parameters       # IDE autocomplete works
```

**Where**: New file `src/statista/distributions/fit_result.py`, imported by `facade.py`.

---

## 2. Cache `fit_all` results — remove `fit_results` parameter from `best_fit`

**Problem**: `best_fit` has a `fit_results` parameter so users can avoid re-fitting. This is an optimization concern leaking into the API. The user must shuttle data between methods:

```python
results = dist.fit_all()
best_name, best_info = dist.best_fit(fit_results=results)  # manual wiring
```

**Fix**: `fit_all` stores results on `self._fit_results`. `best_fit` uses the cache automatically. Drop the `fit_results` parameter.

```python
def fit_all(self, ...):
    ...
    self._fit_results = results
    return results

def best_fit(self, ...):
    if self._fit_results is None:
        self.fit_all(method=method, distributions=distributions)
    # select from self._fit_results
```

After:

```python
dist.fit_all()
best_name, best_info = dist.best_fit()  # just works
# or skip fit_all entirely:
best_name, best_info = dist.best_fit()  # calls fit_all internally
```

---

## 3. Add a `summary` property — DataFrame comparison view

**Problem**: After `fit_all`, there is no quick way to compare distributions side-by-side. The user must manually loop through the nested dict.

**Fix**: Add a `summary` property that returns a `pandas.DataFrame` from cached `_fit_results`:

```python
@property
def summary(self) -> pd.DataFrame:
    # columns: distribution, loc, scale, shape, ks_statistic, ks_p_value,
    #          chisquare_statistic, chisquare_p_value
```

After:

```python
dist = Distributions(data=data)
dist.fit_all()
print(dist.summary)
#              loc    scale   shape  ks_stat  ks_pval  chi_stat  chi_pval
# GEV       463.8  220.072  0.0101   0.0741   0.9987      1.23    0.942
# Gumbel    463.8  220.072     NaN   0.0741   0.9987      1.15    0.950
# ...
```

---

## 4. Simplify p-value selection in `best_fit`

**Problem**: The selection loop uses a sentinel value and manual tracking:

```python
best_name = None
best_pvalue = -1.0
for name, info in fit_results.items():
    pvalue = info[criterion][1]
    if pvalue > best_pvalue:
        best_pvalue = pvalue
        best_name = name
```

**Fix**: Use `max()` with a key function:

```python
best_name = max(
    self._fit_results,
    key=lambda name: self._fit_results[name].ks.p_value
    if criterion == "ks"
    else self._fit_results[name].chisquare.p_value,
)
```

Or with the dataclass approach using `getattr`:

```python
best_name = max(
    self._fit_results,
    key=lambda name: getattr(self._fit_results[name], criterion).p_value,
)
```

---

## Implementation Order

1. Create `GoodnessOfFit` and `FitResult` dataclasses
2. Refactor `fit_all` to return `dict[str, FitResult]` and cache on `self._fit_results`
3. Refactor `best_fit` to use cache and `max()` — drop `fit_results` parameter
4. Add `summary` property
5. Update tests and docstring examples
