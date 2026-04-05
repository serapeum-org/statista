# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- **NEVER install packages directly** with `pip install`, `uv pip install`, or `uv sync`. All dependencies are already installed. Do NOT attempt to install or sync anything.
- **NEVER create a `.venv` directory**. The virtual environment is external at `C:\python-environments\uv\statista` and managed outside this repo.
- **NEVER run bare `python`, `pytest`, `mypy`, etc.** Always prefix with `uv run --active` (e.g., `uv run --active pytest`, `uv run --active mypy`).
- **Set `VIRTUAL_ENV` before running commands.** The shell must have the env activated:
  ```bash
  export VIRTUAL_ENV="C:/python-environments/uv/statista"
  ```

## Project Overview

**Statista** is a Python statistical analysis package for probability distributions, extreme value analysis, and sensitivity analysis. Used primarily in hydrology, climate science, and risk assessment. Part of the [serapeum-org](https://github.com/serapeum-org) project family.

## Development Commands

```bash
# Install (uv doesn't manage the Python install; uv.managed = false)
uv sync --active                    # runtime deps
uv sync --active --group dev        # + dev deps
uv sync --active --group docs       # + docs deps

# Tests
uv run --active pytest                                          # all tests
uv run --active pytest tests/test_distributions.py              # one file
uv run --active pytest tests/test_eva.py::test_specific -vvv    # one test, verbose
uv run --active pytest -m "not slow"                            # skip slow tests

# CI runs this exact command:
uv run --active pytest -vvv --cov=src/statista --cov-report=xml --junitxml=junit.xml

# Linting & formatting
uv run --active black src/ tests/
uv run --active isort src/ tests/
uv run --active mypy src/statista

# Docs
uv run --active mkdocs serve

# Build
uv build
```

## Architecture

Source lives in `src/statista/`. The central module is `distributions.py`.

### Distribution Class Hierarchy

```
AbstractDistribution(ABC)        # Abstract base: pdf(), cdf(), fit_model(), ks(), chisquare(), etc.
  ├── Gumbel                     # 2-param: loc, scale
  ├── GEV                        # 3-param: loc, scale, shape
  ├── Exponential                # 2-param: loc, scale
  └── Normal                     # 2-param: loc, scale

Distributions                    # Facade/factory: Distributions("GEV", data) → delegates to GEV class
PlottingPosition                 # Static utility: weibul(), return_period()
```

**`Distributions`** is the main entry point. It accepts a distribution name string and delegates via `__getattr__()` to the matching subclass. The `available_distributions` dict maps names to classes.

### Parameter Estimation

All distributions support three fitting methods via `fit_model(method=...)`:
- `"mle"` / `"mm"` — delegates to `scipy.stats` distribution `.fit()`
- `"lmoments"` — uses `Lmoments` class from `parameters.py` (more robust to outliers)
- `"optimization"` — custom objective via `scipy.optimize.fmin()`, supports truncated/threshold fitting

Parameters are stored as dicts: `{"loc": ..., "scale": ...}` (2-param) or `{"loc": ..., "scale": ..., "shape": ...}` (3-param).

### Module Dependency Flow

```
distributions.py → confidence_interval.py (ConfidenceInterval.boot_strap)
                 → parameters.py          (Lmoments estimation)
                 → plot.py                (Plot static methods for visualization)
                 → utils.py               (merge_small_bins helper)

eva.py           → distributions.py       (uses Distributions facade for AMS analysis)

sensitivity.py   → standalone (numpy/pandas/matplotlib only)
descriptors.py   → standalone (numpy + sklearn.metrics for r2_score)
```

### Plotting Integration

Most methods accept `plot_figure=True` which adds `(Figure, Axes)` to the return tuple. The `Plot` class in `plot.py` provides static methods (`Plot.pdf()`, `Plot.cdf()`, `Plot.details()`) used internally.

## Commit Conventions

Uses **commitizen** with a custom schema. Allowed types: `feat`, `fix`, `refactor`, `perf`, `build`, `ci`, `chore`.

Format: `<type>: <description>` (e.g., `feat: add gamma distribution support`)

Pattern enforced: `(feat|fix|refactor|perf|build|ci|chore)(\(.+\))?: (.+)`

## Testing

- Markers: `slow`, `fast`, `e2e`, `unit`, `integration`
- `test_eva.py` tests are marked `@pytest.mark.slow`
- Fixtures and test data defined in `conftest.py` (distribution parameters, time series, expected outputs)
- CI matrix: Python 3.11, 3.12, 3.13 on ubuntu-latest

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`: flake8, isort, black, bandit (security), gitleaks, detect-secrets, conventional commit validation, no-commit-to-branch (main), and full pytest suite.

## Code Style

- Line length limit: **120 characters** in all files (code, tests, configs).
  Break lines that exceed 120 characters.
- **Single return per function.** Do not use multiple return statements.
  Use a result variable and return it once at the end.
