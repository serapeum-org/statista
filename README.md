# Statista - Advanced Statistical Analysis Package

[![Python Versions](https://img.shields.io/pypi/pyversions/statista.svg)](https://pypi.org/project/statista/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://serapieum-of-alex.github.io/statista/latest/)
[![codecov](https://codecov.io/gh/Serapieum-of-alex/statista/branch/main/graph/badge.svg?token=GQKhcj2pFK)](https://codecov.io/gh/Serapieum-of-alex/statista)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![GitHub last commit](https://img.shields.io/github/last-commit/Serapieum-of-alex/statista)](https://github.com/Serapieum-of-alex/statista/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/Serapieum-of-alex/statista)](https://github.com/Serapieum-of-alex/statista/issues)
[![GitHub stars](https://img.shields.io/github/stars/Serapieum-of-alex/statista)](https://github.com/Serapieum-of-alex/statista/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Serapieum-of-alex/statista)](https://github.com/Serapieum-of-alex/statista/network/members)

## Overview

**Statista** is a comprehensive Python package for statistical analysis, focusing on probability distributions, extreme value analysis, and sensitivity analysis. It provides robust tools for researchers, engineers, and data scientists working with statistical models, particularly in hydrology, climate science, and risk assessment.

Current release info
====================

| Name                                                                                                                      | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Version                                                                                                                                                                                                                                                                                                                                                 | Platforms                                                                                                                                                                                                                                                                                                                                 |
|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Conda Recipe](https://img.<br/>shields.io/badge/recipe-statista-green.svg)](https://anaconda.org/conda-forge/statista) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/statista.svg)](https://anaconda.org/conda-forge/statista) [![Downloads](https://pepy.tech/badge/statista)](https://pepy.tech/project/statista) [![Downloads](https://pepy.tech/badge/statista/month)](https://pepy.tech/project/statista)  [![Downloads](https://pepy.tech/badge/statista/week)](https://pepy.tech/project/statista)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/statista?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/statista.svg)](https://anaconda.org/conda-forge/statista) [![PyPI version](https://badge.fury.io/py/statista.svg)](https://badge.fury.io/py/statista) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/statista/badges/version.svg)](https://anaconda.org/conda-forge/statista) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/statista.svg)](https://anaconda.org/conda-forge/statista) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |


conda-forge feedstock
=====================
[Conda-forge feedstock](https://github.com/conda-forge/statista-feedstock)

## Installation

### Conda (Recommended)

```bash
conda install -c conda-forge statista
```

### PyPI

```bash
pip install statista
```

### Development Version

```bash
pip install git+https://github.com/Serapieum-of-alex/statista
```

## Main Features

### Statistical Distributions
- **Probability Distributions**: GEV, Gumbel, Normal, Exponential, and more
- **Multi-Distribution Fitting**: Fit all distributions at once and select the best fit
- **Parameter Estimation Methods**: Maximum Likelihood (ML), L-moments, Method of Moments (MOM)
- **Goodness-of-fit Tests**: Kolmogorov-Smirnov, Chi-square
- **Truncated Distributions**: Focus analysis on values above a threshold

### Extreme Value Analysis
- **Return Period Calculation**: Estimate extreme events for different return periods
- **Confidence Intervals**: Calculate confidence bounds using various methods
- **Plotting Positions**: Weibull, Gringorten, and other empirical distribution functions

### Sensitivity Analysis
- **One-at-a-time (OAT)**: Analyze parameter sensitivity individually
- **Sobol Visualization**: Visualize parameter interactions and importance

### Statistical Tools
- **Descriptive Statistics**: Comprehensive statistical descriptors
- **Time Series Analysis**: Auto-correlation and other time series tools
- **Visualization**: Publication-quality plots for statistical analysis

## Quick Start

### Single Distribution

```python
import numpy as np
from statista.distributions import Distributions

# Load your data
data = np.loadtxt("examples/data/time_series2.txt")

# Create a distribution object and fit parameters
dist = Distributions("Gumbel", data=data)
params = dist.fit_model(method="lmoments", test=False)
print(params.loc, params.scale)

# Calculate PDF and CDF
pdf = dist.pdf(plot_figure=True)
cdf, _, _ = dist.cdf(plot_figure=True)

# Goodness-of-fit tests
ks_stat, ks_pvalue = dist.ks()
chi_stat, chi_pvalue = dist.chisquare()
```

### Multi-Distribution Fitting

```python
from statista.distributions import Distributions

# Fit all distributions and find the best one
dist = Distributions(data=data)
best_name, best_info = dist.best_fit()
print(f"Best: {best_name}")
print(f"Parameters: {best_info['parameters']}")

# Or fit all and inspect results
results = dist.fit()
for name, info in results.items():
    print(f"{name}: KS p-value={info['ks'][1]:.4f}")
```

### Extreme Value Analysis

```python
from statista.distributions import Distributions, PlottingPosition

# Fit a GEV distribution using L-moments
gev_dist = Distributions("GEV", data=data)
params = gev_dist.fit_model(method="lmoments")

# Calculate non-exceedance probabilities
cdf_weibul = PlottingPosition.weibul(data)

# Calculate confidence intervals
lower_bound, upper_bound, fig, ax = gev_dist.confidence_interval(
    plot_figure=True
)
```

For more examples and detailed documentation, visit [Statista Documentation](https://serapieum-of-alex.github.io/statista)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Statista in your research, please cite it as:

```
Farrag, M. (2023). Statista: A Python package for statistical analysis, extreme value analysis, and sensitivity analysis.
https://github.com/Serapieum-of-alex/statista
```

BibTeX:
```bibtex
@software{statista2023,
  author = {Farrag, Mostafa},
  title = {Statista: A Python package for statistical analysis, extreme value analysis, and sensitivity analysis},
  url = {https://github.com/Serapieum-of-alex/statista},
  year = {2023}
}
```
