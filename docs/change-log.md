# Changelog


## 0.8.0 (2026-04-15)


- fix: update mkdocs-deploy action version to use specific mkdocs tag
- feat(goodness_of_fit): introduce GoodnessOfFitResult for KS and Chi-square tests
- feat(goodness_of_fit): introduce GoodnessOfFitResult for KS and Chi-square tests
- feat(time_series): add TimeSeries subpackage with 53 analysis methods (#99)
- Introduce a pandas.DataFrame subclass composed from 12 mixins providing                   
  descriptive statistics, visualization, missing-data handling,                         
  autocorrelation, stationarity, trend detection, change-point analysis,                    
  distribution fitting, decomposition, seasonal, hydrological, and                          
  comparison / anomaly methods. Refactor distribution goodness-of-fit to                    
  return a structured result type, and wire a content-hashed notebook                       
  cache into the mkdocs CI pipeline so executed outputs are no longer                       
  committed to git.
-   - Add src/statista/time_series/ (12 mixins, ~53 methods, shared
    _TimeSeriesStub for clean mypy inheritance)
  - Move StatTestResult -> GoodnessOfFitResult into distributions subpackage
    and wire it through AbstractDistribution.ks and .chisquare; preserve
    tuple unpacking, len, and index access for backward compatibility
  - Delete unused TrendTestResult and ChangePointResult dataclasses
  - Add scripts/strip_notebook_outputs.py and scripts/prep_notebooks.py
    plus nbstripout pre-commit hook; remove committed outputs from the
    12 tutorial notebooks
  - Add content-hashed jupyter-cache step to mkdocs deploy workflow keyed
    on notebooks + source + pyproject; restore .jupyter_cache/ gitignore
  - Add .gitattributes to normalize LF line endings across platforms
  - Fix ADF/KPSS constant-series conclusion to trivially stationary
  - Add 12 tutorial notebooks and 12 mkdocs reference pages with mermaid
    diagrams; drop stale docs/reference/time_series/ tree
  - 838 pytest tests + 606 doctests passing; mypy clean
- ref: #93, #162, ..., #181
- chore: rename organisation from Serapieum-of-alex to serapeum-org (#161)
- Update all references across pyproject.toml files, GitHub Actions
  workflows, mkdocs.yml, READMEs, and docs (94 occurrences, 28 files).
- fix:issues in parameter class (#160)
- feat(distributions)!: introduce Parameters dataclass replacing dict-based parameters (#157)
- feat(distributions)!: introduce Parameters dataclass replacing dict-based parameters
-   - Add frozen Parameters dataclass with loc, scale,
    optional shape fields and scale validation
  - Add StatistaError/ParameterError custom exception
    hierarchy in exceptions.py
  - Migrate all 42 method signatures across 6 distribution
    modules from dict[str, float] to Parameters
  - Replace all .get("loc")/["scale"] access with
    attribute access (.loc, .scale, .shape)
  - Add dict-to-Parameters auto-conversion at 17 entry
    points with deprecation warnings for dict-style access
  - Rename fit_all() to fit() in Distributions facade
  - Remove fit_results parameter from best_fit()
  - Split monolithic test_parameters.py into 12
    per-distribution files under tests/parameters/
  - Add 52 dedicated tests for Parameters dataclass
-   BREAKING CHANGE: fit_all() renamed to fit() and
  best_fit() no longer accepts fit_results parameter.
  Distribution methods now return Parameters instances
  instead of dicts.
- ref: #159, #87, # 152
- feat(distributions): add multi-distribution fitting and best-fit selection (#151)
- Add fit_all and best_fit methods to the Distributions facade class,
  enabling users to fit all available distributions to a dataset and
  select the best one by goodness-of-fit criterion (KS or Chi-square).
-   - Support dual-mode initialization: single-distribution wrapping
    (existing) and multi-distribution comparison (distribution=None)
  - fit_all fits selected or all distributions with KS and Chi-square
    goodness-of-fit tests
  - best_fit selects the distribution with the highest p-value, calling
    fit_all internally if needed
  - Add input validation for method, distribution names, and empty lists
  - Eliminate redundant data storage via _data property delegation
  - Add comprehensive test suite (51 tests, 100% facade coverage)
  - Restrict CI test workflow to main branch pushes and pull requests
- ref: #156
- ci(workflows): chain release pipelines and fix commit message template (#149)
- - Add workflow_run trigger to docs deployment after github-release
  - Guard PyPI publish job to skip on failed upstream workflow
  - Fix lockfile commit message to use inputs.increment instead of
    inputs.package
-  Closes #150
- ci(release): harden github-release workflow with admin check and lockfile sync (#146)
- ci(release): harden github-release workflow with admin check and lockfile sync
-   - Rename release.yml to github-release.yml
  - Add admin permission check before creating release
  - Add explicit checkout with full git history
  - Sync uv.lock after commitizen version bump
  - Remove redundant release-branch input pass-through
-   Closes #147
- refactor!: restructure package into submodules with bug fixes and modern typing (#132)
- refactor!: restructure package into submodules with bug fixes and modern typing
-   - Split monolithic distributions.py into submodules:
    base, gumbel, gev, exponential, normal, facade
  - Split parameters.py into submodules:
    lmoments, extreme_value, normal_family, other
  - Fix any(cdf) returning bool instead of element-wise check
    in all inverse_cdf methods
  - Fix Normal._cdf_eq rejecting valid loc<=0 values
  - Fix duplicate return period (50 appeared twice) in eva.py
  - Fix assert ValueError() never raising in sensitivity.sobol()
  - Fix in-place sort mutating caller data in Plot.details()
    and Plot.confidence_level()
  - Fix confidence_interval.py scale parameter index typo
  - Fix save_to silently defaulting to cwd in eva.py
    (now raises ValueError when save_plots=True without save_to)
  - Unify return_period() API: keyword-only args across
    Gumbel and GEV
  - Modernize typing: PEP 604 unions, from __future__ import
    annotations, full mypy compliance across all modules
  - Correct Normal distribution PDF/CDF docstring equations
    (were copy-pasted from Exponential)
  - Convert all docstring math from RST to MathJax-compatible
    Markdown ($$...$$ and \(...\))
  - Add pymdownx.arithmatex for rendered equations in docs
  - Add mypy pre-commit hook and docs guide
  - Add comprehensive test suites for descriptors, sensitivity,
    plot, and individual distribution modules (213+ new tests)
  - Rename tests_utils.py to test_utils.py for pytest discovery
  - Update pypi-release workflow to use composite action
  - Set MPLBACKEND=Agg via pytest-env for stable doctest runs
-   BREAKING CHANGE: statista.distributions is now a package
  (directory) instead of a single module. Direct imports like
  `from statista.distributions import GEV` still work via
  __init__.py re-exports, but `import statista.distributions`
  as a module object has changed.
- ref: #134, #135, #136, #137, #138, #139, #140, #141, #142, #143, #144, #145
- ci(release): refactor workflows to use composite actions (#129)
- ci(release): refactor workflows to use composite actions (#129)
- build: migrate from poetry to uv package manager (#128)
- build: migrate from poetry to uv package manager
-   - Replace Poetry with uv as the primary package manager
  - Convert project.optional-dependencies to dependency-groups (PEP 735)
  - Update all CI/CD workflows to use uv with composite actions
  - Consolidate release workflows (remove release-bump.yml and github-release.yml)
  - Update actions to latest versions (checkout@v5, composite actions v1)
  - Add comprehensive mypy configuration with module-specific overrides
  - Fix deprecated pandas offset alias "A-OCT" to "YE-OCT" in eva.py
  - Simplify mkdocs deployment workflow with reusable composite actions
  - Update PyPI release workflow to use uv build
- ci(release-bump): update step name to clarify version bump and tagging process
- ci(release-bump): remove GitHub token from release-bump workflow
- ci(release-bump): add GitHub token for release bump in main
- chore(pyproject): remove update_changelog_on_bump configuration
- chore(release): add Commitizen and GitHub release workflow (#126)
- chore(release): add Commitizen and GitHub release workflow
- - Added Commitizen for conventional commit management and versioning
- Created GitHub Actions workflow to publish releases using Commitizen
- Updated Commitizen to v4.8.3 and replaced deprecated commands
- Removed `dependabot.yml` and Poetry configuration
- ref: #127
- chore(templates): refine issue templates and add performance and documentation categories (#124)
- - Updated bug report template with clearer sections and reproducibility requirements
- Enhanced feature request template with detailed prompts for API changes and performance considerations
- Added new templates for performance issues and documentation improvements
- Configured default behavior to disable blank issues and added links for Q&A and documentation access
- ref: #125
- ci: update workflow triggers and improve Codecov reporting (#122)
- ci: update workflow triggers and improve Codecov reporting
- - Trigger PyPI publish workflow only on release published event
- Adjust tests.yml trigger to fix Codecov reporting for main branch
- Added step to upload detailed Codecov test report
- Removed Codecov token and restricted trigger to pushes on main
- ref: #123
- ci(pypi): trigger publish workflow only on release published event (#120)
- ref: #121

## 0.6.3 (2025-08-08)
##### Distributions
* fix the `chisquare` method to all distributions.

## 0.6.2 (2025-07-31)
##### Docs
* add complete documentation for all modules.

#### Dev
* refactor all modules.
* fix pre-commit hooks.


## 0.6.1 (2025-06-03)
##### Dev
* replace the setup.py with pyproject.toml.
* migrate the documentation to use mkdocs-material.
* add complete documentation for all modules.


## 0.6.0 (2024-08-18)

##### dev
* Add documentations for the `distributions`, and `eva` modules.
* Add autodoc for all modules.
* Test docstrings as part of CI and pre-commit hooks.
* Test notebooks as part of CI.
* Simplify test for the distributions module

##### distributions
* move the `cdf` and `parameters` for all the methods to be optional parameters.
* rename `theoretical_estimate` method to `inverse_cdf`.
* All distributions can be instantiated with the parameters and/or data.
* rename the `probability_plot` method to `plot`.
* move the `confidence_interval` plot from the `probability_plot/plot` to the method `confidence_interval` and can be
  called by activating the `plot_figure=True`.

##### descriptors
* rename the `metrics` module to `descriptors`.

## 0.5.0 (2023-12-11)

* Unify the all the methods for the distributions.
* Use factory design pattern to create the distributions.
* add tests for the eva module.
* use snake_case for the methods and variables.

## 0.4.0 (2023-11-23)

* add Pearson 3 distribution
* Use setup.py instead of pyproject.toml.
* Correct pearson correlation coefficient and add documentation .
* replace the pdf and cdf by the methods from scipy package.

## 0.3.0 (2023-02-19)

* add documentations for both GEV and gumbel distributions.
* add lmoment parameter estimation method for all distributions.
* add exponential and normal distributions
* modify the pdf, cdf, and probability plot plots
* create separate plot and confidence_interval modules.

## 0.2.0 (2023-02-08)

* add eva (Extreme value analysis) module
* fix bug in obtaining distribution parameters using optimization method

## 0.1.8 (2023-01-31)

* bump up versions

## 0.1.7 (2022-12-26)

* lock numpy to version 1.23.5

## 0.1.0 (2022-05-24)

* First release on PyPI.
