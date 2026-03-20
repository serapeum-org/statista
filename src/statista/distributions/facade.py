"""Distributions facade class.

Provides the ``Distributions`` facade that wraps individual distribution
classes (GEV, Gumbel, Exponential, Normal) behind a single entry point,
and exposes methods for fitting all distributions at once and selecting
the best fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from statista.distributions.base import AbstractDistribution
from statista.distributions.exponential import Exponential
from statista.distributions.gev import GEV
from statista.distributions.gumbel import Gumbel
from statista.distributions.normal import Normal
from statista.distributions.parameters import Parameters


class Distributions:
    """Facade for working with probability distributions.

    ``Distributions`` can be used in two modes:

    1. **Single-distribution mode**: pass a distribution name to wrap a
       specific distribution and delegate all method calls to it.
    2. **Multi-distribution mode**: pass only data (no distribution name)
       and use ``fit`` / ``best_fit`` to compare all distributions.

    Args:
        distribution: Name of the distribution to use. Must be one of the
            keys in ``available_distributions`` ('GEV', 'Gumbel',
            'Exponential', 'Normal'). If None, no single distribution is
            wrapped — use ``fit`` or ``best_fit`` instead.
        data: Data time series as a list or numpy array.
        parameters: Distribution parameters as a ``Parameters`` instance
            or a dictionary (auto-converted).
            ```python
            Parameters(loc=0.0, scale=1.0)
            ```

    Attributes:
        available_distributions (dict[str, type[AbstractDistribution]]):
            Registry mapping distribution names to their classes.
        distribution (AbstractDistribution | None): The underlying
            distribution instance (None in multi-distribution mode).

    Raises:
        ValueError: If the distribution name is not in
            ``available_distributions``.
        ValueError: If neither distribution nor data is provided.

    Examples:
        - Single-distribution mode — wrap a Gumbel and fit:
            ```python
            >>> import numpy as np
            >>> from statista.distributions import Distributions
            >>> data = np.loadtxt("examples/data/time_series2.txt")
            >>> dist = Distributions("Gumbel", data=data)
            >>> params = dist.fit_model(method="lmoments", test=False)
            >>> sorted(params.keys())
            ['loc', 'scale']

            ```
        - Multi-distribution mode — find the best fit in one call:
            ```python
            >>> import numpy as np
            >>> from statista.distributions import Distributions
            >>> data = np.loadtxt("examples/data/time_series2.txt")
            >>> dist = Distributions(data=data)
            >>> best_name, best_info = dist.best_fit() # doctest: +ELLIPSIS
            -----KS Test--------
            ...
            >>> best_name
            'GEV'

            ```
        - Create a distribution from known parameters:
            ```python
            >>> from statista.distributions import Distributions, Parameters
            >>> params = Parameters(loc=500, scale=200)
            >>> dist = Distributions("Normal", parameters=params)
            >>> dist.parameters.loc
            500

            ```
        - Invalid distribution name raises ValueError:
            ```python
            >>> from statista.distributions import Distributions
            >>> Distributions("InvalidDist", data=[1, 2, 3])
            Traceback (most recent call last):
                ...
            ValueError: InvalidDist not supported

            ```

    See Also:
        Gumbel: Gumbel (Extreme Value Type I) distribution.
        GEV: Generalized Extreme Value distribution.
        Exponential: Exponential distribution.
        Normal: Normal (Gaussian) distribution.

    """

    available_distributions: dict[str, type[AbstractDistribution]] = {
        "GEV": GEV,
        "Gumbel": Gumbel,
        "Exponential": Exponential,
        "Normal": Normal,
    }

    def __init__(
        self,
        distribution: str | None = None,
        data: list | np.ndarray | None = None,
        parameters: dict[str, Any] | Parameters | None = None,
    ):
        if distribution is not None:
            if distribution not in self.available_distributions:
                raise ValueError(f"{distribution} not supported")
            if data is None and parameters is None:
                raise ValueError(
                    "data or parameters must be provided when"
                    " specifying a distribution"
                )
            dist_class = self.available_distributions[distribution]
            self.distribution: AbstractDistribution | None = dist_class(
                data, parameters
            )
            self.__data = None
        else:
            if data is None:
                raise ValueError("Either distribution or data must be provided")
            self.distribution = None
            self.__data = np.array(data)

    @property
    def _data(self) -> np.ndarray | None:
        """Return the raw data array.

        In single-distribution mode, delegates to the underlying
        distribution's data to avoid storing a redundant copy. In
        multi-distribution mode, returns the locally stored array.
        """
        if self.distribution is not None:
            return self.distribution.data
        return self.__data

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying distribution instance.

        Any attribute or method not defined directly on ``Distributions``
        is looked up on the wrapped distribution object. This allows
        transparent access to ``pdf``, ``cdf``, ``fit_model``, ``ks``,
        ``chisquare``, ``inverse_cdf``, ``confidence_interval``, ``plot``,
        and all other methods of the concrete distribution.

        Args:
            name: Attribute name to look up.

        Returns:
            The attribute from the underlying distribution instance.

        Raises:
            AttributeError: If neither ``Distributions`` nor the underlying
                distribution has the requested attribute.
        """
        if self.distribution is not None:
            try:
                return getattr(self.distribution, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def fit(
        self,
        method: str = "lmoments",
        distributions: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fit multiple distributions to the data and evaluate goodness of fit.

        Fits each distribution using the specified method, then runs both
        the Kolmogorov-Smirnov and Chi-square goodness-of-fit tests. NaN
        values are removed and the data is sorted before fitting.

        Args:
            method: Fitting method ('mle', 'mm', 'lmoments', or
                'optimization'). Default is 'lmoments'.
            distributions: List of distribution names to fit. If None,
                fits all available distributions ('GEV', 'Gumbel',
                'Exponential', 'Normal').

        Returns:
            Dictionary keyed by distribution name, each value is a dict
            containing:
                - 'distribution': the fitted ``AbstractDistribution``
                  instance
                - 'parameters': ``Parameters`` instance (e.g.,
                  ``Parameters(loc=..., scale=...)``)
                - 'ks': tuple of (statistic, p-value) from the
                  Kolmogorov-Smirnov test
                - 'chisquare': tuple of (statistic, p-value) from the
                  Chi-square test

        Raises:
            ValueError: If a distribution name is not in
                ``available_distributions``.

        Examples:
            - Fit all distributions and inspect the result keys:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Distributions
                >>> data = np.loadtxt("examples/data/time_series2.txt")
                >>> dist = Distributions(data=data)
                >>> results = dist.fit() # doctest: +ELLIPSIS
                -----KS Test--------
                ...
                >>> sorted(results.keys())
                ['Exponential', 'GEV', 'Gumbel', 'Normal']

                ```
            - Fit only a subset of distributions:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Distributions
                >>> data = np.loadtxt("examples/data/time_series2.txt")
                >>> dist = Distributions(data=data)
                >>> results = dist.fit(
                ...     distributions=["Gumbel", "GEV"]
                ... ) # doctest: +ELLIPSIS
                -----KS Test--------
                ...
                >>> sorted(results.keys())
                ['GEV', 'Gumbel']

                ```
            - Access fitted parameters and KS p-value:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Distributions
                >>> data = np.loadtxt("examples/data/time_series2.txt")
                >>> dist = Distributions(data=data)
                >>> results = dist.fit(
                ...     distributions=["Gumbel"]
                ... ) # doctest: +ELLIPSIS
                -----KS Test--------
                ...
                >>> sorted(results["Gumbel"]["parameters"].keys())
                ['loc', 'scale']
                >>> bool(0 <= results["Gumbel"]["ks"][1] <= 1)
                True

                ```

        See Also:
            best_fit: Fit all distributions and directly return the best
                one.

        """
        valid_methods = ("mle", "mm", "lmoments", "optimization")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        data = np.array(self._data)
        data = data[~np.isnan(data)]
        data = np.sort(data)

        dist_names = (
            distributions
            if distributions is not None
            else list(self.available_distributions.keys())
        )

        if not dist_names:
            raise ValueError("distributions list must not be empty")

        results: dict[str, dict[str, Any]] = {}
        for name in dist_names:
            if name not in self.available_distributions:
                raise ValueError(f"{name} not supported")

            dist_class = self.available_distributions[name]
            dist_instance = dist_class(data=data)
            parameters = dist_instance.fit_model(method=method, test=False)
            ks_result = dist_instance.ks()
            chisquare_result = dist_instance.chisquare()

            results[name] = {
                "distribution": dist_instance,
                "parameters": parameters,
                "ks": ks_result,
                "chisquare": chisquare_result,
            }

        return results

    def best_fit(
        self,
        method: str = "lmoments",
        distributions: list[str] | None = None,
        criterion: str = "ks",
    ) -> tuple[str, dict[str, Any]]:
        """Find the best-fitting distribution for the data.

        Fits all (or selected) distributions and returns the one with
        the highest goodness-of-fit p-value.

        Args:
            method: Fitting method ('mle', 'mm', 'lmoments', or
                'optimization'). Default is 'lmoments'.
            distributions: List of distribution names to fit. If None,
                fits all available distributions.
            criterion: Goodness-of-fit criterion for selection.
                'ks' selects by highest Kolmogorov-Smirnov p-value.
                'chisquare' selects by highest Chi-square p-value.
                Default is 'ks'.

        Returns:
            Tuple of (distribution_name, result_dict) for the best fit.
            The result dict contains:
                - 'distribution': the fitted distribution instance
                - 'parameters': ``Parameters`` instance
                - 'ks': (statistic, p-value) tuple
                - 'chisquare': (statistic, p-value) tuple

        Raises:
            ValueError: If ``criterion`` is not 'ks' or 'chisquare'.

        Examples:
            - Find the best distribution directly from data:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Distributions
                >>> data = np.loadtxt("examples/data/time_series2.txt")
                >>> dist = Distributions(data=data)
                >>> best_name, best_info = dist.best_fit() # doctest: +ELLIPSIS
                -----KS Test--------
                ...
                >>> best_name
                'GEV'
                >>> sorted(best_info["parameters"].keys())
                ['loc', 'scale', 'shape']

                ```
            - Select by Chi-square criterion among specific distributions:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Distributions
                >>> data = np.loadtxt("examples/data/time_series2.txt")
                >>> dist = Distributions(data=data)
                >>> best_name, best_info = dist.best_fit(
                ...     distributions=["Gumbel", "GEV"],
                ...     criterion="chisquare",
                ... ) # doctest: +ELLIPSIS
                -----KS Test--------
                ...
                >>> best_name in ("Gumbel", "GEV")
                True

                ```

        See Also:
            fit: Fit multiple distributions and return all results.

        """
        if criterion not in ("ks", "chisquare"):
            raise ValueError(
                f"criterion must be 'ks' or 'chisquare', got '{criterion}'"
            )

        results = self.fit(method=method, distributions=distributions)

        best_name = next(iter(results))
        best_p_value = -1.0
        for name, info in results.items():
            p_value = info[criterion][1]
            if p_value > best_p_value:
                best_p_value = p_value
                best_name = name

        return best_name, results[best_name]
