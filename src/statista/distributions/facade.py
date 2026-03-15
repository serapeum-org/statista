"""Distributions facade class."""

from __future__ import annotations

from numbers import Number

import numpy as np

from statista.distributions.exponential import Exponential
from statista.distributions.gev import GEV
from statista.distributions.gumbel import Gumbel
from statista.distributions.normal import Normal


class Distributions:
    """Distributions."""

    available_distributions = {
        "GEV": GEV,
        "Gumbel": Gumbel,
        "Exponential": Exponential,
        "Normal": Normal,
    }

    def __init__(
        self,
        distribution: str,
        data: list | np.ndarray | None = None,
        parameters: dict[str, Number] = None,
    ):
        if distribution not in self.available_distributions.keys():
            raise ValueError(f"{distribution} not supported")

        self.distribution = self.available_distributions[distribution](data, parameters)  # type: ignore[abstract, arg-type]

    def __getattr__(self, name: str):
        """Delegate method calls to the subclass"""
        # Retrieve the attribute or method from the distribution object
        try:
            # Retrieve the attribute or method from the subclasses
            attribute = getattr(self.distribution, name)

            # If the attribute is a method, return a callable function
            if callable(attribute):

                def method(*args, **kwargs):
                    """A callable function that simply calls the attribute if it is a method"""
                    return attribute(*args, **kwargs)

                return method

            # If it's a regular attribute, return its value
            return attribute

        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
