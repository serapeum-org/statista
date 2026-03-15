"""Distributions facade class."""

from __future__ import annotations

from typing import Any

import numpy as np

from statista.distributions.base import AbstractDistribution
from statista.distributions.exponential import Exponential
from statista.distributions.gev import GEV
from statista.distributions.gumbel import Gumbel
from statista.distributions.normal import Normal


class Distributions:
    """Distributions."""

    available_distributions: dict[str, type[AbstractDistribution]] = {
        "GEV": GEV,
        "Gumbel": Gumbel,
        "Exponential": Exponential,
        "Normal": Normal,
    }

    def __init__(
        self,
        distribution: str,
        data: list | np.ndarray | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        if distribution not in self.available_distributions:
            raise ValueError(f"{distribution} not supported")

        dist_class = self.available_distributions[distribution]
        self.distribution = dist_class(data, parameters)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying distribution instance."""
        try:
            return getattr(self.distribution, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
