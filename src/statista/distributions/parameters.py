"""Distribution parameters dataclass."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterator

from statista.exceptions import ParameterError

_DEPRECATION_MSG = (
    "Dict-style access on Parameters is deprecated. "
    "Use attribute access instead: params.loc, params.scale, "
    "params.shape"
)


@dataclass(frozen=True)
class Parameters:
    """Distribution parameters with named fields and dict compatibility.

    Provides named access to distribution parameters while remaining
    backward-compatible with dict-based access patterns (``[]``,
    ``.get()``, ``.keys()``, ``in``, ``len``).

    Args:
        loc: Location parameter.
        scale: Scale parameter (must be positive).
        shape: Shape parameter (only for 3-parameter distributions
            like GEV). Default is None.

    Raises:
        ParameterError: If scale is not positive.

    Examples:
        - Create 2-parameter distribution parameters:
            ```python
            >>> from statista.distributions import Parameters
            >>> params = Parameters(loc=500.0, scale=200.0)
            >>> params.loc
            500.0
            >>> params.scale
            200.0
            >>> params.shape is None
            True
            >>> len(params)
            2

            ```
        - Create 3-parameter distribution parameters:
            ```python
            >>> from statista.distributions import Parameters
            >>> params = Parameters(loc=500.0, scale=200.0, shape=-0.2)
            >>> params.shape
            -0.2
            >>> len(params)
            3

            ```
        - Attribute access (recommended):
            ```python
            >>> from statista.distributions import Parameters
            >>> params = Parameters(loc=500.0, scale=200.0)
            >>> params.loc
            500.0
            >>> params.shape is None
            True
            >>> sorted(params.keys())
            ['loc', 'scale']

            ```
        - Invalid scale raises ParameterError:
            ```python
            >>> from statista.distributions import Parameters
            >>> Parameters(loc=0.0, scale=-1.0)
            Traceback (most recent call last):
                ...
            statista.exceptions.ParameterError: scale must be positive, got -1.0

            ```

    """

    loc: float
    scale: float
    shape: float | None = None

    def __repr__(self) -> str:
        """Return a repr that omits shape when it is None."""
        if self.shape is not None:
            result = (
                f"Parameters(loc={self.loc!r}, scale={self.scale!r},"
                f" shape={self.shape!r})"
            )
        else:
            result = f"Parameters(loc={self.loc!r}, scale={self.scale!r})"
        return result

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ParameterError(f"scale must be positive, got {self.scale}")

    # -- Dict-compatibility layer ------------------------------------------
    # The methods below replicate the dict interface so that existing code
    # using params["loc"], params.get("scale"), "shape" in params, etc.
    # continues to work without modification after the migration from
    # dict[str, float] to Parameters.

    def __getitem__(self, key: str) -> float | None:
        """Access parameter by name, dict-style.

        Replaces ``params["loc"]`` previously used on plain dicts.

        .. deprecated::
            Use attribute access instead: ``params.loc``,
            ``params.scale``, ``params.shape``.
        """
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        if key not in ("loc", "scale", "shape"):
            raise KeyError(key)
        result = getattr(self, key)
        return result

    def get(self, key: str, default: Any = None) -> float | None:
        """Get parameter value with a default for missing keys.

        Replaces ``params.get("scale")`` previously used on plain dicts.
        When shape is None (2-param distributions), behaves as if the
        key is absent — returns the default, matching the old dict
        behavior where 2-param dicts simply had no "shape" key.

        .. deprecated::
            Use attribute access instead: ``params.loc``,
            ``params.scale``, ``params.shape``.
        """
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        if key not in ("loc", "scale", "shape"):
            result = default
        else:
            val = getattr(self, key)
            result = val if val is not None else default
        return result

    def keys(self) -> list[str]:
        """Return parameter names (excludes shape if None).

        Replaces ``params.keys()`` previously used on plain dicts.
        For 2-param distributions returns ["loc", "scale"], matching
        the old dict that had no "shape" key.
        """
        if self.shape is not None:
            result = ["loc", "scale", "shape"]
        else:
            result = ["loc", "scale"]
        return result

    def values(self) -> list[float]:
        """Return parameter values (excludes shape if None).

        Replaces ``params.values()`` previously used on plain dicts.
        Preserves insertion order: [loc, scale] or [loc, scale, shape].
        """
        if self.shape is not None:
            result = [self.loc, self.scale, self.shape]
        else:
            result = [self.loc, self.scale]
        return result

    def items(self) -> list[tuple[str, float]]:
        """Return (name, value) pairs (excludes shape if None).

        Replaces ``params.items()`` previously used on plain dicts.
        """
        result = [(k, getattr(self, k)) for k in self.keys()]
        return result

    def __contains__(self, key: object) -> bool:
        """Check if a parameter name is present.

        Replaces ``"scale" in params`` previously used on plain dicts.
        Returns False for "shape" when shape is None, matching the old
        dict behavior where 2-param dicts had no "shape" key.
        """
        if key == "shape":
            result = self.shape is not None
        else:
            result = key in ("loc", "scale")
        return result

    def __len__(self) -> int:
        """Return number of parameters (2 or 3).

        Replaces ``len(params)`` previously used on plain dicts.
        Used by ``AbstractDistribution.chisquare()`` to set the
        degrees-of-freedom adjustment.
        """
        if self.shape is not None:
            result = 3
        else:
            result = 2
        return result

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names.

        Replaces ``for key in params`` previously used on plain dicts.
        """
        if self.shape is not None:
            result = iter(["loc", "scale", "shape"])
        else:
            result = iter(["loc", "scale"])
        return result

    def __eq__(self, other: object) -> bool:
        """Compare with another Parameters or a dict.

        Supports comparison with plain dicts for backward compatibility,
        so ``Parameters(loc=1, scale=2) == {"loc": 1, "scale": 2}``
        returns True.
        """
        if isinstance(other, Parameters):
            result = (
                self.loc == other.loc
                and self.scale == other.scale
                and self.shape == other.shape
            )
        elif isinstance(other, dict):
            expected = {"loc": self.loc, "scale": self.scale}
            if self.shape is not None:
                expected["shape"] = self.shape
            result = expected == other
        else:
            result = NotImplemented
        return result
