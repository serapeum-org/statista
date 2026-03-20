"""Distribution parameters dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
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
        ValueError: If scale is not positive.

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
        - Dict-compatible access:
            ```python
            >>> from statista.distributions import Parameters
            >>> params = Parameters(loc=500.0, scale=200.0)
            >>> params["loc"]
            500.0
            >>> params.get("shape", 0.0)
            0.0
            >>> sorted(params.keys())
            ['loc', 'scale']
            >>> "scale" in params
            True

            ```
        - Invalid scale raises ValueError:
            ```python
            >>> from statista.distributions import Parameters
            >>> Parameters(loc=0.0, scale=-1.0)
            Traceback (most recent call last):
                ...
            ValueError: scale must be positive, got -1.0

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
            result = (
                f"Parameters(loc={self.loc!r}, scale={self.scale!r})"
            )
        return result

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(
                f"scale must be positive, got {self.scale}"
            )

    def __getitem__(self, key: str) -> float | None:
        """Access parameter by name, dict-style."""
        if key not in ("loc", "scale", "shape"):
            raise KeyError(key)
        result = getattr(self, key)
        return result

    def get(self, key: str, default: Any = None) -> float | None:
        """Get parameter value with a default for missing keys."""
        if key not in ("loc", "scale", "shape"):
            result = default
        else:
            val = getattr(self, key)
            result = val if val is not None else default
        return result

    def keys(self) -> list[str]:
        """Return parameter names (excludes shape if None)."""
        if self.shape is not None:
            result = ["loc", "scale", "shape"]
        else:
            result = ["loc", "scale"]
        return result

    def values(self) -> list[float]:
        """Return parameter values (excludes shape if None)."""
        if self.shape is not None:
            result = [self.loc, self.scale, self.shape]
        else:
            result = [self.loc, self.scale]
        return result

    def items(self) -> list[tuple[str, float]]:
        """Return (name, value) pairs (excludes shape if None)."""
        result = [(k, self[k]) for k in self.keys()]
        return result

    def __contains__(self, key: object) -> bool:
        """Check if a parameter name is present."""
        if key == "shape":
            result = self.shape is not None
        else:
            result = key in ("loc", "scale")
        return result

    def __len__(self) -> int:
        """Return number of parameters (2 or 3)."""
        if self.shape is not None:
            result = 3
        else:
            result = 2
        return result

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names."""
        return iter(self.keys())

    def __eq__(self, other: object) -> bool:
        """Compare with another Parameters or a dict."""
        if isinstance(other, Parameters):
            result = (
                self.loc == other.loc
                and self.scale == other.scale
                and self.shape == other.shape
            )
        elif isinstance(other, dict):
            result = dict(self.items()) == other
        else:
            result = NotImplemented
        return result
