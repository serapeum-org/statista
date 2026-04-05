"""Type stubs for mixin classes — only evaluated during type checking."""

# flake8: noqa: E704

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame, Index

    class _TimeSeriesStub:
        """Declares the interface that all mixins can rely on.

        At runtime this class does not exist — mixins inherit from ``object``.
        During type-checking, mypy sees this as a parent and resolves all
        DataFrame attributes and shared helper methods.
        """

        columns: Index
        index: Index
        shape: tuple[int, ...]
        values: np.ndarray
        ndim: int

        @staticmethod
        def _get_ax_fig(n_subplots: int = 1, **kwargs: Any) -> tuple[Figure, Axes]: ...

        @staticmethod
        def _adjust_axes_labels(
            ax: Axes, tick_labels: list[str] | None = None, **kwargs: Any
        ) -> Axes: ...

        def __getitem__(self, key: str) -> DataFrame: ...

        def describe(self) -> DataFrame: ...

        def corr(self, method: str = "pearson", **kwargs: Any) -> DataFrame: ...

        def resample(self, rule: str, **kwargs: Any) -> Any: ...

        def rolling(self, window: int, **kwargs: Any) -> Any: ...

        def ewm(self, **kwargs: Any) -> Any: ...
