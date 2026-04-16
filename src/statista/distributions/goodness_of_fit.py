"""Structured result type for distribution goodness-of-fit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class GoodnessOfFitResult:
    """Result of a distribution goodness-of-fit test (KS, Chi-square, etc.).

    Supports backward-compatible tuple unpacking so existing callers that used
    ``statistic, p_value = dist.ks()`` continue to work.

    Attributes:
        test_name: Name of the test (e.g., "Kolmogorov-Smirnov", "Chi-square").
        statistic: Test statistic value.
        p_value: p-value of the test.
        conclusion: Plain-English interpretation.
        alpha: Significance level used for the conclusion.
        details: Test-specific extras (critical values, degrees of freedom, etc.).

    Examples:
        Named-field access:

        >>> from statista.distributions import GoodnessOfFitResult
        >>> result = GoodnessOfFitResult(
        ...     test_name="Kolmogorov-Smirnov",
        ...     statistic=0.019,
        ...     p_value=0.9937,
        ...     conclusion="Accept Hypothesis",
        ...     alpha=0.05,
        ...     details={"reference": "Weibull plotting position"},
        ... )
        >>> result.statistic
        0.019
        >>> result.p_value < result.alpha
        False

        Backward-compatible tuple unpacking (statistic, p_value):

        >>> stat, p = result
        >>> stat
        0.019
        >>> p
        0.9937
    """

    test_name: str
    statistic: float
    p_value: float
    conclusion: str = ""
    alpha: float = 0.05
    details: dict = field(default_factory=dict)

    def __iter__(self) -> Iterator[float]:
        """Yield ``(statistic, p_value)`` for backward-compatible tuple unpacking."""
        yield self.statistic
        yield self.p_value

    def __len__(self) -> int:
        """Length matches the backward-compatible ``(statistic, p_value)`` tuple."""
        return 2

    def __getitem__(self, index: int) -> float:
        """Index access mirrors the ``(statistic, p_value)`` tuple (0 or 1)."""
        return (self.statistic, self.p_value)[index]
