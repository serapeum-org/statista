"""Structured result types for statistical hypothesis tests."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StatTestResult:
    """Result of a statistical hypothesis test.

    Attributes:
        test_name: Name of the test (e.g., "Augmented Dickey-Fuller").
        statistic: Test statistic value.
        p_value: p-value of the test.
        conclusion: Plain-English interpretation (e.g., "Stationary at 5% significance level").
        alpha: Significance level used for the conclusion.
        details: Test-specific extras (critical values, lags used, etc.).

    Examples:
        ```python
        >>> from statista.stat_result import StatTestResult
        >>> result = StatTestResult(
        ...     test_name="Augmented Dickey-Fuller",
        ...     statistic=-3.45,
        ...     p_value=0.009,
        ...     conclusion="Stationary at 5% significance level",
        ...     alpha=0.05,
        ...     details={"used_lag": 2, "n_obs": 100},
        ... )
        >>> result.test_name
        'Augmented Dickey-Fuller'
        >>> result.p_value < result.alpha
        True

        ```
    """

    test_name: str
    statistic: float
    p_value: float
    conclusion: str
    alpha: float = 0.05
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrendTestResult:
    """Result of a trend test (Mann-Kendall family).

    Field names follow pymannkendall conventions for familiarity.

    Attributes:
        trend: Direction string — "increasing", "decreasing", or "no trend".
        h: Whether the null hypothesis (no trend) is rejected.
        p_value: Two-tailed p-value.
        z: Standardized test statistic.
        tau: Kendall's tau correlation coefficient.
        s: Mann-Kendall S statistic.
        var_s: Variance of the S statistic.
        slope: Sen's slope estimate.
        intercept: Intercept of Sen's line.

    Examples:
        ```python
        >>> from statista.stat_result import TrendTestResult
        >>> result = TrendTestResult(
        ...     trend="increasing",
        ...     h=True,
        ...     p_value=0.001,
        ...     z=3.29,
        ...     tau=0.45,
        ...     s=156.0,
        ...     var_s=2200.0,
        ...     slope=0.12,
        ...     intercept=1.5,
        ... )
        >>> result.trend
        'increasing'
        >>> result.h
        True

        ```
    """

    trend: str
    h: bool
    p_value: float
    z: float
    tau: float
    s: float
    var_s: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class ChangePointResult:
    """Result of a change point test (Pettitt, SNHT, Buishand).

    Attributes:
        test_name: Name of the test (e.g., "Pettitt").
        h: Whether the null hypothesis (homogeneous) is rejected. True means a change point was detected.
        change_point: Index position of the detected change point.
        change_point_date: Datetime of the change point if the input had a DatetimeIndex, otherwise None.
        statistic: Test statistic value.
        p_value: p-value (analytical or Monte Carlo).
        mean_before: Mean of the series before the change point.
        mean_after: Mean of the series after the change point.

    Examples:
        ```python
        >>> from statista.stat_result import ChangePointResult
        >>> result = ChangePointResult(
        ...     test_name="Pettitt",
        ...     h=True,
        ...     change_point=50,
        ...     change_point_date=None,
        ...     statistic=1234.0,
        ...     p_value=0.003,
        ...     mean_before=10.5,
        ...     mean_after=15.2,
        ... )
        >>> result.h
        True
        >>> result.mean_after > result.mean_before
        True

        ```
    """

    test_name: str
    h: bool
    change_point: int
    change_point_date: object
    statistic: float
    p_value: float
    mean_before: float
    mean_after: float
