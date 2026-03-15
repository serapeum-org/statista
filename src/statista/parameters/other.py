"""L-moments parameter estimation for other distributions.

Distributions: Exponential, Gamma, Generalized Logistic, Wakeby.
"""

from __future__ import annotations

import numpy as np

SMALL = 1e-6

LMOMENTS_INVALID_ERROR = "L-Moments Invalid"


def exponential(lmoments: list[float | int]) -> list[float | int] | None:
    """Estimate parameters for the Exponential distribution.

    The Exponential distribution is used to model the time between events in a Poisson process.
    It is characterized by two parameters: location and scale.

    Args:
        lmoments: A list of L-moments [l1, l2, ...] where:
            - l1 is the mean (first L-moment)
            - l2 is the L-scale (second L-moment)
            At least 2 L-moments must be provided.

    Returns:
        A list of distribution parameters [location, scale] where:
            - location: Shifts the distribution along the x-axis (minimum value)
            - scale: Controls the spread of the distribution (rate parameter)
        Returns None if the L-moments are invalid.

    Examples:
        - Estimate Exponential parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [0.5, 1.2, 2.3, 3.7, 4.1, 5.6, 6.9]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=2)

          ```
          - Estimate Exponential parameters
          ```python
          >>> params = Lmoments.exponential(l_moments)
          >>> if params:
          ...    print(f"Location: {params[0]}, Scale: {params[1]}")
          Location: 0.6333333333333329, Scale: 2.8380952380952382

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [5.0, 2.5]

          ```
          # Estimate Exponential parameters
          ```python
          >>> params = Lmoments.exponential(l_moments)
          >>> if params:
          ...   print(f"Location: {params[0]}, Scale: {params[1]}")
          Location: 0.0, Scale: 5.0

          ```

    Note:
        The Exponential distribution has the probability density function:
        f(x) = (1/beta) * exp(-(x-mu)/beta) for x >= mu

        Where mu is the location parameter and beta is the scale parameter.

        The method returns None if the second L-moment (l2) is less than or equal to zero,
        as this indicates invalid L-moments for the Exponential distribution.
    """
    if lmoments[1] <= 0:
        print(LMOMENTS_INVALID_ERROR)
        para = None
    else:
        para = [lmoments[0] - 2 * lmoments[1], 2 * lmoments[1]]

    return para


def gamma(lmoments: list[float | int]) -> list[float | int] | None:
    """Estimate parameters for the Gamma distribution.

    The Gamma distribution is a two-parameter family of continuous probability distributions
    used to model positive-valued random variables. It is characterized by a shape parameter
    and a scale parameter.

    Args:
        lmoments: A list of L-moments [l1, l2, ...] where:
            - l1 is the mean (first L-moment)
            - l2 is the L-scale (second L-moment)
            At least 2 L-moments must be provided.

    Returns:
        A list of distribution parameters [shape, scale] where:
            - shape (alpha): Controls the shape of the distribution
            - scale (beta): Controls the spread of the distribution
        Returns None if the L-moments are invalid.

    Examples:
        - Estimate Gamma parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```

          - Calculate L-moments from data
          ```python
          >>> data = [0.8, 1.5, 2.3, 3.7, 4.1, 5.6, 6.9]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=2)

          ```
          - Estimate Gamma parameters
          ```python
          >>> params = Lmoments.gamma(l_moments)
          >>> if params:
          ...   print(f"Shape (alpha): {params[0]}, Scale (beta): {params[1]}")
          Shape (alpha): 1.9539748509411916, Scale (beta): 1.8204650154168824

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 3.0]

          ```
          - Estimate Gamma parameters
          ```python
          >>> params = Lmoments.gamma(l_moments)
          >>> if params:
          ...    print(f"Shape (alpha): {params[0]}, Scale (beta): {params[1]}")
          Shape (alpha): 3.278019029280183, Scale (beta): 3.0506229252109893

          ```

    Note:
        The Gamma distribution has the probability density function:
        f(x) = (x^(alpha-1) * e^(-x/beta)) / (beta^alpha * Gamma(alpha)) for x > 0

        Where alpha is the shape parameter, beta is the scale parameter, and Gamma is the gamma function.

        The method returns None if:
        - The second L-moment (l2) is less than or equal to zero
        - The first L-moment (l1) is less than or equal to the second L-moment (l2)

        These conditions indicate invalid L-moments for the Gamma distribution.
    """
    a1 = -0.3080
    a2 = -0.05812
    a3 = 0.01765
    b1 = 0.7213
    b2 = -0.5947
    b3 = -2.1817
    b4 = 1.2113

    if lmoments[0] <= lmoments[1] or lmoments[1] <= 0:
        print(LMOMENTS_INVALID_ERROR)
        para = None
    else:
        cv = lmoments[1] / lmoments[0]
        if cv >= 0.5:
            t = 1 - cv
            alpha = t * (b1 + t * b2) / (1 + t * (b3 + t * b4))
        else:
            t = np.pi * cv**2
            alpha = (1 + a1 * t) / (t * (1 + t * (a2 + t * a3)))

        para = [alpha, lmoments[0] / alpha]
    return para


def generalized_logistic(
    lmoments: list[float | int],
) -> list[float | int] | None:
    """Estimate parameters for the Generalized Logistic distribution.

    The Generalized Logistic distribution is a flexible three-parameter distribution
    that can model a variety of shapes. It is characterized by location, scale, and
    shape parameters.

    Args:
        lmoments: A list of L-moments [l1, l2, l3, ...] where:
            - l1 is the mean (first L-moment)
            - l2 is the L-scale (second L-moment)
            - l3 is the L-skewness (third L-moment)
            At least 3 L-moments must be provided.

    Returns:
        A list of distribution parameters [location, scale, shape] where:
            - location: Shifts the distribution along the x-axis
            - scale: Controls the spread of the distribution
            - shape: Controls the shape of the distribution
        Returns None if the L-moments are invalid.

    Examples:
        - Estimate Generalized Logistic parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [0.8, 1.5, 2.3, 3.7, 4.1, 5.6, 6.9]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=3)

          ```
          - Estimate Generalized Logistic parameters
          ```python
          >>> params = Lmoments.generalized_logistic(l_moments)
          >>> if params:
          ...   print(f"Location: {params[0]}, Scale: {params[1]}, Shape: {params[2]}")
          Location: 3.346599291165189, Scale: 1.3275318522784219, Shape: -0.09540636042402825

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 2.0, -0.1]  # Negative L-skewness

          ```

          - Estimate Generalized Logistic parameters
          ```python
          >>> params = Lmoments.generalized_logistic(l_moments)
          >>> if params:
          ...   print(f"Location: {params[0]}, Scale: {params[1]}, Shape: {params[2]}")
          Location: 10.327367138330683, Scale: 1.967263286166932, Shape: 0.1

          ```

    Note:
        The Generalized Logistic distribution has the cumulative distribution function:
        F(x) = 1 / (1 + exp(-((x-mu)/alpha))^(1/k)) for k != 0
        F(x) = 1 / (1 + exp(-(x-mu)/alpha)) for k = 0

        Where mu is the location parameter, alpha is the scale parameter, and k is the shape parameter.

        The method returns None if:
        - The second L-moment (l2) is less than or equal to zero
        - The absolute value of the negative third L-moment (g = -l3) is greater than or equal to 1

        These conditions indicate invalid L-moments for the Generalized Logistic distribution.

        When the absolute value of g is very small (<= 1e-6), the shape parameter is set to 0,
        resulting in the standard Logistic distribution.
    """
    g = -lmoments[2]
    if lmoments[1] <= 0 or abs(g) >= 1:
        print(LMOMENTS_INVALID_ERROR)
        para = None
    else:
        if abs(g) <= SMALL:
            para = [lmoments[0], lmoments[1], 0]
            return para

        gg = g * np.pi / np.sin(g * np.pi)
        a = lmoments[1] / gg
        para1 = lmoments[0] - a * (1 - gg) / g
        para = [para1, a, g]
    return para


def wakeby(lmoments: list[float | int]) -> list[float | int] | None:
    """Estimate parameters for the Wakeby distribution.

    The Wakeby distribution is a flexible five-parameter distribution that can model
    a wide variety of shapes. It is particularly useful for modeling extreme events
    in hydrology and other fields.

    Args:
        lmoments: A list of L-moments [l1, l2, l3, l4, l5] where:
            - l1 is the mean (first L-moment)
            - l2 is the L-scale (second L-moment)
            - l3 is the L-skewness (third L-moment)
            - l4 is the L-kurtosis (fourth L-moment)
            - l5 is the fifth L-moment
            All 5 L-moments must be provided.

    Returns:
        A list of distribution parameters [xi, a, b, c, d] where:
            - xi: Location parameter
            - a, b: Scale and shape parameters for the first component
            - c, d: Scale and shape parameters for the second component
        Returns None if the L-moments are invalid.

    Examples:
        - Estimate Wakeby parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [0.8, 1.5, 2.3, 3.7, 4.1, 5.6, 6.9, 8.2, 9.5, 10.3]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=5)

          ```
          - Estimate Wakeby parameters
          ```python
          >>> params = Lmoments.wakeby(l_moments)
          >>> if params:
          ...     print(f"xi: {params[0]}, a: {params[1]}, b: {params[2]}, c: {params[3]}, d: {params[4]}")
          xi: -0.3090923196276183, a: 9.89505997215804, b: 0.7672614429790535, c: 0, d: 0

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 2.0, 0.1, 0.05, 0.02]

          ```
          - Estimate Wakeby parameters
          ```python
          >>> params = Lmoments.wakeby(l_moments)
          >>> if params:
          ...    print(f"xi: {params[0]}, a: {params[1]}, b: {params[2]}, c: {params[3]}, d: {params[4]}")
          xi: 4.51860465116279, a: 4.00999858552907, b: 3.296933739370589, c: 6.793895411225928, d: -0.49376393504801414

          ```

    Note:
        The Wakeby distribution has the quantile function:
        x(F) = xi + (a/(1-b)) * (1-(1-F)^b) - (c/(1+d)) * (1-(1-F)^(-d))

        Where xi, a, b, c, and d are the distribution parameters, and F is the cumulative probability.

        The method returns None if:
        - The second L-moment (l2) is less than or equal to zero
        - The absolute value of any of the L-moments l3, l4, or l5 is greater than or equal to 1

        These conditions indicate invalid L-moments for the Wakeby distribution.

        The Wakeby distribution is very flexible and can approximate many other distributions.
        Special cases include:
        - When c = d = 0, it reduces to the Generalized Pareto distribution
        - When b = d = 0, it reduces to a shifted exponential distribution
    """
    if lmoments[1] <= 0:
        print("Invalid L-Moments")
        return None
    if abs(lmoments[2]) >= 1 or abs(lmoments[3]) >= 1 or abs(lmoments[4]) >= 1:
        print("Invalid L-Moments")
        return None

    alam1 = lmoments[0]
    alam2 = lmoments[1]
    alam3 = lmoments[2] * alam2
    alam4 = lmoments[3] * alam2
    alam5 = lmoments[4] * alam2

    xn1 = 3 * alam2 - 25 * alam3 + 32 * alam4
    xn2 = -3 * alam2 + 5 * alam3 + 8 * alam4
    xn3 = 3 * alam2 + 5 * alam3 + 2 * alam4
    xc1 = 7 * alam2 - 85 * alam3 + 203 * alam4 - 125 * alam5
    xc2 = -7 * alam2 + 25 * alam3 + 7 * alam4 - 25 * alam5
    xc3 = 7 * alam2 + 5 * alam3 - 7 * alam4 - 5 * alam5

    xa = xn2 * xc3 - xc2 * xn3
    xb = xn1 * xc3 - xc1 * xn3
    xc = xn1 * xc2 - xc1 * xn2
    disc = xb * xb - 4 * xa * xc
    skip20 = 0
    if disc < 0:
        pass
    else:
        disc = np.sqrt(disc)
        root1 = 0.5 * (-xb + disc) / xa
        root2 = 0.5 * (-xb - disc) / xa
        b = max(root1, root2)
        d = -min(root1, root2)
        if d >= 1:
            pass
        else:
            a = (
                (1 + b)
                * (2 + b)
                * (3 + b)
                / (4 * (b + d))
                * ((1 + d) * alam2 - (3 - d) * alam3)
            )
            c = (
                -(1 - d)
                * (2 - d)
                * (3 - d)
                / (4 * (b + d))
                * ((1 - b) * alam2 - (3 + b) * alam3)
            )
            xi = alam1 - a / (1 + b) - c / (1 - d)
            if c >= 0 and a + c >= 0:
                skip20 = 1

    if skip20 == 0:
        d = -(1 - 3 * lmoments[2]) / (1 + lmoments[2])
        c = (1 - d) * (2 - d) * lmoments[1]
        b = 0
        a = 0
        xi = lmoments[0] - c / (1 - d)

        if d <= 0:
            a = c
            b = -d
            c = 0
            d = 0

    para = [xi, a, b, c, d]
    return para
