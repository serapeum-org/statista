"""L-moments parameter estimation for normal-family distributions.

Distributions: Normal, Generalized Normal, Pearson Type III.
"""

from __future__ import annotations

import numpy as np
import scipy.special as _spsp

LMOMENTS_INVALID_ERROR = "L-Moments Invalid"


def normal(lmoments: list[float | int]) -> list[float | int] | None:
    """Estimate parameters for the Normal (Gaussian) distribution.

    The Normal distribution is a symmetric, bell-shaped distribution that is
    completely characterized by its mean and standard deviation. It is one of the
    most widely used probability distributions in statistics.

    Args:
        lmoments: A list of L-moments [l1, l2, ...] where:
            - l1 is the mean (first L-moment)
            - l2 is the L-scale (second L-moment)
            At least 2 L-moments must be provided.

    Returns:
        A list of distribution parameters [location, scale] where:
            - location: The mean of the distribution
            - scale: The standard deviation of the distribution
        Returns None if the L-moments are invalid.

    Examples:
        - Estimate Normal parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [0.8, 1.5, 2.3, 3.7, 4.1, 5.6, 6.9]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=2)

          ```
          - Estimate Normal parameters
          ```python
          >>> params = Lmoments.normal(l_moments)
          >>> if params:
          ...       print(f"Mean: {params[0]}, Standard Deviation: {params[1]}")
          Mean: 3.557142857142857, Standard Deviation: 2.3885925705060047

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 2.0]

          ```
          - Estimate Normal parameters
          ```python
          >>> params = Lmoments.normal(l_moments)
          >>> if params:
          ...    print(f"Mean: {params[0]}, Standard Deviation: {params[1]}")
          Mean: 10.0, Standard Deviation: 3.5449077018110318

          ```

    Note:
        The Normal distribution has the probability density function:
        f(x) = (1/(sigma*sqrt(2*pi))) * exp(-((x-mu)^2/(2*sigma^2)))

        Where mu is the location parameter (mean) and sigma is the scale parameter (standard deviation).

        The method returns None if the second L-moment (l2) is less than or equal to zero,
        as this indicates invalid L-moments for the Normal distribution.

        The relationship between the second L-moment (l2) and the standard deviation (sigma) is:
        sigma = l2 * sqrt(pi)
    """
    if lmoments[1] <= 0:
        print(LMOMENTS_INVALID_ERROR)
        return None
    else:
        para = [lmoments[0], lmoments[1] * np.sqrt(np.pi)]
        return para


def generalized_normal(
    lmoments: list[float | int] | None,
) -> list[float | int] | None:
    """Estimate parameters for the Generalized Normal distribution.

    The Generalized Normal distribution (also known as the Generalized Error Distribution)
    is a three-parameter family of symmetric distributions that includes the normal
    distribution as a special case. It is characterized by location, scale, and shape parameters.

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
            - shape: Controls the shape of the distribution (kurtosis)
        Returns None if the L-moments are invalid.
        Returns [0, -1, 0] if the absolute value of the third L-moment is very large (>= 0.95).

    Examples:
        - Estimate Generalized Normal parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [0.8, 1.5, 2.3, 3.7, 4.1, 5.6, 6.9]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=3)

          ```
          - Estimate Generalized Normal parameters
          ```python
          >>> params = Lmoments.generalized_normal(l_moments)
          >>> if params:
          ...     print(f"Location: {params[0]}, Scale: {params[1]}, Shape: {params[2]}")
          Location: 3.32492783574149, Scale: 2.3507769936100464, Shape: -0.1956793126965343

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```

          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 2.0, 0.1]

          ```
          - Estimate Generalized Normal parameters
          ```python
          >>> params = Lmoments.generalized_normal(l_moments)
          >>> if params:
          ...    print(f"Location: {params[0]}, Scale: {params[1]}, Shape: {params[2]}")
          Location: 9.638928100246755, Scale: 3.4832722896983213, Shape: -0.2051440978274827

          ```

    Note:
        The Generalized Normal distribution has the probability density function:
        f(x) = (beta/(2*alpha*Gamma(1/beta))) * exp(-(|x-mu|/alpha)^beta)

        Where mu is the location parameter, alpha is the scale parameter, beta is the shape parameter,
        and Gamma is the gamma function.

        The method returns None if:
        - The second L-moment (l2) is less than or equal to zero
        - The absolute value of the third L-moment (l3) is greater than or equal to 1

        These conditions indicate invalid L-moments for the Generalized Normal distribution.

        When the absolute value of the third L-moment is very large (>= 0.95), the method
        returns [0, -1, 0] as a special case.
    """
    a0 = 0.20466534e01
    a1 = -0.36544371e01
    a2 = 0.18396733e01
    a3 = -0.20360244e00
    b1 = -0.20182173e01
    b2 = 0.12420401e01
    b3 = -0.21741801e00

    if lmoments is None:
        return None

    t3 = lmoments[2]
    if lmoments[1] <= 0 or abs(t3) >= 1:
        print(LMOMENTS_INVALID_ERROR)
        return None

    if abs(t3) >= 0.95:
        para: list[float | int] = [0, -1, 0]
        return para

    tt = t3**2
    g = (
        -t3
        * (a0 + tt * (a1 + tt * (a2 + tt * a3)))
        / (1 + tt * (b1 + tt * (b2 + tt * b3)))
    )
    exp_val = np.exp(0.5 * g**2)
    import scipy as sp

    a = lmoments[1] * g / (exp_val * sp.special.erf(0.5 * g))
    u = lmoments[0] + a * (exp_val - 1) / g
    para = [u, a, g]
    return para


def pearson_3(lmoments: list[float | int]) -> list[float | int]:
    """Estimate parameters for the Pearson Type III (PE3) distribution.

    The Pearson Type III distribution, also known as the three-parameter Gamma distribution,
    is a continuous probability distribution used in hydrology and other fields. It extends
    the Gamma distribution by adding a location parameter, allowing for greater flexibility.

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
            - shape: Controls the skewness of the distribution
        Returns [0, 0, 0] if the L-moments are invalid.

    Examples:
        - Estimate Pearson Type III parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [0.8, 1.5, 2.3, 3.7, 4.1, 5.6, 6.9]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=3)

          ```
          - Estimate Pearson Type III parameters
          ```python
          >>> params = Lmoments.pearson_3(l_moments)
          >>> print(f"Location: {params[0]}, Scale: {params[1]}, Shape: {params[2]}")
          Location: 3.557142857142857, Scale: 2.4141230211542557, Shape: 0.5833688019377993

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 2.0, 0.2]  # Positive skewness

          ```
          - Estimate Pearson Type III parameters
          ```python
          >>> params = Lmoments.pearson_3(l_moments)
          >>> print(f"Location: {params[0]}, Scale: {params[1]}, Shape: {params[2]}")
          Location: 10.0, Scale: 3.70994578417498, Shape: 1.2099737178678576

          ```

    Note:
        The Pearson Type III distribution has the probability density function:
        f(x) = ((x-mu)/beta)^(alpha-1) * exp(-(x-mu)/beta) / (beta * Gamma(alpha))

        Where mu is the location parameter, beta is the scale parameter, alpha is the shape parameter,
        and Gamma is the gamma function.

        The method returns [0, 0, 0] if:
        - The second L-moment (l2) is less than or equal to zero
        - The absolute value of the third L-moment (l3) is greater than or equal to 1

        These conditions indicate invalid L-moments for the Pearson Type III distribution.

        When the absolute value of the third L-moment is very small (<= 1e-6), the shape parameter
        is set to 0, resulting in a normal distribution.

        The sign of the shape parameter is determined by the sign of the third L-moment (l3),
        with negative l3 resulting in negative shape (left-skewed) and positive l3 resulting in
        positive shape (right-skewed).
    """
    small = 1e-6
    # Constants used in Minimax Approx:

    c1 = 0.2906
    c2 = 0.1882
    c3 = 0.0442
    d1 = 0.36067
    d2 = -0.59567
    d3 = 0.25361
    d4 = -2.78861
    d5 = 2.56096
    d6 = -0.77045

    t3 = abs(lmoments[2])
    if lmoments[1] <= 0 or t3 >= 1:
        para: list[float | int] = [0] * 3
        print(LMOMENTS_INVALID_ERROR)
        return para

    if t3 <= small:
        para = [lmoments[0], lmoments[1] * np.sqrt(np.pi), 0]
        return para

    if t3 >= (1.0 / 3):
        t = 1 - t3
        alpha = t * (d1 + t * (d2 + t * d3)) / (1 + t * (d4 + t * (d5 + t * d6)))
    else:
        t = 3 * np.pi * t3 * t3
        alpha = (1 + c1 * t) / (t * (1 + t * (c2 + t * c3)))

    rtalph = np.sqrt(alpha)
    beta = (
        np.sqrt(np.pi)
        * lmoments[1]
        * np.exp(_spsp.gammaln(alpha) - _spsp.gammaln(alpha + 0.5))
    )
    para = [lmoments[0], beta * rtalph, 2 / rtalph]
    if lmoments[2] < 0:
        para[2] = -para[2]

    return para
