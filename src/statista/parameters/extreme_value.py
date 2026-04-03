"""L-moments parameter estimation for extreme value distributions.

Distributions: GEV, Gumbel, Generalized Pareto.
"""

from __future__ import annotations

import numpy as np
import scipy as sp

ninf = 1e-5
MAXIT = 20
EPS = 1e-6
# Euler's constant
EU = 0.577215664901532861

LMOMENTS_INVALID_ERROR = "L-Moments Invalid"


class ConvergenceError(Exception):
    """Custom exception for convergence errors in L-moment calculations."""

    pass


def gev(lmoments: list[float | int]) -> list[float | int]:
    """Estimate parameters for the Generalized Extreme Value (GEV) distribution.

    The Generalized Extreme Value distribution combines the Gumbel, Frechet, and Weibull
    distributions into a single family to model extreme values. The distribution is
    characterized by three parameters: shape, location, and scale.

    Args:
        lmoments: A list of L-moments [l1, l2, l3, ...] where:
            - l1 is the mean (first L-moment)
            - l2 is the L-scale (second L-moment)
            - l3 is the L-skewness (third L-moment)
            At least 3 L-moments must be provided.

    Returns:
        A list of distribution parameters [shape, location, scale] where:
            - shape: Controls the tail behavior of the distribution
            - location: Shifts the distribution along the x-axis
            - scale: Controls the spread of the distribution

    Raises:
        ValueError: If the L-moments are invalid (l2 <= 0 or |l3| >= 1).
        ConvergenceError: If the parameter estimation algorithm fails to converge.

    Examples:
        - Estimate GEV parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [10.2, 15.7, 20.3, 25.9, 30.1, 35.6, 40.2]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=3)

          ```
          - Estimate GEV parameters
          ```python
          >>> params = Lmoments.gev(l_moments)
          >>> print(f"Shape: {params[0]}, Location: {params[1]}, Scale: {params[2]}")
          Shape: 0.3055099485469931, Location: 21.413657588990556, Scale: 11.868352699813734

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 2.0, 0.1]

          ```
          - Estimate GEV parameters
          ```python
          >>> params = Lmoments.gev(l_moments)
          >>> print(f"Shape: {params[0]}, Location: {params[1]}, Scale: {params[2]}")
          Shape: 0.11189502871959642, Location: 8.490058310239982, Scale: 3.1676863588272224

          ```

    Note:
        The GEV distribution has the cumulative distribution function:
        F(x) = exp(-[1 + xi((x-mu)/sigma)]^(-1/xi)) for xi != 0
        F(x) = exp(-exp(-(x-mu)/sigma)) for xi = 0 (Gumbel case)

        Where xi is the shape parameter, mu is the location parameter, and sigma is the scale parameter.
    """
    dl2 = np.log(2)
    dl3 = np.log(3)
    # COEFFICIENTS OF RATIONAL-FUNCTION APPROXIMATIONS FOR XI
    a0 = 0.28377530
    a1 = -1.21096399
    a2 = -2.50728214
    a3 = -1.13455566
    a4 = -0.07138022
    b1 = 2.06189696
    b2 = 1.31912239
    b3 = 0.25077104
    c1 = 1.59921491
    c2 = -0.48832213
    c3 = 0.01573152
    d1 = -0.64363929
    d2 = 0.08985247

    t3 = lmoments[2]
    # if std <= 0 or third moment > 1
    if lmoments[1] <= 0 or abs(t3) >= 1:
        raise ValueError(LMOMENTS_INVALID_ERROR)

    if t3 <= 0:
        G = (a0 + t3 * (a1 + t3 * (a2 + t3 * (a3 + t3 * a4)))) / (
            1 + t3 * (b1 + t3 * (b2 + t3 * b3))
        )
        if t3 >= -0.8:
            shape = G
            gam = np.exp(sp.special.gammaln(1 + G))
            scale = lmoments[1] * G / (gam * (1 - 2 ** (-G)))
            loc = lmoments[0] - scale * (1 - gam) / G
            para = [shape, loc, scale]
            return para

        if t3 <= -0.97:
            G = 1 - np.log(1 + t3) / dl2

        t0 = (t3 + 3) * 0.5

        for _ in range(1, MAXIT):
            x2 = 2 ** (-G)
            x3 = 3 ** (-G)
            xx2 = 1 - x2
            xx3 = 1 - x3
            t = xx3 / xx2
            deriv = (xx2 * x3 * dl3 - xx3 * x2 * dl2) / (xx2**2)
            gold = G
            G = G - (t - t0) / deriv
            if abs(G - gold) <= EPS * G:
                shape = G
                gam = np.exp(sp.special.gammaln(1 + G))
                scale = lmoments[1] * G / (gam * (1 - 2 ** (-G)))
                loc = lmoments[0] - scale * (1 - gam) / G
                para = [shape, loc, scale]
                return para
        raise ConvergenceError("Iteration has not converged")
    else:
        Z = 1 - t3
        G = (-1 + Z * (c1 + Z * (c2 + Z * c3))) / (1 + Z * (d1 + Z * d2))
        if abs(G) < ninf:
            # Gumbel
            scale = lmoments[1] / dl2
            loc = lmoments[0] - EU * scale
            para = [0, loc, scale]
        else:
            # GEV
            shape = G
            gam = np.exp(sp.special.gammaln(1 + G))
            scale = lmoments[1] * G / (gam * (1 - 2 ** (-G)))
            loc = lmoments[0] - scale * (1 - gam) / G
            # multiply the shape by -1 to follow the + ve shape parameter equation (+ve value means heavy tail)
            # para = [-1 * shape, loc, scale]
            para = [shape, loc, scale]

        return para


def gumbel(lmoments: list[float | int]) -> list[float | int]:
    """Estimate parameters for the Gumbel distribution.

    The Gumbel distribution (also known as the Type I Extreme Value distribution) is
    used to model the maximum or minimum of a number of samples of various distributions.
    It is characterized by two parameters: location and scale.

    Args:
        lmoments: A list of L-moments [l1, l2, ...] where:
            - l1 is the mean (first L-moment)
            - l2 is the L-scale (second L-moment)
            At least 2 L-moments must be provided.

    Returns:
        A list of distribution parameters [location, scale] where:
            - location: Shifts the distribution along the x-axis
            - scale: Controls the spread of the distribution

    Raises:
        ValueError: If the L-moments are invalid (l2 <= 0).

    Examples:
        - Estimate Gumbel parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [10.2, 15.7, 20.3, 25.9, 30.1, 35.6, 40.2]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=2)

          ```
          - Estimate Gumbel parameters
          ```python
          >>> params = Lmoments.gumbel(l_moments)
          >>> print(f"Location: {params[0]}, Scale: {params[1]}")
          Location: 19.892792078673775, Scale: 9.590487033719015

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 2.0]

          ```
          - Estimate Gumbel parameters
          ```python
          >>> params = Lmoments.gumbel(l_moments)
          >>> print(f"Location: {params[0]}, Scale: {params[1]}")
          Location: 8.334507645446266, Scale: 2.8853900817779268

          ```

    Note:
        The Gumbel distribution has the cumulative distribution function:
        F(x) = exp(-exp(-(x-mu)/beta))

        Where mu is the location parameter and beta is the scale parameter.

        The Gumbel distribution is a special case of the GEV distribution with shape parameter = 0.
    """
    if lmoments[1] <= 0:
        raise ValueError(LMOMENTS_INVALID_ERROR)
    else:
        para2 = lmoments[1] / np.log(2)
        para1 = lmoments[0] - EU * para2
        para = [para1, para2]
        return para


def generalized_pareto(
    lmoments: list[float | int],
) -> list[float] | None:
    """Estimate parameters for the Generalized Pareto distribution.

    The Generalized Pareto distribution is a flexible three-parameter family of distributions
    used to model the tails of other distributions. It is characterized by location, scale,
    and shape parameters.

    Args:
        lmoments: A list of L-moments [l1, l2, l3, ...] where:
            - l1 is the mean (first L-moment)
            - l2 is the L-scale (second L-moment)
            - l3 is the L-skewness (third L-moment)
            At least 3 L-moments must be provided.

    Returns:
        A list of distribution parameters [location, scale, shape] where:
            - location: Shifts the distribution along the x-axis (lower bound)
            - scale: Controls the spread of the distribution
            - shape: Controls the tail behavior of the distribution
        Returns None if the L-moments are invalid.

    Examples:
        - Estimate Generalized Pareto parameters from L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Calculate L-moments from data
          ```python
          >>> data = [0.8, 1.5, 2.3, 3.7, 4.1, 5.6, 6.9]
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=3)

          ```
          - Estimate Generalized Pareto parameters
          ```python
          >>> params = Lmoments.generalized_pareto(l_moments)
          >>> if params:
          ...   print(f"Location: {params[0]}, Scale: {params[1]}, Shape: {params[2]}")
          Location: -0.016221198156681993, Scale: 5.901814181656014, Shape: 0.6516129032258066

          ```

        - Using predefined L-moments:
          ```python
          >>> from statista.parameters import Lmoments

          ```
          - Predefined L-moments
          ```python
          >>> l_moments = [10.0, 2.0, 0.1]

          ```
          - Estimate Generalized Pareto parameters
          ```python
          >>> params = Lmoments.generalized_pareto(l_moments)
          >>> if params:
          ...     print(f"Location: {params[0]}, Scale: {params[1]}, Shape: {params[2]}")
          Location: 4.7272727272727275, Scale: 8.628099173553718, Shape: 0.6363636363636362

          ```

    Note:
        The Generalized Pareto distribution has the cumulative distribution function:
        F(x) = 1 - [1 - k(x-mu)/alpha]^(1/k) for k != 0
        F(x) = 1 - exp(-(x-mu)/alpha) for k = 0

        Where mu is the location parameter, alpha is the scale parameter, and k is the shape parameter.

        The method returns None if:
        - The second L-moment (l2) is less than or equal to zero
        - The absolute value of the third L-moment (l3) is greater than or equal to 1

        These conditions indicate invalid L-moments for the Generalized Pareto distribution.

        The shape parameter determines the tail behavior:
        - k < 0: The distribution has an upper bound
        - k = 0: The distribution is exponential
        - k > 0: The distribution has a heavy upper tail
    """
    t3 = lmoments[2]
    if lmoments[1] <= 0:
        print(LMOMENTS_INVALID_ERROR)
        return None

    if abs(t3) >= 1:
        print(LMOMENTS_INVALID_ERROR)
        return None

    g = (1 - 3 * t3) / (1 + t3)

    para3 = g
    para2 = (1 + g) * (2 + g) * lmoments[1]
    para1 = lmoments[0] - para2 / (1 + g)
    para = [para1, para2, para3]
    return para
