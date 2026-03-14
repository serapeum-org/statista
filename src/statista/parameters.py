"""Parameters estimation module for statistical distributions.

This module provides functionality for estimating parameters of various statistical
distributions using L-moments method. L-moments are analogous to conventional moments
but can be estimated by linear combinations of order statistics (L-statistics).

The module contains the Lmoments class which implements methods for:
    - Calculating L-moments from data samples
    - Estimating parameters for various distributions using L-moments

Available distributions:
    - Generalized Extreme Value (GEV)
    - Gumbel
    - Exponential
    - Gamma
    - Generalized Logistic
    - Generalized Normal
    - Generalized Pareto
    - Normal
    - Pearson Type III
    - Wakeby
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy as sp
import scipy.special as _spsp
from numpy import ndarray

ninf = 1e-5
MAXIT = 20
EPS = 1e-6
SMALL = 1e-6
# Euler's constant
EU = 0.577215664901532861

LMOMENTS_INVALID_ERROR = "L-Moments Invalid"


class Lmoments:
    """Class for calculating L-moments and estimating distribution parameters.

    L-moments are statistics used to summarize the shape of a probability distribution.
    Introduced by Hosking (1990), they are analogous to conventional moments but can be
    estimated by linear combinations of order statistics (L-statistics).

    L-moments have several advantages over conventional moments:
        - They can characterize a wider range of distributions
        - They are more robust to outliers in the data
        - They are less subject to bias in estimation
        - They approximate their asymptotic normal distribution more closely

    The L-moments of order r are denoted by λr and defined as:
        λ1 = α0 = β0                                      (mean)
        λ2 = α0 - 2α1 = 2β1 - β0                          (L-scale)
        λ3 = α0 - 6α1 + 6α2 = 6β2 - 6β1 + β0              (L-skewness)
        λ4 = α0 - 12α1 + 30α2 - 20α3 = 20β3 - 30β2 + 12β1 - β0  (L-kurtosis)

    Attributes:
        data: The input data for which L-moments will be calculated.

    Examples:
        - Basic usage to calculate L-moments:
          ```python
          >>> import numpy as np
          >>> from statista.parameters import Lmoments

          ```
          - Create sample data
          ```python
          >>> data = np.random.normal(loc=10, scale=2, size=100)

          ```
          - Initialize Lmoments with the data
          ```python
          >>> lmom = Lmoments(data)

          ```
          - Calculate the first 4 L-moments
          ```python
          >>> l_moments = lmom.calculate(nmom=4)
          >>> print(l_moments) #doctest: +SKIP
          [np.float64(10.166325002460868), np.float64(1.0521820576994685), np.float64(0.0015331221093457831), np.float64(0.16527008148561118)]

          ```

        - Estimating distribution parameters using L-moments:
          ```python
          >>> import numpy as np
          >>> from statista.parameters import Lmoments

          ```
          - Create sample data
          ```python
          >>> data = np.random.normal(loc=10, scale=2, size=100)

          ```
          - Calculate L-moments
          ```python
          >>> lmom = Lmoments(data)
          >>> l_moments = lmom.calculate(nmom=3)

          ```
          - Estimate parameters for normal distribution
          ```python
          >>> params = Lmoments.normal(l_moments)
          >>> print(f"Location: {params[0]}, Scale: {params[1]}") #doctest: +SKIP
          Location: 9.531376405859064, Scale: 2.074884534193713

          ```
    """

    def __init__(self, data):
        """Initialize the Lmoments class with data.

        Args:
            data: A sequence of numerical values for which L-moments will be calculated.
                Can be a list, numpy array, or any iterable containing numeric values.

        Examples:
            - Initialize with a list of values:
              ```python
              >>> from statista.parameters import Lmoments
              >>> data = [1.2, 3.4, 5.6, 7.8, 9.0]
              >>> lmom = Lmoments(data)

              ```

            - Initialize with a numpy array:
              ```python
              >>> import numpy as np
              >>> from statista.parameters import Lmoments
              >>> data = np.random.normal(loc=10, scale=2, size=100)
              >>> lmom = Lmoments(data)

              ```
        """
        self.data = data

    def calculate(self, nmom=5):
        """Calculate the L-moments for the data.

        This method calculates the first `nmom` L-moments of the data. For nmom <= 5,
        it uses the more efficient `_samlmusmall` method. For nmom > 5, it uses the
        more general `_samlmularge` method.

        Args:
            nmom: An integer specifying the number of L-moments to calculate.
                Default is 5.

        Returns:
            A list containing the first `nmom` L-moments if nmom > 1.
            If nmom=1, returns only the first L-moment (the mean) as a float.

        Raises:
            ValueError: If nmom <= 0 or if the length of data is less than nmom.

        Examples:
            - Calculate the first 4 L-moments:
              ```python
              >>> from statista.parameters import Lmoments
              >>> data = [1.2, 3.4, 5.6, 7.8, 9.0]
              >>> lmom = Lmoments(data)
              >>> l_moments = lmom.calculate(nmom=4)
              >>> print(l_moments)  # Output: [5.4, 1.68, 0.1, 0.05]
              [np.float64(5.4), 2.0, -0.09999999999999988, -0.09999999999999998]

              ```

            - Calculate only the first L-moment (mean):
              ```python
              >>> from statista.parameters import Lmoments
              >>> data = [1.2, 3.4, 5.6, 7.8, 9.0]
              >>> lmom = Lmoments(data)
              >>> mean = lmom.calculate(nmom=1)
              >>> print(mean)  # Output: 5.4
              [np.float64(5.4)]

              ```
        """
        if nmom <= 5:
            var = self._samlmusmall(nmom)
        else:
            var = self._samlmularge(nmom)

        return var

    @staticmethod
    def _comb(n, k):
        """Calculate the binomial coefficient (n choose k).

        This method computes the binomial coefficient, which is the number of ways
        to choose k items from a set of n items without regard to order.

        Args:
            n: A non-negative integer representing the total number of items.
            k: A non-negative integer representing the number of items to choose.

        Returns:
            An integer representing the binomial coefficient (n choose k).
            Returns 0 if k > n, n < 0, or k < 0.

        Examples:
            - Calculate 5 choose 2:
              ```python
              >>> from statista.parameters import Lmoments
              >>> result = Lmoments._comb(5, 2)
              >>> print(result)  # Output: 10
              10

              ```

            - Calculate 10 choose 3:
              ```python
              >>> from statista.parameters import Lmoments
              >>> result = Lmoments._comb(10, 3)
              >>> print(result)  # Output: 120
              120

              ```

            - Invalid inputs return 0:
              ```python
              >>> from statista.parameters import Lmoments
              >>> result = Lmoments._comb(3, 5)  # k > n
              >>> print(result)
              0
              >>> result = Lmoments._comb(-1, 2)  # n < 0
              >>> print(result)
              0

              ```
        """
        if (k > n) or (n < 0) or (k < 0):
            val = 0
        else:
            val = 1
            for j in range(min(k, n - k)):
                val = (val * (n - j)) // (j + 1)  # // is floor division
        return val

    def _samlmularge(self, nmom: int = 5) -> list[ndarray | float | int | Any]:
        """Calculate L-moments for large samples or higher order moments.

        This method implements a general algorithm for calculating L-moments of any order.
        It is more computationally intensive than _samlmusmall but works for any number
        of moments.

        Args:
            nmom: An integer specifying the number of L-moments to calculate.
                Default is 5.

        Returns:
            A list containing the first `nmom` L-moments if nmom > 1.
            If nmom=1, returns only the first L-moment (the mean) as a float.

        Raises:
            ValueError: If nmom <= 0 or if the length of data is less than nmom.

        Examples:
            - Calculate the first 6 L-moments:
              ```python
              >>> from statista.parameters import Lmoments
              >>> data = [1.2, 3.4, 5.6, 7.8, 9.0, 2.3, 4.5, 6.7, 8.9]
              >>> lmom = Lmoments(data)
              >>> l_moments = lmom._samlmularge(nmom=6)
              >>> print(l_moments)
              [5.488888888888888, 1.722222222222222, -0.06451612903225806, -0.0645161290322581, -0.0645161290322581, -0.06451612903225817]

              ```

        Note:
            This method is primarily used internally by the `calculate` method when
            nmom > 5. For most applications, use the `calculate` method instead.
        """
        x = self.data
        if nmom <= 0:
            raise ValueError("Invalid number of Sample L-Moments")

        x = sorted(x)
        n = len(x)

        if n < nmom:
            raise ValueError("Insufficient length of data for specified nmoments")

        # Calculate first order
        coef_l1 = 1.0 / self._comb(n, 1)
        sum_l1 = sum(x)
        lmoments = [coef_l1 * sum_l1]

        if nmom == 1:
            return lmoments[0]

        # Setup comb table, where comb[i][x] refers to comb(x,i)
        comb: list[list[float | int]] = []
        for i in range(1, nmom):
            comb.append([])
            for j in range(n):
                comb[-1].append(self._comb(j, i))

        for mom in range(2, nmom + 1):
            coefl = 1.0 / mom * 1.0 / self._comb(n, mom)
            xtrans = []
            for i in range(0, n):
                coef_temp: list[float | int] = []
                for _ in range(0, mom):
                    coef_temp.append(1)

                for j in range(0, mom - 1):
                    coef_temp[j] = coef_temp[j] * comb[mom - j - 2][i]

                for j in range(1, mom):
                    coef_temp[j] = coef_temp[j] * comb[j - 1][n - i - 1]

                for j in range(0, mom):
                    coef_temp[j] = coef_temp[j] * self._comb(mom - 1, j)

                for j in range(0, int(0.5 * mom)):
                    coef_temp[j * 2 + 1] = -coef_temp[j * 2 + 1]
                coef_sum: Any = sum(coef_temp)
                xtrans.append(x[i] * coef_sum)

            if mom > 2:
                lmoments.append(coefl * sum(xtrans) / lmoments[1])
            else:
                lmoments.append(coefl * sum(xtrans))
        return lmoments

    def _samlmusmall(self, nmom: int = 5) -> list[ndarray | float | int | Any] | None:
        """Calculate L-moments for small samples or lower order moments.

        This method implements an optimized algorithm for calculating L-moments up to order 5.
        It is more efficient than _samlmularge for nmom <= 5.

        Args:
            nmom: An integer specifying the number of L-moments to calculate.
                Must be between 1 and 5 (inclusive). Default is 5.

        Returns:
            A list containing the first `nmom` L-moments if nmom > 1.
            If nmom=1, returns only the first L-moment (the mean) as a float.

        Raises:
            ValueError: If nmom <= 0 or if the length of data is less than nmom.

        Examples:
            - Calculate the first 3 L-moments:
              ```python
              >>> from statista.parameters import Lmoments
              >>> data = [1.2, 3.4, 5.6, 7.8, 9.0]
              >>> lmom = Lmoments(data)
              >>> l_moments = lmom._samlmusmall(nmom=3)
              >>> print(l_moments)
              [np.float64(5.4), 2.0, -0.09999999999999988]

              ```

        Note:
            This method is primarily used internally by the `calculate` method when
            nmom <= 5. For most applications, use the `calculate` method instead.

            The implementation uses a direct formula for each L-moment order, which
            is more efficient than the general algorithm used in _samlmularge.
        """
        sample = self.data

        if nmom <= 0:
            raise ValueError("Invalid number of Sample L-Moments")

        sample = sorted(sample)
        n = len(sample)

        if n < nmom:
            raise ValueError("Insufficient length of data for specified nmoments")

        l_moment_1 = np.mean(sample)
        if nmom == 1:
            return [l_moment_1]

        comb1 = range(0, n)
        comb2 = range(n - 1, -1, -1)

        coefl2 = 0.5 * 1.0 / self._comb(n, 2)
        xtrans = []
        for i in range(0, n):
            coef_temp = comb1[i] - comb2[i]
            xtrans.append(coef_temp * sample[i])

        l_moment_2 = coefl2 * sum(xtrans)

        if nmom == 2:
            return [l_moment_1, l_moment_2]

        # Calculate Third order
        # comb terms appear elsewhere, this will decrease calc time
        # for nmom > 2, and shouldn't decrease time for nmom == 2
        # comb3 = comb(i-1,2)
        # comb4 = comb3.reverse()
        comb3 = []
        comb4: list[float | int] = []
        for i in range(0, n):
            comb_temp = self._comb(i, 2)
            comb3.append(comb_temp)
            comb4.insert(0, comb_temp)

        coefl3 = 1.0 / 3 * 1.0 / self._comb(n, 3)
        xtrans = []
        for i in range(0, n):
            coef_temp = comb3[i] - 2 * comb1[i] * comb2[i] + comb4[i]
            xtrans.append(coef_temp * sample[i])

        l_moment_3 = coefl3 * sum(xtrans) / l_moment_2

        if nmom == 3:
            return [l_moment_1, l_moment_2, l_moment_3]

        # Calculate Fourth order
        comb5 = []
        comb6: list[float | int] = []
        for i in range(0, n):
            comb_temp = self._comb(i, 3)
            comb5.append(comb_temp)
            comb6.insert(0, comb_temp)

        coefl4 = 1.0 / 4 * 1.0 / self._comb(n, 4)
        xtrans = []
        for i in range(0, n):
            coef_temp = (
                comb5[i] - 3 * comb3[i] * comb2[i] + 3 * comb1[i] * comb4[i] - comb6[i]
            )
            xtrans.append(coef_temp * sample[i])

        l_moment_4 = coefl4 * sum(xtrans) / l_moment_2

        if nmom == 4:
            return [l_moment_1, l_moment_2, l_moment_3, l_moment_4]

        # Calculate Fifth order
        comb7 = []
        comb8: list[float | int] = []
        for i in range(0, n):
            comb_temp = self._comb(i, 4)
            comb7.append(comb_temp)
            comb8.insert(0, comb_temp)

        coefl5 = 1.0 / 5 * 1.0 / self._comb(n, 5)
        xtrans = []
        for i in range(0, n):
            coef_temp = (
                comb7[i]
                - 4 * comb5[i] * comb2[i]
                + 6 * comb3[i] * comb4[i]
                - 4 * comb1[i] * comb6[i]
                + comb8[i]
            )
            xtrans.append(coef_temp * sample[i])

        l_moment_5 = coefl5 * sum(xtrans) / l_moment_2

        if nmom == 5:
            return [l_moment_1, l_moment_2, l_moment_3, l_moment_4, l_moment_5]
        return None

    @staticmethod
    def gev(lmoments: list[float | int]) -> list[float | int]:
        """Estimate parameters for the Generalized Extreme Value (GEV) distribution.

        The Generalized Extreme Value distribution combines the Gumbel, Fréchet, and Weibull
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
            Exception: If the parameter estimation algorithm fails to converge.

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
            F(x) = exp(-[1 + ξ((x-μ)/σ)]^(-1/ξ)) for ξ ≠ 0
            F(x) = exp(-exp(-(x-μ)/σ)) for ξ = 0 (Gumbel case)

            Where ξ is the shape parameter, μ is the location parameter, and σ is the scale parameter.
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

    @staticmethod
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
            F(x) = exp(-exp(-(x-μ)/β))

            Where μ is the location parameter and β is the scale parameter.

            The Gumbel distribution is a special case of the GEV distribution with shape parameter = 0.
        """
        if lmoments[1] <= 0:
            raise ValueError(LMOMENTS_INVALID_ERROR)
        else:
            para2 = lmoments[1] / np.log(2)
            para1 = lmoments[0] - EU * para2
            para = [para1, para2]
            return para

    @staticmethod
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
            f(x) = (1/β) * exp(-(x-μ)/β) for x ≥ μ

            Where μ is the location parameter and β is the scale parameter.

            The method returns None if the second L-moment (l2) is less than or equal to zero,
            as this indicates invalid L-moments for the Exponential distribution.
        """
        if lmoments[1] <= 0:
            print(LMOMENTS_INVALID_ERROR)
            para = None
        else:
            para = [lmoments[0] - 2 * lmoments[1], 2 * lmoments[1]]

        return para

    @staticmethod
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
            f(x) = (x^(α-1) * e^(-x/β)) / (β^α * Γ(α)) for x > 0

            Where α is the shape parameter, β is the scale parameter, and Γ is the gamma function.

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

    @staticmethod
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
            F(x) = 1 / (1 + exp(-((x-μ)/α))^(1/k)) for k ≠ 0
            F(x) = 1 / (1 + exp(-(x-μ)/α)) for k = 0

            Where μ is the location parameter, α is the scale parameter, and k is the shape parameter.

            The method returns None if:
            - The second L-moment (l2) is less than or equal to zero
            - The absolute value of the negative third L-moment (g = -l3) is greater than or equal to 1

            These conditions indicate invalid L-moments for the Generalized Logistic distribution.

            When the absolute value of g is very small (≤ 1e-6), the shape parameter is set to 0,
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

    @staticmethod
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
            Returns [0, -1, 0] if the absolute value of the third L-moment is very large (≥ 0.95).

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
            f(x) = (β/(2αΓ(1/β))) * exp(-(|x-μ|/α)^β)

            Where μ is the location parameter, α is the scale parameter, β is the shape parameter,
            and Γ is the gamma function.

            The method returns None if:
            - The second L-moment (l2) is less than or equal to zero
            - The absolute value of the third L-moment (l3) is greater than or equal to 1

            These conditions indicate invalid L-moments for the Generalized Normal distribution.

            When the absolute value of the third L-moment is very large (≥ 0.95), the method
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
        a = lmoments[1] * g / (exp_val * sp.special.erf(0.5 * g))
        u = lmoments[0] + a * (exp_val - 1) / g
        para = [u, a, g]
        return para

    @staticmethod
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
            F(x) = 1 - [1 - k(x-μ)/α]^(1/k) for k ≠ 0
            F(x) = 1 - exp(-(x-μ)/α) for k = 0

            Where μ is the location parameter, α is the scale parameter, and k is the shape parameter.

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

    @staticmethod
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
            f(x) = (1/(σ√(2π))) * exp(-((x-μ)²/(2σ²)))

            Where μ is the location parameter (mean) and σ is the scale parameter (standard deviation).

            The method returns None if the second L-moment (l2) is less than or equal to zero,
            as this indicates invalid L-moments for the Normal distribution.

            The relationship between the second L-moment (l2) and the standard deviation (σ) is:
            σ = l2 * √π
        """
        if lmoments[1] <= 0:
            print(LMOMENTS_INVALID_ERROR)
            return None
        else:
            para = [lmoments[0], lmoments[1] * np.sqrt(np.pi)]
            return para

    @staticmethod
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
            f(x) = ((x-μ)/β)^(α-1) * exp(-(x-μ)/β) / (β * Γ(α))

            Where μ is the location parameter, β is the scale parameter, α is the shape parameter,
            and Γ is the gamma function.

            The method returns [0, 0, 0] if:
            - The second L-moment (l2) is less than or equal to zero
            - The absolute value of the third L-moment (l3) is greater than or equal to 1

            These conditions indicate invalid L-moments for the Pearson Type III distribution.

            When the absolute value of the third L-moment is very small (≤ 1e-6), the shape parameter
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

    @staticmethod
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


class ConvergenceError(Exception):
    """Custom exception for convergence errors in L-moment calculations."""

    pass
