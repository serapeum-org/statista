"""Core L-moments calculation class.

This module provides the Lmoments class for calculating L-moments from data samples
and estimating distribution parameters using L-moments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from statista.parameters.extreme_value import (
    ConvergenceError,
    gev as _gev,
    gumbel as _gumbel,
    generalized_pareto as _generalized_pareto,
)
from statista.parameters.normal_family import (
    normal as _normal,
    generalized_normal as _generalized_normal,
    pearson_3 as _pearson_3,
)
from statista.parameters.other import (
    exponential as _exponential,
    gamma as _gamma,
    generalized_logistic as _generalized_logistic,
    wakeby as _wakeby,
)


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

    The L-moments of order r are denoted by lambda_r and defined as:
        lambda_1 = alpha_0 = beta_0                                      (mean)
        lambda_2 = alpha_0 - 2*alpha_1 = 2*beta_1 - beta_0              (L-scale)
        lambda_3 = alpha_0 - 6*alpha_1 + 6*alpha_2 = 6*beta_2 - 6*beta_1 + beta_0 (L-skewness)
        lambda_4 = alpha_0 - 12*alpha_1 + 30*alpha_2 - 20*alpha_3 = 20*beta_3 - 30*beta_2 + 12*beta_1 - beta_0 (L-kurtosis)

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

    # Static method delegates to module-level functions for backward compatibility
    @staticmethod
    def gev(lmoments: list[float | int]) -> list[float | int]:
        """Estimate parameters for the GEV distribution. See extreme_value.gev for details."""
        return _gev(lmoments)

    @staticmethod
    def gumbel(lmoments: list[float | int]) -> list[float | int]:
        """Estimate parameters for the Gumbel distribution. See extreme_value.gumbel for details."""
        return _gumbel(lmoments)

    @staticmethod
    def exponential(lmoments: list[float | int]) -> list[float | int] | None:
        """Estimate parameters for the Exponential distribution. See other.exponential for details."""
        return _exponential(lmoments)

    @staticmethod
    def gamma(lmoments: list[float | int]) -> list[float | int] | None:
        """Estimate parameters for the Gamma distribution. See other.gamma for details."""
        return _gamma(lmoments)

    @staticmethod
    def generalized_logistic(lmoments: list[float | int]) -> list[float | int] | None:
        """Estimate parameters for the Generalized Logistic distribution. See other.generalized_logistic for details."""
        return _generalized_logistic(lmoments)

    @staticmethod
    def generalized_normal(lmoments: list[float | int] | None) -> list[float | int] | None:
        """Estimate parameters for the Generalized Normal distribution. See normal_family.generalized_normal for details."""
        return _generalized_normal(lmoments)

    @staticmethod
    def generalized_pareto(lmoments: list[float | int]) -> list[float] | None:
        """Estimate parameters for the Generalized Pareto distribution. See extreme_value.generalized_pareto for details."""
        return _generalized_pareto(lmoments)

    @staticmethod
    def normal(lmoments: list[float | int]) -> list[float | int] | None:
        """Estimate parameters for the Normal distribution. See normal_family.normal for details."""
        return _normal(lmoments)

    @staticmethod
    def pearson_3(lmoments: list[float | int]) -> list[float | int]:
        """Estimate parameters for the Pearson Type III distribution. See normal_family.pearson_3 for details."""
        return _pearson_3(lmoments)

    @staticmethod
    def wakeby(lmoments: list[float | int]) -> list[float | int] | None:
        """Estimate parameters for the Wakeby distribution. See other.wakeby for details."""
        return _wakeby(lmoments)
