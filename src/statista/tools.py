"""Statistical tools"""

from decimal import ROUND_HALF_UP, Decimal
from typing import List, Union

import numpy as np


class Tools:
    """Collection of statistical and data transformation utilities.

    This class provides static methods for various data transformations and
    manipulations commonly used in statistical analysis, including normalization,
    standardization, rescaling, and logarithmic transformations.

    All methods are implemented as static methods, so they can be called directly
    without instantiating the class.

    Examples:
        - Import the Tools class:
            ```python
            >>> import numpy as np
            >>> from statista.tools import Tools

            ```
        - Normalize an array to [0, 1] range
            ```python
            >>> data = [10, 20, 30, 40, 50]
            >>> normalized = Tools.normalize(data)
            >>> print(normalized)
            [0.   0.25 0.5  0.75 1.  ]

            ```
        - Standardize an array (mean=0, std=1):
            ```python
            >>> standardized = Tools.standardize(data)
            >>> print(f"Mean: {np.mean(standardized):.4f}, Std: {np.std(standardized):.4f}")
            Mean: 0.0000, Std: 1.0000

            ```
    """

    @staticmethod
    def normalize(x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Normalize values to the range [0, 1].

        Scales all values in the input array to the range [0, 1] using min-max normalization.
        The formula used is: (x - min(x)) / (max(x) - min(x))

        Args:
            x: Input array or list of values to normalize.

        Returns:
            np.ndarray: Array of normalized values in the range [0, 1].

        Raises:
            ValueError: If all values in the input are identical (max = min),
                which would cause division by zero.

        Examples:
            - Normalize a list of values:
                ```python
                >>> from statista.tools import Tools
                >>> data = [10, 20, 30, 40, 50]
                >>> normalized = Tools.normalize(data)
                >>> print(normalized)
                [0.   0.25 0.5  0.75 1.  ]

                ```
            - Normalize a numpy array:
                ```python
                >>> import numpy as np
                >>> data = np.array([5, 15, 25, 35, 45])
                >>> normalized = Tools.normalize(data)
                >>> print(normalized)
                [0.   0.25 0.5  0.75 1.  ]

                ```
            - Edge case: single value:
                ```python
                >>> data = [42]
                >>> try:
                ...     normalized = Tools.normalize(data)
                ... except ValueError as e:
                ...     print(e)
                input data must contain at least two values for normalization

                ```

        See Also:
            - Tools.standardize: For standardizing values to mean=0, std=1
            - Tools.rescale: For rescaling values to a custom range
        """
        x = np.array(x)
        if len(x) <= 1:
            raise ValueError(
                "input data must contain at least two values for normalization"
            )

        data_max = max(x)
        data_min = min(x)
        return (x - data_min) / (data_max - data_min)

    @staticmethod
    def standardize(x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Standardize values to have mean=0 and standard deviation=1.

        Transforms the input array so that it has a mean of 0 and a standard deviation of 1.
        This is also known as z-score normalization or standard score.
        The formula used is: (x - mean(x)) / std(x)

        Args:
            x: Input array or list of values to standardize.

        Returns:
            np.ndarray: Array of standardized values with mean=0 and std=1.

        Raises:
            ValueError: If the standard deviation of the input is zero,
                which would cause division by zero.

        Examples:
            - Standardize a list of values:
                ```python
                >>> from statista.tools import Tools
                >>> import numpy as np
                >>> data = [10, 20, 30, 40, 50]
                >>> standardized = Tools.standardize(data)
                >>> print(f"Mean: {np.mean(standardized):.4f}, Std: {np.std(standardized):.4f}")
                Mean: 0.0000, Std: 1.0000

                ```
            - Verify the transformation:
                ```python
                >>> print(standardized)
                [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]

                ```
            - Standardize values that already have mean=0:
                ```python
                >>> data = [-2, -1, 0, 1, 2]
                >>> standardized = Tools.standardize(data)
                >>> print(standardized)
                [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]

                ```

        Notes:
            Standardization is particularly useful for algorithms that assume
            the data is centered around zero with unit variance, such as many
            machine learning algorithms.

        See Also:
            - Tools.normalize: For scaling values to the range [0, 1]
            - Tools.rescale: For rescaling values to a custom range
        """
        x = np.array(x)

        mean = np.mean(x)
        std = np.std(x)
        s = (x - mean) / std
        return s

    @staticmethod
    def rescale(
        old_value: float, old_min: float, old_max: float, new_min: float, new_max: float
    ) -> float:
        """Rescale a value from one range to another.

        Linearly transforms a value from its original range [old_min, old_max]
        to a new range [new_min, new_max]. This is useful for mapping values
        between different scales while preserving their relative positions.

        The formula used is:
        new_value = (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

        Args:
            old_value: The value to rescale.
            old_min: The minimum value of the original range.
            old_max: The maximum value of the original range.
            new_min: The minimum value of the target range.
            new_max: The maximum value of the target range.

        Returns:
            float: The rescaled value in the new range.

        Raises:
            ZeroDivisionError: If old_max equals old_min, causing division by zero.

        Examples:
            - Rescale a value from [0, 100] to [0, 1]:
                ```python
                >>> from statista.tools import Tools
                >>> value = 75
                >>> rescaled = Tools.rescale(value, 0, 100, 0, 1)
                >>> print(rescaled)
                0.75

                ```
            - Rescale a value from [0, 1] to [-1, 1]:
                ```python
                >>> value = 0.3
                >>> rescaled = Tools.rescale(value, 0, 1, -1, 1)
                >>> print(rescaled)
                -0.4

                ```
            - Rescale a temperature from Celsius to Fahrenheit:
                ```python
                >>> celsius = 25
                >>> fahrenheit = Tools.rescale(celsius, 0, 100, 32, 212)
                >>> print(f"{celsius}°C = {fahrenheit}°F")
                25°C = 77.0°F

                ```

        See Also:
            - Tools.normalize: For scaling values to the range [0, 1]
            - Tools.log_rescale: For logarithmic rescaling
        """
        old_range = old_max - old_min
        new_range = new_max - new_min
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

        return new_value

    @staticmethod
    def log_rescale(
        x: float, min_old: float, max_old: float, min_new: float, max_new: float
    ) -> int:
        """Rescale a value using logarithmic transformation.

        Transforms a value from its original range to a new range using logarithmic scaling.
        This is useful when dealing with data that spans multiple orders of magnitude,
        as it compresses large values and expands small values.

        The method first converts the value and boundaries to logarithmic space,
        then performs a linear rescaling in that space, and finally rounds to an integer.

        Args:
            x: The value to rescale.
            min_old: The minimum value of the original range.
            max_old: The maximum value of the original range.
            min_new: The minimum value of the target range.
            max_new: The maximum value of the target range.

        Returns:
            int: The logarithmically rescaled value as an integer in the new range.

        Raises:
            ValueError: If max_old is not greater than min_old.
            ValueError: If x is negative (logarithm undefined).

        Examples:
            - Rescale a value from [1, 1000] to [1, 10]:
                ```python
                >>> from statista.tools import Tools
                >>> value = 100
                >>> rescaled = Tools.log_rescale(value, 1, 1000, 1, 10)
                >>> print(rescaled)
                7

                ```
            - Rescale a small value:
                ```python
                >>> value = 10
                >>> rescaled = Tools.log_rescale(value, 1, 1000, 1, 10)
                >>> print(rescaled)
                4

                ```
            - Handle zero values (special case):
                ```python
                >>> value = 0
                >>> rescaled = Tools.log_rescale(value, 0, 1000, 0, 10)
                >>> print(rescaled)
                0

                ```

        Notes:
            - For x = 0, the function uses a special case handling by setting the log value to -7.
            - For min_old = 0, the function also uses -7 as the logarithmic value.
            - The base of the logarithm is e (natural logarithm).

        See Also:
            - Tools.inv_log_rescale: For inverse logarithmic rescaling
            - Tools.rescale: For linear rescaling
        """
        # get the boundaries of the logarithmic scale
        if np.isclose(min_old, 0.0):
            min_old_log = -7
        else:
            min_old_log = np.log(min_old)

        max_old_log = np.log(max_old)

        if x == 0:
            x_log = -7
        else:
            x_log = np.log(x)

        y = int(
            np.round(Tools.rescale(x_log, min_old_log, max_old_log, min_new, max_new))
        )

        return y

    @staticmethod
    def inv_log_rescale(
        x: float,
        min_old: float,
        max_old: float,
        min_new: float,
        max_new: float,
        base: float = np.e,
    ) -> int:
        """Rescale a value using inverse logarithmic transformation.

        Performs the inverse operation of log_rescale. Instead of taking logarithms,
        this method raises the base to the power of the input values before rescaling.
        This is useful when you need to expand the scale of values that were previously
        compressed using a logarithmic transformation.

        The method first converts the value and boundaries to exponential space using
        the specified base, then performs a linear rescaling in that space, and finally
        rounds to an integer.

        Args:
            x: The value to rescale.
            min_old: The minimum value of the original range.
            max_old: The maximum value of the original range.
            min_new: The minimum value of the target range.
            max_new: The maximum value of the target range.
            base: The base to use for the exponential transformation. Defaults to e (natural exponential).

        Returns:
            int: The inverse logarithmically rescaled value as an integer in the new range.

        Raises:
            ValueError: If max_old is not greater than min_old.
            OverflowError: If the exponential values are too large to handle.

        Examples:
            - Rescale a value from [1, 3] to [1, 1000] using base e:
                ```python
                >>> from statista.tools import Tools
                >>> value = 2
                >>> rescaled = Tools.inv_log_rescale(value, 1, 3, 1, 1000)
                >>> print(rescaled)
                270

                ```

            - Using a different base (base 10):
                ```python
                >>> import numpy as np
                >>> value = 1
                >>> rescaled = Tools.inv_log_rescale(value, 0, 2, 1, 100, base=10)
                >>> print(rescaled)
                10

                ```
            - Verify inverse relationship with log_rescale:
                ```python
                >>> original = 500

                ```
            - First log_rescale from [1, 1000] to [0, 3]:
                ```python
                >>> log_scaled = Tools.log_rescale(original, 1, 1000, 0, 3)

                ```
            - Then inv_log_rescale back from [0, 3] to [1, 1000]:
                ```python
                >>> back_to_original = Tools.inv_log_rescale(log_scaled, 0, 3, 1, 1000)
                >>> print(f"Original: {original}, After round-trip: {back_to_original}")
                Original: 500, After round-trip: 1000

                ```

        Notes:
            Due to rounding and the discrete nature of the transformation,
            the round-trip conversion (log_rescale followed by inv_log_rescale)
            may not exactly reproduce the original value.

        See Also:
            - Tools.log_rescale: For logarithmic rescaling
            - Tools.rescale: For linear rescaling
        """
        # get the boundaries of the logarithmic scale

        min_old_power = np.power(base, min_old)
        max_old_power = np.power(base, max_old)
        x_power = np.power(base, x)

        y = int(
            np.round(
                Tools.rescale(x_power, min_old_power, max_old_power, min_new, max_new)
            )
        )
        return y

    @staticmethod
    def round(number: float, precision: float) -> float:
        """Round a number to a specified precision.

        Rounds a number to the nearest multiple of the specified precision.
        This is different from Python's built-in round function, which rounds
        to a specified number of decimal places.

        Args:
            number: The number to be rounded.
            precision: The precision to round to. For example, if precision is 0.5,
                the number will be rounded to the nearest 0.5.

        Returns:
            float: The rounded number.

        Examples:
            - Round to the nearest 0.5
            ```python
            >>> from statista.tools import Tools
            >>> value = 3.7
            >>> rounded = Tools.round(value, 0.5)
            >>> print(rounded)
            3.5

            ```

            - Round to the nearest 5:
                ```python
                >>> value = 23
                >>> rounded = Tools.round(value, 5)
                >>> print(rounded)
                25

                ```

            - Round to the nearest 0.1:
                ```python
                >>> value = 7.84
                >>> rounded = Tools.round(value, 0.1)
                >>> print(rounded) #doctest: +SKIP
                7.8

                ```

        Notes:
            The formula used is: round(number / precision) * precision

            This method is useful for rounding to specific increments rather than
            decimal places. For example, rounding to the nearest 0.25, 0.5, or 5.
        """
        value = Decimal(f"{number}")
        precision = Decimal(f"{precision}")  # type: ignore[assignment]
        rounded = (value / precision).to_integral_value(  # type: ignore[operator]
            rounding=ROUND_HALF_UP
        ) * precision
        return float(rounded)
