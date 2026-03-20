"""Exponential distribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.optimize as so
from matplotlib.figure import Figure
from scipy.stats import expon

from statista.distributions.base import (
    CDF_INVALID_VALUE_ERROR,
    OBJ_FUNCTION_THRESHOLD_ERROR,
    SCALE_PARAMETER_ERROR,
    AbstractDistribution,
)
from statista.distributions.parameters import Parameters
from statista.parameters import Lmoments


class Exponential(AbstractDistribution):
    """Exponential distribution.

    - The exponential distribution assumes that small values occur more frequently than large values.

    - The probability density function (PDF) of the Exponential distribution is:

        $$
        f(x; \\delta, \\beta) =
        \\begin{cases}
            \\frac{1}{\\beta} e^{-\\frac{x - \\delta}{\\beta}} & \\quad x \\geq \\delta \\\\
            0 & \\quad x < \\delta
        \\end{cases}
        $$

    - The probability density function above uses the location parameter \\(\\delta\\) and the scale parameter
        \\(\\beta\\) to define the distribution in a standardized form.
    - A common parameterization for the exponential distribution is in terms of the rate parameter \\(\\lambda\\),
        such that \\(\\lambda = 1 / \\beta\\).
    - The Location Parameter (\\(\\delta\\)): This shifts the starting point of the distribution. The distribution is
        defined for \\(x \\geq \\delta\\).
    - Scale Parameter (\\(\\beta\\)): This determines the spread of the distribution. The rate parameter
        \\(\\lambda\\) is the inverse of the scale parameter, so \\(\\lambda = \\frac{1}{\\beta}\\).

    - The cumulative distribution functions.

        $$
        F(x; \\delta, \\beta) =
        \\begin{cases}
            1 - e^{-\\frac{x - \\delta}{\\beta}} & \\quad x \\geq \\delta \\\\
            0 & \\quad x < \\delta
        \\end{cases}
        $$

    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: Parameters | None = None,
    ):
        """Exponential Distribution.

        Args:
            data (list):
                data time series.
            parameters (Parameters):
                Parameters(loc=val, scale=val)

                - loc (numeric):
                    location parameter of the exponential distribution.
                - scale (numeric):
                    scale parameter of the exponential distribution.
        """
        super().__init__(data, parameters)

    @staticmethod
    def _pdf_eq(
        data: list | np.ndarray, parameters: Parameters
    ) -> np.ndarray:
        loc = parameters.loc
        scale = parameters.scale

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        pdf = expon.pdf(data, loc=loc, scale=scale)
        return pdf

    def pdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: Parameters | None = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x.

        Args:
            parameters (Parameters, optional):
                if not provided, the parameters provided in the class initialization will be used.
                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.
                ```python
                Parameters(loc=val, scale=val)
                ```
                default is None.
            data (np.ndarray):
                array if you want to calculate the pdf for different data than the time series given to the constructor
                method. default is None.
            plot_figure (bool):
                Default is False.
            kwargs (dict[str, Any]):
                fig_size(tuple):
                    Default is (6, 5).
                xlabel (str):
                    Default is "Actual data".
                ylabel (str):
                    Default is "pdf".
                fontsize (int):
                    Default is 15

        Returns:
            pdf (array):
                probability density function pdf.
            fig (matplotlib.figure.Figure):
                Figure object. returned only if `plot_figure` is True.
            ax (matplotlib.axes.Axes):
                Axes object. returned only if `plot_figure` is True.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.distributions import Exponential
            >>> data = np.loadtxt("examples/data/expo.txt")
            >>> parameters = Parameters(loc=0, scale=2)
            >>> expo_dist = Exponential(data, parameters)
            >>> _ = expo_dist.pdf(plot_figure=True)

            ```
            ![exponential-pdf](./../../_images/distributions/exponential-pdf-2.png)
        """
        result = super().pdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )  # type: ignore[misc]

        return result

    def random(
        self,
        size: int,
        parameters: Parameters | None = None,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """Generate Random Variable.

        Args:
            size (int):
                size of the random generated sample.
            parameters (Parameters):
                - loc (numeric):
                    location parameter of the gumbel distribution.
                - scale (numeric):
                    scale parameter of the gumbel distribution.
                ```python
                Parameters(loc=val, scale=val)
                ```

        Returns:
            data (np.ndarray):
                random generated data.

        Examples:
            - To generate a random sample that follow the gumbel distribution with the parameters loc=0 and scale=1.
                ```python
                >>> from statista.distributions import Exponential
                >>> parameters = Parameters(loc=0, scale=2)
                >>> expon_dist = Exponential(parameters=parameters)
                >>> random_data = expon_dist.random(1000)

                ```
            - then we can use the `pdf` method to plot the pdf of the random data.
                ```python
                >>> _ = expon_dist.pdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![exponential-pdf](./../../_images/distributions/exponential-pdf.png)

                ```python
                >>> _ = expon_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![exponential-cdf](./../../_images/distributions/exponential-cdf.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)

        loc = parameters.loc
        scale = parameters.scale
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        random_data = expon.rvs(loc=loc, scale=scale, size=size)
        return random_data

    @staticmethod
    def _cdf_eq(
        data: list | np.ndarray, parameters: Parameters
    ) -> np.ndarray:
        """
        old cdf equation.
        ```python
        >>> ts = np.array([1, 2, 3, 4, 5, 6]) # any value
        >>> loc = 0 # any value
        >>> scale = 2 # any value
        >>> Y = (ts - loc) / scale
        >>> cdf = 1 - np.exp(-Y)
        >>> for i in range(0, len(cdf)):
        ...     if cdf[i] < 0:
        ...         cdf[i] = 0

        ```
        """
        loc = parameters.loc
        scale = parameters.scale
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        cdf = expon.cdf(data, loc=loc, scale=scale)
        return cdf

    def cdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: Parameters | None = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[np.ndarray, Figure, Any] | np.ndarray
    ):  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        Args:
            parameters (Parameters, optional):
                if not provided, the parameters provided in the class initialization will be used. default is None.
                - loc (numeric):
                    location parameter of the gumbel distribution.
                - scale (numeric):
                    scale parameter of the gumbel distribution.
                ```python
                Parameters(loc=val, scale=val)
                ```
            data (np.ndarray):
                array if you want to calculate the cdf for different data than the time series given to the constructor
                method. default is None.
            plot_figure (bool):
                Default is False.
            kwargs (dict[str, Any]):
                fig_size: [tuple]
                    Default is (6, 5).
                xlabel (str):
                    Default is "Actual data".
                ylabel (str):
                    Default is "cdf".
                fontsize (int):
                    Default is 15.

        Returns:
            cdf (array):
                probability density function cdf.
            fig (matplotlib.figure.Figure):
                Figure object is returned only if `plot_figure` is True.
            ax (matplotlib.axes.Axes):
                Axes object is returned only if `plot_figure` is True.

        Examples:
            ```python
            >>> import numpy as np
            >>> from statista.distributions import Exponential
            >>> data = np.loadtxt("examples/data/expo.txt")
            >>> parameters = Parameters(loc=0, scale=2)
            >>> expo_dist = Exponential(data, parameters)
            >>> _ = expo_dist.cdf(plot_figure=True)

            ```
            ![gamma-pdf](./../../_images/distributions/expo-random-cdf.png)
        """
        result = super().cdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )  # type: ignore[misc]
        return result

    def fit_model(
        self,
        method: str = "mle",
        obj_func=None,
        threshold: int | float | None = None,
        test: bool = True,
    ) -> Parameters:
        """fit_model.

        fit_model estimates the distribution parameter based on MLM
        (Maximum likelihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for the values below (x < C), now the likeliest parameters
        are those at the max value of multiplication between two functions max(L1*L2).

        In this case, the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Args:
            obj_func (function):
                function to be used to get the distribution parameters.
            threshold (numeric):
                Value you want to consider only the greater values.
            method (str):
                'mle', 'mm', 'lmoments', optimization
            test (bool):
                Default is True

        Returns:
            param (list):
                shape, loc, scale parameter of the gumbel distribution in that order.

        Examples:
            - Instantiate the `Exponential` class only with the data.
                ```python
                >>> data = np.loadtxt("examples/data/expo.txt")
                >>> expo_dist = Exponential(data)

                ```
            - Then use the `fit_model` method to estimate the distribution parameters. the method takes the method as
                parameter, the default is 'mle'. the `test` parameter is used to perform the Kolmogorov-Smirnov and chisquare
                test.

                ```python
                >>> parameters = expo_dist.fit_model(method="mle", test=True) # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.019
                Accept Hypothesis
                P value = 0.9937026761524456
                Out[14]: Parameters(loc=0.0009, scale=2.0498075)
                >>> print(parameters) # doctest: +SKIP
                Parameters(loc=0, scale=2)

                ```
            - You can also use the `lmoments` method to estimate the distribution parameters.
                ```python
                >>> parameters = expo_dist.fit_model(method="lmoments", test=True) # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.021
                Accept Hypothesis
                P value = 0.9802627322900355
                >>> print(parameters) # doctest: +SKIP
                Parameters(loc=-0.00805012182182141, scale=2.0587576218218215)

                ```
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        method = super().fit_model(method=method)  # type: ignore[assignment]

        if method == "mle" or method == "mm":
            param_list: Any = list(expon.fit(self.data, method=method))
        elif method == "lmoments":
            lm = Lmoments(self.data)
            lmu = lm.calculate()
            param_list = Lmoments.exponential(lmu)
        elif method == "optimization":
            if obj_func is None or threshold is None:
                raise TypeError(OBJ_FUNCTION_THRESHOLD_ERROR)

            param_list = expon.fit(self.data, method="mle")
            # then we use the result as starting value for your truncated Gumbel fit
            param_list = so.fmin(
                obj_func,
                [threshold, param_list[0], param_list[1]],
                args=(self.data,),
                maxiter=500,
                maxfun=500,
            )
            param_list = [param_list[1], param_list[2]]
        else:
            raise ValueError(f"The given: {method} does not exist")

        param = Parameters(loc=param_list[0], scale=param_list[1])
        self.parameters = param

        if test:
            self.ks()
            self.chisquare()

        return param

    def inverse_cdf(
        self,
        cdf: np.ndarray | list[float] | None = None,
        parameters: Parameters | None = None,
    ) -> np.ndarray:
        """Theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on a given  non-exceedance probability

        Args:
            parameters (Parameters):
                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.
                ```python
                Parameters(loc=val, scale=val)
                ```
            cdf (list):
                cumulative distribution function/ Non-Exceedance probability.

        Returns:
            theoretical value (numeric):
                Value based on the theoretical distribution

        Examples:
            - Instantiate the Exponential class only with the data.
                ```python
                >>> data = np.loadtxt("examples/data/expo.txt")
                >>> parameters = Parameters(loc=0, scale=2)
                >>> expo_dist = Exponential(data, parameters)

                ```
            - We will generate a random numbers between 0 and 1 and pass it to the inverse_cdf method as a probabilities
                to get the data that coresponds to these probabilities based on the distribution.
                ```python
                >>> cdf = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                >>> data_values = expo_dist.inverse_cdf(cdf)
                >>> print(data_values)
                [0.21072103 0.4462871  1.02165125 1.83258146 3.21887582 4.60517019]

                ```
        """
        if parameters is None:
            parameters = self.parameters
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)

        loc = parameters.loc
        scale = parameters.scale

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        cdf = np.array(cdf)
        if np.any(cdf < 0) or np.any(cdf > 1):
            raise ValueError(CDF_INVALID_VALUE_ERROR)

        # the main equation from scipy
        q_th = expon.ppf(cdf, loc=loc, scale=scale)
        return q_th

    def ks(self):
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static, the more likely that the two samples are drawn from the same distribution
        IF Pvalue < significance level ------ reject

        Returns:
            Dstatic (numeric):
                The smaller the D static the more likely that the two samples are drawn from the same distribution
            Pvalue (numeric):
                IF Pvalue < significance level ------ reject the null hypothesis
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """chisquare test"""
        return super().chisquare()
