"""Gumbel distribution."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.optimize as so
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gumbel_r, norm

from statista.distributions.base import (
    CDF_INVALID_VALUE_ERROR,
    PDF_XAXIS_LABEL,
    PROB_NON_EXCEEDENCE_ERROR,
    SCALE_PARAMETER_ERROR,
    AbstractDistribution,
    PlottingPosition,
)
from statista.distributions.parameters import Parameters
from statista.parameters import Lmoments
from statista.plot import Plot


class Gumbel(AbstractDistribution):
    """Gumbel distribution (Maximum - Right Skewed) for extreme value analysis.

    The Gumbel distribution is used to model the distribution of the maximum (or the minimum)
    of a number of samples of various distributions. It is commonly used in hydrology,
    meteorology, and other fields to model extreme events like floods, rainfall, and wind speeds.

    The Gumbel distribution is a special case of the Generalized Extreme Value (GEV)
    distribution with shape parameter ξ = 0.

    Attributes:
        _data (np.ndarray): The data array used for distribution calculations.
        _parameters (dict[str, float]): Distribution parameters (loc and scale).

    - The probability density function (PDF) of the Gumbel distribution is:

        $$
        f(x; \\zeta, \\delta) = \\frac{1}{\\delta}
        \\exp\\left(-\\frac{x - \\zeta}{\\delta}\\right)
        \\exp\\left(-\\exp\\left(-\\frac{x - \\zeta}{\\delta}\\right)\\right)
        $$

        Where \\(\\zeta\\) (zeta) is the location parameter and \\(\\delta\\) (delta)
        is the scale parameter.

    - The cumulative distribution function (CDF) is:

        $$
        F(x; \\zeta, \\delta) = \\exp\\left(-\\exp\\left(-\\frac{x - \\zeta}{\\delta}\\right)\\right)
        $$

    - The location parameter \\(\\zeta\\) shifts the distribution along the x-axis, determining
      the mode (peak) of the distribution. It can range from negative to positive infinity.
    - The scale parameter \\(\\delta\\) controls the spread of the distribution. A larger scale
      parameter results in a wider distribution. It must always be positive.
    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: dict[str, float] = None,
    ):
        """Initialize a Gumbel distribution with data or parameters.

        Args:
            data:
                Data time series as a list or numpy array.
            parameters:
                - loc (numeric):
                    Location parameter of the Gumbel distribution
                - scale (numeric):
                    Scale parameter of the Gumbel distribution (must be positive)
                ```python
                {"loc": 0.0, "scale": 1.0}
                ```

        Raises:
            ValueError: If neither data nor parameters are provided.
            TypeError: If data is not a list or numpy array, or if parameters is not a dictionary.

        Examples:
            - Import necessary libraries
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel

                ```
            - Load sample data:
                ```python
                >>> data = np.loadtxt("examples/data/gumbel.txt")

                ```
            - Initialize with data only
                ```python
                >>> gumbel_dist = Gumbel(data)

                ```
            - Initialize with both data and parameters
                ```python
                >>> parameters = {"loc": 0, "scale": 1}
                >>> gumbel_dist = Gumbel(data, parameters)

                ```
            - Initialize with parameters only
                ```python
                >>> gumbel_dist = Gumbel(parameters={"loc": 0, "scale": 1})

                ```
        """
        super().__init__(data, parameters)

    @staticmethod
    def _pdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        """Calculate the probability density function (PDF) values for Gumbel distribution.

        This method implements the Gumbel PDF equation:
        f(x; ζ, δ) = (1/δ) * exp(-(x-ζ)/δ) * exp(-exp(-(x-ζ)/δ))

        Args:
            data:
                Data points for which to calculate PDF values.
            parameters:
                Dictionary of distribution parameters.
                Must contain:
                    - "loc": Location parameter (ζ)
                    - "scale": Scale parameter (δ), must be positive

        Returns:
            Numpy array containing the PDF values for each data point.

        Raises:
            ValueError: If the scale parameter is negative or zero.

        old code:
        ```python
        >>> ts = np.array([1, 2, 3, 4, 5]) # any value
        >>> loc = 0.0 # any value
        >>> scale = 1.0 # any value
        >>> z = (ts - loc) / scale
        >>> pdf = (1.0 / scale) * (np.exp(-(z + (np.exp(-z)))))

        ```
        """
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        pdf = gumbel_r.pdf(data, loc=loc, scale=scale)
        return pdf

    def pdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: dict[str, Any] = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, Figure, Any]:
        """Calculate the probability density function (PDF) values for Gumbel distribution.

        This method calculates the PDF values for the given data using the specified
        Gumbel distribution parameters. It can also generate a plot of the PDF.

        Args:
            plot_figure:
                Whether to generate a plot of the PDF. Default is False.
            parameters:
                    - loc (Numberic):
                        Location parameter of the Gumbel distribution
                    - scale (Numberic):
                        Scale parameter of the Gumbel distribution (must be positive)
                    ```python
                    {"loc": 0.0, "scale": 1.0}
                    ```
                    If None, uses the parameters provided during initialization.
            data:
                Data points for which to calculate PDF values. If None, uses the data provided during initialization.
            *args:
                Variable length argument list to pass to the parent class method.
            **kwargs:
                Arbitrary keyword arguments to pass to the plotting function.
                the possible keyword arguments are:
                    - fig_size:
                        Size of the figure as a tuple (width, height). Default is (6, 5).
                    - xlabel:
                        Label for the x-axis. Default is "Actual data".
                    - ylabel:
                        Label for the y-axis. Default is "pdf".
                    - fontsize:
                        Font size for plot labels. Default is 15.

        Returns:
            If plot_figure is False:
                Numpy array containing the PDF values for each data point.
            If plot_figure is True:
                Tuple containing:
                - Numpy array of PDF values
                - Figure object
                - Axes object

        Examples:
            - Import libraries:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel

                ```
            - Load sample data:
                ```python
                >>> data = np.loadtxt("examples/data/gumbel.txt")

                ```
            - Calculate PDF values with default parameters:
                ```python
                >>> gumbel_dist = Gumbel(data)
                >>> gumbel_dist.fit_model() # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.019
                Accept Hypothesis
                P value = 0.9937026761524456
                {'loc': np.float64(0.010101355750222706), 'scale': 1.0313042643102108}
                >>> pdf_values = gumbel_dist.pdf() # doctest: +SKIP

                ```
            - Generate a PDF plot:
                ```python
                >>> pdf_values, fig, ax = gumbel_dist.pdf(
                ...     plot_figure=True,
                ...     xlabel="Values",
                ...     ylabel="Density",
                ...     fig_size=(8, 6)
                ... ) # doctest: +SKIP

                ```
                ![gamma-pdf](./../../_images/distributions/gamma-pdf-1.png)

            - Calculate PDF with custom parameters:
                ```python
                >>> parameters = {'loc': 0, 'scale': 1}
                >>> pdf_custom = gumbel_dist.pdf(parameters=parameters)
                >>> print(pdf_custom) #doctest: +SKIP
                array([5.44630532e-02, 1.55313724e-01, 3.29857975e-01, 7.01082330e-02,
                       3.54572987e-01, 1.46804327e-01, 3.36843753e-01, 1.01491310e-01,
                       2.38861650e-01, 3.42034071e-01, 2.59606975e-01, 3.33403275e-01,
                       3.52075676e-01, 1.24617619e-01, 6.37994991e-02, 3.67871923e-01,
                       ...
                       2.12529308e-01, 3.13383427e-01, 3.62783762e-01, 4.09957082e-02,
                       2.61395400e-01, 2.58511435e-01, 1.94640967e-01, 3.37392659e-01])
                ```
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
        parameters: dict[str, float | Any] = None,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """Generate random samples from the Gumbel distribution.

        This method generates random samples following the Gumbel distribution
        with the specified parameters.

        Args:
            size:
                Number of random samples to generate.
            parameters:
                    - loc (Numberic):
                        Location parameter of the Gumbel distribution
                    - scale (Numberic):
                        Scale parameter of the Gumbel distribution (must be positive)
                    ```python
                    {"loc": 0.0, "scale": 1.0}
                    ```
                    If None, uses the parameters provided during initialization.

        Returns:
            Numpy array containing the generated random samples.

        Raises:
            ValueError: If the parameters are not provided and not available from initialization.

        Examples:
            - import the required modules and generate random samples:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel
                >>> parameters = {'loc': 0, 'scale': 1}
                >>> gumbel_dist = Gumbel(parameters=parameters)
                >>> random_data = gumbel_dist.random(1000)

                ```
            - Analyze the generated data:
                - Plot the PDF of the random data:
                ```python
                >>> _ = gumbel_dist.pdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![gamma-pdf](./../../_images/distributions/gamma-random-1.png)

                - Plot the CDF of the random data:
                    ```python
                    >>> _ = gumbel_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

                    ```
                    ![gamma-cdf](./../../_images/distributions/gamma-cdf-1.png)

            - Verify the parameters by fitting the model to the random data
                ```python
                >>> gumbel_dist = Gumbel(data=random_data)
                >>> fitted_params = gumbel_dist.fit_model() #doctest: +SKIP
                -----KS Test--------
                Statistic = 0.018
                Accept Hypothesis
                P value = 0.9969602438295625
                >>> print(f"Fitted parameters: {fitted_params}") #doctest: +SKIP
                Fitted parameters: {'loc': np.float64(-0.010212105435018243), 'scale': 1.010287499893525}

                ```
            - Should be close to the original parameters {'loc': 0, 'scale': 1}
            ```
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        random_data = gumbel_r.rvs(loc=loc, scale=scale, size=size)
        return random_data

    @staticmethod
    def _cdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        """Calculate the cumulative distribution function (CDF) values for Gumbel distribution.

        This method implements the Gumbel CDF equation:
        F(x; ζ, δ) = exp(-exp(-(x-ζ)/δ))

        Args:
            data: Data points for which to calculate CDF values.
            parameters: Dictionary of distribution parameters.
                Must contain:
                - "loc": Location parameter (ζ)
                - "scale": Scale parameter (δ), must be positive

        Returns:
            Numpy array containing the CDF values for each data point.

        Raises:
            ValueError: If the scale parameter is negative or zero.

        old code:
        ```python
        >>> ts = np.array([1, 2, 3, 4, 5]) # any value
        >>> loc = 0.0 # any value
        >>> scale = 1.0 # any value
        >>> z = (ts - loc) / scale
        >>> cdf = np.exp(-np.exp(-z))

        ```
        """
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        cdf = gumbel_r.cdf(data, loc=loc, scale=scale)
        return cdf

    def cdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: dict[str, Any] = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> (
        np.ndarray | tuple[np.ndarray, Figure, Axes]
    ):  # pylint: disable=arguments-differ
        """Calculate the cumulative distribution function (CDF) values for Gumbel distribution.

        This method calculates the CDF values for the given data using the specified
        Gumbel distribution parameters. It can also generate a plot of the CDF.

        Args:
            plot_figure:
                Whether to generate a plot of the CDF. Default is False.
            parameters:
                - loc:
                    Location parameter of the Gumbel distribution
                - scale:
                    Scale parameter of the Gumbel distribution (must be positive)
                ```python
                {"loc": 0.0, "scale": 1.0}
                ```
                If None, uses the parameters provided during initialization.
            data:
                Data points for which to calculate CDF values. If None, uses the data provided during initialization.
            *args:
                Variable length argument list to pass to the parent class method.
            **kwargs:
                - fig_size:
                    Size of the figure as a tuple (width, height). Default is (6, 5).
                - xlabel:
                    Label for the x-axis. Default is "Actual data".
                - ylabel:
                    Label for the y-axis. Default is "cdf".
                - fontsize:
                    Font size for plot labels. Default is 15.

        Returns:
            If plot_figure is False:
                Numpy array containing the CDF values for each data point.
            If plot_figure is True:
                Tuple containing:
                - Numpy array of CDF values
                - Figure object
                - Axes object

        Examples:
            -  Load sample data:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel
                >>> data = np.loadtxt("examples/data/gumbel.txt")

                ```
            -  Calculate CDF values with default parameters:
                ```python
                >>> gumbel_dist = Gumbel(data)
                >>> gumbel_dist.fit_model() # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.019
                Accept Hypothesis
                P value = 0.9937026761524456
                {'loc': np.float64(0.010101355750222706), 'scale': 1.0313042643102108}
                >>> cdf_values = gumbel_dist.cdf() # doctest: +SKIP

                ```
            -  Generate a CDF plot:
                ```python
                >>> cdf_values, fig, ax = gumbel_dist.cdf(
                ...     plot_figure=True,
                ...     xlabel="Values",
                ...     ylabel="Probability",
                ...     fig_size=(8, 6)
                ... ) # doctest: +SKIP

                ```
                ![gamma-cdf](./../../_images/distributions/gamma-cdf-2.png)

            -  Calculate CDF with custom parameters:
                ```python
                >>> parameters = {'loc': 0, 'scale': 1}
                >>> cdf_custom = gumbel_dist.cdf(parameters=parameters)

                ```
            -  Calculate exceedance probability (1-CDF):
                ```python
                >>> exceedance_prob = 1 - cdf_values # doctest: +SKIP

                ```
            ```
        """
        result = super().cdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )  # type: ignore[misc]
        return result

    def return_period(
        self,
        *,
        data: bool | list[float] | None = None,
        parameters: dict[str, float | Any] = None,
    ) -> np.ndarray:
        """Calculate return periods for given data values.

        The return period is the average time between events of a given magnitude.
        It is calculated as 1/(1-F(x)), where F(x) is the cumulative distribution function.

        Args:
            data:
                Values for which to calculate return periods. Can be a single value, list, or array.
                If None, uses the data provided during initialization.
            parameters:
                - loc (Numeric):
                    Location parameter of the Gumbel distribution
                - scale (Numeric):
                    Scale parameter of the Gumbel distribution (must be positive)
                ```
                {"loc": 0.0, "scale": 1.0}
                ```
                If None, uses the parameters provided during initialization.

        Returns:
            np.ndarray:
                Return periods corresponding to the input data values.
                - If input is a single value, returns a single value.
                - If input is a list or array, returns an array of return periods.

        Examples:
            - Import necessary libraries:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel

                ```
            -  Calculate return periods for specific values
                ```python
                >>> data = np.loadtxt("examples/data/gumbel.txt")
                >>> gumbel_dist = Gumbel(data=data,parameters={"loc": 0, "scale": 1})
                >>> return_periods = gumbel_dist.return_period()

                ```
            -  Calculate the 100-year return level:
                - First, find the CDF value corresponding to a 100-year return period
                - F(x) = 1 - 1/T, where T is the return period
                ```python
                >>> cdf_value = 1 - 1/100

                ```
            - Then, find the quantile corresponding to this CDF value:
                ```python
                >>> return_level_100yr = gumbel_dist.inverse_cdf([cdf_value], parameters={"loc": 0, "scale": 1})[0]
                >>> print(f"100-year return level: {return_level_100yr:.4f}")
                100-year return level: 4.6001

                ```
        """
        if data is None:
            ts: Any = self.data
        else:
            ts = data

        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        cdf: np.ndarray = self.cdf(parameters=parameters, data=ts)  # type: ignore[assignment]

        rp = 1 / (1 - cdf)

        return rp

    @staticmethod
    def truncated_distribution(opt_parameters: list[float], data: list[float]) -> float:
        """Calculate a negative log-likelihood for a truncated Gumbel distribution.

        This function calculates the negative log-likelihood of a Gumbel distribution
        that is truncated (i.e., the data only includes values above a certain threshold).
        It is used as an objective function for parameter optimization when fitting
        a truncated Gumbel distribution to data.

        This approach is useful when the dataset is incomplete or when data is only
        available above a certain threshold, a common scenario in environmental sciences,
        finance, and other fields dealing with extremes.

        Args:
            opt_parameters:
                List of parameters to optimize:
                    - opt_parameters[0]: Threshold value
                    - opt_parameters[1]: Location parameter (loc)
                    - opt_parameters[2]: Scale parameter (scale)
            data:
                Data points to fit the truncated distribution to.

        Returns:
            Negative log-likelihood value. Lower values indicate better fit.

        Notes:
            The negative log-likelihood is calculated as the sum of two components:
                - L1: Log-likelihood for values below the threshold
                - L2: Log-likelihood for values above the threshold

        Reference:
            https://stackoverflow.com/questions/23217484/how-to-find-parameters-of-gumbels-distribution-using-scipy-optimize

        Examples:
            - import the required modules and generate sample data:
                ```python
                >>> import numpy as np
                >>> from scipy.optimize import minimize
                >>> from statista.distributions import Gumbel
                >>> data = np.random.gumbel(loc=10, scale=2, size=1000)

                ```
            - Initial parameter guess [threshold, loc, scale]:
                ```python
                >>> initial_params = [5.0, 8.0, 1.5]

                ```
            - Optimize parameters:
                ```python
                >>> result = minimize(
                ...     Gumbel.truncated_distribution,
                ...     initial_params,
                ...     args=(data,),
                ...     method='Nelder-Mead'
                ... )

                ```
            - Extract optimized parameters:
                ```python
                >>> threshold, loc, scale = result.x
                >>> print(f"Optimized parameters: threshold={threshold}, loc={loc}, scale={scale}")
                Optimized parameters: threshold=4.0, loc=9.599999999999994, scale=1.5

                ```
        """
        threshold = opt_parameters[0]
        loc = opt_parameters[1]
        scale = opt_parameters[2]

        non_truncated_data = data[data < threshold]  # type: ignore[operator]
        nx2 = len(data[data >= threshold])  # type: ignore[arg-type, operator]
        # pdf with a scaled pdf
        # L1 is pdf based
        parameters = Parameters(loc=loc, scale=scale)
        pdf = Gumbel._pdf_eq(non_truncated_data, parameters)  # type: ignore[arg-type]
        #  the CDF at the threshold is used because the data is assumed to be truncated, meaning that observations below
        #  this threshold are not included in the dataset. When dealing with truncated data, it's essential to adjust
        #  the likelihood calculation to account for the fact that only values above the threshold are observed. The
        #  CDF at the threshold effectively normalizes the distribution, ensuring that the probabilities sum to 1 over
        #  the range of the observed data.
        cdf_at_threshold = 1 - Gumbel._cdf_eq(threshold, parameters)  # type: ignore[arg-type]
        # calculates the negative log-likelihood of a Gumbel distribution
        # Adjust the likelihood for the truncation
        # likelihood = pdf / (1 - adjusted_cdf)

        l1 = (-np.log((pdf / scale))).sum()
        # L2 is cdf based
        l2 = (-np.log(cdf_at_threshold)) * nx2

        return l1 + l2

    def fit_model(
        self,
        method: str = "mle",
        obj_func: Callable = None,
        threshold: None | float | int = None,
        test: bool = True,
    ) -> Parameters:
        """Estimate the parameters of the Gumbel distribution from data.

        This method fits the Gumbel distribution to the data using various estimation
        methods, including Maximum Likelihood Estimation (MLE), Method of Moments (MM),
        L-moments, or custom optimization.

        When using the 'optimization' method with a threshold, the method employs two
        likelihood functions:
            - L1: For values below the threshold
            - L2: For values above the threshold

        The parameters are estimated by maximizing the product L1*L2.

        Args:
            method:
                Estimation method to use. Default is 'mle'.
                Options:
                    - 'mle' (Maximum Likelihood Estimation),
                    - 'mm' (Method of Moments),
                    - 'lmoments' (L-moments),
                    - 'optimization' (Custom optimization)
            obj_func (callable | None):
                Custom objective function to use for parameter estimation. Only used when method is 'optimization'.
                Default is None.
            threshold (float | int | None):
                Value above which to consider data points. If provided, only data points above this threshold are
                used for estimation when using the 'optimization' method. Default is None (use all data points).
            test:
                Whether to perform goodness-of-fit tests after estimation. Default is True.

        Returns:
            Dict:
                - loc (Numeric):
                    Location parameter of the Gumbel distribution
                - scale (Numeric):
                    Scale parameter of the Gumbel distribution
                ```python
                {"loc": 0.0, "scale": 1.0}
                ```

        Raises:
            ValueError: If an invalid method is specified or if required parameters are missing.

        Examples:
            - Import necessary libraries:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel

                ```
            - Load sample data:
                ```python
                >>> data = np.loadtxt("examples/data/gumbel.txt")
                >>> gumbel_dist = Gumbel(data)

                ```
            - Fit using Maximum Likelihood Estimation (default):
                ```python
                >>> parameters = gumbel_dist.fit_model(method="mle", test=True)
                -----KS Test--------
                Statistic = 0.019
                Accept Hypothesis
                P value = 0.9937026761524456
                >>> print(parameters)
                {'loc': np.float64(0.010101355750222706), 'scale': 1.0313042643102108}

                ```
            - Fit using L-moments:
                ```python
                >>> parameters = gumbel_dist.fit_model(method="lmoments", test=True)
                -----KS Test--------
                Statistic = 0.019
                Accept Hypothesis
                P value = 0.9937026761524456
                >>> print(parameters)
                {'loc': np.float64(0.006700226367219564), 'scale': np.float64(1.0531061622114444)}

                ```
            - Fit using optimization with a threshold:
                ```python
                >>> threshold = np.quantile(data, 0.80)
                >>> print(threshold)
                1.5717000000000005
                >>> parameters = gumbel_dist.fit_model(
                ...     method="optimization",
                ...     obj_func=Gumbel.truncated_distribution,
                ...     threshold=threshold
                ... )
                Optimization terminated successfully.
                         Current function value: 0.000000
                         Iterations: 39
                         Function evaluations: 116
                -----KS Test--------
                Statistic = 0.107
                reject Hypothesis
                P value = 2.0977827855404345e-05

                ```
            # Note: When P value is less than the significance level, we reject the null hypothesis,
            # but in this case we're fitting the distribution to part of the data, not the whole data.
            ```
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        method = super().fit_model(method=method)  # type: ignore[assignment]

        if method == "mle" or method == "mm":
            param_list: Any = list(gumbel_r.fit(self.data, method=method))
        elif method == "lmoments":
            lm = Lmoments(self.data)
            lmu = lm.calculate()
            param_list = Lmoments.gumbel(lmu)
        elif method == "optimization":
            if obj_func is None or threshold is None:
                raise TypeError("threshold should be numeric value")

            param_list = gumbel_r.fit(self.data, method="mle")
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
        parameters: dict[str, float] = None,
    ) -> np.ndarray:
        """Calculate the inverse of the cumulative distribution function (quantile function).

        This method calculates the theoretical values (quantiles) corresponding to the given
        CDF values using the specified Gumbel distribution parameters.

        Args:
            cdf: CDF values (non-exceedance probabilities) for which to calculate the quantiles.
                Values should be between 0 and 1.
            parameters (dict[str, float]):
                If None, uses the parameters provided during initialization.
                    - loc (Numeric):
                        Location parameter of the Gumbel distribution
                    - scale (Numeric):
                        Scale parameter of the Gumbel distribution (must be positive)
                    ```python
                    {"loc": 0.0, "scale": 1.0}
                ```

        Returns:
            Numpy array containing the quantile values corresponding to the given CDF values.

        Raises:
            ValueError: If any CDF value is less than or equal to 0 or greater than 1.

        Examples:
            - Load sample data and initialize distribution:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel
                >>> data = np.loadtxt("examples/data/gumbel.txt")
                >>> parameters = {'loc': 0, 'scale': 1}
                >>> gumbel_dist = Gumbel(data, parameters)

                ```
            - Calculate quantiles for specific probabilities:
                ```python
                >>> cdf = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                >>> data_values = gumbel_dist.inverse_cdf(cdf)
                >>> print(data_values) # doctest: +SKIP
                [-0.83403245 -0.475885 0.08742157 0.67172699 1.49993999 2.25036733]

                ```

            - Calculate return levels for specific return periods:
                ```python
                >>> return_periods = [10, 50, 100]
                >>> probs = 1 - 1/np.array(return_periods)
                >>> return_levels = gumbel_dist.inverse_cdf(probs)
                >>> print(f"10-year return level: {return_levels[0]:.2f}")
                10-year return level: 2.25
                >>> print(f"50-year return level: {return_levels[1]:.2f}")
                50-year return level: 3.90
                >>> print(f"100-year return level: {return_levels[2]:.2f}")
                100-year return level: 4.60

                ```
        """
        if parameters is None:
            parameters = self.parameters

        cdf = np.array(cdf)
        if np.any(cdf < 0) or np.any(cdf > 1):
            raise ValueError(CDF_INVALID_VALUE_ERROR)

        qth = self._inv_cdf(cdf, parameters)  # type: ignore[arg-type]

        return qth

    @staticmethod
    def _inv_cdf(
        cdf: np.ndarray | list[float], parameters: dict[str, float]
    ) -> np.ndarray:
        """Calculate the inverse CDF (quantile function) values for Gumbel distribution.

        This method implements the Gumbel inverse CDF equation:
        Q(p) = loc - scale * ln(-ln(p))

        Args:
            cdf: CDF values (non-exceedance probabilities) for which to calculate quantiles.
                Values should be between 0 and 1.
            parameters: Dictionary of distribution parameters.
                Must contain:
                - "loc": Location parameter (ζ)
                - "scale": Scale parameter (δ), must be positive

        Returns:
            Numpy array containing the quantile values corresponding to the given CDF values.

        Raises:
            ValueError: If the scale parameter is negative or zero.
        """
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        qth = gumbel_r.ppf(cdf, loc=loc, scale=scale)

        return qth

    def ks(self) -> tuple:
        """Perform the Kolmogorov-Smirnov (KS) test for goodness of fit.

        This method tests whether the data follows the fitted Gumbel distribution using
        the Kolmogorov-Smirnov test. The test compares the empirical CDF of the data
        with the theoretical CDF of the fitted distribution.

        Returns:
            Tuple:
                - 0:
                    D statistic: The maximum absolute difference between the empirical and theoretical CDFs.
                    The smaller the D statistic, the more likely the data follows the distribution.
                    The KS test statistic measures the maximum distance between the empirical CDF
                    (Weibull plotting position) and the CDF of the reference distribution.
                - 1:
                    p-value The probability of observing a D statistic as extreme as the one calculated, assuming the
                    null hypothesis is true (data follows the distribution).
                    A high p-value (close to 1) suggests that there is a high probability that the sample comes from
                    the specified distribution.
                    If p-value < significance level (typically 0.05), reject the null hypothesis.

        Raises:
            ValueError:
                If the distribution parameters have not been estimated.

        Examples:
            - Import necessary libraries and initialize the Gumbel distribution:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel

                ```
            - Perform KS test:
                ```python
                >>> data = np.loadtxt("examples/data/gumbel.txt")
                >>> gumbel_dist = Gumbel(data)
                >>> gumbel_dist.fit_model()
                -----KS Test--------
                Statistic = 0.019
                Accept Hypothesis
                P value = 0.9937026761524456
                {'loc': np.float64(0.010101355750222706), 'scale': 1.0313042643102108}
                >>> d_stat, p_value = gumbel_dist.ks()
                -----KS Test--------
                Statistic = 0.019
                Accept Hypothesis
                P value = 0.9937026761524456

                ```
            - Interpret the results:
                ```python
                >>> alpha = 0.05
                >>> if p_value < alpha:
                ...     print(f"Reject the null hypothesis (p-value: {p_value:.4f} < {alpha})")
                ...     print("The data does not follow the fitted Gumbel distribution.")
                ... else:
                ...     print(f"Cannot reject the null hypothesis (p-value: {p_value:.4f} >= {alpha})")
                ...     print("The data may follow the fitted Gumbel distribution.")
                Cannot reject the null hypothesis (p-value: 0.9937 >= 0.05)
                The data may follow the fitted Gumbel distribution.

                ```
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """Perform the Chi-square test for goodness of fit.

        This method tests whether the data follows the fitted Gumbel distribution using the Chi-square test. The test
        compares the observed frequencies with the expected frequencies under the fitted distribution.

        Returns:
            Tuple:
                - Chi-square statistic:
                    The test statistic measuring the difference between observed and expected frequencies.
                - p-value:
                    The probability of observing a Chi-square statistic as extreme as the one calculated,
                    assuming the null hypothesis is true (data follows the distribution).
                    If p-value < significance level (typically 0.05), reject the null hypothesis. Returns None if the test
                    fails due to an exception.

        Raises:
            ValueError:
                If the distribution parameters have not been estimated.

        Examples:
            - Perform Chi-square test:
                ```python
                >>> import numpy as np
                >>> from statista.distributions import Gumbel
                >>> data = np.loadtxt("examples/data/gumbel.txt")
                >>> gumbel_dist = Gumbel(data)
                >>> gumbel_dist.fit_model()
                -----KS Test--------
                Statistic = 0.019
                Accept Hypothesis
                P value = 0.9937026761524456
                {'loc': np.float64(0.010101355750222706), 'scale': 1.0313042643102108}
                >>> gumbel_dist.chisquare() #doctest: +SKIP

                ```
            - Interpret the results:
                ```python
                >>> alpha = 0.05
                >>> if p_value < alpha: #doctest: +SKIP
                ...     print(f"Reject the null hypothesis (p-value: {p_value:.4f} < {alpha})")
                ...     print("The data does not follow the fitted Gumbel distribution.")
                >>> else: #doctest: +SKIP
                ...     print(f"Cannot reject the null hypothesis (p-value: {p_value:.4f} >= {alpha})")
                ...     print("The data may follow the fitted Gumbel distribution.")
                ```
        """
        return super().chisquare()

    def confidence_interval(  # type: ignore[override]
        self,
        alpha: float = 0.1,
        prob_non_exceed: np.ndarray = None,
        parameters: dict[str, float | Any] = None,
        plot_figure: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Figure, Axes]:
        """Calculate confidence intervals for the Gumbel distribution quantiles.

        This method calculates the upper and lower bounds of the confidence interval
        for the quantiles of the Gumbel distribution. It can also generate a plot of the
        confidence intervals.

        Args:
            alpha (float):
                Significance level for the confidence interval. Default is 0.1 (90% confidence interval).
            prob_non_exceed: Non-exceedance probabilities for which to calculate quantiles.
                If None, uses the empirical CDF calculated using Weibull plotting positions.
            parameters (dict[str, Any]):
                If None, uses the parameters provided during initialization.
                - loc (Numeric):
                    Location parameter of the Gumbel distribution
                - scale (Numeric):
                    Scale parameter of the Gumbel distribution (must be positive)
                ```python
                {"loc": 0.0, "scale": 1.0}
                ```
            plot_figure (bool):
                Whether to generate a plot of the confidence intervals. Default is False.
            **kwargs:
                Additional keyword arguments to pass to the plotting function.
                    - fig_size:
                        Size of the figure as a tuple (width, height). Default is (6, 6).
                    - fontsize:
                        Font size for plot labels. Default is 11.
                    - marker_size:
                        Size of markers in the plot.

        Returns:
            If plot_figure is False:
                Tuple containing:
                - Numpy array of upper bound values
                - Numpy array of lower bound values
            If plot_figure is True:
                Tuple containing:
                - Numpy array of upper bound values
                - Numpy array of lower bound values
                - Figure object
                - Axes object

        Raises:
            ValueError: If the scale parameter is negative or zero.

        Examples:
            - Load data and initialize distribution:
                ```python
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from statista.distributions import Gumbel
                >>> data = np.loadtxt("examples/data/time_series2.txt")
                >>> parameters = {"loc": 463.8040, "scale": 220.0724}
                >>> gumbel_dist = Gumbel(data, parameters)

                ```
            - Calculate confidence intervals
                ```python
                >>> upper, lower = gumbel_dist.confidence_interval(alpha=0.1)

                ```
            - Generate a confidence interval plot:
                ```python
                >>> upper, lower, fig, ax = gumbel_dist.confidence_interval(
                ...     alpha=0.1,
                ...     plot_figure=True,
                ...     marker_size=10
                ... )
                >>> plt.show()

                ```
            ![image](./../../_images/distributions/gumbel-confidence-interval.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        if prob_non_exceed is None:
            prob_non_exceed = PlottingPosition.weibul(self.data)

        qth = self._inv_cdf(prob_non_exceed, parameters)
        y = [-np.log(-np.log(j)) for j in prob_non_exceed]
        std_error = [
            (scale / np.sqrt(len(self.data)))
            * np.sqrt(1.1087 + 0.5140 * j + 0.6079 * j**2)
            for j in y
        ]
        v = norm.ppf(1 - alpha / 2)
        q_upper = np.array([qth[j] + v * std_error[j] for j in range(len(qth))])
        q_lower = np.array([qth[j] - v * std_error[j] for j in range(len(qth))])

        if plot_figure:
            # if the prob_non_exceed is given, check if the length is the same as the data
            if len(prob_non_exceed) != len(self.data):
                raise ValueError(PROB_NON_EXCEEDENCE_ERROR)

            fig, ax = Plot.confidence_level(
                qth, self.data, q_lower, q_upper, alpha=alpha, **kwargs  # type: ignore[arg-type]
            )
            return q_upper, q_lower, fig, ax
        else:
            return q_upper, q_lower

    def plot(
        self,
        fig_size: tuple[float, float] = (10, 5),
        xlabel: str = PDF_XAXIS_LABEL,
        ylabel: str = "cdf",
        fontsize: int = 15,
        cdf: np.ndarray | list | None = None,
        parameters: dict[str, float | Any] = None,
    ) -> tuple[Figure, tuple[Axes, Axes]]:  # pylint: disable=arguments-differ
        """Probability plot.

        Probability Plot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculates the confidence interval.

        Args:
            fig_size: tuple, Default is (10, 5).
                Size of the figure.
            cdf: [np.ndarray]
                theoretical cdf calculated using weibul or using the distribution cdf function.
            fig_size: [tuple]
                Default is (10, 5)
            xlabel: [str]
                Default is "Actual data"
            ylabel: [str]
                Default is "cdf"
            fontsize: [float]
                Default is 15.
            parameters: dict[str, str]
                {"loc": val, "scale": val}
                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.

        Returns:
            Figure:
                matplotlib figure object
            tuple[Axes, Axes]:
                matplotlib plot axes

        Examples:
        - Instantiate the Gumbel class with the data and the parameters:
            ```python
            >>> import matplotlib.pyplot as plt
            >>> data = np.loadtxt("examples/data/time_series2.txt")
            >>> parameters = {"loc": 463.8040, "scale": 220.0724}
            >>> gumbel_dist = Gumbel(data, parameters)

            ```
        - To calculate the confidence interval, we need to provide the confidence level (`alpha`).
            ```python
            >>> fig, ax = gumbel_dist.plot()
            >>> print(fig)
            Figure(1000x500)
            >>> print(ax)
            (<Axes: xlabel='Actual data', ylabel='pdf'>, <Axes: xlabel='Actual data', ylabel='cdf'>)

            ```
            ![gumbel-plot](./../../_images/gumbel-plot.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        scale = parameters.get("scale")

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        if cdf is None:
            cdf = PlottingPosition.weibul(self.data)
        else:
            # if the cdf is given, check if the length is the same as the data
            if len(cdf) != len(self.data):
                raise ValueError(
                    "Length of cdf does not match the length of data, use the `PlottingPosition.weibul(data)` "
                    "to the get the non-exceedance probability"
                )

        q_x = np.linspace(
            float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
        )
        pdf_fitted: Any = self.pdf(parameters=parameters, data=q_x)
        cdf_fitted: Any = self.cdf(parameters=parameters, data=q_x)

        fig, ax = Plot.details(
            q_x,
            self.data,
            pdf_fitted,
            cdf_fitted,
            cdf,
            fig_size=fig_size,
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize=fontsize,
        )

        return fig, ax
