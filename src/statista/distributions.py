"""Statistical distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from statistics import mode
from typing import Any, Callable

import numpy as np
import scipy.optimize as so
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy.stats import chisquare, expon, genextreme, gumbel_r, ks_2samp, norm

from statista.confidence_interval import ConfidenceInterval
from statista.parameters import Lmoments
from statista.plot import Plot
from statista.utils import merge_small_bins

ninf = 1e-5

__all__ = [
    "PlottingPosition",
    "Gumbel",
    "GEV",
    "Exponential",
    "Normal",
    "Distributions",
]

SCALE_PARAMETER_ERROR = "Scale parameter is negative"
CDF_INVALID_VALUE_ERROR = "cdf Value Invalid"
OBJ_FUNCTION_THRESHOULD_ERROR = "obj_func and threshold should be numeric value"
PROB_NON_EXCEEDENCE_ERROR = """
Length of prob_non_exceed does not match the length of data, use the `PlottingPosition.weibul(data)`
to the get the non-exceedance probability
"""
PDF_XAXIS_LABEL = "Actual data"


class PlottingPosition:
    """PlottingPosition."""

    @staticmethod
    def return_period(prob_non_exceed: list | np.ndarray) -> np.ndarray:
        """Return Period.

        Args:
            prob_non_exceed:
                non-exceedance probability.

        Returns:
            array:
                calculated return period.

        Examples:
            - First generate some random numbers between 0 and 1 as a non-exceedance probability. then use this non-exceedance
                to calculate the return period.
                ```python
                >>> import numpy as np
                >>> from statista.distributions import PlottingPosition
                >>> data = np.random.random(15)
                >>> rp = PlottingPosition.return_period(data)
                >>> print(rp) # doctest: +SKIP
                [ 1.33088992  4.75342173  2.46855419  1.42836548  2.75320582  2.2268505
                  8.06500888 10.56043917 18.28884687  1.10298241  1.2113997   1.40988022
                  1.02795867  1.01326322  1.05572108]

                ```
        """
        if any(np.asarray(prob_non_exceed) > 1):
            raise ValueError("Non-exceedance probability should be less than 1")
        prob_non_exceed = np.array(prob_non_exceed)
        t = 1 / (1 - prob_non_exceed)
        return t

    @staticmethod
    def weibul(data: list | np.ndarray, return_period: int = False) -> np.ndarray:
        """Weibul.

        Weibul method to calculate the cumulative distribution function cdf or
        return period.

        Args:
            data:
                list/array of the data.
            return_period:
                False to calculate the cumulative distribution function cdf or True to calculate the return period.
                Default=False

        Returns:
            cdf/T:
                cumulative distribution function or return period.

        Examples:
            ```python
            >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> cdf = PlottingPosition.weibul(data)
            >>> print(cdf)
            [0.09090909 0.18181818 0.27272727 0.36363636 0.45454545 0.54545455
             0.63636364 0.72727273 0.81818182 0.90909091]

            ```
        """
        data = np.array(data)
        data.sort()
        n = len(data)
        cdf = np.array(range(1, n + 1)) / (n + 1)
        if not return_period:
            return cdf
        else:
            t = PlottingPosition.return_period(cdf)
            return t


class AbstractDistribution(ABC):
    """Abstract base class for probability distributions.

    This class defines the interface for all probability distribution classes in the package.
    It provides common functionality for calculating probability density functions (PDF),
    cumulative distribution functions (CDF), fitting models to data, and more.

    Attributes:
        _data (np.ndarray): The data array used for distribution calculations.
        _parameters (dict[str, float]): Distribution parameters.
    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: dict[str, float] = None,
    ):
        """Initialize the distribution with data or parameters.

        Args:
            data:
                Data time series as a list or numpy array.
            parameters:
                Dictionary of distribution parameters.
                - loc: Location parameter
                - scale: Scale parameter
                - shape: Shape parameter (if applicable)
                ```
                {"loc": 0.0, "scale": 1.0, "shape": 0.0}
                ```

        Raises:
            ValueError:
                If neither data nor parameters are provided.
            TypeError:
                If data is not a list or numpy array, or if parameters argument is not a dictionary.
        """
        if data is None and parameters is None:
            raise ValueError("Either data or parameters must be provided")

        self._data: np.ndarray | None
        if isinstance(data, list) or isinstance(data, np.ndarray):
            self._data = np.array(data)
        elif data is None:
            self._data = data
        else:
            raise TypeError("The `data` argument should be list or numpy array")

        self._parameters: dict[str, float] | None
        if isinstance(parameters, dict) or parameters is None:
            self._parameters = parameters
        else:
            raise TypeError("The `parameters` argument should be dictionary")

    def __str__(self) -> str:
        message = ""
        if self.data is not None:
            message += f"""
                    Dataset of {len(self.data)} value
                    min: {np.min(self.data)}
                    max: {np.max(self.data)}
                    mean: {np.mean(self.data)}
                    median: {np.median(self.data)}
                    mode: {mode(self.data)}
                    std: {np.std(self.data)}
                    Distribution : {self.__class__.__name__}
                    parameters: {self.parameters}
                    """
        if self.parameters is not None:
            message += f"""
                Distribution : {self.__class__.__name__}
                parameters: {self.parameters}
                """
        return message

    @property
    def parameters(self) -> dict[str, float]:
        """Get the distribution parameters.

        Returns:
            Dictionary of distribution parameters (e.g., {"loc": 0.0, "scale": 1.0}).
        """
        return self._parameters  # type: ignore[return-value]

    @parameters.setter
    def parameters(self, value: dict[str, float]):
        """Set the distribution parameters.

        Args:
            value: Dictionary of distribution parameters.
                Example: {"loc": 0.0, "scale": 1.0}
        """
        self._parameters = value

    @property
    def data(self) -> ndarray:
        """Get the data array.

        Returns:
            Numpy array containing the data used for distribution calculations.
        """
        return self._data  # type: ignore[return-value]

    @property
    def data_sorted(self) -> ndarray:
        """Get the data array sorted in ascending order.

        Returns:
            Numpy array containing the sorted data.
        """
        return np.sort(self.data)

    @property
    def kstable(self) -> float:
        """Get the Kolmogorov-Smirnov test critical value.

        Returns:
            Critical value for the Kolmogorov-Smirnov test (1.22/sqrt(n)).
        """
        return 1.22 / np.sqrt(len(self.data))

    @property
    def cdf_weibul(self) -> ndarray:
        """Get the empirical CDF using Weibull plotting position.

        Returns:
            Numpy array containing the empirical CDF values.
        """
        return PlottingPosition.weibul(self.data)

    @staticmethod
    @abstractmethod
    def _pdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        """Calculate the probability density function (PDF) values.

        This is an abstract method that must be implemented by subclasses.

        Args:
            data: Data points for which to calculate PDF values.
            parameters: Dictionary of distribution parameters.
                Example: {"loc": 0.0, "scale": 1.0}

        Returns:
            Numpy array containing the PDF values for each data point.
        """
        pass

    @abstractmethod
    def pdf(
        self,
        parameters: dict[str, float | Any] = None,
        plot_figure: bool = False,
        fig_size: tuple = (6, 5),
        xlabel: str = PDF_XAXIS_LABEL,
        ylabel: str = "pdf",
        fontsize: float | int = 15,
        data: list[float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, Figure, Axes]:
        """Calculate the probability density function (PDF) values.

        This method calculates the PDF values for the given data using the specified
        distribution parameters. It can also generate a plot of the PDF.

        Args:
            parameters: Dictionary of distribution parameters.
                Example: {"loc": 0.0, "scale": 1.0}
                If None, uses the parameters provided during initialization.
            plot_figure: Whether to generate a plot of the PDF.
                Default is False.
            fig_size: Size of the figure as a tuple (width, height).
                Default is (6, 5).
            xlabel: Label for the x-axis.
                Default is "Actual data".
            ylabel: Label for the y-axis.
                Default is "pdf".
            fontsize: Font size for plot labels.
                Default is 15.
            data: Data points for which to calculate PDF values.
                If None, uses the data provided during initialization.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            If plot_figure is False:
                Numpy array containing the PDF values for each data point.
            If plot_figure is True:
                Tuple containing:
                - Numpy array of PDF values
                - Figure object
                - Axes object
        """

        if data is None:
            ts: Any = self.data
            data_sorted: Any = self.data_sorted
        else:
            ts = data
            data_sorted = np.sort(data)

        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        pdf = self._pdf_eq(ts, parameters)  # type: ignore[arg-type]

        if plot_figure:
            qx = np.linspace(float(data_sorted[0]), 1.5 * float(data_sorted[-1]), 10000)
            pdf_fitted = self.pdf(parameters=parameters, data=qx)

            fig, ax = Plot.pdf(
                qx,
                pdf_fitted,  # type: ignore[arg-type]
                data_sorted,
                fig_size=fig_size,
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=int(fontsize),
            )
            return pdf, fig, ax
        else:
            return pdf

    @staticmethod
    @abstractmethod
    def _cdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        """Calculate the cumulative distribution function (CDF) values.

        This is an abstract method that must be implemented by subclasses.

        Args:
            data: Data points for which to calculate CDF values.
            parameters: Dictionary of distribution parameters.
                Example: {"loc": 0.0, "scale": 1.0}

        Returns:
            Numpy array containing the CDF values for each data point.
        """
        pass

    @abstractmethod
    def cdf(
        self,
        parameters: dict[str, float | Any] = None,
        plot_figure: bool = False,
        fig_size: tuple = (6, 5),
        xlabel: str = "data",
        ylabel: str = "cdf",
        fontsize: int = 15,
        data: list[float] | np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, Figure, Axes]:
        """Calculate the cumulative distribution function (CDF) values.

        This method calculates the CDF values for the given data using the specified
        distribution parameters. It can also generate a plot of the CDF.

        Args:
            parameters: Dictionary of distribution parameters.
                Example: {"loc": 0.0, "scale": 1.0}
                If None, uses the parameters provided during initialization.
            plot_figure: Whether to generate a plot of the CDF.
                Default is False.
            fig_size: Size of the figure as a tuple (width, height).
                Default is (6, 5).
            xlabel: Label for the x-axis.
                Default is "data".
            ylabel: Label for the y-axis.
                Default is "cdf".
            fontsize: Font size for plot labels.
                Default is 15.
            data: Data points for which to calculate CDF values.
                If None, uses the data provided during initialization.

        Returns:
            If plot_figure is False:
                Numpy array containing the CDF values for each data point.
            If plot_figure is True:
                Tuple containing:
                - Numpy array of CDF values
                - Figure object
                - Axes object
        """
        if data is None:
            ts: Any = self.data
            data_sorted: Any = self.data_sorted
        else:
            ts = data
            data_sorted = np.sort(data)

        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        cdf = self._cdf_eq(ts, parameters)  # type: ignore[arg-type]

        if plot_figure:
            qx = np.linspace(float(data_sorted[0]), 1.5 * float(data_sorted[-1]), 10000)
            cdf_fitted = self.cdf(parameters=parameters, data=qx)

            cdf_weibul = PlottingPosition.weibul(data_sorted)

            fig, ax = Plot.cdf(
                qx,
                cdf_fitted,  # type: ignore[arg-type]
                data_sorted,
                cdf_weibul,
                fig_size=fig_size,
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=fontsize,
            )

            return cdf, fig, ax
        else:
            return cdf

    @abstractmethod
    def fit_model(
        self,
        method: str = "mle",
        obj_func: Callable | None = None,
        threshold: None | float | int = None,
        test: bool = True,
    ) -> dict[str, str] | Any:
        """Fit the distribution parameters to the data.

        This method estimates the distribution parameters based on the provided data.
        It supports different estimation methods, including Maximum Likelihood Estimation (MLE),
        Method of Moments (MM), and L-moments.

        When a threshold is provided, the method uses a partial likelihood approach:
        - L1: likelihood for values above the threshold (x >= threshold)
        - L2: probability that the threshold will be exceeded (1-F(threshold))
        The parameters are estimated by maximizing the product L1*L2.

        Args:
            method: Estimation method to use.
                Options: 'mle' (Maximum Likelihood Estimation),
                         'mm' (Method of Moments),
                         'lmoments' (L-moments),
                         'optimization' (Custom optimization)
                Default is 'mle'.
            obj_func: Custom objective function to use for parameter estimation.
                Only used when method is 'optimization'.
                Default is None.
            threshold: Value above which to consider data points.
                If provided, only data points above this threshold are used for estimation.
                Default is None (use all data points).
            test: Whether to perform goodness-of-fit tests after estimation.
                Default is True.

        Returns:
            Dictionary of estimated distribution parameters.
            Example: {"loc": 0.0, "scale": 1.0}

        Raises:
            ValueError: If the data is not sufficient for parameter estimation.
        """
        method = method.lower()
        if method not in ["mle", "mm", "lmoments", "optimization"]:
            raise ValueError(
                f"{method} value should be 'mle', 'mm', 'lmoments' or 'optimization'"
            )
        return method

    @abstractmethod
    def inverse_cdf(
        self,
        cdf: np.ndarray | list[float],
        parameters: dict[str, float | Any],
    ) -> np.ndarray:
        """Calculate the inverse of the cumulative distribution function (quantile function).

        This method calculates the theoretical values corresponding to the given CDF values
        using the specified distribution parameters.

        Args:
            cdf: CDF values (non-exceedance probabilities) for which to calculate the quantiles.
                Values should be between 0 and 1.
            parameters: Dictionary of distribution parameters.
                Example: {"loc": 0.0, "scale": 1.0}

        Returns:
            Numpy array containing the quantile values corresponding to the given CDF values.
        """
        pass

    @abstractmethod
    def ks(self) -> tuple:
        """Perform the Kolmogorov-Smirnov (KS) test for goodness of fit.

        This method tests whether the data follows the fitted distribution using
        the Kolmogorov-Smirnov test. The test compares the empirical CDF of the data
        with the theoretical CDF of the fitted distribution.

        Returns:
            Tuple containing:
            - D statistic: The maximum absolute difference between the empirical and theoretical CDFs.
              The smaller the D statistic, the more likely the data follows the distribution.
            - p-value: The probability of observing a D statistic as extreme as the one calculated,
              assuming the null hypothesis is true (data follows the distribution).
              If p-value < significance level (typically 0.05), reject the null hypothesis.

        Raises:
            ValueError: If the distribution parameters have not been estimated.
        """
        if self.parameters is None:
            raise ValueError(
                "The Value of parameters is unknown. Please use 'fit_model' to estimate the distribution parameters"
            )
        qth = self.inverse_cdf(self.cdf_weibul, self.parameters)

        test = ks_2samp(self.data, qth)

        print("-----KS Test--------")
        print(f"Statistic = {test.statistic}")
        if test.statistic < self.kstable:
            print("Accept Hypothesis")
        else:
            print("reject Hypothesis")
        print(f"P value = {test.pvalue}")
        return test.statistic, test.pvalue

    @abstractmethod
    def chisquare(self) -> tuple[float, float]:
        """Perform the Chi-square test for goodness of fit.

        - `chisquare test` refers to Pearson’s chi square goodness of fit test. It is designed for
        categorical/count data: you observe how many points fall into each bin and compare those counts with the
        frequencies expected under some hypothesis

        This method tests whether the data follows the fitted distribution using the Chi-square test.
        The test compares the observed frequencies (number of values in each category/histogram bin) with the
        expected frequencies under the fitted distribution.

        Returns:
            Tuple containing:
            - Chi-square statistic: The test statistic measuring the difference between
              observed and expected frequencies.
              The χ² statistic is simply a measure of how far your observed counts deviate from the counts you would
              expect if the fitted distribution were correct. For each bin 𝑖 we compute the squared difference
              between the observed count 𝑂𝑖 and the expected count 𝐸𝑖, scaled by 𝐸𝑖, and then sum over all bins:

            - p-value: The probability of observing a Chi-square statistic as extreme as the one calculated,
              assuming the null hypothesis is true (data follows the distribution).
              If p-value < significance level (typically 0.05), reject the null hypothesis.
            Returns None if the test fails due to an exception.

        Raises:
            ValueError: If the distribution parameters have not been estimated.
        """
        if self.parameters is None:
            raise ValueError(
                "The Value of parameters is unknown. Please use 'fit_model' to estimate the distribution parameters"
            )

        bin_edges = np.histogram_bin_edges(self.data, bins="sturges")
        obs_counts, _ = np.histogram(self.data, bins=bin_edges)

        expected_prob = np.diff(self._cdf_eq(bin_edges, self.parameters))
        expected_counts = expected_prob * len(self.data)

        # Pearson’s χ² test assumes each expected count is sufficiently large (at least about 5); otherwise the asymptotic χ² approximation is unreliable
        merged_obs, merged_exp = merge_small_bins(
            obs_counts.tolist(), expected_counts.tolist()
        )

        test = chisquare(merged_obs, f_exp=merged_exp, ddof=len(self.parameters))
        return test.statistic, test.pvalue

    def confidence_interval(  # type: ignore[empty-body]
        self,
        alpha: float = 0.1,
        plot_figure: bool = False,
        prob_non_exceed: np.ndarray = None,
        parameters: dict[str, float | Any] = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Figure, Axes]:
        """Calculate confidence intervals for the distribution quantiles.

        This method calculates the upper and lower bounds of the confidence interval
        for the quantiles of the distribution. It can also generate a plot of the
        confidence intervals.

        Args:
            alpha: Significance level for the confidence interval.
                Default is 0.1 (90% confidence interval).
            plot_figure: Whether to generate a plot of the confidence intervals.
                Default is False.
            prob_non_exceed: Non-exceedance probabilities for which to calculate quantiles.
                If None, uses the empirical CDF calculated using Weibull plotting positions.
            parameters: Dictionary of distribution parameters.
                Example: {"loc": 0.0, "scale": 1.0}
                If None, uses the parameters provided during initialization.

        Returns:
            If plot_figure is False:
                Tuple containing:
                - Numpy array of upper-bound values
                - Numpy array of lower-bound values
            If plot_figure is True:
                Tuple containing:
                - Numpy array of upper-bound values
                - Numpy array of lower-bound values
                - Figure object
                - Axes object
        """
        ...  # pragma: no cover

    def plot(
        self,
        fig_size: tuple = (10, 5),
        xlabel: str = PDF_XAXIS_LABEL,
        ylabel: str = "cdf",
        fontsize: int = 15,
        cdf: np.ndarray | None = None,
        parameters: dict[str, float | Any] | None = None,
    ) -> Any:
        """Generate probability plots for the distribution.

        This method creates probability plots comparing the empirical distribution
        of the data with the theoretical distribution. It calculates theoretical values
        based on the distribution parameters and can also display confidence intervals.

        Args:
            fig_size: Size of the figure as a tuple (width, height).
                Default is (10, 5).
            xlabel: Label for the x-axis.
                Default is "Actual data".
            ylabel: Label for the y-axis.
                Default is "cdf".
            fontsize: Font size for plot labels.
                Default is 15.
            cdf: Theoretical CDF values.
                If None, uses the empirical CDF calculated using Weibull plotting positions.
            parameters: Dictionary of distribution parameters.
                Example: {"loc": 0.0, "scale": 1.0}
                If None, uses the parameters provided during initialization.

        Returns:
            Tuple containing:
            - List of Figure objects
            - List of Axes objects
        """
        ...  # pragma: no cover


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

    Mathematical Details:
        - Probability Density Function (PDF):
          f(x; ζ, δ) = (1/δ) * exp(-(x-ζ)/δ) * exp(-exp(-(x-ζ)/δ))

          where ζ (zeta) is the location parameter, and δ (delta) is the scale parameter.

        - Cumulative Distribution Function (CDF):
          F(x; ζ, δ) = exp(-exp(-(x-ζ)/δ))

        - The location parameter ζ shifts the distribution along the x-axis, determining
          the mode (peak) of the distribution. It can range from negative to positive infinity.

        - The scale parameter δ controls the spread of the distribution. A larger scale
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
                ![gamma-pdf](./../_images/distributions/gamma-pdf-1.png)

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
                ![gamma-pdf](./../_images/distributions/gamma-random-1.png)

                - Plot the CDF of the random data:
                    ```python
                    >>> _ = gumbel_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

                    ```
                    ![gamma-cdf](./../_images/distributions/gamma-cdf-1.png)

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
                ![gamma-cdf](./../_images/distributions/gamma-cdf-2.png)

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
        parameters = {"loc": loc, "scale": scale}
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
    ) -> dict[str, float]:
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

        param: dict[str, float] = {"loc": param_list[0], "scale": param_list[1]}
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

        if any(cdf) <= 0 or any(cdf) > 1:  # type: ignore[arg-type]
            raise ValueError(CDF_INVALID_VALUE_ERROR)

        cdf = np.array(cdf)
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
            ![image](./../_images/distributions/gumbel-confidence-interval.png)
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
            ![gumbel-plot](./../_images/gumbel-plot.png)
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


class GEV(AbstractDistribution):
    """GEV (Generalized Extreme value statistics)

    - The Generalized Extreme Value (GEV) distribution is used to model the largest or smallest value among a large
        set of independent, identically distributed random values.
    - The GEV distribution encompasses three types of distributions: Gumbel, Fréchet, and Weibull, which are
        distinguished by a shape parameter (:math:`\\xi` (xi)).

    - The probability density function (PDF) of the Generalized-extreme-value distribution is:

        .. math::
            f(x; \\zeta, \\delta, \\xi)=\\frac{1}{\\delta}\\mathrm{*}{\\mathrm{Q(x)}}^{\\xi+1}\\mathrm{
            *} e^{\\mathrm{-Q(x)}}

        .. math::
            Q(x; \\zeta, \\delta, \\xi)=
            \\begin{cases}
                \\left(1+ \\xi \\left(\\frac{x-\\zeta}{\\delta} \\right) \\right)^\\frac{-1}{\\xi} &
                \\quad\\land\\xi\\neq 0 \\\\
                e^{- \\left(\\frac{x-\\zeta}{\\delta} \\right)} & \\quad \\land \\xi=0
            \\end{cases}
          :label: gev-pdf

        Where the :math:`\\delta` (delta) is the scale parameter, :math:`\\zeta` (zeta) is the location parameter,
        and :math:`\\xi` (xi) is the shape parameter.

    - The location parameter :math:`\\zeta` shifts the distribution along the x-axis. It essentially determines the mode
        (peak) of the distribution and its location. Changing the location parameter moves the distribution left or
        right without altering its shape. The location parameter ranges from negative infinity to positive infinity.
    - The scale parameter :math:`\\delta` controls the spread or dispersion of the distribution. A larger scale parameter
        results in a wider distribution, while a smaller scale parameter results in a narrower distribution. It must
        always be positive.
    - The shape parameter :math:`\\xi` (xi) determines the shape of the distribution. The shape parameter can be positive,
        negative, or zero. The shape parameter is used to classify the GEV distribution into three types: :math:`\\xi = 0`
        Gumbel (Type I), :math:`\\xi > 0` Fréchet (Type II), and :math:`\\xi < 0` Weibull (Type III). The shape
        parameter determines the tail behavior of the distribution.

        In hydrology, the distribution is reparametrized with :math:`k=-\\xi` (xi) (El Adlouni et al., 2008)
        The cumulative distribution functions.

    - The cumulative distribution functions.

        .. math::
            F(x; \\zeta, \\delta, \\xi)=
            \\begin{cases}
                \\exp\\left(- \\left(1+ \\xi \\left(\\frac{x-\\zeta}{\\delta} \\right) \\right)^\\frac{-1}{\\xi} \\right) &
                \\quad\\land\\xi\\neq 0 and 1 + \\xi \\left( \\frac{x-\\zeta}{\\delta}\\right) \\\\
                \\exp\\left(- \\exp\\left(- \\frac{x-\\zeta}{\\delta} \\right) \\right) & \\quad \\land \\xi=0
            \\end{cases}
          :label: gev-cdf

    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: dict[str, float] = None,
    ):
        """GEV.

        Args:
            data: [list]
                data time series.
            parameters: dict[str, str]
                {"loc": val, "scale": val, "shape": value}

                - loc: [numeric]
                    location parameter of the GEV distribution.
                - scale: [numeric]
                    scale parameter of the GEV distribution.
                - shape: [numeric]
                    shape parameter of the GEV distribution.

        Examples:
            - First load the sample data.
                ```python
                >>> data = np.loadtxt("examples/data/gev.txt")

                ```
        - I nstantiate the Gumbel class only with the data.
            ```python
            >>> gev_dist = GEV(data)
            >>> print(gev_dist) # doctest: +SKIP
            <statista.distributions.Gumbel object at 0x000001CDDE9563F0>

            ```
        - You can also instantiate the Gumbel class with the data and the parameters if you already have them.
            ```python
            >>> parameters = {"loc": 0, "scale": 1, "shape": 0.1}
            >>> gev_dist = GEV(data, parameters)
            >>> print(gev_dist) # doctest: +SKIP
            <statista.distributions.Gumbel object at 0x000001CDDEB32C00>
            ```
        """
        super().__init__(data, parameters)

    @staticmethod
    def _pdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")

        pdf = genextreme.pdf(data, loc=loc, scale=scale, c=shape)
        return pdf

    def pdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: dict[str, float] = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """pdf.

        Returns the value of GEV's pdf with parameters loc and scale at x.

        Args:
            parameters: dict[str, float], optional, default is None.
                if not provided, the parameters provided in the class initialization will be used.
                {"loc": val, "scale": val, "shape": value}

                - loc: [numeric]
                    location parameter of the GEV distribution.
                - scale: [numeric]
                    scale parameter of the GEV distribution.
                - shape: [numeric]
                    shape parameter of the GEV distribution.
            data : np.ndarray, default is None.
                array if you want to calculate the pdf for different data than the time series given to the constructor
                method.
            plot_figure: [bool]
                Default is False.
            kwargs:
                fig_size: [tuple]
                    Default is (6, 5).
                xlabel: [str]
                    Default is "Actual data".
                ylabel: [str]
                    Default is "pdf".
                fontsize: [int]
                    Default is 15

        Returns:
            pdf: [np.ndarray]
                probability density function pdf.
            fig: matplotlib.figure.Figure, if `plot_figure` is True.
                Figure object.
            ax: matplotlib.axes.Axes, if `plot_figure` is True.
                Axes object.

        Examples:
            - To calculate the pdf of the GEV distribution, we need to provide the parameters.
            ```python
            >>> import numpy as np
            >>> from statista.distributions import GEV
            >>> data = np.loadtxt("examples/data/gev.txt")
            >>> parameters = {"loc": 0, "scale": 1, "shape": 0.1}
            >>> gev_dist = GEV(data, parameters)
            >>> _ = gev_dist.pdf(plot_figure=True)

            ```
            ![gev-random-pdf](./../_images/gev-random-pdf.png)
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
        """Generate Random Variable.

        Args:
            size: int
                size of the random generated sample.
            parameters: dict[str, str]
                {"loc": val, "scale": val}

                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.

        Returns:
            data: [np.ndarray]
                random generated data.

        Examples:
            - To generate a random sample that follow the gumbel distribution with the parameters loc=0 and scale=1.
                ```python
                >>> parameters = {'loc': 0, 'scale': 1, "shape": 0.1}
                >>> gev_dist = GEV(parameters=parameters)
                >>> random_data = gev_dist.random(100)

                ```
            - then we can use the `pdf` method to plot the pdf of the random data.
                ```python
                >>> _ = gev_dist.pdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![gev-random-pdf](./../_images/gev-random-pdf.png)
                ```
                >>> _ = gev_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![gev-random-cdf](./../_images/gev-random-cdf.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        random_data = genextreme.rvs(loc=loc, scale=scale, c=shape, size=size)
        return random_data

    @staticmethod
    def _cdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")
        # equation https://www.rdocumentation.org/packages/evd/versions/2.3-6/topics/fextreme
        # z = (ts - loc) / scale
        # if shape == 0:
        #     # GEV is Gumbel distribution
        #     cdf = np.exp(-np.exp(-z))
        # else:
        #     y = 1 - shape * z
        #     cdf = list()
        #     for y_i in y:
        #         if y_i > ninf:
        #             logY = -np.log(y_i) / shape
        #             cdf.append(np.exp(-np.exp(-logY)))
        #         elif shape < 0:
        #             cdf.append(0)
        #         else:
        #             cdf.append(1)
        #
        # cdf = np.array(cdf)
        cdf = genextreme.cdf(data, c=shape, loc=loc, scale=scale)
        return cdf

    def cdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: dict[str, float | Any] = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[np.ndarray, Figure, Axes] | np.ndarray
    ):  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        Args:
            parameters: dict[str, str], optional, default is None.
                if not provided, the parameters provided in the class initialization will be used.
                {"loc": val, "scale": val}

                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.
            data : np.ndarray, default is None.
                array if you want to calculate the cdf for different data than the time series given to the constructor
                method.
            plot_figure: [bool]
                Default is False.
            kwargs:
                fig_size: [tuple]
                    Default is (6, 5).
                xlabel: [str]
                    Default is "Actual data".
                ylabel: [str]
                    Default is "cdf".
                fontsize: [int]
                    Default is 15.

        Returns:
            cdf: [array]
                cumulative distribution function cdf.
            fig: matplotlib.figure.Figure, if `plot_figure` is True.
                Figure object.
            ax: matplotlib.axes.Axes, if `plot_figure` is True.
                Axes object.

        Examples:
            - To calculate the cdf of the GEV distribution, we need to provide the parameters.
                ```python
                >>> data = np.loadtxt("examples/data/gev.txt")
                >>> parameters = {"loc": 0, "scale": 1, "shape": 0.1}
                >>> gev_dist = GEV(data, parameters)
                >>> _ = gev_dist.cdf(plot_figure=True)

                ```
            ![gev-random-cdf](./../_images/gev-random-cdf.png)
        """
        result = super().cdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )  # type: ignore[misc]
        return result

    def return_period(self, parameters: dict[str, Any], data: np.ndarray):
        """return_period.

            calculate return period calculates the return period for a list/array of values or a single value.

        Args:
            data (list/array/float):
                value you want the coresponding return value for
            parameters (dict[str, Any]):
                {"loc": val, "scale": val, "shape": value}

                - shape (float):
                    shape parameter
                - loc (float):
                    location parameter
                - scale (float):
                    scale parameter

        Returns:
            float:
                return period
        """
        cdf: Any = self.cdf(parameters=parameters, data=data)

        rp = 1 / (1 - cdf)

        return rp

    def fit_model(
        self,
        method: str = "mle",
        obj_func=None,
        threshold: int | float | None = None,
        test: bool = True,
    ) -> dict[str, float]:
        """Fit model.

        fit_model estimates the distribution parameter based on MLM
        (Maximum likelihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for the values below (x < C), now the likeliest parameters
        are those at the max value of multiplication between two functions max(L1*L2).

        In this case, the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Args:
            obj_func : [function]
                function to be used to get the distribution parameters.
            threshold : [numeric]
                Value you want to consider only the greater values.
            method : [string]
                'mle', 'mm', 'lmoments', optimization
            test: bool
                Default is True

        Returns:
            dict[str, str]:
                {"loc": val, "scale": val}

                - loc: [numeric]
                    location parameter of the GEV distribution.
                - scale: [numeric]
                    scale parameter of the GEV distribution.
                - shape: [numeric]
                    shape parameter of the GEV distribution.

        Examples:
            - Instantiate the Gumbel class only with the data.
                ```python
                >>> data = np.loadtxt("examples/data/gev.txt")
                >>> gev_dist = GEV(data)

                ```
            - Then use the `fit_model` method to estimate the distribution parameters. the method takes the method as
                parameter, the default is 'mle'. the `test` parameter is used to perform the Kolmogorov-Smirnov and chisquare
                test.
                ```python
                >>> parameters = gev_dist.fit_model(method="mle", test=True)
                -----KS Test--------
                Statistic = 0.06
                Accept Hypothesis
                P value = 0.9942356257694902
                >>> print(parameters) # doctest: +SKIP
                {'loc': -0.05962776672431072, 'scale': 0.9114319092295455, 'shape': 0.03492066094614391}

                ```
            - You can also use the `lmoments` method to estimate the distribution parameters.
                ```python
                >>> parameters = gev_dist.fit_model(method="lmoments", test=True)
                -----KS Test--------
                Statistic = 0.05
                Accept Hypothesis
                P value = 0.9996892272702655
                >>> print(parameters) # doctest: +SKIP
                {'loc': -0.07182150513604696, 'scale': 0.9153288314267931, 'shape': 0.018944589308927475}

                ```
            - You can also use the `fit_model` method to estimate the distribution parameters using the 'optimization'
                method. the optimization method requires the `obj_func` and `threshold` parameter. the method
                will take the `threshold` number and try to fit the data values that are greater than the threshold.
                ```python
                >>> threshold = np.quantile(data, 0.80)
                >>> print(threshold)
                1.39252

                ```
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))

        method = super().fit_model(method=method)  # type: ignore[assignment]
        if method == "mle" or method == "mm":
            param_list: Any = list(genextreme.fit(self.data, method=method))
        elif method == "lmoments":
            lm = Lmoments(self.data)
            lmu = lm.calculate()
            param_list = Lmoments.gev(lmu)
        elif method == "optimization":
            if obj_func is None or threshold is None:
                raise TypeError(OBJ_FUNCTION_THRESHOULD_ERROR)

            param_list = genextreme.fit(self.data, method="mle")
            # then we use the result as starting value for your truncated Gumbel fit
            param_list = so.fmin(
                obj_func,
                [threshold, param_list[0], param_list[1], param_list[2]],
                args=(self.data,),
                maxiter=500,
                maxfun=500,
            )
            param_list = [param_list[1], param_list[2], param_list[3]]
        else:
            raise ValueError(f"The given: {method} does not exist")

        param: dict[str, float] = {
            "loc": param_list[1],
            "scale": param_list[2],
            "shape": param_list[0],
        }
        self.parameters = param

        if test:
            self.ks()
            self.chisquare()

        return param

    def inverse_cdf(
        self,
        cdf: np.ndarray | list[float] | None = None,
        parameters: dict[str, float | Any] = None,
    ) -> np.ndarray:
        """Theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on a given non-exceedance probability

        Args:
            parameters: [list]
                location and scale parameters of the gumbel distribution.
            cdf: [list]
                cumulative distribution function/ Non-Exceedance probability.

        Returns:
            theoretical value: [numeric]
                Value based on the theoretical distribution

        Examples:
            - Instantiate the Gumbel class only with the data.
                ```python
                >>> data = np.loadtxt("examples/data/gev.txt")
                >>> parameters = {'loc': 0, 'scale': 1, "shape": 0.1}
                >>> gev_dist = GEV(data, parameters)

                ```
            - We will generate a random numbers between 0 and 1 and pass it to the inverse_cdf method as a probabilities
                to get the data that coresponds to these probabilities based on the distribution.
                ```python
                >>> cdf = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                >>> data_values = gev_dist.inverse_cdf(cdf)
                >>> print(data_values)
                [-0.86980039 -0.4873901   0.08704056  0.64966292  1.39286858  2.01513112]

                ```
        """
        if parameters is None:
            parameters = self.parameters

        if any(cdf) < 0 or any(cdf) > 1:  # type: ignore[arg-type]
            raise ValueError(CDF_INVALID_VALUE_ERROR)

        q_th = self._inv_cdf(cdf, parameters)  # type: ignore[arg-type]
        return q_th

    @staticmethod
    def _inv_cdf(cdf: np.ndarray | list[float], parameters: dict[str, float]):
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        if shape is None:
            raise ValueError("Shape parameter should not be None")

        # the main equation from scipy
        q_th = genextreme.ppf(cdf, shape, loc=loc, scale=scale)
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

    def confidence_interval(  # type: ignore[override]
        self,
        alpha: float = 0.1,
        plot_figure: bool = False,
        prob_non_exceed: np.ndarray = None,
        parameters: dict[str, float | Any] = None,
        state_function: Callable | None = None,
        n_samples: int = 100,
        method: str = "lmoments",
        **kwargs: Any,
    ) -> (
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Figure, Axes]
    ):  # pylint: disable=arguments-differ
        """confidence_interval.

        Args:
            parameters: dict[str, str], optional, default is None.
                if not provided, the parameters provided in the class initialization will be used.
                {"loc": val, "scale": val, "shape": value}

                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.
            prob_non_exceed : [list]
                Non-Exceedance probability
            alpha : [numeric]
                alpha or SignificanceLevel is a value of the confidence interval.
            state_function: callable, Default is GEV.ci_func
                function to calculate the confidence interval.
            n_samples: [int]
                number of samples generated by the bootstrap method Default is 100.
            method: [str]
                method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
                "lmoments".
            plot_figure: bool, optional, default is False.
                to plot the confidence interval.

        Returns:
            q_upper: [list]
                upper-bound coresponding to the confidence interval.
            q_lower: [list]
                lower-bound coresponding to the confidence interval.
            fig: matplotlib.figure.Figure
                Figure object.
            ax: matplotlib.axes.Axes
                Axes object.

        Examples:
            - Instantiate the GEV class with the data and the parameters.
                ```python
                >>> import matplotlib.pyplot as plt
                >>> data = np.loadtxt("examples/data/time_series1.txt")
                >>> parameters = {"loc": 16.3928, "scale": 0.70054, "shape": -0.1614793,}
                >>> gev_dist = GEV(data, parameters)

                ```
            - to calculate the confidence interval, we need to provide the confidence level (`alpha`).
                ```python
                >>> upper, lower = gev_dist.confidence_interval(alpha=0.1)

                ```
            - You can also plot confidence intervals
                ```python
                >>> upper, lower, fig, ax = gev_dist.confidence_interval(alpha=0.1, plot_figure=True, marker_size=10)

                ```
            ![gev-confidence-interval](./../_images/gev-confidence-interval.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        if prob_non_exceed is None:
            prob_non_exceed = PlottingPosition.weibul(self.data)
        else:
            # if the prob_non_exceed is given, check if the length is the same as the data
            if len(prob_non_exceed) != len(self.data):
                raise ValueError(PROB_NON_EXCEEDENCE_ERROR)
        if state_function is None:
            state_function = GEV.ci_func

        ci = ConfidenceInterval.boot_strap(
            self.data,
            state_function=state_function,
            gevfit=parameters,
            F=prob_non_exceed,
            alpha=alpha,
            n_samples=n_samples,
            method=method,
            **kwargs,
        )
        q_lower = ci["lb"]
        q_upper = ci["ub"]

        if plot_figure:
            qth = self._inv_cdf(prob_non_exceed, parameters)
            fig, ax = Plot.confidence_level(
                qth, self.data, q_lower, q_upper, alpha=alpha, **kwargs  # type: ignore[arg-type]
            )
            return q_upper, q_lower, fig, ax
        else:
            return q_upper, q_lower

    def plot(
        self,
        fig_size: tuple = (10, 5),
        xlabel: str = PDF_XAXIS_LABEL,
        ylabel: str = "cdf",
        fontsize: int = 15,
        cdf: np.ndarray | list | None = None,
        parameters: dict[str, float | Any] = None,
    ) -> tuple[Figure, tuple[Axes, Axes]]:
        """Probability Plot.

        Probability Plot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculates the confidence interval.

        Args:
            parameters (dict[str, str]):
                {"loc": val, "scale": val, shape: val}

                - loc (numeric):
                    Location parameter of the GEV distribution.
                - scale (numeric):
                    Scale parameter of the GEV distribution.
                - shape (float | int):
                    Shape parameter for the GEV distribution.
            cdf (list):
                Theoretical cdf calculated using weibul or using the distribution cdf function.
            fontsize (numeric):
                Font size of the axis labels and legend
            ylabel (str):
                y label string
            xlabel (str):
                X label string
            fig_size (int):
                size of the pdf and cdf figure

        Returns:
            Figure:
                matplotlib figure object
            tuple[Axes, Axes]:
                matplotlib plot axes

        Examples:
            - Instantiate the Gumbel class with the data and the parameters.
                ```python
                >>> import numpy as np
                >>> data = np.loadtxt("examples/data/time_series1.txt")
                >>> parameters = {"loc": 16.3928, "scale": 0.70054, "shape": -0.1614793,}
                >>> gev_dist = GEV(data, parameters)

                ```
            - to calculate the confidence interval, we need to provide the confidence level (`alpha`).
                ```python
                >>> fig, ax = gev_dist.plot()
                >>> print(fig)
                Figure(1000x500)
                >>> print(ax)
                (<Axes: xlabel='Actual data', ylabel='pdf'>, <Axes: xlabel='Actual data', ylabel='cdf'>)

                ```
            ![gev-plot](./../_images/gev-plot.png)
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
            # if the prob_non_exceed is given, check if the length is the same as the data
            if len(cdf) != len(self.data):
                raise ValueError(PROB_NON_EXCEEDENCE_ERROR)

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

        # The function to bootstrap

    @staticmethod
    def ci_func(data: list | np.ndarray, **kwargs: Any):
        """GEV distribution function.

        Parameters
        ----------
        data: [list, np.ndarray]
            time series
        kwargs (dict[str, Any]):
            gevfit: [list]
                GEV parameter [shape, location, scale]
            F: [list]
                Non-Exceedance probability
            method: [str]
                method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
                "lmoments".
        """
        gevfit = kwargs["gevfit"]
        prob_non_exceed = kwargs["F"]
        method = kwargs["method"]
        # generate theoretical estimates based on a random cdf, and the dist parameters
        sample = GEV._inv_cdf(np.random.rand(len(data)), gevfit)  # type: ignore[arg-type]

        # get parameters based on the new generated sample
        dist = GEV(sample)
        new_param = dist.fit_model(method=method, test=False)  # type: ignore[arg-type]

        # return period
        # T = np.arange(0.1, 999.1, 0.1) + 1
        # +1 in order not to make 1- 1/0.1 = -9
        # T = np.linspace(0.1, 999, len(data)) + 1
        # coresponding theoretical estimate to T
        # prob_non_exceed = 1 - 1 / T
        q_th = GEV._inv_cdf(prob_non_exceed, new_param)  # type: ignore[arg-type]

        res = list(new_param.values())
        res.extend(q_th)
        return tuple(res)


class Exponential(AbstractDistribution):
    """Exponential distribution.

    - The exponential distribution assumes that small values occur more frequently than large values.

    - The probability density function (PDF) of the Exponential distribution is:

        .. math::
            f(x; \\delta, \\beta) =
            \\begin{cases}
                f(x; \\delta, \\beta) = \\frac{1}{\\beta} e^{-\\frac{x - \\delta}{\\beta}} & \\quad x \\geq 0 \\\\
                0 & \\quad x < 0
            \\end{cases}
          :label: exp-equation

    - The probability density function above uses the location parameter :math:`\\delta` and the scale parameter
        :math:`\\beta` to define the distribution in a standardized form.
    - A common parameterization for the exponential distribution is in terms of the rate parameter :math:`\\lambda`,
        such that :math:`\\lambda = 1 / \\beta`.
    - The Location Parameter (:math:`\\delta`): This shifts the starting point of the distribution. The distribution is
        defined for :math:`x \\geq \\delta`.
    - Scale Parameter (:math:`\\beta`): This determines the spread of the distribution. The rate parameter
        :math:`\\lambda` is the inverse of the scale parameter, so :math:`\\lambda = \\frac{1}{\\beta}`.

    - The cumulative distribution functions.

        .. math::
            F(x; \\delta, \\beta) =
            \\begin{cases}
                F(x; \\delta, \\beta) = 1 - e^{-\\frac{x - \\delta}{\\beta}} & \\quad x \\geq 0 \\\\
                0 & \\quad x < 0
            \\end{cases}
          :label: exp-cdf

    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: dict[str, float] = None,
    ):
        """Exponential Distribution.

        Args:
            data (list):
                data time series.
            parameters (dict[str, float]):
                {"loc": val, "scale": val}

                - loc (numeric):
                    location parameter of the exponential distribution.
                - scale (numeric):
                    scale parameter of the exponential distribution.
        """
        super().__init__(data, parameters)

    @staticmethod
    def _pdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        pdf = expon.pdf(data, loc=loc, scale=scale)
        return pdf

    def pdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: dict[str, float] = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x.

        Args:
            parameters (dict[str, float], optional):
                if not provided, the parameters provided in the class initialization will be used.
                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.
                ```python
                {"loc": val, "scale": val}. default is None.

                ```
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
            >>> parameters = {'loc': 0, 'scale': 2}
            >>> expo_dist = Exponential(data, parameters)
            >>> _ = expo_dist.pdf(plot_figure=True)

            ```
            ![exponential-pdf](./../_images/distributions/exponential-pdf-2.png)
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
        """Generate Random Variable.

        Args:
            size (int):
                size of the random generated sample.
            parameters (dict[str, str]):
                - loc (numeric):
                    location parameter of the gumbel distribution.
                - scale (numeric):
                    scale parameter of the gumbel distribution.
                ```python
                {"loc": val, "scale": val}

                ```

        Returns:
            data (np.ndarray):
                random generated data.

        Examples:
            - To generate a random sample that follow the gumbel distribution with the parameters loc=0 and scale=1.
                ```python
                >>> from statista.distributions import Exponential
                >>> parameters = {'loc': 0, 'scale': 2}
                >>> expon_dist = Exponential(parameters=parameters)
                >>> random_data = expon_dist.random(1000)

                ```
            - then we can use the `pdf` method to plot the pdf of the random data.
                ```python
                >>> _ = expon_dist.pdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![exponential-pdf](./../_images/distributions/exponential-pdf.png)

                ```python
                >>> _ = expon_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![exponential-cdf](./../_images/distributions/exponential-cdf.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        random_data = expon.rvs(loc=loc, scale=scale, size=size)
        return random_data

    @staticmethod
    def _cdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
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
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        cdf = expon.cdf(data, loc=loc, scale=scale)
        return cdf

    def cdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: dict[str, float | Any] = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[np.ndarray, Figure, Any] | np.ndarray
    ):  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        Args:
            parameters (dict[str, str], optional):
                if not provided, the parameters provided in the class initialization will be used. default is None.
                - loc (numeric):
                    location parameter of the gumbel distribution.
                - scale (numeric):
                    scale parameter of the gumbel distribution.
                ```python
                {"loc": val, "scale": val}
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
            >>> parameters = {'loc': 0, 'scale': 2}
            >>> expo_dist = Exponential(data, parameters)
            >>> _ = expo_dist.cdf(plot_figure=True)

            ```
            ![gamma-pdf](./../_images/distributions/expo-random-cdf.png)
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
    ) -> dict[str, float]:
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
                Out[14]: {'loc': 0.0009, 'scale': 2.0498075}
                >>> print(parameters) # doctest: +SKIP
                {'loc': 0, 'scale': 2}

                ```
            - You can also use the `lmoments` method to estimate the distribution parameters.
                ```python
                >>> parameters = expo_dist.fit_model(method="lmoments", test=True) # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.021
                Accept Hypothesis
                P value = 0.9802627322900355
                >>> print(parameters) # doctest: +SKIP
                {'loc': -0.00805012182182141, 'scale': 2.0587576218218215}

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
                raise TypeError(OBJ_FUNCTION_THRESHOULD_ERROR)

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

        param: dict[str, float] = {"loc": param_list[0], "scale": param_list[1]}
        self.parameters = param

        if test:
            self.ks()
            self.chisquare()

        return param

    def inverse_cdf(
        self,
        cdf: np.ndarray | list[float] | None = None,
        parameters: dict[str, float | Any] = None,
    ) -> np.ndarray:
        """Theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on a given  non-exceedance probability

        Args:
            parameters (dict[str, str]):
                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.
                ```python
                {"loc": val, "scale": val}
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
                >>> parameters = {'loc': 0, 'scale': 2}
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

        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        if any(cdf) < 0 or any(cdf) > 1:  # type: ignore[arg-type]
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


class Normal(AbstractDistribution):
    """Normal Distribution.

    - The probability density function (PDF) of the Normal distribution is:

        .. math::
            f(x: threshold, scale) = (1/scale) e **(- (x-threshold)/scale)
          :label: normal-equation

    - The cumulative distribution functions.

        .. math::
            F(x: threshold, scale) = 1 - e **(- (x-threshold)/scale)
          :label: normal-cdf
    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: dict[str, float] = None,
    ):
        """Gumbel.

        Ars:
            data (list):
                data time series.
            parameters (dict[str, str]):
                - loc: [numeric]
                    location parameter of the exponential distribution.
                - scale: [numeric]
                    scale parameter of the exponential distribution.
                ```python
                {"loc": val, "scale": val}
                ```
        """
        super().__init__(data, parameters)

    @staticmethod
    def _pdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)
        pdf = norm.pdf(data, loc=loc, scale=scale)

        return pdf

    def pdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: dict[str, float] = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x.

        Args:
            parameters (dict[str, str], optional):
                if not provided, the parameters provided in the class initialization will be used. default is None.
                - loc: [numeric]
                    location parameter of the normal distribution.
                - scale: [numeric]
                    scale parameter of the normal distribution.
                ```python
                {"loc": val, "scale": val}
                ```
            data (np.ndarray):
                array if you want to calculate the pdf for different data than the time series given to the constructor
                method. default is None.
            plot_figure (bool):
                Default is False.
            kwargs (dict[str, Any]):
                fig_size: [tuple]
                    Default is (6, 5).
                xlabel: [str]
                    Default is "Actual data".
                ylabel: [str]
                    Default is "pdf".
                fontsize: [int]
                    Default is 15

        Returns:
            pdf (array):
                probability density function pdf.
            fig (matplotlib.figure.Figure):
                Figure object is returned only if `plot_figure` is True.
            ax (matplotlib.axes.Axes):
                Axes object is returned only if `plot_figure` is True.
        """
        result = super().pdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )  # type: ignore[misc]

        return result

    @staticmethod
    def _cdf_eq(
        data: list | np.ndarray, parameters: dict[str, float | Any]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)
        if loc is None or loc <= 0:
            raise ValueError("Threshold parameter should be greater than zero")

        cdf = norm.cdf(data, loc=loc, scale=scale)
        return cdf

    def cdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: dict[str, float | Any] = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """cdf.

        cdf calculates the value of Normal distribution cdf with parameters loc and scale at x.

        Args:
            parameters (dict[str, str], optional):
                if not provided, the parameters provided in the class initialization will be used. default is None.
                - loc (numeric):
                    location parameter of the Normal distribution.
                - scale (numeric):
                    scale parameter of the Normal distribution.
                ```python
                {"loc": val, "scale": val, "shape": value}
                ```
            data (np.ndarray):
                array if you want to calculate the pdf for different data than the time series given to the constructor
                method. default is None.
            plot_figure (bool):
                Default is False.
            kwargs (dict[str, Any]):
                fig_size (tuple):
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
    ) -> dict[str, float]:
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
            parameters (list):
                shape, loc, scale parameter of the gumbel distribution in that order.
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        method = super().fit_model(method=method)  # type: ignore[assignment]

        if method == "mle" or method == "mm":
            param_list: Any = list(norm.fit(self.data, method=method))
        elif method == "lmoments":
            lm = Lmoments(self.data)
            lmu = lm.calculate()
            param_list = Lmoments.normal(lmu)
        elif method == "optimization":
            if obj_func is None or threshold is None:
                raise TypeError(OBJ_FUNCTION_THRESHOULD_ERROR)

            param_list = norm.fit(self.data, method="mle")
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

        param: dict[str, float] = {"loc": param_list[0], "scale": param_list[1]}
        self.parameters = param

        if test:
            self.ks()
            self.chisquare()

        return param

    def inverse_cdf(
        self,
        cdf: np.ndarray | list[float] | None = None,
        parameters: dict[str, float | Any] = None,
    ) -> np.ndarray:
        """Theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on a given  non exceedence probability

        Args:
            parameters (dict[str, str]):
                {"loc": val, "scale": val}

                - loc (numeric):
                    location parameter of the Normal distribution.
                - scale (numeric):
                    scale parameter of the Normal distribution.
            cdf (list):
                cumulative distribution function/ Non-Exceedance probability.

        Returns:
            numeric:
                Value based on the theoretical distribution
        """
        if parameters is None:
            parameters = self.parameters

        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        if any(cdf) < 0 or any(cdf) > 1:  # type: ignore[arg-type]
            raise ValueError(CDF_INVALID_VALUE_ERROR)

        # the main equation from scipy
        q_th = norm.ppf(cdf, loc=loc, scale=scale)
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


class Distributions:
    """Distributions."""

    available_distributions = {
        "GEV": GEV,
        "Gumbel": Gumbel,
        "Exponential": Exponential,
        "Normal": Normal,
    }

    def __init__(
        self,
        distribution: str,
        data: list | np.ndarray | None = None,
        parameters: dict[str, Number] = None,
    ):
        if distribution not in self.available_distributions.keys():
            raise ValueError(f"{distribution} not supported")

        self.distribution = self.available_distributions[distribution](data, parameters)  # type: ignore[abstract, arg-type]

    def __getattr__(self, name: str):
        """Delegate method calls to the subclass"""
        # Retrieve the attribute or method from the distribution object
        try:
            # Retrieve the attribute or method from the subclasses
            attribute = getattr(self.distribution, name)

            # If the attribute is a method, return a callable function
            if callable(attribute):

                def method(*args, **kwargs):
                    """A callable function that simply calls the attribute if it is a method"""
                    return attribute(*args, **kwargs)

                return method

            # If it's a regular attribute, return its value
            return attribute

        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
