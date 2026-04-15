"""Base classes for statistical distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mode
from typing import Any, Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy.stats import chisquare, ks_2samp

from statista.distributions.parameters import Parameters
from statista.distributions.goodness_of_fit import GoodnessOfFitResult
from statista.exceptions import ParameterError
from statista.plot import Plot
from statista.utils import merge_small_bins

ninf = 1e-5

SCALE_PARAMETER_ERROR = "Scale parameter is negative"
CDF_INVALID_VALUE_ERROR = "cdf Value Invalid"
OBJ_FUNCTION_THRESHOLD_ERROR = "obj_func and threshold should be numeric value"
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
        _parameters (Parameters): Distribution parameters.
    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: dict[str, float] | Parameters | None = None,
    ):
        """Initialize the distribution with data or parameters.

        Args:
            data:
                Data time series as a list or numpy array.
            parameters:
                Distribution parameters as a ``Parameters`` instance or
                a dictionary with keys 'loc', 'scale', and optionally
                'shape'. Dicts are converted to ``Parameters``
                automatically.
                ```python
                Parameters(loc=0.0, scale=1.0)
                ```

        Raises:
            ValueError:
                If neither data nor parameters are provided.
            TypeError:
                If data is not a list or numpy array, or if parameters
                is not a dict or Parameters.
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

        self._parameters: Parameters | None
        if isinstance(parameters, Parameters) or parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            try:
                self._parameters = Parameters(**parameters)
            except TypeError as e:
                raise ParameterError(
                    "parameters dict must contain only 'loc',"
                    " 'scale', and optionally 'shape' keys:"
                    f" {e}"
                ) from e
        else:
            raise TypeError(
                "The `parameters` argument should be a Parameters"
                " instance or dictionary"
            )

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
    def parameters(self) -> Parameters | None:
        """Get the distribution parameters.

        Returns:
            Parameters instance (e.g., ``Parameters(loc=0.0, scale=1.0)``),
            or None if parameters have not been set or estimated yet.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: dict[str, float] | Parameters):
        """Set the distribution parameters.

        Args:
            value: Parameters instance or dictionary of distribution
                parameters. Dicts are converted to Parameters
                automatically.
        """
        if isinstance(value, dict):
            try:
                self._parameters = Parameters(**value)
            except TypeError as e:
                raise ParameterError(
                    "parameters dict must contain only 'loc',"
                    " 'scale', and optionally 'shape' keys:"
                    f" {e}"
                ) from e
        else:
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
    def _pdf_eq(data: list | np.ndarray, parameters: Parameters) -> np.ndarray:
        """Calculate the probability density function (PDF) values.

        This is an abstract method that must be implemented by subclasses.

        Args:
            data: Data points for which to calculate PDF values.
            parameters: Distribution parameters.
                Example: Parameters(loc=0.0, scale=1.0)

        Returns:
            Numpy array containing the PDF values for each data point.
        """
        pass

    @abstractmethod
    def pdf(
        self,
        parameters: Parameters | dict[str, float] | None = None,
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
            parameters: Distribution parameters.
                Example: Parameters(loc=0.0, scale=1.0)
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
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)

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
    def _cdf_eq(data: list | np.ndarray, parameters: Parameters) -> np.ndarray:
        """Calculate the cumulative distribution function (CDF) values.

        This is an abstract method that must be implemented by subclasses.

        Args:
            data: Data points for which to calculate CDF values.
            parameters: Distribution parameters.
                Example: Parameters(loc=0.0, scale=1.0)

        Returns:
            Numpy array containing the CDF values for each data point.
        """
        pass

    @abstractmethod
    def cdf(
        self,
        parameters: Parameters | dict[str, float] | None = None,
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
            parameters: Distribution parameters.
                Example: Parameters(loc=0.0, scale=1.0)
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
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)

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
            Parameters instance with estimated distribution parameters.
            Example: Parameters(loc=0.0, scale=1.0)

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
        parameters: Parameters,
    ) -> np.ndarray:
        """Calculate the inverse of the cumulative distribution function (quantile function).

        This method calculates the theoretical values corresponding to the given CDF values
        using the specified distribution parameters.

        Args:
            cdf: CDF values (non-exceedance probabilities) for which to calculate the quantiles.
                Values should be between 0 and 1.
            parameters: Distribution parameters.
                Example: Parameters(loc=0.0, scale=1.0)

        Returns:
            Numpy array containing the quantile values corresponding to the given CDF values.
        """
        pass

    @abstractmethod
    def ks(self) -> GoodnessOfFitResult:
        """Perform the Kolmogorov-Smirnov (KS) test for goodness of fit.

        This method tests whether the data follows the fitted distribution using
        the Kolmogorov-Smirnov test. The test compares the empirical CDF of the data
        with the theoretical CDF of the fitted distribution.

        Returns:
            GoodnessOfFitResult with:
            - ``statistic``: The maximum absolute difference between the empirical and
              theoretical CDFs. The smaller the D statistic, the more likely the data
              follows the distribution.
            - ``p_value``: The probability of observing a D statistic as extreme as the
              one calculated, assuming the null hypothesis is true (data follows the
              distribution). If ``p_value < alpha`` (typically 0.05), reject the null.
            - ``conclusion``: "Accept Hypothesis" or "Reject Hypothesis" based on the
              tabulated critical value.

            Tuple unpacking ``stat, p = dist.ks()`` continues to work for backward
            compatibility.

        Raises:
            ValueError: If the distribution parameters have not been estimated.
        """
        if self.parameters is None:
            raise ValueError(
                "The Value of parameters is unknown. Please use 'fit_model' to estimate the distribution parameters"
            )
        qth = self.inverse_cdf(self.cdf_weibul, self.parameters)

        test = ks_2samp(self.data, qth)

        accept = test.statistic < self.kstable
        conclusion = "Accept Hypothesis" if accept else "Reject Hypothesis"

        print("-----KS Test--------")
        print(f"Statistic = {test.statistic}")
        print(conclusion.replace("Reject", "reject"))
        print(f"P value = {test.pvalue}")

        return GoodnessOfFitResult(
            test_name="Kolmogorov-Smirnov",
            statistic=float(test.statistic),
            p_value=float(test.pvalue),
            conclusion=conclusion,
            details={"kstable": float(self.kstable)},
        )

    @abstractmethod
    def chisquare(self) -> GoodnessOfFitResult:
        """Perform the Chi-square test for goodness of fit.

        - `chisquare test` refers to Pearson's chi square goodness of fit test. It is designed for
        categorical/count data: you observe how many points fall into each bin and compare those counts with the
        frequencies expected under some hypothesis

        This method tests whether the data follows the fitted distribution using the Chi-square test.
        The test compares the observed frequencies (number of values in each category/histogram bin) with the
        expected frequencies under the fitted distribution.

        Returns:
            GoodnessOfFitResult with:
            - ``statistic``: The Chi-square statistic measuring the difference between
              observed and expected frequencies. For each bin ``i`` we compute the squared
              difference between the observed count ``O_i`` and the expected count ``E_i``,
              scaled by ``E_i``, and sum over all bins.
            - ``p_value``: The probability of observing a Chi-square statistic as extreme
              as the one calculated, assuming the null hypothesis is true (data follows
              the distribution). If ``p_value < alpha`` (typically 0.05), reject the null.
            - ``details``: ``{"ddof": int}`` — degrees of freedom used.

            Tuple unpacking ``stat, p = dist.chisquare()`` continues to work for backward
            compatibility.

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

        # Pearson's χ² test assumes each expected count is sufficiently large (at least about 5); otherwise the asymptotic χ² approximation is unreliable
        merged_obs, merged_exp = merge_small_bins(
            obs_counts.tolist(), expected_counts.tolist()  # type: ignore[arg-type]
        )

        ddof = len(self.parameters)
        test = chisquare(merged_obs, f_exp=merged_exp, ddof=ddof)
        return GoodnessOfFitResult(
            test_name="Chi-square",
            statistic=float(test.statistic),
            p_value=float(test.pvalue),
            details={"ddof": ddof},
        )

    def confidence_interval(  # type: ignore[empty-body]
        self,
        alpha: float = 0.1,
        plot_figure: bool = False,
        prob_non_exceed: np.ndarray = None,
        parameters: Parameters | dict[str, float] | None = None,
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
            parameters: Distribution parameters.
                Example: Parameters(loc=0.0, scale=1.0)
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
        parameters: Parameters | dict[str, float] | None = None,
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
            parameters: Distribution parameters.
                Example: Parameters(loc=0.0, scale=1.0)
                If None, uses the parameters provided during initialization.

        Returns:
            Tuple containing:
            - List of Figure objects
            - List of Axes objects
        """
        ...  # pragma: no cover
