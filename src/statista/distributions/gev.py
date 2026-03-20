"""Generalized Extreme Value (GEV) distribution."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.optimize as so
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import genextreme

from statista.confidence_interval import ConfidenceInterval
from statista.distributions.base import (
    CDF_INVALID_VALUE_ERROR,
    OBJ_FUNCTION_THRESHOLD_ERROR,
    PDF_XAXIS_LABEL,
    PROB_NON_EXCEEDENCE_ERROR,
    SCALE_PARAMETER_ERROR,
    AbstractDistribution,
    PlottingPosition,
)
from statista.distributions.parameters import Parameters
from statista.parameters import Lmoments
from statista.plot import Plot


class GEV(AbstractDistribution):
    r"""GEV (Generalized Extreme value statistics)

    - The Generalized Extreme Value (GEV) distribution is used to model the largest or smallest value among a large
        set of independent, identically distributed random values.
    - The GEV distribution encompasses three types of distributions: Gumbel, Fréchet, and Weibull, which are
        distinguished by a shape parameter (\(\\xi\) (xi)).

    - The probability density function (PDF) of the Generalized-extreme-value distribution is:

        $$
        f(x; \\zeta, \\delta, \\xi)=\\frac{1}{\\delta}\\mathrm{*}{\\mathrm{Q(x)}}^{\\xi+1}\\mathrm{
        *} e^{\\mathrm{-Q(x)}}
        $$

        $$
        Q(x; \\zeta, \\delta, \\xi)=
        \\begin{cases}
            \\left(1+ \\xi \\left(\\frac{x-\\zeta}{\\delta} \\right) \\right)^\\frac{-1}{\\xi} &
            \\quad\\land\\xi\\neq 0 \\\\
            e^{- \\left(\\frac{x-\\zeta}{\\delta} \\right)} & \\quad \\land \\xi=0
        \\end{cases}
        $$

        Where the \(\\delta\) (delta) is the scale parameter, \(\\zeta\) (zeta) is the location parameter,
        and \(\\xi\) (xi) is the shape parameter.

    - The location parameter \(\\zeta\) shifts the distribution along the x-axis. It essentially determines the mode
        (peak) of the distribution and its location. Changing the location parameter moves the distribution left or
        right without altering its shape. The location parameter ranges from negative infinity to positive infinity.
    - The scale parameter \(\\delta\) controls the spread or dispersion of the distribution. A larger scale parameter
        results in a wider distribution, while a smaller scale parameter results in a narrower distribution. It must
        always be positive.
    - The shape parameter \(\\xi\) (xi) determines the shape of the distribution. The shape parameter can be positive,
        negative, or zero. The shape parameter is used to classify the GEV distribution into three types: \(\\xi = 0\)
        Gumbel (Type I), \(\\xi > 0\) Fréchet (Type II), and \(\\xi < 0\) Weibull (Type III). The shape
        parameter determines the tail behavior of the distribution.

        In hydrology, the distribution is reparametrized with \(k=-\\xi\) (xi) (El Adlouni et al., 2008).

    - The cumulative distribution function (CDF) is:

        $$
        F(x; \\zeta, \\delta, \\xi)=
        \\begin{cases}
            \\exp\\left(- \\left(1+ \\xi \\left(\\frac{x-\\zeta}{\\delta} \\right) \\right)^\\frac{-1}{\\xi} \\right) &
            \\quad\\land\\xi\\neq 0 \\land 1 + \\xi \\left( \\frac{x-\\zeta}{\\delta}\\right) > 0 \\\\
            \\exp\\left(- \\exp\\left(- \\frac{x-\\zeta}{\\delta} \\right) \\right) & \\quad \\land \\xi=0
        \\end{cases}
        $$

    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: Parameters | dict[str, float] | None = None,
    ):
        """GEV.

        Args:
            data: [list]
                data time series.
            parameters: Parameters
                Distribution parameters instance.

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
            >>> parameters = Parameters(loc=0, scale=1, shape=0.1)
            >>> gev_dist = GEV(data, parameters)
            >>> print(gev_dist) # doctest: +SKIP
            <statista.distributions.Gumbel object at 0x000001CDDEB32C00>
            ```
        """
        super().__init__(data, parameters)

    @staticmethod
    def _pdf_eq(data: list | np.ndarray, parameters: Parameters) -> np.ndarray:
        loc = parameters.loc
        scale = parameters.scale
        shape = parameters.shape

        pdf = genextreme.pdf(data, loc=loc, scale=scale, c=shape)
        return pdf

    def pdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: Parameters | dict[str, float] | None = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """pdf.

        Returns the value of GEV's pdf with parameters loc and scale at x.

        Args:
            parameters: Parameters, optional, default is None.
                if not provided, the parameters provided in the class initialization will be used.

                - loc: [numeric]
                    location parameter of the GEV distribution.
                - scale: [numeric]
                    scale parameter of the GEV distribution.
                - shape: [numeric]
                    shape parameter of the GEV distribution.
            data: np.ndarray, default is None.
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
            >>> parameters = Parameters(loc=0, scale=1, shape=0.1)
            >>> gev_dist = GEV(data, parameters)
            >>> _ = gev_dist.pdf(plot_figure=True)

            ```
            ![gev-random-pdf](./../../_images/gev-random-pdf.png)
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
        parameters: Parameters | dict[str, float] | None = None,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """Generate Random Variable.

        Args:
            size: int
                size of the random generated sample.
            parameters: Parameters
                Distribution parameters instance.

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
                >>> parameters = Parameters(loc=0, scale=1, shape=0.1)
                >>> gev_dist = GEV(parameters=parameters)
                >>> random_data = gev_dist.random(100)

                ```
            - then we can use the `pdf` method to plot the pdf of the random data.
                ```python
                >>> _ = gev_dist.pdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![gev-random-pdf](./../../_images/gev-random-pdf.png)
                ```
                >>> _ = gev_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

                ```
                ![gev-random-cdf](./../../_images/gev-random-cdf.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)

        loc = parameters.loc
        scale = parameters.scale
        shape = parameters.shape

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        random_data = genextreme.rvs(loc=loc, scale=scale, c=shape, size=size)
        return random_data

    @staticmethod
    def _cdf_eq(data: list | np.ndarray, parameters: Parameters) -> np.ndarray:
        loc = parameters.loc
        scale = parameters.scale
        shape = parameters.shape
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
        parameters: Parameters | dict[str, float] | None = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[np.ndarray, Figure, Axes] | np.ndarray
    ):  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        Args:
            parameters: Parameters, optional, default is None.
                if not provided, the parameters provided in the class initialization will be used.

                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.
            data: np.ndarray, default is None.
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
                >>> parameters = Parameters(loc=0, scale=1, shape=0.1)
                >>> gev_dist = GEV(data, parameters)
                >>> _ = gev_dist.cdf(plot_figure=True)

                ```
            ![gev-random-cdf](./../../_images/gev-random-cdf.png)
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
        data: np.ndarray | None = None,
        parameters: Parameters | dict[str, float] | None = None,
    ) -> np.ndarray:
        """return_period.

            calculate return period calculates the return period for a list/array of values or a single value.

        Args:
            data (list/array/float):
                value you want the coresponding return value for
            parameters (Parameters):
                Distribution parameters instance.

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
        if data is None:
            data = self.data

        if parameters is None:
            parameters = self.parameters
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)

        cdf: Any = self.cdf(parameters=parameters, data=data)

        rp = 1 / (1 - cdf)

        return rp

    def fit_model(
        self,
        method: str = "mle",
        obj_func=None,
        threshold: int | float | None = None,
        test: bool = True,
    ) -> Parameters:
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
            obj_func (Callable | None):
                function to be used to get the distribution parameters.
            threshold (int | float | None):
                Value you want to consider only the greater values.
            method (str):
                'mle', 'mm', 'lmoments', optimization
            test (bool):
                Default is True

        Returns:
            Parameters:
                Distribution parameters instance.

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
                Parameters(loc=-0.05962776672431072, scale=0.9114319092295455, shape=0.03492066094614391)

                ```
            - You can also use the `lmoments` method to estimate the distribution parameters.
                ```python
                >>> parameters = gev_dist.fit_model(method="lmoments", test=True)
                -----KS Test--------
                Statistic = 0.05
                Accept Hypothesis
                P value = 0.9996892272702655
                >>> print(parameters) # doctest: +SKIP
                Parameters(loc=-0.07182150513604696, scale=0.9153288314267931, shape=0.018944589308927475)

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
                raise TypeError(OBJ_FUNCTION_THRESHOLD_ERROR)

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

        param = Parameters(loc=param_list[1], scale=param_list[2], shape=param_list[0])
        self.parameters = param

        if test:
            self.ks()
            self.chisquare()

        return param

    def inverse_cdf(
        self,
        cdf: np.ndarray | list[float] | None = None,
        parameters: Parameters | dict[str, float] | None = None,
    ) -> np.ndarray:
        """Theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on a given non-exceedance probability

        Args:
            parameters: Parameters
                Distribution parameters instance.
            cdf: [list]
                cumulative distribution function/ Non-Exceedance probability.

        Returns:
            theoretical value: [numeric]
                Value based on the theoretical distribution

        Examples:
            - Instantiate the Gumbel class only with the data.
                ```python
                >>> data = np.loadtxt("examples/data/gev.txt")
                >>> parameters = Parameters(loc=0, scale=1, shape=0.1)
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
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)

        cdf = np.array(cdf)
        if np.any(cdf < 0) or np.any(cdf > 1):
            raise ValueError(CDF_INVALID_VALUE_ERROR)

        q_th = self._inv_cdf(cdf, parameters)  # type: ignore[arg-type]
        return q_th

    @staticmethod
    def _inv_cdf(cdf: np.ndarray | list[float], parameters: Parameters):
        loc = parameters.loc
        scale = parameters.scale
        shape = parameters.shape

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
        parameters: Parameters | dict[str, float] | None = None,
        state_function: Callable | None = None,
        n_samples: int = 100,
        method: str = "lmoments",
        **kwargs: Any,
    ) -> (
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Figure, Axes]
    ):  # pylint: disable=arguments-differ
        """confidence_interval.

        Args:
            parameters: Parameters, optional, default is None.
                if not provided, the parameters provided in the class initialization will be used.

                - loc: [numeric]
                    location parameter of the gumbel distribution.
                - scale: [numeric]
                    scale parameter of the gumbel distribution.
            prob_non_exceed: [list]
                Non-Exceedance probability
            alpha: [numeric]
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
                >>> parameters = Parameters(loc=16.3928, scale=0.70054, shape=-0.1614793)
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
            ![gev-confidence-interval](./../../_images/gev-confidence-interval.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)

        scale = parameters.scale
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
        parameters: Parameters | dict[str, float] | None = None,
    ) -> tuple[Figure, tuple[Axes, Axes]]:
        """Probability Plot.

        Probability Plot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculates the confidence interval.

        Args:
            parameters (Parameters):
                Distribution parameters instance.

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
                >>> parameters = Parameters(loc=16.3928, scale=0.70054, shape=-0.1614793)
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
            ![gev-plot](./../../_images/gev-plot.png)
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters
        elif isinstance(parameters, dict):
            parameters = Parameters(**parameters)
        scale = parameters.scale

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
            gevfit: Parameters
                GEV distribution parameters instance.
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
