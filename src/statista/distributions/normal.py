"""Normal distribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.optimize as so
from matplotlib.figure import Figure
from scipy.stats import norm

from statista.distributions.base import (
    CDF_INVALID_VALUE_ERROR,
    OBJ_FUNCTION_THRESHOLD_ERROR,
    SCALE_PARAMETER_ERROR,
    AbstractDistribution,
)
from statista.distributions.parameters import Parameters
from statista.parameters import Lmoments


class Normal(AbstractDistribution):
    """Normal Distribution.

    - The probability density function (PDF) of the Normal distribution is:

        $$
        f(x; \\mu, \\sigma) = \\frac{1}{\\sigma \\sqrt{2\\pi}}
        \\exp\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)
        $$

        Where \\(\\mu\\) is the location (mean) parameter and \\(\\sigma\\) is the scale
        (standard deviation) parameter.

    - The cumulative distribution function (CDF) is:

        $$
        F(x; \\mu, \\sigma) = \\frac{1}{2}\\left[1 + \\mathrm{erf}
        \\left(\\frac{x - \\mu}{\\sigma \\sqrt{2}}\\right)\\right]
        $$
    """

    def __init__(
        self,
        data: list | np.ndarray | None = None,
        parameters: Parameters | None = None,
    ):
        """Normal.

        Args:
            data (list):
                data time series.
            parameters (Parameters):
                - loc: [numeric]
                    location (mean) parameter of the Normal distribution.
                - scale: [numeric]
                    scale (standard deviation) parameter of the Normal distribution.
                ```python
                Parameters(loc=val, scale=val)
                ```
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
        pdf = norm.pdf(data, loc=loc, scale=scale)

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
                if not provided, the parameters provided in the class initialization will be used. default is None.
                - loc: [numeric]
                    location parameter of the normal distribution.
                - scale: [numeric]
                    scale parameter of the normal distribution.
                ```python
                Parameters(loc=val, scale=val)
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
        data: list | np.ndarray, parameters: Parameters
    ) -> np.ndarray:
        loc = parameters.loc
        scale = parameters.scale

        if scale is None or scale <= 0:
            raise ValueError(SCALE_PARAMETER_ERROR)

        cdf = norm.cdf(data, loc=loc, scale=scale)
        return cdf

    def cdf(  # type: ignore[override]
        self,
        plot_figure: bool = False,
        parameters: Parameters | None = None,
        data: list[float] | np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, Figure, Any] | np.ndarray:
        """cdf.

        cdf calculates the value of Normal distribution cdf with parameters loc and scale at x.

        Args:
            parameters (Parameters, optional):
                if not provided, the parameters provided in the class initialization will be used. default is None.
                - loc (numeric):
                    location parameter of the Normal distribution.
                - scale (numeric):
                    scale parameter of the Normal distribution.
                ```python
                Parameters(loc=val, scale=val)
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
                raise TypeError(OBJ_FUNCTION_THRESHOLD_ERROR)

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

        Theoretical Estimate method calculates the theoretical values based on a given  non exceedence probability

        Args:
            parameters (Parameters):
                Parameters(loc=val, scale=val)

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
