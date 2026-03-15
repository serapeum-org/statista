"""Plotting functions for statista package."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Plot:
    """Visualization utilities for statistical distributions and analyses.

    This class provides static methods for creating various types of statistical plots
    including probability density functions (PDF), cumulative distribution functions (CDF),
    detailed distribution plots, and confidence interval visualizations.

    All methods return matplotlib Figure and Axes objects, allowing for further customization
    if needed before saving or displaying the plots.

    Examples:
        - Generate some sample data:
            ```python
            >>> import numpy as np
            >>> from statista.plot import Plot
            >>> from statista.distributions import Normal
            >>> np.random.seed(42)
            >>> data = np.random.normal(loc=10, scale=2, size=100)

            ```
        - Fit a normal distribution:
            ```python
            >>> normal_dist = Normal(data)
            >>> normal_dist.fit_model() # doctest: +SKIP
            -----KS Test--------
            Statistic = 0.09
            Accept Hypothesis
            P value = 0.8154147124661313
            {'loc': np.float64(9.876997051725278), 'scale': np.float64(2.010896054339655)}

            ```
        - Generate points for plotting:
            ```python
            >>> x = np.linspace(min(data), max(data), 10000)
            >>> parameters = {'loc': 9.876997051725278, 'scale': 2.010896054339655}
            >>> pdf_values = normal_dist.pdf(data=x, parameters=parameters)

            ```
        - Create a PDF plot:
            ```python
            >>> fig, ax = Plot.pdf(x, pdf_values, data)

            ```
            ![PDF Plot Example](./../_images/plot/plot-pdf.png)
    """

    @staticmethod
    def pdf(
        qx: np.ndarray,
        pdf_fitted: np.ndarray | list,
        data_sorted: np.ndarray,
        fig_size: tuple = (6, 5),
        xlabel: str = "Actual data",
        ylabel: str = "pdf",
        fontsize: int = 11,
    ) -> tuple[Figure, Axes]:
        """Create a probability density function (PDF) plot.

        Generates a plot showing both the fitted probability density function curve
        and a histogram of the actual data for visual comparison.

        Args:
            qx (np.ndarray):
                Array of x-values for plotting the fitted PDF curve. Typically generated as a linspace between the
                min and max of the actual data.
            pdf_fitted (Union[np.ndarray, list]):
                Array of PDF values corresponding to each point in qx. Usually obtained from a distribution's pdf
                method.
            data_sorted (np.ndarray):
                The actual data to be plotted as a histogram.
            fig_size (Tuple):
                Figure size as (width, height) in inches. Defaults to (6, 5).
            xlabel (str):
                Label for the x-axis. Defaults to "Actual data".
            ylabel (str):
                Label for the y-axis. Defaults to "pdf".
            fontsize (int):
                Font size for labels. Defaults to 11.

        Returns:
            tuple: A tuple containing:
                - Figure: The matplotlib Figure object
                - Axes: The matplotlib Axes object containing the plot

        Examples:
            - Generate some sample data:
                ```python
                >>> import numpy as np
                >>> from statista.plot import Plot
                >>> from statista.distributions import Normal
                >>> data = np.random.normal(loc=10, scale=2, size=100)

                ```
            - Fit a normal distribution:
                ```python
                >>> normal_dist = Normal(data)
                >>> normal_dist.fit_model() # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.08
                Accept Hypothesis
                P value = 0.9084105017744525
                {'loc': np.float64(10.031759532159755), 'scale': np.float64(1.819201407871162)}

                ```
            - Generate points for plotting
                ```python
                >>> x = np.linspace(min(data), max(data), 1000)
                >>> parameters = {'loc': 10.031759532159755, 'scale': 1.819201407871162}
                >>> pdf_values = normal_dist.pdf(data=x, parameters=parameters)

                ```
            - Create a PDF plot:
                ```python
                >>> fig, ax = Plot.pdf(x, pdf_values, data)

                ```

            - Further customize the plot if needed
                ```python
                >>> ax.set_title("Normal Distribution PDF") # doctest: +SKIP
                >>> ax.grid(True) # doctest: +SKIP

                ```
                ![PDF Plot Example](./../_images/plot/plot-pdf-2.png)

        See Also:
            - Plot.cdf: For plotting cumulative distribution functions
            - Plot.details: For plotting both PDF and CDF together
        """
        fig = plt.figure(figsize=fig_size)
        # gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax = fig.add_subplot()
        ax.plot(qx, pdf_fitted, "-", color="#27408B", linewidth=2)
        ax.hist(
            data_sorted, density=True, histtype="stepfilled", color="#DC143C"
        )  # , alpha=0.2
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        plt.show()
        return fig, ax

    @staticmethod
    def cdf(
        qx: np.ndarray,
        cdf_fitted: np.ndarray,
        data_sorted: np.ndarray,
        cdf_weibul: np.ndarray,
        fig_size: tuple[float, float] = (6, 5),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 11,
    ) -> tuple[Figure, Axes]:
        """Create a cumulative distribution function (CDF) plot.

        Generates a plot showing both the fitted cumulative distribution function curve
        and the empirical CDF points from the actual data for visual comparison.

        Args:
            qx: Array of x-values for plotting the fitted CDF curve. Typically generated
                as a linspace between the min and max of the actual data.
            cdf_fitted: Array of CDF values corresponding to each point in qx.
                Usually obtained from a distribution's cdf method.
            data_sorted: The sorted actual data points.
            cdf_weibul: The empirical CDF values, typically calculated using the Weibull formula
                or another plotting position formula.
            fig_size: Figure size as (width, height) in inches. Defaults to (6, 5).
            xlabel: Label for the x-axis. Defaults to "Actual data".
            ylabel: Label for the y-axis. Defaults to "cdf".
            fontsize: Font size for labels and legend. Defaults to 11.

        Returns:
            tuple: A tuple containing:
                - Figure: The matplotlib Figure object
                - Axes: The matplotlib Axes object containing the plot

        Examples:
            - Generate some sample data:
                ```python
                >>> import numpy as np
                >>> from statista.plot import Plot
                >>> from statista.distributions import Normal
                >>> np.random.seed(42)
                >>> data = np.random.normal(loc=10, scale=2, size=100)
                >>> data_sorted = np.sort(data)

                ```
            - Calculate empirical CDF using Weibull formula:
                ```python
                >>> n = len(data_sorted)
                >>> cdf_empirical = np.arange(1, n + 1) / (n + 1)  # Weibull formula

                ```
            - Fit a normal distribution:
                ```python
                >>> normal_dist = Normal(data)
                >>> normal_dist.fit_model() # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.08
                Accept Hypothesis
                P value = 0.9084105017744525
                {'loc': np.float64(9.62108385209537), 'scale': np.float64(2.1593427284432147)}

                ```
            - Generate points for plotting:
                ```python
                >>> x = np.linspace(min(data), max(data), 1000)
                >>> parameters = {'loc': 9.62108385209537, 'scale': 2.1593427284432147}
                >>> cdf_values = normal_dist.cdf(data=x, parameters=parameters)

                ```
            - Create a CDF plot
                ```python
                >>> fig, ax = Plot.cdf(x, cdf_values, data_sorted, cdf_empirical)

                ```
            - Further customize the plot if needed
                ```python
                >>> ax.set_title("Normal Distribution CDF") # doctest: +SKIP
                >>> ax.grid(True)   # doctest: +SKIP

                ```
                ![CDF Plot Example](./../_images/plot/plot-cdf.png)

        See Also:
            - Plot.pdf: For plotting probability density functions
            - Plot.details: For plotting both PDF and CDF together
        """
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot()
        ax.plot(
            qx, cdf_fitted, "-", label="Estimated CDF", color="#27408B", linewidth=2
        )
        ax.scatter(
            data_sorted,
            cdf_weibul,
            label="Empirical CDF",
            color="orangered",
            facecolors="none",
        )
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize, framealpha=1)
        plt.show()
        return fig, ax

    @staticmethod
    def details(
        qx: np.ndarray | list,
        q_act: np.ndarray | list,
        pdf: np.ndarray | list,
        cdf_fitted: np.ndarray | list,
        cdf: np.ndarray | list,
        fig_size: tuple[float, float] = (10, 5),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 11,
    ) -> tuple[Figure, tuple[Axes, Axes]]:
        """Create a detailed distribution plot with both PDF and CDF.

        Generates a side-by-side plot showing both the probability density function (PDF)
        and cumulative distribution function (CDF) for a fitted distribution compared
        with the actual data. This provides a comprehensive view of how well the
        distribution fits the data.

        Args:
            qx: Array of x-values for plotting the fitted curves. Typically generated
                as a linspace between the min and max of the actual data.
            q_act: The actual data points.
            pdf: Array of PDF values corresponding to each point in qx.
                Usually obtained from a distribution's pdf method.
            cdf_fitted: Array of CDF values corresponding to each point in qx.
                Usually obtained from a distribution's cdf method.
            cdf: The empirical CDF values, typically calculated using the Weibull formula
                or another plotting position formula.
            fig_size: Figure size as (width, height) in inches. Defaults to (10, 5).
            xlabel: Label for the x-axis. Defaults to "Actual data".
            ylabel: Label for the y-axis of the CDF plot. Defaults to "cdf".
            fontsize: Font size for labels. Defaults to 11.

        Returns:
            tuple: A tuple containing:
                - Figure: The matplotlib Figure object
                - tuple: A tuple of two Axes objects (ax1, ax2) where:
                    - ax1: The left subplot containing the PDF
                    - ax2: The right subplot containing the CDF

        Examples:
            - Import necessary libraries:
                ```python
                >>> import numpy as np
                >>> from statista.plot import Plot
                >>> from statista.distributions import Normal

                ```
            - Generate some sample data:
                ```python
                >>> np.random.seed(42)
                >>> data = np.random.normal(loc=10, scale=2, size=100)
                >>> data_sorted = np.sort(data)

                ```
            - Calculate empirical CDF using Weibull formula:
                ```python
                >>> n = len(data_sorted)
                >>> cdf_empirical = np.arange(1, n + 1) / (n + 1)  # Weibull formula

                ```
            - Fit a normal distribution:
                ```python
                >>> normal_dist = Normal(data_sorted)
                >>> normal_dist.fit_model() # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.06
                Accept Hypothesis
                P value = 0.9942356257694902
                {'loc': np.float64(10.061702421737607), 'scale': np.float64(1.857026806934038)}

                ```
            - Generate points for plotting:
                ```python
                >>> x = np.linspace(min(data), max(data), 1000)
                >>> parameters = {'loc': 10.061702421737607, 'scale': 1.857026806934038}
                >>> pdf_values = normal_dist.pdf(data=x, parameters=parameters)
                >>> cdf_values = normal_dist.cdf(data=x, parameters=parameters)

                ```
            - Create a detailed plot with both PDF and CDF:
                ```python
                >>> fig, (ax1, ax2) = Plot.details(x, data, pdf_values, cdf_values, cdf_empirical)

                ```
            - Further customize the plots if needed:
                ```python
                >>> ax1.set_title("PDF Comparison") # doctest: +SKIP
                >>> ax2.set_title("CDF Comparison") # doctest: +SKIP
                >>> fig.suptitle("Normal Distribution Fit", fontsize=14) # doctest: +SKIP
                >>> ax1.grid(True) # doctest: +SKIP
                >>> ax2.grid(True) # doctest: +SKIP

                ```
                ![Details Plot Example](./../_images/plot/plot-detailed.png)
            ```

        See Also:
            - Plot.pdf: For plotting only the probability density function
            - Plot.cdf: For plotting only the cumulative distribution function
        """
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(qx, pdf, "-", color="#27408B", linewidth=2)
        ax1.hist(q_act, density=True, histtype="stepfilled", color="#DC143C")
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel("pdf", fontsize=fontsize)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(qx, cdf_fitted, "-", color="#27408B", linewidth=2)

        q_act = np.sort(q_act)
        ax2.scatter(q_act, cdf, color="#DC143C", facecolors="none")
        ax2.set_xlabel(xlabel, fontsize=fontsize)
        ax2.set_ylabel(ylabel, fontsize=15)
        plt.show()
        return fig, (ax1, ax2)

    @staticmethod
    def confidence_level(
        qth: np.ndarray | list,
        q_act: np.ndarray | list,
        q_lower: np.ndarray | list,
        q_upper: np.ndarray | list,
        fig_size: tuple[float, float] = (6, 6),
        fontsize: int = 11,
        alpha: float = 0.05,
        marker_size: int = 10,
    ) -> tuple[Figure, Axes]:
        """Create a confidence interval plot for distribution quantiles.

        Generates a plot showing the theoretical quantiles, actual data points, and
        confidence interval bounds. This is useful for assessing how well a distribution
        fits the data and visualizing the uncertainty in the fit.

        Args:
            qth: Theoretical quantiles (obtained using the inverse_cdf method).
                These values represent what the distribution predicts for each quantile.
            q_act: Actual data points, which will be sorted within the function.
                These are compared against the theoretical quantiles.
            q_lower: Lower limit of the confidence interval for each theoretical quantile.
                Usually calculated based on the distribution parameters and a significance level.
            q_upper: Upper limit of the confidence interval for each theoretical quantile.
                Usually calculated based on the distribution parameters and a significance level.
            fig_size: Figure size as (width, height) in inches. Defaults to (6, 6).
            fontsize: Font size for labels and legend. Defaults to 11.
            alpha: Significance level used for the confidence intervals (e.g., 0.05 for 95% CI).
                Used only for labeling the legend; the actual intervals must be pre-calculated.
            marker_size: Size of the markers for the upper and lower bounds. Defaults to 10.

        Returns:
            tuple: A tuple containing:
                - Figure: The matplotlib Figure object
                - Axes: The matplotlib Axes object containing the plot

        Examples:
            - Import necessary libraries:
                ```python
                >>> import numpy as np
                >>> from statista.plot import Plot
                >>> from statista.distributions import Normal

                ```
            - Generate some sample data:
                ```python
                >>> np.random.seed(42)
                >>> data = np.random.normal(loc=10, scale=2, size=100)

                ```
            - Fit a normal distribution:
                ```python
                >>> normal_dist = Normal(data)
                >>> normal_dist.fit_model() # doctest: +SKIP
                -----KS Test--------
                Statistic = 0.07
                Accept Hypothesis
                P value = 0.9684099261397212
                {'loc': np.float64(10.51674893337459), 'scale': np.float64(2.002961856532672)}

                ```
            - Generate theoretical quantiles:
                ```python
                >>> p = np.linspace(0.01, 0.99, 100)  # Probability points
                >>> parameters = {'loc': 10.51674893337459, 'scale': 2.002961856532672}
                >>> theoretical_quantiles = normal_dist.inverse_cdf(p, parameters=parameters)

                ```
            - Calculate confidence intervals (simplified example):
            - In practice, these would be calculated based on the distribution parameters
                ```python
                >>> std_error = 0.5  # Example standard error
                >>> z_value = 1.96  # For 95% confidence interval
                >>> lower_ci = theoretical_quantiles - z_value * std_error
                >>> upper_ci = theoretical_quantiles + z_value * std_error

                ```
            - Create the confidence interval plot:
                ```python
                >>> fig, ax = Plot.confidence_level(
                ...     theoretical_quantiles, data, lower_ci, upper_ci, alpha=0.05
                ... )

                ```
            - Further customize the plot if needed
                ```python
                >>> ax.set_title("Normal Distribution Quantile Plot with 95% CI") # doctest: +SKIP
                >>> ax.grid(True) # doctest: +SKIP

                ```
                ![Confidence Level Plot Example](./../_images/plot/plot-confidence-level.png)

        Notes:
            The function automatically sorts the actual data points (q_act) before plotting.

            The 1:1 line represents perfect agreement between theoretical and actual values.
            Points falling along this line indicate a good fit of the distribution to the data.

            Points falling outside the confidence intervals suggest potential issues with
            the distribution fit at those quantiles.

        See Also:
            - Plot.details: For plotting PDF and CDF together
        """
        q_act = np.sort(q_act)

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot()
        ax.plot(qth, qth, "-.", color="#3D59AB", linewidth=2, label="Theoretical Data")
        # confidence interval
        ax.plot(
            qth,
            q_lower,
            "*--",
            color="grey",
            markersize=marker_size,
            label=f"Lower limit ({int((1 - alpha) * 100)} % CI)",
        )
        ax.plot(
            qth,
            q_upper,
            "*--",
            color="grey",
            markersize=marker_size,
            label=f"Upper limit ({int((1 - alpha) * 100)} % CI)",
        )
        ax.scatter(
            qth,
            q_act,
            color="#DC143C",
            facecolors="none",
            label="Actual Data",
            zorder=10,
        )
        ax.legend(fontsize=fontsize, framealpha=1)
        ax.set_xlabel("Theoretical Values", fontsize=fontsize)
        ax.set_ylabel("Actual Values", fontsize=fontsize)
        plt.show()
        return fig, ax
