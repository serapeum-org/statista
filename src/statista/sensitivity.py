"""Sensitivity Analysis module for evaluating parameter influence on model outputs.

This module provides tools for conducting sensitivity analysis on model parameters.
It includes methods for One-At-a-Time (OAT) sensitivity analysis and visualization
of sensitivity results using Sobol plots.

The module is designed to help users understand how changes in input parameters
affect model outputs, which is crucial for model calibration, uncertainty analysis,
and decision-making processes.
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


class Sensitivity:
    """A class for performing sensitivity analysis on model parameters.

    This class provides methods for conducting sensitivity analysis to evaluate how changes
    in model parameters affect model outputs. It supports One-At-a-Time (OAT) sensitivity
    analysis and visualization of results through Sobol plots.

    Attributes:
        parameter (DataFrame): DataFrame containing parameter values with parameter names as index.
        lower_bound (List[Union[int, float]]): Lower bounds for each parameter.
        upper_bound (List[Union[int, float]]): Upper bounds for each parameter.
        function (callable): The model function to evaluate.
        NoValues (int): Number of parameter values to test between bounds.
        return_values (int): Specifies return type (1 for single value, 2 for value and calculations).
        num_parameters (int): Number of parameters to analyze.
        positions (List[int]): Positions of parameters to analyze.
        sen (dict): Dictionary storing sensitivity analysis results.
        MarkerStyleList (List[str]): List of marker styles for plotting.

    Examples:
        - Import necessary libraries:
            ```python
            >>> import pandas as pd
            >>> import numpy as np
            >>> from statista.sensitivity import Sensitivity

            ```
        - Define a simple model function:
            ```python
            >>> def model_function(params, *args, **kwargs):
            ...     # A simple quadratic function
            ...     return params[0]**2 + params[1]

            ```
        - Create parameter DataFrame:
            ```python
            >>> parameters = pd.DataFrame({'value': [2.0, 3.0]}, index=['param1', 'param2'])

            ```
        - Define parameter bounds:
            ```python
            >>> lower_bounds = [0.5, 1.0]
            >>> upper_bounds = [4.0, 5.0]

            ```
        - Create sensitivity analysis object:
            ```python
            >>> sensitivity = Sensitivity(parameters, lower_bounds, upper_bounds, model_function)

            ```
        - Perform one-at-a-time sensitivity analysis:
            ```python
            >>> sensitivity.one_at_a_time()
            0-param1 -0
            3.25
            0-param1 -1
            4.891
            0-param1 -2
            7.0
            ...
            1-param2 -3
            7.0
            1-param2 -4
            8.0
            1-param2 -5
            9.0

            ```
        - Plot results:
            ```python
            >>> fig, ax = sensitivity.sobol(
            ...     title="Parameter Sensitivity",
            ...     xlabel="Relative Parameter Value",
            ...     ylabel="Model Output"
            ... )

            ```
            ![one-at-a-time](./../_images/sensitivity/one-at-a-time.png)
    """

    MarkerStyleList = [
        "--o",
        ":D",
        "-.H",
        "--x",
        ":v",
        "--|",
        "-+",
        "-^",
        "--s",
        "-.*",
        "-.h",
    ]

    def __init__(
        self,
        parameter: DataFrame,
        lower_bound: list[int | float],
        upper_bound: list[int | float],
        function: Callable[..., Any],
        positions=None,
        n_values=5,
        return_values=1,
    ):
        """Initialize the Sensitivity analysis object.

        This constructor sets up the sensitivity analysis by defining the parameters to analyze,
        their bounds, the model function to evaluate, and configuration options for the analysis.

        Args:
            parameter (DataFrame): DataFrame with parameter names as index and a column named 'value'
                containing the parameter values.
            lower_bound (List[Union[int, float]]): List of lower bounds for each parameter.
            upper_bound (List[Union[int, float]]): List of upper bounds for each parameter.
            function (callable): The model function to evaluate. Should accept a list of parameter
                values as its first argument, followed by any additional args and kwargs.
            positions (List[int], optional): Positions of parameters to analyze (0-indexed).
                If None, all parameters will be analyzed. Defaults to None.
            n_values (int, optional): Number of parameter values to test between bounds.
                The parameter's current value will be included in addition to these points.
                Defaults to 5.
            return_values (int, optional): Specifies the return type of the function:
                - 1: Function returns a single metric value
                - 2: Function returns a tuple of (metric, calculated_values)
                Defaults to 1.

        Raises:
            AssertionError: If the lengths of parameter, lower_bound, and upper_bound don't match.
            AssertionError: If the provided function is not callable.

        Examples:
            - Import necessary libraries:
                ```python
                >>> import pandas as pd
                >>> from statista.sensitivity import Sensitivity

                ```
            - Define a simple model function:
                ```python
                >>> def model_function(params):
                ...     return params[0] + 2 * params[1]

                ```
            - Create parameter DataFrame:
                ```python
                >>> parameters = pd.DataFrame({'value': [1.0, 2.0]}, index=['x', 'y'])

                ```
            - Define parameter bounds:
                ```python
                >>> lower_bounds = [0.1, 0.5]
                >>> upper_bounds = [2.0, 3.0]

                ```
            - Create sensitivity analysis object for all parameters:
                ```python
                >>> sensitivity_all = Sensitivity(parameters, lower_bounds, upper_bounds, model_function)

                ```
            - Create sensitivity analysis object for specific parameters:
                ```python
                >>> sensitivity_specific = Sensitivity(
                ...     parameters, lower_bounds, upper_bounds, model_function, positions=[1]
                ... )

                ```
        """
        self.parameter = parameter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        assert (
            len(self.parameter) == len(self.lower_bound) == len(self.upper_bound)
        ), "The Length of the boundary should be of the same length as the length of the parameters"
        assert callable(
            function
        ), "function should be of type-callable (function that takes arguments)"
        self.function = function

        self.NoValues = n_values
        self.return_values = return_values
        # if the Position argument is empty list, the sensitivity will be done for all parameters
        if positions is None:
            self.num_parameters = len(parameter)
            self.positions = list(range(len(parameter)))
        else:
            self.num_parameters = len(positions)
            self.positions = positions

    @staticmethod
    def marker_style(style: int) -> str:
        """Get a marker style for plotting sensitivity analysis results.

        This static method returns a marker style string from a predefined list of styles.
        If the style index exceeds the list length, it wraps around using modulo operation.

        Args:
            style (int):
                Index of the marker style to retrieve.

        Returns:
            str:
                A matplotlib-compatible marker style string (e.g., "--o", ":D").

        Examples:
            - Import necessary libraries:
                ```python
                >>> from statista.sensitivity import Sensitivity

                ```
            - Get the first marker style:
                ```python
                >>> style1 = Sensitivity.marker_style(0)
                >>> print(style1)
                --o

                ```
            - Get the second marker style:
                ```python
                >>> style2 = Sensitivity.marker_style(1)
                >>> print(style2)
                :D

                ```
            - Demonstrate wrapping behavior:
                ```python
                >>> style_wrapped = Sensitivity.marker_style(len(Sensitivity.MarkerStyleList) + 2)
                >>> print(style_wrapped == Sensitivity.marker_style(2))
                True

                ```
        """
        if style > len(Sensitivity.MarkerStyleList) - 1:
            style %= len(Sensitivity.MarkerStyleList)
        return Sensitivity.MarkerStyleList[style]

    def one_at_a_time(self, *args, **kwargs: dict[str, Any]):
        """Perform One-At-a-Time (OAT) sensitivity analysis.

        This method performs OAT sensitivity analysis by varying each parameter one at a time
        while keeping others constant. For each parameter, it generates a range of values
        between the lower and upper bounds, evaluates the model function for each value,
        and stores the results.

        The results are stored in the `sen` attribute, which is a dictionary with parameter
        names as keys. Each value is a list containing:
        1. Relative parameter values (ratio to original value)
        2. Corresponding metric values from the model function
        3. Actual parameter values used
        4. Additional calculated values (if return_values=2)

        Args:
            *args (Any): Variable length argument list passed to the model function.
            **kwargs (dict[str, Any]): Arbitrary keyword arguments passed to the model function.

        Raises:
            ValueError: If the function returns more than one value when return_values=1,
                or doesn't return the expected format when return_values=2.

        Examples:
            - Import libraries:
                ```python
                >>> import pandas as pd
                >>> import numpy as np
                >>> from statista.sensitivity import Sensitivity

                ```
            - Define a model function:
                ```python
                >>> def model_function(params, multiplier=1):
                ...     return multiplier * (params[0]**2 + params[1])

                ```
            - Create parameter DataFrame:
                ```python
                >>> parameters = pd.DataFrame({'value': [2.0, 3.0]}, index=['param1', 'param2'])

                ```
            - Define parameter bounds:
                ```python
                >>> lower_bounds = [0.5, 1.0]
                >>> upper_bounds = [4.0, 5.0]

                ```
            - Create sensitivity analysis object:
                ```python
                >>> sensitivity = Sensitivity(parameters, lower_bounds, upper_bounds, model_function)

                ```
            - Perform OAT sensitivity analysis with additional argument:
                ```python
                >>> sensitivity.one_at_a_time(multiplier=2)
                0-param1 -0
                6.5
                0-param1 -1
                9.781
                0-param1 -2
                14.0
                0-param1 -3
                16.125
                0-param1 -4
                ...
                1-param2 -2
                14.0
                1-param2 -3
                14.0
                1-param2 -4
                16.0
                1-param2 -5
                18.0

                ```
            - Access results for the first parameter:
                ```python
                >>> param_name = parameters.index[0]
                >>> relative_values = sensitivity.sen[param_name][0]
                >>> metric_values = sensitivity.sen[param_name][1]
                >>> actual_values = sensitivity.sen[param_name][2]

                ```
            - Print a sample result:
                ```python
                >>> print(f"When {param_name} = {actual_values[0]}, metric = {metric_values[0]}")
                When param1 = 0.5, metric = 6.5

                ```
        """
        self.sen: dict[str, list[Any]] = {}

        for i in range(self.num_parameters):
            k = self.positions[i]
            if self.return_values == 1:
                self.sen[self.parameter.index[k]] = [[], [], []]
            else:
                self.sen[self.parameter.index[k]] = [[], [], [], []]
            # generate 5 random values between the high and low parameter bounds
            rand_value = np.linspace(
                self.lower_bound[k], self.upper_bound[k], self.NoValues
            )
            # add the value of the calibrated parameter and sort the values
            rand_value = np.sort(np.append(rand_value, self.parameter["value"].iloc[k]))
            # store the relative values of the parameters in the first list in the dict
            self.sen[self.parameter.index[k]][0] = [
                (h / self.parameter["value"].iloc[k]) for h in rand_value
            ]

            random_param = self.parameter["value"].tolist()
            for j in range(len(rand_value)):
                random_param[k] = rand_value[j]

                if self.return_values == 1:
                    metric = self.function(random_param, *args, **kwargs)
                else:
                    metric, calculated_values = self.function(
                        random_param, *args, **kwargs
                    )
                    self.sen[self.parameter.index[k]][3].append(calculated_values)
                try:
                    # store the metric value in the second list in the dict
                    self.sen[self.parameter.index[k]][1].append(round(metric, 3))
                except TypeError:
                    message = """the Given function returns more than one value,
                    the function should return only one value for return_values=1, or
                    two values for return_values=2.
                    """
                    raise ValueError(message)
                # store the real values of the parameter in the third list in the dict
                self.sen[self.parameter.index[k]][2].append(round(rand_value[j], 4))
                print(str(k) + "-" + self.parameter.index[k] + " -" + str(j))
                print(round(metric, 3))

    def sobol(
        self,
        real_values: bool = False,
        title: str = "",
        xlabel: str = "xlabel",
        ylabel: str = "Metric values",
        labelfontsize=12,
        plotting_from="",
        plotting_to="",
        title2: str = "",
        xlabel2: str = "xlabel2",
        ylabel2: str = "ylabel2",
        spaces=None,
    ) -> tuple:
        """Plot sensitivity analysis results using Sobol-style visualization.

        This method creates plots to visualize the results of sensitivity analysis.
        It can generate either a single plot (when return_values=1) or two plots
        (when return_values=2) to show both the metric values and additional calculated values.

        Args:
            real_values (bool, optional):
                If True, plots actual parameter values on x-axis instead of relative values. Works best when
                analyzing a single parameter since parameter ranges may differ. Defaults to False.
            title (str, optional):
                Title for the main plot. Defaults to "".
            xlabel (str, optional):
                X-axis label for the main plot. Defaults to "xlabel".
            ylabel (str, optional):
                Y-axis label for the main plot. Defaults to "Metric values".
            labelfontsize (int, optional):
                Font size for axis labels. Defaults to 12.
            plotting_from (str or int, optional):
                Starting index for plotting calculated values in the second plot. Defaults to "" (start from beginning).
            plotting_to (str or int, optional): Ending index for plotting calculated values
                in the second plot. Defaults to "" (plot until end).
            title2 (str, optional): Title for the second plot (when return_values=2).
                Defaults to "".
            xlabel2 (str, optional):
                X-axis label for the second plot. Defaults to "xlabel2".
            ylabel2 (str, optional):
                Y-axis label for the second plot. Defaults to "ylabel2".
            spaces (List[float], optional):
                Spacing parameters for subplot adjustment [left, bottom, right, top, wspace, hspace]. Defaults to None.

        Returns:
            tuple: When return_values=1, returns (fig, ax) where fig is the matplotlib figure
                and ax is the axis. When return_values=2, returns (fig, (ax1, ax2)) where
                ax1 is the main plot axis and ax2 is the calculated values plot axis.

        Raises:
            ValueError:
                If attempting to plot calculated values when return_values is not 2.

        Examples:
            - Import necessary libraries:
                ```python
                >>> import pandas as pd
                >>> import numpy as np
                >>> from statista.sensitivity import Sensitivity
                >>> import matplotlib.pyplot as plt

                ```
            - Define a model function:
                ```python
                >>> def model_function(params):
                ...     return params[0]**2 + params[1]

                ```
            - Create parameter DataFrame:
                ```python
                >>> parameters = pd.DataFrame({'value': [2.0, 3.0]}, index=['param1', 'param2'])

                ```
            - Define parameter bounds:
                ```python
                >>> lower_bounds = [0.5, 1.0]
                >>> upper_bounds = [4.0, 5.0]

                ```
            - Create sensitivity analysis object:
                ```python
                >>> sensitivity = Sensitivity(parameters, lower_bounds, upper_bounds, model_function)

                ```
            - Perform OAT sensitivity analysis:
                ```python
                >>> sensitivity.one_at_a_time()
                0-param1 -0
                3.25
                0-param1 -1
                4.891
                0-param1 -2
                7.0
                0-param1 -3
                ...
                1-param2 -2
                7.0
                1-param2 -3
                7.0
                1-param2 -4
                8.0
                1-param2 -5
                9.0

                ```
            - Plot results with relative parameter values:
                ```python
                >>> fig, ax = sensitivity.sobol(
                ...     title="Parameter Sensitivity Analysis",
                ...     xlabel="Relative Parameter Value",
                ...     ylabel="Model Output"
                ... )
                >>> plt.show()

                ```
                ![one-at-a-time](./../_images/sensitivity/one-at-a-time.png)

            - Plot results with actual parameter values:
                ```python
                >>> fig2, ax2 = sensitivity.sobol(
                ...     real_values=True,
                ...     title="Parameter Sensitivity Analysis",
                ...     xlabel="Parameter Value",
                ...     ylabel="Model Output"
                ... )

                ```
                ![one-at-a-time](./../_images/sensitivity/real_values.png)
        """
        if self.return_values == 1:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

            for i in range(self.num_parameters):
                k = self.positions[i]
                if real_values:
                    ax.plot(
                        self.sen[self.parameter.index[k]][2],
                        self.sen[self.parameter.index[k]][1],
                        Sensitivity.marker_style(k),
                        linewidth=3,
                        markersize=10,
                        label=self.parameter.index[k],
                    )
                else:
                    ax.plot(
                        self.sen[self.parameter.index[k]][0],
                        self.sen[self.parameter.index[k]][1],
                        Sensitivity.marker_style(k),
                        linewidth=3,
                        markersize=10,
                        label=self.parameter.index[k],
                    )

            ax.set_title(title, fontsize=12)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            ax.tick_params(axis="both", which="major", labelsize=labelfontsize)

            ax.legend(fontsize=12)
            plt.tight_layout()
            return fig, ax
        else:  # self.return_values == 2 and CalculatedValues
            try:
                fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 6))

                for i in range(self.num_parameters):
                    # for i in range(len(self.sen[self.parameter.index[0]][0])):
                    k = self.positions[i]
                    if real_values:
                        ax1.plot(
                            self.sen[self.parameter.index[k]][2],
                            self.sen[self.parameter.index[k]][1],
                            Sensitivity.marker_style(k),
                            linewidth=3,
                            markersize=10,
                            label=self.parameter.index[k],
                        )
                    else:
                        ax1.plot(
                            self.sen[self.parameter.index[k]][0],
                            self.sen[self.parameter.index[k]][1],
                            Sensitivity.marker_style(k),
                            linewidth=3,
                            markersize=10,
                            label=self.parameter.index[k],
                        )

                ax1.set_title(title, fontsize=12)
                ax1.set_xlabel(xlabel, fontsize=12)
                ax1.set_ylabel(ylabel, fontsize=12)
                ax1.tick_params(axis="both", which="major", labelsize=labelfontsize)

                ax1.legend(fontsize=12)

                for i in range(self.num_parameters):
                    k = self.positions[i]
                    # for j in range(self.n_values):
                    for j in range(len(self.sen[self.parameter.index[k]][0])):
                        if plotting_from == "":
                            plotting_from = 0
                        if plotting_to == "":
                            plotting_to = len(
                                self.sen[self.parameter.index[k]][3][j].values
                            )

                        ax2.plot(
                            self.sen[self.parameter.index[k]][3][j].values[
                                plotting_from:plotting_to
                            ],
                            label=self.sen[self.parameter.index[k]][2][j],
                        )

                # ax2.legend(fontsize=12)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(loc=6, fancybox=True, bbox_to_anchor=(1.015, 0.5))

                ax2.set_title(title2, fontsize=12)
                ax2.set_xlabel(xlabel2, fontsize=12)
                ax2.set_ylabel(ylabel2, fontsize=12)

                plt.subplots_adjust(
                    left=spaces[0],
                    bottom=spaces[1],
                    right=spaces[2],
                    top=spaces[3],
                    wspace=spaces[4],
                    hspace=spaces[5],
                )

            except ValueError:
                raise ValueError(
                    "To plot calculated values, you should choose return_values==2 in the sensitivity object"
                )

            plt.tight_layout()
            plt.show()
            return fig, (ax1, ax2)
