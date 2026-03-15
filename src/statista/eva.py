"""Extreme value analysis.

Annual Maximum Series (AMS) Analysis is a statistical method commonly used in fields like hydrology, meteorology, and
environmental engineering to analyze extreme events, such as floods, rainfall, or temperatures. The primary goal of AMS
analysis is to assess the frequency and magnitude of extreme events over time.

Key Concepts of AMS Analysis

Definition:
    The Annual Maximum Series is a time series composed of the maximum values observed within each year. For example,
    in hydrology, the AMS might consist of the highest daily flow recorded in each year for a river.

Purpose:
    The AMS is used to model and predict the probability of extreme events occurring in the future. This is crucial for
    risk assessment and the design of infrastructure to withstand such events (e.g., dams, levees, drainage systems).

Advantages of AMS Analysis
    - Simplicity: AMS analysis is straightforward and focuses on extreme events, which are often of primary interest.
    - Historical Context: Provides insights based on historical extreme values, which are directly relevant for
        planning and design.

Limitations of AMS Analysis
    - Data Limitations: The accuracy of AMS analysis depends on the availability and quality of long-term data.
    - Ignores Sub-Annual Events: AMS considers only one value per year, potentially ignoring significant events that
        occur more than once in a year.

Common Applications:
    - Flood Frequency Analysis: AMS is often used to estimate the probability of extreme flood events to help design
        flood control infrastructure.
    - Rainfall Analysis: Used to assess the risk of extreme rainfall events for urban drainage design.
    - Temperature Extremes: AMS can be used to evaluate the risk of extremely high or low temperatures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame

from statista.distributions import Distributions


def ams_analysis(
    time_series_df: DataFrame,
    ams: bool = False,
    ams_start: str = "YE-OCT",
    save_plots: bool = False,
    save_to: str | None = None,
    filter_out: float | int | None = None,
    distribution: str = "GEV",
    method: str = "lmoments",
    obj_func: Callable | None = None,
    quartile: float = 0,
    alpha: float = 0.1,
) -> tuple[DataFrame, DataFrame]:
    """Annual Maximum Series analysis.

    ams analysis method reads resamples all the time series in the given dataframe to annual maximum, then fits
    the time series to a given distribution and parameter estimation method.

    Args:
        time_series_df (DataFrame):
            DataFrame containing multiple time series to do the statistical analysis on.
        ams (bool):
            True if the given time series is annual mean series. Default is False.
        ams_start (str):
            The beginning of the year which is used to resample the time series to get the annual maximum series.
            Default is "A-OCT".
        save_plots (bool):
            True if you want to save the plots.
        save_to (str):
            The rdir where you want to save the statistical properties.
        filter_out (bool):
            For observed or hydraulic model data it has gaps of times where the model did not run or gaps in the observed
            data if these gap days are filled with a specific value and you want to ignore it here
            give filter_out = Value you want
        distribution (str):
            distribution name. Default is "GEV".
        method (str):
            available methods are 'mle', 'mm', 'lmoments', 'optimization'. Default is "lmoments".
        obj_func (callable):
            objective function to be used in the optimization method, default is None. for Gumbel distribution there is the
            `Gumbel.truncated_distribution` and similarly for the GEV distribution there is the GEV.truncated_distribution.
        quartile (float):
            the quartile is only used when estimating the distribution parameters based on optimization and a threshould
            value, the threshold value will be calculated as the quartile coresponding to the value of this parameter.
        alpha (float, optional):
            alpha or Significance level is a value of the confidence interval. Default is [0.1].

    Returns:
        DataFrame:
            Statistical properties like mean, std, min, 5%, 25%, median, 75%, 95%, max, start_year, end_year, nyr, q1.5,
            q2, q5, q10, q25, q50, q100, q200, q500, q1000.
        DataFrame:
            Distribution properties like the shape, location, and scale parameters of the fitted distribution, plus the
            D-static and P-Value of the KS test.

    Examples:
    - First read the data as `pandas.DataFrame`.
        ```python
        >>> import pandas as pd
        >>> ams_gauges = pd.read_csv(f"examples/data/ams-gauges.csv", index_col=0)
        >>> print(ams_gauges) # doctest: +SKIP
            Frankfurt  Mainz  Kaub  Andernach  Cologne   Rees
        date
        1951         -9   4250  4480       6080     6490   6830
        1952         -9   4490  4610       6970     7110   7340
        1953         -9   4270  4380       7300     7610   7970
        1954         -9   2850  2910       3440     3620   3840
        1955         -9   5940  6050       9460     9460   9500
        1956         -9   5000  5150       7140     7270   7540
        1957         -9   4500  4520       6650     6750   6950
        .....
        1998       1060   4720  4790       6910     6700   6150
        1999       1420   5480  5730       8160     8530   9240
        2000        625   3750  3900       6390     6370   6550
        2001       1140   5420  5710       8320     8410   8410
        2002       1170   4950  5140       7260     7240   7940
        2003       1800   5090  5350       8620     8840   9470
        2004        197   1150  1190       1470     1580   1810

        ```
    - The time series data we have just read are the annual maximum series of the gauges, the first column is an
        index of the year (54 years in total) and the rest are dischate values in m3/s for each the station. a value 0f
        "-9" is used to fill the missing data.
    - The `ams_analysis` function takes the time series `DataFrame` as the first and only positional argument,
        all the other arguments are optional. Since the time series is annual maximum series already, so we don't
        need the function to do any resampling, we set `ams=True`. The `ams_start` could be used to provide the
        beginning of the year to resample the time series to ams (i.e., `ams_start = "A-OCT"`).
    - We want to save the plots, so we set `save_plots=True` and provide the directory where we want to save the plots in
        `save_to`.
    - We also want to filter out the missing data, so we set `filter_out=-9`.
    - In order to fit the time series to a distribution we also to provide the parameter estimation method (i.e.,
        `lmoments`, `mle`, `mm`, `optimization`), the default is the `lmoments`, and you need to provide the name of
        the distribution you want to fit the time series to (i.e., `GEV`, `Gumbel`). So for that we
        will use `method="lmoments"`, and `distribution="GEV"`.
    - The `alpha` is the significance level of the confidence interval, the default is 0.1. The `alpha` parameter is
        necessary for the confidence interval calculation.
        ```python
        >>> method = "lmoments"
        >>> save_to = "examples/data/gauges"
        >>> statistical_properties, distribution_properties = ams_analysis(
        ...     time_series_df=ams_gauges,
        ...     ams=True,
        ...     save_plots=True,
        ...     save_to=save_to,
        ...     filter_out=-9,
        ...     method=method,
        ...     alpha=0.05,
        ... ) # doctest: +SKIP
        -----KS Test--------
        Statistic = 0.07317073170731707
        Accept Hypothesis
        P value = 0.9999427584427157
        -----KS Test--------
        Statistic = 0.07317073170731707
        Accept Hypothesis
        P value = 0.9999427584427157
        2024-08-18 12:45:04.779 | DEBUG    | statista.confidence_interval:boot_strap:104 - Some values used top 10 low/high samples; results may be unstable.
        2024-08-18 12:45:05.221 | INFO     | statista.eva:ams_analysis:300 - Gauge Frankfurt done.

        ```
    - The `ams_analysis` function will iterate over all the gauges in the time series and fit the time series to the
        distribution and calculate the statistical properties and the distribution properties of the fitted distribution.
    - One of the outputs of the function is the statistical properties of the time series, which includes the mean, std,
        min, and  some quantile (5%, 25%, ..., 95%, max).
        ```python
        >>> print(statistical_properties.loc[:, statistical_properties.columns[:9]]) # doctest: +SKIP
                          mean          std     min       5%      25%  median      75%       95%      max
        id
        Frankfurt   917.439024   433.982918   197.0   347.00   548.00   882.0  1170.00   1760.00   1990.0
        Mainz      4153.333333  1181.707804  1150.0  2286.50  3415.00  4190.0  4987.50   5914.00   6920.0
        Kaub       4327.092593  1243.019565  1190.0  2394.50  3635.00  4350.0  5147.50   6383.50   7160.0
        Andernach  6333.407407  2016.211257  1470.0  3178.00  5175.00  6425.0  7412.50   9717.00  10400.0
        Cologne    6489.277778  2037.005658  1580.0  3354.50  5277.50  6585.0  7560.00   9728.85  10700.0
        Rees       6701.425926  2074.994365  1810.0  3556.50  5450.00  6575.0  7901.75  10005.00  11300.0

        ```
    - The rest of the columns in the `statistical_properties` are start_year, end_year, nyr, q1.5, q2, q5, q10, q25,
        q50, q100, q200, q500, q1000, which are the return periods of the fitted distribution.
        ```python
        >>> print(statistical_properties.loc[:, statistical_properties.columns[9:]]) # doctest: +SKIP
                   start_year  end_year   nyr         q1.5           q2  ...          q200          q500         q1000
        id
        Frankfurt      1964.0    2004.0  40.0   683.254634   855.296864  ...   2258.332886   2460.823383   2717.037039
        Mainz          1951.0    2004.0  53.0  3627.907224  4164.824744  ...   6734.883442   6919.948680   7110.767115
        Kaub           1951.0    2004.0  53.0  3761.253314  4321.114689  ...   7131.430892   7348.738113   7577.263513
        Andernach      1951.0    2004.0  53.0  5450.050443  6369.734950  ...  10654.874462  10950.940916  11252.770123
        Cologne        1951.0    2004.0  53.0  5583.579049  6507.694660  ...  10940.851299  11261.139356  11591.687060
        Rees           1951.0    2004.0  53.0  5759.172691  6693.471602  ...  11368.384249  11728.167908  12106.027638

        ```
    - The other output is the distribution properties of the fitted distribution, which includes the shape, location, and
        scale parameters of the fitted distribution, plus the D-static and P-Value of the KS test.
        ```python
        >>> print(distribution_properties) # doctest: +SKIP
                          c          loc        scale  D-static   P-Value
        id
        Frankfurt  0.051852   718.720761   376.188608  0.073171  0.999943
        Mainz      0.307295  3743.806013  1214.617042  0.055556  0.999998
        Kaub       0.282580  3881.573477  1262.426086  0.055556  0.999998
        Andernach  0.321513  5649.076008  2084.383132  0.074074  0.998738
        Cologne    0.306146  5783.017454  2090.224037  0.074074  0.998738
        Rees       0.284227  5960.022503  2107.197210  0.074074  0.998738

        ```
    - Since we have set `save_plots=True`, the function will save the plots in the directory we have provided in `save_to`.
        For example, the plot of Frankfurt's time series data is saved as "Frankfurt.png" for the `pdf` and `cdf` and
        "f-Frankfurt.png" for the confidince interval plot in the specified directory.'

        ![Frankfurt](./../_images/Frankfurt.png)

        ![f-Frankfurt](./../_images/f-Frankfurt.png)

    """
    gauges = time_series_df.columns.tolist()
    # List of the table output, including some general data and the return periods.
    col_csv = [
        "mean",
        "std",
        "min",
        "5%",
        "25%",
        "median",
        "75%",
        "95%",
        "max",
        "start_year",
        "end_year",
        "nyr",
    ]
    rp_name = [
        "q1.5",
        "q2",
        "q5",
        "q10",
        "q25",
        "q50",
        "q100",
        "q200",
        "q500",
        "q1000",
    ]
    col_csv += rp_name

    # In a table where duplicates are removed (np.unique), find the number of
    # gauges contained in the .csv file.
    # Declare a dataframe for the output file, with as index the gauge numbers
    # and as columns all the output names.
    statistical_properties = pd.DataFrame(np.nan, index=gauges, columns=col_csv)
    statistical_properties.index.name = "id"

    if distribution == "GEV":
        cols = ["c", "loc", "scale", "D-static", "P-Value"]
    else:
        cols = ["loc", "scale", "D-static", "P-Value"]

    distribution_properties = pd.DataFrame(
        np.nan,
        index=gauges,
        columns=cols,
    )
    distribution_properties.index.name = "id"
    # required return periods
    return_period_list = [1.5, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
    return_period = np.array(return_period_list)
    # these values are the Non Exceedance probability (F) of the chosen
    # return periods non_exceed_prop = 1 - (1/return_period)
    # Non Exceedance probabilities
    # non_exceed_prop = [1/3, 0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.995, 0.998]
    non_exceed_prop = 1 - (1 / return_period)
    if save_plots and save_to is None:
        raise ValueError(
            "save_to must be provided when save_plots is True"
        )

    save_path = Path(save_to) if save_to is not None else Path(".")
    # Iteration over all the gauge numbers.
    if save_plots:
        rpath = save_path.joinpath("figures")
        if not rpath.exists():
            rpath.mkdir(parents=True, exist_ok=True)

    for i in gauges:
        q_ts = time_series_df.loc[:, i].to_frame()
        # The time series is resampled to the annual maxima, and turned into a numpy array.
        # The hydrological year is 1-Nov/31-Oct (from Petrow and Merz, 2009, JoH).
        if not ams:
            ams_df = q_ts.resample(ams_start).max()
            ams_arr = ams_df.values
        else:
            ams_df = q_ts
            ams_arr = q_ts.values

        if filter_out is not None:
            ams_df = ams_df.loc[ams_df[ams_df.columns[0]] != filter_out, :]
            ams_arr = ams_arr[ams_arr != filter_out]

        dist = Distributions(distribution, data=ams_arr)
        # estimate the parameters through the given method
        try:
            threshold = np.quantile(ams_arr, quartile)
            param_dist = dist.fit_model(
                method=method,
                obj_func=obj_func,
                threshold=threshold,
            )
        except Exception as e:
            logger.warning(
                f"The gauge {i} parameters could not be estimated because of {e}"
            )
            continue

        (
            distribution_properties.loc[i, "D-static"],
            distribution_properties.loc[i, "P-Value"],
        ) = dist.ks()

        if distribution == "GEV":
            distribution_properties.loc[i, "c"] = param_dist["shape"]
            distribution_properties.loc[i, "loc"] = param_dist["loc"]
            distribution_properties.loc[i, "scale"] = param_dist["scale"]
        else:
            distribution_properties.loc[i, "loc"] = param_dist["loc"]
            distribution_properties.loc[i, "scale"] = param_dist["scale"]

        # Return periods from the fitted distribution are stored.
        # get the Discharge coresponding to the return periods
        q_rp = dist.inverse_cdf(non_exceed_prop, param_dist)

        # Gumbel.plot method calculates the theoretical values
        # based on the Gumbel distribution
        # parameters, theoretical cdf (or weibul), and calculate the confidence interval
        if save_plots:
            fig, _ = dist.plot()
            _, _, fig2, _ = dist.confidence_interval(
                method=method, plot_figure=True, alpha=alpha
            )

            fig.savefig(f"{save_path}/figures/{i}.png", format="png")
            plt.close()

            fig2.savefig(f"{save_path}/figures/f-{i}.png", format="png")
            plt.close()

        quantiles = np.quantile(ams_arr, [0.05, 0.25, 0.50, 0.75, 0.95])
        statistical_properties.loc[i, "mean"] = ams_arr.mean()
        statistical_properties.loc[i, "std"] = ams_arr.std()
        statistical_properties.loc[i, "min"] = ams_arr.min()
        statistical_properties.loc[i, "5%"] = quantiles[0]
        statistical_properties.loc[i, "25%"] = quantiles[1]
        statistical_properties.loc[i, "median"] = quantiles[2]
        statistical_properties.loc[i, "75%"] = quantiles[3]
        statistical_properties.loc[i, "95%"] = quantiles[4]
        statistical_properties.loc[i, "max"] = ams_arr.max()
        statistical_properties.loc[i, "start_year"] = ams_df.index.min()
        statistical_properties.loc[i, "end_year"] = ams_df.index.max()

        if ams:
            statistical_properties.loc[i, "nyr"] = (
                statistical_properties.loc[i, "end_year"]
                - statistical_properties.loc[i, "start_year"]
            )
        else:
            statistical_properties.loc[i, "nyr"] = (
                statistical_properties.loc[i, "end_year"]
                - statistical_properties.loc[i, "start_year"]
            ).days / 365.25

        for irp, irp_name in zip(q_rp, rp_name):
            statistical_properties.loc[i, irp_name] = irp

        # Print for prompt and check progress.
        logger.info(f"Gauge {i} done.")
    return statistical_properties, distribution_properties
