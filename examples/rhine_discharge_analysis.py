"""
Rhine River Discharge Analysis using Statista Distributions

This script demonstrates how to use the statista.distributions module to analyze discharge
time series data from the Rhine River. We'll explore different probability distributions,
fit them to the data, and calculate return periods and flood frequency curves.

The Rhine River is one of Europe's major rivers, flowing through several countries including
Switzerland, Liechtenstein, Austria, Germany, France and the Netherlands. Analyzing discharge
data is crucial for flood risk assessment, water resource management, and understanding the
hydrological behavior of the river.

In this script, we will:
1. Load and preprocess discharge time series data from multiple gauges along the Rhine River
2. Fit different probability distributions to the data
3. Evaluate the goodness of fit for each distribution
4. Calculate return periods and flood frequency curves
5. Visualize the results
"""

import matplotlib.pyplot as plt

# Import necessary libraries
import numpy as np
import pandas as pd
from scipy import stats

# Import statista distributions module
from statista.distributions import (
    GEV,
    Distributions,
    Exponential,
    Gumbel,
    Normal,
    PlottingPosition,
)

# Set plot style
plt.style.use('ggplot')

# Display all columns in pandas DataFrames
pd.set_option('display.max_columns', None)

"""
Data Loading and Preprocessing
-----------------------------
We'll load the Rhine River discharge data from the CSV file. The data contains daily
discharge measurements from multiple gauges along the river. The first column is the date,
and the remaining columns represent different gauges.

We need to handle missing values (empty strings) in the data and convert the date column
to a datetime format.
"""


# %%
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Rhine River discharge data.

    Args:
        file_path: Path to the CSV file containing the discharge data

    Returns:
        pandas.DataFrame: Preprocessed discharge data
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Display the first few rows of the data
    print(f"Data shape: {df.shape}")
    print(df.head())

    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Set the date column as the index
    df.set_index('date', inplace=True)

    # Check for missing values
    print("\nNumber of missing values in each column:")
    print(df.isna().sum())

    # Convert empty strings to NaN
    df = df.replace('', np.nan)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for missing values again
    print("\nNumber of missing values after conversion:")
    print(df.isna().sum())

    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe())

    return df


# %%
"""
Exploratory Data Analysis
------------------------
Let's explore the data to understand the discharge patterns at different gauges along
the Rhine River. We'll visualize the time series, examine the distribution of discharge
values, and identify any seasonal patterns.
"""


def plot_time_series(df, selected_gauges):
    """
    Plot time series for selected gauges.

    Args:
        df: DataFrame containing the discharge data
        selected_gauges: List of gauge names to plot
    """
    plt.figure(figsize=(14, 8))
    for gauge in selected_gauges:
        if gauge in df.columns:
            plt.plot(df.index, df[gauge], label=gauge)
    plt.title('Discharge Time Series for Selected Gauges')
    plt.xlabel('Date')
    plt.ylabel('Discharge (m³/s)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_histograms(df, selected_gauges):
    """
    Create histograms for selected gauges.

    Args:
        df: DataFrame containing the discharge data
        selected_gauges: List of gauge names to plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, gauge in enumerate(selected_gauges):
        if gauge in df.columns and i < len(axes):
            # Use matplotlib's histogram function
            data = df[gauge].dropna()
            axes[i].hist(data, bins=20, density=True, alpha=0.7)

            # Add a density curve
            from scipy import stats

            min_val, max_val = data.min(), data.max()
            x = np.linspace(min_val, max_val, 1000)
            kde = stats.gaussian_kde(data)
            axes[i].plot(x, kde(x), 'r-', linewidth=2)

            axes[i].set_title(f'Distribution of Discharge at {gauge}')
            axes[i].set_xlabel('Discharge (m³/s)')
            axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.show()


def extract_annual_maxima(df):
    """
    Extract annual maximum discharge for each gauge.

    Args:
        df: DataFrame containing the discharge data

    Returns:
        pandas.DataFrame: Annual maximum discharge for each gauge
    """
    # Extract annual maximum discharge for each gauge
    annual_max = df.resample('Y').max()

    return annual_max


def plot_annual_maxima(annual_max, selected_gauges):
    """
    Plot annual maximum discharge for selected gauges.

    Args:
        annual_max: DataFrame containing annual maximum discharge
        selected_gauges: List of gauge names to plot
    """
    plt.figure(figsize=(14, 8))
    for gauge in selected_gauges:
        if gauge in annual_max.columns:
            plt.plot(annual_max.index, annual_max[gauge], 'o-', label=gauge)
    plt.title('Annual Maximum Discharge for Selected Gauges')
    plt.xlabel('Year')
    plt.ylabel('Maximum Discharge (m³/s)')
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
"""
Fitting Probability Distributions
-------------------------------
Now we'll fit different probability distributions to the annual maximum discharge data
for each gauge. We'll use the following distributions from the statista.distributions module:

1. Gumbel distribution
2. Generalized Extreme Value (GEV) distribution
3. Normal distribution
4. Exponential distribution

We'll evaluate the goodness of fit using statistical tests and visual inspection.
"""


def fit_distributions(data, method="lmoments"):
    """
    Fit different distributions to the data and evaluate goodness of fit.

    Args:
        data: numpy array of discharge values

    Returns:
        dict: Dictionary of fitted distribution objects and test results
    """
    # Remove NaN values
    data = data[~np.isnan(data)]

    # Sort data in ascending order
    data = np.sort(data)

    # Initialize distributions
    gumbel = Gumbel(data=data)
    gev = GEV(data=data)
    normal = Normal(data=data)
    exponential = Exponential(data=data)

    # Fit distributions
    gumbel_params = gumbel.fit_model(method=method)
    gev_params = gev.fit_model(method=method)
    normal_params = normal.fit_model(method=method)
    exponential_params = exponential.fit_model(method=method)

    # Perform Kolmogorov-Smirnov test
    gumbel_ks = gumbel.ks()
    gev_ks = gev.ks()
    normal_ks = normal.ks()
    exponential_ks = exponential.ks()

    # Return results
    return {
        'Gumbel': {'dist': gumbel, 'params': gumbel_params, 'ks': gumbel_ks},
        'GEV': {'dist': gev, 'params': gev_params, 'ks': gev_ks},
        'Normal': {'dist': normal, 'params': normal_params, 'ks': normal_ks},
        'Exponential': {
            'dist': exponential,
            'params': exponential_params,
            'ks': exponential_ks,
        },
    }


def plot_fitted_distributions(data, fitted_dists, gauge_name):
    """
    Plot the empirical and fitted distributions.

    Args:
        data: numpy array of discharge values
        fitted_dists: dictionary of fitted distribution objects
        gauge_name: name of the gauge
    """
    # Remove NaN values
    data = data[~np.isnan(data)]

    # Sort data in ascending order
    data = np.sort(data)

    # Calculate empirical CDF using Weibull plotting position
    pp = PlottingPosition.weibul(data)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot PDF
    ax1.hist(data, bins=20, density=True, alpha=0.5, label='Empirical')
    x = np.linspace(min(data), max(data), 1000)

    for name, dist_info in fitted_dists.items():
        dist = dist_info['dist']
        params = dist_info['params']

        # Plot PDF
        y_pdf = dist._pdf_eq(x, params)
        ax1.plot(x, y_pdf, label=f'{name} (KS p-value: {dist_info["ks"][1]:.4f})')

    ax1.set_title(f'Probability Density Function - {gauge_name}')
    ax1.set_xlabel('Discharge (m³/s)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True)

    # Plot CDF
    ax2.plot(data, pp, 'o', label='Empirical')

    for name, dist_info in fitted_dists.items():
        dist = dist_info['dist']
        params = dist_info['params']

        # Plot CDF
        y_cdf = dist._cdf_eq(x, params)
        ax2.plot(x, y_cdf, label=name)

    ax2.set_title(f'Cumulative Distribution Function - {gauge_name}')
    ax2.set_xlabel('Discharge (m³/s)')
    ax2.set_ylabel('Probability of Non-Exceedance')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_flood_frequency_curve(data, fitted_dists, gauge_name):
    """
    Calculate return periods and plot flood frequency curves.

    Args:
        data: numpy array of discharge values
        fitted_dists: dictionary of fitted distribution objects
        gauge_name: name of the gauge
    """
    # Remove NaN values
    data = data[~np.isnan(data)]

    # Sort data in ascending order
    data = np.sort(data)

    # Calculate empirical return periods using Weibull plotting position
    pp = PlottingPosition.weibul(data)
    rp = PlottingPosition.return_period(pp)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot empirical return periods
    plt.semilogx(rp, data, 'o', label='Empirical')

    # Generate return periods for plotting
    return_periods = np.logspace(0, 3, 1000)  # 1 to 1000 years
    non_exceed_prob = 1 - 1 / return_periods

    # Plot theoretical return periods for each distribution
    for name, dist_info in fitted_dists.items():
        dist = dist_info['dist']
        params = dist_info['params']

        # Calculate quantiles for each return period
        quantiles = dist.inverse_cdf(non_exceed_prob, params)

        # Plot flood frequency curve
        plt.semilogx(return_periods, quantiles, label=name)

    plt.title(f'Flood Frequency Curve - {gauge_name}')
    plt.xlabel('Return Period (years)')
    plt.ylabel('Discharge (m³/s)')
    plt.grid(True)
    plt.legend()

    # Add vertical lines for common return periods
    common_rp = [2, 5, 10, 25, 50, 100, 200, 500]
    for rp_val in common_rp:
        plt.axvline(x=rp_val, color='gray', linestyle='--', alpha=0.5)
        plt.text(
            rp_val, plt.ylim()[0], str(rp_val), ha='center', va='bottom', alpha=0.7
        )

    plt.show()


def analyze_distributions(annual_max, selected_gauges):
    """
    Fit distributions to annual maximum discharge for selected gauges.

    Args:
        annual_max: DataFrame containing annual maximum discharge
        selected_gauges: List of gauge names to analyze

    Returns:
        dict: Dictionary of fitted distribution results for each gauge
    """
    results = {}

    for gauge in selected_gauges:
        if gauge in annual_max.columns:
            print(f"\nFitting distributions to {gauge}...")
            data = annual_max[gauge].values
            results[gauge] = fit_distributions(data)

            # Print goodness of fit results
            print(f"\nGoodness of fit results for {gauge}:")
            for dist_name, dist_info in results[gauge].items():
                ks_stat = dist_info['ks'][0]
                ks_pvalue = dist_info['ks'][1]
                print(
                    f"{dist_name}: KS statistic = {ks_stat:.4f}, p-value = {ks_pvalue:.4f}"
                )

            # Plot fitted distributions
            plot_fitted_distributions(data, results[gauge], gauge)

            # Plot flood frequency curve
            plot_flood_frequency_curve(data, results[gauge], gauge)

    return results


"""
Calculating Design Floods
-----------------------
Design floods are discharge values associated with specific return periods. They are used
in the design of hydraulic structures, flood protection measures, and risk assessment.
Let's calculate design floods for common return periods using the best-fitting distribution
for each gauge.
"""


def find_best_distribution(fitted_dists):
    """
    Find the best-fitting distribution based on KS test p-value.

    Args:
        fitted_dists: dictionary of fitted distribution objects

    Returns:
        tuple: (best distribution name, distribution info)
    """
    best_dist = None
    best_pvalue = -1

    for name, dist_info in fitted_dists.items():
        pvalue = dist_info['ks'][1]
        if pvalue > best_pvalue:
            best_pvalue = pvalue
            best_dist = (name, dist_info)

    return best_dist


def calculate_design_floods(results):
    """
    Calculate design floods for common return periods.

    Args:
        results: Dictionary of fitted distribution results for each gauge

    Returns:
        pandas.DataFrame: Design floods for different return periods
    """
    common_rp = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
    design_floods = {}

    for gauge, fitted_dists in results.items():
        best_dist_name, best_dist_info = find_best_distribution(fitted_dists)
        dist = best_dist_info['dist']
        params = best_dist_info['params']

        # Calculate non-exceedance probabilities for common return periods
        non_exceed_prob = 1 - 1 / np.array(common_rp)

        # Calculate quantiles (design floods)
        quantiles = dist.inverse_cdf(non_exceed_prob, params)

        # Store results
        design_floods[gauge] = {
            'best_dist': best_dist_name,
            'return_periods': common_rp,
            'design_floods': quantiles,
        }

    # Create a DataFrame to display design floods
    design_flood_df = pd.DataFrame(index=common_rp)
    for gauge, info in design_floods.items():
        design_flood_df[f"{gauge} ({info['best_dist']})"] = info['design_floods']

    design_flood_df.index.name = 'Return Period (years)'
    design_flood_df.columns.name = 'Gauge (Best Distribution)'

    return design_flood_df


"""
Confidence Intervals
------------------
Let's calculate confidence intervals for the flood frequency curves to account for
uncertainty in the parameter estimation. We'll use the confidence interval methods
provided by the statista.distributions module.
"""


def plot_flood_frequency_with_ci(data, fitted_dists, gauge_name):
    """
    Plot flood frequency curve with confidence intervals.

    Args:
        data: numpy array of discharge values
        fitted_dists: dictionary of fitted distribution objects
        gauge_name: name of the gauge
    """
    # Remove NaN values
    data = data[~np.isnan(data)]

    # Sort data in ascending order
    data = np.sort(data)

    # Calculate empirical return periods using Weibull plotting position
    pp = PlottingPosition.weibul(data)
    rp = PlottingPosition.return_period(pp)

    # Find best distribution
    best_dist_name, best_dist_info = find_best_distribution(fitted_dists)
    dist = best_dist_info['dist']
    params = best_dist_info['params']

    # Generate return periods for plotting
    return_periods = np.logspace(0, 3, 1000)  # 1 to 1000 years
    non_exceed_prob = 1 - 1 / return_periods

    # Calculate confidence intervals
    ci = dist.confidence_interval(alpha=0.1, prob_non_exceed=non_exceed_prob)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot empirical return periods
    plt.semilogx(rp, data, 'o', label='Empirical')

    # Calculate quantiles for each return period
    quantiles = dist.inverse_cdf(non_exceed_prob, params)

    # Plot flood frequency curve
    plt.semilogx(return_periods, quantiles, label=best_dist_name)

    # Plot confidence intervals
    plt.fill_between(
        return_periods, ci[1], ci[0], alpha=0.2, label='90% Confidence Interval'
    )

    plt.title(f'Flood Frequency Curve with Confidence Intervals - {gauge_name}')
    plt.xlabel('Return Period (years)')
    plt.ylabel('Discharge (m³/s)')
    plt.grid(True)
    plt.legend()

    # Add vertical lines for common return periods
    common_rp = [2, 5, 10, 25, 50, 100, 200, 500]
    for rp_val in common_rp:
        plt.axvline(x=rp_val, color='gray', linestyle='--', alpha=0.5)
        plt.text(
            rp_val, plt.ylim()[0], str(rp_val), ha='center', va='bottom', alpha=0.7
        )

    plt.show()


def plot_confidence_intervals(annual_max, results, selected_gauges):
    """
    Plot flood frequency curves with confidence intervals for selected gauges.

    Args:
        annual_max: DataFrame containing annual maximum discharge
        results: Dictionary of fitted distribution results for each gauge
        selected_gauges: List of gauge names to plot
    """
    for gauge in selected_gauges:
        if gauge in results:
            print(
                f"\nPlotting flood frequency curve with confidence intervals for {gauge}..."
            )
            data = annual_max[gauge].values
            plot_flood_frequency_with_ci(data, results[gauge], gauge)


# %%
"""
Main function to run the analysis
-------------------------------
"""


def main():
    # Define the path to the data file
    file_path = 'examples/data/rhine-2.csv'

    # Load and preprocess the data
    print("Loading and preprocessing the data...")
    df = load_and_preprocess_data(file_path)

    # Define selected gauges for analysis
    selected_gauges = ['rees-0', 'cologne-0', 'kaub-0', 'mainz-0']

    # Exploratory data analysis
    print("\nPlotting time series for selected gauges...")
    plot_time_series(df, selected_gauges)

    print("\nPlotting histograms for selected gauges...")
    plot_histograms(df, selected_gauges)

    # Extract annual maximum discharge
    print("\nExtracting annual maximum discharge...")
    annual_max = extract_annual_maxima(df)

    print("\nPlotting annual maximum discharge for selected gauges...")
    plot_annual_maxima(annual_max, selected_gauges)

    # Fit distributions and analyze results
    print("\nFitting distributions and analyzing results...")
    results = analyze_distributions(annual_max, selected_gauges)

    # Calculate design floods
    print("\nCalculating design floods...")
    design_flood_df = calculate_design_floods(results)
    print("\nDesign Floods (m³/s) for Different Return Periods:")
    print(design_flood_df)

    # Plot confidence intervals
    print("\nPlotting flood frequency curves with confidence intervals...")
    plot_confidence_intervals(annual_max, results, selected_gauges)

    print("\nAnalysis complete!")

    # Summary of findings
    print("\nKey Findings:")
    print(
        "- The best-fitting distribution varies between gauges, highlighting the importance of testing multiple distributions"
    )
    print(
        "- The GEV and Gumbel distributions generally provide good fits for annual maximum discharge data, which is consistent with extreme value theory"
    )
    print(
        "- Confidence intervals widen for higher return periods, reflecting increased uncertainty in estimating rare events"
    )
    print(
        "- Design floods increase with return period, but the rate of increase varies between gauges"
    )


if __name__ == "__main__":
    main()
