import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

# %%
series = pd.read_csv('examples/data/temp.csv', header=0, index_col=0)
series.plot()
plt.show()
# %%
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(series)
plt.show()
import matplotlib.pyplot as plt

# %%
import pandas as pd
from statsmodels.tsa.stattools import acf

from statista.time_series import TimeSeries

# Sample time series data
time_series_1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
time_series_2 = pd.Series([2, 1, 2, 4, 3, 5, 4, 6, 7, 8])
df = pd.DataFrame({'Time Series 1': time_series_1, 'Time Series 2': time_series_2})

ax, fig = TimeSeries._get_ax_fig(1)


# %%
def auto_correlation():
    for i in range(len(df.columns)):
        plot = acf(df.iloc[:, i], nlags=10)
        ax.plot(plot, label=df.columns[i])


import matplotlib.pyplot as plt

# %%
import pandas as pd
from statsmodels.tsa.stattools import acf

# Sample time series data
time_series_1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
time_series_2 = pd.Series([2, 1, 2, 4, 3, 5, 4, 6, 7, 8])


# Plot autocorrelation
plt.figure(figsize=(10, 6))
plt.show()


def calculate_plot_acf(
    time_series_1, nlag: int = 10, name: str = None, marker: str = 'o', color='blue'
):
    """Calculate and plot the auto-correlation of a time series with confidence intervals.

    Parameters
    ----------
    time_series_1:
    nlag: int, optional, default: 10
        Number of lags to calculate.
    name: str, optional, default: None
        Name of the time series.
    marker: str, optional, default: 'o'
        Marker style for the plot.
    color: str, optional, default: 'blue'
        Color of the plot.

    Returns
    -------

    """
    # Calculate auto-correlation with confidence intervals (95% and 99%)
    auto_corr, ci_1_95 = acf(time_series_1, nlags=nlag, alpha=0.05, fft=False)
    _, ci_1_99 = acf(time_series_1, nlags=nlag, alpha=0.01, fft=False)

    plt.plot(auto_corr, label=name, marker=marker)
    plt.fill_between(
        range(len(auto_corr)),
        ci_1_95[:, 0],
        ci_1_95[:, 1],
        color=color,
        alpha=0.2,
        label=f"95% CI ({name})",
    )
    plt.fill_between(
        range(len(auto_corr)),
        ci_1_99[:, 0],
        ci_1_99[:, 1],
        color=color,
        alpha=0.1,
        label=f"99% CI ({name})",
    )


calculate_plot_acf(time_series_1, name='Time Series 1', marker='o', color="blue")
calculate_plot_acf(time_series_2, name='Time Series 2', marker='x', color="green")

# Customize plot
plt.title('Autocorrelation of Multiple Time Series with Confidence Intervals')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.axhline(0, color='black', lw=1, linestyle='--')  # Add a horizontal line at 0
plt.legend()
plt.show()
