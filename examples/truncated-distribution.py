import matplotlib

matplotlib.use("TkAgg")
import pandas as pd

from statista.distributions import Distributions, Gumbel, PlottingPosition

time_series1 = pd.read_csv("examples/data/time_series1.txt", header=None)[0].tolist()
time_series2 = pd.read_csv("examples/data/time_series2.txt", header=None)[0].tolist()
# %%
gumbel_series_1 = Distributions("Gumbel", time_series1)
param_lmoments = gumbel_series_1.fit_model(method="lmoments")
gumbel_series_1.ks()
gumbel_series_1.chisquare()
print(param_lmoments)
# calculate and plot the pdf
pdf = gumbel_series_1.pdf(plot_figure=True)
cdf, _, _ = gumbel_series_1.cdf(plot_figure=True)
upper, lower, fig, ax = gumbel_series_1.confidence_interval(alpha=0.1, plot_figure=True)
# %%
# calculate the F (Non-Exceedance probability based on weibul)
cdf_weibul = PlottingPosition.weibul(time_series1)
# %%
import numpy as np


def truncated_distribution(p, x, threshold):
    # threshold = p[0]
    loc = p[1]
    scale = p[2]

    truncated_data = x[x < threshold]
    nx2 = len(x[x >= threshold])
    # pdf with a scaled pdf
    # L1 is pdf based
    parameters = {"loc": loc, "scale": scale}
    pdf = Gumbel._pdf_eq(truncated_data, parameters)
    #  the CDF at the threshold is used because the data is assumed to be truncated, meaning that observations below
    #  this threshold are not included in the dataset. When dealing with truncated data, it's essential to adjust
    #  the likelihood calculation to account for the fact that only values above the threshold are observed. The
    #  CDF at the threshold effectively normalizes the distribution, ensuring that the probabilities sum to 1 over
    #  the range of the observed data.
    adjusted_cdf = 1 - Gumbel._cdf_eq(threshold, parameters)
    # calculates the negative log-likelihood of a Gumbel distribution
    # Adjust the likelihood for the truncation
    # likelihood = pdf / (1 - adjusted_cdf)

    l1 = (-np.log((pdf / scale))).sum()
    # L2 is cdf based
    l2 = (-np.log(adjusted_cdf)) * nx2
    # print x1, nx2, L1, L2
    return l1 * l2  # -np.sum(np.log(likelihood))


# %%
threshold = 18
param_dist = gumbel_series_1.fit_model(
    method="optimization", obj_func=truncated_distribution, threshold=threshold
)
print(param_dist)
# gumbel_series_1.plot(parameters=param_dist)
upper, lower, fig, ax = gumbel_series_1.confidence_interval(
    parameters=param_dist, alpha=0.1, plot_figure=True
)
# %%
