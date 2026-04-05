import pandas as pd

from statista.distributions import Distributions

# Load your time series data
data = pd.read_csv("examples/data/time_series1.txt", header=None)[0].tolist()
# Create a distribution object (e.g., Gumbel)
dist = Distributions("Gumbel", data)

# Fit the distribution using maximum likelihood
params = dist.fit_model(method="mle")
print(params)

# Calculate and plot the PDF and CDF
pdf = dist.pdf(plot_figure=True)
cdf, _, _ = dist.cdf(plot_figure=True)

# Perform goodness-of-fit tests
ks_test = dist.ks()
chi2_test = dist.chisquare()

# Create a probability plot with confidence intervals
fig, ax = dist.plot()
# %%
from statista.distributions import GEV, PlottingPosition

# Create a GEV distribution
gev_dist = Distributions("GEV", data)

# Fit using L-moments
params = gev_dist.fit_model(method="lmoments")

# Calculate non-exceedance probabilities
cdf_weibul = PlottingPosition.weibul(data)

# Calculate confidence intervals
lower_bound, upper_bound, fig, ax = gev_dist.confidence_interval(plot_figure=True)
# %%
