"""Rhine gauges example"""

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd

from statista.distributions import Distributions

# %%
ams = pd.read_csv("examples/data/rhine.csv")
ams.head()
ams.replace(0, np.nan, inplace=True)
ams.dropna(axis=0, inplace=True)
# %%
rees_gauge = ams.loc[:, "rees"].values
cologne_gauge = ams.loc[:, "cologne"].values
maxau_gauge = ams.loc[:, "maxau"].values
rockenau_gauge = ams.loc[:, "rockenau"].values
# %% Exponential distribution (mle)
dist_obj = Distributions("Exponential", cologne_gauge)
# default parameter estimation method is maximum liklihood method
mle_param = dist_obj.fit_model(method="mle")
dist_obj.ks()
dist_obj.chisquare()

print(mle_param)
# calculate and plot the pdf
pdf, fig, ax = dist_obj.pdf(plot_figure=True)
cdf, _, _ = dist_obj.cdf(plot_figure=True)
# %% exponential distribution (lmoments)
dist_obj = Distributions("Exponential", cologne_gauge)
# default parameter estimation method is maximum likelihood method
mle_param = dist_obj.fit_model(method="lmoments")
dist_obj.ks()
dist_obj.chisquare()

print(mle_param)
# calculate and plot the pdf
pdf, fig, ax = dist_obj.pdf(plot_figure=True)
cdf, _, _ = dist_obj.cdf(plot_figure=True)
# %% GEV (mle)
gev_cologne = Distributions("GEV", cologne_gauge)
# default parameter estimation method is maximum likelihood method
mle_param = gev_cologne.fit_model(method="mle")
gev_cologne.ks()
gev_cologne.chisquare()

print(mle_param)
# shape = -1 * mle_param[0]
# calculate and plot the pdf
pdf, fig, ax = gev_cologne.pdf(plot_figure=True)
cdf, _, _ = gev_cologne.cdf(plot_figure=True)
# %% cologne (lmoment)
gev_cologne = Distributions("GEV", cologne_gauge)
# default parameter estimation method is maximum likelihood method
lmom_param = gev_cologne.fit_model(method="lmoments")
gev_cologne.ks()
gev_cologne.chisquare()

print(lmom_param)
# shape = -1 * `lmom_param[0]
# calculate and plot the pdf
pdf, fig, ax = gev_cologne.pdf(plot_figure=True)
cdf, _, _ = gev_cologne.cdf(plot_figure=True)
