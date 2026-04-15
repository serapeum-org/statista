"""Heavy tail example."""

import pandas as pd

rdir = rf"examples/data"
from statista.distributions import GEV, Distributions

# %%
dung = pd.read_csv(f"{rdir}/pdf_obs.txt", delimiter=" ")
dung.sort_values(by="ams", inplace=True)

scale = 2132.938715
loc = 6582.059315
shape = 0.0486556
parameters = {
    "loc": loc,
    "scale": scale,
    "shape": shape,
}
ams = dung["ams"].values

gev_positive = Distributions("GEV", ams, parameters)
pdf = gev_positive.pdf(parameters, plot_figure=False)
dung["scipy +ve"] = pdf

parameters["shape"] = -0.0486556
dist_negative = Distributions("GEV", ams, parameters)
pdf = dist_negative.pdf(parameters, plot_figure=False)
dung["scipy -ve"] = pdf
# %%
method = "lmoments"  # "mle"
parameters_lm = dist_negative.fit_model(method=method)
parameters_mle = dist_negative.fit_model(method="mle")
