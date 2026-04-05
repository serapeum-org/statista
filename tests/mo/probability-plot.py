import matplotlib

matplotlib.use("TkAgg")
import scipy as sp
from scipy.stats import skew

from statista.distributions import Gumbel

# %%

parameters = {'loc': 0, 'scale': 1}
gumbel_dist = Gumbel(parameters=parameters)
random_data = gumbel_dist.random(100)
gumbel_dist.pdf(data=random_data, plot_figure=True)
import matplotlib.pyplot as plt

# %%
import numpy as np
import scipy.stats as stats

data = np.loadtxt("examples/data/gev.txt")
stats.probplot(data, dist="genextreme", sparams=(0, 1, 0.1))  # plot=plt
plt.show()
# %%

from scipy.stats import genextreme

genextreme.sf()
