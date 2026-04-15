import numpy as np
from scipy.stats import chisquare, gumbel_r

from statista.distributions import Gumbel

# %%
from statista.utils import merge_small_bins

merge_small_bins([10, 3, 2], [10, 3, 2])
# %%

data = np.loadtxt("examples/data/gumbel.txt")
gumbel_dist = Gumbel(data)
gumbel_dist.fit_model()
gumbel_dist.chisquare()
from scipy.stats import chi2

parameters = gumbel_dist.parameters
loc = parameters["loc"]
scale = parameters["scale"]

bin_edges = np.histogram_bin_edges(data, bins="sturges")
obs_counts, _ = np.histogram(data, bins=bin_edges)


# compute expected counts under the fitted distribution
expected_prob = np.diff(gumbel_r.cdf(bin_edges, loc=loc, scale=scale))
expected_counts = expected_prob * len(data)

# Pearson’s χ² test assumes each expected count is sufficiently large (at least about 5); otherwise the asymptotic χ² approximation is unreliable
merged_obs, merged_exp = merge_small_bins(obs_counts, expected_counts)

# merge bins with bin_count_fitted_data < 5 and rescale expected to match bin_count_observed.sum()
# (see previous message for details) -> new_obs, new_exp

# finally call chisquare on the binned counts
degrees_of_freedom = len(merged_obs) - 1 - 2  # subtract loc and scale

chi2_stat, p = chisquare(merged_obs, f_exp=merged_exp, ddof=2)
