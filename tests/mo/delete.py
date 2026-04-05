import numpy as np

from statista.confidence_interval import ConfidenceInterval
from statista.descriptors import rmse_hf

data = [3.1, 2.4, 5.6, 8.4]
indeces = ConfidenceInterval.bs_indexes(data, n_samples=2)
[i for i in indeces]
next(indeces)
