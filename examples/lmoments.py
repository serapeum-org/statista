import numpy as np
import pandas as pd

from statista.parameters import Lmoments

time_series1 = pd.read_csv("examples/data/time_series1.txt", header=None)[0].tolist()
time_series2 = pd.read_csv("examples/data/time_series2.txt", header=None)[0].tolist()
# %%
L = Lmoments(time_series1)
l1, l2, l3, l4 = L.calculate(4)

sample = np.array(time_series1)
n = len(sample)
# sort descinding
sample = np.sort(sample.reshape(n))[::-1]
b0 = np.mean(sample)
lmom1 = b0
b1 = np.array([(n - j - 1) * sample[j] / n / (n - 1) for j in range(n)]).sum()
lmom2 = 2 * b1 - b0

b2 = np.array(
    [(n - j - 1) * (n - j - 2) * sample[j] / n / (n - 1) / (n - 2) for j in range(n)]
).sum()

lmom3 = 6 * (b2 - b1) + b0

b3 = np.array(
    [
        (n - j - 1)
        * (n - j - 2)
        * (n - j - 3)
        * sample[j]
        / n
        / (n - 1)
        / (n - 2)
        / (n - 3)
        for j in range(n - 1)
    ]
).sum()

lmom4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
lmom1, lmom2, lmom3, lmom4
