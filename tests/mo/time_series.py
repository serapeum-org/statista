import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

print(plt.style.available)
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot, lag_plot

from statista.time_series import TimeSeries

# %%
data = np.random.randn(100)
ts = TimeSeries(data)

# Access the stats
print(ts.stats)

# Plot the different visualizations with custom options
# Example of plotting on a new figure and axes
fig_box, ax_box = ts.box_plot(
    title='Custom Box Plot',
    xlabel='Custom X',
    ylabel='Custom Y',
    color={'boxes': 'purple', 'whiskers': 'orange'},
)

# Example of using only a provided figure
fig = plt.figure(figsize=(12, 8))
fig_error, ax_error = ts.plot_error_bar(
    title='Custom Error Bar Plot on Existing Figure',
    xlabel='Custom X',
    ylabel='Custom Y',
    color='red',
    linestyle='--',
    marker='s',
    capsize=5,
    fig=fig,
)

# Example of using only a provided axes
fig, ax = plt.subplots(figsize=(10, 6))
fig_violin, ax_violin = ts.violin(
    title='Custom Violin Plot on Existing Axes',
    xlabel='Custom X',
    ylabel='Custom Y',
    mean=False,
    median=True,
    ax=ax,
)

# %%
alpha = 0.95
mean_color = "#f6eff7"
mean_prop = dict(marker='x', markeredgecolor='w', markerfacecolor='firebrick')
# %%
data = pd.read_csv("tests/mo/data1.csv", index_col=0)
data2 = pd.read_csv("tests/mo/data2.csv", index_col=0)
ylabel = "Shape parameter"
# %%
labels = list(data.index)
pos = list(range(1, 2 * len(labels) + 1))
positions1 = [i for i in pos if i % 2 != 0]
positions2 = [i for i in pos if i % 2 == 0]

box_plot_data = [row[~np.isnan(row)] for row in data.values]
box_plot_data2 = [row[~np.isnan(row)] for row in data2.values]

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))

bp_1d = ax.boxplot(
    box_plot_data,
    positions=positions1,
    notch='True',
    patch_artist=True,
    showmeans=False,
    meanline=False,
    # labels=labels,
    meanprops=mean_prop,
    boxprops=dict(facecolor="#27408B", alpha=alpha),
)
bp_2d = ax.boxplot(
    box_plot_data2,
    positions=positions2,
    notch='True',
    patch_artist=True,
    showmeans=False,
    meanline=False,
    # labels=labels2,
    meanprops=mean_prop,
    boxprops=dict(facecolor="#DC143C", alpha=alpha),
)
# for patch in bp['boxes']:  # zip(bp['boxes'], colors)
#     patch.set_facecolor("#27408B")

# for patch in bp_1d['means']: #zip(bp['boxes'], colors)
#     patch.set(color='red', linewidth=3)
#     patch.set_color("red")

for median in bp_1d['medians']:
    median.set(color='k', linewidth=2.5)
for median in bp_2d['medians']:
    median.set(color='k', linewidth=2.5)

for mean in bp_1d['means']:
    mean.set(color=mean_color, linewidth=1.5)
for median in bp_2d['means']:
    median.set(color=mean_color, linewidth=1.5)

# for patch in bp_2d['means']: #zip(bp['boxes'], colors)
#     patch.set(
#         color='red',
#         linewidth=3
#     )
#     patch.set_color("red")


# plt.xticks(rotation = 90)
ax.set_ylabel(ylabel, fontsize=12)
ax.set_xlabel("Gauges", fontsize=12)
ax.grid(axis="both", linestyle='-.', linewidth=0.3)  # color='r',
pos = [(i + j) / 2 for i, j in zip(positions1, positions2)]
ax.xaxis.set_ticks(pos)
ax.xaxis.set_ticks(pos)
# ax.plot(pos, gauge_data, "o", label="Historical data", alpha=0.8, color="dimgrey", markeredgecolor="k")
ax.set_xticklabels(data.index.to_list())
handles, labels = ax.get_legend_handles_labels()
new_handles = [bp_1d["boxes"][0], bp_2d["boxes"][0]] + handles
new_labels = ['1D', '1D-2D'] + labels
ax.legend(new_handles, new_labels, loc='best')

plt.subplots_adjust(
    wspace=0.5, hspace=0.5, top=0.96, bottom=0.15, left=0.10, right=0.95
)
plt.show()

# %%

import matplotlib.pyplot as plt
import numpy as np


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


# create test data
np.random.seed(19680801)
data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
plt.show()
ax1.set_title('Default violin plot')
ax1.set_ylabel('Observed values')
ax1.violinplot(data)

ax2.set_title('Customized violin plot')
plt.show()
parts = ax2.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

plt.show()
for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

plt.show()
quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)

whiskers = np.array(
    [
        calculate_wiskers(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)
    ]
)
plt.show()
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
plt.show()
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
plt.show()
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
plt.show()

# set style for the axes
labels = ['A', 'B', 'C', 'D']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()
