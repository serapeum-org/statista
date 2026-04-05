import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

# %%
# Sample data
np.random.seed(10)
data = [np.random.normal(0, 1, 100) for _ in range(4)]


# %%
def raincloud_plot(
    data,
    overlay=True,
    violin_width=0.4,
    scatter_offset=0.15,
    boxplot_width=0.1,
    order=['violin', 'scatter', 'box'],
):
    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = len(data)
    positions = np.arange(1, n_groups + 1)

    # Dictionary to map plot types to the functions
    plot_funcs = {
        'violin': lambda pos, d: ax.violinplot(
            [d],
            positions=[pos],
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=violin_width,
        ),
        'scatter': lambda pos, d: ax.scatter(
            np.random.normal(pos, 0.04, size=len(d)),
            d,
            alpha=0.6,
            color='black',
            s=10,
            edgecolor='white',
            linewidth=0.5,
        ),
        'box': lambda pos, d: ax.boxplot(
            [d],
            positions=[pos],
            widths=boxplot_width,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'),
        ),
    }

    # Plot elements according to the specified order and selected plots
    for i, d in enumerate(data):
        base_pos = positions[i]
        if overlay:
            for plot_type in order:
                plot_funcs[plot_type](base_pos, d)
        else:
            for j, plot_type in enumerate(order):
                offset = (j - 1) * scatter_offset
                plot_funcs[plot_type](base_pos + offset, d)

    # Customize the appearance of violins if they are included
    if 'violin' in order:
        for (
            pc
        ) in ax.collections:  # all polygons created by violinplot are in ax.collections
            if isinstance(pc, PolyCollection):
                pc.set_facecolor('skyblue')
                pc.set_edgecolor('blue')
                pc.set_alpha(0.3)
                pc.set_linewidth(1)
                pc.set_linestyle('-')

    # Set x-tick labels
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Group {i + 1}' for i in range(n_groups)])

    # Add grid lines for better readability
    ax.yaxis.grid(True)

    # Display the plot
    plt.show()


# %%
# Sample data
np.random.seed(10)
data = [np.random.normal(0, 1, 100) for _ in range(4)]

# Call the function to plot
raincloud_plot(
    data, overlay=True, order=['violin', 'scatter', 'box']
)  # Overlaid with default order
raincloud_plot(
    data, overlay=True, order=['violin', 'scatter']
)  # Overlaid with default order
raincloud_plot(
    data, overlay=False, order=['scatter', 'box', 'violin']
)  # Side-by-side with custom order
raincloud_plot(
    data, overlay=False, order=['box', 'violin']
)  # Side-by-side with custom order
raincloud_plot(
    data, overlay=False, order=['violin', 'scatter', 'box']
)  # Side-by-side with custom order
