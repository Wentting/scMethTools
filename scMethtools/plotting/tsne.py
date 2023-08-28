#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月25日

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_tsne(x, labels, main="A tSNE visualization", n=20,
              pad=0.1, cex=0.65, pch=19, add=False, legend_suffix="",
              cex_main=1, cex_legend=1):
    col_vector = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(np.unique(labels))))
    layout = x

    xylim = np.ptp(layout, axis=0)
    xylim = xylim + (xylim * pad) * np.array([-0.5, 0.5])

    if not add:
        plt.figure(figsize=(10, 10))
        plt.xlim(xylim[0])
        plt.ylim(xylim[1])
        plt.gca().set_axis_off()
        plt.gca().add_patch(plt.Rectangle((xylim[0][0], xylim[1][0]),
                                          xylim[0][1] - xylim[0][0], xylim[1][1] - xylim[1][0],
                                          fill=False, color="#aaaaaa", linewidth=0.25))

    plt.scatter(layout[:, 0], layout[:, 1], c=col_vector[labels.astype(int)],
                s=cex * 100, marker=pch)
    plt.title(main, fontsize=cex_main * 10)

    labels_u = np.unique(labels)
    legend_pos = "upper right"
    legend_text = labels_u.astype(str)
    if add:
        legend_pos = "lower right"
        legend_text = [label + legend_suffix for label in legend_text]
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col_vector[i],
                                  markersize=cex * 10) for i in labels_u]
    plt.legend(legend_elements, legend_text, loc=legend_pos,
               prop={'size': cex_legend * 10}, frameon=False)


# Example usage
# Replace 'x', 'labels', and other parameters with your actual data
# plot_tsne(x, labels)

plt.show()
