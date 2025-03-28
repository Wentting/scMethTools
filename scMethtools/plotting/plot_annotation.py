#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年11月30日

"""

import matplotlib.pyplot as plt
import numpy as np


def plot_peak_annotation(annotation, col='Feature',out_file=None):
    '''
    Generates a pie chart of the genomic annotations of identified VMR

    Args:
        annnotation           Connection to the annotation result to use
        out_file              Name and location of the output file

    Remarks:

    '''
    simple_annotation = np.array([x.split(' ')[0].lower() for x in annotation[col]])

    counts = {}
    for x in np.unique(simple_annotation):
        counts[x] = np.sum(simple_annotation == x)

    labels = 'promoter(1-5kb)','promoter(<=1kb)','five_prime_utr','three_prime_utr'

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.rcParams["font.size"] = "10"

    size = 0.4
    vals = np.array([counts[x] for x in labels])

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(2) * 4 + 1)[[1, 0], :]
    inner_colors = cmap(np.array([0, 1, 2, 3, 4]))[::-1, :]

    ax.pie([vals[0], sum(vals[1:])], radius=1, colors=outer_colors,
           wedgeprops=dict(width=size, edgecolor='w'), startangle=90)

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(vals.flatten(), radius=1 - size, colors=inner_colors,
                                      autopct=lambda pct: func(pct, vals), pctdistance=1.2,
                                      wedgeprops=dict(width=size, edgecolor='w'), startangle=90)

    ax.set(aspect="equal")
    ax.set_title('Intra- vs. Intergenic Peaks', fontsize=24)

    ax.legend(wedges, labels,
              title="Regions",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), frameon=False)