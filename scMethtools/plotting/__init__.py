#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月22日

"""

from ._scatter import embedding,pca,tsne,umap
from ._palette import palette,red_palette,blue_palette,green_palette,purple_palette,ditto_palette,zeileis_palette
from .dendrogram import dendrogram
from .marker import plot_marker
from ._profile import profile
from .basic import grouped_value_boxplot,stacked_plot