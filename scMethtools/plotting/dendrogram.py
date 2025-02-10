
import numpy as np
import pandas as pd
from matplotlib import gridspec, patheffects, rcParams
from matplotlib import pyplot as plt
import seaborn as sns
from collections import OrderedDict
from collections.abc import Collection, Iterable, Mapping, Sequence
from itertools import product
from typing import TYPE_CHECKING, Literal, Union
if TYPE_CHECKING:
    from anndata import AnnData
    from cycler import Cycler
    from matplotlib.axes import Axes
    from seaborn import FacetGrid
    from seaborn.matrix import ClusterGrid  
import logging as logg
    
def dendrogram(
                adata,
                groupby,
                dendro_ax =  None,
                dendrogram_key = None,
                orientation:Literal['top',"bottom", "left", "right"] = "top",
                color = 'lightgreen',
                remove_labels: bool = False,
                show = None,
                save = None,
                ax = None,
                ticks = None
              ):
    if dendro_ax is None:
        _, dendro_ax = plt.subplots(dpi=250)

    if not isinstance(dendrogram_key, str):
        if isinstance(groupby, str):
            dendrogram_key = f"dendrogram_{groupby}"
        elif isinstance(groupby, list):
            dendrogram_key = f'dendrogram_{"_".join(groupby)}'
            
    if dendrogram_key not in adata.uns:
        import scanpy as sc
        sc.tl.dendrogram(adata, groupby, key_added=dendrogram_key)
    
    if "dendrogram_info" not in adata.uns[dendrogram_key]:
        raise ValueError(
            f"The given dendrogram key ({dendrogram_key!r}) does not contain "
            "valid dendrogram information."
        )
    
    def translate_pos(pos_list, new_ticks, old_ticks):
        # of given coordinates.

        if not isinstance(old_ticks, list):
            # assume that the list is a numpy array
            old_ticks = old_ticks.tolist()
        new_xs = []
        for x_val in pos_list:
            if x_val in old_ticks:
                new_x_val = new_ticks[old_ticks.index(x_val)]
            else:
                # find smaller and bigger indices
                idx_next = np.searchsorted(old_ticks, x_val, side="left")
                idx_prev = idx_next - 1
                old_min = old_ticks[idx_prev]
                old_max = old_ticks[idx_next]
                new_min = new_ticks[idx_prev]
                new_max = new_ticks[idx_next]
                new_x_val = ((x_val - old_min) / (old_max - old_min)) * (
                    new_max - new_min
                ) + new_min
            new_xs.append(new_x_val)
        return new_xs

    dendro_info = adata.uns[dendrogram_key]["dendrogram_info"]
    leaves = dendro_info["ivl"]
    icoord = np.array(dendro_info["icoord"])
    dcoord = np.array(dendro_info["dcoord"])

    orig_ticks = np.arange(5, len(leaves) * 10 + 5, 10).astype(float)
    # check that ticks has the same length as orig_ticks
    if ticks is not None and len(orig_ticks) != len(ticks):
        logg.warning(
            "ticks argument does not have the same size as orig_ticks. "
            "The argument will be ignored"
        )
        ticks = None

    for xs, ys in zip(icoord, dcoord):
        if ticks is not None:
            xs = translate_pos(xs, ticks, orig_ticks,linewidth=1.5)
        if orientation in ["right", "left"]:
            xs, ys = ys, xs
            print(xs, ys)
            if np.any(np.isclose(xs, 0.0)):  # 使用 np.isclose() 来判断浮点数是否接近零，找到叶节点
                idx = np.where(np.isclose(xs, 0.0))
                print(xs[idx],ys[idx])
                dendro_ax.scatter(xs[idx],ys[idx], clip_on=False,zorder=3,c=color,s=60)
        dendro_ax.plot(xs, ys, color="gray")
        if np.any(np.isclose(ys, 0.0)):  # 使用 np.isclose() 来判断浮点数是否接近零，找到叶节点
            idx = np.where(np.isclose(ys, 0.0))
            print(xs[idx],ys[idx])
            dendro_ax.scatter(xs[idx],ys[idx], clip_on=False,zorder=3,c=color,s=60)

    dendro_ax.tick_params(bottom=False, top=False, left=False, right=False)
    ticks = ticks if ticks is not None else orig_ticks
    dendro_ax.spines["left"].set_visible(False)
    dendro_ax.spines["right"].set_visible(False)
    dendro_ax.spines["top"].set_visible(False)
    dendro_ax.spines["bottom"].set_visible(False)
    if orientation in ["right", "left"]:
        dendro_ax.set_yticks(ticks)
        dendro_ax.set_yticklabels(leaves, fontsize="small", rotation=0)
        dendro_ax.tick_params(labelbottom=True, labeltop=False) #轴线
        dendro_ax.spines["bottom"].set_visible(True)
        xmin, xmax = dendro_ax.get_xlim()
        if orientation == "left":
            dendro_ax.set_xlim(xmax, 0.0)
            dendro_ax.tick_params(labelleft=False, labelright=True)
            dendro_ax.tick_params(bottom=True) #刻度线
        else:
            dendro_ax.set_xlim(0.0, xmax)
            dendro_ax.tick_params(labelleft=True, labelright=False)
            dendro_ax.tick_params(bottom=True)
    else:
        dendro_ax.set_xticks(ticks)
        dendro_ax.set_xticklabels(leaves, fontsize="small", rotation=90) #绘制节点标签
        dendro_ax.tick_params(labelleft=True, labelright=False)
        dendro_ax.spines["left"].set_visible(True)
        if orientation == "bottom":
            ymin, ymax = dendro_ax.get_ylim()
            dendro_ax.set_ylim(ymax, 0.00)
            dendro_ax.tick_params(labeltop=True, labelbottom=False)
            dendro_ax.tick_params(left=True)
        else:
            ymin, ymax = dendro_ax.get_ylim()
            dendro_ax.set_ylim(0.00,ymax)
            dendro_ax.tick_params(labeltop=False, labelbottom=True)
            dendro_ax.tick_params(left=True)

    if remove_labels:
        dendro_ax.tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )
    return dendro_ax
    # dendro_ax.grid(True)