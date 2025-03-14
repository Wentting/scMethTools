#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年10月22日

"""
from ._scatter import embedding
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scanpy as sc


def plot_marker(adata,basis="umap",top=10,key_added='rank_genes_groups',cluster=None, ncols=5, axes_size=3, dpi=150,**kwargs):
    
    groups = list(adata.uns[key_added]["names"].dtype.names)
    method = adata.uns[key_added]["params"]["method"]
    if cluster is None:
        raise ValueError("The 'cluster' parameter must be specified.")
    # 检查cluster是否为字符串
    if not isinstance(cluster, str):
        raise ValueError("The 'cluster' parameter must be a string.")
    if cluster not in groups:
        raise ValueError(f"cluster '{cluster}' should be one of: {groups}")
    # figure
    dmr_data = adata.uns[key_added]['names'][cluster]
    dmr_list = dmr_data[:top]
    nrows = dmr_list.shape[0]  // ncols + 1
    if nrows <= 1:
        nrows = 1  # 确保至少有一行
    fig = plt.figure(figsize=(ncols * axes_size, nrows * axes_size), dpi=dpi)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    # print("nrows:",nrows,ncols)
    # gene axes
    for i, item in enumerate(dmr_list):
        col = i % ncols
        row = i // ncols
        ax = fig.add_subplot(gs[row, col])
    
        is_first_col = (col == 0)
        is_last_row = (row == nrows - 1)
    
        if is_first_col and is_last_row:
            frameon = 'small'
        else:
            frameon = False        
        # print(item)
        embedding(adata,basis=basis,ax=ax,frameon=frameon, color=item,size=50,show=False,wspace=0.1)
        ax.set_title(f'{item}',fontsize=8)
    fig.suptitle(f'{cluster} Top {top} Markers') 
    fig.subplots_adjust(top=0.9)  #设置顶部间距为 90%，可以根据需要调整具体数值
    return None

def plot_marker_violin(adata,top=20,groupby=None,key_added='rank_genes_groups', figsize=None, fontsize=None,dpi=None, show=None, save=None, **kwargs):
    
    _figsize = figsize if figsize is not None else plt.rcParams["figure.figsize"]
    _fontsize = fontsize if fontsize is not None else plt.rcParams["font.size"]
    _dpi = dpi if dpi is not None else plt.rcParams["figure.dpi"]
    
    # Extract the top marker genes for each group
    try:
        var_names = []
        gene_dict = adata.uns[key_added]['names']  # Dictionary of ranked gene names
        for group in gene_dict:  # Iterate over all groups
            var_names.extend(gene_dict[group][:top])  # Select top N genes per group
        var_names = list(set(var_names))  # Remove duplicates
    except Exception as e:
        raise ValueError(f"Error while retrieving gene names: {e}")

    # Plot violin plots for the selected genes
    sc.pl.rank_genes_groups_violin(
        adata, n_genes=top, groupby=groupby, key=key_added, var_names=var_names, 
        use_raw=True, log=False, figsize=_figsize, show=show, save=save, **kwargs
    )

    return None