#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月21日

"""
import os
import psutil
import pandas as pd
import numpy as np
import anndata as ad
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype,is_numeric_dtype
from matplotlib import patheffects, rcParams
from scMethtools import settings
from scMethtools import logging as logg

# from snapatac2._snapatac2 import AnnData, AnnDataSet, read

def set_figure_params(context='notebook',style='white',palette='deep',font='sans-serif',font_scale=1.1,color_codes=True,
                      dpi=80,dpi_save=150,figsize=[5.4, 4.8],rc=None):
    """ Set global parameters for figures. Modified from sns.set()
    Parameters
    ----------
    context : string or dict
        Plotting context parameters, see seaborn :func:`plotting_context
    style: `string`,optional (default: 'white')
        Axes style parameters, see seaborn :func:`axes_style`
    palette : string or sequence
        Color palette, see seaborn :func:`color_palette`
    font_scale: `float`, optional (default: 1.3)
        Separate scaling factor to independently scale the size of the font elements.        
    color_codes : `bool`, optional (default: True)
        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand
        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.
    dpi: `int`,optional (default: 80)
        Resolution of rendered figures.
    dpi_save: `int`,optional (default: 150)
        Resolution of saved figures.
    rc: `dict`,optional (default: None)
        rc settings properties. Parameter mappings to override the values in the preset style.
        Please see https://matplotlib.org/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
    """
#     mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_theme(context=context,style=style,palette=palette,font=font,font_scale=font_scale,color_codes=color_codes,
            rc={'figure.dpi':dpi,
                'savefig.dpi':dpi_save,
                'figure.figsize':figsize,
                'image.cmap': 'viridis',
                'lines.markersize':6,
                'legend.columnspacing':0.1,
                'legend.borderaxespad':0.1,
                'legend.handletextpad':0.1,
                'pdf.fonttype':42,})
    if(rc is not None):
        assert isinstance(rc,dict),"rc must be dict"  
        for key, value in rc.items():
            if key in plt.rcParams.keys():
                plt.rcParams[key] = value
            else:
                raise Exception("unrecognized property '%s'" % key)
            
def get_colors(adata,ann):
    df_cell_colors = pd.DataFrame(index=adata.obs.index)
    if(is_numeric_dtype(adata.obs[ann])):
        cm = mpl.cm.get_cmap()
        norm = mpl.colors.Normalize(vmin=0, vmax=max(adata.obs[ann]),clip=True)
        df_cell_colors[ann+'_color'] = [mpl.colors.to_hex(cm(norm(x))) for x in adata.obs[ann]]
    else:
        if(ann+'_color' not in adata.uns_keys()):  
            ### a hacky way to generate colors from seaborn
            tmp = pd.DataFrame(index=adata.obs_names,
                   data=np.random.rand(adata.shape[0], 2))
            tmp[ann] = adata.obs[ann]
            fig = plt.figure()
            ax_i = fig.add_subplot(1,1,1)
            sc_i=sns.scatterplot(ax=ax_i,x=0, y=1,hue=ann,data=tmp,linewidth=0)             
            colors_sns = sc_i.get_children()[0].get_facecolors()
            colors_sns_scaled = (255*colors_sns).astype(int)
            ax_i.remove()
            adata.uns[ann+'_color'] = {tmp[ann][i]:'#%02x%02x%02x' % (colors_sns_scaled[i][0], colors_sns_scaled[i][1], colors_sns_scaled[i][2])
                                       for i in np.unique(tmp[ann],return_index=True)[1]}            
        dict_color = adata.uns[ann+'_color']
        df_cell_colors[ann+'_color'] = ''
        for x in dict_color.keys():
            id_cells = np.where(adata.obs[ann]==x)[0]
            df_cell_colors.loc[df_cell_colors.index[id_cells],ann+'_color'] = dict_color[x]
    return(df_cell_colors[ann+'_color'].tolist())

def HowManyTime(tbegin,tend):
    """
    to calculate the time to evaluate the speed
    """
    tTotal=tend-tbegin
    tsec=tTotal%60
    ttolmin=tTotal//60
    thour=ttolmin//60
    tmin=ttolmin%60
    suretime="running time is %d hour, %d minutes, %.2f seconds"%(thour,tmin,tsec)
    return suretime

def _memory_usage_psutil():
    """
    获取当前进程的内存使用情况，单位为MB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)

def binary_beta(beta_value,cut_ceil,cut_off):
    if beta_value > cut_ceil:
        beta_value=1
    elif beta_value <= cut_off:
        beta_value=0
    else:
        return -1
    return beta_value

def get_igraph_from_adjacency(adj):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    vcount = max(adj.shape)
    sources, targets = adj.nonzero()
    edgelist = list(zip(list(sources), list(targets)))
    weights = np.ravel(adj[(sources, targets)])
    gr = ig.Graph(n=vcount, edges=edgelist, directed=False, edge_attrs={"weight": weights})
    return gr


def chunks(mat, chunk_size: int):
    """
    Return chunks of the input matrix
    """
    n = mat.shape[0]
    for i in range(0, n, chunk_size):
        j = max(i + chunk_size, n)
        yield mat[i:j, :]


def find_elbow(x, saturation=0.01):
    accum_gap = 0
    for i in range(1, len(x)):
        gap = x[i - 1] - x[i]
        accum_gap = accum_gap + gap
        if gap < saturation * accum_gap:
            return i
    return None


def fetch_seq(fasta, region):
    chr, x = region.split(':')
    start, end = x.split('-')
    start = int(start)
    end = int(end)
    seq = fasta[chr][start:end].seq
    l1 = len(seq)
    l2 = end - start
    if l1 != l2:
        raise NameError(
            "sequence fetch error: expected length: {}, but got {}.".format(l2, l1)
        )
    else:
        return seq


def pcorr(A, B):
    """Compute pairwsie correlation between two matrices.

    A
        n_sample x n_feature
    B
        n_sample x n_feature
    """
    N = B.shape[0]

    sA = A.sum(0)
    sB = B.sum(0)

    p1 = N * np.einsum('ij,ik->kj', A, B)
    p2 = sA * sB[:, None]
    p3 = N * ((B ** 2).sum(0)) - (sB ** 2)
    p4 = N * ((A ** 2).sum(0)) - (sA ** 2)

    return (p1 - p2) / np.sqrt(p4 * p3[:, None])


def savefig(
    writekey=str,
    show= None,
    dpi=None,
    ext= None,
    save=None,
):
    """

    Parameters:
    -----------
    writekey: `str`
        The name of the file to save the figure to.
    show: `bool`, optional (default: None)
        Whether to display the figure.
    dpi: `int`, optional (default: None)
        The resolution of the figure in dots per inch.  
    ext: `str`, optional (default: None)
        The file extension of the figure. 
    save: `bool`, optional (default: None)
        Whether to save the figure.
    ____________________________________________________________
    savefig(dpi=dpi, save=save, show=show)
    """
    if isinstance(save, str):
        # check whether `save` contains a figure extension
        if ext is None:
            for try_ext in [".svg", ".pdf", ".png"]:
                if save.endswith(try_ext):
                    ext = try_ext[1:]
                    save = save.replace(try_ext, "")
                    break
        # append it
        writekey += save
        save = True
    save = settings.autosave if save is None else save
    show = settings.autoshow if show is None else show
    if save:
        if dpi is None:
            # needed in nb b/c internal figures are also influenced by 'savefig.dpi'.
            if (
                not isinstance(rcParams["savefig.dpi"], str)
                and rcParams["savefig.dpi"] < 150
            ):
                if settings._low_resolution_warning:
                    logg.warn(
                        "You are using a low resolution (dpi<150) for saving figures.\n"
                        "Consider running `set_figure_params(dpi_save=...)`, which "
                        "will adjust `matplotlib.rcParams['savefig.dpi']`"
                    )
                    settings._low_resolution_warning = False
            else:
                dpi = rcParams["savefig.dpi"]
        if len(settings.figdir) > 0:
            if settings.figdir[-1] != "/":
                settings.figdir += "/"
            if not os.path.exists(settings.figdir):
                os.makedirs(settings.figdir)
        if ext is None:
            ext = settings.file_format_figs
        filepath = f"{settings.figdir}{settings.plot_prefix}{writekey}"
        #print('writekey',writekey)
        if "/" in writekey:
            filepath = f"{writekey}"
        try:
            #plot_suffix = scm
            filename = filepath + f"{settings.plot_suffix}.{ext}"
            plt.savefig(filename, dpi=dpi,bbox_inches='tight')
        except ValueError as e:
            # save as .png if .pdf is not feasible (e.g. specific streamplots)
            filename = filepath + f"{settings.plot_suffix}.png"
            plt.savefig(filename, dpi=dpi,bbox_inches='tight')
            logg_message = (
                f"figure cannot be saved as {ext}, using png instead "
                f"({e.__str__().lower()})."
            )
            logg.msg(logg_message, v=1)
        logg.msg("saving figure to file", filename, v=1)
        print(f"Fig saved at {filename}")
    if show:
        plt.show()
    if save:
        plt.close()  # clear figure
        

def savefig(
    writekey: str = "",
    show: bool = None,
    dpi: int = None,
    save: bool = None,
):
    """
    保存 Matplotlib 生成的图像，并根据参数决定是否显示。

    参数:
    -----------
    writekey: `str`
        要保存的文件名（不带扩展名）。
    show: `bool`, 可选 (默认: None)
        是否显示图像。
    dpi: `int`, 可选 (默认: None)
        分辨率（每英寸点数）。
    ext: `str`, 可选 (默认: None)
        需要保存的文件扩展名（如 "png", "pdf", "svg"）。
    save: `bool`, 可选 (默认: None)
        是否保存图像。

    """
    # 确定是否保存和显示
    save = settings.autosave if save is None else save
    show = settings.autoshow if show is None else show
    
    # 确定保存路径和文件名
    if "/" in writekey or "\\" in writekey:  # 如果 writekey 是完整路径
        filepath = writekey
        directory = os.path.dirname(filepath)
    else:  # 否则，使用默认的 settings.figdir
        directory = settings.figdir.rstrip("/")
        filepath = os.path.join(directory, writekey)
    
    # 自动识别扩展名
    ext = None
    for try_ext in [".svg", ".pdf", ".png"]:
        if filepath.endswith(try_ext):
            ext = try_ext[1:]  # 去掉前导 `.`
            filepath = filepath.rsplit(".", 1)[0]  # 去掉扩展名
            break
    if ext is None:
        ext = settings.file_format_figs  # 默认格式
        
    # 确保目标文件夹存在
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 最终文件路径
    final_filepath = f"{filepath}{settings.plot_suffix}.{ext}"    
    
    if dpi is None:
        dpi = rcParams["savefig.dpi"]
      # 保存图像
    try:
        plt.savefig(final_filepath, dpi=dpi, bbox_inches='tight')
    except ValueError:
        # 若 .pdf 失败，降级为 .png
        final_filepath = final_filepath.rsplit(".", 1)[0] + ".png"
        plt.savefig(final_filepath, dpi=dpi, bbox_inches='tight')
    
    print(f"Figure save at: {final_filepath}")
    # 显示或关闭图像
    if show:
        plt.show()
    if save:
        plt.close()  # 关闭图像，避免过多窗口累积
    plt.close()  