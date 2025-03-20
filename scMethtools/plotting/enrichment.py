from itertools import zip_longest
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, ColorConverter
from matplotlib import cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from math import ceil
from pandas.core.arrays.categorical import Categorical
from scMethtools import _utils

def map_colors(ax, c, palette, add_legend = True, hue_order = None, na_color = 'lightgrey',
        legend_kwargs = {}, cbar_kwargs = {}, vmin = None, vmax = None, log = False,
        normalizer = Normalize):

    assert(isinstance(c, (np.ndarray, list, Categorical)))

    if isinstance(c, Categorical):
        c = c.astype(str)
    
    if isinstance(c, list):
        c = np.array(c)
    c = np.ravel(c)

    if log:
        c = np.log1p(c)

    if np.issubdtype(c.dtype, np.number):
        
        na_mask = np.isnan(c)

        colormapper=cm.ScalarMappable(normalizer(
            np.nanmin(c) if vmin is None else vmin,
            np.nanmax(c) if vmax is None else vmax), 
            cmap=palette)
        c = colormapper.to_rgba(c)

        if na_mask.sum() > 0:
            c[na_mask] = ColorConverter().to_rgba(na_color)

        if add_legend:
            plt.colorbar(colormapper, ax=ax, **cbar_kwargs)

        return c

    else:
        na_mask = c == 'nan'
        
        classes = list(
            dict(zip(c, range(len(c)))).keys()
        )[::-1] #set, order preserved

        if isinstance(palette, (list, np.ndarray)):
            num_colors = len(palette)
            palette_obj = lambda i : np.array(palette)[i]
        else:
            palette_obj = cm.get_cmap(palette)
            num_colors = len(palette_obj.colors)

        if num_colors > 24:
            color_scaler = (num_colors-1)/(len(classes)-1)

            color_wheel = palette_obj(
                (color_scaler * np.arange(len(classes))).astype(int) % num_colors
            )
        else:
            color_wheel =palette_obj(np.arange(len(classes)) % num_colors)
        
        if hue_order is None:
            class_colors = dict(zip(classes, color_wheel))
        else:
            assert(len(hue_order) == len(classes))
            class_colors = dict(zip(hue_order, color_wheel))

        c = np.array([class_colors[c_class] for c_class in c])
        
        if na_mask.sum() > 0:
            c[na_mask] = ColorConverter().to_rgba(na_color)
        
        if add_legend:
            ax.legend(handles = [
                Patch(color = color, label = str(c_class)) for c_class, color in class_colors.items() if not c_class == 'nan'
            ], **legend_kwargs)

        return c

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def compact_string(x, max_wordlen = 4, join_spacer = ' ', sep = ' ', label_genes = []):
    return '\n'.join(
        [
            join_spacer.join([x + ('*' if x in label_genes else '') for x in segment if not x == '']) for segment in grouper(x.split(sep), max_wordlen, fillvalue='')
        ]
    )

def plot_enrichment(
    enrich_result,
    ax=None,
    color_by_adj = True,
    label_genes = [], #可以用来设置特别标注的基因名
    pval_threshold = 1e-5,
    palette = 'Reds',
    gene_fontsize=10,
    show_top = 5,
    show_genes = True,
    max_genes = 10,
    text_color = 'black',
    barcolor = 'lightgrey',
    figsize = None,
    fontsize = None,
    dpi = None,
    show=None,
    save=None,
    **kwargs
    
):
    """
    Plot enrichment results from a enrichment analysis (Enrichr).
    Parameters
    ----------
    go_results : pd.DataFrame
        Results from a enrichment analysis (Enrichr).
    ax : matplotlib.Axes, optional
        The axes object to plot on. If None, a new figure and axes is created.
    color_by_adj : bool, optional
        Whether to color bars by adjusted p-value.
    label_genes : list, optional
        List of gene names to label.
    pval_threshold : float, optional
        P-value threshold for coloring.
    palette : str or list, optional
        Color palette to use.
    gene_fontsize : int, optional
        Font size for gene labels.
        
    show_top : int, optional
        Number of top terms to show.
    show_genes : bool, optional
        Whether to show gene labels.
    max_genes : int, optional
        Maximum number of genes to show.
    text_color : str, optional
        Color for gene labels.
    barcolor : str, optional
        Color for bars.
    figsize : tuple, optional
        Figure size.
    fontsize : int, optional
        Font size.
    dpi : int, optional
        Resolution of the figure.
    show : bool, optional
        Whether to show the figure.
    save : bool, optional
        Whether to save the figure.
    kwargs : dict
        Additional arguments to pass to `plt.savefig`. 
        
    ----
    go_result : Index(['Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value',
       'Odds Ratio', 'Combined Score', 'Genes'],
      dtype='object')
    """
     # 使用传入的参数或全局设置
    _figsize = figsize if figsize is not None else plt.rcParams["figure.figsize"]
    _fontsize = fontsize if fontsize is not None else plt.rcParams["font.size"]
    _dpi = dpi if dpi is not None else plt.rcParams["figure.dpi"]
    
    # df = go_results.results
    df = enrich_result
    results = df[df['P-value'] < 0.05]

    if ax is None:
        fig, ax = plt.subplots(figsize=_figsize,dpi=_dpi)
    else:
        fig = ax.get_figure()

    terms, genes, pvals, adj_pvals = [],[],[],[]
    # 计算 -log10 调整后 P 值
    for _, result in results[:show_top].iterrows():  # 使用 iterrows() 逐行遍历
        # 确保 'Term' 存在
        terms.append(result['Term'])
        genes_list = result['Genes'].split(';') if isinstance(result['Genes'], str) else result['Genes']
        genes.append(' '.join(genes_list[:max_genes]) if genes_list else '')
        pvals.append(-np.log10(result['P-value']))
        adj_pvals.append(-np.log10(result['Adjusted P-value']))


    if color_by_adj:
        edgecolor = map_colors(ax, np.array(adj_pvals), palette, add_legend = True, 
            cbar_kwargs = dict(
                    location = 'right', pad = 0.1, shrink = 0.5, aspect = 15, label = '-log10 Adj P-value',
                ), vmin = 0, vmax = -np.log10(pval_threshold))
            
        ax.barh(np.arange(len(terms)), pvals, edgecolor = edgecolor, color = barcolor, linewidth = 2)
    else:
        ax.barh(np.arange(len(terms)), pvals, color = barcolor)

    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set(title = '', xlabel = '-log10 pvalue')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if show_genes:
        for j, p in enumerate(ax.patches):
            print 
            _y = p.get_y() + p.get_height() - p.get_height()/3
            ax.text(0.1, _y, compact_string(genes[j], max_wordlen=10, join_spacer = ', ', label_genes = label_genes), 
                ha="left", color = text_color, fontsize = gene_fontsize)
    _utils.savefig("enrichment", show=show, save=save,dpi=_dpi)
    
    plt.show()