import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scMethtools._utils import savefig
from scanpy.plotting import _utils 
from ..get import rank_genes_groups_df
from typing import List, Optional, Union


def set_colors(adata, key, colors=None, palette=None, n_colors=None):
    """
    Set or modify color palette in adata.uns for categorical variables.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to modify.
    key : str
        The name of the categorical variable in adata.obs.
    colors : list, optional
        List of colors to use. If None, will use palette or default.
    palette : str, optional
        Name of the matplotlib/seaborn color palette to use.
    n_colors : int, optional
        Number of colors to generate. If None, will use the number of categories.
        
    Returns
    -------
    adata : AnnData
        The modified AnnData object.
        
    Examples
    --------
    >>> import scMethtools as scm
    >>> # Set colors for 'cell_type' using a list of colors
    >>> adata = scm.pl.set_colors(adata, 'cell_type', colors=['#FF0000', '#00FF00', '#0000FF'])
    >>> 
    >>> # Set colors for 'treatment' using a seaborn palette
    >>> adata = scm.pl.set_colors(adata, 'treatment', palette='Set2')
    >>> 
    >>> # Set colors for 'condition' with a specific number of colors
    >>> adata = scm.pl.set_colors(adata, 'condition', palette='viridis', n_colors=5)
    """
    # 确保key存在于adata.obs中
    if key not in adata.obs.columns:
        raise ValueError(f"'{key}' not found in adata.obs")
    
    # 确保分类变量是分类类型
    if not pd.api.types.is_categorical_dtype(adata.obs[key]):
        adata.obs[key] = adata.obs[key].astype('category')
    
    # 获取类别数量
    categories = adata.obs[key].cat.categories
    n_cats = len(categories)
    
    # 如果没有指定n_colors，使用类别数量
    if n_colors is None:
        n_colors = n_cats
    
    # 生成颜色
    if colors is not None:
        # 确保颜色数量足够
        if len(colors) < n_cats:
            # 如果颜色不够，循环使用
            colors = colors * (n_cats // len(colors) + 1)
        color_list = colors[:n_cats]
    elif palette is not None:
        # 使用指定的调色板
        try:
            color_list = sns.color_palette(palette, n_colors)
        except ValueError:
            # 如果调色板名称无效，使用默认调色板
            print(f"Invalid palette name '{palette}'. Using default palette.")
            color_list = sns.color_palette("tab10", n_colors)
    else:
        # 使用默认调色板
        if n_cats > 10:
            color_list = sns.color_palette("tab20", n_cats)
        else:
            color_list = sns.color_palette("tab10", n_cats)
    
    # 将颜色列表转换为numpy数组并存储在adata.uns中
    color_key = f"{key}_colors"
    adata.uns[color_key] = np.array(color_list)
    
    return adata

def get_colors_from_adata(adata, key, colors=None):
    """
    Get colors for a categorical variable from adata.uns or use provided colors.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data.
    key : str
        The name of the categorical variable.
    colors : list, optional
        List of colors to use. If provided, these colors will be used instead of
        those in adata.uns.
        
    Returns
    -------
    list
        List of colors to use for plotting.
    """
    color_key = f"{key}_colors"
    
    # 如果提供了颜色，优先使用
    if colors is not None:
        return colors
    
    # 检查adata.uns中是否有颜色定义
    if color_key in adata.uns:
        print(f"Get colors from adata.uns{color_key} ...")
        return adata.uns[color_key]
    
    # 如果没有定义颜色，返回None（使用默认颜色）
    return None

def grouped_value_boxplot(adata, color_by, value_column, colors=None,
                          figsize=None, fontsize=None, dpi=None,
                          show=None, save=None, title=None,
                          median_labels=True, median_fmt='.2f',
                          jitter=True, jitter_alpha=0.5, jitter_size=4,
                          box_width=0.6, theme='ticks', palette=None,
                          **kwargs):
    """
    Generate a boxplot grouped by a categorical variable.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing data and grouping information.
    color_by : str
        Column name for grouping.
    value_column : str
        Column name for the numeric values to plot.
    colors : list, optional
        List of colors for different groups.
    figsize : tuple, optional
        Figure size, if None use global settings.
    fontsize : int, optional
        Font size, if None use global settings.
    dpi : int, optional
        Figure resolution, if None use global settings.
    show : bool, optional
        Whether to show the plot, if None use global settings.
    save : str, optional
        Filename to save the plot, default None.
    title : str, optional
        Custom title for the plot. If None, a default title is generated.
    median_labels : bool, optional
        Whether to show median value labels, default True.
    median_fmt : str, optional
        Format string for median labels, default '.2f'.
    jitter : bool, optional
        Whether to add jittered points to the boxplot, default True.
    jitter_alpha : float, optional
        Alpha value for jittered points, default 0.5.
    jitter_size : int, optional
        Size of jittered points, default 4.
    box_width : float, optional
        Width of the boxes, default 0.6.
    theme : str, optional
        Seaborn theme to use, default 'ticks'.
    palette : str or list, optional
        Color palette to use, overrides colors if provided.
    **kwargs : dict
        Additional parameters passed to sns.boxplot.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.

    Examples
    --------
    >>> import scMethtools as scm
    >>> # Basic usage with default parameters
    >>> scm.pl.grouped_value_boxplot(adata, 'cell_type', 'gene_expression')
    >>> 
    >>> # Add jittered points with custom appearance
    >>> scm.pl.grouped_value_boxplot(adata, 'cell_type', 'gene_expression', 
    ...                             jitter=True, jitter_alpha=0.7, jitter_size=3)
    >>> 
    >>> # Use a custom color palette
    >>> scm.pl.grouped_value_boxplot(adata, 'cell_type', 'gene_expression',
    ...                             palette='viridis')
    """
    # Use provided parameters or global settings
    _figsize = figsize if figsize is not None else plt.rcParams["figure.figsize"]
    _fontsize = fontsize if fontsize is not None else plt.rcParams["font.size"]
    _dpi = dpi if dpi is not None else plt.rcParams["figure.dpi"]
    
    # Set seaborn style
    with sns.axes_style(theme):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=_figsize, dpi=_dpi)
        
        # Get unique categories and their count
        categories = adata.obs[color_by].unique()
        n_categories = len(categories)
        
        # Determine colors/palette
        if palette is not None:
            # Use provided palette name or list
            color_palette = palette
        elif colors is not None:
            # If colors is a list, create a dictionary mapping categories to colors
            if isinstance(colors, list):
                # Make sure we have enough colors
                if len(colors) < n_categories:
                    # Extend the color list if needed
                    colors = colors * (n_categories // len(colors) + 1)
                # Create a dictionary mapping categories to colors
                color_palette = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
            else:
                color_palette = colors
        else:
            color_palette = get_colors_from_adata(adata, color_by)
            if color_palette is None:
                # Use default colormap
                if n_categories > 10:
                    color_palette = sns.color_palette("tab20", n_categories)
                else:
                    color_palette = sns.color_palette("tab10", n_categories)
        
        # Draw boxplot
        boxplot = sns.boxplot(x=color_by, y=value_column, data=adata.obs, 
                   palette=color_palette, ax=ax, width=box_width, **kwargs)
        
        # Add jittered points if requested
        if jitter:
            sns.stripplot(x=color_by, y=value_column, data=adata.obs,
                         palette=color_palette, ax=ax, size=jitter_size, 
                         alpha=jitter_alpha, jitter=True, dodge=True)
        
        # Add median values if requested
        if median_labels:
            medians = adata.obs.groupby(color_by)[value_column].median()
            xtick_labels = list(categories)
            
            for xtick, median in zip(ax.get_xticks(), medians):
                ax.text(xtick, median, f'{median:{median_fmt}}', 
                       ha='center', va='bottom', color='k', 
                       fontsize=_fontsize*0.8, fontweight='bold')
        
        # Set x-ticks and labels
        ax.set_xticks(np.arange(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, fontsize=_fontsize)
        ax.tick_params(axis='y', labelsize=_fontsize)
        
        # Set title and labels
        if title is None:
            title = f'Grouped Boxplot of {value_column} by {color_by}'
        ax.set_title(title, fontsize=_fontsize*1.1, fontweight='bold')
        ax.set_xlabel(color_by, fontsize=_fontsize)
        ax.set_ylabel(value_column, fontsize=_fontsize)
        
        # Remove top and right spines
        sns.despine(ax=ax)
        
        # Save and show figure
        _utils.savefig_or_show("boxplot", show=show, save=save)   
        return fig, ax

def stacked_plot(adata,
                 groupby,
                 colorby,
                 orientation='vertical',
                 ax=None,
                 color=None,
                 dpi=None,
                 figsize=None,
                 fontsize=None, 
                 show=None,
                 save=None,
                 legend_fontsize=None,
                 legend_loc='best',  # 添加图例位置参数
                 legend_bbox_to_anchor=None,  # 添加图例锚点参数
                 **kwargs
                ):
    """
    generate a stacked bar plot of groupby by colorby

    Args:
    adata (AnnData): Annotated data matrix 
    groupby (str): The column name of the data matrix to group by
    colorby (str): The column name of the data matrix to color by
    orientation (str): The orientation of the plot, either 'vertical' or 'horizontal'
    ax (matplotlib.axes.Axes): The axes to plot on
    color (list): The color palette to use for the plot
    figsize (tuple): The size of the figure
    fontsize (int): The fontsize of the labels
    show (bool): Whether to show the plot
    save (str): png pdf or svg file to save the plot,default None
        
    ---------
    stacked_bar(adata,groupby='Cell_type',orientation='horizontal',colorby='Treatment',color=scm.pl.ditto_palette())
    
    """
     # 使用传入的参数或全局设置
    _figsize = figsize if figsize is not None else plt.rcParams["figure.figsize"]
    _fontsize = fontsize if fontsize is not None else plt.rcParams["font.size"]
    _dpi = dpi if dpi is not None else plt.rcParams["figure.dpi"]
     # 获取 obs 数据
    obs = adata.obs  
    # 创建透视表，用于绘制堆积图
    pivot_table = obs.pivot_table(index=groupby, columns=colorby, aggfunc='size', fill_value=0, observed=False)
    #print(pivot_table)
    if color is not None:
        colors = color[:len(pivot_table.columns)]
    else:
        colors = get_colors_from_adata(adata, colorby)

    if ax is None:
        fig, ax = plt.subplots(figsize=_figsize,dpi=_dpi)
    else:
        fig = ax.get_figure()
        
    # 绘制堆积条形图
    if orientation == 'horizontal':
        pivot_table.plot(kind='barh', stacked=True, color=colors, ax=ax)
        ax.set_xlabel('Counts', fontsize=_fontsize)
        ax.set_ylabel(groupby, fontsize=_fontsize)
    else:
        pivot_table.plot(kind='bar', stacked=True, color=colors, ax=ax)
        ax.set_xlabel(groupby, fontsize=_fontsize)
        ax.set_ylabel('Counts', fontsize=_fontsize)
    
    # 设置图形标题和标签
    ax.set_title(f'Stacked Bar Plot of {groupby} by {colorby}', fontsize=_fontsize)
    
    # 设置图例字体大小
    if legend_fontsize is None:
        legend_fontsize = _fontsize * 0.8  # 默认图例字体稍小于主字体
    
    # 设置图例并应用字体大小
    legend = ax.legend(title=colorby)
    plt.setp(legend.get_title(), fontsize=_fontsize)  # 设置图例标题字体大小
    plt.setp(legend.get_texts(), fontsize=legend_fontsize)  # 设置图例文本字体大小

    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)

    # 设置左边和下边的坐标刻度为透明色
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.xaxis.set_tick_params(color='none')
    ax.yaxis.set_tick_params(color='none')
    
    # 显示图形
    # savefig("stacked", save=save, show=show)
    _utils.savefig_or_show("stacked", show=show, save=save)  
def propotion(adata,groupby:str,color_by:str,
                       groupby_list=None,figsize:tuple=(4,6),
                       ticks_fontsize:int=12,labels_fontsize:int=12,ax=None,
                       legend:bool=False):
    """
    绘制堆叠图
    
    """

    b=pd.DataFrame(columns=['cell_type','value','Week'])
    visual_clusters=groupby
    visual_li=groupby_list
    if visual_li==None:
        adata.obs[visual_clusters]=adata.obs[visual_clusters].astype('category')
        visual_li=adata.obs[visual_clusters].cat.categories
    
    for i in visual_li:
        b1=pd.DataFrame()
        test=adata.obs.loc[adata.obs[visual_clusters]==i,color_by].value_counts()
        b1['cell_type']=test.index
        b1['value']=test.values/test.sum()
        b1['Week']=i
        b=pd.concat([b,b1])
    
    plt_data2=adata.obs[color_by].value_counts()
    plot_data2_color_dict=dict(zip(adata.obs[color_by].cat.categories,adata.uns['{}_colors'.format(color_by)]))
    plt_data3=adata.obs[visual_clusters].value_counts()
    plot_data3_color_dict=dict(zip([i.replace('Retinoblastoma_','') for i in adata.obs[visual_clusters].cat.categories],adata.uns['{}_colors'.format(visual_clusters)]))
    b['cell_type_color'] = b['cell_type'].map(plot_data2_color_dict)
    b['stage_color']=b['Week'].map(plot_data3_color_dict)
    
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
    #用ax控制图片
    #sns.set_theme(style="whitegrid")
    #sns.set_theme(style="ticks")
    n=0
    all_celltype=adata.obs[color_by].cat.categories
    for i in all_celltype:
        if n==0:
            test1=b[b['cell_type']==i]
            ax.bar(x=test1['Week'],height=test1['value'],width=0.8,color=list(set(test1['cell_type_color']))[0], label=i)
            bottoms=test1['value'].values
        else:
            test2=b[b['cell_type']==i]
            ax.bar(x=test2['Week'],height=test2['value'],bottom=bottoms,width=0.8,color=list(set(test2['cell_type_color']))[0], label=i)
            test1=test2
            bottoms+=test1['value'].values
        n+=1
    if legend!=False:
        plt.legend(bbox_to_anchor=(1.05, -0.05), loc=3, borderaxespad=0,fontsize=10)
    
    plt.grid(False)
    
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 设置左边和下边的坐标刻度为透明色
    #ax.yaxis.tick_left()
    #ax.xaxis.tick_bottom()
    #ax.xaxis.set_tick_params(color='none')
    #ax.yaxis.set_tick_params(color='none')

    # 设置左边和下边的坐标轴线为独立的线段
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    plt.xticks(fontsize=ticks_fontsize,rotation=90)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel(groupby,fontsize=labels_fontsize)
    # plt.ylabel('Cells per Stage',fontsize=labels_fontsize)
    #fig.tight_layout()
    if ax==None:
        return fig,ax
    
def plot_volcano(adata: ad.AnnData,
    group: Union[str, List[str]],
    key: Optional[str] = "rank_genes_groups",
    log2_fc_col='logfoldchanges', pval_col='pvals', alpha=0.05, lfc_threshold=2.0,point_size=None, 
    color_palette=None, show_name=True,figsize=None, fontsize=None, dpi=None,
                          show=None, save=None, title=None, **kwargs):
    """
    Generate a volcano plot for the differential analysis results.
    Parameters:
    ----------
    adata (ad.AnnData): AnnData object containing the differential analysis results.
    group (str or list): Group to plot results for.
    key (str): Key in adata.uns containing the differential analysis results.
    log2_fc_col (str): Column name for log2 fold change.
    pval_col (str): Column name for p-value.
    alpha (float): Significance threshold for p-value.
    lfc_threshold (float): Threshold for log2 fold change to consider a gene significantly differentially methylated.
    
    Examples
    --------
    >>> import scMethtools as scm
    >>> # Basic usage with default parameters
    >>> scm.pl.plot_volcano(adata, group=None)
    >>> plot_volcano(adata,group='0')
    #filter
    >>> plot_volcano(adata,group='0',show_name=False,log2fc_max=10)

    """
        # Use provided parameters or global settings
    _figsize = figsize if figsize is not None else plt.rcParams["figure.figsize"]
    _fontsize = fontsize if fontsize is not None else plt.rcParams["font.size"]
    _dpi = dpi if dpi is not None else plt.rcParams["figure.dpi"]
    _title_fontsize= _fontsize * 1.1
    # 默认点大小
    if point_size is None:
        point_size = 30  # 默认大小
    # 默认颜色
    if color_palette is None:
        color_palette = {'hypermethylated': 'red', 'hypomethylated': 'blue', 'nonsignificant': 'gray'}
        
    # Pull dataframe from adata object, and select columns of interest
    results_df = sc.get.rank_genes_groups_df(adata, group=group, key=key,**kwargs)
    # Ensure p-value column is numeric
    results_df[pval_col] = pd.to_numeric(results_df[pval_col], errors='coerce')
    # Calculate -log10(p-value)
    results_df['-log10_pvalue'] = -np.log10(results_df[pval_col])
    # Determine significance and direction of regulation
    results_df['significance'] = (results_df[pval_col] < alpha) & (np.abs(results_df[log2_fc_col]) > lfc_threshold)
    results_df['regulation'] = ['hypermethylated' if lfc > 0 and sig else 'hypomethylated' if lfc < 0 and sig else 'nonsignificant'
                                for lfc, sig in zip(results_df[log2_fc_col], results_df['significance'])]
    
    # Get the top significantly expressed genes
    top_sig_genes = results_df[results_df['significance']].nsmallest(10, 'pvals_adj')['names'].tolist()
    if len(top_sig_genes) < 10:
        top_sig_gene_labels = top_sig_genes
    else:
        top_sig_gene_labels = [f'{gene_name[:15]}...' for gene_name in top_sig_genes]
    
    # Create the volcano plot
    plt.figure(figsize=_figsize, dpi=int(_dpi))
    sns.scatterplot(x=log2_fc_col, y='-log10_pvalue', data=results_df,
                    hue='regulation', palette=color_palette,
                    alpha=0.4,s=point_size)
    plt.axhline(-np.log10(alpha), ls='--', color='black', lw=0.5)
    plt.axvline(lfc_threshold, ls='--', color='black', lw=0.5)
    plt.axvline(-lfc_threshold, ls='--', color='black', lw=0.5)
    plt.xlabel('Log2Fold Change',fontsize=_fontsize)
    plt.ylabel('-Log10 p-value',fontsize=_fontsize)
    #plt.title('Volcano Plot of Differential Methylation',fontsize=_title_fontsize)
    plt.legend(title='Regulation', loc='upper left',bbox_to_anchor=(1, 1), fontsize=_fontsize*0.8)
    
    if show_name:
        # Add gene names for top significantly features
        for i, gene_name in enumerate(top_sig_gene_labels):
            plt.annotate(gene_name, (results_df.loc[results_df['names'] == top_sig_genes[i], log2_fc_col].values[0],
                                    results_df.loc[results_df['names'] == top_sig_genes[i], '-log10_pvalue'].values[0]),
                        fontsize=8)
    
    plt.show()