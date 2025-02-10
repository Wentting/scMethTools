import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scMethtools._utils import savefig

def grouped_value_boxplot(adata, color_by, value_column, colors=None):
    """
    绘制分组箱线图

    参数:
    adata (AnnData): 包含数据和分组信息的AnnData对象
    color_by (str): 用于分组的列名
    value_column (str): 要绘制箱线图的数值列名
    colors (list): 可选参数，用于指定颜色的列表
    """
    plt.figure(figsize=(10, 6))
    
    if colors is None:
        # 使用Matplotlib的调色盘为不同分组着色，将其转化为Seaborn颜色列表
        n_colors = len(adata.obs[color_by].unique())
        if n_colors > 10:
            colors = [plt.cm.get_cmap('tab20', n_colors)(i) for i in range(n_colors)]
        else:
            colors = [plt.cm.get_cmap('tab10', n_colors)(i) for i in range(n_colors)]
    
    # 绘制箱线图
    ax = sns.boxplot(x=color_by, y=value_column, data=adata.obs, palette=colors)
    
    # 绘制中线
    medians = adata.obs.groupby(color_by)[value_column].median()
    xtick_labels = list(adata.obs[color_by].unique())
    
    for xtick, median in zip(ax.get_xticks(), medians):
        ax.text(xtick, median, f'{median:.2f}', ha='center', va='bottom', color='k', fontsize=10)
    
    plt.xticks(np.arange(len(xtick_labels)), xtick_labels)
    
    plt.title(f'Grouped Boxplot of {value_column} by {color_by}')
    plt.xlabel(color_by)
    plt.ylabel(value_column)
    plt.show()

def stacked_plot(adata,
                 groupby,
                 colorby,
                 orientation='vertical',
                 ax=None,
                 color=None,
                 figsize=(10,6),
                 fontsize=10, 
                 show=True,
                 save=None):
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
     # 获取 obs 数据
    obs = adata.obs
    # 检查 uns 字典中是否有颜色信息
    if f'{colorby}_colors' in adata.uns:
        colors = adata.uns[f'{colorby}_colors']
        
    # 创建透视表，用于绘制堆积图
    pivot_table = obs.pivot_table(index=groupby, columns=colorby, aggfunc='size', fill_value=0, observed=False)
    #print(pivot_table)
    if color is not None:
        colors = color[:len(pivot_table.columns)]
    else:
        colors = None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize,dpi=200)
    else:
        fig = ax.get_figure()
        
    # 绘制堆积条形图
    if orientation == 'horizontal':
        pivot_table.plot(kind='barh', stacked=True, color=colors, ax=ax)
        ax.set_xlabel('Counts')
        ax.set_ylabel(groupby)
    else:
        pivot_table.plot(kind='bar', stacked=True, color=colors, ax=ax)
        ax.set_xlabel(groupby)
        ax.set_ylabel('Counts')
    
    # 设置图形标题和标签
    ax.set_title(f'Stacked Bar Plot of {groupby} by {colorby}')
    ax.legend(title=colorby)

    # 设置左边和下边的坐标刻度为透明色
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.xaxis.set_tick_params(color='none')
    ax.yaxis.set_tick_params(color='none')
    # 显示图形
    savefig("stacked", save=save, show=show)

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
    
def map_ensembl_to_gene_name(de_results, gene_names_file):
    """
    Map Ensembl IDs to gene names and remove genes with no mapping.

    Parameters
    ----------
    de_results : pd.DataFrame
        DataFrame containing the differential expression analysis results.
    gene_names_file : str
        Path to the file containing the mapping between Ensembl IDs and gene names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Ensembl IDs mapped to gene names and genes with no mapping removed.
    """
    # Read the gene names file
    gene_names = pd.read_csv(gene_names_file, sep='\t', header=None, names=['ensembl_id', 'gene_name'])

    # Create a dictionary for mapping Ensembl IDs to gene names
    id_to_name = dict(zip(gene_names['ensembl_id'], gene_names['gene_name']))

    # Map Ensembl IDs to gene names in the results DataFrame
    de_results['gene_name'] = de_results['gene'].map(id_to_name)

    # Remove genes with no mapping
    de_results = de_results.dropna(subset=['gene_name'])

    return de_results

def plot_volcano(results_df, log2_fc_col='log2_fold_change', pval_col='p_value', alpha=0.05, lfc_threshold=1.0):
    """
    Generate a volcano plot for the differential expression results.
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the differential expression analysis results.
    log2_fc_col (str): Column name for log2 fold change.
    pval_col (str): Column name for p-value.
    alpha (float): Significance threshold for p-value.
    lfc_threshold (float): Threshold for log2 fold change to consider a gene significantly differentially expressed.
    """
    # Ensure p-value column is numeric
    results_df[pval_col] = pd.to_numeric(results_df[pval_col], errors='coerce')
    # Calculate -log10(p-value)
    results_df['-log10_pvalue'] = -np.log10(results_df[pval_col])
    # Determine significance and direction of regulation
    results_df['significance'] = (results_df[pval_col] < alpha) & (np.abs(results_df[log2_fc_col]) > lfc_threshold)
    results_df['regulation'] = ['upregulated' if lfc > 0 and sig else 'downregulated' if lfc < 0 and sig else 'nonsignificant'
                                for lfc, sig in zip(results_df[log2_fc_col], results_df['significance'])]
    
    # Get the top significantly expressed genes
    top_sig_genes = results_df[results_df['significance']].nsmallest(10, 'padj')['gene_name'].tolist()
    if len(top_sig_genes) < 10:
        top_sig_gene_labels = top_sig_genes
    else:
        top_sig_gene_labels = [f'{gene_name[:15]}...' for gene_name in top_sig_genes]
    
    # Create the volcano plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=log2_fc_col, y='-log10_pvalue', data=results_df,
                    hue='regulation', palette={'upregulated': 'red', 'downregulated': 'blue', 'nonsignificant': 'gray'},
                    alpha=0.6)
    plt.axhline(-np.log10(alpha), ls='--', color='black', lw=0.5)
    plt.axvline(lfc_threshold, ls='--', color='black', lw=0.5)
    plt.axvline(-lfc_threshold, ls='--', color='black', lw=0.5)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.title('Volcano Plot of Differential Expression')
    plt.legend(title='Regulation', loc='upper right')
    
    # Add gene names for top significantly expressed genes
    for i, gene_name in enumerate(top_sig_gene_labels):
        plt.annotate(gene_name, (results_df.loc[results_df['gene_name'] == top_sig_genes[i], log2_fc_col].values[0],
                                 results_df.loc[results_df['gene_name'] == top_sig_genes[i], '-log10_pvalue'].values[0]),
                     fontsize=8)
    
    plt.show()

def plot_heatmap(results_df, adata, top_n=20):
    """
    Plot a heatmap of the top differentially expressed genes.
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the differential expression analysis results.
    adata (anndata.AnnData): AnnData object containing the counts matrix and metadata.
    top_n (int): Number of top genes to display in the heatmap.
    """
    # Select the top_n differentially expressed genes by adjusted p-value
    top_genes = results_df.nsmallest(top_n, 'padj')['gene']
    top_genes_data = adata[:, top_genes].X
    
    # Ensure the data is in the correct format
    if not isinstance(top_genes_data, np.ndarray):
        top_genes_data = top_genes_data.toarray()
    
    # Get the corresponding gene names
    top_gene_names = results_df.loc[results_df['gene'].isin(top_genes), 'gene_name'].tolist()
    
    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_genes_data, yticklabels=adata.obs.index, xticklabels=top_gene_names, cmap='RdBu_r', cbar=True)
    plt.title(f'Top {top_n} Differentially Expressed Genes')
    plt.xlabel('Gene Names')
    plt.ylabel('Samples')
    plt.show()
