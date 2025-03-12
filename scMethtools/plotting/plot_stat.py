#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月22日

"""
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scMethtools._utils import savefig
from ._palette import *

color = ["#f9766e","#fbbab6","#e1c548","#f0e2a3","#5fa664","#abd0a7","#ca6a6b","#e5b5b5","#4e79a6","#bac4d0","#45337f","#a199be","#aedd2f","#d7ee96"]

def joint_scatter(data,x,y,color='grey',hue=None,ax=None):
    sns.jointplot(
        x=x,
        y=y,
        data=data,
        hue=hue,
        s=20,
        color='grey',
        ax=ax
    )   
def hist_cutoff_pct(data,x,bins=20, color='grey',density=False):
    data=data[x]
    ax = plt.subplot() 
    # 创建直方图
    n, bins, patches = plt.hist(data, bins=bins,color=color,density=density)
    # 计算不同截断值下的保留百分比
    cutoffs = np.sort(data)
    percentiles = 1- np.arange(1, len(cutoffs) + 1) / len(cutoffs)
    # 将百分数形式转换为小数并保留两位小数
    percentiles_decimal = [round(100 * p, 2) for p in percentiles]
    # 添加右侧百分比y轴
    ax2 = plt.twinx()
    # adding horizontal grid lines 
    ax2.yaxis.grid(True) 
    plt.ylabel('passed %')
    # 绘制保留百分比曲线
    ax2 = plt.plot(cutoffs,percentiles_decimal, 'r--', label= 'passed percent')
     # 添加标签和标题
    plt.ylabel(x)
    # 设置图例
    plt.legend(loc='upper right')
    
    # 显示图形
    plt.show()

def cat_plot(df,x,y,hue=None, palette=None, kind='box',save=False, show=True, writekey="catgory_boxplot" ,height=5, aspect=2, **kwargs):
    """
    plot catplot
    Args:
    df: pandas DataFrame
    x: str, column name of df
    y: str, column name of df
    hue: str, column name of df
    palette: str, color palette
    kind: str, kind of plot | box, violin, strip: scatter, swarm,  bar ,etc
    height: int, height of plot
    aspect: int, aspect ratio of plot
    kwargs: additional keyword arguments
    ------
    https://seaborn.pydata.org/generated/seaborn.catplot.html
    -----
    example:
    cat_plot(df_obs, x='Cell_type', y='global_meth_level', hue='Treatment', kind='violin', height=5, aspect=2)
    """
    plt.figure(dpi=250)
    sns.set_theme(style="ticks")# 或 "darkgrid", "ticks" 等
    df[hue] = df[hue].fillna('Unknown')
    if palette is None:
        palette = sc_color #_palette.palette()
    if hue is None:
        hue = x
    #sns.catplot(kind='violin', data=df_obs, x="Cell_type", y="global_meth_level", hue="Treatment", height=5, aspect=2)
    sns.catplot(kind=kind, data=df, x=x, y=y, hue=hue, height=height,palette=palette, aspect=aspect, **kwargs)
    if save:
        savefig(writekey=writekey, save=save, show=show)
    #aspect宽高比

def grouped_boxplot(adata, group_by, color_by, value_column):
    plt.figure(figsize=(20, 6))

    # 使用Seaborn的color_palette来自动生成足够多的颜色
    colors = sns.color_palette("Set3", n_colors=len(adata.obs[color_by].unique()))

    print(colors)

    # 使用stripplot绘制数据点
    ax = sns.stripplot(x=group_by, y=value_column, data=adata.obs, palette=colors, jitter=True, hue=color_by)

    # 绘制中值线
    medians = adata.obs.groupby(group_by)[value_column].median()
    xtick_labels = list(adata.obs[group_by].unique())

    for xtick, median in zip(ax.get_xticks(), medians):
        ax.text(xtick, median, f'{median:.2f}', ha='center', va='bottom', color='k', fontsize=10)

    # 倾斜xtick标签
    plt.xticks(rotation=45)

    plt.title(f'Grouped Boxplot of {value_column} by {group_by}')
    plt.xlabel(group_by)
    plt.ylabel(value_column)

    # 添加color_by的图例，位于坐标轴正下方
    ax.legend(title=color_by, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(adata.obs[color_by].unique()))
    plt.show()



def plot_qc(adata,obs_hue=None, save=False, show=True,dpi=None,
    ext= None,writekey="quality_control"):
    feature = 'Regions'
    df_obs = adata.obs
    df_var = adata.var
    cell_num = adata.shape[0]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (16,8))
    axs = axs.flatten()
    # 设置左侧标注
    axs[0].set_ylabel('Cells', rotation=0, size='large', labelpad=50,color='blue')
    axs[2].set_ylabel(feature, rotation=0, size='large', labelpad=50,color='blue')
    # sns.scatterplot(df_obs,x='global_meth_level',y='sites',color='grey',hue=obs_hue,ax=axs[0])
    sns.histplot(df_obs,x='global_meth_level',color='grey',hue=obs_hue,ax=axs[0])
    sns.histplot(data=df_obs, x='sites', color="grey",hue=obs_hue,ax=axs[1])
    sns.histplot(data=df_var, x='covered_cell', color="grey",binwidth=10,ax=axs[2])
    sns.histplot(data=df_var, x='var', color="grey",binwidth=0.1,ax=axs[3])
    fig.tight_layout()
    savefig(writekey=writekey, save=save, show=show,dpi=dpi,ext= ext)
        
def plot_qc_pc(mdata, obs_hue=None, save=False, show=True, writekey="quality_control"):
    """
    Plot quality control metrics for methylation data.
    
    Parameters:
    mdata (MethylationData): MethylationData object containing data to plot
    obs_hue (str): Column name in mdata.obs to use for coloring in plots
    save (str): Whether to save the figure to a file
                if save is a string, it is the path to save the figure
    show (bool): Whether to display the figure
    """
    # Determine the number of modalities
    num_modalities = len(mdata.mod)
    
    # Create figure with two rows: the first one with two columns, the second one with a number equal to the number of modalities
    nrows = 2
    ncols_first_row = 2
    ncols_second_row = num_modalities
    
    # Setting up the figure and axes
    fig = plt.figure(figsize=(16, 8))  # Adjust overall figure size as needed
    
    # Create subplots for the first row
    ax1 = fig.add_subplot(2, max(ncols_first_row, ncols_second_row), 1)
    ax2 = fig.add_subplot(2, max(ncols_first_row, ncols_second_row), 2)
    
    # General plots from mdata.obs
    df_obs = mdata.obs
    sns.scatterplot(data=df_obs, x='global_meth_level', y='sites', color='grey', hue=obs_hue, ax=ax1)
    sns.histplot(data=df_obs, x='sites', color="grey", hue=obs_hue, ax=ax2)
    
    # Loop over modalities for the second row of plots
    for i, (mod_name, mod_data) in enumerate(mdata.mod.items()):
        # Add subplot for each modality in the second row
        ax = fig.add_subplot(2, ncols_second_row, ncols_second_row + i + 1)
        
        # Access the 'var' DataFrame for the current modality
        df_var = mod_data.var
        # Plot a histogram for 'covered_cell' feature of current modality
        sns.histplot(data=df_var, x='covered_cell', color="blue", binwidth=10, ax=ax)
        ax.set_title(f"{mod_name} covered_cell")  # Set title for the subplot
        
    # Adjust layout for better readability
    fig.tight_layout()

    savefig(writekey=writekey, save=save, show=show)
    
    # # Save or show the figure based on user input
    # if save_fig:
    #     if fig_path is not None:
    #         plt.savefig(os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches='tight')
    #     plt.close(fig)  # Close the figure after saving to release memory
    # else:
    #     plt.show()  # Display the figure if not saving
        
      
def density_scatter(data,x,y):
    from scipy import stats
    values = np.vstack([data[x], data[y]]).astype(float)
    kernel = stats.gaussian_kde(values)(values)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        c=kernel,
        cmap="viridis",
        ax=ax,
        size=2
    )

