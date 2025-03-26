#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年10月17日

"""
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import roc_auc_score
import matplotlib.pylab as plt
from multiprocessing import Pool
import seaborn as sns
from PyComplexHeatmap import *
import logging
from ..preprocessing.get_df import dmr_df
from typing import Union, Iterable
from typing_extensions import Literal

top_n = 1000
auroc_cutoff = 0.8
adj_p_cutoff = 0.001
fc_cutoff = 0.8
max_cluster_cells = 2000
max_other_fold = 5
cpu = 10

def cosg(adata,groupby):
    
    import cosg as cosg
    #缺失填充
    adata.X = np.nan_to_num(adata.X)
    import cosg as cosg
    cosg.cosg(adata,key_added='cosg',mu=1,n_genes_user=100,groupby=groupby)
    colnames = ['names', 'scores']
    test = [pd.DataFrame(adata.uns["cosg"][c]) for c in colnames]
    test = pd.concat(test, axis=1, names=[None, 'group'], keys=colnames)
    
    markers = {}
    cats = adata.obs[groupby].cat.categories
    percentile = 5
    for i, c in enumerate(cats):
        cell_type_df = test.loc[:, 'names'][c]
        scores_df = test.loc[:, 'scores'][c]
        percentile_index = int(len(cell_type_df.values.tolist()) * percentile / 100)
        markers[c] = cell_type_df.values.tolist()[:percentile_index]    
    sc.pl.dotplot(adata, 
                  var_names = markers, 
                  groupby=groupby,
                  cmap='Spectral_r',
                  standard_scale = 'var')

def mdiff_pairwise(adata,group_by:str,cluster:str,reference:str,method='t-test',key_added = "rank_genes_groups",top_n=None,matrix=False,**kwargs):
    """   
    Perform differential methylation analysis on single-cell data.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    group_by : str
        The key of the observation grouping to consider.
    groups : list[str]
        List of groups to compare.
    reference : str
        The reference group to compare against.
    method : str, optional (default: 'wilcoxon')
        The method to use for differential analysis. Options are 't-test' or 'wilcoxon'.
    key_added : str, optional (default: "rank_genes_groups")
        The key under which the results will be stored in `adata.uns`.
    top_n : int, optional (default: 100)
        Number of top differentially methylated genes to consider.
    matrix : bool, optional (default: False)
        If True, returns a DataFrame with the results.
    Returns:
    --------
    result : DataFrame or None
        If `matrix` is True, returns a DataFrame with the differential methylation results.
        Otherwise, returns None.
    Raises:
    -------
    ValueError
        If the `method` is not 't-test' or 'wilcoxon'.
    Notes:
    ------
    The function logs the value counts of the groups and the progress of the analysis.
    """

    logging.info(adata.obs[group_by].value_counts())
    
    if isinstance(cluster, str):
        groups = [cluster]
    print(f"... Running differential methylation analysis between {cluster} and reference {reference}")
    if method == "t-test":
        sc.tl.rank_genes_groups(adata, groupby=group_by, groups = groups, reference = reference, key_added = key_added, method='t-test_overestim_var', n_genes=top_n,**kwargs)
    elif method == "wilcoxon":
        sc.tl.rank_genes_groups(adata, groupby=group_by, groups = groups, reference = reference, key_added = key_added, method='wilcoxon', n_genes=top_n,**kwargs)
    else:
        raise ValueError("method must be 't-test' or 'wilcoxon'")
    if reference is not None:
        adata.uns[key_added]['params'] = {'reference': reference, 'groups': groups, 'method': method} 
    if matrix == True:
        result = dmr_df(adata, key_added=key_added)
        return result
    else:       
        return None
    
def mdiff_specific(
    adata,
    group_by: str = 'rank_genes_groups',
    clusters: Union[Literal['all'], Iterable[str]] = 'all',
    reference: str = 'rest',
    method: str ='wilcoxon',
    key_added:str = "rank_genes_groups",
    top_n: bool = None,
    matrix: bool =False,
    **kwargs
    ):
    """   
    Perform differential methylation analysis on single-cell data.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    group_by : str
        The key of the observation grouping to consider.
    clusters : list[str], optional (default: None)    
        List of groups to compare.
    reference : str, optional (default: 'rest')
        The reference group to compare against. 
    method : str, optional (default: 't-test')
        The method to use for differential analysis. Options are 't-test' or 'wilcoxon'.
    key_added : str, optional (default: "rank_genes_groups")    
        The key under which the results will be stored in `adata.uns`.
    top_n : int, optional (default: None)
        Number of top differentially methylated genes to consider.
    matrix : bool, optional (default: False)
        If True, returns a DataFrame with the results.
    Returns:
    --------
    result : DataFrame or None
        If `matrix` is True, returns a DataFrame with the differential methylation results.
        Otherwise, returns None.
    """

    logging.info(adata.obs[group_by].value_counts())
    if clusters == 'all':
        clusters = adata.obs[group_by].cat.categories.tolist()
    print(f"... Running differential methylation analysis between {clusters} and reference {reference}")
    if reference is None:
        #TODO：这里应该是子集进行差异分析的逻辑代码，但是因为不知道怎么合并到对象里面，暂时没有实现
        reference = 'rest'
        # #如果指定参考组为None，则需要对groups里面的每个组进行两两比较
        # if not isinstance(clusters, list) or len(clusters) < 2:
        #     raise ValueError("clusters must be a list with at least two elements")
        # # 生成所有两两组合
        # pairwise_comparisons = list(itertools.combinations(clusters, 2))
        # # 存储结果
        # results = {}

        # for group1, group2 in pairwise_comparisons:
        #     print(f"Comparing {group1} vs {group2}")
            
        #     # 只筛选 group1 和 group2 细胞
        #     adata_subset = adata[adata.obs[group_by].isin([group1, group2])].copy()
           
        #     # 运行差异分析
        #     sc.tl.rank_genes_groups(adata_subset, groupby=[group_by], reference=group2)
            
        #     # 存储结果
        #     results[f"{group1}_vs_{group2}"] = adata_subset.uns['rank_genes_groups']        
    else:
        if method == "t-test":
            sc.tl.rank_genes_groups(adata, groupby=group_by, groups = clusters, reference = reference, key_added = key_added, method='t-test_overestim_var', n_genes=top_n, **kwargs)
        elif method == "wilcoxon":
            sc.tl.rank_genes_groups(adata, groupby=group_by, groups = clusters, reference = reference, key_added = key_added, method='wilcoxon', n_genes=top_n,**kwargs)
        else:
            raise ValueError("method must be 't-test' or 'wilcoxon'")
        adata.uns[key_added]['params'] = {'reference': reference, 'groups': clusters, 'method': method} 
        
    if matrix == True:
        result = dmr_df(adata, key_added=key_added)
        return result
    else:       
        return None
      
def one_vs_rest(adata,obs_dim,cluster,top_n = 100,method="wilcoxon"):
    cluster_judge = adata.obs[obs_dim] == cluster
    one_cells = cluster_judge[cluster_judge]
    rest_cells = cluster_judge[~cluster_judge]
    cell_label = pd.concat([one_cells, rest_cells])
    adata.obs['dmr_groups'] = cell_label.astype('category')
    # 计算差异基因
    sc.tl.rank_genes_groups(adata,
                            groupby='dmr_groups',
                            n_genes=top_n,
                            method=method)
    dmr_result = pd.DataFrame({
        data_key: pd.DataFrame(adata.uns['rank_genes_groups'][data_key]).stack()
        for data_key in ['names', 'pvals_adj']
    })
    dmr_result = dmr_result[dmr_result.index.get_level_values(1).astype(bool)].reset_index(drop=True)
    # add fold change
    in_cells_mean = adata.X[adata.obs['dmr_groups'].astype(bool),].mean(axis=0)
    out_cells_mean = adata.X[~adata.obs['dmr_groups'].astype(bool),].mean(axis=0)
    fc = pd.Series(in_cells_mean / out_cells_mean, index=adata.var_names)
    dmr_result['fc'] = dmr_result['names'].map(fc)
    # filter
    dmr_result = dmr_result[(dmr_result['pvals_adj'] < adj_p_cutoff) & (
            dmr_result['fc'] < fc_cutoff)].copy()
    dmr_result = dmr_result.set_index('names').drop_duplicates()

    # add AUROC and filter again
    auroc = {}
    for gene, row in dmr_result.iterrows():
        yscore = adata.obs_vector(gene)
        ylabel = adata.obs['dmr_groups'] == True
        score = roc_auc_score(ylabel, yscore)
        score = abs(score - 0.5) + 0.5
        auroc[gene] = score
    dmr_result['AUROC'] = pd.Series(auroc)
    dmr_result = dmr_result[(dmr_result['AUROC'] > auroc_cutoff)].copy()
    dmr_result['cluster'] = cluster

    return dmr_result


def dmr_clusters_in_parallel(adata, obs_dim, cluster_list, top_n=100, method="wilcoxon", cpu=5):
    # 创建一个空的 DataFrame 以存储结果
    combined_df = pd.DataFrame(columns=['pvals_adj', 'fc', 'AUROC', 'cluster'])

    # 使用 Pool 创建多个进程
    with Pool(cpu) as pool:
        # 使用 apply_async 并行处理任务
        results = [pool.apply_async(one_vs_rest, args=(adata, obs_dim, cluster, top_n, method)) for cluster in
                   cluster_list]

        # 获取并等待所有结果
        results = [result.get() for result in results]

    # 将结果合并到 combined_df
    for result in results:
        combined_df = pd.concat([combined_df, result], ignore_index=True)

    return combined_df

# 安装fastcluster更好
# 绘制差异基因热图
# df_row 是 1onRest 一对多分析的差异基因组
# df_col 是meta表中的数据组
# data 是热图中的数据
def plot_dmr_heatmap(adata, dmr_df, obs_dim):
    if isinstance(adata.X, np.ndarray):
        # 如果 adata.X 是 numpy.ndarray，则进行转换
        data_df = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)
    else:
        # 如果 adata.X 不是 numpy.ndarray，返回原始数据
        return adata.X
    # 差异基因列表
    df_row = dmr_df.sort_values('cluster')
    df_col = adata.obs[obs_dim].sort_index()
    groups = adata.obs[obs_dim].tolist()
    # 获取Seaborn的颜色调色盘，可以根据需要选择不同的调色盘
    colors = sns.color_palette("tab20", len(groups))
    # 创建分组颜色字典
    col_colors_dict = {group: color for group, color in zip(groups, colors)}

    col_ha = HeatmapAnnotation(label=anno_label(df_col, merge=True, rotation=90, extend=True,
                                                colors=col_colors_dict, adjust_color=True, luminance=0.75,
                                                relpos=(0.5, 0)),  # fontsize=10
                               Group=anno_simple(df_col, colors=col_colors_dict),  # legend_kws={'fontsize':4}
                               verbose=0, axis=1)

    row_ha = HeatmapAnnotation(
        Group=anno_simple(df_row['cluster'], legend=True,
                          colors=col_ha.annotations[1].color_dict),
        verbose=0, axis=0, plot_legend=False)  # label_kws={'rotation':90,'rotation_mode':'anchor','color':'black'}

    plt.figure(figsize=(12, 10))
    # print(data.loc[df_col.index.tolist(),df_row.index.tolist()])
    cm = ClusterMapPlotter(data=data_df.loc[df_col.index.tolist(), df_row.index.tolist()].T,
                           top_annotation=col_ha, left_annotation=row_ha,
                           row_cluster=True, col_cluster=True,
                           label='beta', row_dendrogram=False, legend_gap=7,
                           # row_split=df_row.Group,col_split=df_col.Group,
                           # row_split_gap=0.2,col_split_gap=0.1
                           # row_split_order=df_row.Group.unique().tolist(),
                           # col_split_order=df_row.Group.unique().tolist(),
                           cmap='parula', rasterized=True)
    plt.show()



def dmr_df_to_gene(dmr_df, key_added='gene_annotation', maohao=True):
    # https://github.com/ddb-qiwang/MoClust/blob/39d426c4c0023a3fc8ff3564da3c8d7edd5be06f/gene_matrix.py#L43
    feature_type = 'gene'
    annotation = 'HAVANA'
    upstream = 3000
    gtf = {}

    # read annotation file in GTF format
    with open('D://Test/GSE97693/genome/gencode.v41.basic.annotation.gtf') as f:
        for line in f:
            if line[0:2] != '##' and '\t' + feature_type + '\t' in line:  # and '\t'+annotation+'\t' in line:
                line = line.rstrip('\n').split('\t')
                # forward strand
                if line[6] == '-':
                    if line[0] not in gtf.keys():
                        gtf[line[0]] = [[int(line[3]), int(line[4]) + upstream, line[-1].split(';')[:-1]]]
                    else:
                        gtf[line[0]].append([int(line[3]), int(line[4]) + upstream, line[-1].split(';')[:-1]])
                else:
                    if line[0] not in gtf.keys():
                        gtf[line[0]] = [[int(line[3]) - upstream, int(line[4]), line[-1].split(';')[:-1]]]
                    else:
                        gtf[line[0]].append([int(line[3]) - upstream, int(line[4]), line[-1].split(';')[:-1]])
    raw_adata_features = {}
    feature_index = 0
    for line in dmr_df.index:
        if maohao:
            tmp = []
            tmp_ = line.split(':')
            tmp.append(tmp_[0])
            tmp_ = tmp_[1].split('-')
            tmp.append(tmp_[0])
            tmp.append(tmp_[1])
            line = tmp
        else:
            line = line.split('-')
        if line[0] not in raw_adata_features.keys():
            raw_adata_features[line[0]] = [[int(line[1]), int(line[2]), feature_index]]
        else:
            raw_adata_features[line[0]].append([int(line[1]), int(line[2]), feature_index])
        feature_index += 1
    gene_index = []

    # error: gene index are sorted by features chrom, not raw features. so when combined two result will be misordered
    for chrom in raw_adata_features.keys():
        # raw_adta_features are not sorted
        if chrom in gtf.keys():
            chrom_index = 0
            previous_features_index = 0
            for feature in raw_adata_features[chrom]:  # annotated by chrom
                gene_name = []
                feature_start = feature[0]
                feature_end = feature[1]
                for gene in gtf[chrom]:
                    if (gene[1] <= feature_start):  # the gene is before the feature. we need to test the next gene.
                        continue
                    elif (feature_end <= gene[0]):  # the gene is after the feature. we need to test the next feature.
                        break
                    else:  # the window is overlapping the gene.
                        for n in gene[-1]:
                            if 'gene_name' in n:
                                gene_name.append(n.lstrip('gene_name "').rstrip('""'))

                if gene_name == []:
                    gene_index.append('intergenic')
                elif len(gene_name) == 1:
                    gene_index.append(gene_name[0])
                else:
                    gene_index.append(";".join(list(set(gene_name))))

        else:
            for feature in raw_adata_features[chrom]:
                gene_index.append("unassigned")
    # get the variable metadata
    gene_name = []
    if feature_type == 'transcript':
        for x in gene_index:
            for y in x:
                if 'transcript_name' in y:
                    gene_name.append(y.lstrip(' transcript_name "').rstrip('"'))
        # gene_name = [x[7].lstrip(' transcript_name "').rstrip('"') for x in gene_index]
    elif feature_type == 'gene':
        for x in gene_index:
            for y in x:
                if 'gene_name' in y:
                    gene_name.append(y.lstrip(' gene_name "').rstrip('"'))
        # gene_name = [x[4].lstrip(' gene_name "').rstrip('"') for x in gene_index]

    metadata_genes = {'gene_id': [],
                      'transcript_id': [],
                      'gene_type': [],
                      'gene_name': [],
                      'transcript_type': [],
                      'transcript_name': [],
                      'protein_id': []}

    for line in gene_index:
        dico_line = {}
        for element in line:
            if ' "' in element:
                dico_line[element.rstrip('"').lstrip(" ").split(' "')[0]] = element.rstrip('"').lstrip(" ").split(' "')[
                    1]

        for key in metadata_genes.keys():
            if key in dico_line.keys():
                metadata_genes[key].append(dico_line[key])
            else:
                metadata_genes[key].append('NA')
    dmr_df[key_added] = gene_index
    return dmr_df


