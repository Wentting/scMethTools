from typing import TYPE_CHECKING, List, Optional, Union
import pandas as pd
from anndata import AnnData
from packaging.version import Version
from collections.abc import Iterable


# TODO: implement diffxpy method, make singledispatch
def rank_genes_groups_df(
    adata: AnnData,
    group: Optional[Union[str, Iterable[str]]],
    *,
    key: str = "rank_genes_groups",
    pval_cutoff: Optional[float] = None,  
    log2fc_min: Optional[float] = None, 
    log2fc_max: Optional[float] = None,   
    gene_symbols: Optional[str] = None    
) -> pd.DataFrame:
    """\
    :func:`scanpy.tl.rank_genes_groups` results in the form of a
    :class:`~pandas.DataFrame`.

    Params
    ------
    adata
        Object to get results from.
    group
        Which group (as in :func:`scanpy.tl.rank_genes_groups`'s `groupby`
        argument) to return results from. Can be a list. All groups are
        returned if groups is `None`.
    key
        Key differential expression groups were stored under.
    pval_cutoff
        Return only adjusted p-values below the  cutoff.
    log2fc_min
        Minimum logfc to return.
    log2fc_max
        Maximum logfc to return.
    gene_symbols
        Column name in `.var` DataFrame that stores gene symbols. Specifying
        this will add that column to the returned dataframe.

    Example
    -------
    >>> import scanpy as sc
    >>> pbmc = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(pbmc, groupby="louvain", use_raw=True)
    >>> dedf = sc.get.rank_genes_groups_df(pbmc, group="0")
    """
    if isinstance(group, str):
        group = [group]
    if group is None:
        group = list(adata.uns[key]["names"].dtype.names)
    method = adata.uns[key]["params"]["method"]
    if method == "logreg":
        colnames = ["names", "scores"]
    else:
        colnames = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]

    d = [pd.DataFrame(adata.uns[key][c])[group] for c in colnames]
    d = pd.concat(d, axis=1, names=[None, "group"], keys=colnames)
    if Version(pd.__version__) >= Version("2.1"):
        d = d.stack(level=1, future_stack=True).reset_index()
    else:
        d = d.stack(level=1).reset_index()
    d["group"] = pd.Categorical(d["group"], categories=group)
    d = d.sort_values(["group", "level_0"]).drop(columns="level_0")

    if method != "logreg":
        if pval_cutoff is not None:
            d = d[d["pvals_adj"] < pval_cutoff]
        if log2fc_min is not None:
            d = d[d["logfoldchanges"] > log2fc_min]
        if log2fc_max is not None:
            d = d[d["logfoldchanges"] < log2fc_max]
    if gene_symbols is not None:
        d = d.join(adata.var[gene_symbols], on="names")

    for pts, name in {"pts": "pct_nz_group", "pts_rest": "pct_nz_reference"}.items():
        if pts in adata.uns[key]:
            pts_df = (
                adata.uns[key][pts][group]
                .rename_axis(index="names")
                .reset_index()
                .melt(id_vars="names", var_name="group", value_name=name)
            )
            d = d.merge(pts_df)

    # remove group column for backward compat if len(group) == 1
    if len(group) == 1:
        d.drop(columns="group", inplace=True)

    return d.reset_index(drop=True)



def get_group_dmr(
    adata: AnnData,
    key_added: str = "rank_genes_groups",
    groups: Optional[Union[List[str], str]] = None,
    gene_symbols: Optional[str] = None,
    direction: str = "both"  # "up", "down", or "both"
) -> dict[str, list[str]]:
    """
    Get genes associated with DMRs.

    Args:
        adata (AnnData): AnnData object containing single-cell data.
        key_added (str, optional): Key in adata.uns where differential analysis results are stored. Defaults to "rank_genes_groups".
        groups (Optional[Union[List[str], str]], optional): Groups to get genes for. Can be a string or a list of strings. Defaults to None, meaning all groups.
        gene_symbols (Optional[str], optional): Column name for gene symbols if gene names need to be converted. Defaults to None.
        direction (str, optional): Direction for gene selection, can be "up" (upregulated genes), "down" (downregulated genes), or "both" (all genes). Defaults to "both".

    Raises:
        ValueError: If the specified key_added is not found in adata.uns.

    Returns:
        dict[str, list[str]]: Dictionary of gene lists for each group.

    ---
    Example:
    
    """
    
    if key_added not in adata.uns:
        raise ValueError(f"{key_added} was not found in `adata.uns`. Please check whether differential analysis has been performed.")
    if isinstance(groups, str):
        groups = [groups]
    # 获取所有分组
    if groups is None:
        groups = list(adata.uns[key_added]['names'].dtype.names)
    
    # 获取基因名和 logFC 数据
    gene_names = pd.DataFrame(adata.uns[key_added]['names'])
    logfc = pd.DataFrame(adata.uns[key_added]['logfoldchanges'], index=gene_names.index)

    result_genes = {}

    for group in groups:
        # 获取该组的基因
        genes = gene_names[group].tolist()
        fc_values = logfc[group]

        # 筛选基因
        if direction == "up":
            selected_genes = [genes[i] for i in range(len(genes)) if fc_values[i] > 0]
        elif direction == "down":
            selected_genes = [genes[i] for i in range(len(genes)) if fc_values[i] < 0]
        else:  # both
            selected_genes = genes

        # 如果需要转换基因名
        if gene_symbols and gene_symbols in adata.var.columns:
            selected_genes = adata.var.loc[selected_genes, gene_symbols].dropna().tolist()

        result_genes[group] = selected_genes

    return result_genes



def get_region_genes(
    adata: AnnData,
    regions: Optional[str],
    use_gene_col: str = 'Gene',
    upper: bool = False,
) -> None:
    """\
    Get genes associated with genomic regions.

    Params
    ------
    adata
        Object to get results from.
    region_key
        Key in `adata.var` that stores region information.
    key_added
        Key to store results under.
    gene_symbols
        Column name in `.var` DataFrame that stores gene symbols. Specifying
        this will add that column to the returned dataframe.

    Example
    -------
    >>> import scanpy as sc
    >>> pbmc = sc.datasets.pbmc68k_reduced()
    >>> sc.get.get_region_genes(pbmc, region_key="gene_id")
    """
    # 确保 names 是列表
    if not isinstance(regions, list):
        raise ValueError("regions must be a list") 
    # 确保 all_genes 存在于 adata.var.index
    valid_regions= [region for region in regions if region in adata.var.index]
    #print(valid_regions)
    if  upper:
        # gene needs to be upper if doing enrichment analysis in scmethtools especially for offline analysis
        gene_list= adata.var.loc[valid_regions, use_gene_col].dropna().str.upper().tolist()
    else:
        gene_list= adata.var.loc[valid_regions, use_gene_col].dropna().tolist()
    return gene_list
        
def get_dmr_genes(
    adata: AnnData,
    key_added: str = "rank_genes_groups",
    groups: Optional[Union[List[str], str]] = None,
    gene_symbols: Optional[str] = None,
    direction: str = "both" ,
    use_gene_col: str = 'Gene',
    upper: bool = True,
    ):
    
    if key_added not in adata.uns:
        raise ValueError(f"{key_added} was not found in `adata.uns`. Please check whether differential analysis has been performed.")
    if isinstance(groups, str):
        groups = [groups]
    # 获取所有分组
    if groups is None:
        groups = list(adata.uns[key_added]['names'].dtype.names)
    group_region = get_group_dmr(adata, key_added, groups, gene_symbols, direction) #返回的是一个字典

    group_genes={}
    
    for g in groups:
        gene_list = get_region_genes(adata, group_region[g],upper=upper,use_gene_col= use_gene_col)
        group_genes[g] = gene_list
    return group_genes
    