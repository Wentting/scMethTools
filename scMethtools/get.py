from typing import TYPE_CHECKING, TypeVar
import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from packaging.version import Version
if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from typing import Any, Literal

    from anndata._core.sparse_dataset import BaseCompressedSparseDataset
    from anndata._core.views import ArrayView


# TODO: implement diffxpy method, make singledispatch
def rank_genes_groups_df(
    adata: AnnData,
    group: str | Iterable[str] | None,
    *,
    key: str = "rank_genes_groups",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
    gene_symbols: str | None = None,
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


def get_dmr_genes(adata, key_added="dmr_genes", groups=None, gene_symbols=None):
    
    list(adata.uns[key_added]['names'].dtype.names)
    group = "celltype_A"
    gene_list =  list(set(np.concatenate([adata.uns[key_added]['names'][group] for group in ['ESC 2i', 'MII oocyte ']])))
    logfc = pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'], index=adata.uns['rank_genes_groups']['names'])
    up_genes = logfc[group][logfc[group] > 0].index.tolist()   # 上调基因
    down_genes = logfc[group][logfc[group] < 0].index.tolist() # 下调基因
    
    pass

def get_region_genes(
    adata: AnnData,
    region_key: str,
    *,
    key_added: str = "region_genes",
    gene_symbols: str | None = None,
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
        raise ValueError("regions 参数必须是列表")
    
    #把region换成gene，前提是已经做过注释了
    adata.var[adata.var.index.get_indexer(all_genes)]['Gene']
    # 确保 all_genes 存在于 adata.var.index
    valid_genes = [gene for gene in all_genes if gene in adata.var.index]

    # 提取对应的基因信息
    gene_list= adata.var.loc[valid_genes, 'Gene'].dropna().str.upper().tolist()
    
    if gene_symbols is not None:
        adata.var[gene_symbols] = adata.var_names

    adata.uns[key_added] = {}
    for region in adata.var[region_key].unique():
        genes = adata.var_names[adata.var[region_key] == region]
        if gene_symbols is not None:
            genes = adata.var[gene_symbols][adata.var[region_key] == region]
        adata.uns[key_added][region] = genes