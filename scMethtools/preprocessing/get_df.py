import pandas as pd
import numpy as np
from packaging.version import Version

def dmr_df(adata,
            cluster=None,
            key_added="rank_genes_groups",
            pval_cutoff = None,
            log2fc_min = None,
            log2fc_max = None,
            gene_symbols = None,
           ):
    """ create a dataframe of DMRs for a given groupby condition

    Args:
        adata (_type_): _description_
        groupby (_type_, optional): _description_. Defaults to None.
        key_added (_type_, optional): _description_. Defaults to None.
        gene_annotation (_type_, optional): _description_. Defaults to None.
        
    Example
    -------
    >>> dedf = scm.pp.dmr_df(adata,key_added="wilcoxon")
    """
    if isinstance(cluster, str):
        group = [cluster]
    # For pairwise comparison
    if cluster is None: 
        group = list(adata.uns[key_added]["names"].dtype.names)
    method = adata.uns[key_added]["params"]["method"]
    
    if method == "logreg":
        colnames = ["names", "scores"]
    else:
        #t-test,wilcxon
        colnames = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
    
    d = [pd.DataFrame(adata.uns[key_added][c])[group] for c in colnames]
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
        if pts in adata.uns[key_added]:
            pts_df = (
                adata.uns[key_added][pts][group]
                .rename_axis(index="names")
                .reset_index()
                .melt(id_vars="names", var_name="group", value_name=name)
            )
            d = d.merge(pts_df)

    # remove group column for backward compat if len(group) == 1 如果group只有一个说明是一对一的比较，一般是指定了refrence的，所以就删掉了
    if len(group) == 1:
        d.drop(columns="group", inplace=True)
        ref = adata.uns[key_added]["params"]["reference"]
        print (f"Importing differential methylated region dataframe for {group} v.s. {ref}")
        

    return d.reset_index(drop=True)
        
    
    
    
