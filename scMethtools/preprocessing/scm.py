#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年10月18日

"""
# https://github.com/scverse/scirpy/blob/ad457499f8449e1e21cf444bdc4895d592c6920c/src/scirpy/datasets/_processing_scripts/maynard2020.py#L42
import logging
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from mudata import MuData
import seaborn as sns
import os
from ..plotting.plot_stat import plot_qc

# import scirpy as ir

# adata = sc.AnnData(
#     var=res[0]["var"],
#     X=tpm_mat[sample_id_mask, :],
#     obs=meta.loc[sample_ids[sample_id_mask], :],
# )

# adata.layers["counts"] = count_mat[sample_id_mask, :]
# adata.layers["counts_length_scaled"] = count_mat_scaled[sample_id_mask, :]

# adata.var.set_index("symbol", inplace=True)

# # ## add IR information

# adata_tcr = ir.io.read_tracer(DATASET_DIR / "smartseq2_pipeline/TraCeR")

# adata_bcr = ir.io.read_bracer(DATASET_DIR / "smartseq2_pipeline/BraCeR/filtered_BCR_summary/changeodb.tab")

# adata_airr = ir.pp.merge_airr(adata_tcr, adata_bcr)

# mdata = MuData({"gex": adata, "airr": adata_airr})

# # ## check that all is right

# mdata_vis = mdata.copy()

# adata.layers

# adata_vis = mdata_vis["gex"].copy()
# sc.pp.log1p(adata_vis)
# sc.pp.highly_variable_genes(adata_vis, n_top_genes=4000, flavor="cell_ranger")
# sc.tl.pca(adata_vis)
# sc.pp.neighbors(adata_vis)
# sc.tl.umap(adata_vis)

# sc.pl.umap(adata_vis, color=["sample", "patient", "origin", "CD8A", "CD14"])

# ir.tl.chain_qc(mdata_vis)

# mdata_vis.update_obs()

# _ = ir.pl.group_abundance(mdata_vis, groupby="airr:receptor_subtype", target_col="gex:patient")

# _ = ir.pl.group_abundance(mdata_vis, groupby="airr:chain_pairing", target_col="gex:patient")

# # ## save MuData

# mdata.write_h5mu("maynard2020.h5mu", compression="lzf")

def scm(adata):
    def __init__(self, adata):
        self.adata = adata
        self.obs = adata.obs
        self.var = adata.var
        self.X = adata.X
        self.layers = adata.layers
        self.uns = adata.uns
        self.obs_keys = adata.obs_keys()
        self.var_keys = adata.var_keys()
        self.layers_keys = adata.layers_keys()
        self.uns_keys = adata.uns_keys()
        self.shape = adata.shape
        self.obs_names = adata.obs_names
        self.var_names = adata.var_names #index of var
    def load_scm(self, file, show=True):
        adata = ad.read_h5ad(file)
        dir_path = os.path.dirname(file)
        if adata.obs.shape[1] == 0:
            data_df = pd.read_csv(os.path.join(dir_path, "basic_stats.csv"))
            adata.obs = data_df
        if show:
            plot_qc(adata)
        return adata
    def add_meta(self, meta_file, sep='\t', index_col='cell'):
        df_metadata = pd.read_csv(meta_file, sep=sep)
        self.obs = pd.merge(self.obs, df_metadata, how='left', left_on='cell_name', right_on=index_col)
        if('label' not in df_metadata.columns):
            print("No column 'label' found in metadata, \'unknown\' is used as the default cell labels")
            df_metadata['label'] = 'unknown'
        if('label_color' in df_metadata.columns):
            self.uns['label_color'] = pd.Series(data=df_metadata.label_color.tolist(),index=df_metadata.label.tolist()).to_dict()
        else:
            print("No column 'label_color' found in metadata, random color is generated for each cell label")
            labels_unique = df_metadata['label'].unique()
            if(len(labels_unique)==1):
                self.uns['label_color'] = {labels_unique[0]:'gray'}
            else:
                list_colors = sns.color_palette("hls",n_colors=len(labels_unique)).as_hex()
                self.uns['label_color'] = {x:list_colors[i] for i,x in enumerate(labels_unique)}
            df_metadata['label_color'] = ''
            for x in labels_unique:
                id_cells = np.where(df_metadata['label']==x)[0]
                df_metadata.loc[df_metadata.index[id_cells],'label_color'] = self.uns['label_color'][x]
        return
    def _write_h5ad(self, aggr_path, silent):
        self.var = _to_categorical(self.var)
        self.obs = _to_categorical(self.obs)

        h5ad_path = os.path.join(aggr_path, "adata.raw.h5ad")
        if not silent:
            print("\nWriting to: {}..\n".format(h5ad_path))
        self.write_h5ad(h5ad_path)
        


def load_scm(file,show=True):
    """load h5 file and basic_stats.csv to adata.obs

    Args:
        file (_type_): _description_
        show (bool, optional): show quality plot or not. Defaults to True.
    Returns:
        _anndata_: _description_
        obs: 'cell_id', 'cell_name', 'sites', 'meth', 'n_total', 'global_meth_level'
        var: 'chromosome', 'start', 'end', 'covered_cell', 'var', 'sr_var'
    """
    adata = ad.read_h5ad(file)
    dir_path = os.path.dirname(file)
    if  adata.obs.shape[1] == 0:
        data_df = pd.read_csv(os.path.join(dir_path, "basic_stats.csv"))
        adata.obs = data_df
    if show:
        plot_qc(adata)
    return adata


def add_meta(adata,meta_file,sep='\t',index_col='cell'):
    """_summary_

    Args:
        adata (anndata): _description_
        meta_file (file_path): absolute path for meta file prepared
        index: index for meta to merge with adata.obs
    """
    df_metadata = pd.read_csv(meta_file, sep =sep)
    # df_metadata.index.name = index_col
    #adata.obs = adata.obs.join(df_metadata, how='left', on=index_col)
    adata.obs = pd.merge(adata.obs, df_metadata, how='left', left_on='cell_name', right_on=index_col)
    if('label' not in df_metadata.columns):
        print("No column 'label' found in metadata, \'unknown\' is used as the default cell labels")
        df_metadata['label'] = 'unknown'
    if('label_color' in df_metadata.columns):
        adata.uns['label_color'] = pd.Series(data=df_metadata.label_color.tolist(),index=df_metadata.label.tolist()).to_dict()
    else:
        print("No column 'label_color' found in metadata, random color is generated for each cell label")
        labels_unique = df_metadata['label'].unique()
        if(len(labels_unique)==1):
            adata.uns['label_color'] = {labels_unique[0]:'gray'}
        else:
            list_colors = sns.color_palette("hls",n_colors=len(labels_unique)).as_hex()
            adata.uns['label_color'] = {x:list_colors[i] for i,x in enumerate(labels_unique)}
        df_metadata['label_color'] = ''
        for x in labels_unique:
            id_cells = np.where(df_metadata['label']==x)[0]
            df_metadata.loc[df_metadata.index[id_cells],'label_color'] = adata.uns['label_color'][x]
    return
    
    
def remove_chrom(adata,exclude_chromosome):
    judge = adata.var[f'chromosome'].isin(exclude_chromosome)
    print(f'{int(judge.sum())} features in {exclude_chromosome} are removed.')
    adata= adata[:,~judge]
    return adata


def filter_features(adata,
                    min_n_cells = None, max_n_cells=None,
                    min_pct_cells = None, max_pct_cells=None):
    """
    Filter out features based on different metrics.
    Arguments:
    ----------
    adata         - AnnData object.
                    Annotated data matrix.
    min_n_cells   - int, default None.
                    Minimum number of cells expressing one feature.
    max_n_cells   - int, default None.
                    Maximum number of cells expressing one feature.
    min_pct_cells - float, default None.
                    Minimum percentage of cells expressing one feature.
    max_pct_cells - float, default None.
                    Maximum percentage of cells expressing one feature. 
    """
    feature = 'regions'
    # if('covered_cell' in adata.var_keys()): 
    #     n_cells = adata.var['covered_cell']
    # else:
    if sp.issparse(adata.X):
        dense_X = adata.X.toarray()
        feature_mean = np.nanmean(dense_X, axis=0)
        n_cells = np.sum(~np.isnan(dense_X), axis=0).astype(int)
    else:
        n_cells = np.sum(~np.isnan(adata.X), axis=0).astype(int)
        feature_mean = np.nanmean(adata.X, axis=0)
    adata.var['covered_cell'] = n_cells
    adata.var['mean'] = feature_mean
        
    if('pct_cell' in adata.var_keys()): 
        pct_cells = adata.var['pct_cell']
    else:
        pct_cells = n_cells/adata.shape[0]
        adata.var['pct_cell'] = pct_cells

        
    if(sum(list(map(lambda x: x is None,[min_n_cells,min_pct_cells,
                                         max_n_cells,max_pct_cells,])))==4):
        print('No filtering')
    else:
        feature_subset = np.ones(len(adata.var_names),dtype=bool)
        if(min_n_cells!=None):
            print('Filter '+feature+' based on min_n_cells')
            feature_subset = (n_cells>=min_n_cells) & feature_subset
        if(max_n_cells!=None):
            print('Filter '+feature+' based on max_n_cells')
            feature_subset = (n_cells<=max_n_cells) & feature_subset
        if(min_pct_cells!=None):
            print('Filter '+feature+' based on min_pct_cells')
            feature_subset = (pct_cells>=min_pct_cells) & feature_subset
        if(max_pct_cells!=None):
            print('Filter '+feature+' based on max_pct_cells')
            feature_subset = (pct_cells<=max_pct_cells) & feature_subset
        adata._inplace_subset_var(feature_subset)
        print('After filtering out low coveraged '+feature+': ')
        print(str(adata.shape[0])+' cells, ' + str(adata.shape[1])+' '+feature)
    return None

def filter_cells(adata,
                 min_n_features = None, max_n_features = None,
                 min_pct_features = None, max_pct_features = None,
                 min_mc_level =None, max_mc_level =None,
                 min_n_sites = None,
                 assay=None):
    feature = 'regions'         
    if('sites' in adata.obs_keys()):
        n_sites = adata.obs['sites']                
    if('global_meth_level' in adata.obs_keys()):
        methylation_level = adata.obs['global_meth_level']
    if('features_number' in adata.obs_keys()):
        n_features = adata.obs['features_number']
    else:
        if sp.issparse(adata.X):
            n_features = np.sum(~np.isnan(adata.X.toarray()), axis=1).astype(int)
        else:
            n_features = np.sum(~np.isnan(adata.X), axis=1).astype(int)
        adata.obs['n_features'] = n_features       
    if('pct_features' in adata.obs_keys()):
        pct_features = adata.obs['pct_features']
    else:
        pct_features = n_features/adata.shape[1]
        adata.obs['pct_peaks'] = pct_features           

    if(sum(list(map(lambda x: x is None,[min_n_features,min_pct_features,min_n_sites,min_mc_level,
                                         max_n_features,max_pct_features,max_mc_level])))==7):
        print('No filtering')    
    else:
        cell_subset = np.ones(len(adata.obs_names),dtype=bool)
        if(min_n_features!=None):
            print('filter cells based on min_n_features >= ', min_n_features)
            cell_subset = (n_features>=min_n_features) & cell_subset
        if(max_n_features!=None):
            print('filter cells based on max_n_features <=', max_n_features)
            cell_subset = (n_features<=max_n_features) & cell_subset
        if(min_pct_features!=None):
            print('filter cells based on min_pct_features >= ',min_pct_features)
            cell_subset = (pct_features>=min_pct_features) & cell_subset
        if(max_pct_features!=None):
            print('filter cells based on max_pct_features <= ',max_pct_features)
            cell_subset = (pct_features<=max_pct_features) & cell_subset
        adata._inplace_subset_obs(cell_subset)
        print('after filtering out low-quality cells: ')
        print(str(adata.shape[0])+' cells, ' + str(adata.shape[1])+' '+feature)
    return None


def drop_cells_from_list(adata, droplist):
    '''Inplace remove cells from given list.
    Arguments:
    ----------
    adata - AnnData object.
            Annotated data matrix.
    droplist - iterables
               An array with cell identifiers as elements.
    Returns:
    ----------
    updates `adata` with a subset of cells that are not in the droplist.
    '''
    droplist = set(droplist)
    if len(droplist) == 0:
        return
    remainingIdx = []
    dropped = 0
    droppedCells = []
    for i in range(adata.obs_names.size):
        if adata.obs_names[i] not in droplist:
            remainingIdx.append(i)
        else:
            dropped += 1
            droppedCells.append(adata.obs_names[i])
    adata._inplace_subset_obs(remainingIdx)
    adata.uns['removedCells'] = droppedCells

# TODO: Add docstrings
def random_subsample(adata, fraction=0.1, return_subset=False, copy=False):
    """TODO."""
    adata_sub = adata.copy() if copy else adata
    p, size = fraction, adata.n_obs
    subset = np.random.choice([True, False], size=size, p=[p, 1 - p])
    adata_sub._inplace_subset_obs(subset)
    return adata_sub if copy else subset if return_subset else None

def log2Transformation(adata,layer=None):
    '''
    Log2(N + 1) transformation on the  sparse data.
    Arguments:
    ----------
    adata - AnnData object.
            Annotated data matrix.
    Returns:
    -----
    updates `adata` with the following fields.
    X - numpy.ndarray, at adata.X
    The transformed data matrix will replace the original array.
    '''
    if layer is None:
        adata.layers['raw'] = adata.X
        adata.X = adata.X.log1p() 
    else:
        adata.layers['raw'] = adata.layers[layer]
        adata.layers[layer] = adata.layers[layer].log1p() 

def filter_genes(adata, min_num_cells = None, min_pct_cells = None,
                 min_count = None, expr_cutoff = 1):
    '''
    Filter out genes based on different metrics.
    from https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L331
    Arguments:
    ----------
    adata         - AnnData object.
                    Annotated data matrix.
    min_num_cells - int, default None.
                    Minimum number of cells expressing one gene
    min_pct_cells - float, default None.
                    Minimum percentage of cells expressing one gene
    min_count     - int, default None.
                    Minimum number of read count for one gene
    expr_cutoff   - float, default 1.
                    Expression cutoff. If greater than expr_cutoff, the gene
                    is considered 'expressed'.
    Returns:
    ----------
    updates `adata` with a subset of genes that pass the filtering.
    '''
    n_counts = np.sum(adata.X, axis = 0)
    adata.var['n_counts'] = n_counts
    n_cells = np.sum(adata.X > expr_cutoff, axis = 0)
    adata.var['n_cells'] = n_cells
    if sum(list(map(lambda x: x is None,[min_num_cells,min_pct_cells,min_count]))) == 3:
        logging.info('No gene filtering')
    else:
        gene_subset = np.ones(len(adata.var_names), dtype = bool)
        if min_num_cells != None:
            logging.info('Filter genes based on min_num_cells')
            gene_subset = (n_cells > min_num_cells) & gene_subset
        if min_pct_cells != None:
            logging.info('Filter genes based on min_pct_cells')
            gene_subset = (n_cells > adata.shape[0] * min_pct_cells) & gene_subset
        if min_count != None:
            logging.info('Filter genes based on min_count')
            gene_subset = (n_counts > min_count) & gene_subset
        adata._inplace_subset_var(gene_subset)
        logging.info('After filtering out low-expressed genes: {} cells, {} genes'.format(adata.shape[0], adata.shape[1]))
        
# def feature_select(adata,select_by,top=500,copy=False):
#     df = adata.var.copy()
#     feature_subset = df.index.isin(df.sort_values(select_by, ascending=False).index[:top])
#     df['feature_select'] = feature_subset
#     if(copy):
#         return df
#     else:
#         adata.var = df
#         return
    
    
def feature_select(adata,top=3000):
    print(adata.shape)
    adata.raw = adata.copy()
    df = adata.var
    feature_subset_sr = df.index.isin(df.sort_values('sr_var', ascending=False).index[:top])
    feature_subset_var = df.index.isin(df.sort_values('sr_var', ascending=False).index[:top])
    df['feature_select'] = feature_subset_sr
    df['feature_select_var'] = feature_subset_var
    adata.var = df
    # data_filter = adata[:,adata.var['feature_select']].copy()
    return adata  


def order_genes_by_gtf(adata, GTF_file_name, ident='gene_name'):
    '''
    Place the gene in the order that follows the annotations from a GTF file.
    Arguments:
    ----------
    adata         - AnnData object.
                    Annotated data matrix.
    GTF_file_name - str, path like.
                    The file name of the GTF file.
    ident         - str, default "gene_name"
                    The identifier type of the genes in the matrix. Choose
                    based on the ninth column of the GTF file.
    Returns:
    ----------
    adata - AnnData object.
            A new object where the order of genes updated.
    '''
    if ident not in {'gene_id', 'gene_name'}:
        raise ValueError("Identifier must be set within {'gene_id', 'gene_name'}")
    ordered_idx = []
    chrlist = []
    with open(GTF_file_name, 'r') as gtfFile:
        for line in gtfFile:
            line = line.rstrip('\r\n')
            if line:
                if line.startswith('#'):
                    continue
                tokens = line.split('\t')
            else:
                break
            if tokens[2] != 'gene':
                continue
            geneIdent = None
            for info in tokens[8].split('; '):
                if info.split(' ')[0] == ident:
                    geneIdent = info.split(' ')[1].strip('"')
                    break
            if geneIdent != None:
                idx = np.where(adata.var_names == geneIdent)
                ordered_idx.extend(list(idx[0]))
                chrlist.extend([tokens[0]] * len(idx[0]))

    adata_tmp = adata.T[ordered_idx].copy()
    adata = adata_tmp.T
    adata.var['Chr'] = chrlist
    adata.var['chr'] = adata.var['Chr'].astype('category')
    del adata.var['Chr']
    return adata

def zscore_norm(adata, against = None):
    '''
    Z-score normalization of the expression profile.
    Arguments:
    ----------
    adata   - AnnData object.
              Annotated data matrix.
    against - AnnData object, default None.
              Another adata where a contol expression profile is saved in.
              If None, normalization will be done against the adata itself.
    Returns:
    ----------
    updates `adata` with the following fields.
    normalized - dict, with keys 'data' - numpy.ndarray, 'against' - str,
                 at adata.uns['normalized'].
                 The normalized data matrix and against.uns['name']
    '''
    logging.info('Applying z-score normalization')
    input_data = adata.X.copy()

    if against == None:
        against = adata
        Mean = np.mean(input_data, axis = 0)
        VAR = np.var(input_data, axis = 0)
    else:
        Mean = np.mean(against.X, axis = 0)
        VAR = np.var(against.X, axis = 0)

    Z = (input_data - Mean) / (VAR + 1)
    adata.uns['normalized'] = {'data': Z, 'against': against.uns['name']}


def keep_genes_as_list(adata, geneList):
    ordered_idx = []
    notFound = 0
    for gene in geneList:
        idx = np.where(adata.var_names == gene)
        try:
            ordered_idx.append(idx[0][0])
        except IndexError:
            notFound += 1
    adata_tmp = adata.T[ordered_idx].copy()
    logging.info('Forced to keep {} genes from given list'.format(len(geneList)))
    return adata_tmp.T


def _to_categorical(df):
    for column in df.columns:
        df[column] = pd.Categorical(df[column])
    return df

def _write_h5ad(adata, aggr_path, silent):
    adata.var = _to_categorical(adata.var)
    adata.obs = _to_categorical(adata.obs)

    h5ad_path = os.path.join(aggr_path, "adata.raw.h5ad")
    if not silent:
        print("\nWriting to: {}..\n".format(h5ad_path))
    adata.write_h5ad(h5ad_path)

def anndata_sparse(adata):
    """
    Set adata.X to csr_matrix

    Arguments:
        adata: AnnData

    Returns:
        adata: AnnData

    """

    from scipy.sparse import csr_matrix
    x = csr_matrix(adata.X.copy())
    adata.X=x
    return adata

def store_layers(adata,layers='counts'):
    """
    Store the X of adata in adata.uns['layers_{}'.format(layers)]

    Arguments:
        adata: AnnData
        layers: the layers name to store, default 'counts'
    """


    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.uns['layers_{}'.format(layers)]=ad.AnnData(sp.csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    elif sp.issparse(adata.X):
        adata.uns['layers_{}'.format(layers)]=ad.AnnData(adata.X.copy(),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                           var=pd.DataFrame(index=adata.var.index),)
    else:
        adata.uns['layers_{}'.format(layers)]=ad.AnnData(sp.csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    print('......The X of adata have been stored in {}'.format(layers))
    

def load_layers(adata,layers='counts'):
    """
    Load the X of adata from adata.uns['layers_{}'.format(layers)]

    Arguments:
        adata: AnnData
        layers: the layers name to load, default 'counts'
    """

    adata.X=adata.uns['layers_{}'.format(layers)].X
    print('......The X of adata have been loaded from {}'.format(layers))
    return adata



