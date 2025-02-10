
import scanpy as sc
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pandas.testing as pdt
import scanpy as sc
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from typing import Optional, Union

from collections import Counter
import math
from sklearn.feature_selection import f_classif, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
#feature select
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA 

from openTSNE import TSNEEmbedding, affinity, initialization
#tsne methods
from typing import Callable, Union

class my_PCA:
    """
    A class to store the results of a sklearn PCA (i.e., embeddings, loadings and 
    explained variance ratios).
    """
    def __init__(self):
        self.n_pcs = None
        self.embs = None
        self.loads = None
        self.var_ratios = None

    def calculate_PCA(self, M, n_components=50):
        '''
        Perform PCA decomposition of some input obs x genes matrix.
        '''
        self.n_pcs = n_components
        # Convert to dense np.array if necessary)
        if isinstance(M, np.ndarray) == False:
            M = M.toarray()

        # Perform PCA
        model = PCA(n_components=n_components, svd_solver='arpack',random_state=1234)
        # Store results accordingly
        self.embs = np.round(model.fit_transform(M), 2) # Round for reproducibility
        self.loads = model.components_.T
        self.var_ratios = model.explained_variance_ratio_
        self.cum_sum_eigenvalues = np.cumsum(self.var_ratios)

        return self
    def impute(self, M , method='mean'):
        from sklearn.impute import SimpleImputer
        if method == 'mean':
            imp = SimpleImputer(missing_values=np.nan, strategy="mean")
            M  = imp.fit_transform(M)
        elif method == 'median':
            imp = SimpleImputer(missing_values=np.nan, strategy="median")
            M  = imp.fit_transform(M)
        elif method == 'most_frequent':
            imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            M  = imp.fit_transform(M)
        elif method == 'constant':
            imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
            M  = imp.fit_transform(M)
        elif method == 'knn':
            #TO DO
            #稀疏矩阵报错
            from sklearn.impute import KNNImputer
            imp = KNNImputer(n_neighbors=5, weights="uniform")
            M  = imp.fit_transform(M)
        return M
    
def pca(adata,use_hvm=False,hvm_obs='feature_select',key='X',impute='mean',n_pcs=50,layer=None,plot=False,first_pc = False,n_pc = 15,fig_size=(4,4),
                                    pad=1.08,w_pad=None,h_pad=None):
    """
    Perform PCA on the input AnnData object.
    Parameters:
    -----------
    adata: AnnData
        Annotated data matrix.
    use_hvm: `bool`, optional (default: False)
        If True, use the highly variable genes obtained from select_variable_genes() for PCA.
    key: `str`, optional (default: 'X')
        The key in adata to store the PCA results.
    impute: `str`, optional (default: 'mean')
        The method to impute missing values. Choose from {'mean', 'median', 'most_frequent', 'constant', 'knn'}.
    
    """
    model = my_PCA()
    if use_hvm:
        print(f'... using top variable features based on {hvm_obs}')
        if hvm_obs not in adata.var.keys():
            raise KeyError('Please run select_hvm() first!')
        if layer is not None:
            if layer in adata.layers: 
                X = adata[:,adata.var[hvm_obs]].layers[layer]
                key = f'{layer}'
            else:
                raise KeyError(f'Selected layer {layer} is not present. Compute it first!')
        else:
            X = adata[:,adata.var[hvm_obs]].X
    else:
        print('... using all features')
        if layer is not None:
            if layer in adata.layers: 
                X = adata.layers[layer]
                key = f'{layer}'
            else:
                raise KeyError(f'Selected layer {layer} is not present. Compute it first!')
        else:
            X = adata.X
            
    model = my_PCA()
    X = model.impute(X,method=impute)
    model.calculate_PCA(X, n_components=n_pcs)
    adata.obsm[key + '_pca'] = model.embs
    # adata.varm[key + '_pca_loadings'] = model.loads
    adata.uns[key + '_pca_var_ratios'] = model.var_ratios
    # adata.uns[key + '_cum_sum_eigenvalues'] = np.cumsum(model.var_ratios)

    if plot:
        pca_variance_ratio = model.var_ratios
        max_pc =  pca_variance_ratio.shape[0]
        fig = plt.figure(figsize=fig_size)
        plt.plot(range(max_pc),pca_variance_ratio[:max_pc])
        if(first_pc):
            plt.axvline(n_pc,c='red',ls = '--')
        else:
            plt.axvline(1,c='red',ls = '--')
            plt.axvline(n_pc+1,c='red',ls = '--')
            plt.xlabel('Principal Component')
            plt.ylabel('Variance Ratio')
            plt.locator_params(axis='x',nbins=5)
            plt.locator_params(axis='y',nbins=5)
            plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    # if(save_fig):
    #     plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
    #     plt.close(fig)
    return adata

def tsne(adata, use_rep='X_pca',
         metric: Union[str, Callable] = "cosine",
         exaggeration: float = -1,
         perplexity: int = 30,
         n_jobs: int = -1):
    if use_rep not in adata.obsm.keys():
        raise KeyError(f'The representation {use_rep} is not present in adata.obsm. Please run the PCA first!')
    else:
        X = adata.obsm[use_rep]
    Z = tsne_method(X=X, metric=metric, exaggeration=exaggeration, perplexity=perplexity, n_jobs=n_jobs)
    adata.obsm['X_tsne'] = Z
    return

def umap(adata):
    import scanpy as sc
    sc.tl.umap(adata)
    
def tsne_method(X: np.ndarray,
                metric: Union[str, Callable] = "euclidean",
                exaggeration: float = -1,
                perplexity: int = 30,
                n_jobs: int = -1) -> TSNEEmbedding:
    """
    Implementation of Dmitry Kobak and Philipp Berens
    "The art of using t-SNE for single-cell transcriptomics" based on openTSNE.
    See https://doi.org/10.1038/s41467-019-13056-x | www.nature.com/naturecommunications
    Args:
        X				The data matrix of shape (n_cells, n_genes) i.e. (n_samples, n_features)
        metric			Any metric allowed by PyNNDescent (default: 'euclidean')
        exaggeration	The exaggeration to use for the embedding
        perplexity		The perplexity to use for the embedding

    Returns:
        The embedding as an opentsne.TSNEEmbedding object (which can be cast to an np.ndarray)
    """
    n = X.shape[0]
    if n > 100_000:
        if exaggeration == -1:
            exaggeration = 1 + n / 333_333
        # Subsample, optimize, then add the remaining cells and optimize again
        # Also, use exaggeration == 4
        logging.info(f"Creating subset of {n // 40} elements")
        # Subsample and run a regular art_of_tsne on the subset
        indices = np.random.permutation(n)
        reverse = np.argsort(indices)
        X_sample, X_rest = X[indices[:n // 40]], X[indices[n // 40:]]
        logging.info(f"Embedding subset")
        Z_sample = tsne_method(X_sample)

        logging.info(
            f"Preparing partial initial embedding of the {n - n // 40} remaining elements"
        )
        if isinstance(Z_sample.affinities, affinity.Multiscale):
            rest_init = Z_sample.prepare_partial(X_rest,
                                                 k=1,
                                                 perplexities=[1 / 3, 1 / 3])
        else:
            rest_init = Z_sample.prepare_partial(X_rest, k=1, perplexity=1 / 3)
        logging.info(f"Combining the initial embeddings, and standardizing")
        init_full = np.vstack((Z_sample, rest_init))[reverse]
        init_full = init_full / (np.std(init_full[:, 0]) * 10000)

        logging.info(f"Creating multiscale affinities")
        affinities = affinity.PerplexityBasedNN(X,
                                                perplexity=perplexity,
                                                metric=metric,
                                                method="approx",
                                                n_jobs=n_jobs)
        logging.info(f"Creating TSNE embedding")
        Z = TSNEEmbedding(init_full,
                          affinities,
                          negative_gradient_method="fft",
                          n_jobs=n_jobs)
        logging.info(f"Optimizing, stage 1")
        Z.optimize(n_iter=250,
                   inplace=True,
                   exaggeration=12,
                   momentum=0.5,
                   learning_rate=n / 12,
                   n_jobs=n_jobs)
        logging.info(f"Optimizing, stage 2")
        Z.optimize(n_iter=750,
                   inplace=True,
                   exaggeration=exaggeration,
                   momentum=0.8,
                   learning_rate=n / 12,
                   n_jobs=n_jobs)
    elif n > 3_000:
        if exaggeration == -1:
            exaggeration = 1
        # Use multiscale perplexity
        affinities_multiscale_mixture = affinity.Multiscale(
            X,
            perplexities=[perplexity, n / 100],
            metric=metric,
            method="approx",
            n_jobs=n_jobs)
        init = initialization.pca(X)
        Z = TSNEEmbedding(init,
                          affinities_multiscale_mixture,
                          negative_gradient_method="fft",
                          n_jobs=n_jobs)
        Z.optimize(n_iter=250,
                   inplace=True,
                   exaggeration=12,
                   momentum=0.5,
                   learning_rate=n / 12,
                   n_jobs=n_jobs)
        Z.optimize(n_iter=750,
                   inplace=True,
                   exaggeration=exaggeration,
                   momentum=0.8,
                   learning_rate=n / 12,
                   n_jobs=n_jobs)
    else:
        if exaggeration == -1:
            exaggeration = 1
        # Just a plain TSNE with high learning rate
        lr = max(200, n / 12)
        aff = affinity.PerplexityBasedNN(X,
                                         perplexity=perplexity,
                                         metric=metric,
                                         method="approx",
                                         n_jobs=n_jobs)
        init = initialization.pca(X)
        Z = TSNEEmbedding(init,
                          aff,
                          learning_rate=lr,
                          n_jobs=n_jobs,
                          negative_gradient_method="fft")
        Z.optimize(250, exaggeration=12, momentum=0.5, inplace=True, n_jobs=n_jobs)
        Z.optimize(750,
                   exaggeration=exaggeration,
                   momentum=0.8,
                   inplace=True,
                   n_jobs=n_jobs)
    return Z



def getNClusters(adata,n_cluster,range_min=0,range_max=3,max_steps=20):
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        print('step ' + str(this_step))
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.louvain(adata,resolution=this_resolution)
        sc.tl.leiden(adata,resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()
        
        print('got ' + str(this_clusters) + ' at resolution ' + str(this_resolution))
        
        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
            return(this_resolution, adata)
        this_step += 1
    
    print('Cannot find the number of clusters')
    print('Clustering solution from last iteration is used:' + str(this_clusters) + ' at resolution ' + + str(this_resolution))


def hvm_basic(adata,n_features=None):
    print(adata.shape)
    adata.raw = adata.copy()
    df = adata.var
    feature_subset_sr = df.index.isin(df.sort_values('sr_var', ascending=False).index[:n_features])
    feature_subset_var = df.index.isin(df.sort_values('var', ascending=False).index[:n_features])
    df['feature_select'] = feature_subset_sr
    df['feature_select_var'] = feature_subset_var
    adata.var = df

    
def downsample(adata,downsample,fraction=None,copy=True):
    n_cells = adata.shape[0]
    if downsample and n_cells > downsample:
        # make a downsampled mcds
        print(f'Downsample cells to {downsample} to calculate HVF.')
        if copy:
        #by count
            bdata = sc.pp.subsample(data=adata, n_obs=10, random_state=1, copy=True)
            return bdata
        else:
            sc.pp.subsample(data=adata, n_obs=10, random_state=1, copy=False)
    if fraction:
         #by fraction
        if copy:
            bdata = sc.pp.subsample(data=adata, fraction=0.9, random_state=1, copy=True)
            return bdata
        else:
            sc.pp.subsample(data=adata, fraction=0.9, random_state=1, copy=True)

        