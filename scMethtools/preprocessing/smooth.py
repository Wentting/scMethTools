#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Adapted From scbs 
> Created Time: 2023年10月02日

"""
import numba
import numpy as np
import os
from scipy.sparse import coo_matrix, save_npz, csr_matrix
from scipy import sparse
import datetime as datetime
from typing import Union
from sklearn.neighbors import NearestNeighbors
import anndata as ad
from ..io  import *
import scMethtools.logging as logg

# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")
class Smoother(object):
    def __init__(self, sparse_mat, bandwidth=1000, weigh=False):
        # create the tricube kernel
        self.hbw = bandwidth // 2
        rel_dist = np.abs((np.arange(bandwidth) - self.hbw) / self.hbw)
        self.kernel = (1 - (rel_dist ** 3)) ** 3  #kernel
        # calculate (unsmoothed) methylation fraction across the chromosome
        n_obs = sparse_mat.getnnz(axis=1)
        n_meth = np.ravel(np.sum(sparse_mat > 0, axis=1))
        #self.mfracs = np.divide(n_meth, n_obs) 
        #Avoid division by zero and set positions with zero results to NaN.
        self.mfracs = np.where(n_obs > 0, np.divide(n_meth, n_obs), np.nan)
        self.cpg_pos = (~np.isnan(self.mfracs)).nonzero()[0]
        #self.entropy = self.calculate_entropy(self.mfracs)
        assert n_obs.shape == n_meth.shape == self.mfracs.shape
        if weigh:
            self.weights = np.log1p(n_obs)
        self.weigh = weigh
        return

    def calculate_entropy(self, mfracs):
        valid_indices = ~np.isnan(mfracs)  # Get indices of non-NaN values
        p = mfracs[valid_indices]  # Get valid methylation fractions
        # Calculate entropy, handling 0 and 1 explicitly to avoid log(0)
        entropy = np.zeros(mfracs.shape) * np.nan  # Initialize with NaNs
        entropy[valid_indices] = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        entropy[np.isnan(entropy)] = 0  # Handle NaNs
        return entropy
    
    def smooth_whole_chrom(self):
        smoothed = {}
        for i in self.cpg_pos:
            window = self.mfracs[i - self.hbw : i + self.hbw]
            nz = ~np.isnan(window)
            try:
                k = self.kernel[nz]
                if self.weigh:
                    w = self.weights[i - self.hbw : i + self.hbw][nz]
                    smooth_val = np.divide(np.sum(window[nz] * k * w), np.sum(k * w))
                else:
                    smooth_val = np.divide(np.sum(window[nz] * k), np.sum(k))
                smoothed[i] = smooth_val
            except IndexError:
                smoothed[i] = np.nan
        return smoothed
    
    def smooth_whole_entory(self):
        smoothed = {}
        for i in self.cpg_pos:
            window = self.entropy[i - self.hbw : i + self.hbw]
            nz = ~np.isnan(window)
            try:
                k = self.kernel[nz]
                if self.weigh:
                    w = self.weights[i - self.hbw : i + self.hbw][nz]
                    smooth_val = np.divide(np.sum(window[nz] * k * w), np.sum(k * w))
                else:
                    smooth_val = np.divide(np.sum(window[nz] * k), np.sum(k))
                smoothed[i] = smooth_val
            except IndexError:
                smoothed[i] = np.nan
        return smoothed
    # def smooth(data_dir, bandwidth, use_weights):
    #     out_dir = os.path.join(data_dir, "smoothed")
    #     os.makedirs(out_dir, exist_ok=True)
    #     for mat_path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
    #         chrom = os.path.basename(os.path.splitext(mat_path)[0])
    #         print(f"Reading chromosome {chrom} data from {mat_path} ...")
    #         mat = sparse.load_npz(mat_path)
    #         sm = Smoother(mat, bandwidth, use_weights)
    #         print(f"Smoothing chromosome {chrom} ...")
    #         smoothed_chrom = sm.smooth_whole_chrom()
    #         with open(os.path.join(out_dir, f"{chrom}.csv"), "w") as smooth_out:
    #             for pos, smooth_val in smoothed_chrom.items():
    #                 smooth_out.write(f"{pos},{smooth_val}\n")
    #     print('smoothing done')
    #     return
    
class AdaptiveSmoother(object):
    def __init__(self, sparse_mat, n_neighbors=10, weigh=False):
        self.sparse_mat = sparse_mat
        self.n_neighbors = n_neighbors
        self.weigh = weigh

        # Calculate (unsmoothed) methylation fraction across the chromosome
        n_obs = sparse_mat.getnnz(axis=1)
        n_meth = np.ravel(np.sum(sparse_mat > 0, axis=1))
        self.mfracs = np.divide(n_meth, n_obs)
        self.cpg_pos = (~np.isnan(self.mfracs)).nonzero()[0]
        assert n_obs.shape == n_meth.shape == self.mfracs.shape
        if weigh:
            self.weights = np.log1p(n_obs)

    def fit_bandwidths(self):
        # Fit nearest neighbors to find adaptive bandwidths
        self.bandwidths = np.zeros_like(self.mfracs)
        if len(self.cpg_pos) > self.n_neighbors:
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
            nbrs.fit(self.cpg_pos[:, np.newaxis])
            distances, _ = nbrs.kneighbors(self.cpg_pos[:, np.newaxis])
            self.bandwidths[self.cpg_pos] = distances[:, -1]
        else:
            self.bandwidths[self.cpg_pos] = np.inf  # Avoid division by zero
  
    def smooth_whole_chrom(self):
        self.fit_bandwidths()
        smoothed = {}

        for i in self.cpg_pos:
            bandwidth = self.bandwidths[i]
            if bandwidth > 3000:
                smoothed[i] = np.nan
                continue
            start = max(i - int(bandwidth), 0)
            end = min(i + int(bandwidth), len(self.mfracs))
            window = self.mfracs[start:end]
            nz = ~np.isnan(window)

            if bandwidth > 0 and nz.any():
                distances = np.abs(np.arange(-bandwidth, bandwidth)[:len(window)])
                kernel = np.exp(-(distances / bandwidth) ** 2)  # Gaussian kernel

                if self.weigh:
                    weights = self.weights[start:end][nz]
                    smooth_val = np.sum(window[nz] * kernel[nz] * weights) / np.sum(kernel[nz] * weights)
                else:
                    smooth_val = np.sum(window[nz] * kernel[nz]) / np.sum(kernel[nz])
                smoothed[i] = smooth_val
            else:
                smoothed[i] = np.nan

        return smoothed
    
def _smoothing_cell(data_path, keep=True):
    for coo_file in sorted(find_files_with_suffix(data_path, ".txt")):
        chrom = os.path.basename(os.path.splitext(coo_file)[0])
        with open(coo_file, 'r') as file:
            lines = file.readlines()
        rows, cols, data = zip(*(map(int, line.split()) for line in lines))
        coo_matrix_loaded = coo_matrix((data, (rows, cols)))
        csr_matrix_result = csr_matrix(coo_matrix_loaded)
        sm = Smoother(csr_matrix_result,bandwidth=1000, weigh=True)
        smoothed_chrom = sm.smooth_whole_chrom()
        save_npz(os.path.join(data_path, f"{chrom}.npz"), csr_matrix_result)
        smooth_path = os.path.join(data_path,"smooth")
        os.makedirs(smooth_path, exist_ok=True)
        with open(os.path.join(smooth_path, f"{chrom}.csv"), "w") as smooth_out:
            for pos, smooth_val in smoothed_chrom.items():
                smooth_out.write(f"{pos},{smooth_val}\n")
        del smoothed_chrom
        if keep == False:
            os.remove(coo_file)
    return

# def _smoothing_chrom(csr_matrix_chrom,chrom,output_dir):
#     echo(f"...smoothing {chrom}")
#     csr_matrix_chrom = sparse.save_npz(os.path.join(output_dir, "{chrom}.npz"))
#     sm = Smoother(csr_matrix_chrom,bandwidth=1000, weigh=True)
#     smoothed_chrom = sm.smooth_whole_chrom()
#     smooth_path = os.path.join(output_dir,"smooth")
#     os.makedirs(smooth_path, exist_ok=True)
#     with open(os.path.join(smooth_path, f"{chrom}.csv"), "w") as smooth_out:
#         for pos, smooth_val in smoothed_chrom.items():
#             smooth_out.write(f"{pos},{smooth_val}\n")
#     echo(f"...smoothing {chrom} end") 
#     return smoothed_chrom
def _smoothing_chrom(chrom,npz_path,output_dir,adaptive=False):
    logg.info(f"...smoothing {chrom}")
    csr_matrix_chrom = sparse.load_npz(os.path.join(npz_path, f"{chrom}.npz"))
    if adaptive:
        sm = AdaptiveSmoother(csr_matrix_chrom, weigh=True)
    else:
        sm = Smoother(csr_matrix_chrom,bandwidth=1000, weigh=True)
    smoothed_chrom = sm.smooth_whole_chrom()
    smooth_path = os.path.join(output_dir,"smooth")
    os.makedirs(smooth_path, exist_ok=True)
    with open(os.path.join(smooth_path, f"{chrom}.csv"), "w") as smooth_out:
        for pos, smooth_val in smoothed_chrom.items():
            smooth_out.write(f"{pos},{smooth_val}\n")
    logg.info(f"...smoothing {chrom} end") 
    return smoothed_chrom

def _smoothing_chrom_adaptive(chrom,npz_path,output_dir):
    logg.info(f"...smoothing {chrom}")
    csr_matrix_chrom = sparse.load_npz(os.path.join(npz_path, f"{chrom}.npz"))
    sm = AdaptiveSmoother(csr_matrix_chrom, weigh=True)
    smoothed_chrom = sm.smooth_whole_chrom()
    smooth_path = os.path.join(output_dir,"smooth")
    os.makedirs(smooth_path, exist_ok=True)
    with open(os.path.join(smooth_path, f"{chrom}.csv"), "w") as smooth_out:
        for pos, smooth_val in smoothed_chrom.items():
            smooth_out.write(f"{pos},{smooth_val}\n")
    logg.info(f"...smoothing {chrom} end") 
    return smoothed_chrom