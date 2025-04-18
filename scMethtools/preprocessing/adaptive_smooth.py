#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Adapted From scbs 
> Created Time: 2023年10月02日

"""
import numpy as np
import os
from scipy import sparse
import datetime as datetime
from ..io  import *
import scMethtools.logging as logg
from numba import jit, prange
from sklearn.neighbors import NearestNeighbors


# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")

@jit(nopython=True)
def gaussian_kernel(distances, bandwidth):
    return np.exp(-(distances / bandwidth) ** 2)

# 使用numba加速核心计算函数
@jit(nopython=True)
def calculate_tricube_kernel(bandwidth):
    """计算三次方核函数"""
    hbw = bandwidth // 2
    rel_dist = np.abs((np.arange(bandwidth) - hbw) / hbw)
    return (1 - (rel_dist ** 3)) ** 3

@jit(nopython=True)
def calculate_gaussian_kernel(distances, bandwidth):
    """计算高斯核函数"""
    return np.exp(-(distances / bandwidth) ** 2)

@jit(nopython=True)
def compute_mfrac(data, n_obs):
    """计算甲基化分数"""
    mfracs = np.full(len(n_obs), np.nan, dtype=np.float32)
    for i in range(len(n_obs)):
        if n_obs[i] > 0:
            mfracs[i] = data[i] / n_obs[i]
    return mfracs

@jit(nopython=True)
def smooth_window(window, kernel, weights=None):
    """平滑单个窗口"""
    nz = ~np.isnan(window)
    if not np.any(nz):
        return np.nan   
    valid_window = window[nz]
    valid_kernel = kernel[nz]    
    if weights is not None:
        valid_weights = weights[nz]
        return np.sum(valid_window * valid_kernel * valid_weights) / np.sum(valid_kernel * valid_weights)
    return np.sum(valid_window * valid_kernel) / np.sum(valid_kernel)

@jit(nopython=True, parallel=True)
def smooth_all_positions(mfracs, positions, kernel, hbw, weights=None):
    """并行处理所有位置的平滑操作"""
    result = np.zeros(len(positions), dtype=np.float32)
    
    for i in prange(len(positions)):
        pos = positions[i]
        
        start = max(0, pos - hbw)
        end = min(len(mfracs), pos + hbw)
        window = mfracs[start:end]
        nz = ~np.isnan(window)
        window_kernel = kernel[:end-start][nz]
        #print("weights:",weights)
        if weights is not None:
            window_weights = weights[start:end][nz]
            result[i] = smooth_window(window[nz], window_kernel, window_weights)
            #print("fast:", window[nz], window_kernel, window_weights)
        else:
            result[i] = smooth_window(window[nz], window_kernel)
            #print("fast:", window[nz], window_kernel)
    return result

class FastSmoother:
    def __init__(self, sparse_mat, bandwidth=1000, weigh=True):
        self.hbw = bandwidth // 2
        self.weigh = weigh
        
        # 计算基础统计量
        n_obs = sparse_mat.getnnz(axis=1)
        n_meth = np.ravel(np.sum(sparse_mat > 0, axis=1))
        
        # 计算甲基化分数
        self.mfracs = compute_mfrac(n_meth, n_obs)
        self.cpg_pos = (~np.isnan(self.mfracs)).nonzero()[0]
        
        # 预计算核函数
        self.kernel = calculate_tricube_kernel(bandwidth)
        assert n_obs.shape == n_meth.shape == self.mfracs.shape
        if weigh:
            self.weights = np.log1p(n_obs)
        else:
            self.weights = None

    def smooth_whole_chrom(self):
        # 使用numba加速的并行平滑
        smoothed_values = smooth_all_positions(
            self.mfracs, 
            self.cpg_pos, 
            self.kernel, 
            self.hbw, 
            self.weights
        )
        # 转换为字典格式
        return dict(zip(self.cpg_pos, smoothed_values))
    
@jit(nopython=True)
def gaussian_kernel(distances, bandwidth):
    return np.exp(-(distances / bandwidth) ** 2)

@jit(nopython=True)
def smooth_window_with_kernel(window, distances, bandwidth, weights=None):
    nz = ~np.isnan(window)
    if not np.any(nz):
        return np.nan
    kernel = gaussian_kernel(distances[nz], bandwidth)
    if weights is not None:
        return np.sum(window[nz] * kernel * weights[nz]) / np.sum(kernel * weights[nz])
    else:
        return np.sum(window[nz] * kernel) / np.sum(kernel)

@jit(nopython=True, parallel=True)
def fast_adaptive_smoothing(mfracs, cpg_pos, bandwidths, weights=None):
    result = np.empty(len(cpg_pos), dtype=np.float32)
    
    for i in prange(len(cpg_pos)):
        center = cpg_pos[i]
        bandwidth = bandwidths[center]
        if bandwidth <= 0 or np.isnan(bandwidth) or bandwidth > 1500:
            result[i] = np.nan
            continue

        start = max(0, center - int(bandwidth))
        end = min(len(mfracs), center + int(bandwidth))
        window = mfracs[start:end]
        distances = np.abs(np.arange(start, end) - center).astype(np.float32)

        if weights is not None:
            window_weights = weights[start:end]
            result[i] = smooth_window_with_kernel(window, distances, bandwidth, window_weights)
        else:
            result[i] = smooth_window_with_kernel(window, distances, bandwidth)

    return result

class AdaptiveSmoother:
    def __init__(self, sparse_mat, n_neighbors=10, weigh=False, max_bandwidth=1500):
        self.sparse_mat = sparse_mat
        self.n_neighbors = n_neighbors
        self.weigh = weigh
        self.max_bandwidth = max_bandwidth

        # 计算基本信息
        n_obs = sparse_mat.getnnz(axis=1)
        n_meth = np.ravel(np.sum(sparse_mat > 0, axis=1))
        self.mfracs = np.divide(n_meth, n_obs)
        self.cpg_pos = (~np.isnan(self.mfracs)).nonzero()[0]
        assert n_obs.shape == n_meth.shape == self.mfracs.shape

        if weigh:
            self.weights = np.log1p(n_obs).astype(np.float32)
        else:
            self.weights = None

    def fit_bandwidths(self):
        self.bandwidths = np.zeros_like(self.mfracs, dtype=np.float32)
        if len(self.cpg_pos) > self.n_neighbors:
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
            nbrs.fit(self.cpg_pos[:, np.newaxis])
            distances, _ = nbrs.kneighbors(self.cpg_pos[:, np.newaxis])
            bw = np.minimum(distances[:, -1], self.max_bandwidth)
            self.bandwidths[self.cpg_pos] = bw
        else:
            self.bandwidths[self.cpg_pos] = self.max_bandwidth

    def smooth_whole_chrom(self):
        self.fit_bandwidths()
        smoothed_values = fast_adaptive_smoothing(
            self.mfracs.astype(np.float32),
            self.cpg_pos.astype(np.int32),
            self.bandwidths.astype(np.float32),
            self.weights
        )
        return dict(zip(self.cpg_pos, smoothed_values))
    
        
def _smoothing_chrom_fast(chrom,npz_path,output_dir,adaptive=False):
    logg.info(f"...smoothing {chrom}")
    csr_matrix_chrom = sparse.load_npz(os.path.join(npz_path, f"{chrom}.npz"))
    if adaptive:
        sm = AdaptiveSmoother(csr_matrix_chrom, weigh=True)
    else:
        sm = FastSmoother(csr_matrix_chrom,bandwidth=1000, weigh=True)
    smoothed_chrom = sm.smooth_whole_chrom()
    smooth_path = os.path.join(output_dir,"smooth")
    os.makedirs(smooth_path, exist_ok=True)
    with open(os.path.join(smooth_path, f"{chrom}.csv"), "w") as smooth_out:
        for pos, smooth_val in smoothed_chrom.items():
            smooth_out.write(f"{pos},{smooth_val}\n")
    logg.info(f"...smoothing {chrom} end") 
    return smoothed_chrom
