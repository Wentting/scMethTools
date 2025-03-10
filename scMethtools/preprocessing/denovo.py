#adapt from scbs 
from numba import njit, prange
import time
from scipy.stats import betabinom
import numpy as np
import pandas as pd
import numba
import os
from glob import glob
from scipy import sparse
import multiprocessing as mp
import math

# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")

def denovo(input_dir, exclude_chroms=None, window_size=3000, step_size=500,cpu=10):
    """
    Calculate the methylation variability for each cell in each window of the genome.
    """
     #检查smooth和data文件夹是否存在
    if not os.path.exists(os.path.join(input_dir, "smooth")):
        print(f"Could not find smooth folder at {os.path.join(input_dir, 'smooth')}, please run smooth first.")
        return
    if not os.path.exists(os.path.join(input_dir, "data")):
        print(f"Could not find data folder at {os.path.join(input_dir, 'data')}")
        return
    npz_files = glob(os.path.join(input_dir, "data", "*.npz"))
    
    features = {}
    cpu = min(cpu, 24)
    results = []
    pool  = mp.Pool(processes=cpu)

    for npz_file in npz_files:
        chrom = os.path.basename(os.path.splitext(npz_file)[0])
        if exclude_chroms and chrom in exclude_chroms:
            print(f"Skipping chromosome {chrom}") 
        result = pool.apply_async(find_chrom, args=(input_dir,chrom,window_size,step_size))
        results.append((chrom, result))
    for chrom, result in results:
        try:
            features[chrom] = result.get()  # Get the result from AsyncResult
        except Exception as e:
            print(f"Error retrieving result for {chrom}: {e}")
            features[chrom] = None
    return features
    
    

@njit
def _populate_smooth_value_dict(smooth_arr):
    #执行以下格式转换防止numba报错
    #返回 DictType[int64,float64]<iv=None>({3050375: 1.0,
    typed_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64,
    )
    for i in range(smooth_arr.shape[0]):
        typed_dict[int(smooth_arr[i, 0])] = smooth_arr[i, 1]
        
    return typed_dict


def _load_smoothed_chrom(data_dir, chrom):
    smoothed_path = os.path.join(data_dir, "smooth", f"{chrom}.csv")
    if not os.path.isfile(smoothed_path):
        raise Exception(
            "Could not find smoothed methylation data for "
            f"chromosome {chrom} at {smoothed_path} . "
        )
    smoo_df = pd.read_csv(smoothed_path, delimiter=",", header=None, dtype="float")
    typed_dict = _populate_smooth_value_dict(smoo_df.values)
    return typed_dict

def _load_chrom_mat(data_dir, chrom):
    mat_path = os.path.join(data_dir, "data" ,f"{chrom}.npz")
    #print(f"loading chromosome {chrom} from {mat_path} ... \n")
    try:
        mat = sparse.load_npz(mat_path)
    except FileNotFoundError:
        mat = None
    return mat


def find_chrom(input_dir,chrom,window_size=1000,step_size=300):
    """
    Find all chromosomes in the data directory.
    """
    print(f"Finding denovo windows for chromosome {chrom} ...\n")
    mat = _load_chrom_mat(data_dir=input_dir, chrom=chrom)   #csr矩阵正常载入
    smoothed_cpg_vals = _load_smoothed_chrom(data_dir=input_dir, chrom=chrom) #为了numba执行需要进行格式转换
    chrom_len, n_cells = mat.shape #矩阵大小
    cpg_pos_chrom = np.nonzero(mat.getnnz(axis=1))[0] #矩阵的起始位置，即从基因组哪里开始有第一个值
    
    data_chrom = mat.data
    indices_chrom = mat.indices
    indptr_chrom = mat.indptr
    half_bw = window_size // 2  # half the window size
    start = cpg_pos_chrom[0] + half_bw + 1
    end = cpg_pos_chrom[-1] - half_bw - 1
    
    windows, mean, var, num = _move_windows_worker(
    start=start,
    end=end,
    stepsize=step_size,
    half_bw=half_bw,
    data_chrom=data_chrom,
    indices_chrom=indices_chrom,
    indptr_chrom=indptr_chrom,
    smoothed_vals=smoothed_cpg_vals,
    n_cells=n_cells,
    chrom_len=chrom_len,
    )
    result =  merge(windows, mean, var, num, half_bw)
    return result

@njit(nogil=True)
def merge(windows, mean, var, num, half_bw):
    sr_var_values = var[~np.isnan(var)]
    
        # 如果有效的 sr_var 值少于 2 个，则无法计算标准差
    if len(sr_var_values) < 2:
        print("No valid sr_var values found.")
        return
    else:
        # 计算平均值和标准差
        mean_sr_var = np.mean(sr_var_values)
        std_sr_var = np.std(sr_var_values)
    
        # 定义上下一个标准差的阈值
        lower_threshold = mean_sr_var - 2 *std_sr_var
        upper_threshold = mean_sr_var + 2 *std_sr_var
        #print("threshold:",lower_threshold,"-",upper_threshold)
    
    filtered_windows = []
        
    # 遍历 var 中的每一项，基于标准差阈值进行过滤
    for i, item in enumerate(var):
        if item < lower_threshold or item > upper_threshold:
            # 假设 var 中的每个 item 是一个包含 'start' 和 'end' 的字典
            # 如果 item 是字典格式，可以这样访问 'start' 和 'end'
            filtered_windows.append((windows[i]-half_bw, windows[i]+half_bw))
            
            #print(windows,(windows[i]-half_bw, windows[i]+half_bw))
    
    # 合并相邻或重叠的窗口
    merged_windows = []
    if filtered_windows:
        filtered_windows.sort()
        current_start, current_end = filtered_windows[0]
        for start, end in filtered_windows[1:]:
            if start <= current_end:  # 如果窗口重叠或相邻
                current_end = max(current_end, end)
            else:
                merged_windows.append((current_start, current_end))
                current_start, current_end = start, end
        merged_windows.append((current_start, current_end))  # 添加最后一个窗口
    return merged_windows

@njit(parallel=True)
def _move_windows_worker(
    start,
    end,
    stepsize,
    half_bw,
    data_chrom,
    indices_chrom,
    indptr_chrom,
    smoothed_vals,
    n_cells,
    chrom_len,
):
    """
    Move the sliding window along the whole chromosome.
    For each window, calculate the mean shrunken residuals,
    i.e. 1 methylation value per cell for that window. Then
    calculate the variance of shrunken residuals. This is our
    measure of methylation variability for that window.
    """
    windows = np.arange(start, end, stepsize)
    var = np.empty(windows.shape, dtype=np.float64)
    mean = np.empty(windows.shape, dtype=np.float64)
    # alpha = np.empty(windows.shape, dtype=np.float64)
    # beta = np.empty(windows.shape, dtype=np.float64)
    # gamma = np.empty(windows.shape, dtype=np.float64)
    # beta_var = np.empty(windows.shape, dtype=np.float64)
    f_number = np.empty(windows.shape, dtype=np.int64)
    for i in prange(windows.shape[0]):
        pos = windows[i]
        #要保证第一个和最后一个窗口向左右延伸一点   
        sample_meth_proteior = _calc_mean_shrunken_residuals(
            data_chrom,
            indices_chrom,
            indptr_chrom,
            pos - half_bw,
            pos + half_bw,
            smoothed_vals,
            n_cells,
            chrom_len,
        )
        var[i] = np.nanvar(sample_meth_proteior)
        mean[i] = np.nanmean(sample_meth_proteior)
        f_number[i] = np.sum(~np.isnan(sample_meth_proteior)) #非 NaN 元素的个数
    
    return windows, mean, var, f_number
    #return windows, mean, var

# @njit
def _calc_beta_binomial_var(windows, mean, var, num):
    
    beta_var = np.empty(windows.shape, dtype=np.float64)
    scipy_var = np.empty(windows.shape, dtype=np.float64)
    gamma = np.empty(windows.shape, dtype=np.float64)
    for i in range(len(windows)):
        mu = mean[i]
        sigma2 = var [i]
        n = num[i]
        if mu is None:
            beta_var[i] = np.nan
            gamma[i] = np.nan
            continue
        alpha = mu * (mu * (1 - mu) / sigma2 - 1)
        beta = (1 - mu) * (mu * (1 - mu) / sigma2 - 1)
        
        mean_p = alpha / (alpha + beta)
        gamma[i] = 1 / (alpha + beta + 1)
   
        #beta_var[i] = (gamma**2 * m * (1 - m * (1-gamma) -gamma))/ (1 - gamma)
        #beta_var[i] = n * m * (1 - m)* (1+ ((n -1)/ (alpha + beta +n -1)))
        var_p = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        beta_var[i] = n * mean_p * (1 - mean_p) * (1 + alpha) / (1 + alpha + beta)
        # 理论方差
    # mean_p = alpha / (alpha + beta_param)
    # theoretical_variance = n * mean_p * (1 - mean_p) * (1 + alpha) / (1 + alpha + beta_param)
    # print("var_p:",var_p)
    # print("beta_var:",beta_var)
    # print("var_scipy:",scipy_var)
    return windows, beta_var, gamma

@njit(nogil=True)
def _calc_mean_shrunken_residuals(
    data_chrom, #csr
    indices_chrom, #
    indptr_chrom,
    region_start,
    region_end,
    smoothed_vals,
    n_cells,
    chrom_len,
    shrinkage_factor=1,
):
    shrunken_resid = np.full(n_cells, np.nan, dtype=np.float32)
    sample_meth_proteior = np.full(n_cells, np.nan, dtype=np.float32)
    start = max(region_start - 1, 0)
    end = min(region_end - 1, chrom_len)
    if start >= chrom_len or start >= end:
        return sample_meth_proteior
    # slice the methylation values so that we only keep the values in the window
    data = data_chrom[indptr_chrom[start] : indptr_chrom[end]]
    #print(f"in:{start}-{end}")
    if data.size == 0:
        # return NaN for regions without coverage or regions without CpGs
        return shrunken_resid
    # slice indices
    indices = indices_chrom[indptr_chrom[start] : indptr_chrom[end]]
    # slice index pointer
    indptr = indptr_chrom[start : end + 1] - indptr_chrom[start]
    indptr_diff = np.diff(indptr)

    n_obs = np.zeros(n_cells, dtype=np.int64)
    n_obs_start = np.bincount(indices)
    n_obs[0 : n_obs_start.shape[0]] = n_obs_start

    meth_sums = np.zeros(n_cells, dtype=np.int64)
    smooth_sums = np.zeros(n_cells, dtype=np.float64)
    cpg_idx = 0
    nobs_cpg = indptr_diff[cpg_idx]
    # nobs_cpg: how many of the next values correspond to the same CpG
    # e.g. a value of 3 means that the next 3 values are of the same CpG
    #print(f"data:{start}-{end}",data)
    for i in range(data.shape[0]):
        while nobs_cpg == 0:
            cpg_idx += 1
            nobs_cpg = indptr_diff[cpg_idx]
        nobs_cpg -= 1
        cell_idx = indices[i]
        smooth_sums[cell_idx] += smoothed_vals[start + cpg_idx]
        meth_value = data[i]
        if meth_value == -1:
            continue  # skip 0 meth values when summing
        meth_sums[cell_idx] += meth_value
        
    for i in range(n_cells):
        if n_obs[i] > 0:
            shrunken_resid[i] = (meth_sums[i] - smooth_sums[i]) / (
                n_obs[i] + shrinkage_factor
            )
            sample_meth_proteior[i] = meth_sums[i] / n_obs[i]
    # print("sum:",meth_sums)
    # print("n_obs:",n_obs)
    return  sample_meth_proteior
