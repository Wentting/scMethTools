
import os
import numpy as np
import pandas as pd
import scMethtools.logging as logg
import scipy.sparse as sparse
import click


def echo(*args, **kwargs):
    click.echo(*args, **kwargs, err=True)
    return


def secho(*args, **kwargs):
    click.secho(*args, **kwargs, err=True)
    return

def _feature_methylation_chrom(regions,chrom,output_dir,npz_path,relative,smooth):
    """
    calculate methylation level and residual for each chromosome

    Args:
        regions (_type_): _description_
        chrom (_type_): _description_
        output_dir (_type_): _description_
        relative (_type_): _description_
        smooth (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        #echo(f"...saving single-cytosine sparse matrix for {chrom}")
        # csr_matrix_chrom = _save_npz(tmp_path,output_dir,chrom)
        #_save_npz(tmp_path,output_dir,chrom)     
        if relative: 
         
            csv_file_path = os.path.join(os.path.join(output_dir,"smooth"), f"{chrom}.csv")
            df = pd.read_csv(csv_file_path, header=None, names=["pos", "smooth_val"])
            # Convert DataFrame to a dictionary
            smooth_dict = dict(zip(df["pos"], df["smooth_val"]))   
            logg.info(f"...caculate {chrom} relative methylation level")    
            r,m,var = _caculate_relative_mean_chrom(npz_path,chrom,regions,smooth_dict)
            return r,m,var
        else:
            logg.info(f"...caculate {chrom} methylation level")  
            m,var = _caculate_mean_chrom(npz_path,chrom,regions)
            return m,var
    except Exception as e:
        secho("Warning: ", fg="red", nl=False)
        echo(f"An error occurred: {e}")
        

def _caculate_mean_chrom(npz_path,chrom,regions):
    """
    calculate methylation level of features for each chromosome

    Args:
        npz_path (_type_): _description_
        chrom (_type_): _description_
        regions (_type_): _description_

    Returns:
        _type_: _description_
    """
    mean_bins = []
    region_mtx = []
    try:
        # Load the sparse matrix for the specified chromosome
        csr_matrix_chrom = sparse.load_npz(os.path.join(npz_path, f"{chrom}.npz"))
    except FileNotFoundError:
        print(f"File {chrom}.npz not found in {npz_path}.")
        return [],[]
      
    chrom_len, n_cells = csr_matrix_chrom.shape
    for region_start, region_end in regions: #annotation regions 
        #print("annotation-region",region_start,'-',region_end)
        #返回值不是元组，而是两个独立的列表。因此，在调用 _calc_mean_shrunken_residuals 函数时，只能使用一个变量来接收返回值
        #mean_shrunk_resid, mean_level = _calc_mean_shrunken_residuals
        result = _calMean(
            csr_matrix_chrom,
            region_start,
            region_end,
            n_cells,
            chrom_len,
        )
        mean_bins.append(result)
        #统计var
        region_mtx.append(_calVar(chrom,region_start,region_end,sr=None,mean=result))
    return mean_bins,region_mtx

def _calMean(data_chrom,region_start, region_end,n_cells, chrom_len,):
    mean_level = np.full(n_cells, np.nan)
    start = max(region_start - 1, 0)
    end = min(region_end - 1, chrom_len)
    if start >= chrom_len or start >= end:
        return mean_level
    selected_rows = data_chrom[start:end + 1, :].toarray() #已测试numpy方法更快
    if selected_rows.size == 0:
        return mean_level
    cell_sums = np.sum(np.where(selected_rows == 1, selected_rows, 0), axis=0)
    n_obs = np.sum(selected_rows != 0, axis=0)
    nonzero_mask = n_obs > 0
    mean_level[nonzero_mask] = np.round(cell_sums[nonzero_mask] / n_obs[nonzero_mask], 3)
    return mean_level

def _calVar(chromosome, start, end, sr, mean):
    mean_mask = ~np.isnan(mean)
    covered_cell = np.count_nonzero(mean_mask)
    mean_var = np.nanvar(mean) if covered_cell > 1 else np.nan

    result = {
        'chromosome': chromosome,
        'start': start,
        'end': end,
        'covered_cell': covered_cell,
        'var': mean_var
    }
    if sr is not None:
        sr_mask = ~np.isnan(sr)
        sr_var = np.nanvar(sr) if np.count_nonzero(sr_mask) > 1 else np.nan
        result['sr_var'] = sr_var

    return result

def _calcRMean(
    data_chrom,
    region_start,
    region_end,
    smoothed_vals,
    n_cells,
    chrom_len,
    shrinkage_factor=1,
):
    shrunken_resid = np.full(n_cells, np.nan, dtype=np.float32)
    mean_level = np.full(n_cells, np.nan, dtype=np.float32)
    start = max(region_start - 1, 0)
    end = min(region_end - 1, chrom_len)
    if start >= chrom_len or start >= end:
        return shrunken_resid, mean_level
    # 获取选定区域的行
    selected_rows = data_chrom[start:end + 1, :]
    if selected_rows.nnz == 0:
        return shrunken_resid, mean_level

    # 直接在稀疏矩阵上计算 cell_sums，跳过值为 -1 的元素
    cell_sums = np.zeros(n_cells, dtype=np.float32)
    n_obs = np.zeros(n_cells, dtype=np.int32)
    for start, end in zip(selected_rows.indptr[:-1], selected_rows.indptr[1:]):
        for index in range(start, end):
            col = selected_rows.indices[index]
            value = selected_rows.data[index]
            if value != -1:
                cell_sums[col] += value
            if value != 0: 
                n_obs[col] += 1
    # 计算 smooth_sum
    smooth_sum = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        if n_obs[i] > 0:
            non_zero_indices  = selected_rows[:, i].nonzero()[0] + region_start -1 # must -1
            smooth_sum[i] = sum(smoothed_vals.get(j, 0) for j in non_zero_indices)
    
    valid_obs_mask = n_obs > 0
    # 计算 shrunken_resid 和 mean_level
    shrunken_resid[valid_obs_mask] = (cell_sums[valid_obs_mask] - smooth_sum[valid_obs_mask]) / \
                                     (n_obs[valid_obs_mask] + shrinkage_factor)
    mean_level[valid_obs_mask] = np.round(cell_sums[valid_obs_mask] / n_obs[valid_obs_mask], 3)
    return shrunken_resid, mean_level

def _caculate_relative_mean_chrom(npz_path,chromosome,regions,smooth_dict):
    meth_shrunken_bins = []
    mean_bins = []
    region_mtx = []  
    try:
        data_chrom = sparse.load_npz(os.path.join(npz_path, f"{chromosome}.npz"))
    except FileNotFoundError:
        secho("Warning: ", fg="red", nl=False)
        echo(
            f"Couldn't load methylation data for chromosome {chromosome} at {npz_path} "
        )
        data_chrom = None
    chrom_len, n_cells = data_chrom.shape
    for region in regions:#annotation regions 
        if len(region) == 2:
            region_start, region_end = region
        else:
            region_start, region_end, *additional_info = region
        #print("annotation-region",region_start,'-',region_end)
        #返回值不是元组，而是两个独立的列表。因此，在调用 _calc_mean_shrunken_residuals 函数时，只能使用一个变量来接收返回值
        #mean_shrunk_resid, mean_level = _calc_mean_shrunken_residuals
        result = _calcRMean(
            data_chrom,
            region_start,
            region_end,
            smooth_dict,
            n_cells,
            chrom_len
        )
    
        meth_shrunken_bins.append(result[0])
        mean_bins.append(result[1])
        #统计var
        region_mtx.append(_calVar(chromosome,region_start,region_end,result[0],result[1]))
    return meth_shrunken_bins,mean_bins,region_mtx

 