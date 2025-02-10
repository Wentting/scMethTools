#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月21日

"""

import pandas as pd
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path

from anndata import ImplicitModificationWarning
from numba import jit
from scipy.sparse import coo_matrix
from scipy import sparse
import datetime as datetime
from typing import Union
import anndata as ad
import subprocess
import pathlib
from .._genome import Genome, hg38
import collections
import click
import time
import warnings

__all__ = ['import_bed_file','convert_coo_file_to_matrix','convert_coo_to_csr_without_imputation','convert_coo_to_csr_with_imputation']

def echo(*args, **kwargs):
    click.echo(*args, **kwargs, err=True)
    return


def secho(*args, **kwargs):
    click.secho(*args, **kwargs, err=True)
    return

def find_files_with_suffix(path, suffix):
    matching_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                matching_files.append(os.path.join(root, file))
    return matching_files


def filter_cells(
        data: ad.AnnData,
        min_counts: int = 1000,
        min_tsse: float = 5.0,
        max_counts: int = 30000,
        max_tsse: float = 10.0,
        inplace: bool = True,
):
    """
    Filter cell outliers based on counts and numbers of genes expressed.
    For instance, only keep cells with at least `min_counts` counts or
    `min_tsse` TSS enrichment scores. This is to filter measurement outliers,
    i.e. "unreliable" observations.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_tsse
        Minimum TSS enrichemnt score required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_tsse
        Maximum TSS enrichment score expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.

    Returns
    -------
    np.ndarray | None:
        If `inplace = True`, directly subsets the data matrix. Otherwise return
        a boolean index mask that does filtering, where `True` means that the
        cell is kept, `False` means the cell is removed.
    """
    selected_cells = True
    if min_counts: selected_cells &= data.obs["n_fragment"] >= min_counts
    if max_counts: selected_cells &= data.obs["n_fragment"] <= max_counts
    if min_tsse: selected_cells &= data.obs["tsse"] >= min_tsse
    if max_tsse: selected_cells &= data.obs["tsse"] <= max_tsse

    if inplace:
        if data.isbacked:
            data.subset(selected_cells)
        else:
            data._inplace_subset_obs(selected_cells)
    else:
        return selected_cells


def stat(
        cov_df: str,
        cell_name: str,
        cell_id: str,
        out_dir: Path,
):
    print("*****summary file generate *****")
    sta_df = cov_df.groupby('mc_class').agg({"methylated": "sum", "mc_class": "count", "total": "sum"})
    sta_df['overall_mc_level'] = sta_df['methylated'] / sta_df['total']
    # 使用 stack() 函数将 DataFrame 转换为单列的 Series 对象
    df_transposed = sta_df.stack().to_frame().T
    # 将列名修改为需要的格式
    df_transposed.columns = [f'{col}_{row}' for row in sta_df.index for col in sta_df.columns]
    df_transposed['sample'] = cell_name
    df_transposed['sample_id'] = cell_id
    stat_file = os.path.join(out_dir, "basic_summary.csv")
    try:
        if os.path.exists(stat_file):
            df_transposed.to_csv(stat_file, index=False, header=False, sep=',', mode ='a')
        else:
            df_transposed.to_csv(stat_file, index=False, sep=',', mode='a')
    except PermissionError:
        print(f"文件 '{stat_file}' 正在被另一个进程锁定。等待关闭后重试...")
    print('*****summary file end ****')
    return stat_file


def _iter_chunks(data_dir, chrom):
    # print(os.path.join(data_dir, f"{chrom}_chunk*.coo"))
    chunk_paths = glob.glob(os.path.join(data_dir, "{}_chunk*.coo".format(chrom)))
    for chunk_path in sorted(chunk_paths):
        chunk = pd.read_csv(chunk_path, delimiter=",", header=None).values  # array
        # os.remove(chunk_file) #删除中间文件
        # sort_chunk = np.sort(chunk) #numpy.ndarray  ############################################这里要排序，报错了这里
        yield chunk


def _read_coo_file(coo_file):
    rows, cols, data = coo_file[:, 0], coo_file[:, 1], coo_file[:, 2]
    # print(rows,cols,data)
    return rows, cols, data

def _convert_cov_to_coo(
        output_dir: Path,
        cov_file: str,
        cell_id: int,
        pipeline: str = 'Bismark',
        out_file: str = None,
        chunksize: int = 1000000,
        round_sites: bool = True,
):
    """
    convert single methylation cov/bed file into coo_matrix with fixed batch in temp dir
    :param out_dir:
    :param cov_file:
    :param chunksize:
    :param cell_id:
    :param out_file:
    :param round_sites:
    :return:
    """


    print('step1 : begin build coo matrix ----- \n')
    # print('pipeline is ' +pipeline+'\n')


    if out_file is None:
        name = os.path.basename(cov_file)
        sample_name = os.path.splitext(name)[0]
        print('processed cell ', sample_name, '...\n')

    cov_df = pd.read_csv(cov_file, sep='\t', header=None,
                         names=['chr', 'strand', 'pos', 'mc_class', 'mc_context', 'mc_level', 'methylated',
                                'total'])  # usecols=['chr','pos','mc_level','total']
    cov_df = cov_df.sort_values(['chr', 'pos'])
    chrom_sizes = {}
    current_chrom = None
    current_chunk = None
    current_file = None
    # record qc into stats file
    stat_file = stat(cov_df, sample_name, cell_id,output_dir)

    # 测试用，不测试的时候删除
    #valid_values = ["chr1", "chr2"]
    print("enter...")
    for i, row in cov_df.iterrows():
        chrom, pos, mc_level,context = row['chr'], row['pos'], row['mc_level'],row['mc_context']
        if 'N' in context:
            continue
        chunk = int(pos // chunksize)
        #if chrom in valid_values:  # 测试用，不测试删除这个判断
        if chrom != current_chrom or chunk != current_chunk:
            current_chrom = chrom
            current_chunk = chunk
            if current_file:
                current_file.close()  # 每次结束一个染色体的一个chunk区域后要关闭文件
            filename = '{}_chunk{:07d}.coo'.format(current_chrom, current_chunk)
            current_file = open(os.path.join(output_dir, filename), 'a')
            if chrom not in chrom_sizes:
                chrom_sizes[chrom] = 0
        current_file.write('{}, {} ,{}\n'.format(pos, cell_id, mc_level))
        if pos > chrom_sizes[current_chrom]:
            chrom_sizes[current_chrom] = pos  # 记录的是测到的最大碱基的位置不是真正的基因组位置
    if current_file:
        current_file.close()
    return chrom_sizes


def caculate_methylation_feature(csr_mat, start, end, n_cells, min_cov=0):
    print('-----------------------calutate mean---------------------')
    mean_array = np.zeros(n_cells,dtype=np.int64)
    mean_list = []
    sliced_data = csr_mat[start - 1:end, :] #sliced by rows
    sliced_data.data[sliced_data.data == -1] = 0
    print('-----------------------calutating...---------------------')
    # 计算平均值、方差、标准差和离散度
    for col_idx in range(csr_mat.shape[1]):
        col_data = csr_mat[start - 1:end, col_idx].toarray()
        col_data = col_data[col_data != 0]# 跳过值为0的数据
        col_data[col_data == -1] = 0# 在计算的时候把-1当成0计算
        if col_data.size == 0 or col_data.size < min_cov:
            mean_array[col_idx] = np.NaN
        else:
            mean_array[col_idx] = np.mean(col_data)
    return mean_array

@jit(nopython=True)  # 使用nopython=True以启用Numba的编译优化
def calc_cell_features(csr_mtx, start, end, smooth_vals, n_cells, chrom):
    print('-----------------------cal for', chrom, ':', start, '-', end)
    shrunken_resid = np.full(n_cells, np.nan)
    smooth_sums = np.zeros(n_cells, dtype=np.float64)
    non_zero_counts = np.zeros(n_cells, dtype=np.float64)
    meth_sums = np.zeros(n_cells, dtype=np.float64)
    meth_means = np.full(n_cells, np.nan)
    # slice csr matrix
    csr_matrix_slice = csr_mtx[start-1:end, :]
    nonzero_indices = csr_matrix_slice.nonzero()
    # 输出非零元素的行索引和列索引
    for row, col in zip(nonzero_indices[0], nonzero_indices[1]):
        meth_value = csr_matrix_slice[row, col]
        non_zero_counts[col] += 1
        pos_idx = start + row - 1
        smooth_value = smooth_vals[smooth_vals[0] == pos_idx][1]
        smooth_sums[col] += smooth_value
        if meth_value == -1:
            continue
        meth_sums[col] += meth_value
    for i in range(n_cells):
        if non_zero_counts[i] > 0:
            shrunken_resid[i] = (meth_sums[i] - smooth_sums[i]) / (
                    non_zero_counts[i] + 1)
            meth_means[i] = meth_sums[i] / non_zero_counts[i]
    return shrunken_resid, meth_means

def calc_cell_features(csr_mtx, start, end, smooth_vals ,n_cells,chrom):
    print('-----------------------cal for',chrom,':',start,'-',end)
    shrunken_resid = np.full(n_cells, np.nan)
    smooth_sums = np.zeros(n_cells, dtype=np.float64)
    non_zero_counts = np.zeros(n_cells, dtype=np.float64)
    meth_sums = np.zeros(n_cells, dtype=np.float64)
    meth_means = np.full(n_cells, np.nan)
    #slice csr matrix
    csr_matrix_slice = csr_mtx[start-1:end, :]
    nonzero_indices = csr_matrix_slice.nonzero()
    # 输出非零元素的行索引和列索引
    for row, col in zip(nonzero_indices[0], nonzero_indices[1]):
        meth_value = csr_matrix_slice[row, col]
        non_zero_counts[col] += 1
        pos_idx = start + row -1
        smooth_value = smooth_vals[smooth_vals[0] == pos_idx][1]
        smooth_sums[col] += smooth_value
        if meth_value == -1:
            continue
        meth_sums[col] += meth_value
    for i in range(n_cells):
        if non_zero_counts[i] > 0:
            shrunken_resid[i] = (meth_sums[i] - smooth_sums[i]) / (
                    non_zero_counts[i] + 1)
            meth_means[i] = meth_sums[i]/non_zero_counts[i]
    return shrunken_resid,meth_means


def convert_coo_file_to_matrix(data_dir, chrom):
    coo_matrix_path = os.path.join(data_dir, "{}_coo_matrix.npz".format(chrom))
    chunk_row = []
    chunk_col = []
    chunk_data = []
    for chunk_file in _iter_chunks(data_dir, chrom):  # 按顺序处理染色体的chunk
        file_rows, file_cols, file_data = _read_coo_file(chunk_file)
        chunk_row.extend(file_rows.astype(int))
        chunk_col.extend(file_cols.astype(int))
        chunk_data.extend(file_data)
    coo_matrix_result = coo_matrix((chunk_data, (chunk_row, chunk_col)))
    sparse.save_npz(coo_matrix_path, coo_matrix_result)
    return coo_matrix_path, coo_matrix_result

def _load_chrom_size_file(chrom_file,remove_chr_list=None):
    with open(chrom_file) as f:
        chrom_dict = collections.OrderedDict()
        for line in f:
            # *_ for other format like fadix file
            chrom, length, *_ = line.strip('\n').split('\t')
            if chrom in remove_chr_list:
                continue
            chrom_dict[chrom] = int(length)
    return chrom_dict
    pass


def _sliding_windows_with_step_size(window_size, step_size, ref, chrom_size=None, chrom_file=None):
    """

    :param window_size:
    :param step_size:
    :param ref:
        A Genome object, providing gene annotation and chromosome sizes.
        If not set, `gff_file` and `chrom_size` must be provided.
        `genome` has lower priority than `gff_file` and `chrom_size`.
    :param chrom_file:
        File name of the gene annotation file in BED or GFF or GTF format.
        This is required if `ref` is not set.
        Setting `chrom_file` will override the annotations from the `genome` parameter.
    :param chrom_size:
        A dictionary containing chromosome sizes, for example,
        `{"chr1": 2393, "chr2": 2344, ...}`.
        This is required if `genome` is not set.
        Setting `chrom_size` will override the chrom_size from the `genome` parameter.
    :return:
    """
    chrom_size_dict = {}

    if step_size is None:
        step_size = window_size
    if chrom_file is None:
        if ref is not None:
            chrom_size_dict = ref.chrom_sizes
    else:
        chrom_size_dict = _load_chrom_size_file(chrom_file)  # user defined reference, especially for other species

    records = []
    for chrom, chrom_length in chrom_size_dict.items():
        bin_start = np.array(list(range(0, chrom_length, step_size)))
        bin_end = bin_start + window_size
        bin_end[np.where(bin_end > chrom_length)] = chrom_length
        chrom_df = pd.DataFrame(dict(start=bin_start, end=bin_end))
        chrom_df['chrom'] = chrom
        records.append(chrom_df)
    total_df = pd.concat(records)[['chrom', 'start', 'end']].reset_index(drop=True)
    return total_df


def caculate_smooth_residuals(csr_mtx, start, end, smoothed_vals, n_cells):
    shrunken_resid = np.full(n_cells, np.nan)
    smooth_sums = np.zeros(n_cells, dtype=np.float64)
    non_zero_counts = np.zeros(n_cells, dtype=np.float64)
    meth_sums = np.zeros(n_cells, dtype=np.float64)
    # slice csr matrix
    csr_matrix_slice = csr_mtx[start - 1:end, :]
    # #get c-count by col
    # non_zero_counts = csr_matrix_slice.getnnz(axis=0)
    # #get meth_total sum by col(sample)
    # # 将CSR矩阵中的值为-1的元素替换为0
    # csr_matrix_slice.data[csr_matrix_slice.data == -1] = 0
    # # 计算每列的加和
    # meth_sums = np.array(csr_matrix_slice.sum(axis=0)).flatten()

    nonzero_indices = csr_matrix_slice.nonzero()
    # 输出非零元素的行索引和列索引
    for row, col in zip(nonzero_indices[0], nonzero_indices[1]):
        meth_value = csr_matrix_slice[row, col]
        non_zero_counts[col] += 1
        pos_idx = start + row - 1
        # print('position:',pos_idx)
        smooth_sums[col] += smoothed_vals[smoothed_vals[0] == pos_idx][1]
        if meth_value == -1:
            continue
        meth_sums[col] += meth_value
    print('non_zero', non_zero_counts)
    print('sums', meth_sums)
    print('smooth', smooth_sums)
    for i in range(n_cells):
        if non_zero_counts[i] > 0:
            # print(meth_sums[i])
            # print(non_zero_counts[i])
            # print(smooth_sums[i])
            shrunken_resid[i] = (meth_sums[i] - smooth_sums[i]) / (
                    non_zero_counts[i] + 1)
            # print('shrunken:',shrunken_resid)
    return shrunken_resid


def _process_smooth_chromosome(temp_dir, chrom_data):
    temp_dir = Path(temp_dir)
    chrom, chrom_records = chrom_data
    # 使用 glob 模块查找文件
    matching_files = glob.glob(os.path.join(temp_dir, f'{chrom}*npz'))
    if not matching_files:
        print(f"No matching files found for {chrom}. Returning empty result.")
        return {}  # 返回空的结果，一个空的字典
    else:
        secho(f"\n merge {chrom} coo_matrix and bin it ", fg="green")
        # for coo_path in sorted(glob(os.path.join(temp_dir, "chr1*.npz"))):
        coo_mat = sparse.vstack([sparse.load_npz(path) for path in temp_dir.glob(f'{chrom}*npz')])
        coo_mat.data = np.where(coo_mat.data == 0, -1,
                                coo_mat.data)  # change all zero to -1 which means unmethylated for calculating
        csr_mat = coo_mat.tocsr()
        n_cells = csr_mat.shape[1]
        secho(f"\n merge {chrom} coo_matrix and bin it for {n_cells} cell ", fg="green")
        # 遍历每一行
        feature_mtx = {}
        residual_mtx = {}
        smoothed_data = pd.read_csv(f'{temp_dir}/smoothed/{chrom}_coo_matrix.csv', header=None)
        for index, row in chrom_records.iterrows():
            chr = row['chrom']
            start = row['start']
            end = row['end']
            smoothed_vals = smoothed_data[(smoothed_data[0] >= start-1) & (smoothed_data[0] <= end)]
            shrunken_residuals,mean = calc_cell_features(csr_mat, start, end, smoothed_vals,n_cells,chrom)
            feature_name = f"{chr}_{start}_{end}"
            feature_mtx[feature_name] = mean
            residual_mtx[feature_name] = shrunken_residuals
        print(f'merge finished for {chrom}')
        return feature_mtx,residual_mtx

def _process_chromosome(temp_dir, chrom_data):
    temp_dir = Path(temp_dir)
    chrom, chrom_records = chrom_data
    # 使用 glob 模块查找文件
    matching_files = glob.glob(os.path.join(temp_dir, f'{chrom}*npz'))
    if not matching_files:
        print(f"No matching files found for {chrom}. Returning empty result.")
        return {}  # 返回空的结果，一个空的字典
    else:
        secho(f"\n merge {chrom} coo_matrix and bin it ", fg="green")
        # for coo_path in sorted(glob(os.path.join(temp_dir, "chr1*.npz"))):
        coo_mat = sparse.vstack([sparse.load_npz(path) for path in temp_dir.glob(f'{chrom}*npz')])
        coo_mat.data = np.where(coo_mat.data == 0, -1,
                                coo_mat.data)  # change all zero to -1 which means unmethylated for calculating
        csr_mat = coo_mat.tocsr()
        n_cells = csr_mat.shape[1]
        secho(f"\n merge {chrom} coo_matrix and bin it for {n_cells} cell ", fg="green")
        # 遍历每一行
        feature_mtx = {}
        for index, row in chrom_records.iterrows():
            chr = row['chrom']
            start = row['start']
            end = row['end']
            mean = caculate_methylation_feature(csr_mat, start, end, n_cells)

            feature_name = f"{chr}_{start}_{end}"
            feature_mtx[feature_name] = mean
        print(f'merge finished for {chrom}')
        return feature_mtx

def _process_chromosome_hvf(temp_dir, chrom_data):
    temp_dir = Path(temp_dir)
    chrom, chrom_records = chrom_data
    # 使用 glob 模块查找文件
    matching_files = glob.glob(os.path.join(temp_dir, f'{chrom}*npz'))
    if not matching_files:
        print(f"No matching files found for {chrom}. Returning empty result.")
        return {}  # 返回空的结果，一个空的字典
    else:
        secho(f"\n merge {chrom} coo_matrix and bin it ", fg="green")
        # for coo_path in sorted(glob(os.path.join(temp_dir, "chr1*.npz"))):
        coo_mat = sparse.vstack([sparse.load_npz(path) for path in temp_dir.glob(f'{chrom}*npz')])
        coo_mat.data = np.where(coo_mat.data == 0, -1,
                                coo_mat.data)  # change all zero to -1 which means unmethylated for calculating
        csr_mat = coo_mat.tocsr()
        n_cells = csr_mat.shape[1]
        secho(f"\n merge {chrom} coo_matrix and bin it for {n_cells} cell ", fg="green")
        # 遍历每一行
        feature_mtx = {}
        for index, row in chrom_records.iterrows():
            chr = row['chrom']
            start = row['start']
            end = row['end']
            mean = caculate_methylation_feature(csr_mat, start, end, n_cells)
            feature_name = f"{chr}_{start}_{end}"
            feature_mtx[feature_name] = mean
        print(f'merge finished for {chrom}')

        # Calculate entropy and mean_shrunken_residuals
        entropy_values = []
        mean_shrunken_residuals_values = []
        for feature_name, values in feature_mtx.items():
            entropy = calculate_entropy(values)
            mean_shrunken_residuals = calculate_mean_shrunken_residuals(values)
            entropy_values.append(entropy)
            mean_shrunken_residuals_values.append(mean_shrunken_residuals)

        # Create a DataFrame to store the results
        result_df = pd.DataFrame({
            'FeatureName': list(feature_mtx.keys()),
            'Mean': list(feature_mtx.values()),
            'Entropy': entropy_values,
            'MeanShrunkenResiduals': mean_shrunken_residuals_values
        })

        # Save the results to the output file
        result_df.to_csv(output_file, index=False)

        print(f'merge finished for {chrom} and saved results to {output_file}')

        return feature_mtx



def _load_feature(feature_file,chrom,file_format = None):
    if file_format is None:
        file_format = feature_file[-3:]
    if file_format not in ['bed','txt','gtf','gff']:
        print (f"not support {file_format}  format right now, please check your file again")
    # need specify gtf gff loading method
    else:
        features_chrom = {}
        with open(feature_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                value = line.strip().split("\t")
                if value[0] in chrom: #check if the chrom is valid
                    yield value[0],int(value[1]),int(value[2]),value[3:] #返回染色体，起始终止位置和文件的其他列即feature的名称等


# def _load_csr_into_anndata(coo_mtx,out_dir) -> Path:
#     if os.path.exists(out_dir):
#         return out_dir
#
#     feature_mtx = convert_coo_to_csr_with_annotation(coo_mtx)
#     total_matrix = sparse.vstack([sparse.load_npz(path) for path in matrix_paths])
#
#     adata = ad.AnnData(X=total_matrix,
#                     obs=pd.DataFrame([], index=obs_names),
#                     var=var_df[['chrom']],
#                     uns=dict(bin_size=bin_size,
#                              chrom_size_path=chrom_size_path,
#                              mc_type=mc_type,
#                              count_type=count_type,
#                              step_size=step_size,
#                              strandness=strandness))
#     adata.write(output_path,
#                 compression=compression,
#                 compression_opts=compression_opts)
#     return output_path


def convert_coo_to_csr_without_imputation(temp_dir,window_size=100000,step_size=None,feature_file=None,out_file=None,cpu=5,ref=hg38):
    # adata = ad.AnnData() if out_file is None else ad.AnnData(filename=out_file)
    feature = _sliding_windows_with_step_size(100000, 100000, ref)
    # all chromosome or defined chromosome
    chrom_data = [(chrom, feature[feature['chrom'] == chrom]) for chrom in feature['chrom'].unique()]
    pool = mp.Pool(processes=min(cpu, len(chrom_data)))
    # 将染色体数据分组传递给处理函数
    results = []
    print("开始执行注释合并")
    start_time = time.time()
    for i in range(len(chrom_data)):
        chrom = chrom_data[i]
        results.append(pool.apply_async(_process_chromosome, args=(temp_dir, chrom)))
    # 关闭线程池
    pool.close()
    pool.join()
    # # 将处理结果合并成一个 DataFrame
    #按行合并
    # result_df = pd.concat([pd.DataFrame.from_dict(r.get()) for r in results], axis=1, ignore_index=True)
    result_df = pd.concat([pd.DataFrame.from_dict(r.get()) for r in results], axis=1)
    print("注释合并进程结束耗时%s" % (time.time() - start_time))

    # 步骤 1: 将 DataFrame 转换为 CSR 矩阵
    # 步骤 1: 填充 DataFrame 中的 NaN 值
    df_filled = result_df.fillna(0)  # 这里将 NaN 填充为你希望的缺失值填充值（例如 0）
    csr_matrix = sparse.csr_matrix(df_filled.values)
    # 保存行名和列名
    row_names = result_df.columns.tolist()
    col_names = result_df.index.tolist()
    # 步骤 2: 存储 CSR 矩阵到 AnnData 对象
    # `var` must have number of columns of `X` (2490), but has 30894 rows.
    # 步骤 3: 将 DataFrame 保存为文本文件
    result_df.to_csv(f'{temp_dir}/test_chr1.txt', sep='\t', index=False)

    data_df = pd.read_csv('D://Test/GSE56789/temp/basic_summary.csv')
    # df_sorted = data_df[data_df['sample_id'].isin(col_names)].sort_values(by='sample_id')
    df_sorted = data_df.query('sample_id in @col_names').sort_values(by='sample_id')
    # 将整数索引转换为字符串
    # 将每个元素转换为字符串
    var_df = pd.DataFrame({'feature_name': row_names})
    adata = ad.AnnData(X=csr_matrix, obs=df_sorted, var=var_df)
    adata.obs.index = adata.obs.index.astype(str)

    # 保存 AnnData 对象到 HDF5 文件（如果需要）
    adata.write_h5ad(f'{temp_dir}/smoothed_result_anndata.h5')

def convert_coo_to_csr_with_imputation(temp_dir,window_size=100000,step_size=None,feature_file=None,out_file=None,cpu=5,ref=hg38):
    # adata = ad.AnnData() if out_file is None else ad.AnnData(filename=out_file)
    feature = _sliding_windows_with_step_size(100000, 100000, ref)
    # all chromosome or defined chromosome
    chrom_data = [(chrom, feature[feature['chrom'] == chrom]) for chrom in feature['chrom'].unique()]
    pool = mp.Pool(processes=min(cpu, len(chrom_data)))
    # 将染色体数据分组传递给处理函数
    results = []
    print("开始执行注释合并")
    start_time = time.time()
    for i in range(len(chrom_data)):
        chrom = chrom_data[i]
        results.append(pool.apply_async(_process_smooth_chromosome, args=(temp_dir, chrom)))
    # 关闭线程池
    pool.close()
    pool.join()
    for result in results:
        residual,mean = result.get()
        result_mean_df = pd.concat([pd.DataFrame.from_dict(mean)],axis=1)
        result_re_df = pd.concat([pd.DataFrame.from_dict(residual)],axis=1)
    print("注释合并进程结束耗时%s" % (time.time() - start_time))

    # 步骤 1: 将 DataFrame 转换为 CSR 矩阵
    # 步骤 1: 填充 DataFrame 中的 NaN 值
    df_filled = result_mean_df.fillna(0)  # 这里将 NaN 填充为你希望的缺失值填充值（例如 0）
    csr_matrix = sparse.csr_matrix(df_filled.values)
    # 保存行名和列名
    row_names = result_mean_df.columns.tolist()
    col_names = result_mean_df.index.tolist()
    # 步骤 2: 存储 CSR 矩阵到 AnnData 对象
    # `var` must have number of columns of `X` (2490), but has 30894 rows.
    # 步骤 3: 将 DataFrame 保存为文本文件
    result_mean_df.to_csv(f'{temp_dir}/test_chr1.txt', sep='\t', index=False)
    result_re_df.to_csv(f'{temp_dir}/test_smooth_chr1.txt', sep='\t', index=False)
    data_df = pd.read_csv(f'{temp_dir}/basic_summary.csv')
    # df_sorted = data_df[data_df['sample_id'].isin(col_names)].sort_values(by='sample_id')
    df_sorted = data_df.query('sample_id in @col_names').sort_values(by='sample_id')
    # 将整数索引转换为字符串
    # 将每个元素转换为字符串
    var_df = pd.DataFrame({'feature_name': row_names})
    adata = ad.AnnData(X=csr_matrix, obs=df_sorted, var=var_df)
    adata.obs.index = adata.obs.index.astype(str)

    # 保存 AnnData 对象到 HDF5 文件（如果需要）
    adata.write_h5ad(f'{temp_dir}/smoothed_result_anndata.h5')

def import_bed_file(
        data_dir: Path,
        output_dir: Path,
        file: str = None,
        suffix: str = "bed",
        cpu: int = 10,
        pipeline : str = 'Bismark',
        chunksize: int = 100000,
        low_memory: bool = True,
        sorted_by_name: bool = True,
):
    """
    import single cov/bed methylation file into one csr sparse matrix for each chromsome,
    csr matrix will be saved at tmp_file and it is single-nucleotide
    :param data_dir:
    :param file:
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used.
    :param chrom:
    :param low_memory:
    :param sorted_by_name:
    :param blacklist:
        File name or a list of barcodes. If it is a file name, each line
        must contain a valid barcode. When provided, only barcodes in the whitelist
        will be retained.
    :param tempdir:
    :return:
    -------
    AnnData | ad.AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to regions. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.
    """
    # make output_dir if not exits
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = datetime.datetime.now()
    samples = find_files_with_suffix(data_dir, suffix)
    print(samples)
    n_cells = len(samples)
    chunksize = 10000
    chrom_size = {}
    processes = min(cpu, n_cells)
    print(processes)
    time1 = time.time()
    print('Import starting --- --- ' + str(start_time) + '---')
    print('process ', n_cells, 'samples with', (min(cpu, n_cells)), 'cpu paraller')
    chroms = []
    pool = mp.Pool(processes=min(cpu, n_cells))
    for cell_id, cov_file in enumerate(samples):
        # save cell id and name table
        # with open(output_dir + "file_list.csv", 'w') as f:
        #     f.write(f"{cell_n}\t{cov_file}\n")
        chroms.append(
            pool.apply_async(
                _convert_cov_to_coo,
                args=(
                    output_dir,
                    cov_file,
                    cell_id,
                    pipeline,
                ),
            )
        )
    pool.close()
    pool.join()
    print('Conver coo to matrix at--- --- ' + str(datetime.datetime.now()) + '---')
    # conver to coo_matrix .npz format
    pool = mp.Pool(processes=cpu)
    coo_matrix_results = []
    for chrom, size in hg38.chrom_sizes.items():
        coo_matrix_results.append(
            pool.apply_async(convert_coo_file_to_matrix, args=(output_dir,chrom)))
    pool.close()
    pool.join()
    print('Conver coo to matrix at--- --- ' + str(datetime.datetime.now()) + '---')
    print("**** Basic statistics summary has been saved at " + output_dir)
    print('Convet coo end --- --- ' + str(time.time() - time1) + '---')
    time3 = datetime.datetime.now()
    # aggregate all the cells
    print('Aggregate cells into adata --- ' + str(time3) + ' ---')
    temp_dir = output_dir
    convert_coo_to_csr_without_imputation(temp_dir,cpu=cpu,out_file=file)

